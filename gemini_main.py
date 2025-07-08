import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import asyncio
import warnings
import time
import threading
from abc import ABC, abstractmethod

# This is the correct, environment-provided data fetcher.
from datastream.yfinance_ohlcv import YFinanceOHLCVFetcher as Fetcher

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ==============================================================================
# === LEGO BRICK 1: DATA HANDLER ===============================================
# ==============================================================================
class DataHandler:
    """
    Handles fetching and storing of OHLCV data using the provided Fetcher.
    """

    def __init__(self, symbols, period="5y", resolution="1d"):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.resolution = resolution
        period_map = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260, "10y": 2520}
        self.lookback = period_map.get(period.lower(), 1260)
        self.data = {}
        self.data_lock = threading.Lock()
        self.fetcher = Fetcher()

    async def fetch_all_data(self):
        """Fetches data for all symbols using the provided Fetcher."""
        print("Fetching data for all symbols using internal Fetcher...")
        loop = asyncio.get_event_loop()
        with self.data_lock:
            for symbol in self.symbols:
                try:
                    data = await loop.run_in_executor(
                        None,
                        self.fetcher.fetch_ohlcv,
                        symbol,
                        self.resolution,
                        self.lookback,
                    )
                    if data is not None and not data.empty and len(data) > 100:
                        data.rename(
                            columns={
                                "close": "Close",
                                "volume": "Volume",
                                "open": "Open",
                                "high": "High",
                                "low": "Low",
                            },
                            inplace=True,
                            errors="ignore",
                        )
                        self.data[symbol] = data
                        print(
                            f"✓ Successfully fetched {len(data)} data points for {symbol}"
                        )
                    else:
                        print(f"✗ Failed to fetch sufficient data for {symbol}")
                except Exception as e:
                    print(f"Error fetching data for {symbol} with Fetcher: {e}")
        print("Data fetching complete.")

    def get_data(self, symbol):
        with self.data_lock:
            return self.data.get(symbol)

    def get_latest_data(self, symbol, n=1):
        with self.data_lock:
            if symbol in self.data:
                return self.data[symbol].iloc[-n:]
            return None


# ==============================================================================
# === LEGO BRICK 2: FEATURE EXTRACTORS (Corrected) =============================
# ==============================================================================
# This version has no external dependencies like 'ta'.
# ==============================================================================
class BaseFeatureExtractor(ABC):
    @abstractmethod
    def transform(self, df):
        pass


class DefaultFeatureExtractor(BaseFeatureExtractor):
    """A self-sufficient feature extractor using only pandas/numpy."""

    def transform(self, df):
        try:
            if "Close" not in df.columns or "Volume" not in df.columns:
                print("Error: DataFrame must contain 'Close' and 'Volume' columns.")
                return pd.DataFrame()
            if len(df) < 50:
                return None

            features = pd.DataFrame(index=df.index)

            # Standard features
            features["returns"] = df["Close"].pct_change()
            for window in [5, 10, 20, 50]:
                features[f"sma_{window}"] = df["Close"].rolling(window).mean()
                features[f"price_to_sma_{window}"] = (
                    df["Close"] / features[f"sma_{window}"]
                )

            # Manual RSI Calculation
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # Manual MACD Calculation
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            features["macd"] = ema12 - ema26
            features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()

            # Manual Bollinger Bands Calculation
            sma20 = df["Close"].rolling(20).mean()
            std20 = df["Close"].rolling(20).std()
            features["bb_upper"] = sma20 + (std20 * 2)
            features["bb_lower"] = sma20 - (std20 * 2)

            # Other features
            features["volume_sma"] = df["Volume"].rolling(20).mean()
            features["volatility"] = df["Close"].pct_change().rolling(20).std()
            for lag in [1, 2, 3, 5]:
                features[f"returns_lag_{lag}"] = features["returns"].shift(lag)

            return features.dropna()
        except Exception as e:
            print(f"Error creating features: {e}")
            return pd.DataFrame()


# ==============================================================================
# === LEGO BRICK 3: LABELING STRATEGIES ========================================
# ==============================================================================
class BaseLabelingStrategy(ABC):
    @abstractmethod
    def generate(self, df):
        pass


class ForwardReturnLabeling(BaseLabelingStrategy):
    """Labels data based on future returns."""

    def __init__(self, forward_days=3, threshold=0.01):
        self.forward_days = forward_days
        self.threshold = threshold

    def generate(self, df):
        returns = (
            df["Close"].pct_change(periods=self.forward_days).shift(-self.forward_days)
        )
        labels = pd.Series(0, index=df.index)
        labels[returns > self.threshold] = 1
        labels[returns < -self.threshold] = -1
        return labels


# ==============================================================================
# === LEGO BRICK 4: TRADING MODELS =============================================
# ==============================================================================
class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs

    @abstractmethod
    def train(self, X, y):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        defaults = {
            "n_estimators": 50,
            "max_depth": 8,
            "random_state": 42,
            "n_jobs": -1,
        }
        params = {**defaults, **self.params}
        self.model = RandomForestClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y)


class GradientBoostingModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        defaults = {
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        params = {**defaults, **self.params}
        self.model = GradientBoostingClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y)


# ==============================================================================
# === LEGO BRICK 5: STRATEGY ENGINE ============================================
# ==============================================================================
class BaseStrategy(ABC):
    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def generate_signal(self, symbol, current_idx):
        pass


class MLStrategy(BaseStrategy):
    """A machine learning-based strategy."""

    def __init__(
        self,
        data_handler,
        feature_extractor,
        labeling_strategy,
        model_class,
        **model_params,
    ):
        self.data_handler = data_handler
        self.feature_extractor = feature_extractor
        self.labeling_strategy = labeling_strategy
        self.model_class = model_class
        self.model_params = model_params
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        self.model_lock = threading.Lock()
        self.min_accuracy_threshold = 0.35

    def pre_train(self):
        print("\nPre-training all models...")
        for symbol in self.data_handler.symbols:
            with self.model_lock:
                df = self.data_handler.get_data(symbol)
                if df is None or df.empty:
                    continue
                features = self.feature_extractor.transform(df)
                labels = self.labeling_strategy.generate(df)
                data = features.join(labels.rename("target")).dropna()
                if len(data) < 100:
                    print(f"✗ Insufficient data to train for {symbol}")
                    continue
                X = data.drop("target", axis=1)
                y = data["target"]
                if len(y.unique()) < 2:
                    print(f"✗ Not enough class diversity to train for {symbol}")
                    continue
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model_instance = self.model_class(**self.model_params)
                split_idx = int(len(X_scaled) * 0.8)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                model_instance.train(X_train, y_train)
                accuracy = accuracy_score(y_test, model_instance.predict(X_test))
                if accuracy >= self.min_accuracy_threshold:
                    model_instance.train(X_scaled, y)
                    self.models[symbol] = model_instance
                    self.scalers[symbol] = scaler
                    self.feature_cols[symbol] = X.columns
                    print(f"✓ Model for {symbol} trained. Accuracy: {accuracy:.2f}")
                else:
                    print(
                        f"✗ Model for {symbol} accuracy ({accuracy:.2f}) below threshold."
                    )

    def generate_signal(self, symbol, current_idx):
        with self.model_lock:
            if symbol not in self.models:
                return {"signal": "HOLD", "confidence": 0.0}
            df = self.data_handler.get_data(symbol)
            features_df = self.feature_extractor.transform(df.iloc[: current_idx + 1])
            if features_df is None or features_df.empty:
                return {"signal": "HOLD", "confidence": 0.0}
            current_features = features_df.iloc[-1:][self.feature_cols[symbol]]
            if current_features.isnull().any().any():
                return {"signal": "HOLD", "confidence": 0.0}
            X_scaled = self.scalers[symbol].transform(current_features)
            prediction = self.models[symbol].predict(X_scaled)[0]
            confidence = np.max(self.models[symbol].predict_proba(X_scaled)[0])
            signal_map = {1: "LONG", -1: "SHORT", 0: "HOLD"}
            return {
                "signal": signal_map.get(prediction, "HOLD"),
                "confidence": confidence,
            }


# ==============================================================================
# === LEGO BRICK 6: PORTFOLIO & EXECUTION ======================================
# ==============================================================================
class Portfolio:
    """Manages the trading account, supporting long and short positions."""

    def __init__(
        self,
        symbols,
        initial_balance=100000,
        transaction_cost=0.001,
        max_position_size=0.1,
    ):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.trades = []

    def update(self, signal_event):
        symbol, signal, price, date = (
            signal_event["symbol"],
            signal_event["signal"],
            signal_event["price"],
            signal_event["date"],
        )
        target_shares = (self.balance * self.max_position_size) / price
        current_pos = self.positions[symbol]

        if signal == "LONG" and current_pos <= 0:
            if current_pos < 0:
                self._execute_trade(symbol, -current_pos, price, date, "COVER")
            self._execute_trade(symbol, target_shares, price, date, "BUY")
        elif signal == "SHORT" and current_pos >= 0:
            if current_pos > 0:
                self._execute_trade(symbol, -current_pos, price, date, "SELL")
            self._execute_trade(symbol, -target_shares, price, date, "SHORT")
        elif signal != "LONG" and current_pos > 0:
            self._execute_trade(symbol, -current_pos, price, date, "SELL")
        elif signal != "SHORT" and current_pos < 0:
            self._execute_trade(symbol, -current_pos, price, date, "COVER")

    def _execute_trade(self, symbol, shares, price, date, action):
        cost, fee = abs(shares) * price, abs(shares) * price * self.transaction_cost
        if action in ["BUY", "SHORT"]:
            self.balance -= cost + fee
        else:
            self.balance += cost - fee
        self.positions[symbol] += shares
        self.trades.append(
            {
                "date": date,
                "symbol": symbol,
                "action": action,
                "shares": shares,
                "price": price,
                "fee": fee,
            }
        )
        print(
            f"{pd.to_datetime(date).date()}: {action} {abs(shares):.2f} {symbol} @ ${price:.2f}"
        )

    def get_total_value(self, current_prices):
        position_values = sum(
            self.positions[s] * current_prices.get(s, 0) for s in self.symbols
        )
        return self.balance + position_values


# ==============================================================================
# === LEGO BRICK 7: BACKTESTING ENGINE =========================================
# ==============================================================================
class EventDrivenBacktester:
    def __init__(self, data_handler, strategy, portfolio, start_date, end_date):
        self.data_handler, self.strategy, self.portfolio = (
            data_handler,
            strategy,
            portfolio,
        )
        self.start_date, self.end_date = (
            pd.to_datetime(start_date),
            pd.to_datetime(end_date),
        )
        self.signal_confidence_threshold = 0.5

    def run(self):
        print("\nRunning Event-Driven Backtest...")
        ref_symbol, ref_data = (
            self.data_handler.symbols[0],
            self.data_handler.get_data(self.data_handler.symbols[0]),
        )
        if ref_data is None:
            print("Cannot run backtest, no reference data available.")
            return
        date_range = ref_data.loc[self.start_date : self.end_date].index
        progress_interval = max(1, len(date_range) // 20)
        for i, date in enumerate(date_range):
            if i > 0 and i % progress_interval == 0:
                print(f"  Progress: {i / len(date_range) * 100:.1f}%")
            for symbol in self.data_handler.symbols:
                symbol_data = self.data_handler.get_data(symbol)
                if symbol_data is not None and date in symbol_data.index:
                    current_idx = symbol_data.index.get_loc(date)
                    if current_idx > 50:
                        signal_data = self.strategy.generate_signal(symbol, current_idx)
                        if signal_data["confidence"] > self.signal_confidence_threshold:
                            event = {
                                "date": date,
                                "symbol": symbol,
                                "signal": signal_data["signal"],
                                "price": symbol_data["Close"].iloc[current_idx],
                            }
                            self.portfolio.update(event)
        print("Backtest complete.")
        self.generate_results()

    def generate_results(self):
        print("\n" + "=" * 60 + "\nBACKTEST RESULTS\n" + "=" * 60)
        final_prices = {
            s: (
                d["Close"].iloc[-1]
                if (d := self.data_handler.get_latest_data(s)) is not None
                else 0
            )
            for s in self.portfolio.symbols
        }
        final_value = self.portfolio.get_total_value(final_prices)
        initial_value = self.portfolio.initial_balance
        total_return = (final_value - initial_value) / initial_value * 100
        print(f"Initial Portfolio Value: ${initial_value:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print(f"Total Return:            {total_return:.2f}%")
        print(f"Total Trades Executed:   {len(self.portfolio.trades)}")
        trades_df = pd.DataFrame(self.portfolio.trades)
        if not trades_df.empty:
            print("\nTrades Breakdown:\n", trades_df["action"].value_counts())


# ==============================================================================
# === THE GLUE: MAIN TRADING SYSTEM ============================================
# ==============================================================================
class TradingSystem:
    def __init__(self, symbols, strategy_config, portfolio_config, data_config):
        self.symbols = symbols
        self.data_handler = DataHandler(symbols, **data_config)
        self.feature_extractor = strategy_config["feature_extractor"]()
        self.labeling_strategy = strategy_config["labeling_strategy"]()
        self.strategy = MLStrategy(
            self.data_handler,
            self.feature_extractor,
            self.labeling_strategy,
            strategy_config["model_class"],
            **strategy_config["model_params"],
        )
        self.portfolio = Portfolio(symbols, **portfolio_config)

    async def initialize(self):
        await self.data_handler.fetch_all_data()
        self.strategy.pre_train()

    def run_backtest(self, start_date, end_date):
        EventDrivenBacktester(
            self.data_handler, self.strategy, self.portfolio, start_date, end_date
        ).run()

    def generate_live_signal(self, symbol):
        print(f"\nGenerating live signal for {symbol}...")
        if symbol not in self.data_handler.data:
            print("Error: No data available for symbol.")
            return None
        df, latest_idx = (
            self.data_handler.get_data(symbol),
            len(self.data_handler.get_data(symbol)) - 1,
        )
        signal, price = (
            self.strategy.generate_signal(symbol, latest_idx),
            df["Close"].iloc[latest_idx],
        )
        print(
            f"  -> Recommendation for {symbol}: {signal['signal']} (Confidence: {signal['confidence']:.2f}) at Price: ${price:.2f}"
        )
        return {
            "symbol": symbol,
            "recommendation": signal["signal"],
            "confidence": signal["confidence"],
            "price": price,
        }


# ==============================================================================
# === MAIN EXECUTION EXAMPLE ===================================================
# ==============================================================================
async def main():
    print("Initializing Modular Trading System...")
    system = TradingSystem(
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
        data_config={"period": "5y"},
        strategy_config={
            "feature_extractor": DefaultFeatureExtractor,
            "labeling_strategy": ForwardReturnLabeling,
            "model_class": RandomForestModel,
            "model_params": {"n_estimators": 75, "max_depth": 10},
        },
        portfolio_config={
            "initial_balance": 100000,
            "transaction_cost": 0.001,
            "max_position_size": 0.15,
        },
    )
    await system.initialize()
    system.run_backtest(start_date="2022-01-01", end_date="2023-12-31")
    for symbol in system.symbols:
        system.generate_live_signal(symbol)


if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
