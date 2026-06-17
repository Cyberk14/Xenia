import sys
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import asyncio
import warnings
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import ta

from datastream.yfinance_ohlcv import YFinanceOHLCVFetcher as DatastreamFetcher
from alpaca_markets import AlpacaTrader
from sp500 import sp500_tickers

warnings.filterwarnings("ignore")

@dataclass
class DataFetcherConfig:
    """Configuration for data fetcher"""

    symbol: str
    resolution: str = "1d"
    lookback_days: Optional[int] = None
    period: str = "5y"
    max_retries: int = 3
    rate_limit_interval: float = 1.0

@dataclass
class EmailConfig:
    """Configuration for email notifications"""
    enabled: bool = True
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = "void.ynd@gmail.com"
    send_on_signals: bool = True
    send_on_trades: bool = True
    send_on_errors: bool = True
    send_daily_summary: bool = True

@dataclass
class TradingConfig:
    """Configuration for trading parameters"""

    initial_balance: float = 10000
    transaction_cost: float = 0.002
    min_accuracy_threshold: float = 0.35
    signal_threshold: float = 0.15
    confidence_threshold: float = 0.4
    max_position_size: float = 0.2
    min_fetch_interval: float = 1.0

    # Trading parameters
    live_trading: bool = False
    enable_monte_carlo: bool = False
    monte_carlo_simulations: int = 10000
    forecast_days: int = 252
    
    # Timing parameters
    run_interval_hours: int = 24
    market_timezone: str = "US/Eastern"  # NYSE/NASDAQ timezone
    local_timezone: str = "Africa/Kampala"  # Uganda timezone
    
    # Portfolio parameters
    risk_per_trade: float = 0.1  # buying power per trade
    
    # API credentials (to be loaded from environment or config)
    api_key: str = ""
    api_secret: str = ""
    
    # Trading symbols
    symbols: list = None
    
    # Email configuration
    email_config: EmailConfig = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                "AAPL",  # AI investments + positioning in "Magnificent Seven"
                "MSFT",  # Cloud/AI infrastructure benefit
                "NVDA",  # Upgraded targets; Blackwell chips; continued AI tailwinds
                "AMZN",  # Heavy cloud spending, large-cap AI exposure
                # Add more symbols as needed
            ]
        
        if self.email_config is None:
            self.email_config = EmailConfig()


@dataclass
class ModelConfig:
    """Configuration for ML models"""

    model_type: str = "RandomForest"
    n_estimators: int = 50
    max_depth: int = 8
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    random_state: int = 42
    cv_splits: int = 3


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_spans: List[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_window: int = 14
    bb_window: int = 20
    bb_std: float = 2.0
    momentum_periods: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 20])


@dataclass
class LabelConfig:
    """Configuration for label creation"""

    forward_days: int = 3
    threshold: float = 0.01
    method: str = "threshold"  # 'threshold', 'quantile', 'volatility_adjusted'


# Abstract Base Classes
class DataFetcher(ABC):
    """Abstract base class for data fetchers"""

    @abstractmethod
    async def fetch_data(self, config: DataFetcherConfig) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        pass


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering"""

    @abstractmethod
    def create_features(
        self, df: pd.DataFrame, config: FeatureConfig, current_idx: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        pass


class LabelCreator(ABC):
    """Abstract base class for label creation"""

    @abstractmethod
    def create_labels(
        self, df: pd.DataFrame, config: LabelConfig
    ) -> Optional[np.ndarray]:
        pass


class ModelTrainer(ABC):
    """Abstract base class for model training"""

    @abstractmethod
    def train_model(
        self, X: np.ndarray, y: np.ndarray, config: ModelConfig
    ) -> Optional[Dict[str, Any]]:
        pass


class SignalGenerator(ABC):
    """Abstract base class for signal generation"""

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        current_idx: int,
        data: pd.DataFrame,
        model_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        pass


class TradeExecutor(ABC):
    """Abstract base class for trade execution"""

    @abstractmethod
    def execute_trade(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        price: float,
        date: datetime,
        positions: Dict[str, float],
        balance: float,
        config: TradingConfig,
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        pass


# Concrete Implementations
class YFinanceDataFetcher(DataFetcher):
    """YFinance data fetcher implementation"""

    def __init__(self):
        self.last_fetch_time = {}
        self.datastream_fetcher = DatastreamFetcher()

    def _rate_limit(self, symbol: str, interval: float):
        """Apply rate limiting"""
        now = time.time()
        if symbol in self.last_fetch_time:
            elapsed = now - self.last_fetch_time[symbol]
            if elapsed < interval:
                time.sleep(interval - elapsed)
        self.last_fetch_time[symbol] = time.time()

    def _fetch_yfinance_data(self, config: DataFetcherConfig) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance"""
        try:
            self._rate_limit(config.symbol, config.rate_limit_interval)

            for attempt in range(config.max_retries):
                try:
                    ticker = yf.Ticker(config.symbol)

                    # Validate ticker
                    try:
                        info = ticker.info
                        if info is None or len(info) == 0:
                            if attempt < config.max_retries - 1:
                                time.sleep(1)
                                continue
                    except Exception:
                        if attempt < config.max_retries - 1:
                            time.sleep(1)
                            continue

                    # Get historical data
                    data = ticker.history(period=config.period)

                    if data is None or data.empty:
                        if attempt < config.max_retries - 1:
                            time.sleep(1)
                            continue
                        return None

                    print(
                        f"✓ Successfully fetched {len(data)} days of data for {config.symbol}"
                    )
                    return data

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {config.symbol}: {e}")
                    if attempt < config.max_retries - 1:
                        time.sleep(2**attempt)
                        continue
                    else:
                        return None

            return None

        except Exception as e:
            print(f"Critical error fetching data for {config.symbol}: {e}")
            return None

    def _fetch_datastream_data(
        self, config: DataFetcherConfig
    ) -> Optional[pd.DataFrame]:
        """Fetch data using datastream fetcher"""
        try:
            if config.lookback_days:
                data = self.datastream_fetcher._fetch_ohlcv_sync(
                    config.symbol,
                    resolution=config.resolution,
                    lookback_days=config.lookback_days,
                )
                return data
            return None
        except Exception as e:
            print(f"Error fetching datastream data for {config.symbol}: {e}")
            return None

    async def fetch_data(self, config: DataFetcherConfig) -> Optional[pd.DataFrame]:
        """Fetch data asynchronously"""
        try:
            # Try datastream first if lookback_days is specified
            if config.lookback_days:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self._fetch_datastream_data, config
                )
                keys = {
                    'open': "Open",
                    'high': 'High',
                    'low': "Low",
                    'close': "Close",
                    'volume': "Volume"
                }

                data.rename(columns=keys, inplace=True)
                if data is not None and self.validate_data(data):
                    
                    return data
            
                # Fallback to yfinance
            data = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_yfinance_data, config
            )

            if data is not None and self.validate_data(data):
                return data

            return None

        except Exception as e:
            print(f"Async error fetching data for {config.symbol}: {e}")
            return None

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data"""
        if data is None or data.empty:
            return False

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_columns):
            return False

        if len(data) < 100:  # Minimum data requirement
            return False

        return True


class TechnicalFeatureEngineer(FeatureEngineer):
    """Technical analysis feature engineering implementation"""

    def create_features(
        self, df: pd.DataFrame, config: FeatureConfig, current_idx: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Create technical analysis features"""
        try:
            if current_idx is not None:
                df = df.iloc[: current_idx + 1].copy()

            if len(df) < 50:
                return None

            features = pd.DataFrame(index=df.index)

            # Basic price features
            features["returns"] = df["Close"].pct_change()
            features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

            # Simple moving averages
            for window in config.sma_windows:
                if len(df) > window:
                    features[f"sma_{window}"] = df["Close"].rolling(window).mean()
                    features[f"price_to_sma_{window}"] = (
                        df["Close"] / features[f"sma_{window}"]
                    )

            # Exponential moving averages
            for span in config.ema_spans:
                features[f"ema_{span}"] = df["Close"].ewm(span=span).mean()
                features[f"price_to_ema_{span}"] = df["Close"] / features[f"ema_{span}"]

            # RSI
            try:
                features["rsi"] = ta.momentum.RSIIndicator(
                    df["Close"], window=config.rsi_window
                ).rsi()
            except:
                # Fallback RSI calculation
                delta = df["Close"].diff()
                gain = (
                    (delta.where(delta > 0, 0)).rolling(window=config.rsi_window).mean()
                )
                loss = (
                    (-delta.where(delta < 0, 0))
                    .rolling(window=config.rsi_window)
                    .mean()
                )
                rs = gain / loss
                features["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            try:
                macd_line = ta.trend.MACD(df["Close"]).macd()
                macd_signal = ta.trend.MACD(df["Close"]).macd_signal()
                features["macd"] = macd_line
                features["macd_signal"] = macd_signal
                features["macd_histogram"] = macd_line - macd_signal
            except:
                # Fallback MACD
                ema12 = df["Close"].ewm(span=12).mean()
                ema26 = df["Close"].ewm(span=26).mean()
                features["macd"] = ema12 - ema26
                features["macd_signal"] = features["macd"].ewm(span=9).mean()
                features["macd_histogram"] = features["macd"] - features["macd_signal"]

            # Bollinger Bands
            sma_bb = df["Close"].rolling(config.bb_window).mean()
            std_bb = df["Close"].rolling(config.bb_window).std()
            features["bb_upper"] = sma_bb + (std_bb * config.bb_std)
            features["bb_lower"] = sma_bb - (std_bb * config.bb_std)
            features["bb_position"] = (df["Close"] - features["bb_lower"]) / (
                features["bb_upper"] - features["bb_lower"]
            )

            # Volume features
            features["volume"] = df["Volume"]
            features["volume_sma"] = df["Volume"].rolling(20).mean()
            features["volume_ratio"] = features["volume"] / features["volume_sma"]

            # Volatility
            features["volatility"] = df["Close"].pct_change().rolling(20).std()

            # Price momentum
            for period in config.momentum_periods:
                features[f"momentum_{period}"] = (
                    df["Close"] / df["Close"].shift(period) - 1
                )

            # Lag features
            for lag in config.lag_periods:
                features[f"returns_lag_{lag}"] = features["returns"].shift(lag)
                features[f"rsi_lag_{lag}"] = features["rsi"].shift(lag)

            # Rolling statistics
            for window in config.rolling_windows:
                features[f"returns_mean_{window}"] = (
                    features["returns"].rolling(window).mean()
                )
                features[f"returns_std_{window}"] = (
                    features["returns"].rolling(window).std()
                )

            return features

        except Exception as e:
            print(f"Error creating features: {e}")
            return None


class ThresholdLabelCreator(LabelCreator):
    """Threshold-based label creation implementation"""

    def create_labels(
        self, df: pd.DataFrame, config: LabelConfig
    ) -> Optional[np.ndarray]:
        """Create labels based on threshold method"""
        try:
            close_prices = df["Close"].values
            labels = np.full(len(df), 0)  # Default to hold

            for i in range(len(df) - config.forward_days):
                current_price = close_prices[i]
                future_price = close_prices[i + config.forward_days]

                return_pct = (future_price - current_price) / current_price

                if return_pct > config.threshold:
                    labels[i] = 1  # Buy
                elif return_pct < -config.threshold:
                    labels[i] = -1  # Sell
                # else: 0 (Hold)

            return labels

        except Exception as e:
            print(f"Error creating labels: {e}")
            return None


class RandomForestModelTrainer(ModelTrainer):
    """Random Forest model trainer implementation"""

    def train_model(
        self, X: np.ndarray, y: np.ndarray, config: ModelConfig
    ) -> Optional[Dict[str, Any]]:
        """Train Random Forest model"""
        try:
            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2 or min(counts) < 10:
                return None

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Create model
            if config.model_type == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_split=config.min_samples_split,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                    n_jobs=1,
                )
            elif config.model_type == "GradientBoosting":
                model = GradientBoostingClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    random_state=config.random_state,
                )
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=config.cv_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

            avg_score = np.mean(scores)

            # Train final model on all data
            model.fit(X_scaled, y)

            return {
                "model": model,
                "scaler": scaler,
                "accuracy": avg_score,
                "class_distribution": dict(zip(unique, counts)),
            }

        except Exception as e:
            print(f"Error training model: {e}")
            return None


class MLSignalGenerator(SignalGenerator):
    """Machine Learning signal generator implementation"""

    def __init__(
        self, feature_engineer: FeatureEngineer, feature_config: FeatureConfig
    ):
        self.feature_engineer = feature_engineer
        self.feature_config = feature_config
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()

    def generate_signal(
        self,
        symbol: str,
        current_idx: int,
        data: pd.DataFrame,
        model_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Generate ML-based signal"""
        try:
            # Check cache
            cache_key = f"{symbol}_{current_idx}"
            with self.cache_lock:
                if cache_key in self.prediction_cache:
                    return self.prediction_cache[cache_key]

            if model_data is None:
                return 0.0, 0.5

            # Create features
            features_df = self.feature_engineer.create_features(
                data, self.feature_config, current_idx
            )
            if features_df is None or current_idx >= len(features_df):
                return 0.0, 0.5

            # Get feature columns from model training
            feature_cols = [
                col
                for col in features_df.columns
                if col in model_data.get("feature_cols", [])
            ]
            if not feature_cols:
                # Use all features if feature_cols not available
                feature_cols = list(features_df.columns)

            # Get current features
            current_features = features_df[feature_cols].iloc[
                current_idx : current_idx + 1
            ]

            # Check for NaN values
            if current_features.isnull().any().any():
                return 0.0, 0.5

            # Scale and predict
            X_scaled = model_data["scaler"].transform(current_features.values)
            model = model_data["model"]
            prediction = model.predict(X_scaled)[0]

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_scaled)[0]
                confidence = np.max(probas)
            else:
                confidence = 0.6

            signal = float(prediction * 0.5)
            confidence = float(confidence)

            # Cache result
            with self.cache_lock:
                self.prediction_cache[cache_key] = (signal, confidence)
                # Limit cache size
                if len(self.prediction_cache) > 1000:
                    keys_to_remove = list(self.prediction_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.prediction_cache[key]

            return signal, confidence

        except Exception as e:
            print(f"Error generating ML signal for {symbol}: {e}")
            return 0.0, 0.5


class TechnicalSignalGenerator(SignalGenerator):
    """Technical analysis signal generator implementation"""

    def __init__(
        self, feature_engineer: FeatureEngineer, feature_config: FeatureConfig
    ):
        self.feature_engineer = feature_engineer
        self.feature_config = feature_config

    def generate_signal(
        self,
        symbol: str,
        current_idx: int,
        data: pd.DataFrame,
        model_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Generate technical analysis signal"""
        try:
            features_df = self.feature_engineer.create_features(
                data, self.feature_config, current_idx
            )
            if features_df is None or current_idx >= len(features_df):
                return 0.0, 0.5

            current_data = features_df.iloc[current_idx]
            signals = []

            # RSI signal
            rsi = current_data.get("rsi", 50)
            if pd.notna(rsi):
                if rsi < 30:
                    signals.append(0.7)  # Oversold - buy
                elif rsi > 70:
                    signals.append(-0.7)  # Overbought - sell
                else:
                    signals.append(0)

            # MACD signal
            macd = current_data.get("macd", 0)
            macd_signal = current_data.get("macd_signal", 0)
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal:
                    signals.append(0.5)
                elif macd < macd_signal:
                    signals.append(-0.5)
                else:
                    signals.append(0)

            # Moving average signal
            price_to_sma_20 = current_data.get("price_to_sma_20", 1)
            if pd.notna(price_to_sma_20):
                if price_to_sma_20 > 1.02:
                    signals.append(0.4)
                elif price_to_sma_20 < 0.98:
                    signals.append(-0.4)
                else:
                    signals.append(0)

            # Bollinger Bands signal
            bb_position = current_data.get("bb_position", 0.5)
            if pd.notna(bb_position):
                if bb_position < 0.2:
                    signals.append(0.6)  # Near lower band - buy
                elif bb_position > 0.8:
                    signals.append(-0.6)  # Near upper band - sell
                else:
                    signals.append(0)

            # Average the signals
            avg_signal = np.mean(signals) if signals else 0
            confidence = 0.6  # Fixed confidence for technical signals

            return float(avg_signal), float(confidence)

        except Exception as e:
            print(f"Error generating technical signal for {symbol}: {e}")
            return 0.0, 0.5


class BasicTradeExecutor(TradeExecutor):
    """Basic trade executor implementation with fixed P&L calculation"""

    def __init__(self):
        # Track purchase prices for accurate P&L calculation
        self.purchase_prices = {}  # symbol -> average purchase price

    def execute_trade(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        price: float,
        date: datetime,
        positions: Dict[str, float],
        balance: float,
        config: TradingConfig,
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """Execute trade based on signal"""
        try:
            trades = []
            new_positions = positions.copy()
            new_balance = balance

            # Trading logic
            if (
                signal > config.signal_threshold
                and confidence > config.confidence_threshold
            ):
                # Buy signal
                if new_positions.get(symbol, 0) == 0:  # Only if not holding
                    new_positions, new_balance, trade = self._execute_buy(
                        symbol,
                        price,
                        date,
                        signal,
                        confidence,
                        new_positions,
                        new_balance,
                        config,
                    )
                    if trade:
                        trades.append(trade)

            elif (
                signal < -config.signal_threshold
                and confidence > config.confidence_threshold
            ):
                # Sell signal
                if new_positions.get(symbol, 0) > 0:  # Only if holding
                    new_positions, new_balance, trade = self._execute_sell(
                        symbol,
                        price,
                        date,
                        signal,
                        confidence,
                        new_positions,
                        new_balance,
                        config,
                    )
                    if trade:
                        trades.append(trade)

            return new_positions, new_balance, trades

        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")
            return positions, balance, []

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        date: datetime,
        signal: float,
        confidence: float,
        positions: Dict[str, float],
        balance: float,
        config: TradingConfig,
    ) -> Tuple[Dict[str, float], float, Optional[Dict]]:
        """Execute buy order"""
        try:
            # Calculate position size
            position_value = balance * config.max_position_size
            shares = position_value / price
            cost = shares * price
            transaction_fee = cost * config.transaction_cost
            total_cost = cost + transaction_fee

            if total_cost <= balance:
                new_positions = positions.copy()
                new_positions[symbol] = shares
                new_balance = balance - total_cost

                # Store the purchase price for P&L calculation
                self.purchase_prices[symbol] = price

                trade = {
                    "symbol": symbol,
                    "action": "BUY",
                    "price": price,
                    "shares": shares,
                    "cost": total_cost,
                    "signal": signal,
                    "confidence": confidence,
                    "date": date,
                    "balance_after": new_balance,
                }

                # print(
                #     f"BUY {symbol}: {shares:.2f} shares @ ${price:.2f} | Signal: {signal:.3f} | Confidence: {confidence:.3f}"
                # )

                return new_positions, new_balance, trade

            return positions, balance, None

        except Exception as e:
            print(f"Error executing buy for {symbol}: {e}")
            return positions, balance, None

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        date: datetime,
        signal: float,
        confidence: float,
        positions: Dict[str, float],
        balance: float,
        config: TradingConfig,
    ) -> Tuple[Dict[str, float], float, Optional[Dict]]:
        """Execute sell order"""
        try:
            if positions.get(symbol, 0) > 0:
                shares = positions[symbol]
                revenue = shares * price
                transaction_fee = revenue * config.transaction_cost
                net_revenue = revenue - transaction_fee

                new_positions = positions.copy()
                new_positions[symbol] = 0
                new_balance = balance + net_revenue

                # Calculate P&L using stored purchase price
                purchase_price = self.purchase_prices.get(
                    symbol, price
                )  
                cost_basis = shares * purchase_price
                pnl = net_revenue - cost_basis
                pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

                # Clear the purchase price since we're selling all shares
                if symbol in self.purchase_prices:
                    del self.purchase_prices[symbol]

                trade = {
                    "symbol": symbol,
                    "action": "SELL",
                    "price": price,
                    "shares": shares,
                    "revenue": net_revenue,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "signal": signal,
                    "confidence": confidence,
                    "date": date,
                    "balance_after": new_balance,
                    "purchase_price": purchase_price,  # Added for debugging
                }

                # print(
                #     f"SELL {symbol}: {shares:.2f} shares @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Bought @ ${purchase_price:.2f}"
                # )

                return new_positions, new_balance, trade

            return positions, balance, None

        except Exception as e:
            print(f"Error executing sell for {symbol}: {e}")
            return positions, balance, None

# SET STYLE
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ADD THESE CLASSES TO YOUR EXISTING CODE

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd


class MonteCarloSimulator:
    """
    🎲 Advanced Monte Carlo Simulation Engine for Portfolio Risk Analysis
    
    Provides sophisticated statistical modeling for portfolio forecasting
    and risk assessment using Monte Carlo methods.
    """
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.simulations = []
        self.color_palette = {
            'primary': '#00D4AA',
            'secondary': '#6C7CE7',
            'accent': '#FF6B9D',
            'warning': '#FFB800',
            'success': '#00E676',
            'error': '#FF5252'
        }
        
    def run_monte_carlo(self, num_simulations=1000, forecast_days=252):
        """
        Execute Monte Carlo simulation suite
        
        Args:
            num_simulations (int): Number of simulation paths to generate
            forecast_days (int): Trading days to forecast (default: 252 = 1 year)
        
        Returns:
            list: Complete simulation results with statistical metrics
        """
        print(f"\n🎲 Initializing Monte Carlo Engine...")
        print(f"📊 Running {num_simulations:,} simulations over {forecast_days} days")
        print(f"⚡ Processing simulation paths...")
        
        simulations = []
        progress_milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for sim in range(num_simulations):
            progress = (sim + 1) / num_simulations
            
            # Update progress bar in place
            percentage = progress * 100
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r🔄 [{bar}] {percentage:.1f}% Complete", end="")
            
            sim_result = self._run_single_simulation(forecast_days)
            simulations.append(sim_result)

        # Print final bar and newline when done
        print(f"\r🔄 [{bar}] {percentage:.1f}% Complete")
            
        print(f"✅ Monte Carlo simulation complete!")
        print(f"📈 Generated {len(simulations):,} portfolio paths")
        
        self.simulations = simulations
        return simulations
    
    def _run_single_simulation(self, forecast_days):
        """Execute individual simulation path with enhanced risk modeling"""
        portfolio = self.trading_system.get_portfolio_status()
        initial_value = portfolio['total_value']
        
        # Enhanced return calculation with volatility clustering
        daily_returns = {}
        for symbol in self.trading_system.symbols:
            if symbol in self.trading_system.data:
                prices = self.trading_system.data[symbol]['Close']
                returns = prices.pct_change().dropna()
                
                # Calculate advanced statistics
                daily_returns[symbol] = {
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'skew': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'last_price': prices.iloc[-1],
                    'volatility_regime': self._detect_volatility_regime(returns)
                }
        
        # Enhanced portfolio simulation with regime switching
        portfolio_values = [initial_value]
        current_value = initial_value
        daily_returns_series = []
        
        for day in range(forecast_days):
            portfolio_return = 0
            total_weight = 0
            
            # Calculate position-weighted returns with correlation effects
            for symbol in self.trading_system.symbols:
                if symbol in self.trading_system.positions and self.trading_system.positions[symbol] > 0:
                    if symbol in daily_returns:
                        weight = (self.trading_system.positions[symbol] * daily_returns[symbol]['last_price']) / current_value
                        
                        # Enhanced return simulation with fat tails
                        base_return = daily_returns[symbol]['mean']
                        volatility = daily_returns[symbol]['std']
                        
                        # Apply volatility regime adjustment
                        if daily_returns[symbol]['volatility_regime'] == 'high':
                            volatility *= 1.5
                        
                        simulated_return = np.random.normal(base_return, volatility)
                        
                        # Add occasional extreme events (fat tails)
                        if np.random.random() < 0.05:  # 5% chance of extreme event
                            simulated_return *= np.random.choice([-2, 2])
                        
                        portfolio_return += weight * simulated_return
                        total_weight += weight
            
            daily_returns_series.append(portfolio_return)
            current_value *= (1 + portfolio_return)
            portfolio_values.append(current_value)
        
        return {
            'values': portfolio_values,
            'returns': daily_returns_series,
            'final_value': current_value,
            'total_return': (current_value - initial_value) / initial_value * 100,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'volatility': np.std(daily_returns_series) * np.sqrt(252) * 100,
            'var_95': np.percentile(daily_returns_series, 5) * 100,
            'cvar_95': np.mean([r for r in daily_returns_series if r <= np.percentile(daily_returns_series, 5)]) * 100
        }
    
    def _detect_volatility_regime(self, returns):
        """Detect current volatility regime"""
        recent_vol = returns.tail(30).std()
        historical_vol = returns.std()
        
        return 'high' if recent_vol > historical_vol * 1.5 else 'normal'
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown with enhanced precision"""
        if len(values) < 2:
            return 0
            
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
                
        return max_dd * 100
    
    def get_statistics(self):
        """Generate comprehensive Monte Carlo statistics"""
        if not self.simulations:
            return None
            
        final_values = [sim['final_value'] for sim in self.simulations]
        returns = [sim['total_return'] for sim in self.simulations]
        drawdowns = [sim['max_drawdown'] for sim in self.simulations]
        volatilities = [sim['volatility'] for sim in self.simulations]
        vars_95 = [sim['var_95'] for sim in self.simulations]
        cvars_95 = [sim['cvar_95'] for sim in self.simulations]
        
        return {
            'simulation_count': len(self.simulations),
            'final_value': {
                'mean': np.mean(final_values),
                'median': np.median(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'percentile_5': np.percentile(final_values, 5),
                'percentile_10': np.percentile(final_values, 10),
                'percentile_25': np.percentile(final_values, 25),
                'percentile_75': np.percentile(final_values, 75),
                'percentile_90': np.percentile(final_values, 90),
                'percentile_95': np.percentile(final_values, 95)
            },
            'returns': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_10': np.percentile(returns, 10),
                'percentile_25': np.percentile(returns, 25),
                'percentile_75': np.percentile(returns, 75),
                'percentile_90': np.percentile(returns, 90),
                'percentile_95': np.percentile(returns, 95)
            },
            'risk_metrics': {
                'max_drawdown': {
                    'mean': np.mean(drawdowns),
                    'median': np.median(drawdowns),
                    'std': np.std(drawdowns),
                    'max': np.max(drawdowns),
                    'percentile_95': np.percentile(drawdowns, 95)
                },
                'volatility': {
                    'mean': np.mean(volatilities),
                    'median': np.median(volatilities),
                    'std': np.std(volatilities)
                },
                'var_95': {
                    'mean': np.mean(vars_95),
                    'median': np.median(vars_95)
                },
                'cvar_95': {
                    'mean': np.mean(cvars_95),
                    'median': np.median(cvars_95)
                }
            },
            'probability_positive': len([r for r in returns if r > 0]) / len(returns) * 100,
            'probability_loss_10': len([r for r in returns if r < -10]) / len(returns) * 100,
            'probability_gain_20': len([r for r in returns if r > 20]) / len(returns) * 100,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'sortino_ratio': np.mean(returns) / np.std([r for r in returns if r < 0]) if len([r for r in returns if r < 0]) > 0 else 0
        }

# Main Trading System
class XeniaV2Modular:
    """Modular version of Xenia V2 Trading System"""

    def __init__(
        self,
        symbols: List[str],
        data_fetcher: DataFetcher,
        feature_engineer: FeatureEngineer,
        label_creator: LabelCreator,
        model_trainer: ModelTrainer,
        signal_generators: List[SignalGenerator],
        trade_executor: TradeExecutor,
        trading_config: TradingConfig,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        model_config: ModelConfig,
        data_fetcher_configs: Dict[str, DataFetcherConfig],
    ):
        self.symbols = symbols
        self.data_fetcher = data_fetcher
        self.feature_engineer = feature_engineer
        self.label_creator = label_creator
        self.model_trainer = model_trainer
        self.signal_generators = signal_generators
        self.trade_executor = trade_executor

        # Configurations
        self.trading_config = trading_config
        self.feature_config = feature_config
        self.label_config = label_config
        self.model_config = model_config
        self.data_fetcher_configs = data_fetcher_configs

        # State
        self.balance = trading_config.initial_balance
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.data = {}
        self.models = {}
        self.trades = []
        self.win_trades = []
        self.lose_trades = []
        self.latest_signals = {}

        # Thread safety
        self.data_lock = threading.Lock()
        self.model_lock = threading.Lock()

    async def fetch_all_data(self):
        """Fetch data for all symbols"""
        with self.data_lock:
            tasks = []
            for symbol in self.symbols:
                config = self.data_fetcher_configs.get(
                    symbol, DataFetcherConfig(symbol=symbol)
                )
                tasks.append(self.data_fetcher.fetch_data(config))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                symbol = self.symbols[i]
                if isinstance(result, Exception):
                    print(f"Exception fetching data for {symbol}: {result}")
                    continue

                if result is not None:
                    self.data[symbol] = result
                    print(f"✓ Stored {len(result)} days of data for {symbol}")
                else:
                    print(f"✗ Failed to fetch data for {symbol}")

    def train_model_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Train model for a specific symbol"""
        try:
            with self.model_lock:
                print(f"Training model for {symbol}...")

                if symbol not in self.data:
                    print(f"No data available for {symbol}")
                    return None

                df = self.data[symbol]

                # Create features
                features_df = self.feature_engineer.create_features(
                    df, self.feature_config
                )
                if features_df is None:
                    print(f"Failed to create features for {symbol}")
                    return None

                # Create labels
                labels = self.label_creator.create_labels(df, self.label_config)
                if labels is None:
                    print(f"Failed to create labels for {symbol}")
                    return None

                # Combine and clean
                combined_df = features_df.copy()
                combined_df["target"] = labels
                combined_df = combined_df.dropna()

                print(f"{symbol}: {len(combined_df)} samples after cleaning")

                if len(combined_df) < 50:
                    print(f"{symbol}: Insufficient data after cleaning")
                    return None

                # Prepare features
                feature_cols = [col for col in combined_df.columns if col != "target"]
                X = combined_df[feature_cols].values
                y = combined_df["target"].values

                # Train model
                model_data = self.model_trainer.train_model(X, y, self.model_config)
                if model_data is None:
                    print(f"{symbol}: Model training failed")
                    return None

                # Add feature columns to model data
                model_data["feature_cols"] = feature_cols

                print(
                    f"{symbol}: Model trained successfully (accuracy: {model_data['accuracy']:.3f})"
                )
                return model_data

        except Exception as e:
            print(f"Error training model for {symbol}: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def train_all_models(self):
        """Train models for all symbols"""
        print("Training models for all symbols...")

        for symbol in self.symbols:
            if symbol in self.data:
                model_data = self.train_model_for_symbol(symbol)
                if model_data:
                    self.models[symbol] = model_data
                    print(f"✓ {symbol}: Model trained successfully")
                else:
                    print(f"✗ {symbol}: Model training failed")

    def get_combined_signal(self, symbol: str, current_idx: int) -> Tuple[float, float]:
        """Get combined signal from all signal generators"""
        try:
            if symbol not in self.data:
                return 0.0, 0.5

            signals = []
            confidences = []

            for generator in self.signal_generators:
                model_data = (
                    self.models.get(symbol)
                    if isinstance(generator, MLSignalGenerator)
                    else None
                )
                signal, confidence = generator.generate_signal(
                    symbol, current_idx, self.data[symbol], model_data
                )
                signals.append(signal)
                confidences.append(confidence)

            # Weight signals equally for now (could be configurable)
            if signals:
                combined_signal = np.mean(signals)
                combined_confidence = np.mean(confidences)
            else:
                combined_signal = 0.0
                combined_confidence = 0.5

            return float(combined_signal), float(combined_confidence)

        except Exception as e:
            print(f"Error getting combined signal for {symbol}: {e}")
            return 0.0, 0.5

    def execute_trade_for_symbol(self, symbol: str, current_idx: int):
        """Execute trade for a specific symbol"""
        try:
            if symbol not in self.data or current_idx >= len(self.data[symbol]):
                return

            current_price = self.data[symbol]["Close"].iloc[current_idx]
            current_date = self.data[symbol].index[current_idx]

            # Get combined signal
            combined_signal, combined_confidence = self.get_combined_signal(
                symbol, current_idx
            )

            # Store signal for monitoring
            self.latest_signals[symbol] = {
                "combined_signal": combined_signal,
                "confidence": combined_confidence,
                "price": current_price,
                "date": current_date,
            }

            # Execute trade
            new_positions, new_balance, new_trades = self.trade_executor.execute_trade(
                symbol,
                combined_signal,
                combined_confidence,
                current_price,
                current_date,
                self.positions,
                self.balance,
                self.trading_config,
            )

            # Update state
            self.positions = new_positions
            self.balance = new_balance
            self.trades.extend(new_trades)

            # Categorize trades
            for trade in new_trades:
                if trade["action"] == "SELL":
                    if trade.get("pnl", 0) > 0:
                        self.win_trades.append(trade)
                    else:
                        self.lose_trades.append(trade)

        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")
        
    async def run_backtest(self, start_date=None, end_date=None):
        """
        Run backtest on all available data or within specified date range
        
        Args:
            start_date (str or datetime, optional): Start date for backtest (e.g., '2023-01-01')
            end_date (str or datetime, optional): End date for backtest (e.g., '2023-12-31')
        """
        print("\033[96m" + "▓" * 80 + "\033[0m")
        print("\033[96m▓\033[0m" + " " * 78 + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + "\033[92m    ██╗  ██╗███████╗███╗   ██╗██╗ █████╗     ◢◤◢◤◢◤    \033[0m" + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + "\033[92m    ╚██╗██╔╝██╔════╝████╗  ██║██║██╔══██╗    ◢◤◢◤◢◤    \033[0m" + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + "\033[92m     ╚███╔╝ █████╗  ██╔██╗ ██║██║███████║    ◢◤◢◤◢◤    \033[0m" + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + "\033[92m     ██╔██╗ ██╔══╝  ██║╚██╗██║██║██╔══██║    ◢◤◢◤◢◤    \033[0m" + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + "\033[92m    ██╔╝ ██╗███████╗██║ ╚████║██║██║  ██║    ◢◤◢◤◢◤    \033[0m" + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + "\033[92m    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝    ◢◤◢◤◢◤    \033[0m" + "\033[96m▓\033[0m")
        print("\033[96m▓\033[0m" + " " * 78 + "\033[96m▓\033[0m")
        print("\033[96m" + "▓" * 80 + "\033[0m")

        time.sleep(2.0)

        print("\033[93m⚡ INITIALIZING QUANTUM DATA STREAMS...\033[0m")
        await self.fetch_all_data()

        if not self.data:
            print("\033[91m❌ NEURAL NETWORK OFFLINE - NO DATA DETECTED\033[0m")
            return

        print("\033[93m🧠 TRAINING NEURAL MATRICES...\033[0m")
        await self.train_all_models()

        print("\033[92m✓ SYSTEMS ONLINE - COMMENCING TEMPORAL ANALYSIS\033[0m")

        max_symbol = max(self.data.keys(), key=lambda x: len(self.data[x]))
        reference_data = self.data[max_symbol]

        # Filter data by date range if specified
        if start_date is not None or end_date is not None:
            import pandas as pd
            
            # Convert string dates to datetime if needed
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
            
            # Get timezone info from the data index
            data_tz = None
            if hasattr(reference_data.index, 'tz') and reference_data.index.tz is not None:
                data_tz = reference_data.index.tz
            
            # Localize dates to match data timezone
            if data_tz is not None:
                if start_date is not None:
                    if hasattr(start_date, 'tz') and start_date.tz is None:
                        # pandas Timestamp with no timezone
                        start_date = start_date.tz_localize(data_tz)
                    elif hasattr(start_date, 'tz') and start_date.tz != data_tz:
                        # pandas Timestamp with different timezone
                        start_date = start_date.tz_convert(data_tz)
                    elif hasattr(start_date, 'tzinfo') and start_date.tzinfo is None:
                        # datetime.datetime with no timezone
                        start_date = pd.Timestamp(start_date).tz_localize(data_tz)
                    elif hasattr(start_date, 'tzinfo') and start_date.tzinfo != data_tz:
                        # datetime.datetime with different timezone
                        start_date = pd.Timestamp(start_date).tz_convert(data_tz)
                    
                if end_date is not None:
                    if hasattr(end_date, 'tz') and end_date.tz is None:
                        # pandas Timestamp with no timezone
                        end_date = end_date.tz_localize(data_tz)
                    elif hasattr(end_date, 'tz') and end_date.tz != data_tz:
                        # pandas Timestamp with different timezone
                        end_date = end_date.tz_convert(data_tz)
                    elif hasattr(end_date, 'tzinfo') and end_date.tzinfo is None:
                        # datetime.datetime with no timezone
                        end_date = pd.Timestamp(end_date).tz_localize(data_tz)
                    elif hasattr(end_date, 'tzinfo') and end_date.tzinfo != data_tz:
                        # datetime.datetime with different timezone
                        end_date = pd.Timestamp(end_date).tz_convert(data_tz)
            
            # Filter reference data
            if start_date is not None and end_date is not None:
                reference_data = reference_data[(reference_data.index >= start_date) & (reference_data.index <= end_date)]
                date_range_msg = f"FROM {start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}"
            elif start_date is not None:
                reference_data = reference_data[reference_data.index >= start_date]
                date_range_msg = f"FROM {start_date.strftime('%Y-%m-%d')} ONWARDS"
            elif end_date is not None:
                reference_data = reference_data[reference_data.index <= end_date]
                date_range_msg = f"UP TO {end_date.strftime('%Y-%m-%d')}"
            
            print(f"\033[95m📅 DATE RANGE FILTER APPLIED: {date_range_msg}\033[0m")
            
            # Also filter all symbol data to match the date range
            for symbol in self.data:
                if start_date is not None and end_date is not None:
                    self.data[symbol] = self.data[symbol][(self.data[symbol].index >= start_date) & (self.data[symbol].index <= end_date)]
                elif start_date is not None:
                    self.data[symbol] = self.data[symbol][self.data[symbol].index >= start_date]
                elif end_date is not None:
                    self.data[symbol] = self.data[symbol][self.data[symbol].index <= end_date]

        print(f"\033[94m🔍 SCANNING {len(reference_data)} TEMPORAL NODES\033[0m")

        if len(reference_data) == 0:
            print("\033[91m❌ TEMPORAL MATRIX CORRUPTED - NO DATA IN SPECIFIED RANGE\033[0m")
            return

        total_days = len(reference_data)
        bar_width = 60
        start_time = time.time()

        print("\033[96m\n⟦ QUANTUM PROCESSING STATUS ⟧\033[0m")

        time_samples = []

        for i, date in enumerate(reference_data.index):
            loop_start = time.time()

            for symbol in self.symbols:
                if symbol in self.data and date in self.data[symbol].index:
                    try:
                        symbol_idx = self.data[symbol].index.get_loc(date)
                        if symbol_idx >= 50:
                            self.execute_trade_for_symbol(symbol, symbol_idx)
                    except Exception as e:
                        print(f"\033[91m⚠️ {symbol} {date}: {e}\033[0m")
                        continue

            progress = (i + 1) / total_days
            filled_width = int(bar_width * progress)

            bar_parts = []
            for j in range(bar_width):
                relative_pos = j / bar_width
                if j < filled_width:
                    if relative_pos < 0.15:
                        bar_parts.append('\033[38;5;196m█\033[0m')  # Bright Red
                    elif relative_pos < 0.30:
                        bar_parts.append('\033[38;5;202m█\033[0m')  # Orange
                    elif relative_pos < 0.45:
                        bar_parts.append('\033[38;5;220m█\033[0m')  # Yellow
                    elif relative_pos < 0.60:
                        bar_parts.append('\033[38;5;45m█\033[0m')   # Cyan
                    elif relative_pos < 0.75:
                        bar_parts.append('\033[38;5;51m█\033[0m')   # Blue-Cyan
                    else:
                        bar_parts.append('\033[38;5;82m█\033[0m')   # Bright Green
                else:
                    if j % 7 == 0:
                        bar_parts.append('\033[90m┊\033[0m')
                    elif j % 3 == 0:
                        bar_parts.append('\033[90m▒\033[0m')
                    else:
                        bar_parts.append('\033[90m░\033[0m')

            bar = ''.join(bar_parts)

            elapsed_time = time.time() - start_time
            time_samples.append(time.time() - loop_start)

            if len(time_samples) > 5:
                avg_time = sum(time_samples[-20:]) / min(20, len(time_samples))
                eta_seconds = avg_time * (total_days - (i + 1))
            else:
                eta_seconds = None

            def format_time(seconds):
                if seconds is None:
                    return "CALC..."
                elif seconds > 3600:
                    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
                elif seconds > 60:
                    return f"{int(seconds // 60)}m {int(seconds % 60)}s"
                else:
                    return f"{int(seconds)}s"

            eta_display = format_time(eta_seconds)
            elapsed_display = format_time(elapsed_time)

            percent = progress * 100
            pulse = '◉' if i % 2 == 0 else '◎'
            status = "ANALYZING" if percent < 100 else "COMPLETE"
            scan_indicator = '►' if (i % 40) < 20 else '◄'

            print(f'\033[2K\r\033[96m⟦\033[0m{bar}\033[96m⟧\033[0m \033[93m{percent:.1f}%\033[0m \033[94m[{i+1:04d}/{total_days:04d}]\033[0m \033[95m{pulse}{status}{pulse}\033[0m \033[97m{scan_indicator}\033[0m \033[90m|\033[0m \033[97mETA: {eta_display}\033[0m \033[90m|\033[0m \033[97mElapsed: {elapsed_display}\033[0m', end='', flush=True)

        total_time = time.time() - start_time
        final_display = format_time(total_time)

        print(f"\n\033[92m✓ QUANTUM ANALYSIS COMPLETE - TOTAL TIME: {final_display}\033[0m")
        print("\033[92m✓ GENERATING NEURAL INSIGHTS\033[0m")

        self.generate_results()

    def run_professional_analysis(self, num_monte_carlo=10000, forecast_days=252):
        """Run comprehensive professional analysis"""
        print("\n🔬 Running Professional Analysis...")
        
        # Initialize Monte Carlo simulator
        monte_carlo = MonteCarloSimulator(self)
        
        # Run Monte Carlo simulations
        monte_carlo.run_monte_carlo(num_monte_carlo, forecast_days)
        
        # Get statistics
        mc_stats = monte_carlo.get_statistics()
        import sys
        if mc_stats:
            print(f"\n📊 Monte Carlo Results ({num_monte_carlo} simulations):")
            print(f"Expected Return: {mc_stats['returns']['mean']:.2f}% ± {mc_stats['returns']['std']:.2f}%")
            print(f"5th Percentile: {mc_stats['returns']['percentile_5']:.2f}%")
            print(f"95th Percentile: {mc_stats['returns']['percentile_95']:.2f}%")
            print(f"Probability of Positive Return: {mc_stats['probability_positive']:.1f}%")
            print(f"Expected Max Drawdown: {mc_stats['risk_metrics']['max_drawdown']['mean']:.2f}%")
        
        # Create professional visualizations
        visualizer = ProfessionalVisualizer(self, monte_carlo)
        
        # Generate executive dashboard
        visualizer.create_executive_dashboard('executive_dashboard.html')
        
        # Generate investor report
        visualizer.create_investor_report('investor_report.md')
        
        print("\n✅ Professional analysis complete!")
        print("📊 Executive dashboard saved as 'executive_dashboard.html'")
        print("📋 Investor report saved as 'investor_report.md'")
        
        return monte_carlo


    def generate_results(self):
        """Generate comprehensive results"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        if not self.trades:
            print("No trades executed during backtest period")
            print("\nDebugging Info:")
            print("Latest signals for each symbol:")
            for symbol, signal_data in self.latest_signals.items():
                print(
                    f"{symbol}: Combined signal: {signal_data['combined_signal']:.3f}, "
                    f"Confidence: {signal_data['confidence']:.3f}"
                )
            return

        # Calculate final portfolio value
        final_balance = self.balance
        portfolio_value = 0

        for symbol in self.symbols:
            if symbol in self.data and self.positions[symbol] > 0:
                current_price = self.data[symbol]["Close"].iloc[-1]
                position_value = self.positions[symbol] * current_price
                portfolio_value += position_value

        total_value = final_balance + portfolio_value
        total_return = (
            (total_value - self.trading_config.initial_balance)
            / self.trading_config.initial_balance
            * 100
        )
        total_balance_return = (
            (final_balance - self.trading_config.initial_balance)
            / self.trading_config.initial_balance
            * 100
        )

        print(f"Initial Balance: ${self.trading_config.initial_balance:,.2f}")
        print(f"Final Cash: ${final_balance:,.2f}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Realized Return: {total_balance_return:.2f}%")
        print(f"Total Trades: {len(self.trades)}")

        # Trade statistics
        buy_trades = [t for t in self.trades if t["action"] == "BUY"]
        sell_trades = [t for t in self.trades if t["action"] == "SELL"]

        print(f"Buy Trades: {len(buy_trades)}")
        print(f"Sell Trades: {len(sell_trades)}")
        print(f"Profitable Trades: {len(self.win_trades)}")
        print(f"Losing Trades: {len(self.lose_trades)}")

        if sell_trades:
            profitable_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
            win_rate = len(profitable_trades) / len(sell_trades) * 100
            avg_profit = np.mean([t.get("pnl", 0) for t in sell_trades])
            avg_profit_pct = np.mean([t.get("pnl_pct", 0) for t in sell_trades])

            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average P&L per trade: ${avg_profit:.2f} ({avg_profit_pct:.2f}%)")

            if profitable_trades:
                avg_win = np.mean([t.get("pnl", 0) for t in profitable_trades])
                print(f"Average Winning Trade: ${avg_win:.2f}")

            losing_trades = [t for t in sell_trades if t.get("pnl", 0) <= 0]
            if losing_trades:
                avg_loss = np.mean([t.get("pnl", 0) for t in losing_trades])
                print(f"Average Losing Trade: ${avg_loss:.2f}")

        # Symbol performance
        print("\nSymbol Performance:")
        for symbol in self.symbols:
            symbol_trades = [t for t in sell_trades if t["symbol"] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t.get("pnl", 0) for t in symbol_trades)
                symbol_trades_count = len(symbol_trades)
                print(f"{symbol}: {symbol_trades_count} trades, P&L: ${symbol_pnl:.2f}")

        # Recent signals
        print("\nLatest Signals:")
        for symbol, signal_data in self.latest_signals.items():
            print(
                f"{symbol}: Signal: {signal_data['combined_signal']:.3f}, "
                f"Confidence: {signal_data['confidence']:.3f}, "
                f"Price: ${signal_data['price']:.2f}"
            )

        # Model performance
        print("\nModel Performance:")
        for symbol, model_data in self.models.items():
            print(f"{symbol}: Accuracy: {model_data['accuracy']:.3f}")

    def get_current_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get current trading signals for all symbols"""
        signals = {}

        for symbol in self.symbols:
            if symbol in self.data:
                try:
                    latest_idx = len(self.data[symbol]) - 1
                    combined_signal, combined_confidence = self.get_combined_signal(
                        symbol, latest_idx
                    )
                    current_price = self.data[symbol]["Close"].iloc[-1]

                    signals[symbol] = {
                        "combined_signal": float(combined_signal),
                        "confidence": float(combined_confidence),
                        "price": float(current_price),
                        "recommendation": self.get_recommendation(
                            combined_signal, combined_confidence
                        ),
                    }
                except Exception as e:
                    print(f"Error getting signal for {symbol}: {e}")

        return signals

    def get_recommendation(self, signal: float, confidence: float) -> str:
        """Convert signal to human-readable recommendation"""
        if (
            signal > self.trading_config.signal_threshold
            and confidence > self.trading_config.confidence_threshold
        ):
            return "BUY"
        elif (
            signal < -self.trading_config.signal_threshold
            and confidence > self.trading_config.confidence_threshold
        ):
            return "SELL"
        else:
            return "HOLD"

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        portfolio = {
            "cash": float(self.balance),
            "positions": {},
            "total_value": float(self.balance),
        }

        for symbol in self.symbols:
            if self.positions[symbol] > 0 and symbol in self.data:
                current_price = self.data[symbol]["Close"].iloc[-1]
                position_value = self.positions[symbol] * current_price

                portfolio["positions"][symbol] = {
                    "shares": float(self.positions[symbol]),
                    "current_price": float(current_price),
                    "value": float(position_value),
                }
                portfolio["total_value"] += position_value

        portfolio["total_return"] = (
            (portfolio["total_value"] - self.trading_config.initial_balance)
            / self.trading_config.initial_balance
        ) * 100
        portfolio["total_realized_returns"] = (
            (self.balance - self.trading_config.initial_balance)
            / self.trading_config.initial_balance
        ) * 100

        return portfolio


# Factory Functions for Easy Setup
def create_default_system(
    symbols: List[str],
    initial_balance: float = 10000,
    custom_data_configs: Optional[Dict[str, DataFetcherConfig]] = None,
) -> XeniaV2Modular:
    """Create a default trading system with standard configurations"""

    # Create configurations
    trading_config = TradingConfig(initial_balance=initial_balance)
    feature_config = FeatureConfig()
    label_config = LabelConfig()
    model_config = ModelConfig()

    # Create data fetcher configurations
    data_fetcher_configs = {}
    for symbol in symbols:
        if custom_data_configs and symbol in custom_data_configs:
            data_fetcher_configs[symbol] = custom_data_configs[symbol]
        else:
            data_fetcher_configs[symbol] = DataFetcherConfig(symbol=symbol)

    # Create components
    data_fetcher = YFinanceDataFetcher()
    feature_engineer = TechnicalFeatureEngineer()
    label_creator = ThresholdLabelCreator()
    model_trainer = RandomForestModelTrainer()

    # Create signal generators
    ml_signal_generator = MLSignalGenerator(feature_engineer, feature_config)
    tech_signal_generator = TechnicalSignalGenerator(feature_engineer, feature_config)
    signal_generators = [ml_signal_generator, tech_signal_generator]

    # Create trade executor
    trade_executor = BasicTradeExecutor()

    # Create system
    system = XeniaV2Modular(
        symbols=symbols,
        data_fetcher=data_fetcher,
        feature_engineer=feature_engineer,
        label_creator=label_creator,
        model_trainer=model_trainer,
        signal_generators=signal_generators,
        trade_executor=trade_executor,
        trading_config=trading_config,
        feature_config=feature_config,
        label_config=label_config,
        model_config=model_config,
        data_fetcher_configs=data_fetcher_configs,
    )

    return system


def create_custom_system(
    symbols: List[str],
    data_fetcher: DataFetcher,
    feature_engineer: FeatureEngineer,
    label_creator: LabelCreator,
    model_trainer: ModelTrainer,
    signal_generators: List[SignalGenerator],
    trade_executor: TradeExecutor,
    trading_config: TradingConfig,
    feature_config: FeatureConfig,
    label_config: LabelConfig,
    model_config: ModelConfig,
    data_fetcher_configs: Dict[str, DataFetcherConfig],
) -> XeniaV2Modular:
    """Create a custom trading system with user-defined components"""

    return XeniaV2Modular(
        symbols=symbols,
        data_fetcher=data_fetcher,
        feature_engineer=feature_engineer,
        label_creator=label_creator,
        model_trainer=model_trainer,
        signal_generators=signal_generators,
        trade_executor=trade_executor,
        trading_config=trading_config,
        feature_config=feature_config,
        label_config=label_config,
        model_config=model_config,
        data_fetcher_configs=data_fetcher_configs,
    )


def create_default_trader(api_key: Optional[str] = None, secret_key: Optional[str] = None):

    if not api_key and secret_key:
        api_key='PKB6827AE6J1CM0IKLEJ'
        secret_key='bRQuhjlrbz7uVqX3SeBfJ2KaRWsOcAYLXv5rbgZV'

    return AlpacaTrader(
        api_key=api_key,
        secret_key=secret_key
    )

import asyncio
import argparse
import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import pytz
from dataclasses import dataclass, asdict
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xenia_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class EmailNotificationService:
    """Email notification service for trading system"""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self.daily_events = []
        self.last_summary_date = None
        
    def add_event(self, event_type: str, message: str, data: Dict = None):
        """Add an event to the daily summary"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'data': data or {}
        }
        self.daily_events.append(event)
        
    def send_email(self, subject: str, body: str, html_body: str = None) -> bool:
        """Send email notification"""
        if not self.config.enabled or not self.config.sender_email or not self.config.sender_password:
            logger.warning("Email notifications not properly configured")
            return False
            
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.sender_email
            msg['To'] = self.config.recipient_email
            msg['Subject'] = f"[XENIA TRADING] {subject}"
            
            # Add text version
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML version if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Connect and send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.sender_email, self.config.sender_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_signal_notification(self, signals: Dict[str, Dict]):
        """Send notification for new trading signals"""
        if not self.config.send_on_signals:
            return
            
        active_signals = {k: v for k, v in signals.items() if v['recommendation'] != 'HOLD'}
        
        if not active_signals:
            return
            
        subject = f"New Trading Signals - {len(active_signals)} Active"
        
        # Text version
        body = f"New trading signals detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n\n"
        for symbol, signal_data in active_signals.items():
            body += f"{symbol}: {signal_data['recommendation']} | "
            body += f"Signal: {signal_data['combined_signal']:.3f} | "
            body += f"Confidence: {signal_data['confidence']:.3f} | "
            body += f"Price: ${signal_data['price']:.2f}\n"
        
        # HTML version
        html_body = f"""
        <html>
        <body>
        <h2>🚀 New Trading Signals</h2>
        <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Active Signals:</strong> {len(active_signals)}</p>
        
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f2f2f2;">
            <th>Symbol</th>
            <th>Recommendation</th>
            <th>Signal</th>
            <th>Confidence</th>
            <th>Price</th>
        </tr>
        """
        
        for symbol, signal_data in active_signals.items():
            color = "#28a745" if signal_data['recommendation'] == 'BUY' else "#dc3545"
            html_body += f"""
            <tr>
                <td><strong>{symbol}</strong></td>
                <td style="color: {color}; font-weight: bold;">{signal_data['recommendation']}</td>
                <td>{signal_data['combined_signal']:.3f}</td>
                <td>{signal_data['confidence']:.3f}</td>
                <td>${signal_data['price']:.2f}</td>
            </tr>
            """
        
        html_body += """
        </table>
        </body>
        </html>
        """
        
        self.send_email(subject, body, html_body)
        self.add_event("SIGNALS", f"Sent signal notification for {len(active_signals)} symbols", active_signals)
    
    def send_trade_notification(self, trades: Dict[str, tuple]):
        """Send notification for executed trades"""
        if not self.config.send_on_trades or not trades:
            return
            
        subject = f"Trades Executed - {len(trades)} Positions"
        
        # Text version
        body = f"Trades executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n\n"
        for symbol, (signal, price) in trades.items():
            body += f"{symbol}: {signal} at ${price:.2f}\n"
        
        # HTML version
        html_body = f"""
        <html>
        <body>
        <h2>💼 Trades Executed</h2>
        <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Trades:</strong> {len(trades)}</p>
        
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f2f2f2;">
            <th>Symbol</th>
            <th>Action</th>
            <th>Price</th>
        </tr>
        """
        
        for symbol, (signal, price) in trades.items():
            color = "#28a745" if signal == 'BUY' else "#dc3545"
            html_body += f"""
            <tr>
                <td><strong>{symbol}</strong></td>
                <td style="color: {color}; font-weight: bold;">{signal}</td>
                <td>${price:.2f}</td>
            </tr>
            """
        
        html_body += """
        </table>
        </body>
        </html>
        """
        
        self.send_email(subject, body, html_body)
        self.add_event("TRADES", f"Executed {len(trades)} trades", trades)
    
    def send_error_notification(self, error_message: str, context: str = ""):
        """Send notification for errors"""
        if not self.config.send_on_errors:
            return
            
        subject = "❌ Trading System Error"
        
        body = f"Error occurred in Xenia Trading System:\n\n"
        body += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        body += f"Context: {context}\n"
        body += f"Error: {error_message}\n"
        
        html_body = f"""
        <html>
        <body>
        <h2 style="color: #dc3545;">❌ Trading System Error</h2>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Context:</strong> {context}</p>
        <p><strong>Error:</strong></p>
        <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">{error_message}</pre>
        </body>
        </html>
        """
        
        self.send_email(subject, body, html_body)
        self.add_event("ERROR", f"Error in {context}: {error_message}")
    
    def send_daily_summary(self, portfolio_status: Dict = None):
        """Send daily summary of trading activities"""
        if not self.config.send_daily_summary:
            return
            
        today = datetime.now().date()
        if self.last_summary_date == today:
            return
            
        self.last_summary_date = today
        
        subject = f"Daily Trading Summary - {today.strftime('%Y-%m-%d')}"
        
        # Count events by type
        event_counts = {}
        for event in self.daily_events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Text version
        body = f"Daily Trading Summary for {today.strftime('%Y-%m-%d')}:\n\n"
        body += f"Total Events: {len(self.daily_events)}\n"
        
        for event_type, count in event_counts.items():
            body += f"{event_type}: {count}\n"
        
        if portfolio_status:
            body += f"\nPortfolio Status:\n"
            body += f"Cash: ${portfolio_status.get('cash', 0):.2f}\n"
            body += f"Total Value: ${portfolio_status.get('total_value', 0):.2f}\n"
            body += f"Total Return: {portfolio_status.get('total_return', 0):.2f}%\n"
        
        body += "\nRecent Events:\n"
        for event in self.daily_events[-5:]:  # Show last 5 events
            body += f"{event['timestamp']}: {event['type']} - {event['message']}\n"
        
        # HTML version
        html_body = f"""
        <html>
        <body>
        <h2>📊 Daily Trading Summary</h2>
        <p><strong>Date:</strong> {today.strftime('%Y-%m-%d')}</p>
        <p><strong>Total Events:</strong> {len(self.daily_events)}</p>
        
        <h3>Event Summary</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f2f2f2;">
            <th>Event Type</th>
            <th>Count</th>
        </tr>
        """
        
        for event_type, count in event_counts.items():
            html_body += f"""
            <tr>
                <td>{event_type}</td>
                <td>{count}</td>
            </tr>
            """
        
        html_body += "</table>"
        
        if portfolio_status:
            html_body += f"""
            <h3>Portfolio Status</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>Cash</strong></td><td>${portfolio_status.get('cash', 0):.2f}</td></tr>
            <tr><td><strong>Total Value</strong></td><td>${portfolio_status.get('total_value', 0):.2f}</td></tr>
            <tr><td><strong>Total Return</strong></td><td>{portfolio_status.get('total_return', 0):.2f}%</td></tr>
            </table>
            """
        
        html_body += """
        </body>
        </html>
        """
        
        self.send_email(subject, body, html_body)
        
        # Clear daily events after sending summary
        self.daily_events = []


class MarketHoursChecker:
    """Check if the market is open considering holidays and weekends"""
    
    def __init__(self, market_tz: str = "US/Eastern", local_tz: str = "Africa/Kampala"):
        self.market_tz = pytz.timezone(market_tz)
        self.local_tz = pytz.timezone(local_tz)
        
        # Common US market holidays (simplified)
        self.holidays_2024 = [
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # Martin Luther King Jr. Day
            "2024-02-19",  # Presidents' Day
            "2024-03-29",  # Good Friday
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
        ]
        
        self.holidays_2025 = [
            "2025-01-01",  # New Year's Day
            "2025-01-20",  # Martin Luther King Jr. Day
            "2025-02-17",  # Presidents' Day
            "2025-04-18",  # Good Friday
            "2025-05-26",  # Memorial Day
            "2025-06-19",  # Juneteenth
            "2025-07-04",  # Independence Day
            "2025-09-01",  # Labor Day
            "2025-11-27",  # Thanksgiving
            "2025-12-25",  # Christmas
        ]
    
    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open"""
        now_utc = datetime.now(pytz.UTC)
        market_time = now_utc.astimezone(self.market_tz)
        
        # Check if it's a weekend
       # if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
         #   return False
        
        # Check if it's a holiday
        market_date = market_time.date().strftime("%Y-%m-%d")
        if market_date in self.holidays_2024 + self.holidays_2025:
            return False
        
        # Check market hours (9:30 AM - 4:00 PM EST)
        market_hour = market_time.hour
        market_minute = market_time.minute
        
        # Market opens at 9:30 AM
        # if market_hour < 9 or (market_hour == 9 and market_minute < 30):
           # return False
        
        # Market closes at 4:00 PM
        if market_hour >= 16:
            return False
        
        return True
    
    def get_next_market_open(self) -> datetime:
        """Get the next time the market will be open"""
        now_utc = datetime.now(pytz.UTC)
        market_time = now_utc.astimezone(self.market_tz)
        
        # Start checking from the next day if market is closed today
        check_date = market_time.date()
        if market_time.hour >= 16:  # After market close
            check_date += timedelta(days=1)
        
        # Find the next market day
        while True:
            # Skip weekends
            if check_date.weekday() >= 5:
                check_date += timedelta(days=1)
                continue
            
            # Skip holidays
            if check_date.strftime("%Y-%m-%d") in self.holidays_2024 + self.holidays_2025:
                check_date += timedelta(days=1)
                continue
            
            break
        
        # Set to 9:30 AM market time
        next_open = self.market_tz.localize(
            datetime.combine(check_date, datetime.min.time().replace(hour=9, minute=30))
        )
        
        return next_open
    
    def time_until_market_open(self) -> timedelta:
        """Get time remaining until market opens"""
        if self.is_market_open():
            return timedelta(0)
        
        next_open = self.get_next_market_open()
        now_utc = datetime.now(pytz.UTC)
        
        return next_open - now_utc

class XeniaCLI:
    """Command Line Interface for Xenia Trading System"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.config_file = Path("xenia_config.json")
        self.market_checker = MarketHoursChecker()
        self.running = False
        self.email_service = EmailNotificationService(self.config.email_config)
        
    def load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                    # Handle email config separately
                    if 'email_config' in config_data:
                        email_data = config_data.pop('email_config')
                        self.config.email_config = EmailConfig(**email_data)
                    
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                
                # Update email service with new config
                self.email_service = EmailNotificationService(self.config.email_config)
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.email_service.send_error_notification(str(e), "Loading configuration")
        else:
            self.save_config()
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            config_dict = asdict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            self.email_service.send_error_notification(str(e), "Saving configuration")
    
    def show_status(self) -> None:
        """Display current system status"""
        print("\n" + "=" * 60)
        print("XENIA TRADING SYSTEM STATUS")
        print("=" * 60)
        
        market_open = self.market_checker.is_market_open()
        print(f"Market Status: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
        
        if not market_open:
            time_until = self.market_checker.time_until_market_open()
            print(f"Next Market Open: {time_until}")
        
        print(f"Live Trading: {'🟢 ENABLED' if self.config.live_trading else '🔴 DISABLED'}")
        print(f"Monte Carlo: {'🟢 ENABLED' if self.config.enable_monte_carlo else '🔴 DISABLED'}")
        print(f"Email Notifications: {'🟢 ENABLED' if self.config.email_config.enabled else '🔴 DISABLED'}")
        print(f"Run Interval: {self.config.run_interval_hours} hours")
        print(f"Symbols: {', '.join(self.config.symbols)}")
        print(f"System Running: {'🟢 YES' if self.running else '🔴 NO'}")
        
        # Show local time
        local_time = datetime.now(pytz.timezone(self.config.local_timezone))
        market_time = datetime.now(pytz.timezone(self.config.market_timezone))
        print(f"\nLocal Time (Uganda): {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Market Time (EST): {market_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    def toggle_live_trading(self) -> None:
        """Toggle live trading on/off"""
        self.config.live_trading = not self.config.live_trading
        status = "ENABLED" if self.config.live_trading else "DISABLED"
        print(f"Live Trading: {status}")
        self.save_config()
    
    def toggle_monte_carlo(self) -> None:
        """Toggle Monte Carlo analysis on/off"""
        self.config.enable_monte_carlo = not self.config.enable_monte_carlo
        status = "ENABLED" if self.config.enable_monte_carlo else "DISABLED"
        print(f"Monte Carlo Analysis: {status}")
        self.save_config()
    
    def toggle_email_notifications(self) -> None:
        """Toggle email notifications on/off"""
        self.config.email_config.enabled = not self.config.email_config.enabled
        status = "ENABLED" if self.config.email_config.enabled else "DISABLED"
        print(f"Email Notifications: {status}")
        self.save_config()
    
    def configure_email(self, sender_email: str, sender_password: str) -> None:
        """Configure email credentials"""
        self.config.email_config.sender_email = sender_email
        self.config.email_config.sender_password = sender_password
        print("Email credentials configured")
        self.save_config()
        
        # Update email service
        self.email_service = EmailNotificationService(self.config.email_config)
        
        # Send test email
        self.email_service.send_email(
            "Email Configuration Test",
            "Email notifications have been successfully configured for Xenia Trading System.",
            "<h2>✅ Email Configuration Test</h2><p>Email notifications have been successfully configured for Xenia Trading System.</p>"
        )
    
    def set_interval(self, hours: int) -> None:
        """Set run interval in hours"""
        if hours < 1:
            print("Error: Interval must be at least 1 hour")
            return
        
        self.config.run_interval_hours = hours
        print(f"Run interval set to: {hours} hours")
        self.save_config()
    
    def configure_api(self, api_key: str, api_secret: str) -> None:
        """Configure API credentials"""
        self.config.api_key = api_key
        self.config.api_secret = api_secret
        print("API credentials configured")
        self.save_config()
    
    async def run_trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        try:
            print("\n" + "=" * 60)
            print(f"STARTING TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            self.email_service.add_event("CYCLE_START", "Starting trading cycle")
            
            # Check if market is open
            if not self.market_checker.is_market_open():
                print("Market is closed. Skipping trading cycle.")
                self.email_service.add_event("MARKET_CLOSED", "Market is closed, skipping cycle")
                return

            self.config.symbols = ["TSLA", "NVDA", "META", "GOOGL", "PLTR", "^NDX"]
            
            # Import your trading system modules here
            # from your_trading_system import create_default_system, create_default_trader, DataFetcherConfig
            
            # Create custom data fetcher configs
            custom_data_configs = {}
            for symbol in self.config.symbols:
                custom_data_configs[symbol] = DataFetcherConfig(symbol=symbol, period='10y')
                
            
            # Create system using default factory
            system = create_default_system(
                self.config.symbols,
                initial_balance=self.config.initial_balance,
                custom_data_configs=custom_data_configs
            )
            
            # Create trader if live trading is enabled
            trader = None
            if self.config.live_trading and self.config.api_key and self.config.api_secret:
                trader = create_default_trader(self.config.api_key, self.config.api_secret)

            start = datetime(2019, 1, 1)
            end = datetime(2021, 1, 1)
            
            # fetching symbol data
            await system.fetch_all_data()

            # training models for the data
            await system.train_all_models()

            
            # >>> Removed the monte-carlo and the backtest <<<
            
            # Run backtest
            print("Running backtest...")
            await system.run_backtest()
            
            # Run professional analysis if enabled
            if self.config.enable_monte_carlo:
                print("Running Monte Carlo analysis...")
                system.run_professional_analysis(
                    num_monte_carlo=self.config.monte_carlo_simulations,
                    forecast_days=self.config.forecast_days
                )
            
            # Show current signals
            
            print("\n" + "=" * 60)
            print("CURRENT SIGNALS")
            print("=" * 60)
            
            live_signals = {}
            signals = system.get_current_signals()
            
          
            for symbol, signal_data in signals.items():
                if signal_data['recommendation'] != 'HOLD':
                    live_signals[symbol] = [signal_data['recommendation'], signal_data['price']]
                print(
                    f"{symbol}: {signal_data['recommendation']} | "
                    f"Signal: {signal_data['combined_signal']:.3f} | "
                    f"Confidence: {signal_data['confidence']:.3f} | "
                    f"Price: ${signal_data['price']:.2f}"
                )
            
            # Send signal notifications
            if signals:
                self.email_service.send_signal_notification(signals)
            
            # Show portfolio status
            print("\n" + "=" * 60)
            print("PORTFOLIO STATUS")
            print("=" * 60)
            
            portfolio = system.get_portfolio_status()
            
            if portfolio:
                print(f"Cash: ${portfolio['cash']:.2f}")
                print(f"Total Value: ${portfolio['total_value']:.2f}")
                print(f"Total Return: {portfolio['total_return']:.2f}%")
                
                if portfolio.get("positions"):
                    print("\nPositions:")
                    for symbol, position in portfolio["positions"].items():
                        print(
                            f"{symbol}: {position['shares']:.2f} shares @ "
                            f"${position['current_price']:.2f} = ${position['value']:.2f}"
                        )
            
            # Execute live trades if enabled
            executed_trades = {}
            if self.config.live_trading and trader and live_signals:
                print("\n" + "=" * 60)
                print("EXECUTING LIVE TRADES")
                print("=" * 60)
                
                account_info = trader.get_account_info()
                
                if account_info:
                    for symbol, (signal, price) in live_signals.items():
                        quantity = (account_info['buying_power'] * self.config.risk_per_trade) / price
                        
                        print(f"Executing {signal} for {symbol}: {quantity:.2f} shares")
                        
                        trader.execute_signal(
                            signal,
                            symbol,
                            quantity=(quantity if signal.upper() == 'BUY' else None)
                        )
                        
                        executed_trades[symbol] = (signal, price)
                        time.sleep(1)  # Rate limiting
            
            # Send trade notifications
            if executed_trades:
                self.email_service.send_trade_notification(executed_trades)
            
            # Send daily summary
            self.email_service.send_daily_summary(portfolio)
            
            print("\n" + "=" * 60)
            print("TRADING CYCLE COMPLETED")
            print("=" * 60)
            
            self.email_service.add_event("CYCLE_COMPLETE", "Trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            print(f"Error in trading cycle: {e}")
            self.email_service.send_error_notification(str(e), "Trading cycle")
    
    async def start_system(self) -> None:
        """Start the automated trading system"""
        if self.running:
            print("System is already running!")
            return
        
        self.running = True
        print("Starting Xenia Trading System...")
        print("Press Ctrl+C to stop the system")
        
        # Send startup notification
        self.email_service.send_email(
            "🚀 System Started",
            "Xenia Trading System has been started successfully.",
            "<h2>🚀 System Started</h2><p>Xenia Trading System has been started successfully.</p>"
        )

        try:
            while self.running:
                # Check if market is open
                if self.market_checker.is_market_open():
                    await self.run_trading_cycle()
                    
                    # Wait for the specified interval
                    wait_seconds = self.config.run_interval_hours * 3600
                    print(f"\nWaiting {self.config.run_interval_hours} hours until next cycle...")
                    
                    for remaining in range(wait_seconds, 0, -1):
                        if not self.running:
                            break
                        
                        hours = remaining // 3600
                        minutes = (remaining % 3600) // 60
                        seconds = remaining % 60
                        
                        print(f"\rNext cycle in: {hours:02d}:{minutes:02d}:{seconds:02d}", end="")
                        await asyncio.sleep(1)
                    
                    print()  # New line after countdown
                else:
                    # Market is closed, wait until it opens with countdown
                    time_until_open = self.market_checker.time_until_market_open()
                    print(f"Market is closed. Waiting until market opens...")
                    
                    # Convert to total seconds for countdown
                    wait_seconds = int(time_until_open.total_seconds())
                    
                    for remaining in range(wait_seconds, 0, -1):
                        if not self.running:
                            break
                        
                        # Calculate days, hours, minutes, seconds
                        days = remaining // 86400
                        hours = (remaining % 86400) // 3600
                        minutes = (remaining % 3600) // 60
                        seconds = remaining % 60
                        
                        # Format countdown display
                        if days > 0:
                            countdown_str = f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            countdown_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        print(f"\rMarket opens in: {countdown_str}", end="")
                        await asyncio.sleep(1)
                    
                    print()  # New line after countdown               
        except KeyboardInterrupt:
            print("\nShutting down system...")
        finally:
            self.running = False
            # Send shutdown notification
            self.email_service.send_email(
                "⏹️ System Stopped",
                "Xenia Trading System has been stopped.",
                "<h2>⏹️ System Stopped</h2><p>Xenia Trading System has been stopped.</p>"
            )
    
    def stop_system(self) -> None:
        """Stop the automated trading system"""
        if not self.running:
            print("System is not running!")
            return
        
        self.running = False
        print("Stopping system...")
    
    def interactive_mode(self) -> None:
        """Interactive command mode"""
        print("\n🚀 Welcome to Xenia Trading System CLI")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command = input("\nxenia> ").strip().lower()
                
                if command == 'help':
                    self.show_help()
                elif command == 'status':
                    self.show_status()
                elif command == 'toggle-live':
                    self.toggle_live_trading()
                elif command == 'toggle-monte':
                    self.toggle_monte_carlo()
                elif command == 'toggle-email':
                    self.toggle_email_notifications()
                elif command.startswith('interval '):
                    try:
                        hours = int(command.split()[1])
                        self.set_interval(hours)
                    except (IndexError, ValueError):
                        print("Usage: interval <hours>")
                elif command.startswith('email-config '):
                    try:
                        parts = command.split()
                        if len(parts) >= 3:
                            sender_email = parts[1]
                            sender_password = parts[2]
                            self.configure_email(sender_email, sender_password)
                        else:
                            print("Usage: email-config <sender_email> <sender_password>")
                    except (IndexError, ValueError):
                        print("Usage: email-config <sender_email> <sender_password>")
                elif command == 'test-email':
                    self.test_email()
                elif command == 'start':
                    asyncio.run(self.start_system())
                elif command == 'stop':
                    self.stop_system()
                elif command == 'run-once':
                    asyncio.run(self.run_trading_cycle())
                elif command == 'config':
                    self.show_config()
                elif command in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
                self.email_service.send_error_notification(str(e), "Interactive mode")
    
    def test_email(self) -> None:
        """Send a test email"""
        if not self.config.email_config.sender_email or not self.config.email_config.sender_password:
            print("Email not configured. Use 'email-config' command first.")
            return
        
        success = self.email_service.send_email(
            "Test Email",
            "This is a test email from Xenia Trading System.",
            "<h2>📧 Test Email</h2><p>This is a test email from Xenia Trading System.</p><p>If you receive this, email notifications are working correctly!</p>"
        )
        
        if success:
            print("✅ Test email sent successfully!")
        else:
            print("❌ Failed to send test email. Check your email configuration.")
    
    def show_help(self) -> None:
        """Show help information"""
        print("\n📋 Available Commands:")
        print("  help              - Show this help message")
        print("  status            - Show system status")
        print("  toggle-live       - Toggle live trading on/off")
        print("  toggle-monte      - Toggle Monte Carlo analysis on/off")
        print("  toggle-email      - Toggle email notifications on/off")
        print("  interval <n>      - Set run interval to n hours")
        print("  email-config <email> <password> - Configure email credentials")
        print("  test-email        - Send a test email")
        print("  start             - Start automated trading system")
        print("  stop              - Stop automated trading system")
        print("  run-once          - Run one trading cycle immediately")
        print("  config            - Show current configuration")
        print("  exit/quit/q       - Exit the program")
    
    def show_config(self) -> None:
        """Show current configuration"""
        print("\n⚙️  Current Configuration:")
        print(f"  Live Trading: {self.config.live_trading}")
        print(f"  Monte Carlo: {self.config.enable_monte_carlo}")
        print(f"  Monte Carlo Simulations: {self.config.monte_carlo_simulations}")
        print(f"  Forecast Days: {self.config.forecast_days}")
        print(f"  Run Interval: {self.config.run_interval_hours} hours")
        print(f"  Initial Balance: ${self.config.initial_balance}")
        print(f"  Risk per Trade: {self.config.risk_per_trade * 100}%")
        print(f"  Symbols: {', '.join(self.config.symbols)}")
        print(f"  API Key: {'***' if self.config.api_key else 'Not set'}")
        print(f"\n📧 Email Configuration:")
        print(f"  Enabled: {self.config.email_config.enabled}")
        print(f"  Sender Email: {'***' if self.config.email_config.sender_email else 'Not set'}")
        print(f"  Recipient: {self.config.email_config.recipient_email}")
        print(f"  Send Signals: {self.config.email_config.send_on_signals}")
        print(f"  Send Trades: {self.config.email_config.send_on_trades}")
        print(f"  Send Errors: {self.config.email_config.send_on_errors}")
        print(f"  Daily Summary: {self.config.email_config.send_daily_summary}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Xenia Trading System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Enable live trading mode'
    )
    
    parser.add_argument(
        '--monte-carlo',
        action='store_true',
        help='Enable Monte Carlo analysis'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=24,
        help='Run interval in hours (default: 24)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for live trading'
    )
    
    parser.add_argument(
        '--api-secret',
        type=str,
        help='API secret for live trading'
    )
    
    parser.add_argument(
        '--email-sender',
        type=str,
        help='Email sender address'
    )
    
    parser.add_argument(
        '--email-password',
        type=str,
        help='Email sender password'
    )
    
    parser.add_argument(
        '--disable-email',
        action='store_true',
        help='Disable email notifications'
    )
    
    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run one trading cycle and exit'
    )
    
    parser.add_argument(
        '--start',
        action='store_true',
        help='Start the automated trading system'
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = XeniaCLI()
    cli.load_config()
    
    # Apply command line arguments
    if args.live:
        cli.config.live_trading = True
    
    if args.monte_carlo:
        cli.config.enable_monte_carlo = True
    
    if args.interval:
        cli.config.run_interval_hours = args.interval
    
    if args.api_key and args.api_secret:
        cli.configure_api(args.api_key, args.api_secret)
    
    if args.email_sender and args.email_password:
        cli.configure_email(args.email_sender, args.email_password)
    
    if args.disable_email:
        cli.config.email_config.enabled = False
    
    cli.save_config()
    
    # Execute based on arguments
    if args.run_once:
        asyncio.run(cli.run_trading_cycle())
    elif args.start:
        asyncio.run(cli.start_system())
    else:
        # Enter interactive mode
        cli.interactive_mode()

if __name__ == "__main__":
    main()
    