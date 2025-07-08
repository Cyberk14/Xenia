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
class TradingConfig:
    """Configuration for trading parameters"""

    initial_balance: float = 10000
    transaction_cost: float = 0.002
    min_accuracy_threshold: float = 0.35
    signal_threshold: float = 0.15
    confidence_threshold: float = 0.4
    max_position_size: float = 0.2
    min_fetch_interval: float = 1.0


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
                data = await self.datastream_fetcher.fetch_ohlcv(
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


class ProfessionalVisualizer:
    """
    🎨 Professional Trading System Visualization Engine
    
    Creates sophisticated, interactive dashboards and reports for institutional
    and retail investors with modern design principles and comprehensive analytics.
    """
    
    def __init__(self, trading_system, monte_carlo_simulator=None):
        self.trading_system = trading_system
        self.monte_carlo = monte_carlo_simulator
        
        # Professional color palette
        self.colors = {
            'primary': '#00D4AA',      # Teal Green
            'secondary': '#6C7CE7',    # Soft Purple
            'accent': '#FF6B9D',       # Pink Accent
            'warning': '#FFB800',      # Amber
            'success': '#00E676',      # Green
            'error': '#FF5252',        # Red
            'neutral': '#90A4AE',      # Blue Grey
            'dark': '#263238',         # Dark Blue Grey
            'light': '#F8F9FA',        # Light Grey
            'background': '#1E1E1E',   # Dark Background
            'surface': '#2D2D2D',      # Surface
            'text': '#FFFFFF',         # White Text
            'text_secondary': '#B0BEC5' # Secondary Text
        }
        
        # Professional theme configuration
        self.theme = {
            'template': 'plotly_dark',
            'font_family': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'title_font_size': 28,
            'subtitle_font_size': 16,
            'axis_font_size': 12,
            'legend_font_size': 11,
            'paper_bgcolor': self.colors['background'],
            'plot_bgcolor': self.colors['surface'],
            'grid_color': '#404040',
            'line_width': 2.5,
            'marker_size': 8,
            'border_radius': 8
        }
        
    def create_executive_dashboard(self, save_path=None):
        """
        Create a comprehensive executive dashboard with professional styling
        
        Features:
        - Real-time portfolio performance metrics
        - Risk analysis with Monte Carlo projections
        - Interactive charts with hover details
        - Professional color scheme and typography
        - Responsive design for all screen sizes
        """
        print("\n🎨 Creating Executive Dashboard...")
        print("📊 Generating professional visualizations...")
        
        # Create enhanced subplot layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                '📈 Portfolio Performance & Benchmark',
                '📊 Returns Distribution Analysis', 
                '⚡ Risk & Performance Metrics',
                '🎯 Trade Analysis & Execution',
                '🏆 Symbol Performance Ranking',
                '🎲 Monte Carlo Projections',
                '📉 Drawdown & Risk Analysis',
                '🔍 Signal Strength Matrix',
                '📊 Confidence Intervals',
                '💰 Asset Allocation',
                '📈 Performance Attribution',
                '🎯 Risk-Return Profile'
            ),
            specs=[
                [{"secondary_y": True}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"secondary_y": True}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "xy"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Generate all dashboard components
        self._add_enhanced_portfolio_performance(fig, row=1, col=1)
        self._add_enhanced_returns_distribution(fig, row=1, col=2)
        self._add_enhanced_risk_metrics(fig, row=1, col=3)
        self._add_enhanced_trade_analysis(fig, row=2, col=1)
        self._add_enhanced_symbol_performance(fig, row=2, col=2)
        
        if self.monte_carlo:
            self._add_enhanced_monte_carlo_paths(fig, row=2, col=3)
        
        self._add_enhanced_drawdown_analysis(fig, row=3, col=1)
        self._add_enhanced_signal_strength(fig, row=3, col=2)
        
        if self.monte_carlo:
            self._add_enhanced_confidence_intervals(fig, row=3, col=3)
        
        self._add_asset_allocation_pie(fig, row=4, col=1)
        self._add_performance_attribution(fig, row=4, col=2)
        self._add_risk_return_profile(fig, row=4, col=3)
        
        # Apply professional styling
        fig.update_layout(
            height=1600,
            title={
                'text': '🚀 XENIA V2 - Executive Trading Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {
                    'size': self.theme['title_font_size'],
                    'family': self.theme['font_family'],
                    'color': self.colors['text']
                }
            },
            font={
                'family': self.theme['font_family'],
                'size': self.theme['axis_font_size'],
                'color': self.colors['text']
            },
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['surface'],
            showlegend=True,
            legend={
                'bgcolor': 'rgba(45, 45, 45, 0.8)',
                'bordercolor': self.colors['neutral'],
                'borderwidth': 1,
                'font': {'size': self.theme['legend_font_size']}
            },
            margin=dict(l=60, r=60, t=100, b=60),
            template=self.theme['template']
        )
        
        # Update all subplot backgrounds
        for i in range(1, 13):
            fig.update_xaxes(
                gridcolor=self.theme['grid_color'],
                gridwidth=0.5,
                title_font_size=self.theme['axis_font_size'],
                tickfont_size=10
            )
            fig.update_yaxes(
                gridcolor=self.theme['grid_color'],
                gridwidth=0.5,
                title_font_size=self.theme['axis_font_size'],
                tickfont_size=10
            )
        
        if save_path:
            fig.write_html(save_path, config={'displayModeBar': False})
            print(f"✅ Dashboard saved to: {save_path}")
        
        return fig
        
    def _add_enhanced_portfolio_performance(self, fig, row, col):
        """Enhanced portfolio performance with benchmark comparison"""
        if not self.trading_system.trades:
            return
            
        # Calculate portfolio performance over time
        dates = []
        portfolio_values = []
        benchmark_values = []
        current_balance = self.trading_system.trading_config.initial_balance
        
        for i, trade in enumerate(self.trading_system.trades):
            dates.append(trade['date'])
            
            if trade['action'] == 'BUY':
                current_balance -= trade['cost']
            else:
                current_balance += trade['revenue']
                
            portfolio_values.append(current_balance)
            # Simple benchmark: flat growth
            benchmark_values.append(self.trading_system.trading_config.initial_balance * (1 + 0.08 * i / len(self.trading_system.trades)))
        
        # Portfolio performance line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=4),
                hovertemplate='<b>Portfolio Value</b><br>' +
                            'Date: %{x}<br>' +
                            'Value: $%{y:,.2f}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Benchmark line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark_values,
                mode='lines',
                name='Benchmark (8% Annual)',
                line=dict(color=self.colors['neutral'], width=2, dash='dash'),
                hovertemplate='<b>Benchmark</b><br>' +
                            'Date: %{x}<br>' +
                            'Value: $%{y:,.2f}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add performance zones
        if len(portfolio_values) > 1:
            max_val = max(max(portfolio_values), max(benchmark_values))
            min_val = min(min(portfolio_values), min(benchmark_values))
            
            fig.add_hrect(
                y0=min_val, y1=max_val * 0.3,
                fillcolor=self.colors['error'], opacity=0.1,
                line_width=0, row=row, col=col
            )
            fig.add_hrect(
                y0=max_val * 0.7, y1=max_val,
                fillcolor=self.colors['success'], opacity=0.1,
                line_width=0, row=row, col=col
            )
    
    def _add_enhanced_returns_distribution(self, fig, row, col):
        """Enhanced returns distribution with statistical overlays"""
        if not self.trading_system.trades:
            return
            
        sell_trades = [t for t in self.trading_system.trades if t['action'] == 'SELL']
        if not sell_trades:
            return
            
        returns = [t.get('pnl_pct', 0) for t in sell_trades]
        
        # Main histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=25,
                name='Returns Distribution',
                marker_color=self.colors['secondary'],
                marker_line_color=self.colors['text'],
                marker_line_width=1,
                opacity=0.8,
                hovertemplate='<b>Returns Range</b><br>' +
                            'Range: %{x}<br>' +
                            'Count: %{y}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add statistical markers
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        fig.add_vline(
            x=mean_return, line_width=2, line_dash="dash",
            line_color=self.colors['warning'],
            annotation_text=f"Mean: {mean_return:.1f}%",
            row=row, col=col
        )
        
        fig.add_vline(
            x=median_return, line_width=2, line_dash="dot",
            line_color=self.colors['primary'],
            annotation_text=f"Median: {median_return:.1f}%",
            row=row, col=col
        )
    
    def _add_enhanced_risk_metrics(self, fig, row, col):
        """Enhanced risk metrics with color coding"""
        portfolio = self.trading_system.get_portfolio_status()
        
        metrics = ['Total Return %', 'Sharpe Ratio', 'Max Drawdown %', 'Win Rate %']
        values = [
            portfolio.get('total_return', 0),
            self._calculate_sharpe_ratio(),
            -abs(self._calculate_max_drawdown()),  # Negative for visual impact
            self._calculate_win_rate()
        ]
        
        # Color coding based on performance
        colors = []
        for i, val in enumerate(values):
            if i == 0:  # Total Return
                colors.append(self.colors['success'] if val > 0 else self.colors['error'])
            elif i == 1:  # Sharpe Ratio
                colors.append(self.colors['success'] if val > 1 else self.colors['warning'] if val > 0 else self.colors['error'])
            elif i == 2:  # Max Drawdown
                colors.append(self.colors['success'] if val > -10 else self.colors['warning'] if val > -20 else self.colors['error'])
            else:  # Win Rate
                colors.append(self.colors['success'] if val > 60 else self.colors['warning'] if val > 40 else self.colors['error'])
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name='Risk Metrics',
                marker_color=colors,
                marker_line_color=self.colors['text'],
                marker_line_width=1,
                text=[f'{v:.1f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                            'Value: %{y:.2f}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_enhanced_trade_analysis(self, fig, row, col):
        """Enhanced trade analysis with trend lines"""
        sell_trades = [t for t in self.trading_system.trades if t['action'] == 'SELL']
        if not sell_trades:
            return
            
        x_values = list(range(len(sell_trades)))
        y_values = [t.get('pnl_pct', 0) for t in sell_trades]
        trade_dates = [t['date'] for t in sell_trades]
        
        # Color code by performance
        colors = [self.colors['success'] if pnl > 0 else self.colors['error'] for pnl in y_values]
        sizes = [max(8, min(20, abs(pnl) / 2)) for pnl in y_values]  # Size based on magnitude
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                name='Trade P&L',
                marker=dict(
                    color=colors,
                    size=sizes,
                    line=dict(width=1, color=self.colors['text']),
                    opacity=0.8
                ),
                text=[f'Trade {i+1}<br>{date}<br>{pnl:.1f}%' for i, (date, pnl) in enumerate(zip(trade_dates, y_values))],
                hovertemplate='<b>%{text}</b><br>' +
                            'Trade #: %{x}<br>' +
                            'P&L: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add trend line
        if len(y_values) > 1:
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            trend_line = p(x_values)
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color=self.colors['accent'], width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add zero line
        fig.add_hline(
            y=0, line_width=1, line_dash="dot",
            line_color=self.colors['neutral'],
            row=row, col=col
        )
    
    def _add_enhanced_symbol_performance(self, fig, row, col):
        """Enhanced symbol performance with ranking"""
        symbol_performance = {}
        symbol_trades = {}
        
        for trade in self.trading_system.trades:
            if trade['action'] == 'SELL':
                symbol = trade['symbol']
                pnl = trade.get('pnl', 0)
                
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = 0
                    symbol_trades[symbol] = 0
                
                symbol_performance[symbol] += pnl
                symbol_trades[symbol] += 1
        
        if symbol_performance:
            # Sort by performance
            sorted_symbols = sorted(symbol_performance.items(), key=lambda x: x[1], reverse=True)
            symbols = [s[0] for s in sorted_symbols]
            performance = [s[1] for s in sorted_symbols]
            
            # Color gradient based on ranking
            colors = []
            for i, perf in enumerate(performance):
                if perf > 0:
                    intensity = min(1.0, perf / max(performance) if max(performance) > 0 else 0)
                    colors.append(f'rgba(0, 230, 118, {0.5 + 0.5 * intensity})')
                else:
                    intensity = min(1.0, abs(perf) / abs(min(performance)) if min(performance) < 0 else 0)
                    colors.append(f'rgba(255, 82, 82, {0.5 + 0.5 * intensity})')
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=performance,
                    name='Symbol P&L',
                    marker_color=colors,
                    marker_line_color=self.colors['text'],
                    marker_line_width=1,
                    text=[f'${p:.0f}' for p in performance],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' +
                                'P&L: $%{y:.2f}<br>' +
                                'Trades: ' + str([symbol_trades[s] for s in symbols]) + '<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
    
    def _add_enhanced_monte_carlo_paths(self, fig, row, col):
        """Enhanced Monte Carlo paths with confidence bands"""
        if not self.monte_carlo or not self.monte_carlo.simulations:
            return
        
        # Calculate percentiles for confidence bands
        max_length = max(len(sim['values']) for sim in self.monte_carlo.simulations)
        
        # Collect all paths and calculate percentiles
        all_paths = []
        for sim in self.monte_carlo.simulations:
            if len(sim['values']) == max_length:
                all_paths.append(sim['values'])
        
        if not all_paths:
            return
        
        all_paths = np.array(all_paths)
        days = list(range(max_length))
        
        # Calculate confidence intervals
        median_path = np.percentile(all_paths, 50, axis=0)
        upper_95 = np.percentile(all_paths, 95, axis=0)
        lower_5 = np.percentile(all_paths, 5, axis=0)
        upper_75 = np.percentile(all_paths, 75, axis=0)
        lower_25 = np.percentile(all_paths, 25, axis=0)
        
        # Add confidence bands
        fig.add_trace(
            go.Scatter(
                x=days + days[::-1],
                y=np.concatenate([upper_95, lower_5[::-1]]),
                fill='toself',
                fillcolor=f'rgba({int(108)}, {int(124)}, {int(231)}, 0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% Confidence',
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=days + days[::-1],
                y=np.concatenate([upper_75, lower_25[::-1]]),
                fill='toself',
                fillcolor=f'rgba({int(108)}, {int(124)}, {int(231)}, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='50% Confidence',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add median path
        fig.add_trace(
            go.Scatter(
                x=days,
                y=median_path,
                mode='lines',
                name='Median Path',
                line=dict(color=self.colors['secondary'], width=3),
                hovertemplate='<b>Median Projection</b><br>' +
                            'Day: %{x}<br>' +
                            'Value: $%{y:,.2f}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add sample paths for context
        sample_indices = np.random.choice(len(self.monte_carlo.simulations), 
                                        min(10, len(self.monte_carlo.simulations)), 
                                        replace=False)
        for i in sample_indices:
            sim = self.monte_carlo.simulations[i]
            if len(sim['values']) == max_length:
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=sim['values'],
                        mode='lines',
                        name=f'Path {i+1}',
                        line=dict(color=self.colors['neutral'], width=0.5, dash='dot'),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
    
    def _add_enhanced_drawdown_analysis(self, fig, row, col):
        """Enhanced drawdown analysis with underwater curve"""
        if not self.trading_system.trades:
            return
            
        # Calculate running drawdown
        dates = []
        balances = []
        current_balance = self.trading_system.trading_config.initial_balance
        peak_balance = current_balance
        drawdowns = []
        underwater_periods = []
        
        for trade in self.trading_system.trades:
            dates.append(trade['date'])
            
            if trade['action'] == 'BUY':
                current_balance -= trade['cost']
            else:
                current_balance += trade['revenue']
            
            if current_balance > peak_balance:
                peak_balance = current_balance
            
            drawdown = (peak_balance - current_balance) / peak_balance * 100
            drawdowns.append(drawdown)
            
            # Track underwater periods
            if drawdown > 0:
                underwater_periods.append(trade['date'])
        
        # Main drawdown curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[-dd for dd in drawdowns],  # Negative for visual impact
                mode='lines',
                name='Drawdown %',
                line=dict(color=self.colors['error'], width=2.5),
                fill='tozeroy',
                fillcolor=f'rgba(255, 82, 82, 0.3)',
                hovertemplate='<b>Drawdown Analysis</b><br>' +
                            'Date: %{x}<br>' +
                            'Drawdown: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add drawdown severity zones
        fig.add_hrect(
            y0=-5, y1=0,
            fillcolor=self.colors['warning'], opacity=0.1,
            line_width=0, row=row, col=col
        )
        fig.add_hrect(
            y0=-10, y1=-5,
            fillcolor=self.colors['error'], opacity=0.1,
            line_width=0, row=row, col=col
        )
        
        # Add annotations for significant drawdowns
        max_drawdown = max(drawdowns) if drawdowns else 0
        if max_drawdown > 5:  # Only annotate significant drawdowns
            max_dd_index = drawdowns.index(max_drawdown)
            fig.add_annotation(
                x=dates[max_dd_index],
                y=-max_drawdown,
                text=f"Max DD: {max_drawdown:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.colors['error'],
                bgcolor=self.colors['surface'],
                bordercolor=self.colors['error'],
                row=row, col=col
            )
    
    def _add_enhanced_signal_strength(self, fig, row, col):
        """Enhanced signal strength analysis with confidence indicators"""
        signals = self.trading_system.get_current_signals()
        
        if not signals:
            return
        
        symbols = list(signals.keys())
        signal_values = [signals[s]['combined_signal'] for s in symbols]
        confidence_values = [signals[s]['confidence'] for s in symbols]
        
        # Create bubble chart where size represents confidence
        colors = []
        for signal in signal_values:
            if signal > 0.5:
                colors.append(self.colors['success'])
            elif signal > 0:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['error'])
        
        fig.add_trace(
            go.Scatter(
                x=symbols,
                y=signal_values,
                mode='markers',
                name='Signal Strength',
                marker=dict(
                    color=colors,
                    size=[c * 30 + 10 for c in confidence_values],  # Size by confidence
                    line=dict(width=2, color=self.colors['text']),
                    opacity=0.8
                ),
                text=[f'{s}<br>Signal: {sig:.2f}<br>Confidence: {conf:.2f}' 
                      for s, sig, conf in zip(symbols, signal_values, confidence_values)],
                hovertemplate='<b>%{text}</b><br>' +
                            'Signal: %{y:.3f}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add signal zones
        fig.add_hrect(
            y0=0.5, y1=1,
            fillcolor=self.colors['success'], opacity=0.1,
            line_width=0, row=row, col=col
        )
        fig.add_hrect(
            y0=0, y1=0.5,
            fillcolor=self.colors['warning'], opacity=0.1,
            line_width=0, row=row, col=col
        )
        fig.add_hrect(
            y0=-1, y1=0,
            fillcolor=self.colors['error'], opacity=0.1,
            line_width=0, row=row, col=col
        )
    
    def _add_enhanced_confidence_intervals(self, fig, row, col):
        """Enhanced confidence intervals from Monte Carlo with risk metrics"""
        if not self.monte_carlo or not self.monte_carlo.simulations:
            return
            
        stats = self.monte_carlo.get_statistics()
        if not stats:
            return
        
        categories = ['Expected Return', '10th Percentile', '90th Percentile', 'VaR (95%)', 'Best Case']
        values = [
            stats['returns']['mean'],
            stats['returns']['percentile_10'],
            stats['returns']['percentile_90'],
            stats['risk_metrics']['var_95']['mean'],
            stats['returns']['percentile_95']
        ]
        
        # Color coding for risk levels
        colors = [
            self.colors['primary'],    # Expected
            self.colors['error'],      # 10th percentile
            self.colors['success'],    # 90th percentile
            self.colors['warning'],    # VaR
            self.colors['accent']      # Best case
        ]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                name='Monte Carlo Results',
                marker_color=colors,
                marker_line_color=self.colors['text'],
                marker_line_width=1,
                text=[f'{v:.1f}%' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                            'Value: %{y:.2f}%<br>' +
                            f'Based on {stats["simulation_count"]:,} simulations<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add benchmark lines
        fig.add_hline(
            y=0, line_width=1, line_dash="solid",
            line_color=self.colors['neutral'],
            row=row, col=col
        )
    
    def _add_asset_allocation_pie(self, fig, row, col):
        """Add asset allocation pie chart"""
        portfolio = self.trading_system.get_portfolio_status()
        
        if not portfolio.get('positions'):
            return
        
        labels = []
        values = []
        colors = []
        
        # Add positions
        for symbol, position in portfolio['positions'].items():
            labels.append(symbol)
            values.append(position['value'])
            colors.append(self.colors['primary'])
        
        # Add cash
        if portfolio.get('cash', 0) > 0:
            labels.append('Cash')
            values.append(portfolio['cash'])
            colors.append(self.colors['neutral'])
        
        # Create color palette for positions
        color_palette = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                        self.colors['warning'], self.colors['success']]
        colors = [color_palette[i % len(color_palette)] for i in range(len(labels))]
        
        # fig.add_trace(
        #     go.Pie(
        #         labels=labels,
        #         values=values,
        #         name="Asset Allocation",
        #         marker_colors=colors,
        #         textinfo='label+percent',
        #         textposition='auto',
        #         hovertemplate='<b>%{label}</b><br>' +
        #                     'Value: $%{value:,.2f}<br>' +
        #                     'Percentage: %{percent}<br>' +
        #                     '<extra></extra>'
        #     ),
        #     row=row, col=col
        # )
    
    def _add_performance_attribution(self, fig, row, col):
        """Add performance attribution analysis"""
        symbol_performance = {}
        
        for trade in self.trading_system.trades:
            if trade['action'] == 'SELL':
                symbol = trade['symbol']
                pnl_pct = trade.get('pnl_pct', 0)
                
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = []
                symbol_performance[symbol].append(pnl_pct)
        
        if symbol_performance:
            symbols = list(symbol_performance.keys())
            avg_returns = [np.mean(returns) for returns in symbol_performance.values()]
            volatilities = [np.std(returns) for returns in symbol_performance.values()]
            
            colors = [self.colors['success'] if ret > 0 else self.colors['error'] for ret in avg_returns]
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=avg_returns,
                    name='Average Return',
                    marker_color=colors,
                    marker_line_color=self.colors['text'],
                    marker_line_width=1,
                    error_y=dict(
                        type='data',
                        array=volatilities,
                        visible=True,
                        color=self.colors['neutral']
                    ),
                    hovertemplate='<b>%{x}</b><br>' +
                                'Avg Return: %{y:.2f}%<br>' +
                                'Volatility: %{error_y.array:.2f}%<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
        
    
    

    def _add_risk_return_profile(self, fig, row, col):
        import logging
        """Add risk-return scatter plot to the specified subplot."""

        subplot_type = fig.get_subplot(row=row, col=col)
        print(subplot_type)
        symbol_performance = {}
        
        # Collect performance data
        for trade in self.trading_system.trades:
            if trade.get('action') == 'SELL':
                symbol = trade.get('symbol')
                pnl_pct = trade.get('pnl_pct', 0)
                if not isinstance(pnl_pct, (int, float)):
                    logging.warning(f"Invalid pnl_pct for trade {trade}: {pnl_pct}. Using 0.")
                    pnl_pct = 0
                if symbol:
                    if symbol not in symbol_performance:
                        symbol_performance[symbol] = []
                    symbol_performance[symbol].append(pnl_pct)
        
        if not symbol_performance:
            logging.warning("No SELL trades found for risk-return profile.")
            return  # Skip plotting if no data
        
        # Compute metrics
        symbols = list(symbol_performance.keys())
        avg_returns = [np.mean(returns) for returns in symbol_performance.values()]
        volatilities = [np.std(returns) for returns in symbol_performance.values()]
        trade_counts = [len(returns) for returns in symbol_performance.values()]
        sizes = [max(10, min(30, count * 3)) for count in trade_counts]
        colors = [self.colors['success'] if ret > 0 else self.colors['error'] for ret in avg_returns]
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=avg_returns,
                mode='markers+text',
                name='Risk-Return Profile',
                text=symbols,
                textposition='middle center',
                customdata=trade_counts,
                marker=dict(
                    color=colors,
                    size=sizes,
                    line=dict(width=2, color=self.colors.get('text', '#000000')),
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>' +
                            'Return: %{y:.2f}%<br>' +
                            'Risk: %{x:.2f}%<br>' +
                            'Trades: %{customdata}<br>' +
                            '<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add quadrant lines
        if avg_returns and volatilities:
            avg_return = np.mean(avg_returns)
            avg_vol = np.mean(volatilities)
            fig.add_hline(
                y=avg_return,
                line_width=1,
                line_dash="dash",
                line_color=self.colors.get('neutral', '#888888'),
                row=row, col=col
            )
            fig.add_vline(
                x=avg_vol,
                line_width=1,
                line_dash="dash",
                line_color=self.colors.get('neutral', '#888888'),
                row=row, col=col
            )
        
        # Update axis labels
        fig.update_xaxes(title_text="Volatility (%)", row=row, col=col)
        fig.update_yaxes(title_text="Average Return (%)", row=row, col=col)
        
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio with enhanced precision"""
        sell_trades = [t for t in self.trading_system.trades if t['action'] == 'SELL']
        if not sell_trades:
            return 0
            
        returns = [t.get('pnl_pct', 0) for t in sell_trades]
        if len(returns) < 2:
            return 0
            
        # Annualize the Sharpe ratio
        daily_returns = np.array(returns) / 100  # Convert to decimal
        excess_returns = daily_returns - 0.02/252  # Assume 2% risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
            
        return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown with enhanced precision"""
        if not self.trading_system.trades:
            return 0
            
        current_balance = self.trading_system.trading_config.initial_balance
        peak_balance = current_balance
        max_drawdown = 0
        
        for trade in self.trading_system.trades:
            if trade['action'] == 'BUY':
                current_balance -= trade['cost']
            else:
                current_balance += trade['revenue']
            
            if current_balance > peak_balance:
                peak_balance = current_balance
            
            drawdown = (peak_balance - current_balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100
    
    def _calculate_win_rate(self):
        """Calculate win rate with enhanced precision"""
        sell_trades = [t for t in self.trading_system.trades if t['action'] == 'SELL']
        if not sell_trades:
            return 0
            
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(sell_trades) * 100
    
    def create_investor_report(self, save_path=None):
        """
        Create a comprehensive investor report with professional formatting
        
        Generates a detailed markdown report suitable for institutional investors,
        complete with executive summary, risk analysis, and performance metrics.
        """
        print("\n📊 Generating Investor Report...")
        print("📋 Compiling performance metrics...")
        
        portfolio = self.trading_system.get_portfolio_status()
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Calculate additional metrics
        total_trades = len(self.trading_system.trades)
        buy_trades = len([t for t in self.trading_system.trades if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trading_system.trades if t['action'] == 'SELL'])
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        win_rate = self._calculate_win_rate()
        
        # Performance metrics
        total_return = portfolio.get('total_return', 0)
        total_realized = portfolio.get('total_realized_returns', 0)
        current_value = portfolio.get('total_value', 0)
        
        report = f"""
# 🚀 XENIA V2 TRADING SYSTEM
## Institutional Investment Report

**Report Date:** {current_date}  
**Reporting Period:** {self.trading_system.trades[0]['date'] if self.trading_system.trades else 'N/A'} - {self.trading_system.trades[-1]['date'] if self.trading_system.trades else 'N/A'}  
**Strategy:** XENIA V2 Algorithmic Trading System

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|---------|
| **Total Return** | {total_return:.2f}% | {'🟢 Positive' if total_return > 0 else '🔴 Negative'} |
| **Realized Returns** | {total_realized:.2f}% | {'🟢 Positive' if total_realized > 0 else '🔴 Negative'} |
| **Portfolio Value** | ${current_value:,.2f} | {'🟢 Above Initial' if current_value > self.trading_system.trading_config.initial_balance else '🔴 Below Initial'} |
| **Sharpe Ratio** | {sharpe_ratio:.3f} | {'🟢 Excellent' if sharpe_ratio > 1.5 else '🟡 Good' if sharpe_ratio > 1 else '🔴 Poor'} |
| **Maximum Drawdown** | {max_drawdown:.2f}% | {'🟢 Low Risk' if max_drawdown < 10 else '🟡 Moderate Risk' if max_drawdown < 20 else '🔴 High Risk'} |
| **Win Rate** | {win_rate:.1f}% | {'🟢 Strong' if win_rate > 60 else '🟡 Moderate' if win_rate > 40 else '🔴 Weak'} |

---

## 📈 Performance Analysis

### Trading Activity
- **Total Trades Executed:** {total_trades:,}
- **Buy Orders:** {buy_trades:,}
- **Sell Orders:** {sell_trades:,}
- **Average Trade Size:** ${(current_value / max(1, total_trades)):,.2f}

### Risk Assessment
- **Volatility:** {self._calculate_portfolio_volatility():.2f}% (Annualized)
- **Beta:** {self._calculate_portfolio_beta():.3f}
- **Alpha:** {self._calculate_portfolio_alpha():.2f}%
- **Information Ratio:** {self._calculate_information_ratio():.3f}

---

## 🎯 Portfolio Composition

### Current Holdings
"""
        
        if portfolio.get('positions'):
            total_portfolio_value = portfolio['total_value']
            
            report += "\n| Symbol | Position Value | Allocation | P&L | Status |\n"
            report += "|--------|---------------|------------|-----|--------|\n"
            
            for symbol, position in portfolio['positions'].items():
                allocation = (position['value'] / total_portfolio_value) * 100
                pnl = position.get('unrealized_pnl', 0)
                status = '🟢 Profitable' if pnl > 0 else '🔴 Loss' if pnl < 0 else '🟡 Neutral'
                
                report += f"| **{symbol}** | ${position['value']:,.2f} | {allocation:.1f}% | ${pnl:,.2f} | {status} |\n"
        
        cash_allocation = (portfolio['cash'] / portfolio['total_value']) * 100
        report += f"| **Cash** | ${portfolio['cash']:,.2f} | {cash_allocation:.1f}% | - | 🟢 Liquid |\n"
        
        report += f"""

---

## 📊 Risk Analysis
"""
        
        if self.monte_carlo and self.monte_carlo.simulations:
            stats = self.monte_carlo.get_statistics()
            report += f"""
### Monte Carlo Simulation Results
*Based on {stats['simulation_count']:,} simulations*

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Expected Return** | {stats['returns']['mean']:.2f}% | {'🟢 Positive outlook' if stats['returns']['mean'] > 0 else '🔴 Negative outlook'} |
| **Return Volatility** | {stats['returns']['std']:.2f}% | {'🟢 Low volatility' if stats['returns']['std'] < 15 else '🟡 Moderate volatility' if stats['returns']['std'] < 25 else '🔴 High volatility'} |
| **5th Percentile** | {stats['returns']['percentile_5']:.2f}% | Worst-case scenario |
| **95th Percentile** | {stats['returns']['percentile_95']:.2f}% | Best-case scenario |
| **Probability of Profit** | {stats['probability_positive']:.1f}% | {'🟢 High confidence' if stats['probability_positive'] > 60 else '🟡 Moderate confidence' if stats['probability_positive'] > 40 else '🔴 Low confidence'} |
| **Value at Risk (95%)** | {stats['risk_metrics']['var_95']['mean']:.2f}% | Maximum expected daily loss |
| **Expected Shortfall** | {stats['risk_metrics']['cvar_95']['mean']:.2f}% | Average loss beyond VaR |

### Risk Metrics Summary
- **Sortino Ratio:** {stats['sortino_ratio']:.3f}
- **Maximum Simulated Drawdown:** {stats['risk_metrics']['max_drawdown']['percentile_95']:.2f}%
- **Probability of >10% Loss:** {stats['probability_loss_10']:.1f}%
- **Probability of >20% Gain:** {stats['probability_gain_20']:.1f}%
"""
        
        report += f"""

---

## 📋 Trading Performance by Asset

| Symbol | Trades | Total P&L | Avg P&L | Win Rate | Best Trade | Worst Trade |
|--------|---------|-----------|---------|----------|------------|-------------|
"""
        
        symbol_performance = {}
        symbol_trades = {}
        symbol_wins = {}
        symbol_best = {}
        symbol_worst = {}
        
        for trade in self.trading_system.trades:
            if trade['action'] == 'SELL':
                symbol = trade['symbol']
                pnl = trade.get('pnl', 0)
                
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = 0
                    symbol_trades[symbol] = 0
                    symbol_wins[symbol] = 0
                    symbol_best[symbol] = float('-inf')
                    symbol_worst[symbol] = float('inf')
                
                symbol_performance[symbol] += pnl
                symbol_trades[symbol] += 1
                
                if pnl > 0:
                    symbol_wins[symbol] += 1
                
                symbol_best[symbol] = max(symbol_best[symbol], pnl)
                symbol_worst[symbol] = min(symbol_worst[symbol], pnl)
        
        for symbol in symbol_performance:
            trades_count = symbol_trades[symbol]
            total_pnl = symbol_performance[symbol]
            avg_pnl = total_pnl / trades_count if trades_count > 0 else 0
            win_rate_symbol = (symbol_wins[symbol] / trades_count * 100) if trades_count > 0 else 0
            best_trade = symbol_best[symbol] if symbol_best[symbol] != float('-inf') else 0
            worst_trade = symbol_worst[symbol] if symbol_worst[symbol] != float('inf') else 0
            
            report += f"| **{symbol}** | {trades_count} | ${total_pnl:.2f} | ${avg_pnl:.2f} | {win_rate_symbol:.1f}% | ${best_trade:.2f} | ${worst_trade:.2f} |\n"
        
        report += f"""

---

## 🎯 Strategic Recommendations

### Immediate Actions
"""
        
        # Generate recommendations based on performance
        if total_return > 10:
            report += "- ✅ **Continue Current Strategy**: Portfolio is performing well above market average\n"
        elif total_return > 0:
            report += "- 🟡 **Monitor Performance**: Positive returns but room for improvement\n"
        else:
            report += "- ⚠️ **Review Strategy**: Portfolio underperforming, consider adjustments\n"
        
        if max_drawdown > 20:
            report += "- 🔴 **Risk Management**: High drawdown detected, implement stricter stop-losses\n"
        elif max_drawdown > 10:
            report += "- 🟡 **Risk Monitoring**: Moderate drawdown, maintain current risk controls\n"
        else:
            report += "- ✅ **Risk Control**: Excellent drawdown management\n"
        
        if win_rate < 40:
            report += "- 📊 **Strategy Refinement**: Low win rate suggests need for signal optimization\n"
        elif win_rate > 60:
            report += "- ✅ **Signal Quality**: High win rate indicates strong predictive signals\n"
        
        report += f"""

### Risk Management
- **Position Sizing**: Current allocation appears {'well-diversified' if len(portfolio.get('positions', {})) > 3 else 'concentrated'}
- **Cash Management**: {cash_allocation:.1f}% cash position provides {'adequate' if cash_allocation > 10 else 'limited'} liquidity
- **Rebalancing**: {'Recommended' if max([pos['value'] for pos in portfolio.get('positions', {}).values()] + [0]) / portfolio['total_value'] > 0.3 else 'Not needed'} based on current allocations

---

## 📞 Contact Information

**XENIA V2 Trading System**  
*Algorithmic Trading Platform*

For questions regarding this report or the trading system:
- 📧 Email: trading@xenia.ai
- 📱 Support: 1-800-XENIA-V2
- 🌐 Web: www.xenia-trading.com

---

*This report is generated automatically by the XENIA V2 trading system. All metrics are calculated based on actual trading data and Monte Carlo simulations where applicable. Past performance does not guarantee future results.*

**Risk Disclaimer:** Trading involves substantial risk of loss and is not suitable for all investors. The value of investments may go down as well as up. Please consider your risk tolerance and investment objectives before trading.
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Investor report saved to: {save_path}")
        
        print("📊 Report generation complete!")
        return report
    
    def _calculate_portfolio_volatility(self):
        """Calculate annualized portfolio volatility"""
        if not self.trading_system.trades:
            return 0
            
        sell_trades = [t for t in self.trading_system.trades if t['action'] == 'SELL']
        if len(sell_trades) < 2:
            return 0
            
        returns = [t.get('pnl_pct', 0) for t in sell_trades]
        daily_vol = np.std(returns)
        
        # Annualize assuming 252 trading days
        return daily_vol * np.sqrt(252)
    
    def _calculate_portfolio_beta(self):
        """Calculate portfolio beta (simplified)"""
        # This is a simplified beta calculation
        # In practice, you'd need market data for proper beta calculation
        return 1.0  # Placeholder
    
    def _calculate_portfolio_alpha(self):
        """Calculate portfolio alpha"""
        # Simplified alpha calculation
        portfolio_return = self.trading_system.get_portfolio_status().get('total_return', 0)
        benchmark_return = 8.0  # Assuming 8% market return
        
        return portfolio_return - benchmark_return
    
    def _calculate_information_ratio(self):
        """Calculate information ratio"""
        alpha = self._calculate_portfolio_alpha()
        tracking_error = self._calculate_portfolio_volatility()
        
        return alpha / tracking_error if tracking_error > 0 else 0


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
    import time


    async def run_backtest(self):
        """Run backtest on all available data"""
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

        print(f"\033[94m🔍 SCANNING {len(reference_data)} TEMPORAL NODES\033[0m")

        if len(reference_data) == 0:
            print("\033[91m❌ TEMPORAL MATRIX CORRUPTED\033[0m")
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

@dataclass
class TradingConfig:
    """Configuration for the trading system"""
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
    initial_balance: float = 1000.0
    risk_per_trade: float = 0.01  # 1% of buying power per trade
    
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
        if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        market_date = market_time.date().strftime("%Y-%m-%d")
        if market_date in self.holidays_2024 + self.holidays_2025:
            return False
        
        # Check market hours (9:30 AM - 4:00 PM EST)
        market_hour = market_time.hour
        market_minute = market_time.minute
        
        # Market opens at 9:30 AM
        if market_hour < 9 or (market_hour == 9 and market_minute < 30):
            return False
        
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
            
            # Import your trading system modules here
            # from your_trading_system import create_default_system, create_default_trader, DataFetcherConfig
            
            # Create custom data fetcher configs
            custom_data_configs = {}
            for symbol in self.config.symbols:
                custom_data_configs[symbol] = DataFetcherConfig(symbol=symbol, period='1y')
                
            
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