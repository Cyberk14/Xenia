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
from typing import Dict, List, Tuple, Optional
import ta
from scipy.signal import savgol_filter
from functools import lru_cache
import threading

warnings.filterwarnings("ignore")


class XeniaV2:
    def __init__(self, symbols, initial_balance=10000, transaction_cost=0.002):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.data = {}
        self.models = {}
        self.scalers = {}
        self.trades = []
        self.win_trades = []
        self.lose_trades = []
        self.transaction_cost = transaction_cost

        # More realistic thresholds
        self.min_accuracy_threshold = 0.35  # Lowered from 0.35
        self.signal_threshold = 0.15  # Lowered from 0.3
        self.confidence_threshold = 0.4  # Lowered from 0.5
        self.max_position_size = 0.2

        self.latest_signals = {}

        # Add thread locks for Flask safety
        self.data_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()

        # Rate limiting for API calls
        self.last_fetch_time = {}
        self.min_fetch_interval = 1.0  # Minimum seconds between fetches per symbol

    def safe_fetch_ticker_data(self, symbol, period="5y", max_retries=3):
        """Safely fetch ticker data with retry logic and rate limiting"""
        try:
            # Rate limiting
            now = time.time()
            if symbol in self.last_fetch_time:
                elapsed = now - self.last_fetch_time[symbol]
                if elapsed < self.min_fetch_interval:
                    time.sleep(self.min_fetch_interval - elapsed)

            self.last_fetch_time[symbol] = time.time()

            # Retry logic
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(symbol)

                    # Check if ticker object is valid
                    if ticker is None:
                        print(f"Failed to create ticker object for {symbol}")
                        continue

                    # Try to get basic info first to validate ticker
                    try:
                        info = ticker.info
                        if info is None or len(info) == 0:
                            print(
                                f"No info available for {symbol}, attempt {attempt + 1}"
                            )
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                continue
                    except Exception as info_error:
                        print(f"Info check failed for {symbol}: {info_error}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue

                    # Get historical data
                    data = ticker.history(period=period)

                    if data is None or data.empty:
                        print(f"No historical data for {symbol}, attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return None

                    print(
                        f"✓ Successfully fetched {len(data)} days of data for {symbol}"
                    )
                    return data

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"All attempts failed for {symbol}")
                        return None

            return None

        except Exception as e:
            print(f"Critical error fetching data for {symbol}: {e}")
            return None

    async def fetch_stock_data_async(self, symbol, period="5y"):
        """Asynchronously fetch stock data with improved error handling"""
        try:
            # Use thread-safe synchronous call within async context
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.safe_fetch_ticker_data, symbol, period
            )
            return symbol, data
        except Exception as e:
            print(f"Async error fetching data for {symbol}: {e}")
            return symbol, None

    async def fetch_all_data(self):
        """Fetch data for all symbols asynchronously with thread safety"""
        with self.data_lock:
            tasks = [self.fetch_stock_data_async(symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"Exception in fetch_all_data: {result}")
                    continue

                symbol, data = result
                if data is not None and not data.empty and len(data) > 100:
                    self.data[symbol] = data
                    print(f"✓ Stored {len(data)} days of data for {symbol}")
                else:
                    print(f"✗ Failed to fetch sufficient data for {symbol}")

    def create_features(self, df, current_idx=None):
        """Simplified but effective feature engineering with better error handling"""
        try:
            if current_idx is not None:
                df = df.iloc[: current_idx + 1].copy()

            if len(df) < 50:  # Need minimum data for features
                return None

            features = pd.DataFrame(index=df.index)

            # Basic price features
            features["returns"] = df["Close"].pct_change()
            features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

            # Simple moving averages
            for window in [5, 10, 20, 50]:
                if len(df) > window:
                    features[f"sma_{window}"] = df["Close"].rolling(window).mean()
                    features[f"price_to_sma_{window}"] = (
                        df["Close"] / features[f"sma_{window}"]
                    )

            # Exponential moving averages
            for span in [5, 10, 20]:
                features[f"ema_{span}"] = df["Close"].ewm(span=span).mean()
                features[f"price_to_ema_{span}"] = df["Close"] / features[f"ema_{span}"]

            # RSI with fallback
            try:
                features["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            except:
                # Fallback RSI calculation
                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features["rsi"] = 100 - (100 / (1 + rs))

            # MACD with fallback
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
            sma20 = df["Close"].rolling(20).mean()
            std20 = df["Close"].rolling(20).std()
            features["bb_upper"] = sma20 + (std20 * 2)
            features["bb_lower"] = sma20 - (std20 * 2)
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
            for period in [3, 5, 10, 20]:
                features[f"momentum_{period}"] = (
                    df["Close"] / df["Close"].shift(period) - 1
                )

            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f"returns_lag_{lag}"] = features["returns"].shift(lag)
                features[f"rsi_lag_{lag}"] = features["rsi"].shift(lag)

            # Rolling statistics
            features["returns_mean_5"] = features["returns"].rolling(5).mean()
            features["returns_std_5"] = features["returns"].rolling(5).std()
            features["returns_mean_20"] = features["returns"].rolling(20).mean()
            features["returns_std_20"] = features["returns"].rolling(20).std()

            return features

        except Exception as e:
            print(f"Error creating features: {e}")
            return None

    def create_labels(self, df, forward_days=3, threshold=0.01):
        """Simplified labeling strategy"""
        try:
            close_prices = df["Close"].values
            labels = np.full(len(df), 0)  # Default to hold

            for i in range(len(df) - forward_days):
                current_price = close_prices[i]
                future_price = close_prices[i + forward_days]

                return_pct = (future_price - current_price) / current_price

                if return_pct > threshold:
                    labels[i] = 1  # Buy
                elif return_pct < -threshold:
                    labels[i] = -1  # Sell
                # else: 0 (Hold)

            return labels
        except Exception as e:
            print(f"Error creating labels: {e}")
            return None

    def train_model_for_symbol(self, symbol):
        """Train a simplified but robust model with thread safety"""
        try:
            with self.model_lock:
                print(f"Training model for {symbol}...")

                if symbol not in self.data:
                    print(f"No data available for {symbol}")
                    return None

                df = self.data[symbol]

                # Create features and labels
                features_df = self.create_features(df)
                if features_df is None:
                    print(f"Failed to create features for {symbol}")
                    return None

                labels = self.create_labels(df)
                if labels is None:
                    print(f"Failed to create labels for {symbol}")
                    return None

                # Combine and clean
                combined_df = features_df.copy()
                combined_df["target"] = labels

                # Remove rows with NaN values
                combined_df = combined_df.dropna()

                print(f"{symbol}: {len(combined_df)} samples after cleaning")

                if len(combined_df) < 50:
                    print(f"{symbol}: Insufficient data after cleaning")
                    return None

                # Prepare features
                feature_cols = [col for col in combined_df.columns if col != "target"]
                X = combined_df[feature_cols].values
                y = combined_df["target"].values

                # Check class distribution
                unique, counts = np.unique(y, return_counts=True)
                print(f"{symbol}: Class distribution - {dict(zip(unique, counts))}")

                # Skip if too imbalanced or insufficient samples
                if len(unique) < 2 or min(counts) < 10:
                    print(f"{symbol}: Insufficient class diversity")
                    return None

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train model with cross-validation
                model = RandomForestClassifier(
                    n_estimators=50,  # Reduced for speed
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=1,  # Changed from -1 to 1 for Flask safety
                )

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []

                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    scores.append(score)

                avg_score = np.mean(scores)
                print(f"{symbol}: Average CV accuracy: {avg_score:.3f}")

                if avg_score < self.min_accuracy_threshold:
                    print(f"{symbol}: Model accuracy too low")
                    return None

                # Train final model on all data
                model.fit(X_scaled, y)

                return {
                    "model": model,
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "accuracy": avg_score,
                }

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
                    print(
                        f"✓ {symbol}: Model trained successfully (accuracy: {model_data['accuracy']:.3f})"
                    )
                else:
                    print(f"✗ {symbol}: Model training failed")

    def get_prediction(self, symbol, current_idx):
        """Get model prediction for current data point with comprehensive error handling"""
        try:
            # Create cache key
            cache_key = f"{symbol}_{current_idx}"

            # Check cache first
            with self.cache_lock:
                if cache_key in self.prediction_cache:
                    return self.prediction_cache[cache_key]

            # Validate inputs
            if symbol not in self.models:
                print(f"No model available for {symbol}")
                return 0, 0.5

            if symbol not in self.data:
                print(f"No data available for {symbol}")
                return 0, 0.5

            model_data = self.models[symbol]
            df = self.data[symbol]

            # Validate model data
            if not all(
                key in model_data for key in ["model", "scaler", "feature_cols"]
            ):
                print(f"Invalid model data for {symbol}")
                return 0, 0.5

            # Validate current_idx
            if current_idx >= len(df) or current_idx < 0:
                print(
                    f"Invalid index {current_idx} for {symbol} (data length: {len(df)})"
                )
                return 0, 0.5

            # Create features up to current index
            features_df = self.create_features(df, current_idx)

            if features_df is None:
                print(f"Failed to create features for {symbol} at index {current_idx}")
                return 0, 0.5

            if current_idx >= len(features_df):
                print(
                    f"Index {current_idx} out of range for features (length: {len(features_df)})"
                )
                return 0, 0.5

            # Get current features
            try:
                current_features = features_df[model_data["feature_cols"]].iloc[
                    current_idx : current_idx + 1
                ]
            except KeyError as e:
                print(f"Missing feature columns for {symbol}: {e}")
                return 0, 0.5
            except IndexError as e:
                print(f"Index error getting features for {symbol}: {e}")
                return 0, 0.5

            # Check for NaN values
            if current_features.isnull().any().any():
                print(f"NaN values in features for {symbol} at index {current_idx}")
                return 0, 0.5

            # Scale and predict
            try:
                X_scaled = model_data["scaler"].transform(current_features.values)
            except Exception as e:
                print(f"Error scaling features for {symbol}: {e}")
                return 0, 0.5

            # Get prediction and confidence
            try:
                model = model_data["model"]
                prediction = model.predict(X_scaled)[0]

                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_scaled)[0]
                    confidence = np.max(probas)
                else:
                    confidence = 0.6

                # Convert prediction to signal
                signal = float(prediction * 0.5)  # Scale down signal strength
                confidence = float(confidence)

                # Cache the result
                with self.cache_lock:
                    self.prediction_cache[cache_key] = (signal, confidence)
                    # Limit cache size
                    if len(self.prediction_cache) > 1000:
                        # Remove oldest entries
                        keys_to_remove = list(self.prediction_cache.keys())[:100]
                        for key in keys_to_remove:
                            del self.prediction_cache[key]

                return signal, confidence

            except Exception as e:
                print(f"Error in model prediction for {symbol}: {e}")
                return 0, 0.5

        except Exception as e:
            print(f"Critical error getting prediction for {symbol}: {e}")
            import traceback

            traceback.print_exc()
            return 0, 0.5

    def get_technical_signal(self, symbol, current_idx):
        """Get technical analysis signal with improved error handling"""
        try:
            if symbol not in self.data:
                return 0, 0.5

            df = self.data[symbol]

            if current_idx >= len(df) or current_idx < 0:
                return 0, 0.5

            features_df = self.create_features(df, current_idx)

            if features_df is None or current_idx >= len(features_df):
                return 0, 0.5

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
            print(f"Error in technical signal for {symbol}: {e}")
            return 0, 0.5

    def execute_trade(self, symbol, current_idx):
        """Execute trade based on combined signals"""
        try:
            if symbol not in self.data or current_idx >= len(self.data[symbol]):
                return

            current_price = self.data[symbol]["Close"].iloc[current_idx]
            current_date = self.data[symbol].index[current_idx]

            # Get model prediction
            model_signal, model_confidence = self.get_prediction(symbol, current_idx)

            # Get technical signal
            tech_signal, tech_confidence = self.get_technical_signal(
                symbol, current_idx
            )

            # Combine signals (weighted average)
            combined_signal = model_signal * 0.6 + tech_signal * 0.4
            combined_confidence = model_confidence * 0.6 + tech_confidence * 0.4

            # Store signal for monitoring
            self.latest_signals[symbol] = {
                "model_signal": model_signal,
                "tech_signal": tech_signal,
                "combined_signal": combined_signal,
                "confidence": combined_confidence,
                "price": current_price,
                "date": current_date,
            }

            # Trading logic with lower thresholds
            if (
                combined_signal > self.signal_threshold
                and combined_confidence > self.confidence_threshold
            ):
                # Buy signal
                if self.positions[symbol] == 0:  # Only if not holding
                    self.execute_buy(
                        symbol,
                        current_price,
                        current_date,
                        combined_signal,
                        combined_confidence,
                    )

            elif (
                combined_signal < -self.signal_threshold
                and combined_confidence > self.confidence_threshold
            ):
                # Sell signal
                if self.positions[symbol] > 0:  # Only if holding
                    self.execute_sell(
                        symbol,
                        current_price,
                        current_date,
                        combined_signal,
                        combined_confidence,
                    )

        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")

    def execute_buy(self, symbol, price, date, signal, confidence):
        """Execute buy order"""
        try:
            # Calculate position size (percentage of current balance)
            position_value = self.balance * self.max_position_size
            shares = position_value / price
            cost = shares * price
            transaction_fee = cost * self.transaction_cost
            total_cost = cost + transaction_fee

            if total_cost <= self.balance:
                self.positions[symbol] = shares
                self.balance -= total_cost

                trade = {
                    "symbol": symbol,
                    "action": "BUY",
                    "price": price,
                    "shares": shares,
                    "cost": total_cost,
                    "signal": signal,
                    "confidence": confidence,
                    "date": date,
                    "balance_after": self.balance,
                }
                self.trades.append(trade)

                print(
                    f"BUY {symbol}: {shares:.2f} shares @ ${price:.2f} | Signal: {signal:.3f} | Confidence: {confidence:.3f}"
                )

        except Exception as e:
            print(f"Error executing buy for {symbol}: {e}")

    def execute_sell(self, symbol, price, date, signal, confidence):
        """Execute sell order"""
        try:
            if self.positions[symbol] > 0:
                shares = self.positions[symbol]
                revenue = shares * price
                transaction_fee = revenue * self.transaction_cost
                net_revenue = revenue - transaction_fee

                self.balance += net_revenue

                # Calculate P&L
                buy_trades = [
                    t
                    for t in self.trades
                    if t["symbol"] == symbol and t["action"] == "BUY"
                ]
                if buy_trades:
                    last_buy = buy_trades[-1]
                    pnl = net_revenue - last_buy["cost"]
                    pnl_pct = (pnl / last_buy["cost"]) * 100
                else:
                    pnl = 0
                    pnl_pct = 0

                self.positions[symbol] = 0

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
                    "balance_after": self.balance,
                }
                self.trades.append(trade)
                if pnl <= 0:
                    self.lose_trades.append(trade)
                elif pnl > 0:
                    self.win_trades.append(trade)

                print(
                    f"🔴 SELL {symbol}: {shares:.2f} shares @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
                )

        except Exception as e:
            print(f"Error executing sell for {symbol}: {e}")

    async def run_backtest(self, start_date, end_date):
        """Run backtest with improved date handling"""
        print("=" * 60)
        print("IMPROVED ML TRADING SYSTEM - BACKTEST")
        print("=" * 60)

        # Fetch data
        await self.fetch_all_data()

        if not self.data:
            print("No data available")
            return

        # Train models
        await self.train_all_models()

        if not self.models:
            print("No models trained successfully")
            return

        print(f"\nRunning backtest from {start_date} to {end_date}...")

        # Get the symbol with most data for date reference
        max_symbol = max(self.data.keys(), key=lambda x: len(self.data[x]))
        reference_data = self.data[max_symbol]

        # Filter by date range
        mask = (reference_data.index >= start_date) & (reference_data.index < end_date)
        backtest_data = reference_data[mask]

        print(f"Backtesting on {len(backtest_data)} trading days")

        if len(backtest_data) == 0:
            print("No data in specified date range")
            return

        # Execute trades
        progress_interval = max(1, len(backtest_data) // 20)

        for i, date in enumerate(backtest_data.index):
            if i % progress_interval == 0:
                progress = (i / len(backtest_data)) * 100
                print(f"Progress: {progress:.1f}% ({i}/{len(backtest_data)})")

            # Execute trades for each symbol
            for symbol in self.symbols:
                if symbol in self.data and symbol in self.models:
                    try:
                        # Find corresponding index in symbol's data
                        symbol_data = self.data[symbol]
                        if date in symbol_data.index:
                            symbol_idx = symbol_data.index.get_loc(date)
                            # Ensure we have enough historical data for features
                            if symbol_idx >= 50:  # Need at least 50 days for features
                                self.execute_trade(symbol, symbol_idx)
                    except Exception as e:
                        continue

        # Generate results
        self.generate_results()

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
        total_return = (total_value - self.initial_balance) / self.initial_balance * 100
        total_balance_return = (final_balance - self.initial_balance) / self.initial_balance * 100

        print(f"Initial Balance: ${self.initial_balance:,.2f}")
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
            profitable_trades = [t for t in sell_trades if t["pnl"] > 0]
            win_rate = len(profitable_trades) / len(sell_trades) * 100
            avg_profit = np.mean([t["pnl"] for t in sell_trades])
            avg_profit_pct = np.mean([t["pnl_pct"] for t in sell_trades])

            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average P&L per trade: ${avg_profit:.2f} ({avg_profit_pct:.2f}%)")

            if profitable_trades:
                avg_win = np.mean([t["pnl"] for t in profitable_trades])
                print(f"Average Winning Trade: ${avg_win:.2f}")

            losing_trades = [t for t in sell_trades if t["pnl"] <= 0]
            if losing_trades:
                avg_loss = np.mean([t["pnl"] for t in losing_trades])
                print(f"Average Losing Trade: ${avg_loss:.2f}")

        # Symbol performance
        print("\nSymbol Performance:")
        for symbol in self.symbols:
            symbol_trades = [t for t in sell_trades if t["symbol"] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t["pnl"] for t in symbol_trades)
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

    def get_current_signals(self):
        """Get current trading signals for all symbols"""
        signals = {}

        for symbol in self.symbols:
            if symbol in self.data and symbol in self.models:
                try:
                    latest_idx = len(self.data[symbol]) - 1
                    model_signal, model_confidence = self.get_prediction(
                        symbol, latest_idx
                    )
                    tech_signal, tech_confidence = self.get_technical_signal(
                        symbol, latest_idx
                    )

                    combined_signal = model_signal * 0.4 + tech_signal * 0.6
                    combined_confidence = model_confidence * 0.4 + tech_confidence * 0.6

                    current_price = self.data[symbol]["Close"].iloc[-1]

                    signals[symbol] = {
                        "model_signal": float(model_signal),
                        "tech_signal": float(tech_signal),
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

    def get_recommendation(self, signal, confidence):
        """Convert signal to human-readable recommendation"""
        if signal > self.signal_threshold and confidence > self.confidence_threshold:
            return "BUY"
        elif signal < -self.signal_threshold and confidence > self.confidence_threshold:
            return "SELL"
        else:
            return "HOLD"

    def get_portfolio_status(self):
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
            (portfolio["total_value"] - self.initial_balance) / self.initial_balance
        ) * 100
        portfolio["total_realized_returns"] = (
            (self.balance - self.initial_balance) / self.initial_balance
        ) * 100

        return portfolio

# Main execution
async def main():
    """Main function for standalone execution"""
    print("Initializing Xenia V2 Trading System...")

    # Create trading system with popular tech stocks
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    system = XeniaV2(symbols, initial_balance=100000)

    # Run backtest
    await system.run_backtest(start_date="2022-06-01", end_date="2024-01-01")

    # Show current signals
    print("\n" + "=" * 60)
    print("CURRENT SIGNALS")
    print("=" * 60)

    signals = system.get_current_signals()
    for symbol, signal_data in signals.items():
        print(
            f"{symbol}: {signal_data['recommendation']} | "
            f"Signal: {signal_data['combined_signal']:.3f} | "
            f"Confidence: {signal_data['confidence']:.3f} | "
            f"Price: ${signal_data['price']:.2f}"
        )

