"""
*Phase 1*: IndicatorBlock System - Living Adaptive Trading Intelligence

This module implements the foundational layer of the adaptive trading system,
processing raw OHLCV data into normalized, confidence-tagged, probabilistic signals.

Each indicator outputs a structured signal with:
- Normalized signal value [-1, 1]
- Confidence score [0, 1] 
- Probability P(up) of positive move
- Variance/uncertainty estimate
- Human-readable explanation
- Diagnostic metadata
"""


import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.special import expit  # sigmoid function
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class SignalOutput:
    """
    Structured output format for all indicators.
    Every signal must conform to this interface.
    """
    signal_value: float          # Normalized signal score [-1, 1]
    confidence: float            # Confidence level [0, 1]
    probability: float           # P(move > 0) [0, 1]
    variance: float              # Uncertainty/volatility estimate
    explanation: str             # Human-readable reasoning
    metadata: Dict[str, Any]     # Diagnostic information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def __post_init__(self):
        """Validate signal output bounds"""
        assert -1 <= self.signal_value <= 1, f"Signal value {self.signal_value} outside [-1, 1]"
        assert 0 <= self.confidence <= 1, f"Confidence {self.confidence} outside [0, 1]"
        assert 0 <= self.probability <= 1, f"Probability {self.probability} outside [0, 1]"
        assert self.variance >= 0, f"Variance {self.variance} must be non-negative"


class BaseIndicator(ABC):
    """
    Abstract base class for all indicators.
    Enforces consistent interface and probabilistic reasoning.
    """
    
    def __init__(self, window: int = 20, confidence_window: int = 100):
        self.window = window
        self.confidence_window = confidence_window
        self.history: List[Dict[str, float]] = []  # Track performance for confidence
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the raw indicator value and metadata.
        Must be implemented by each indicator.
        
        Returns:
            Tuple of (raw_signal_value, metadata_dict)
        """
        pass
    
    def normalize_signal(self, raw_value: float, volatility: float = 1.0) -> float:
        """
        Normalize raw signal to [-1, 1] range using tanh scaling.
        Volatility adjustment prevents over-normalization in quiet periods.
        """
        if np.isnan(raw_value) or raw_value == 0:
            return 0.0
        
        # Adaptive scaling based on volatility
        scale_factor = max(0.5, min(2.0, 1.0 / volatility)) if volatility > 0 else 1.0
        normalized = np.tanh(raw_value * scale_factor)
        
        return np.clip(normalized, -1.0, 1.0)
    
    def calculate_confidence(self, data: pd.DataFrame, signal_strength: float, 
                           volatility: float) -> float:
        """
        Calculate confidence based on multiple factors:
        1. Signal strength (stronger signals = higher confidence)
        2. Market volatility (lower vol = higher confidence)
        3. Historical accuracy (rolling performance)
        4. Data quality (sufficient history, no gaps)
        """
        confidence_factors = []
        
        # Factor 1: Signal strength
        strength_confidence = abs(signal_strength)
        confidence_factors.append(strength_confidence)
        
        # Factor 2: Volatility adjustment (lower vol = higher confidence)
        vol_confidence = np.exp(-2 * volatility) if volatility > 0 else 0.5
        confidence_factors.append(vol_confidence)
        
        # Factor 3: Historical accuracy (if we have performance history)
        if len(self.history) >= 10:
            recent_accuracy = np.mean([h.get('hit', 0.5) for h in self.history[-20:]])
            accuracy_confidence = recent_accuracy
            confidence_factors.append(accuracy_confidence)
        else:
            # Default moderate confidence when no history
            confidence_factors.append(0.5)
        
        # Factor 4: Data quality
        data_quality = min(1.0, len(data) / self.window) if len(data) > 0 else 0.0
        confidence_factors.append(data_quality)
        
        # Combine factors using geometric mean (conservative)
        final_confidence = np.prod(confidence_factors) ** (1/len(confidence_factors))
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def signal_to_probability(self, signal: float, confidence: float) -> float:
        """
        Convert normalized signal to probability P(up) using sigmoid transformation.
        Higher confidence increases the steepness of the sigmoid.
        """
        # Confidence affects the steepness of probability mapping
        steepness = 1.0 + 3.0 * confidence  # Range: [1, 4]
        probability = expit(signal * steepness)
        
        return np.clip(probability, 0.01, 0.99)  # Avoid extreme probabilities
    
    def estimate_variance(self, data: pd.DataFrame, lookback: int = 20) -> float:
        """
        Estimate the variance of future returns based on recent volatility.
        This helps quantify uncertainty in our predictions.
        """
        if len(data) < lookback:
            return 0.01  # Default small variance
        
        recent_returns = data['close'].pct_change().dropna().tail(lookback)
        if len(recent_returns) == 0:
            return 0.01
        
        return max(0.001, recent_returns.var())
    
    def update_performance(self, predicted_return: float, actual_return: float):
        """
        Update performance history for confidence calibration.
        This enables the system to learn and adapt over time.
        """
        hit = (predicted_return * actual_return) > 0  # Same direction
        accuracy = 1.0 - abs(predicted_return - actual_return) / (abs(actual_return) + 1e-8)
        
        performance_record = {
            'predicted': predicted_return,
            'actual': actual_return,
            'hit': float(hit),
            'accuracy': np.clip(accuracy, 0.0, 1.0),
            'timestamp': pd.Timestamp.now()
        }
        
        self.history.append(performance_record)
        
        # Keep only recent history for adaptive behavior
        if len(self.history) > self.confidence_window:
            self.history = self.history[-self.confidence_window:]
    
    def generate_signal(self, data: pd.DataFrame) -> SignalOutput:
        """
        Main entry point: generate complete structured signal output.
        This is the primary interface used by downstream modules.
        """
        try:
            # Validate input data
            if data is None or len(data) == 0:
                return SignalOutput(
                    signal_value=0.0,
                    confidence=0.0,
                    probability=0.5,
                    variance=0.01,
                    explanation="No data available",
                    metadata={'error': 'empty_data', 'indicator': self.name}
                )
            
            # Calculate raw signal and metadata
            raw_signal, metadata = self.calculate_raw_signal(data)
            
            # Estimate current volatility for normalization and confidence
            volatility = self.estimate_variance(data) ** 0.5
            
            # Normalize signal
            normalized_signal = self.normalize_signal(raw_signal, volatility)
            
            # Calculate confidence
            confidence = self.calculate_confidence(data, abs(normalized_signal), volatility)
            
            # Convert to probability
            probability = self.signal_to_probability(normalized_signal, confidence)
            
            # Estimate variance
            variance = self.estimate_variance(data)
            
            # Generate explanation
            direction = "bullish" if normalized_signal > 0 else "bearish" if normalized_signal < 0 else "neutral"
            strength = "strong" if abs(normalized_signal) > 0.6 else "moderate" if abs(normalized_signal) > 0.3 else "weak"
            conf_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
            
            explanation = f"{strength.title()} {direction} signal from {self.name} with {conf_level} confidence"
            
            # Add diagnostic metadata
            metadata.update({
                'indicator': self.name,
                'raw_signal': raw_signal,
                'volatility': volatility,
                'data_points': len(data),
                'history_count': len(self.history)
            })
            
            return SignalOutput(
                signal_value=normalized_signal,
                confidence=confidence,
                probability=probability,
                variance=variance,
                explanation=explanation,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return SignalOutput(
                signal_value=0.0,
                confidence=0.0,
                probability=0.5,
                variance=0.01,
                explanation=f"Error in {self.name}: {str(e)}",
                metadata={'error': str(e), 'indicator': self.name}
            )


class TrendIndicators:
    """Collection of trend-following indicators"""
    
    class SMACrossover(BaseIndicator):
        """Simple Moving Average Crossover Signal"""
        
        def __init__(self, fast_period: int = 10, slow_period: int = 20, **kwargs):
            super().__init__(**kwargs)
            self.fast_period = fast_period
            self.slow_period = slow_period
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < self.slow_period:
                return 0.0, {'insufficient_data': True}
            
            fast_sma = data['close'].rolling(self.fast_period).mean()
            slow_sma = data['close'].rolling(self.slow_period).mean()
            
            # Signal based on crossover and divergence
            current_fast = fast_sma.iloc[-1]
            current_slow = slow_sma.iloc[-1]
            prev_fast = fast_sma.iloc[-2] if len(fast_sma) > 1 else current_fast
            prev_slow = slow_sma.iloc[-2] if len(slow_sma) > 1 else current_slow
            
            # Crossover detection
            crossover = 0.0
            if prev_fast <= prev_slow and current_fast > current_slow:
                crossover = 1.0  # Bullish crossover
            elif prev_fast >= prev_slow and current_fast < current_slow:
                crossover = -1.0  # Bearish crossover
            
            # Divergence strength
            divergence = (current_fast - current_slow) / current_slow if current_slow != 0 else 0
            
            # Combine crossover and divergence
            raw_signal = crossover * 0.7 + np.tanh(divergence * 10) * 0.3
            
            metadata = {
                'fast_sma': current_fast,
                'slow_sma': current_slow,
                'divergence': divergence,
                'crossover': crossover
            }
            
            return raw_signal, metadata
    
    class EMAMomentum(BaseIndicator):
        """Exponential Moving Average Momentum Signal"""
        
        def __init__(self, period: int = 12, **kwargs):
            super().__init__(**kwargs)
            self.period = period
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < self.period:
                return 0.0, {'insufficient_data': True}
            
            ema = data['close'].ewm(span=self.period).mean()
            
            # Momentum based on EMA slope and price distance
            if len(ema) < 3:
                return 0.0, {'insufficient_slope_data': True}
            
            # EMA slope (rate of change)
            ema_slope = (ema.iloc[-1] - ema.iloc[-3]) / ema.iloc[-3]
            
            # Price position relative to EMA
            price_position = (data['close'].iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
            
            # Combine slope and position
            raw_signal = np.tanh(ema_slope * 20) * 0.6 + np.tanh(price_position * 5) * 0.4
            
            metadata = {
                'ema': ema.iloc[-1],
                'ema_slope': ema_slope,
                'price_position': price_position,
                'current_price': data['close'].iloc[-1]
            }
            
            return raw_signal, metadata


class MomentumIndicators:
    """Collection of momentum-based indicators"""
    
    class RSI(BaseIndicator):
        """Relative Strength Index with overbought/oversold signals"""
        
        def __init__(self, period: int = 14, **kwargs):
            super().__init__(**kwargs)
            self.period = period
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < self.period + 1:
                return 0.0, {'insufficient_data': True}
            
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # RSI signal generation
            if current_rsi > 70:
                raw_signal = -(current_rsi - 70) / 30  # Bearish when overbought
            elif current_rsi < 30:
                raw_signal = (30 - current_rsi) / 30   # Bullish when oversold
            else:
                # Neutral zone - slight trend following
                raw_signal = (current_rsi - 50) / 50 * 0.3
            
            metadata = {
                'rsi': current_rsi,
                'gain': gain.iloc[-1],
                'loss': loss.iloc[-1],
                'overbought': current_rsi > 70,
                'oversold': current_rsi < 30
            }
            
            return raw_signal, metadata
    
    class MACD(BaseIndicator):
        """MACD Signal and Histogram"""
        
        def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs):
            super().__init__(**kwargs)
            self.fast = fast
            self.slow = slow
            self.signal = signal
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < self.slow + self.signal:
                return 0.0, {'insufficient_data': True}
            
            exp1 = data['close'].ewm(span=self.fast).mean()
            exp2 = data['close'].ewm(span=self.slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=self.signal).mean()
            histogram = macd_line - signal_line
            
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2] if len(histogram) > 1 else current_hist
            
            # Signal based on histogram and crossovers
            hist_momentum = current_hist - prev_hist
            
            # Normalize histogram value
            hist_std = histogram.rolling(50).std().iloc[-1]
            if hist_std > 0:
                normalized_hist = current_hist / hist_std
            else:
                normalized_hist = 0
            
            raw_signal = np.tanh(normalized_hist) * 0.7 + np.tanh(hist_momentum * 100) * 0.3
            
            metadata = {
                'macd': macd_line.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': current_hist,
                'histogram_momentum': hist_momentum
            }
            
            return raw_signal, metadata


class VolumeIndicators:
    """Collection of volume-based indicators"""
    
    class VolumeWeightedSignal(BaseIndicator):
        """Volume-weighted price momentum"""
        
        def __init__(self, period: int = 20, **kwargs):
            super().__init__(**kwargs)
            self.period = period
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < self.period or 'volume' not in data.columns:
                return 0.0, {'insufficient_data': True}
            
            # Volume-weighted average price
            vwap = (data['close'] * data['volume']).rolling(self.period).sum() / \
                   data['volume'].rolling(self.period).sum()
            
            # Price position relative to VWAP
            price_position = (data['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
            
            # Volume momentum
            vol_avg = data['volume'].rolling(self.period).mean()
            vol_ratio = data['volume'].iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
            
            # Combine price position with volume confirmation
            raw_signal = np.tanh(price_position * 5) * min(1.5, vol_ratio) * 0.8
            
            metadata = {
                'vwap': vwap.iloc[-1],
                'price_position': price_position,
                'volume_ratio': vol_ratio,
                'current_volume': data['volume'].iloc[-1],
                'avg_volume': vol_avg.iloc[-1]
            }
            
            return raw_signal, metadata
    
    class OBV(BaseIndicator):
        """On-Balance Volume momentum"""
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < 20 or 'volume' not in data.columns:
                return 0.0, {'insufficient_data': True}
            
            # Calculate OBV
            price_change = data['close'].diff()
            obv = (np.sign(price_change) * data['volume']).cumsum()
            
            # OBV momentum (slope)
            if len(obv) < 5:
                return 0.0, {'insufficient_obv_data': True}
            
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / 5
            obv_std = obv.rolling(50).std().iloc[-1]
            
            if obv_std > 0:
                normalized_slope = obv_slope / obv_std
            else:
                normalized_slope = 0
            
            raw_signal = np.tanh(normalized_slope * 0.1)
            
            metadata = {
                'obv': obv.iloc[-1],
                'obv_slope': obv_slope,
                'obv_normalized': normalized_slope
            }
            
            return raw_signal, metadata


class PriceActionIndicators:
    """Collection of candlestick and price action indicators"""
    
    class CandlestickPatterns(BaseIndicator):
        """Basic candlestick pattern recognition"""
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            required_cols = ['open', 'high', 'low', 'close']
            if len(data) < 3 or not all(col in data.columns for col in required_cols):
                return 0.0, {'insufficient_data': True}
            
            current = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else current
            
            signals = []
            patterns = {}
            
            # Doji
            body_size = abs(current['close'] - current['open'])
            candle_range = current['high'] - current['low']
            if candle_range > 0 and body_size / candle_range < 0.1:
                patterns['doji'] = True
                signals.append(0.0)  # Neutral/reversal
            
            # Hammer/Hanging Man
            lower_shadow = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            
            if candle_range > 0 and lower_shadow > 2 * body_size and upper_shadow < body_size:
                if current['close'] > prev['close']:  # Hammer (bullish)
                    patterns['hammer'] = True
                    signals.append(0.6)
                else:  # Hanging man (bearish)
                    patterns['hanging_man'] = True
                    signals.append(-0.6)
            
            # Engulfing patterns
            if len(data) >= 2:
                prev_body = abs(prev['close'] - prev['open'])
                curr_body = abs(current['close'] - current['open'])
                
                if (prev['close'] < prev['open'] and current['close'] > current['open'] and
                    current['open'] < prev['close'] and current['close'] > prev['open']):
                    patterns['bullish_engulfing'] = True
                    signals.append(0.8)
                elif (prev['close'] > prev['open'] and current['close'] < current['open'] and
                      current['open'] > prev['close'] and current['close'] < prev['open']):
                    patterns['bearish_engulfing'] = True
                    signals.append(-0.8)
            
            # Combine signals
            raw_signal = np.mean(signals) if signals else 0.0
            
            metadata = {
                'patterns': patterns,
                'body_size': body_size,
                'candle_range': candle_range,
                'lower_shadow': lower_shadow,
                'upper_shadow': upper_shadow
            }
            
            return raw_signal, metadata
    
    class SupportResistance(BaseIndicator):
        """Support/Resistance level analysis"""
        
        def __init__(self, lookback: int = 50, **kwargs):
            super().__init__(**kwargs)
            self.lookback = lookback
        
        def calculate_raw_signal(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            if len(data) < self.lookback:
                return 0.0, {'insufficient_data': True}
            
            recent_data = data.tail(self.lookback)
            current_price = data['close'].iloc[-1]
            
            # Find recent highs and lows
            highs = recent_data['high'].rolling(5, center=True).max()
            lows = recent_data['low'].rolling(5, center=True).min()
            
            # Identify support and resistance levels
            resistance_levels = highs[highs == recent_data['high']].dropna().values
            support_levels = lows[lows == recent_data['low']].dropna().values
            
            # Calculate distance to nearest levels
            if len(resistance_levels) > 0:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                resistance_distance = (nearest_resistance - current_price) / current_price
            else:
                resistance_distance = float('inf')
            
            if len(support_levels) > 0:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                support_distance = (current_price - nearest_support) / current_price
            else:
                support_distance = float('inf')
            
            # Generate signal based on proximity to levels
            if abs(resistance_distance) < 0.02:  # Within 2% of resistance
                raw_signal = -0.5  # Bearish near resistance
            elif abs(support_distance) < 0.02:  # Within 2% of support
                raw_signal = 0.5   # Bullish near support
            else:
                raw_signal = 0.0   # No significant level nearby
            
            metadata = {
                'current_price': current_price,
                'nearest_resistance': nearest_resistance if len(resistance_levels) > 0 else None,
                'nearest_support': nearest_support if len(support_levels) > 0 else None,
                'resistance_distance': resistance_distance,
                'support_distance': support_distance,
                'num_resistance_levels': len(resistance_levels),
                'num_support_levels': len(support_levels)
            }
            
            return raw_signal, metadata


class IndicatorBlock:
    """
    Main IndicatorBlock class that orchestrates all indicators.
    This is the primary interface for Phase 1 of the system.
    """
    
    def __init__(self):
        self.indicators = {
            # Trend indicators
            'sma_crossover': TrendIndicators.SMACrossover(),
            'ema_momentum': TrendIndicators.EMAMomentum(),
            
            # Momentum indicators
            'rsi': MomentumIndicators.RSI(),
            'macd': MomentumIndicators.MACD(),
            
            # Volume indicators
            'volume_weighted': VolumeIndicators.VolumeWeightedSignal(),
            'obv': VolumeIndicators.OBV(),
            
            # Price action indicators
            'candlestick_patterns': PriceActionIndicators.CandlestickPatterns(),
            'support_resistance': PriceActionIndicators.SupportResistance()
        }
        
        self.results_history: List[Dict[str, Any]] = []
    
    def process_all_indicators(self, data: pd.DataFrame) -> Dict[str, SignalOutput]:
        """
        Process all indicators and return structured outputs.
        This is the main entry point for the IndicatorBlock system.
        """
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                signal_output = indicator.generate_signal(data)
                results[name] = signal_output
                
                # logger.info(f"{name}: Signal={signal_output.signal_value:.3f}, "
                #            f"Confidence={signal_output.confidence:.3f}, "
                #            f"P(up)={signal_output.probability:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                results[name] = SignalOutput(
                    signal_value=0.0,
                    confidence=0.0,
                    probability=0.5,
                    variance=0.01,
                    explanation=f"Error in {name}: {str(e)}",
                    metadata={'error': str(e)}
                )
        
        # Store results for analysis
        self.results_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': {k: v.to_dict() for k, v in results.items()}
        })
        
        return results
    
    def get_indicator_summary(self, results: Dict[str, SignalOutput]) -> Dict[str, Any]:
        """
        Generate summary statistics across all indicators.
        Useful for downstream group aggregation.
        """
        if not results:
            return {'error': 'no_results'}
        
        signals = [r.signal_value for r in results.values()]
        confidences = [r.confidence for r in results.values()]
        probabilities = [r.probability for r in results.values()]
        
        summary = {
            'signal_mean': np.mean(signals),
            'signal_std': np.std(signals),
            'signal_median': np.median(signals),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'probability_mean': np.mean(probabilities),
            'active_indicators': len([s for s in signals if abs(s) > 0.1]),
            'bullish_count': len([s for s in signals if s > 0.1]),
            'bearish_count': len([s for s in signals if s < -0.1]),
            'high_confidence_count': len([c for c in confidences if c > 0.6]),
            'consensus_strength': 1.0 - np.std(signals)  # Low std = high consensus
        }
        
        return summary
    
    def update_all_performance(self, predicted_returns: Dict[str, float], 
                              actual_return: float):
        """
        Update performance history for all indicators.
        This enables adaptive confidence calibration.
        """
        for name, indicator in self.indicators.items():
            if name in predicted_returns:
                indicator.update_performance(predicted_returns[name], actual_return)
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic report.
        Useful for system monitoring and debugging.
        """
        report = {
            'indicator_count': len(self.indicators),
            'total_history_records': len(self.results_history),
            'indicators': {}
        }
        
        for name, indicator in self.indicators.items():
            report['indicators'][name] = {
                'performance_records': len(indicator.history),
                'recent_accuracy': np.mean([h.get('accuracy', 0.5) 
                                          for h in indicator.history[-10:]]) if indicator.history else 0.5,
                'recent_hit_rate': np.mean([h.get('hit', 0.5) 
                                          for h in indicator.history[-10:]]) if indicator.history else 0.5
            }
        
        return report


# # Example usage and testing
# if __name__ == "__main__":
#     # Create sample OHLCV data for testing
#     np.random.seed(42)
#     dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
#     # Generate realistic price series with trend and noise
#     price_base = 100
#     trend = np.cumsum(np.random.normal(0.001, 0.02, 100))  # Random walk with slight upward bias
#     prices = price_base * np.exp(trend)
    
#     # Add intraday noise for OHLC
#     noise = np.random.normal(0, 0.01, 100)
    
#     sample_data = pd.DataFrame({
#         'date': dates,
#         'open': prices * (1 + noise * 0.5),
#         'high': prices * (1 + np.abs(noise)),
#         'low': prices * (1 - np.abs(noise)),
#         'close': prices,
#         'volume': np.random.randint(1000000, 5000000, 100)
#     })
    
#     # Ensure OHLC relationships are valid
#     sample_data['high'] = np.maximum.reduce([
#         sample_data['open'], sample_data['close'], sample_data['high']
#     ])
#     sample_data['low'] = np.minimum.reduce([
#         sample_data['open'], sample_data['close'], sample_data['low']
#     ])
    
#     print("=== Phase 1: IndicatorBlock System Test ===\n")
    
#     # Initialize the IndicatorBlock system
#     indicator_block = IndicatorBlock()
    
#     # Process all indicators
#     print("Processing indicators...")
#     results = indicator_block.process_all_indicators(sample_data)
    
#     print(f"\n=== Results for {len(results)} Indicators ===")
    
#     # Display individual indicator results
#     for name, signal_output in results.items():
#         print(f"\n{name.upper().replace('_', ' ')}:")
#         print(f"  Signal Value: {signal_output.signal_value:+.3f}")
#         print(f"  Confidence:   {signal_output.confidence:.3f}")
#         print(f"  P(up):        {signal_output.probability:.3f}")
#         print(f"  Variance:     {signal_output.variance:.6f}")
#         print(f"  Explanation:  {signal_output.explanation}")
        
#         # Show key metadata
#         if signal_output.metadata:
#             key_metadata = {k: v for k, v in signal_output.metadata.items() 
#                           if k not in ['error', 'indicator'] and isinstance(v, (int, float))}
#             if key_metadata:
#                 print(f"  Key Metrics:  {key_metadata}")
    
#     # Generate summary
#     print("\n=== Indicator Summary ===")
#     summary = indicator_block.get_indicator_summary(results)
    
#     for key, value in summary.items():
#         if isinstance(value, float):
#             print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
#         else:
#             print(f"  {key.replace('_', ' ').title()}: {value}")
    
#     # Generate diagnostic report
#     print("\n=== Diagnostic Report ===")
#     diagnostics = indicator_block.get_diagnostic_report()
    
#     print(f"Total Indicators: {diagnostics['indicator_count']}")
#     print(f"History Records: {diagnostics['total_history_records']}")
    
#     print("\nIndicator Performance:")
#     for name, stats in diagnostics['indicators'].items():
#         print(f"  {name}: Records={stats['performance_records']}, "
#               f"Accuracy={stats['recent_accuracy']:.3f}, "
#               f"Hit Rate={stats['recent_hit_rate']:.3f}")
    
#     # Test performance update functionality
#     print("\n=== Testing Performance Updates ===")
    
#     # Simulate some predictions and outcomes
#     predicted_returns = {name: result.signal_value * 0.01 
#                         for name, result in results.items()}
    
#     # Simulate actual market movement (could be positive or negative)
#     actual_return = np.random.normal(0.005, 0.015)  # 0.5% return with 1.5% volatility
    
#     print(f"Simulated actual return: {actual_return:+.3f}")
#     print("Updating indicator performance...")
    
#     indicator_block.update_all_performance(predicted_returns, actual_return)
    
#     # Show updated diagnostics
#     updated_diagnostics = indicator_block.get_diagnostic_report()
#     print("\nUpdated Performance:")
#     for name, stats in updated_diagnostics['indicators'].items():
#         print(f"  {name}: Records={stats['performance_records']}, "
#               f"Accuracy={stats['recent_accuracy']:.3f}, "
#               f"Hit Rate={stats['recent_hit_rate']:.3f}")
    
#     # Test edge cases
#     print("\n=== Testing Edge Cases ===")
    
#     # Test with insufficient data
#     small_data = sample_data.head(5)
#     print("Testing with insufficient data (5 rows)...")
#     edge_results = indicator_block.process_all_indicators(small_data)
    
#     insufficient_data_count = sum(1 for r in edge_results.values() 
#                                  if 'insufficient_data' in r.explanation.lower())
#     print(f"Indicators with insufficient data: {insufficient_data_count}/{len(edge_results)}")
    
#     # Test with empty data
#     print("Testing with empty data...")
#     empty_results = indicator_block.process_all_indicators(pd.DataFrame())
#     error_count = sum(1 for r in empty_results.values() if r.signal_value == 0.0)
#     print(f"Indicators returning zero signal: {error_count}/{len(empty_results)}")
    
#     # Test signal validation
#     print("\n=== Signal Validation Test ===")
#     validation_passed = True
    
#     for name, result in results.items():
#         # Check bounds
#         if not (-1 <= result.signal_value <= 1):
#             print(f"ERROR: {name} signal {result.signal_value} outside [-1, 1]")
#             validation_passed = False
        
#         if not (0 <= result.confidence <= 1):
#             print(f"ERROR: {name} confidence {result.confidence} outside [0, 1]")
#             validation_passed = False
        
#         if not (0 <= result.probability <= 1):
#             print(f"ERROR: {name} probability {result.probability} outside [0, 1]")
#             validation_passed = False
        
#         if result.variance < 0:
#             print(f"ERROR: {name} variance {result.variance} is negative")
#             validation_passed = False
    
#     if validation_passed:
#         print("✅ All signals passed validation checks")
#     else:
#         print("❌ Some signals failed validation")
    
#     # Export sample results for downstream testing
#     print("\n=== Exporting Sample Results ===")
    
#     export_data = {
#         'timestamp': pd.Timestamp.now().isoformat(),
#         'input_data_shape': sample_data.shape,
#         'results': {name: result.to_dict() for name, result in results.items()},
#         'summary': summary,
#         'diagnostics': diagnostics
#     }
    
#     print("Sample export structure:")
#     print(f"  - Timestamp: {export_data['timestamp']}")
#     print(f"  - Input shape: {export_data['input_data_shape']}")
#     print(f"  - Results count: {len(export_data['results'])}")
#     print(f"  - Summary keys: {list(export_data['summary'].keys())}")
    
#     print("\n🎯 Phase 1 Complete: IndicatorBlock System Successfully Built")
#     print("   - {len(indicator_block.indicators)} indicators implemented")
#     print("   - Probabilistic outputs with confidence scoring")
#     print("   - Structured SignalOutput format")
#     print("   - Performance tracking and adaptation")
#     print("   - Comprehensive error handling")
#     print("   - Validation and diagnostics")
#     print("\n📋 Ready for Phase 2: GroupBlock aggregation system")
#     print("   - Indicators can be grouped by type (trend, momentum, volume, price action)")
#     print("   - Each indicator provides structured output for easy aggregation")
#     print("   - Confidence and performance history available for weighting")
    
#     # Performance benchmark
#     import time
#     print("\n⚡ Performance Benchmark")
    
#     start_time = time.time()
#     for _ in range(10):
#         benchmark_results = indicator_block.process_all_indicators(sample_data)
#     end_time = time.time()
    
#     avg_time = (end_time - start_time) / 10
#     print(f"   - Average processing time: {avg_time:.4f} seconds")
#     print(f"   - Indicators per second: {len(indicator_block.indicators) / avg_time:.1f}")
#     print(f"   - Suitable for real-time applications: {'✅' if avg_time < 0.1 else '⚠️'}")
    
#     print("\n" + "="*60)
#     print("PHASE 1 INDICATOR SYSTEM: FULLY OPERATIONAL")
#     print("="*60)