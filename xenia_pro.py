"""
🧠 Modular AI-Powered Trading System - Lego-Style Architecture
================================================================

A complete, modular trading framework contained in a single file.
Each component is designed as a "Lego brick" that can be easily swapped or extended.

Author: VOID Inc
Date: 2025-06-26
"""

# ============================================================================
# IMPORTS - Keep all your existing imports here
# ============================================================================
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import warnings
import sys


# ML imports (add your existing ML imports here)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from collections import Counter

from datetime import datetime
import json
import asyncio

from datastream.yfinance_ohlcv import YFinanceOHLCVFetcher as Fetcher

warnings.filterwarnings("ignore")

# Technical analysis - Complete TA-Lib implementation
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using custom implementations.")

# If TA-Lib is not available, we'll use custom implementations


# ============================================================================
# CUSTOM TA-LIB IMPLEMENTATIONS - Lego Brick #1.5
# ============================================================================

import numpy as np
from typing import Union, Literal, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AlignmentMethod(Enum):
    """Enumeration of alignment methods for signals of different lengths."""
    ALIGN_RIGHT = "align_right"
    ALIGN_LEFT = "align_left"
    PAD_LEFT = "pad_left"
    PAD_RIGHT = "pad_right"
    INTERPOLATE = "interpolate"
    LATEST_ONLY = "latest_only"


class CombinationMethod(Enum):
    """Enumeration of signal combination methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    PROBABILISTIC = "probabilistic"
    HIERARCHICAL = "hierarchical"
    SCORE_BASED = "score_based"


@dataclass
class SignalCombinerConfig:
    """Configuration for SignalCombiner."""
    model_weight: float = 0.6
    technical_weight: float = 0.4
    alignment_method: AlignmentMethod = AlignmentMethod.ALIGN_RIGHT
    combination_method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE
    threshold: float = 0.3
    pad_value: Union[int, float] = 0
    interpolation_method: str = 'nearest'
    random_state: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if abs(self.model_weight + self.technical_weight - 1.0) > 1e-6:
            raise ValueError("model_weight + technical_weight must equal 1.0")
        if not 0 <= self.model_weight <= 1:
            raise ValueError("model_weight must be between 0 and 1")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")


class SignalCombiner:
    """
    A class for combining model and technical signals with different lengths.
    
    Supports multiple alignment methods and combination strategies while maintaining
    discrete output classes [1, 0, -1].
    """
    
    def __init__(self, config: Optional[SignalCombinerConfig] = None):
        """
        Initialize SignalCombiner.
        
        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or SignalCombinerConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration."""
        if self.config.interpolation_method not in ['nearest', 'linear', 'forward_fill']:
            raise ValueError("interpolation_method must be 'nearest', 'linear', or 'forward_fill'")
    
    def align_signals(
        self, 
        model_signals: List[Union[int, float]], 
        technical_signals: List[Union[int, float]]
    ) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
        """
        Align two signal arrays of different lengths based on configuration.
        
        Args:
            model_signals: Array of model signals
            technical_signals: Array of technical signals
        
        Returns:
            Tuple of aligned signal arrays
        """
        len_model = len(model_signals)
        len_tech = len(technical_signals)
        
        if len_model == len_tech:
            return model_signals, technical_signals
        
        method = self.config.alignment_method
        
        if method == AlignmentMethod.ALIGN_RIGHT:
            min_len = min(len_model, len_tech)
            return model_signals[-min_len:], technical_signals[-min_len:]
        
        elif method == AlignmentMethod.ALIGN_LEFT:
            min_len = min(len_model, len_tech)
            return model_signals[:min_len], technical_signals[:min_len]
        
        elif method == AlignmentMethod.PAD_LEFT:
            return self._pad_signals(model_signals, technical_signals, 'left')
        
        elif method == AlignmentMethod.PAD_RIGHT:
            return self._pad_signals(model_signals, technical_signals, 'right')
        
        elif method == AlignmentMethod.INTERPOLATE:
            return self._interpolate_signals(model_signals, technical_signals)
        
        elif method == AlignmentMethod.LATEST_ONLY:
            # Return single element lists with latest values
            latest_model = [model_signals[-1]] if model_signals else [0]
            latest_tech = [technical_signals[-1]] if technical_signals else [0]
            return latest_model, latest_tech
        
        else:
            raise ValueError(f"Unknown alignment method: {method}")
    
    def _pad_signals(
        self, 
        model_signals: List[Union[int, float]], 
        technical_signals: List[Union[int, float]], 
        side: str
    ) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
        """Pad shorter signal array to match longer one."""
        len_model = len(model_signals)
        len_tech = len(technical_signals)
        
        if len_model > len_tech:
            pad_length = len_model - len_tech
            if side == 'left':
                padded_tech = [self.config.pad_value] * pad_length + technical_signals
            else:
                padded_tech = technical_signals + [self.config.pad_value] * pad_length
            return model_signals, padded_tech
        
        else:
            pad_length = len_tech - len_model
            if side == 'left':
                padded_model = [self.config.pad_value] * pad_length + model_signals
            else:
                padded_model = model_signals + [self.config.pad_value] * pad_length
            return padded_model, technical_signals
    
    def _interpolate_signals(
        self, 
        model_signals: List[Union[int, float]], 
        technical_signals: List[Union[int, float]]
    ) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
        """Interpolate shorter signal array to match longer one."""
        len_model = len(model_signals)
        len_tech = len(technical_signals)
        
        if len_model < len_tech:
            interpolated_model = self._interpolate_array(model_signals, len_tech)
            return interpolated_model, technical_signals
        elif len_tech < len_model:
            interpolated_tech = self._interpolate_array(technical_signals, len_model)
            return model_signals, interpolated_tech
        else:
            return model_signals, technical_signals
    
    def _interpolate_array(
        self, 
        signals: List[Union[int, float]], 
        target_length: int
    ) -> List[Union[int, float]]:
        """Interpolate array to target length."""
        if len(signals) == target_length:
            return signals
        
        method = self.config.interpolation_method
        
        if method == 'nearest':
            indices = np.linspace(0, len(signals) - 1, target_length)
            return [signals[int(round(i))] for i in indices]
        
        elif method == 'forward_fill':
            step = len(signals) / target_length
            result = []
            for i in range(target_length):
                idx = min(int(i * step), len(signals) - 1)
                result.append(signals[idx])
            return result
        
        elif method == 'linear':
            indices = np.linspace(0, len(signals) - 1, target_length)
            interpolated = np.interp(indices, range(len(signals)), signals)
            return [int(round(x)) if x != 0 else 0 for x in interpolated]
        
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def _combine_single_signals(
        self, 
        model_signal: Union[int, float], 
        technical_signal: Union[int, float]
    ) -> Literal[1, 0, -1]:
        """Combine two individual signals based on combination method."""
        method = self.config.combination_method
        
        if method == CombinationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_combine(model_signal, technical_signal)
        
        elif method == CombinationMethod.PROBABILISTIC:
            return self._probabilistic_combine(model_signal, technical_signal)
        
        elif method == CombinationMethod.HIERARCHICAL:
            return self._hierarchical_combine(model_signal, technical_signal)
        
        elif method == CombinationMethod.SCORE_BASED:
            return self._score_based_combine(model_signal, technical_signal)
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def _weighted_average_combine(
        self, 
        model_signal: Union[int, float], 
        technical_signal: Union[int, float]
    ) -> Literal[1, 0, -1]:
        """Combine using weighted average with thresholds."""
        combined = (self.config.model_weight * model_signal + 
                   self.config.technical_weight * technical_signal)
        
        if combined > self.config.threshold:
            return 1
        elif combined < -self.config.threshold:
            return -1
        else:
            return 0
    
    def _probabilistic_combine(
        self, 
        model_signal: Union[int, float], 
        technical_signal: Union[int, float]
    ) -> Literal[1, 0, -1]:
        """Combine using probabilistic voting."""
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
        
        if model_signal == technical_signal:
            return model_signal
        
        if np.random.random() < self.config.model_weight:
            return model_signal
        else:
            return technical_signal
    
    def _hierarchical_combine(
        self, 
        model_signal: Union[int, float], 
        technical_signal: Union[int, float]
    ) -> Literal[1, 0, -1]:
        """Combine using hierarchical decision making."""
        if model_signal == 0:
            return technical_signal
        
        if model_signal * technical_signal < 0:  # Opposite signs
            if self.config.model_weight > 0.7:
                return model_signal
            else:
                return 0
        
        return model_signal
    
    def _score_based_combine(
        self, 
        model_signal: Union[int, float], 
        technical_signal: Union[int, float]
    ) -> Literal[1, 0, -1]:
        """Combine using scoring system."""
        buy_score = 0
        sell_score = 0
        
        if model_signal == 1:
            buy_score += self.config.model_weight
        elif model_signal == -1:
            sell_score += self.config.model_weight
        
        if technical_signal == 1:
            buy_score += self.config.technical_weight
        elif technical_signal == -1:
            sell_score += self.config.technical_weight
        
        if buy_score > sell_score and buy_score > self.config.threshold:
            return 1
        elif sell_score > buy_score and sell_score > self.config.threshold:
            return -1
        else:
            return 0
    
    def combine(
        self, 
        model_signals: List[Union[int, float]], 
        technical_signals: List[Union[int, float]]
    ) -> List[Literal[1, 0, -1]]:
        """
        Combine two signal arrays.
        
        Args:
            model_signals: Array of model signals [-1, 0, 1]
            technical_signals: Array of technical signals [-1, 0, 1]
        
        Returns:
            Array of combined signals [-1, 0, 1]
        """

        model_signals = model_signals.tolist() if hasattr(model_signals, 'tolist') else list(model_signals)
        technical_signals = technical_signals.tolist() if hasattr(technical_signals, 'tolist') else list(technical_signals)

        if not model_signals and not technical_signals:
            return []
        
        # Handle empty arrays
        if not model_signals:
            model_signals = [0] * len(technical_signals)
        if not technical_signals:
            technical_signals = [0] * len(model_signals)
        
        # Align signals
        model_aligned, tech_aligned = self.align_signals(model_signals, technical_signals)
        
        # Combine aligned signals
        combined_signals = []
        for model_sig, tech_sig in zip(model_aligned, tech_aligned):
            combined_signal = self._combine_single_signals(model_sig, tech_sig)
            combined_signals.append(combined_signal)
        
        return combined_signals
    
    def combine_latest(
        self, 
        model_signals: List[Union[int, float]], 
        technical_signals: List[Union[int, float]]
    ) -> Literal[1, 0, -1]:
        """
        Combine only the latest values from each signal array.
        
        Args:
            model_signals: Array of model signals [-1, 0, 1]
            technical_signals: Array of technical signals [-1, 0, 1]
        
        Returns:
            Single combined signal [-1, 0, -1]
        """
        latest_model = model_signals[-1] if model_signals else 0
        latest_tech = technical_signals[-1] if technical_signals else 0
        
        return self._combine_single_signals(latest_model, latest_tech)
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        self._validate_config()



class CustomTALib:
    """
    Custom implementation of common TA-Lib functions.
    Provides fallback when TA-Lib is not available.
    """

    @staticmethod
    def SMA(series: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Simple Moving Average"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.SMA(series.values, timeperiod=timeperiod), index=series.index
            )
        return series.rolling(window=timeperiod).mean()

    @staticmethod
    def EMA(series: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Exponential Moving Average"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.EMA(series.values, timeperiod=timeperiod), index=series.index
            )
        return series.ewm(span=timeperiod).mean()

    @staticmethod
    def RSI(series: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.RSI(series.values, timeperiod=timeperiod), index=series.index
            )

        # Custom RSI implementation
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=timeperiod).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def BBANDS(
        series: pd.Series, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                series.values, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn
            )
            return (
                pd.Series(upper, index=series.index),
                pd.Series(middle, index=series.index),
                pd.Series(lower, index=series.index),
            )

        # Custom BB implementation
        sma = CustomTALib.SMA(series, timeperiod)
        std = series.rolling(window=timeperiod).std()
        upper = sma + (std * nbdevup)
        lower = sma - (std * nbdevdn)
        return upper, sma, lower

    @staticmethod
    def MACD(
        series: pd.Series,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        if TALIB_AVAILABLE:
            macd, signal, hist = talib.MACD(
                series.values,
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod,
            )
            return (
                pd.Series(macd, index=series.index),
                pd.Series(signal, index=series.index),
                pd.Series(hist, index=series.index),
            )

        # Custom MACD implementation
        ema_fast = CustomTALib.EMA(series, fastperiod)
        ema_slow = CustomTALib.EMA(series, slowperiod)
        macd = ema_fast - ema_slow
        signal = CustomTALib.EMA(macd, signalperiod)
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def STOCH(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(
                high.values,
                low.values,
                close.values,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period,
            )
            return (
                pd.Series(slowk, index=close.index),
                pd.Series(slowd, index=close.index),
            )

        # Custom Stochastic implementation
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.rolling(window=slowk_period).mean()
        d_percent = k_percent.rolling(window=slowd_period).mean()
        return k_percent, d_percent

    @staticmethod
    def ATR(
        high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
    ) -> pd.Series:
        """Average True Range"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.ATR(high.values, low.values, close.values, timeperiod=timeperiod),
                index=close.index,
            )

        # Custom ATR implementation
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=timeperiod).mean()

    @staticmethod
    def ADX(
        high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
    ) -> pd.Series:
        """Average Directional Index"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.ADX(high.values, low.values, close.values, timeperiod=timeperiod),
                index=close.index,
            )

        # Custom ADX implementation
        atr = CustomTALib.ATR(high, low, close, timeperiod)

        # Calculate directional movement
        high_diff = high.diff()
        low_diff = low.diff()

        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)

        plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

        # Smooth the directional movements
        plus_dm_smooth = plus_dm.rolling(window=timeperiod).mean()
        minus_dm_smooth = minus_dm.rolling(window=timeperiod).mean()

        # Calculate directional indicators
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=timeperiod).mean()

        return adx

    @staticmethod
    def CCI(
        high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
    ) -> pd.Series:
        """Commodity Channel Index"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.CCI(high.values, low.values, close.values, timeperiod=timeperiod),
                index=close.index,
            )

        # Custom CCI implementation
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=timeperiod).mean()
        mean_deviation = typical_price.rolling(window=timeperiod).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci

    @staticmethod
    def WILLR(
        high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
    ) -> pd.Series:
        """Williams %R"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.WILLR(
                    high.values, low.values, close.values, timeperiod=timeperiod
                ),
                index=close.index,
            )

        # Custom Williams %R implementation
        highest_high = high.rolling(window=timeperiod).max()
        lowest_low = low.rolling(window=timeperiod).min()
        willr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return willr

    @staticmethod
    def MOM(series: pd.Series, timeperiod: int = 10) -> pd.Series:
        """Momentum"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.MOM(series.values, timeperiod=timeperiod), index=series.index
            )

        # Custom Momentum implementation
        return series.diff(timeperiod)

    @staticmethod
    def ROC(series: pd.Series, timeperiod: int = 10) -> pd.Series:
        """Rate of Change"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.ROC(series.values, timeperiod=timeperiod), index=series.index
            )

        # Custom ROC implementation
        return ((series / series.shift(timeperiod)) - 1) * 100

    @staticmethod
    def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)

        # Custom OBV implementation
        obv = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    @staticmethod
    def AD(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Accumulation/Distribution Line"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.AD(high.values, low.values, close.values, volume.values),
                index=close.index,
            )

        # Custom A/D Line implementation
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        return ad_line

    @staticmethod
    def AROON(
        high: pd.Series, low: pd.Series, timeperiod: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """Aroon Up and Aroon Down"""
        if TALIB_AVAILABLE:
            aroon_down, aroon_up = talib.AROON(
                high.values, low.values, timeperiod=timeperiod
            )
            return (
                pd.Series(aroon_up, index=high.index),
                pd.Series(aroon_down, index=high.index),
            )

        # Custom Aroon implementation
        def aroon_up_calc(window):
            return ((timeperiod - window.argmax()) / timeperiod) * 100

        def aroon_down_calc(window):
            return ((timeperiod - window.argmin()) / timeperiod) * 100

        aroon_up = high.rolling(window=timeperiod + 1).apply(aroon_up_calc, raw=True)
        aroon_down = low.rolling(window=timeperiod + 1).apply(aroon_down_calc, raw=True)

        return aroon_up, aroon_down

    @staticmethod
    def AROONOSC(high: pd.Series, low: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Aroon Oscillator"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.AROONOSC(high.values, low.values, timeperiod=timeperiod),
                index=high.index,
            )

        # Custom Aroon Oscillator implementation
        aroon_up, aroon_down = CustomTALib.AROON(high, low, timeperiod)
        return aroon_up - aroon_down

    @staticmethod
    def TRIX(series: pd.Series, timeperiod: int = 14) -> pd.Series:
        """TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.TRIX(series.values, timeperiod=timeperiod), index=series.index
            )

        # Custom TRIX implementation
        ema1 = CustomTALib.EMA(series, timeperiod)
        ema2 = CustomTALib.EMA(ema1, timeperiod)
        ema3 = CustomTALib.EMA(ema2, timeperiod)
        trix = ema3.pct_change() * 10000  # Convert to basis points
        return trix

    @staticmethod
    def DX(
        high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
    ) -> pd.Series:
        """Directional Movement Index"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.DX(high.values, low.values, close.values, timeperiod=timeperiod),
                index=close.index,
            )

        # Custom DX implementation (part of ADX calculation)
        atr = CustomTALib.ATR(high, low, close, timeperiod)

        high_diff = high.diff()
        low_diff = low.diff()

        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)

        plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

        plus_dm_smooth = plus_dm.rolling(window=timeperiod).mean()
        minus_dm_smooth = minus_dm.rolling(window=timeperiod).mean()

        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx


# Create a unified interface that uses TA-Lib if available, otherwise custom implementations
class TALibInterface:
    """Unified interface for technical analysis functions."""

    @staticmethod
    def get_available_functions():
        """Get list of available TA functions."""
        return [
            "SMA",
            "EMA",
            "RSI",
            "BBANDS",
            "MACD",
            "STOCH",
            "ATR",
            "ADX",
            "CCI",
            "WILLR",
            "MOM",
            "ROC",
            "OBV",
            "AD",
            "AROON",
            "AROONOSC",
            "TRIX",
            "DX",
        ]

    def __getattr__(self, name):
        """Dynamically route to appropriate implementation."""
        if hasattr(CustomTALib, name):
            return getattr(CustomTALib, name)
        else:
            raise AttributeError(f"Function '{name}' not available")


# Create global instance
ta = TALibInterface()


class Signal(Enum):
    """Trading signals with support for long, short, and hold positions."""

    HOLD = 0
    LONG = 1
    SHORT = -1

    def __str__(self):
        return self.name


class ModelType(Enum):
    """Supported ML model types."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    MLP = "mlp"
    CUSTOM = "custom"


class BacktestType(Enum):
    """Supported backtesting methods."""

    CLASSIC_CV = "classic_cv"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"


class SimulationType(Enum):
    """Supported simulation methods."""

    EVENT_BASED = "event_based"
    BAR_BASED = "bar_based"
    HYBRID = "hybrid"


# ============================================================================
# ABSTRACT BASE CLASSES - Lego Brick #3
# ============================================================================
class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction components."""

    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from market data."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        pass


class BaseLabelingStrategy(ABC):
    """Abstract base class for labeling strategies."""

    @abstractmethod
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create trading labels from market data."""
        pass


class BaseModel(ABC):
    """Abstract base class for trading models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        pass


class BaseBacktester(ABC):
    """Abstract base class for backtesting engines."""

    @abstractmethod
    def run_backtest(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        feature_extractor: BaseFeatureExtractor,
        labeling_strategy: BaseLabelingStrategy,
    ) -> Dict[str, Any]:
        """Run backtest and return results."""
        pass

    def _combine_signals(self, model_pred, tech_pred, model_threshold=.6, tech_thereshold=.4):
        signal_combiner = SignalCombiner()

        comb_signals = signal_combiner.combine(model_pred, tech_pred)

        return comb_signals

class BaseSimulator(ABC):
    """Abstract base class for trading simulators."""

    @abstractmethod
    def simulate(self, signals: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate trading based on signals."""
        pass

class BasePortfolioManager(ABC):
    """Abstract base class for portfolio management."""

    @abstractmethod
    def calculate_position_size(
        self, signal: Signal, current_price: float, portfolio_value: float
    ) -> float:
        """Calculate position size based on signal and portfolio value."""
        pass

    @abstractmethod
    def update_portfolio(self, signal: Signal, price: float, quantity: float) -> None:
        """Update portfolio state."""
        pass


# ============================================================================
# FEATURE EXTRACTION COMPONENTS - Lego Brick #4
# ============================================================================
class TechnicalFeatureExtractor(BaseFeatureExtractor):
    """Default technical indicator feature extractor."""

    def __init__(self, lookback_periods: List[int] = [5, 10, 20]):
        self.lookback_periods = lookback_periods
        self.feature_names = []

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicators as features."""
        features = pd.DataFrame(index=data.index)

        # Price-based features
        for period in self.lookback_periods:
            features[f"sma_{period}"] = ta.SMA(data["close"], timeperiod=period)
            features[f"ema_{period}"] = ta.EMA(data["close"], timeperiod=period)
            features[f"rsi_{period}"] = ta.RSI(data["close"], timeperiod=period)
            (
                features[f"bb_upper_{period}"],
                features[f"bb_middle_{period}"],
                features[f"bb_lower_{period}"],
            ) = ta.BBANDS(data["close"], timeperiod=period)

        # Volume-based features
        if "volume" in data.columns:
            features["volume_sma_10"] = ta.SMA(data["volume"], timeperiod=10)
            features["volume_ratio"] = data["volume"] / features["volume_sma_10"]
            features["obv"] = ta.OBV(data["close"], data["volume"])
            features["ad_line"] = ta.AD(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Momentum indicators
        features["macd"], features["macd_signal"], features["macd_hist"] = ta.MACD(
            data["close"]
        )
        features["stoch_k"], features["stoch_d"] = ta.STOCH(
            data["high"], data["low"], data["close"]
        )
        features["mom_10"] = ta.MOM(data["close"], timeperiod=10)
        features["roc_10"] = ta.ROC(data["close"], timeperiod=10)

        # Volatility indicators
        features["atr"] = ta.ATR(data["high"], data["low"], data["close"])

        # Trend indicators
        features["adx"] = ta.ADX(data["high"], data["low"], data["close"])
        features["aroon_up"], features["aroon_down"] = ta.AROON(
            data["high"], data["low"]
        )
        features["aroon_osc"] = ta.AROONOSC(data["high"], data["low"])

        # Oscillators
        features["cci"] = ta.CCI(data["high"], data["low"], data["close"])
        features["willr"] = ta.WILLR(data["high"], data["low"], data["close"])
        features["trix"] = ta.TRIX(data["close"])

        # Price ratios
        features["price_to_sma_20"] = data["close"] / features["sma_20"]
        features["high_low_ratio"] = data["high"] / data["low"]

        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f"close_lag_{lag}"] = data["close"].shift(lag)
            if "volume" in data.columns:
                features[f"volume_lag_{lag}"] = data["volume"].shift(lag)

        # Store feature names
        self.feature_names = features.columns.tolist()

        return features.fillna(method="ffill").fillna(0)

    def get_feature_names(self) -> List[str]:
        return self.feature_names


class PriceActionFeatureExtractor(BaseFeatureExtractor):
    """Price action focused feature extractor."""

    def __init__(self, lookback_periods: List[int] = [5, 10, 20]):
        self.lookback_periods = lookback_periods
        self.feature_names = []

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract price action features."""
        features = pd.DataFrame(index=data.index)

        # Returns
        features["returns"] = data["close"].pct_change()
        features["log_returns"] = np.log(data["close"] / data["close"].shift(1))

        # Rolling statistics
        for period in self.lookback_periods:
            features[f"returns_mean_{period}"] = (
                features["returns"].rolling(period).mean()
            )
            features[f"returns_std_{period}"] = (
                features["returns"].rolling(period).std()
            )
            features[f"returns_skew_{period}"] = (
                features["returns"].rolling(period).skew()
            )
            features[f"returns_kurt_{period}"] = (
                features["returns"].rolling(period).kurt()
            )

        # Price patterns
        features["higher_high"] = (data["high"] > data["high"].shift(1)).astype(int)
        features["lower_low"] = (data["low"] < data["low"].shift(1)).astype(int)
        features["inside_bar"] = (
            (data["high"] < data["high"].shift(1))
            & (data["low"] > data["low"].shift(1))
        ).astype(int)

        self.feature_names = features.columns.tolist()
        return features.fillna(0)

    def get_feature_names(self) -> List[str]:
        return self.feature_names


# ============================================================================
# LABELING STRATEGIES - Lego Brick #5
# ============================================================================
class ReturnBasedLabeling(BaseLabelingStrategy):
    """Label based on future returns."""

    def __init__(self, lookforward_periods: int = 5, threshold: float = 0.01):
        self.lookforward_periods = lookforward_periods
        self.threshold = threshold

    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels based on future returns."""
        future_returns = (
            data["close"].shift(-self.lookforward_periods) / data["close"] - 1
        )

        labels = pd.Series(Signal.HOLD.value, index=data.index)
        labels[future_returns > self.threshold] = Signal.LONG.value
        labels[future_returns < -self.threshold] = Signal.SHORT.value

        return labels


class VolatilityAdjustedLabeling(BaseLabelingStrategy):
    """Label based on volatility-adjusted returns."""

    def __init__(
        self,
        lookforward_periods: int = 5,
        volatility_window: int = 20,
        threshold_multiplier: float = 1.0,
    ):
        self.lookforward_periods = lookforward_periods
        self.volatility_window = volatility_window
        self.threshold_multiplier = threshold_multiplier

    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels based on volatility-adjusted returns."""
        returns = data["close"].pct_change()
        rolling_vol = returns.rolling(self.volatility_window).std()

        future_returns = (
            data["close"].shift(-self.lookforward_periods) / data["close"] - 1
        )
        threshold = rolling_vol * self.threshold_multiplier

        labels = pd.Series(Signal.HOLD.value, index=data.index)
        labels[future_returns > threshold] = Signal.LONG.value
        labels[future_returns < -threshold] = Signal.SHORT.value

        return labels


# ============================================================================
# TRADING MODELS - Lego Brick #6
# ============================================================================
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from hmmlearn import hmm
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class ModelType(Enum):
    """Supported ML model types."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    MLP = "mlp"
    HIDDEN_MARKOV = "hidden_markov"
    CUSTOM = "custom"

class BaseModel(ABC):
    """Abstract base class for trading models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        pass


class HiddenMarkovModel(BaseModel):
    """
    Hidden Markov Model for trading regime detection and signal generation.
    Designed for 15-minute timeframes with industrial-grade features.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        lookback_window: int = 96,
        volatility_window: int = 20,
        regime_threshold: float = 0.6,
        **kwargs,
    ):
        """
        Initialize HMM trading model.

        Args:
            n_components: Number of hidden states (default 3: bull, bear, sideways)
            covariance_type: Type of covariance parameters
            lookback_window: Number of periods to look back (96 = 24 hours for 15m)
            volatility_window: Window for volatility calculation
            regime_threshold: Confidence threshold for regime classification
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.regime_threshold = regime_threshold

        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=kwargs.get("n_iter", 1000),
            tol=kwargs.get("tol", 1e-4),
            random_state=kwargs.get("random_state", 42),
        )

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.regime_labels = {0: "Bearish", 1: "Sideways", 2: "Bullish"}

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for HMM training.
        Optimized for 15-minute trading timeframes.
        """
        features = pd.DataFrame(index=data.index)

        # Price-based features
        if "close" in data.columns:
            # Returns at multiple timeframes
            features["returns_1"] = data["close"].pct_change(1)
            features["returns_4"] = data["close"].pct_change(4)  # 1 hour
            features["returns_16"] = data["close"].pct_change(16)  # 4 hours
            features["returns_96"] = data["close"].pct_change(96)  # 24 hours

            # Log returns for better distribution properties
            features["log_returns"] = np.log(data["close"] / data["close"].shift(1))

            # Volatility features
            features["volatility"] = (
                features["returns_1"].rolling(self.volatility_window).std()
            )
            features["volatility_long"] = (
                features["returns_1"].rolling(self.volatility_window * 2).std()
            )

            # Price momentum
            features["momentum_short"] = (
                data["close"] / data["close"].shift(8) - 1
            )  # 2 hours
            features["momentum_medium"] = (
                data["close"] / data["close"].shift(32) - 1
            )  # 8 hours
            features["momentum_long"] = (
                data["close"] / data["close"].shift(96) - 1
            )  # 24 hours

        # Volume-based features (if available)
        if "volume" in data.columns:
            features["volume_ma"] = data["volume"].rolling(20).mean()
            features["volume_ratio"] = data["volume"] / features["volume_ma"]
            features["volume_volatility"] = np.log(data["volume"]).rolling(20).std()

        # Technical indicators optimized for 15m timeframes
        if all(col in data.columns for col in ["high", "low", "close"]):
            # RSI with shorter periods for 15m
            features["rsi_14"] = self._calculate_rsi(data["close"], 14)
            features["rsi_28"] = self._calculate_rsi(data["close"], 28)

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = data["close"].rolling(bb_period).mean()
            bb_std_dev = data["close"].rolling(bb_period).std()
            features["bb_upper"] = bb_ma + (bb_std_dev * bb_std)
            features["bb_lower"] = bb_ma - (bb_std_dev * bb_std)
            features["bb_position"] = (data["close"] - bb_ma) / (bb_std_dev * bb_std)

            # MACD optimized for 15m
            features["macd"] = self._calculate_macd(data["close"], 12, 26, 9)

            # Average True Range
            features["atr"] = self._calculate_atr(
                data["high"], data["low"], data["close"], 14
            )
            features["atr_ratio"] = features["atr"] / data["close"]

        # Market microstructure features
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            # Intraday patterns
            features["hl_ratio"] = (data["high"] - data["low"]) / data["close"]
            features["oc_ratio"] = (data["close"] - data["open"]) / data["open"]
            features["body_ratio"] = abs(data["close"] - data["open"]) / (
                data["high"] - data["low"]
            )

            # Gap analysis
            features["gap"] = (data["open"] - data["close"].shift(1)) / data[
                "close"
            ].shift(1)

        # Time-based features for 15m patterns
        if isinstance(data.index, pd.DatetimeIndex):
            features["hour"] = data.index.hour
            features["minute"] = data.index.minute
            features["day_of_week"] = data.index.dayofweek
            features["is_market_open"] = (
                (data.index.hour >= 9) & (data.index.hour < 16)
            ).astype(int)

        # Market regime indicators
        features["trend_strength"] = self._calculate_trend_strength(data["close"])
        features["regime_volatility"] = (
            features["volatility"] / features["volatility"].rolling(96).mean()
        )

        # Clean and normalize features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method="ffill").fillna(0)

        self.feature_names = features.columns.tolist()
        return features

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    def _calculate_trend_strength(
        self, prices: pd.Series, period: int = 20
    ) -> pd.Series:
        """Calculate trend strength indicator."""
        ma = prices.rolling(period).mean()
        deviations = abs(prices - ma)
        trend_strength = 1 - (deviations.rolling(period).mean() / ma)
        return trend_strength.fillna(0)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Train the Hidden Markov Model.

        Args:
            X: Input features or raw OHLCV data
            y: Not used for HMM (unsupervised learning)
        """
        # Create features if raw OHLCV data is provided
        if any(col in X.columns for col in ["open", "high", "low", "close", "volume"]):
            features = self._create_features(X)
        else:
            features = X.copy()

        # Remove rows with insufficient data
        features = features.iloc[self.lookback_window :]

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit HMM model
        self.model.fit(features_scaled)

        # Store training data for regime analysis
        self.training_features = features_scaled
        self.hidden_states = self.model.predict(features_scaled)

        # Analyze regime characteristics
        self._analyze_regimes(features_scaled)

        self.is_fitted = True

    def _analyze_regimes(self, features: np.ndarray) -> None:
        """Analyze characteristics of each hidden state/regime."""
        states = self.model.predict(features)

        self.regime_stats = {}
        for state in range(self.n_components):
            state_mask = states == state
            state_features = features[state_mask]

            if len(state_features) > 0:
                self.regime_stats[state] = {
                    "mean_features": np.mean(state_features, axis=0),
                    "std_features": np.std(state_features, axis=0),
                    "frequency": np.mean(state_mask),
                    "avg_duration": self._calculate_avg_duration(states, state),
                }

    def _calculate_avg_duration(self, states: np.ndarray, target_state: int) -> float:
        """Calculate average duration of a specific state."""
        durations = []
        current_duration = 0

        for state in states:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals: 1 (buy), 0 (hold), -1 (sell).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        probabilities = self.predict_proba(X)
        predictions = np.zeros(len(probabilities))

        # Convert regime probabilities to trading signals
        for i, probs in enumerate(probabilities):
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]

            if max_prob > self.regime_threshold:
                if max_prob_idx == 2:  # Bullish regime
                    predictions[i] = 1
                elif max_prob_idx == 0:  # Bearish regime
                    predictions[i] = -1
                else:  # Sideways regime
                    predictions[i] = 0
            else:
                predictions[i] = 0  # Hold when uncertain

        return predictions.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate regime probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create features if raw OHLCV data is provided
        if any(col in X.columns for col in ["open", "high", "low", "close", "volume"]):
            features = self._create_features(X)
        else:
            features = X.copy()

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get state probabilities
        log_probabilities = self.model.predict_proba(features_scaled)

        return log_probabilities

    def get_regime_info(self) -> dict:
        """Get information about the identified market regimes."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing regime info")

        return {
            "n_regimes": self.n_components,
            "regime_labels": self.regime_labels,
            "regime_stats": self.regime_stats,
            "transition_matrix": self.model.transmat_,
            "feature_names": self.feature_names,
        }

    def plot_regimes(
        self, prices: pd.Series, start_date: str = None, end_date: str = None
    ):
        """Plot price data with identified regimes (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if start_date:
                mask = prices.index >= start_date
                if end_date:
                    mask = mask & (prices.index <= end_date)
                prices = prices[mask]

            # Get regime predictions for the price data
            regime_data = pd.DataFrame({"close": prices})
            features = self._create_features(regime_data)
            features_scaled = self.scaler.transform(features)
            regimes = self.model.predict(features_scaled)

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

            # Plot prices with regime coloring
            colors = ["red", "gray", "green"]
            for regime in range(self.n_components):
                mask = regimes == regime
                ax1.scatter(
                    prices.index[mask],
                    prices[mask],
                    c=colors[regime],
                    alpha=0.6,
                    s=1,
                    label=self.regime_labels[regime],
                )

            ax1.set_ylabel("Price")
            ax1.set_title("Price Action with Market Regimes")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot regime transitions
            ax2.plot(prices.index, regimes, linewidth=1)
            ax2.set_ylabel("Regime")
            ax2.set_xlabel("Time")
            ax2.set_title("Market Regime Transitions")
            ax2.set_yticks(range(self.n_components))
            ax2.set_yticklabels(
                [self.regime_labels[i] for i in range(self.n_components)]
            )
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")


class MLModel(BaseModel):
    """Wrapper for various ML models including HMM."""

    def __init__(self, model_type: ModelType = ModelType.RANDOM_FOREST, **kwargs):
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.scaler = (
            StandardScaler() if model_type != ModelType.HIDDEN_MARKOV else None
        )
        self.is_fitted = False

    def _create_model(self, **kwargs):
        """Create the underlying ML model."""
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=kwargs.get("random_state", 42),
            )
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=kwargs.get("random_state", 42),
            )
        elif self.model_type == ModelType.MLP:
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (100, 50)),
                max_iter=kwargs.get("max_iter", 1000),
                random_state=kwargs.get("random_state", 42),
            )
        elif self.model_type == ModelType.HIDDEN_MARKOV:
            return HiddenMarkovModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Train the model."""
        if self.model_type == ModelType.HIDDEN_MARKOV:
            self.model.fit(X, y)
        else:
            if y is None:
                raise ValueError("Supervised models require target variable y")
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model_type == ModelType.HIDDEN_MARKOV:
            return self.model.predict(X)
        else:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model_type == ModelType.HIDDEN_MARKOV:
            return self.model.predict_proba(X)
        else:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)


class SignalType(Enum):
    """Types of technical analysis signals."""
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    OSCILLATOR = "oscillator"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    COMBINED = "combined"

class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

class BaseSignalGenerator(ABC):
    """Base class for signal generators."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Get signal strength for each data point."""
        pass

class TechnicalSignalGenerator(BaseSignalGenerator, BaseModel):
    """Technical analysis signal generator using custom TA indicators."""
    
    def __init__(self, signal_type: SignalType = SignalType.COMBINED, **kwargs):
        self.signal_type = signal_type
        self.params = self._set_default_params(**kwargs)
        self.is_fitted = False
        self.signal_history = None
        
    def _set_default_params(self, **kwargs) -> Dict:
        """Set default parameters for technical indicators."""
        defaults = {
            # Moving averages
            'sma_fast': kwargs.get('sma_fast', 10),
            'sma_slow': kwargs.get('sma_slow', 30),
            'ema_fast': kwargs.get('ema_fast', 12),
            'ema_slow': kwargs.get('ema_slow', 26),
            
            # Oscillators
            'rsi_period': kwargs.get('rsi_period', 14),
            'rsi_overbought': kwargs.get('rsi_overbought', 70),
            'rsi_oversold': kwargs.get('rsi_oversold', 30),
            
            'stoch_k': kwargs.get('stoch_k', 14),
            'stoch_d': kwargs.get('stoch_d', 3),
            'stoch_overbought': kwargs.get('stoch_overbought', 80),
            'stoch_oversold': kwargs.get('stoch_oversold', 20),
            
            'cci_period': kwargs.get('cci_period', 14),
            'cci_overbought': kwargs.get('cci_overbought', 100),
            'cci_oversold': kwargs.get('cci_oversold', -100),
            
            'willr_period': kwargs.get('willr_period', 14),
            'willr_overbought': kwargs.get('willr_overbought', -20),
            'willr_oversold': kwargs.get('willr_oversold', -80),
            
            # MACD
            'macd_fast': kwargs.get('macd_fast', 12),
            'macd_slow': kwargs.get('macd_slow', 26),
            'macd_signal': kwargs.get('macd_signal', 9),
            
            # Bollinger Bands
            'bb_period': kwargs.get('bb_period', 20),
            'bb_std': kwargs.get('bb_std', 2),
            
            # Trend indicators
            'adx_period': kwargs.get('adx_period', 14),
            'adx_trend_threshold': kwargs.get('adx_trend_threshold', 25),
            
            'aroon_period': kwargs.get('aroon_period', 14),
            'aroon_threshold': kwargs.get('aroon_threshold', 70),
            
            # Momentum
            'mom_period': kwargs.get('mom_period', 10),
            'roc_period': kwargs.get('roc_period', 10),
            'trix_period': kwargs.get('trix_period', 14),
            
            # Volume
            'atr_period': kwargs.get('atr_period', 14),
        }
        return defaults
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the signal generator to historical data."""
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Generate all technical indicators
        self.indicators = self._calculate_indicators(data)
        self.is_fitted = True
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators."""
        indicators = {}
        
        # Import the ta interface from your provided code
        # Assuming 'ta' is available from your CustomTALib implementation
        
        high, low, close = data['high'], data['low'], data['close']
        volume = data.get('volume', pd.Series(index=data.index))
        
        # Moving Averages
        indicators['sma_fast'] = ta.SMA(close, self.params['sma_fast'])
        indicators['sma_slow'] = ta.SMA(close, self.params['sma_slow'])
        indicators['ema_fast'] = ta.EMA(close, self.params['ema_fast'])
        indicators['ema_slow'] = ta.EMA(close, self.params['ema_slow'])
        
        # Oscillators
        indicators['rsi'] = ta.RSI(close, self.params['rsi_period'])
        indicators['stoch_k'], indicators['stoch_d'] = ta.STOCH(
            high, low, close, 
            self.params['stoch_k'], 
            self.params['stoch_d'], 
            self.params['stoch_d']
        )
        indicators['cci'] = ta.CCI(high, low, close, self.params['cci_period'])
        indicators['willr'] = ta.WILLR(high, low, close, self.params['willr_period'])
        
        # MACD
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = ta.MACD(
            close, 
            self.params['macd_fast'], 
            self.params['macd_slow'], 
            self.params['macd_signal']
        )
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = ta.BBANDS(
            close, self.params['bb_period'], self.params['bb_std'], self.params['bb_std']
        )
        
        # Trend Indicators
        indicators['adx'] = ta.ADX(high, low, close, self.params['adx_period'])
        indicators['aroon_up'], indicators['aroon_down'] = ta.AROON(
            high, low, self.params['aroon_period']
        )
        indicators['aroon_osc'] = ta.AROONOSC(high, low, self.params['aroon_period'])
        
        # Momentum
        indicators['momentum'] = ta.MOM(close, self.params['mom_period'])
        indicators['roc'] = ta.ROC(close, self.params['roc_period'])
        indicators['trix'] = ta.TRIX(close, self.params['trix_period'])
        
        # Volume (if available)
        if not volume.empty:
            indicators['obv'] = ta.OBV(close, volume)
            indicators['ad'] = ta.AD(high, low, close, volume)
        
        # Volatility
        indicators['atr'] = ta.ATR(high, low, close, self.params['atr_period'])
        
        return indicators
    
    def _generate_trend_signals(self) -> pd.Series:
        """Generate trend-following signals."""
        signals = pd.Series(0, index=self.indicators['sma_fast'].index)
        
        # Moving Average Crossover
        ma_signal = np.where(
            self.indicators['sma_fast'] > self.indicators['sma_slow'], 1, -1
        )
        
        # ADX trend strength
        strong_trend = self.indicators['adx'] > self.params['adx_trend_threshold']
        
        # Aroon signals
        aroon_bullish = self.indicators['aroon_up'] > self.params['aroon_threshold']
        aroon_bearish = self.indicators['aroon_down'] > self.params['aroon_threshold']
        
        # Combine signals
        signals = pd.Series(ma_signal, index=signals.index)
        signals[strong_trend & aroon_bullish] = 1
        signals[strong_trend & aroon_bearish] = -1
        
        return signals
    
    def _generate_momentum_signals(self) -> pd.Series:
        """Generate momentum-based signals."""
        signals = pd.Series(0, index=self.indicators['rsi'].index)
        
        # RSI signals
        rsi_oversold = self.indicators['rsi'] < self.params['rsi_oversold']
        rsi_overbought = self.indicators['rsi'] > self.params['rsi_overbought']
        
        # MACD signals
        macd_bullish = (self.indicators['macd'] > self.indicators['macd_signal']) & \
                      (self.indicators['macd_hist'] > 0)
        macd_bearish = (self.indicators['macd'] < self.indicators['macd_signal']) & \
                      (self.indicators['macd_hist'] < 0)
        
        # Momentum signals
        mom_bullish = self.indicators['momentum'] > 0
        mom_bearish = self.indicators['momentum'] < 0
        
        # ROC signals
        roc_bullish = self.indicators['roc'] > 0
        roc_bearish = self.indicators['roc'] < 0
        
        # Combine signals
        signals[rsi_oversold & macd_bullish & mom_bullish] = 1
        signals[rsi_overbought & macd_bearish & mom_bearish] = -1
        signals[roc_bullish & macd_bullish] = 1
        signals[roc_bearish & macd_bearish] = -1
        
        return signals
    
    def _generate_oscillator_signals(self) -> pd.Series:
        """Generate oscillator-based signals."""
        signals = pd.Series(0, index=self.indicators['rsi'].index)
        
        # Multiple oscillator confirmation
        rsi_oversold = self.indicators['rsi'] < self.params['rsi_oversold']
        rsi_overbought = self.indicators['rsi'] > self.params['rsi_overbought']
        
        stoch_oversold = self.indicators['stoch_k'] < self.params['stoch_oversold']
        stoch_overbought = self.indicators['stoch_k'] > self.params['stoch_overbought']
        
        cci_oversold = self.indicators['cci'] < self.params['cci_oversold']
        cci_overbought = self.indicators['cci'] > self.params['cci_overbought']
        
        willr_oversold = self.indicators['willr'] < self.params['willr_oversold']
        willr_overbought = self.indicators['willr'] > self.params['willr_overbought']
        
        # Require multiple oscillator confirmation
        oversold_count = (rsi_oversold.astype(int) + stoch_oversold.astype(int) + 
                         cci_oversold.astype(int) + willr_oversold.astype(int))
        overbought_count = (rsi_overbought.astype(int) + stoch_overbought.astype(int) + 
                           cci_overbought.astype(int) + willr_overbought.astype(int))
        
        signals[oversold_count >= 2] = 1  # Buy when 2+ oscillators show oversold
        signals[overbought_count >= 2] = -1  # Sell when 2+ oscillators show overbought
        
        return signals
    
    def _generate_volume_signals(self) -> pd.Series:
        """Generate volume-based signals."""
        if 'obv' not in self.indicators:
            return pd.Series(0, index=self.indicators['rsi'].index)
        
        signals = pd.Series(0, index=self.indicators['obv'].index)
        
        # OBV trend
        obv_trend = self.indicators['obv'].diff(5)  # 5-period change
        
        # A/D Line trend
        ad_trend = self.indicators['ad'].diff(5)
        
        # Volume confirmation
        signals[obv_trend > 0] = 1
        signals[obv_trend < 0] = -1
        signals[(obv_trend > 0) & (ad_trend > 0)] = 1
        signals[(obv_trend < 0) & (ad_trend < 0)] = -1
        
        return signals
    
    def _generate_volatility_signals(self) -> pd.Series:
        """Generate volatility-based signals."""
        signals = pd.Series(0, index=self.indicators['bb_upper'].index)
        
        close = self.indicators['bb_middle']  # This is the SMA used for BB
        
        # Bollinger Band signals
        bb_squeeze = (self.indicators['bb_upper'] - self.indicators['bb_lower']) / \
                     self.indicators['bb_middle'] < 0.1  # Narrow bands
        
        # Price touching bands
        touching_lower = close <= self.indicators['bb_lower']
        touching_upper = close >= self.indicators['bb_upper']
        
        # ATR for volatility context
        atr_expanding = self.indicators['atr'] > self.indicators['atr'].rolling(5).mean()
        
        signals[touching_lower & ~bb_squeeze] = 1  # Buy at lower band
        signals[touching_upper & ~bb_squeeze] = -1  # Sell at upper band
        signals[bb_squeeze & atr_expanding] = 0  # No signal during squeeze
        
        return signals
    
    def _generate_combined_signals(self) -> pd.Series:
        """Generate combined signals from multiple strategies."""
        trend_signals = self._generate_trend_signals()
        momentum_signals = self._generate_momentum_signals()
        oscillator_signals = self._generate_oscillator_signals()
        volume_signals = self._generate_volume_signals()
        volatility_signals = self._generate_volatility_signals()
        
        # Combine with weighted voting
        combined = pd.DataFrame({
            'trend': trend_signals,
            'momentum': momentum_signals,
            'oscillator': oscillator_signals,
            'volume': volume_signals,
            'volatility': volatility_signals
        })
        
        # Weight the signals (trend and momentum get higher weight)
        weights = {'trend': 0.3, 'momentum': 0.3, 'oscillator': 0.2, 'volume': 0.1, 'volatility': 0.1}
        
        weighted_sum = sum(combined[col] * weights[col] for col in combined.columns)
        
        # Convert to discrete signals
        signals = pd.Series(0, index=weighted_sum.index)
        signals[weighted_sum > 0.3] = 1
        signals[weighted_sum < -0.3] = -1
        
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on the selected strategy."""
        if not self.is_fitted:
            self.fit(data)
        
        if self.signal_type == SignalType.TREND_FOLLOWING:
            signals = self._generate_trend_signals()
        elif self.signal_type == SignalType.MOMENTUM:
            signals = self._generate_momentum_signals()
        elif self.signal_type == SignalType.OSCILLATOR:
            signals = self._generate_oscillator_signals()
        elif self.signal_type == SignalType.VOLUME:
            signals = self._generate_volume_signals()
        elif self.signal_type == SignalType.VOLATILITY:
            signals = self._generate_volatility_signals()
        elif self.signal_type == SignalType.COMBINED:
            signals = self._generate_combined_signals()
        else:
            raise ValueError(f"Unsupported signal type: {self.signal_type}")
        
        self.signal_history = signals
        return signals
    
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength for each data point."""
        if not self.is_fitted:
            self.fit(data)
        
        # Generate base signals
        signals = self.generate_signals(data)
        
        # Calculate strength based on multiple confirmations
        strength = pd.Series(SignalStrength.NEUTRAL.value, index=signals.index)
        
        # Get individual signal types
        trend_signals = self._generate_trend_signals()
        momentum_signals = self._generate_momentum_signals()
        oscillator_signals = self._generate_oscillator_signals()
        
        # Count confirmations
        bullish_count = (
            (trend_signals == 1).astype(int) +
            (momentum_signals == 1).astype(int) +
            (oscillator_signals == 1).astype(int)
        )
        
        bearish_count = (
            (trend_signals == -1).astype(int) +
            (momentum_signals == -1).astype(int) +
            (oscillator_signals == -1).astype(int)
        )
        
        # Assign strength levels
        strength[bullish_count >= 3] = SignalStrength.STRONG_BUY.value
        strength[bullish_count == 2] = SignalStrength.BUY.value
        strength[bearish_count >= 3] = SignalStrength.STRONG_SELL.value
        strength[bearish_count == 2] = SignalStrength.SELL.value
        
        return strength
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions (signals) for compatibility with ML interface."""
        signals = self.generate_signals(data)
        return signals.values
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities based on signal strength."""
        strength = self.get_signal_strength(data)
        
        # Convert strength to probabilities
        # [prob_sell, prob_neutral, prob_buy]
        proba = np.zeros((len(strength), 3))
        
        for i, s in enumerate(strength):
            if s == SignalStrength.STRONG_SELL.value:
                proba[i] = [0.9, 0.05, 0.05]
            elif s == SignalStrength.SELL.value:
                proba[i] = [0.7, 0.2, 0.1]
            elif s == SignalStrength.NEUTRAL.value:
                proba[i] = [0.2, 0.6, 0.2]
            elif s == SignalStrength.BUY.value:
                proba[i] = [0.1, 0.2, 0.7]
            elif s == SignalStrength.STRONG_BUY.value:
                proba[i] = [0.05, 0.05, 0.9]
        
        return proba
    
    def get_signal_details(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get detailed breakdown of all signals and indicators."""
        if not self.is_fitted:
            self.fit(data)
        
        details = pd.DataFrame(index=data.index)
        
        # Add all indicators
        for name, indicator in self.indicators.items():
            details[f'indicator_{name}'] = indicator
        
        # Add individual signal types
        details['signal_trend'] = self._generate_trend_signals()
        details['signal_momentum'] = self._generate_momentum_signals()
        details['signal_oscillator'] = self._generate_oscillator_signals()
        details['signal_volume'] = self._generate_volume_signals()
        details['signal_volatility'] = self._generate_volatility_signals()
        
        # Add final signals and strength
        details['final_signal'] = self.generate_signals(data)
        details['signal_strength'] = self.get_signal_strength(data)
        
        return details

# ============================================================================
# BACKTESTING ENGINES - Lego Brick #7
# ============================================================================

class ClassicCVBacktester(BaseBacktester):
    """Classic cross-validation backtester."""

    def __init__(self, n_splits: int = 5, test_size: int = 252):
        self.n_splits = n_splits
        self.test_size = test_size

    def run_backtest(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        feature_extractor: BaseFeatureExtractor,
        labeling_strategy: BaseLabelingStrategy,
    ) -> Dict[str, Any]:
        """Run classic CV backtest."""

        features = feature_extractor.extract_features(data)
        labels = labeling_strategy.create_labels(data)

        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]

        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)

        results = {
            "predictions": [],
            "actuals": [],
            "dates": [],
            "accuracies": [],
            "split_results": [],
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

            print(f"Fold {fold} class balance: {Counter(y_test)}")

            if not is_balanced(y_test):
                print(f"Skipping fold {fold} due to imbalance or missing class")
                continue

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            results["predictions"].extend(y_pred)
            results["actuals"].extend(y_test)
            results["dates"].extend(X_test.index)
            results["accuracies"].append(accuracy)
            results["split_results"].append(
                {
                    "fold": fold,
                    "accuracy": accuracy,
                    "predictions": y_pred,
                    "actuals": y_test.values,
                }
            )

        results["overall_accuracy"] = (
            np.mean(results["accuracies"]) if results["accuracies"] else 0.0
        )

        return results


class WalkForwardBacktester(BaseBacktester):
    """Walk-forward backtester."""

    def __init__(self, train_size: int = 252, test_size: int = 63, step_size: int = 21):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run_backtest(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        feature_extractor: BaseFeatureExtractor,
        labeling_strategy: BaseLabelingStrategy,
        tech_assistant: TechnicalSignalGenerator,
    ) -> Dict[str, Any]:
        """Run walk-forward backtest."""
        # Extract features and labels
        features = feature_extractor.extract_features(data)
        labels = labeling_strategy.create_labels(data)

        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]

        results = {
            "predictions": [],
            "actuals": [],
            "dates": [],
            "accuracies": [],
            "walk_results": [],
        }

        start_idx = self.train_size
        walk = 0

        while start_idx + self.test_size <= len(features):
            # Define train and test windows
            train_start = start_idx - self.train_size
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + self.test_size, len(features))

            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_test = features.iloc[test_start:test_end]
            y_test = labels.iloc[test_start:test_end]

            # Train model

            model.fit(X_train, y_train)
            

            # Predict
            y_pred = model.predict(X_test)
            if tech_assistant:
                signal_combiner = SignalCombiner()
                tech_pred = tech_assistant.predict(data)
                y_pred = signal_combiner.combine(y_pred, tech_pred)

            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            results["predictions"].extend(y_pred)
            results["actuals"].extend(y_test)
            results["dates"].extend(X_test.index)
            results["accuracies"].append(accuracy)
            results["walk_results"].append(
                {
                    "walk": walk,
                    "accuracy": accuracy,
                    "train_period": (X_train.index[0], X_train.index[-1]),
                    "test_period": (X_test.index[0], X_test.index[-1]),
                }
            )

            # Move to next window
            start_idx += self.step_size
            walk += 1

        results["overall_accuracy"] = np.mean(results["accuracies"])
        return results

class SingleFoldBacktester(BaseBacktester):
    """Single-fold backtester that splits data into one training and one testing period."""

    def __init__(self, train_ratio: float = 0.8):
        """
        Initialize single fold backtester.
        
        Args:
            train_ratio: Proportion of data to use for training (e.g., 0.8 = 80% train, 20% test)
        """
        self.train_ratio = train_ratio

    def run_backtest(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        feature_extractor: BaseFeatureExtractor,
        labeling_strategy: BaseLabelingStrategy,
    ) -> Dict[str, Any]:
        """Run single-fold backtest."""
        # Extract features and labels
        features = feature_extractor.extract_features(data)
        labels = labeling_strategy.create_labels(data)

        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]

        # Calculate split point
        split_idx = int(len(features) * self.train_ratio)

        # Split data
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        tech_pred = TechnicalSignalGenerator().generate_signals(data)
        y_pred = self._combine_signals(y_pred, tech_pred)
        print(tech_pred, y_pred)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        results = {
            "predictions": list(y_pred),
            "actuals": list(y_test),
            "dates": list(X_test.index),
            "accuracies": [accuracy],  # Single accuracy value in list for consistency
            "overall_accuracy": accuracy,
            "train_period": (X_train.index[0], X_train.index[-1]),
            "test_period": (X_test.index[0], X_test.index[-1]),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return results

class EventBasedSimulator(BaseSimulator):
    """Improved event-based trading simulator."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        max_exposure_ratio: float = 0.15,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_exposure_ratio = max_exposure_ratio

    def calculate_portfolio_value(self, cash, position, price):
        return cash + position * price

    def simulate(self, signals: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        cash = self.initial_capital
        position = 0.0
        entry_price = None
        portfolio_value = self.initial_capital

        trades = []
        portfolio_history = []

        for date, signal in signals.items():
            if date not in data.index:
                continue

            price = data.loc[date, "close"]

            # === Close Long ===
            if signal == Signal.SHORT.value and position > 0:
                pnl = position * (price - entry_price)
                cash += position * price * (1 - self.commission)
                trades.append(
                    {
                        "date": date,
                        "action": "sell",
                        "price": price,
                        "quantity": position,
                        "pnl": pnl,
                    }
                )
                position = 0
                entry_price = None

            # === Close Short ===
            elif signal == Signal.LONG.value and position < 0:
                pnl = abs(position) * (entry_price - price)
                cash += abs(position) * price * (1 - self.commission)
                trades.append(
                    {
                        "date": date,
                        "action": "cover_short",
                        "price": price,
                        "quantity": abs(position),
                        "pnl": pnl,
                    }
                )
                position = 0
                entry_price = None

            # === Open Long ===
            if signal == Signal.LONG.value and position == 0:
                quantity = (cash * self.max_exposure_ratio) / price
                cash -= quantity * price * (1 + self.commission)
                position = quantity
                entry_price = price
                trades.append(
                    {
                        "date": date,
                        "action": "buy",
                        "price": price,
                        "quantity": quantity,
                        "pnl": 0,
                    }
                )

            # === Open Short ===
            elif signal == Signal.SHORT.value and position == 0:
                quantity = (cash * self.max_exposure_ratio) / price
                cash += quantity * price * (1 - self.commission)
                position = -quantity
                entry_price = price
                trades.append(
                    {
                        "date": date,
                        "action": "short",
                        "price": price,
                        "quantity": quantity,
                        "pnl": 0,
                    }
                )

            # === Update portfolio ===
            portfolio_value = self.calculate_portfolio_value(cash, position, price)
            portfolio_history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position": position,
                    "price": price,
                }
            )

        # === Final close if position remains ===
        if position != 0 and entry_price is not None:
            last_date = signals.index[-1]
            last_price = data.loc[last_date, "close"]
            if position > 0:
                pnl = position * (last_price - entry_price)
                cash += position * last_price * (1 - self.commission)
                trades.append(
                    {
                        "date": last_date,
                        "action": "sell_final",
                        "price": last_price,
                        "quantity": position,
                        "pnl": pnl,
                    }
                )
            elif position < 0:
                pnl = abs(position) * (entry_price - last_price)
                cash += abs(position) * last_price * (1 - self.commission)
                trades.append(
                    {
                        "date": last_date,
                        "action": "cover_short_final",
                        "price": last_price,
                        "quantity": abs(position),
                        "pnl": pnl,
                    }
                )
            position = 0
            entry_price = None
            portfolio_value = cash

            portfolio_history.append(
                {
                    "date": last_date,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position": position,
                    "price": last_price,
                }
            )

        # === Metrics ===
        df_history = pd.DataFrame(portfolio_history).set_index("date")
        df_history["drawdown"] = (
            df_history["portfolio_value"] - df_history["portfolio_value"].cummax()
        )
        max_drawdown = df_history["drawdown"].min()

        return {
            "trades": trades,
            "portfolio_history": df_history,
            "final_value": portfolio_value,
            "total_return": (portfolio_value - self.initial_capital)
            / self.initial_capital,
            "max_drawdown": max_drawdown,
            "num_trades": len(trades),
        }

class EventBasedSimulatorModified(BaseSimulator):
    """Event-based trading simulator with proper cash management."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        max_exposure_ratio: float = 0.15,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_exposure_ratio = max_exposure_ratio

    def calculate_portfolio_value(self, cash, position, price):
        return cash + position * price

    def simulate(self, signals: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        cash = self.initial_capital
        position = 0.0
        entry_price = None
        portfolio_value = self.initial_capital

        trades = []
        portfolio_history = []
        
        for date, signal in signals.items():
            if date not in data.index:
                continue

            price = data.loc[date, "close"]

            # === Close Long Position ===
            if signal == Signal.SHORT.value and position > 0:
                pnl = position * (price - entry_price)
                cash += position * price * (1 - self.commission)
                trades.append(
                    {
                        "date": date,
                        "action": "sell",
                        "price": price,
                        "quantity": position,
                        "pnl": pnl,
                    }
                )
                position = 0
                entry_price = None

            # === Close Short Position ===
            elif signal == Signal.LONG.value and position < 0:
                pnl = abs(position) * (entry_price - price)
                cost_to_cover = abs(position) * price * (1 + self.commission)
                
                # Check if we have enough cash to cover the short
                if cash >= cost_to_cover:
                    cash -= cost_to_cover
                    trades.append(
                        {
                            "date": date,
                            "action": "cover_short",
                            "price": price,
                            "quantity": abs(position),
                            "pnl": pnl,
                        }
                    )
                    position = 0
                    entry_price = None
                else:
                    # Insufficient cash - return current state
                    portfolio_value = self.calculate_portfolio_value(cash, position, price)
                    df_history = pd.DataFrame(portfolio_history).set_index("date") if portfolio_history else pd.DataFrame()
                    
                    return {
                        "trades": trades,
                        "portfolio_history": df_history,
                        "final_value": portfolio_value,
                        "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
                        "max_drawdown": 0 if df_history.empty else (df_history["portfolio_value"] - df_history["portfolio_value"].cummax()).min(),
                        "num_trades": len(trades),
                        "terminated_early": True,
                        "termination_reason": "Insufficient cash to cover short position"
                    }

            # === Open Long Position ===
            if signal == Signal.LONG.value and position == 0:
                available_capital = cash * self.max_exposure_ratio
                cost_per_share = price * (1 + self.commission)
                quantity = available_capital / cost_per_share
                total_cost = quantity * cost_per_share
                
                if cash >= total_cost:
                    cash -= total_cost
                    position = quantity
                    entry_price = price
                    trades.append(
                        {
                            "date": date,
                            "action": "buy",
                            "price": price,
                            "quantity": quantity,
                            "pnl": 0,
                        }
                    )
                else:
                    # Insufficient cash - return current state
                    portfolio_value = self.calculate_portfolio_value(cash, position, price)
                    df_history = pd.DataFrame(portfolio_history).set_index("date") if portfolio_history else pd.DataFrame()
                    
                    return {
                        "trades": trades,
                        "portfolio_history": df_history,
                        "final_value": portfolio_value,
                        "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
                        "max_drawdown": 0 if df_history.empty else (df_history["portfolio_value"] - df_history["portfolio_value"].cummax()).min(),
                        "num_trades": len(trades),
                        "terminated_early": True,
                        "termination_reason": "Insufficient cash to open long position"
                    }

            # === Open Short Position ===
            elif signal == Signal.SHORT.value and position == 0:
                available_capital = cash * self.max_exposure_ratio
                quantity = available_capital / price
                proceeds = quantity * price * (1 - self.commission)
                
                cash += proceeds
                position = -quantity
                entry_price = price
                trades.append(
                    {
                        "date": date,
                        "action": "short",
                        "price": price,
                        "quantity": quantity,
                        "pnl": 0,
                    }
                )

            # === Update Portfolio History ===
            portfolio_value = self.calculate_portfolio_value(cash, position, price)
            portfolio_history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position": position,
                    "price": price,
                }
            )

        # === Final Position Closure ===
        if position != 0 and entry_price is not None:
            last_date = signals.index[-1]
            last_price = data.loc[last_date, "close"]
            
            if position > 0:
                # Close long position
                pnl = position * (last_price - entry_price)
                cash += position * last_price * (1 - self.commission)
                trades.append(
                    {
                        "date": last_date,
                        "action": "sell_final",
                        "price": last_price,
                        "quantity": position,
                        "pnl": pnl,
                    }
                )
            elif position < 0:
                # Close short position
                pnl = abs(position) * (entry_price - last_price)
                cost_to_cover = abs(position) * last_price * (1 + self.commission)
                
                if cash >= cost_to_cover:
                    cash -= cost_to_cover
                    trades.append(
                        {
                            "date": last_date,
                            "action": "cover_short_final",
                            "price": last_price,
                            "quantity": abs(position),
                            "pnl": pnl,
                        }
                    )
                else:
                    # Mark-to-market the position if we can't cover
                    pnl = abs(position) * (entry_price - last_price)
                    trades.append(
                        {
                            "date": last_date,
                            "action": "mark_to_market",
                            "price": last_price,
                            "quantity": abs(position),
                            "pnl": pnl,
                        }
                    )
            
            position = 0
            entry_price = None
            portfolio_value = cash if position == 0 else self.calculate_portfolio_value(cash, position, last_price)

            portfolio_history.append(
                {
                    "date": last_date,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position": position,
                    "price": last_price,
                }
            )

        # === Calculate Metrics ===
        df_history = pd.DataFrame(portfolio_history).set_index("date")
        
        if not df_history.empty:
            df_history["drawdown"] = (
                df_history["portfolio_value"] - df_history["portfolio_value"].cummax()
            )
            max_drawdown = df_history["drawdown"].min()
        else:
            max_drawdown = 0

        return {
            "trades": trades,
            "portfolio_history": df_history,
            "final_value": portfolio_value,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
            "max_drawdown": max_drawdown,
            "num_trades": len(trades),
            "terminated_early": False,
            "termination_reason": None
        }

# ============================================================================
# PORTFOLIO MANAGEMENT - Lego Brick #9
# ============================================================================
class FixedSizePortfolioManager(BasePortfolioManager):
    """Fixed position size portfolio manager."""

    def __init__(self, position_size: float = 0.1):
        self.position_size = position_size
        self.current_position = 0
        self.cash = 0
        self.portfolio_value = 0

    def calculate_position_size(
        self, signal: Signal, current_price: float, portfolio_value: float
    ) -> float:
        """Calculate fixed position size."""
        if signal == Signal.HOLD:
            return 0
        return portfolio_value * self.position_size / current_price

    def update_portfolio(self, signal: Signal, price: float, quantity: float) -> None:
        """Update portfolio state."""
        if signal == Signal.LONG:
            self.current_position += quantity
            self.cash -= quantity * price
        elif signal == Signal.SHORT:
            self.current_position -= quantity
            self.cash += quantity * price


# ============================================================================
# MAIN TRADING SYSTEM - Lego Brick #10 (The Master Brick)
# ============================================================================
class ModularTradingSystem:
    """
    Main trading system that orchestrates all components.
    This is the "master Lego brick" that connects all other bricks.
    """

    def __init__(self, technical_assistance: bool=False):
        self.feature_extractor: Optional[BaseFeatureExtractor] = None
        self.labeling_strategy: Optional[BaseLabelingStrategy] = None
        self.model: Optional[BaseModel] = None
        self.backtester: Optional[BaseBacktester] = None
        self.simulator: Optional[BaseSimulator] = None
        self.portfolio_manager: Optional[BasePortfolioManager] = None
        self.tech_assistant = TechnicalSignalGenerator()
        self.signal_combiner = SignalCombiner()

        # System state
        self.is_configured = False
        self.is_trained = False
        self.last_data: Optional[pd.DataFrame] = None
        self.last_features: Optional[pd.DataFrame] = None

    def plug_in_feature_extractor(
        self, feature_extractor: BaseFeatureExtractor
    ) -> "ModularTradingSystem":
        """Plug in a feature extractor component."""
        self.feature_extractor = feature_extractor
        self.is_configured = True
        return self

    def plug_in_labeling_strategy(
        self, labeling_strategy: BaseLabelingStrategy
    ) -> "ModularTradingSystem":
        """Plug in a labeling strategy component."""
        self.labeling_strategy = labeling_strategy
        self.is_configured = True
        return self

    def plug_in_model(self, model: BaseModel) -> "ModularTradingSystem":
        """Plug in a trading model component."""
        self.model = model
        self.is_configured = True
        return self

    def plug_in_backtester(self, backtester: BaseBacktester) -> "ModularTradingSystem":
        """Plug in a backtesting engine component."""
        self.backtester = backtester
        self.is_configured = True
        return self

    def plug_in_simulator(self, simulator: BaseSimulator) -> "ModularTradingSystem":
        """Plug in a simulation engine component."""
        self.simulator = simulator
        self.is_configured = True
        return self

    def plug_in_portfolio_manager(
        self, portfolio_manager: BasePortfolioManager
    ) -> "ModularTradingSystem":
        """Plug in a portfolio manager component."""
        self.portfolio_manager = portfolio_manager
        self.is_configured = True
        return self

    def configure_default_system(self) -> "ModularTradingSystem":
        """Configure system with default components."""
        if not self.feature_extractor:
            self.plug_in_feature_extractor(TechnicalFeatureExtractor())
        if not self.labeling_strategy:
            self.plug_in_labeling_strategy(ReturnBasedLabeling())
        if not self.model:
            self.plug_in_model(MLModel(ModelType.GRADIENT_BOOSTING))
        if not self.backtester:
            self.plug_in_backtester(ClassicCVBacktester())
        if not self.simulator:
            self.plug_in_simulator(EventBasedSimulator())
        if not self.portfolio_manager:
            self.plug_in_portfolio_manager(FixedSizePortfolioManager())

        self.is_configured = True
        return self

    def train_system(self, data: pd.DataFrame) -> "ModularTradingSystem":
        """Train the system on historical data."""
        if not self.is_configured:
            self.configure_default_system()

        print("Training system...")

        # Extract features and labels
        features = self.feature_extractor.extract_features(data)
        labels = self.labeling_strategy.create_labels(data)

        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]

        print("Feature shape:", features.shape)
        print("Label shape:", labels.shape)
        print("Label distribution:", labels.value_counts().to_dict())

       
        

        # Train model
        self.model.fit(features, labels)
        self.tech_assistant.fit(data)

        # Store data for live trading
        self.last_data = data
        self.last_features = features
        self.is_trained = True

        print("System trained successfully!")
        return self

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete backtest."""
        if not self.is_configured:
            self.configure_default_system()

        print("Running backtest...")

        # Run backtest
        backtest_results = self.backtester.run_backtest(
            data, self.model, self.feature_extractor, self.labeling_strategy,
        )

        


        # Convert predictions to signals
        signals = pd.Series(
            backtest_results["predictions"], index=backtest_results["dates"]
        )
        

        # Run simulation
        simulation_results = self.simulator.simulate(signals, data)

        # Combine results
        results = {
            "backtest": backtest_results,
            "simulation": simulation_results,
            "summary": {
                "overall_accuracy": backtest_results["overall_accuracy"],
                "total_return": simulation_results["total_return"],
                "num_trades": simulation_results["num_trades"],
                "final_value": simulation_results["final_value"],
            },
        }

        print("Backtest completed!")
        return results

    def generate_live_signal(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate live trading signal based on current market data.
        This is the key function for real-time trading.
        """
        if not self.is_trained:
            raise ValueError("System must be trained before generating live signals")

        # Extract features from current data
        features = self.feature_extractor.extract_features(current_data)

        # Get latest feature row (most recent data point)
        latest_features = features.iloc[-1:].fillna(0)

        # Generate prediction
        prediction = self.model.predict(latest_features)[0]
        prediction_proba = self.model.predict_proba(latest_features)[0]

        # Convert to signal
        signal = Signal(prediction)

        # Get current price
        current_price = current_data["close"].iloc[-1]

        # Calculate position size
        portfolio_value = 100000  # This should come from actual portfolio value
        position_size = self.portfolio_manager.calculate_position_size(
            signal, current_price, portfolio_value
        )

        return {
            "timestamp": current_data.index[-1],
            "signal": signal,
            "signal_value": prediction,
            "confidence": prediction_proba,
            "current_price": current_price,
            "position_size": position_size,
            "raw_prediction": prediction,
            "prediction_probabilities": prediction_proba.tolist(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_configured": self.is_configured,
            "is_trained": self.is_trained,
            "components": {
                "feature_extractor": type(self.feature_extractor).__name__
                if self.feature_extractor
                else None,
                "labeling_strategy": type(self.labeling_strategy).__name__
                if self.labeling_strategy
                else None,
                "model": type(self.model).__name__ if self.model else None,
                "backtester": type(self.backtester).__name__
                if self.backtester
                else None,
                "simulator": type(self.simulator).__name__ if self.simulator else None,
                "portfolio_manager": type(self.portfolio_manager).__name__
                if self.portfolio_manager
                else None,
            },
        }


# ============================================================================
# UTILITY FUNCTIONS - Lego Brick #11
# ============================================================================
def create_sample_data(
    start_date: str = "2020-01-01", end_date: str = "2024-01-01", symbol: str = "SAMPLE"
) -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    # Generate realistic price data using random walk with drift
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns

    # Starting price
    initial_price = 100.0
    prices = [initial_price]

    for i in range(1, n_days):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 0.01))  # Prevent negative prices

    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    data["close"] = prices

    # Generate high/low based on close with realistic spread
    daily_volatility = np.random.uniform(0.005, 0.03, n_days)
    data["high"] = data["close"] * (
        1 + daily_volatility * np.random.uniform(0.3, 0.8, n_days)
    )
    data["low"] = data["close"] * (
        1 - daily_volatility * np.random.uniform(0.3, 0.8, n_days)
    )

    # Generate open (gap from previous close)
    gap_factor = np.random.normal(1.0, 0.005, n_days)
    data["open"] = data["close"].shift(1) * gap_factor
    data["open"].iloc[0] = initial_price

    # Ensure OHLC logic is maintained
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    # Generate volume
    base_volume = 1000000
    volume_factor = np.random.lognormal(0, 0.5, n_days)
    data["volume"] = (base_volume * volume_factor).astype(int)

    return data


def print_live_signal(signal_data: Dict[str, Any]) -> None:
    """Pretty print live signal data."""
    print("\n" + "=" * 60)
    print("LIVE TRADING SIGNAL")
    print("=" * 60)

    print(f"Timestamp: {signal_data['timestamp']}")
    print(f"Signal: {signal_data['signal']} ({signal_data['signal_value']})")
    print(f"Current Price: ${signal_data['current_price']:.2f}")
    print(f"Position Size: {signal_data['position_size']:.2f}")
    print(f"Confidence: {signal_data['confidence']}")
    print("=" * 60)


def save_system_config(system: ModularTradingSystem, filepath: str) -> None:
    """Save system configuration to file."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "system_status": system.get_system_status(),
        "components": system.get_system_status()["components"],
    }

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

    print(f"System configuration saved to {filepath}")


def is_balanced(
    y: pd.Series, required_classes={-1, 0, 1}, min_samples_per_class=100
) -> bool:
    """Check if all required classes are present with enough samples."""
    class_counts = Counter(y)
    for cls in required_classes:
        if class_counts.get(cls, 0) < min_samples_per_class:
            return False
    return True


# ============================================================================
# ADVANCED COMPONENTS - Optional Lego Bricks #12
# ============================================================================


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all models in ensemble."""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted voting
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        return np.round(weighted_predictions).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        probabilities = []
        for model in self.models:
            proba = model.predict_proba(X)
            probabilities.append(proba)

        # Weighted average of probabilities
        return np.average(probabilities, axis=0, weights=self.weights)


class RiskAdjustedPortfolioManager(BasePortfolioManager):
    """Risk-adjusted position sizing based on volatility."""

    def __init__(self, target_volatility: float = 0.15, lookback_period: int = 20):
        self.target_volatility = target_volatility
        self.lookback_period = lookback_period
        self.current_position = 0
        self.cash = 0
        self.portfolio_value = 0
        self.price_history = []

    def calculate_position_size(
        self, signal: Signal, current_price: float, portfolio_value: float
    ) -> float:
        """Calculate risk-adjusted position size."""
        if signal == Signal.HOLD:
            return 0

        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)

        # Calculate historical volatility
        if len(self.price_history) < 2:
            volatility = 0.02  # Default volatility
        else:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility

        # Adjust position size based on volatility
        volatility_adjustment = self.target_volatility / max(volatility, 0.01)
        base_position_size = portfolio_value * 0.1  # 10% base allocation

        adjusted_position_size = base_position_size * min(
            volatility_adjustment, 2.0
        )  # Cap at 2x

        return adjusted_position_size / current_price

    def update_portfolio(self, signal: Signal, price: float, quantity: float) -> None:
        """Update portfolio state."""
        if signal == Signal.LONG:
            self.current_position += quantity
            self.cash -= quantity * price
        elif signal == Signal.SHORT:
            self.current_position -= quantity
            self.cash += quantity * price


class MonteCarloBacktester(BaseBacktester):
    """Monte Carlo backtester for robust performance evaluation."""

    def __init__(
        self, n_simulations: int = 1000, train_size: int = 252, test_size: int = 63
    ):
        self.n_simulations = n_simulations
        self.train_size = train_size
        self.test_size = test_size

    def run_backtest(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        feature_extractor: BaseFeatureExtractor,
        labeling_strategy: BaseLabelingStrategy,
    ) -> Dict[str, Any]:
        """Run Monte Carlo backtest with validation and error protection."""
        # Extract features and labels
        features = feature_extractor.extract_features(data)
        labels = labeling_strategy.create_labels(data)

        # Drop rows with NaNs in either features or labels
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]

        results = {
            "simulations": [],
            "accuracies": [],
            "predictions": [],
            "actuals_all": [],
        }

        total_samples = len(features)

        # Early exit if not enough data
        if total_samples < self.train_size + self.test_size:
            print("Not enough data to run even one simulation.")
            return results

        for sim in range(self.n_simulations):
            # Calculate valid range for simulation
            max_start = total_samples - self.train_size - self.test_size
            if max_start <= 0:
                print(f"Skipping simulation {sim}: max_start <= 0.")
                continue

            # Randomly pick a valid start index
            start_idx = np.random.randint(0, max_start + 1)

            # Define train and test windows
            train_start = start_idx
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_test = features.iloc[test_start:test_end]
            y_test = labels.iloc[test_start:test_end]

            # Validate sizes
            if len(X_train) == 0 or len(y_train) == 0:
                print(f"Simulation {sim}: Empty training data. Skipping.")
                continue
            if len(X_test) == 0 or len(y_test) == 0:
                print(f"Simulation {sim}: Empty testing data. Skipping.")
                continue

            try:
                # Train the model
                model.fit(X_train, y_train)

                # Predict on test set
                y_pred = model.predict(X_test)
                tech_pred = TechnicalSignalGenerator().generate_signals(data)
                y_pred = self._combine_signals(y_pred, tech_pred)
                

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)

                # Store simulation results
                results["simulations"].append(
                    {
                        "simulation": sim,
                        "accuracy": accuracy,
                        "train_period": (X_train.index[0], X_train.index[-1]),
                        "test_period": (X_test.index[0], X_test.index[-1]),
                    }
                )
                results["accuracies"].append(accuracy)
                results["predictions"].extend(y_pred)
                results["actuals_all"].extend(y_test)
                results.setdefault("prediction_dates", []).extend(X_test.index.tolist())

            except Exception as e:
                print(f"Simulation {sim} failed due to error: {e}")
                continue
        # Post-processing statistics
        if results["accuracies"]:
            accuracies = np.array(results["accuracies"])
            results["overall_accuracy"] = np.mean(accuracies)
            results["accuracy_std"] = np.std(accuracies)
            results["accuracy_confidence_interval"] = (
                np.percentile(accuracies, 5),
                np.percentile(accuracies, 95),
            )
        else:
            results["overall_accuracy"] = None
            results["accuracy_std"] = None
            results["accuracy_confidence_interval"] = (None, None)

        # Add final predictions and associated dates
        results["dates"] = results.get("prediction_dates", [])

        return results


# ============================================================================
# EXAMPLE USAGE AND DEMO - Lego Brick #11
# ============================================================================



async def main(symbols):
    backtested = False
    fetcher = Fetcher()
    data = await fetcher.fetch_multiple(symbols, "15", 50)

    if not data:
        raise ValueError("No data fetched...closing system...")
        return None

    # 🚨 CRITICAL FIX: Create system factory instead of shared instance
    def create_system():
        """Factory function to create isolated system instances"""
        models = [
        MLModel(ModelType.RANDOM_FOREST),
        MLModel(ModelType.GRADIENT_BOOSTING),
        MLModel(ModelType.MLP),
        ]
        ensemble_model = EnsembleModel(models, weights=[0.4, 0.4, 0.2])

        system = ModularTradingSystem(           
            ).plug_in_labeling_strategy(
            VolatilityAdjustedLabeling(lookforward_periods=160)
            ).plug_in_model(
            MLModel(ModelType.RANDOM_FOREST, n_estimators=200)
            ).plug_in_backtester(
            MonteCarloBacktester(n_simulations=200, train_size=500, test_size=160)
            ).plug_in_feature_extractor(
                TechnicalFeatureExtractor([10, 20, 50, 200])
            ).plug_in_portfolio_manager(
                FixedSizePortfolioManager()
            ).plug_in_simulator(
                EventBasedSimulatorModified(max_exposure_ratio=.95)
            )
        
        system.is_configured = True
        return system

    # Fixed: Each symbol gets its own system instance
    def backtest_one_symbol(symbol, s_d):
        # Create isolated system for this thread
        local_system = create_system()
        print(local_system.get_system_status())

        try:
            print(f"[{symbol}] Starting training...")
            local_system.train_system(s_d)
            print(f"[{symbol}] Training complete")
            time.sleep(1)

            print(f"[{symbol}] Running backtest...")
            results = local_system.run_backtest(s_d)
            print(f"[{symbol}] Backtest complete")

            print_backtest_results(results)
            time.sleep(1)  # Pass symbol for identification
            return results

        except Exception as e:
            print(f"[{symbol}] Backtest error: {e}")
            import traceback

            traceback.print_exc()
            return None

    # Fixed: Each symbol gets its own system for live trading
    def get_live_for_one(symbol, resolution, lookback_days):
        # Create isolated system for live trading
        live_system = create_system()
        print(f"[{symbol}] Live system initiated...")

        # First, we need to train the system with recent data
        try:
            recent_data = fetcher._fetch_ohlcv_sync(
                symbol, resolution, 40
            )  # Get more data for training
            live_system.train_system(recent_data)
            print(f"[{symbol}] Live system trained")
        except Exception as e:
            print(f"[{symbol}] Live training error: {e}")
            return

        # Now run live loop
        while True:
            try:
                data = fetcher._fetch_ohlcv_sync(symbol, resolution, lookback_days)
                signal = live_system.generate_live_signal(data)
                print_live_signal(signal)  # Pass symbol for identification

                # Add sleep to prevent hammering the API
                import time

                time.sleep(900)  # Wait 1 minute between signals

            except Exception as e:
                print(f"[{symbol}] Live error: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying

    # Run backtests in parallel using asyncio.to_thread()
    print("Starting backtests...")
    backtest_tasks = [
        asyncio.to_thread(backtest_one_symbol, symbol, s_d)
        for symbol, s_d in data.items()
    ]

    # Collect results
    backtest_results = await asyncio.gather(*backtest_tasks, return_exceptions=True)

    # Check if any backtests failed
    successful_backtests = []
    for i, result in enumerate(backtest_results):
        symbol = list(data.keys())[i]
        if isinstance(result, Exception):
            print(f"[{symbol}] Backtest failed: {result}")

        elif result is not None:
            successful_backtests.append((symbol, result))

    print(f"{len(successful_backtests)} backtests completed successfully.")
    backtested = len(successful_backtests) > 0

    await asyncio.sleep(1)
   
    backtested = False
    # Run live trading only if at least one backtest succeeded
    if backtested:
        print("Starting live trading...")
        live_tasks = [
            asyncio.to_thread(get_live_for_one, symbol, "15", 30) for symbol in symbols
        ]

        try:
            await asyncio.gather(*live_tasks)
        except KeyboardInterrupt:
            print("Live trading stopped by user")
            
        except Exception as e:
            print(f"Live trading error: {e}")
            


def print_backtest_results(results: Dict[str, Any]) -> None:
    """Pretty print backtest results."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    summary = results["summary"]
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    print(f"Max Drawdown: {results['simulation']['max_drawdown']:.2f}")
    print(f"Total Return: {summary['total_return']:.2%}")
    print(f"Final Portfolio Value: ${summary['final_value']:,.2f}")
    print(f"Number of Trades: {summary['num_trades']}")

    if "backtest" in results:
        print(f"Cross-Validation Folds: {len(results['backtest']['accuracies'])}")
        print(
            f"Accuracy per Fold: {[f'{acc:.2%}' for acc in results['backtest']['accuracies']]}"
        )

    print("=" * 80)

# Additional debugging helper
def validate_system_state(system, symbol):
    """Validate that the system is properly initialized"""
    try:
        if hasattr(system, "model") and hasattr(system.model, "scaler"):
            scaler = system.model.scaler
            if hasattr(scaler, "mean_"):
                print(f"[{symbol}] Scaler properly fitted")
            else:
                print(f"[{symbol}] Scaler not fitted")
        return True
    except Exception as e:
        print(f"[{symbol}] System validation failed: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run
    asyncio.run(main(["TSLA", "MSFT"]))

    print("\n Available Systems:")
    print("1. systems[0] - Default system")
    print("2. systems[1] - Custom system")
    print("3. systems[2] - Ensemble system")

    print("\n🎯 Key Functions:")
    print("- system.generate_live_signal(current_data) - Get live trading signal")
    print("- system.run_backtest(data) - Run complete backtest")
    print("- system.train_system(data) - Train system on new data")
    print("- create_sample_data() - Generate sample market data")

    print("\n✨ The system is ready for use!")
    print("You can now plug in your own components or use the existing ones.")
    print("Each component follows the Lego-brick architecture for easy swapping.")
