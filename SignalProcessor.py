"""
Phase 2: GroupBlock System - Aggregate Indicators into Coherent Groups

This module implements the GroupBlock system that aggregates individual indicator
signals into group-level predictions using Bayesian logic, dispersion-based
confidence, and ensemble methods.

Architecture:
- Groups indicators by type (trend, momentum, volume, price_action)
- Uses weighted averaging based on historical performance and confidence
- Computes group-level confidence using dispersion and alignment metrics
- Provides structured output compatible with SystemEngine (Phase 3)

Author: VOID Intelligent Systems
Version: 2.5
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod
import warnings
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class GroupOutput:
    """Structured output for group-level signals."""
    group_name: str
    signal_value: float          # Aggregated signal [-1, 1]
    confidence: float           # Group confidence [0, 1]
    probability: float          # P(up) for the group
    variance: float            # Uncertainty in future returns
    explanation: str           # Human-readable reasoning
    component_count: int       # Number of indicators in group
    alignment_score: float     # How well indicators agree
    dispersion: float         # Signal dispersion (lower = higher confidence)
    weighted_contribution: Dict[str, float]  # Each indicator's contribution
    metadata: Dict[str, Any]   # Diagnostic information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroupOutput':
        """Create from dictionary."""
        return cls(**data)

class BaseAggregator(ABC):
    """Base class for signal aggregation methods."""
    
    @abstractmethod
    def aggregate(self, signals: List[float], weights: List[float]) -> float:
        """Aggregate signals using specific method."""
        pass
    
    @abstractmethod
    def compute_confidence(self, signals: List[float], weights: List[float], 
                          confidences: List[float]) -> float:
        """Compute aggregated confidence."""
        pass

class BayesianAggregator(BaseAggregator):
    """Bayesian weighted aggregation with uncertainty propagation."""
    
    def __init__(self, prior_weight: float = 0.1):
        self.prior_weight = prior_weight
    
    def aggregate(self, signals: List[float], weights: List[float]) -> float:
        """Bayesian weighted average with prior."""
        if not signals or not weights:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights) + self.prior_weight
        normalized_weights = [w / total_weight for w in weights]
        prior_contrib = self.prior_weight / total_weight
        
        # Weighted sum with neutral prior
        weighted_sum = sum(s * w for s, w in zip(signals, normalized_weights))
        return weighted_sum  # Prior is 0 (neutral)
    
    def compute_confidence(self, signals: List[float], weights: List[float], 
                          confidences: List[float]) -> float:
        """Confidence based on weighted average and dispersion."""
        if not signals:
            return 0.0
        
        # Weighted confidence
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_conf = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        
        # Dispersion penalty (agreement bonus)
        if len(signals) > 1:
            signal_std = np.std(signals)
            # Higher dispersion reduces confidence
            dispersion_factor = np.exp(-2.0 * signal_std)  # λ = 2.0
            weighted_conf *= dispersion_factor
        
        return np.clip(weighted_conf, 0.0, 1.0)

class RobustAggregator(BaseAggregator):
    """Robust aggregation using median and MAD."""
    
    def aggregate(self, signals: List[float], weights: List[float]) -> float:
        """Weighted median aggregation."""
        if not signals:
            return 0.0
        
        if len(signals) == 1:
            return signals[0]
        
        # For robust aggregation, use weighted median
        # Sort by signal value
        paired = list(zip(signals, weights))
        paired.sort(key=lambda x: x[0])
        
        total_weight = sum(weights)
        if total_weight == 0:
            return np.median(signals)
        
        # Find weighted median
        cumulative_weight = 0
        for signal, weight in paired:
            cumulative_weight += weight
            if cumulative_weight >= total_weight / 2:
                return signal
        
        return paired[-1][0]  # Fallback
    
    def compute_confidence(self, signals: List[float], weights: List[float], 
                          confidences: List[float]) -> float:
        """Confidence using MAD and weighted average."""
        if not signals:
            return 0.0
        
        # Median confidence
        median_conf = np.median(confidences)
        
        # MAD-based dispersion
        if len(signals) > 1:
            median_signal = np.median(signals)
            mad = np.median([abs(s - median_signal) for s in signals])
            # Lower MAD = higher confidence
            mad_factor = np.exp(-3.0 * mad)  # More aggressive than std
            median_conf *= mad_factor
        
        return np.clip(median_conf, 0.0, 1.0)

class GroupBlock:
    """
    Aggregates individual indicator signals into group-level predictions.
    
    Groups indicators by type and uses ensemble methods to create robust
    group-level signals with confidence scoring and uncertainty quantification.
    """
    
    def __init__(self, 
                 aggregation_method: str = 'bayesian',
                 confidence_threshold: float = 0.1,
                 min_indicators_per_group: int = 1,
                 decay_factor: float = 0.95,
                 max_history: int = 1000):
        """
        Initialize GroupBlock system.
        
        Args:
            aggregation_method: 'bayesian', 'robust', or 'simple'
            confidence_threshold: Minimum confidence for valid signals
            min_indicators_per_group: Minimum indicators needed for group signal
            decay_factor: Exponential decay for historical performance
            max_history: Maximum history records to maintain
        """
        self.aggregation_method = aggregation_method
        self.confidence_threshold = confidence_threshold
        self.min_indicators_per_group = min_indicators_per_group
        self.decay_factor = decay_factor
        self.max_history = max_history
        
        # Initialize aggregator
        self.aggregator = self._create_aggregator(aggregation_method)
        
        # Define indicator groups
        self.indicator_groups = {
            'trend': ['sma_crossover', 'ema_momentum'],
            'momentum': ['rsi', 'macd'],
            'volume': ['volume_weighted', 'obv'],
            'price_action': ['candlestick_patterns', 'support_resistance']
        }
        
        # Group performance history
        self.group_performance_history = defaultdict(lambda: deque(maxlen=max_history))
        self.group_weights = defaultdict(lambda: defaultdict(float))
        
        # Diagnostics
        self.processing_stats = {
            'groups_processed': 0,
            'signals_aggregated': 0,
            'confidence_adjustments': 0,
            'last_update': None
        }
        
        logger.info(f"GroupBlock initialized with {aggregation_method} aggregation")
    
    def _create_aggregator(self, method: str) -> BaseAggregator:
        """Create appropriate aggregator instance."""
        if method == 'bayesian':
            return BayesianAggregator()
        elif method == 'robust':
            return RobustAggregator()
        else:
            return BayesianAggregator()  # Default fallback
    
    def process_indicator_results(self, indicator_results: Dict[str, Any]) -> Dict[str, GroupOutput]:
        """
        Process indicator results and aggregate into group signals.
        
        Args:
            indicator_results: Dict of {indicator_name: SignalOutput}
            
        Returns:
            Dict of {group_name: GroupOutput}
        """
        try:
            group_results = {}

            for group_name, indicator_names in self.indicator_groups.items():
                group_output = self._process_group(
                    group_name, indicator_names, indicator_results
                )
                group_results[group_name] = group_output # <-- Convert to dict here

            self.processing_stats['groups_processed'] += len(group_results)
            self.processing_stats['last_update'] = datetime.now().isoformat()

            # logger.info(f"Processed {len(group_results)} groups")
            return group_results

            
        except Exception as e:
            logger.error(f"Error processing indicator results: {e}")
            # Return empty results with error information
            return {group: GroupOutput(
                group_name=group,
                signal_value=0.0,
                confidence=0.0,
                probability=0.5,
                variance=0.0,
                explanation=f"Error processing group: {str(e)}",
                component_count=0,
                alignment_score=0.0,
                dispersion=1.0,
                weighted_contribution={},
                metadata={'error': str(e), 'processing_time': datetime.now().isoformat()}
            ).to_dict() for group in self.indicator_groups.keys()}
    
    def _process_group(self, group_name: str, indicator_names: List[str], 
                      indicator_results: Dict[str, Any]) -> GroupOutput:
        """Process a single group of indicators."""
        
        # Extract available indicators for this group
        available_indicators = []
        signals = []
        confidences = []
        probabilities = []
        variances = []
        
        for indicator_name in indicator_names:
            if indicator_name in indicator_results:
                indicator_result = indicator_results[indicator_name]
                available_indicators.append(indicator_name)
                signals.append(indicator_result.signal_value)
                confidences.append(indicator_result.confidence)
                probabilities.append(indicator_result.probability)
                variances.append(indicator_result.variance)
        
        # Handle insufficient indicators
        if len(available_indicators) < self.min_indicators_per_group:
            return self._create_insufficient_data_output(group_name, available_indicators)
        
        # Compute weights based on historical performance and current confidence
        weights = self._compute_indicator_weights(group_name, available_indicators, confidences)
        
        # Aggregate signals
        aggregated_signal = self.aggregator.aggregate(signals, weights)
        
        # Compute group confidence
        group_confidence = self.aggregator.compute_confidence(signals, weights, confidences)
        
        # Apply confidence threshold
        if group_confidence < self.confidence_threshold:
            group_confidence *= 0.5  # Penalty for low confidence
            self.processing_stats['confidence_adjustments'] += 1
        
        # Compute group probability with confidence adjustment
        confidence_steepness = 1.0 + 2.0 * group_confidence  # Higher confidence = steeper sigmoid
        group_probability = 1.0 / (1.0 + np.exp(-confidence_steepness * aggregated_signal))
        
        # Aggregate variance (uncertainty propagation)
        if weights:
            total_weight = sum(weights)
            weighted_variance = sum(v * w for v, w in zip(variances, weights)) / total_weight if total_weight > 0 else np.mean(variances)
        else:
            weighted_variance = np.mean(variances) if variances else 0.0
        
        # Compute alignment and dispersion metrics
        alignment_score = self._compute_alignment_score(signals, weights)
        dispersion = np.std(signals) if len(signals) > 1 else 0.0
        
        # Compute weighted contributions
        total_weight = sum(weights) if weights else len(signals)
        weighted_contribution = {}
        for i, (indicator_name, weight) in enumerate(zip(available_indicators, weights)):
            weighted_contribution[indicator_name] = weight / total_weight if total_weight > 0 else 1.0 / len(signals)
        
        # Generate explanation
        explanation = self._generate_group_explanation(
            group_name, available_indicators, aggregated_signal, 
            group_confidence, alignment_score, dispersion
        )
        
        # Create metadata
        metadata = {
            'aggregation_method': self.aggregation_method,
            'raw_signals': signals,
            'raw_confidences': confidences,
            'weights': weights,
            'processing_time': datetime.now().isoformat(),
            'available_indicators': available_indicators
        }
        
        self.processing_stats['signals_aggregated'] += len(signals)
        
        return GroupOutput(
            group_name=group_name,
            signal_value=np.clip(aggregated_signal, -1.0, 1.0),
            confidence=np.clip(group_confidence, 0.0, 1.0),
            probability=np.clip(group_probability, 0.0, 1.0),
            variance=max(0.0, weighted_variance),
            explanation=explanation,
            component_count=len(available_indicators),
            alignment_score=alignment_score,
            dispersion=dispersion,
            weighted_contribution=weighted_contribution,
            metadata=metadata
        ).to_dict()
    
    def _compute_indicator_weights(self, group_name: str, indicator_names: List[str], 
                                 confidences: List[float]) -> List[float]:
        """Compute weights for indicators based on performance and confidence."""
        weights = []
        
        for i, indicator_name in enumerate(indicator_names):
            # Base weight from current confidence
            base_weight = confidences[i]
            
            # Historical performance weight
            if indicator_name in self.group_weights[group_name]:
                historical_weight = self.group_weights[group_name][indicator_name]
                # Combine current confidence with historical performance
                combined_weight = 0.7 * base_weight + 0.3 * historical_weight
            else:
                combined_weight = base_weight
                # Initialize historical weight
                self.group_weights[group_name][indicator_name] = base_weight
            
            weights.append(max(0.01, combined_weight))  # Minimum weight threshold
        
        return weights
    
    def _compute_alignment_score(self, signals: List[float], weights: List[float]) -> float:
        """Compute how well indicators align (agreement measure)."""
        if len(signals) <= 1:
            return 1.0
        
        # Weighted correlation with consensus
        total_weight = sum(weights)
        if total_weight == 0:
            consensus = np.mean(signals)
        else:
            consensus = sum(s * w for s, w in zip(signals, weights)) / total_weight
        
        # Compute alignment as inverse of weighted disagreement
        disagreements = [abs(s - consensus) * w for s, w in zip(signals, weights)]
        avg_disagreement = sum(disagreements) / total_weight if total_weight > 0 else np.mean([abs(s - consensus) for s in signals])
        
        # Convert to alignment score (higher = better alignment)
        alignment_score = np.exp(-2.0 * avg_disagreement)  # Exponential decay
        
        return np.clip(alignment_score, 0.0, 1.0)
    
    def _generate_group_explanation(self, group_name: str, indicator_names: List[str],
                                  signal: float, confidence: float, alignment: float, 
                                  dispersion: float) -> str:
        """Generate human-readable explanation for group signal."""
        
        # Signal direction
        if signal > 0.1:
            direction = "bullish"
        elif signal < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Confidence level
        if confidence > 0.7:
            conf_level = "high"
        elif confidence > 0.4:
            conf_level = "moderate"
        else:
            conf_level = "low"
        
        # Alignment description
        if alignment > 0.8:
            alignment_desc = "strong consensus"
        elif alignment > 0.6:
            alignment_desc = "moderate agreement"
        else:
            alignment_desc = "mixed signals"
        
        explanation = (f"{group_name.title()} group shows {direction} signal "
                      f"(strength: {abs(signal):.2f}) with {conf_level} confidence "
                      f"based on {alignment_desc} among {len(indicator_names)} indicators")
        
        return explanation
    
    def _create_insufficient_data_output(self, group_name: str, 
                                       available_indicators: List[str]) -> GroupOutput:
        """Create output for groups with insufficient indicators."""
        return GroupOutput(
            group_name=group_name,
            signal_value=0.0,
            confidence=0.0,
            probability=0.5,
            variance=0.0,
            explanation=f"Insufficient indicators for {group_name} group ({len(available_indicators)} available, {self.min_indicators_per_group} required)",
            component_count=len(available_indicators),
            alignment_score=0.0,
            dispersion=0.0,
            weighted_contribution={},
            metadata={
                'insufficient_data': True,
                'available_indicators': available_indicators,
                'required_indicators': self.min_indicators_per_group
            }
        ).to_dict()
    
    def update_group_performance(self, group_results: Dict[str, GroupOutput], 
                               actual_return: float) -> None:
        """Update group performance based on actual market outcomes."""
        try:
            for group_name, group_output in group_results.items():
                if group_output.component_count == 0:
                    continue  # Skip groups with no data
                
                # Compute prediction accuracy
                predicted_direction = 1 if group_output.signal_value > 0 else -1
                actual_direction = 1 if actual_return > 0 else -1
                hit = 1 if predicted_direction == actual_direction else 0
                
                # Compute prediction error
                predicted_return = group_output.signal_value * 0.01  # Scale to return
                error = abs(predicted_return - actual_return)
                
                # Create performance record
                performance_record = {
                    'timestamp': datetime.now().isoformat(),
                    'predicted_signal': group_output.signal_value,
                    'predicted_return': predicted_return,
                    'actual_return': actual_return,
                    'hit': hit,
                    'error': error,
                    'confidence': group_output.confidence,
                    'component_count': group_output.component_count
                }
                
                # Store performance history
                self.group_performance_history[group_name].append(performance_record)
                
                # Update indicator weights within group
                self._update_indicator_weights(group_name, group_output, hit, error)
                
            logger.info(f"Updated performance for {len(group_results)} groups")
            
        except Exception as e:
            logger.error(f"Error updating group performance: {e}")
    
    def _update_indicator_weights(self, group_name: str, group_output: GroupOutput, 
                                hit: int, error: float) -> None:
        """Update weights for indicators within a group based on performance."""
        
        # Performance factor (0.5 to 1.5 multiplier)
        performance_factor = 1.0 + 0.5 * (hit - 0.5)  # Hit=1 -> 1.25, Hit=0 -> 0.75
        error_factor = np.exp(-error * 10)  # Lower error = higher factor
        combined_factor = performance_factor * error_factor
        
        # Update weights with exponential decay
        for indicator_name, contribution in group_output.weighted_contribution.items():
            current_weight = self.group_weights[group_name][indicator_name]
            
            # Apply Bayesian update with decay
            updated_weight = (self.decay_factor * current_weight + 
                            (1 - self.decay_factor) * combined_factor * contribution)
            
            self.group_weights[group_name][indicator_name] = np.clip(updated_weight, 0.01, 2.0)
    
    def get_group_summary(self, group_results: Dict[str, GroupOutput]) -> Dict[str, Any]:
        """Generate summary statistics for group processing."""
        if not group_results:
            return {'error': 'No group results available'}
        
        signals = [r.signal_value for r in group_results.values()]
        confidences = [r.confidence for r in group_results.values()]
        probabilities = [r.probability for r in group_results.values()]
        
        # Overall signal consensus
        overall_signal = np.mean(signals)
        signal_consensus = np.std(signals)
        
        # Confidence metrics
        avg_confidence = np.mean(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Group alignment
        bullish_groups = sum(1 for s in signals if s > 0.1)
        bearish_groups = sum(1 for s in signals if s < -0.1)
        neutral_groups = len(signals) - bullish_groups - bearish_groups
        
        # Component analysis
        total_indicators = sum(r.component_count for r in group_results.values())
        avg_alignment = np.mean([r.alignment_score for r in group_results.values()])
        
        return {
            'overall_signal': overall_signal,
            'signal_consensus': signal_consensus,
            'average_confidence': avg_confidence,
            'confidence_range': (min_confidence, max_confidence),
            'group_distribution': {
                'bullish': bullish_groups,
                'bearish': bearish_groups,
                'neutral': neutral_groups
            },
            'total_indicators': total_indicators,
            'average_alignment': avg_alignment,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        
        # Group performance statistics
        group_stats = {}
        for group_name, history in self.group_performance_history.items():
            if history:
                recent_records = list(history)[-50:]  # Last 50 records
                hits = [r['hit'] for r in recent_records]
                errors = [r['error'] for r in recent_records]
                confidences = [r['confidence'] for r in recent_records]
                
                group_stats[group_name] = {
                    'total_records': len(history),
                    'recent_hit_rate': np.mean(hits) if hits else 0.0,
                    'recent_avg_error': np.mean(errors) if errors else 0.0,
                    'recent_avg_confidence': np.mean(confidences) if confidences else 0.0,
                    'current_weights': dict(self.group_weights[group_name])
                }
            else:
                group_stats[group_name] = {
                    'total_records': 0,
                    'recent_hit_rate': 0.0,
                    'recent_avg_error': 0.0,
                    'recent_avg_confidence': 0.0,
                    'current_weights': {}
                }
        
        return {
            'aggregation_method': self.aggregation_method,
            'group_count': len(self.indicator_groups),
            'total_indicator_types': sum(len(indicators) for indicators in self.indicator_groups.values()),
            'processing_stats': self.processing_stats,
            'group_performance': group_stats,
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'min_indicators_per_group': self.min_indicators_per_group,
                'decay_factor': self.decay_factor,
                'max_history': self.max_history
            }
        }

# # Example usage and testing
# if __name__ == "__main__":
#     # This would typically import from Phase 1
#     # For testing, we'll create mock indicator results

#     from IndicatorBlock import IndicatorBlock

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
    
#     indactor_block = IndicatorBlock()

#     indicator_block = {}

#     indicator_results = indactor_block.process_data(sample_data)

#     for name, signal_output in indicator_results.items():
#         indicator_block['name'] = name.upper().replace("_", " ")
#         indicator_block['signal_value'] = signal_output.signal_value
#         indicator_block['confidence'] = signal_output.confidence
#         indicator_block['probability'] = signal_output.probability
#         indicator_block['variance'] = signal_output.variance
#         indicator_block['explanation'] = signal_output.explanation
#         indicator_block['metadata'] = signal_output.metadata


#     print("=== Phase 2: GroupBlock System Test ===\n")
    
#     # # Create mock SignalOutput class for testing
#     # @dataclass
#     # class MockSignalOutput:
#     #     signal_value: float
#     #     confidence: float
#     #     probability: float
#     #     variance: float
#     #     explanation: str
#     #     metadata: Dict[str, Any]
        
#     #     def to_dict(self):
#     #         return asdict(self)
    
#     # Initialize GroupBlock
#     group_block = GroupBlock(
#         aggregation_method='bayesian',
#         confidence_threshold=0.1,
#         min_indicators_per_group=1
#     )
    
#     # # Create mock indicator results (simulating Phase 1 output)
#     # indicator_results = {
#     #     'sma_crossover': MockSignalOutput(
#     #         signal_value=0.3, confidence=0.75, probability=0.65,
#     #         variance=0.001, explanation="SMA crossover bullish",
#     #         metadata={'strength': 0.3}
#     #     ),
#     #     'ema_momentum': MockSignalOutput(
#     #         signal_value=0.2, confidence=0.65, probability=0.58,
#     #         variance=0.0015, explanation="EMA momentum positive",
#     #         metadata={'strength': 0.2}
#     #     ),
#     #     'rsi': MockSignalOutput(
#     #         signal_value=-0.1, confidence=0.5, probability=0.45,
#     #         variance=0.002, explanation="RSI neutral to bearish",
#     #         metadata={'rsi_value': 45}
#     #     ),
#     #     'macd': MockSignalOutput(
#     #         signal_value=0.4, confidence=0.8, probability=0.7,
#     #         variance=0.0008, explanation="MACD strong bullish divergence",
#     #         metadata={'histogram': 0.15}
#     #     ),
#     #     'volume_weighted': MockSignalOutput(
#     #         signal_value=0.15, confidence=0.6, probability=0.55,
#     #         variance=0.0012, explanation="Volume confirmation",
#     #         metadata={'volume_ratio': 1.3}
#     #     ),
#     #     'obv': MockSignalOutput(
#     #         signal_value=0.25, confidence=0.7, probability=0.62,
#     #         variance=0.001, explanation="OBV trending up",
#     #         metadata={'obv_slope': 0.05}
#     #     ),
#     #     'candlestick_patterns': MockSignalOutput(
#     #         signal_value=0.35, confidence=0.85, probability=0.68,
#     #         variance=0.0005, explanation="Bullish engulfing pattern",
#     #         metadata={'pattern': 'bullish_engulfing'}
#     #     ),
#     #     'support_resistance': MockSignalOutput(
#     #         signal_value=0.1, confidence=0.55, probability=0.52,
#     #         variance=0.0018, explanation="Near resistance level",
#     #         metadata={'distance_to_resistance': 0.02}
#     #     )
#     # }
    
#     print("Processing indicator results into groups...")
#     group_results = group_block.process_indicator_results(indicator_results)
    
#     print(f"\n=== Results for {len(group_results)} Groups ===")
    
#     # Display group results
#     for group_name, group_output in group_results.items():
#         print(f"\n{group_name.upper()} GROUP:")
#         print(f"  Signal Value:     {group_output.signal_value:+.3f}")
#         print(f"  Confidence:       {group_output.confidence:.3f}")
#         print(f"  P(up):           {group_output.probability:.3f}")
#         print(f"  Variance:        {group_output.variance:.6f}")
#         print(f"  Component Count: {group_output.component_count}")
#         print(f"  Alignment Score: {group_output.alignment_score:.3f}")
#         print(f"  Dispersion:      {group_output.dispersion:.3f}")
#         print(f"  Explanation:     {group_output.explanation}")
        
#         print("  Weighted Contributions:")
#         for indicator, contribution in group_output.weighted_contribution.items():
#             print(f"    {indicator}: {contribution:.3f}")
    
#     # Generate summary
#     print("\n=== Group Summary ===")
#     summary = group_block.get_group_summary(group_results)
    
#     for key, value in summary.items():
#         if key == 'group_distribution':
#             print("  Group Distribution:")
#             for dist_key, dist_value in value.items():
#                 print(f"    {dist_key}: {dist_value}")
#         elif key == 'confidence_range':
#             print(f"  Confidence Range: {value[0]:.3f} - {value[1]:.3f}")
#         elif isinstance(value, float):
#             print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
#         else:
#             print(f"  {key.replace('_', ' ').title()}: {value}")
    
#     # Test performance updates
#     print("\n=== Testing Performance Updates ===")
    
#     # Simulate actual market movement
#     actual_return = 0.008  # 0.8% positive return
#     print(f"Simulated actual return: {actual_return:+.3f}")
    
#     print("Updating group performance...")
#     group_block.update_group_performance(group_results, actual_return)
    
#     # Test multiple updates to see adaptation
#     print("Testing adaptive learning with multiple updates...")
#     for i in range(5):
#         # Simulate different market outcomes
#         simulated_return = np.random.normal(0.005, 0.01)
#         group_block.update_group_performance(group_results, simulated_return)
    
#     # Show diagnostic report
#     print("\n=== Diagnostic Report ===")
#     diagnostics = group_block.get_diagnostic_report()
    
#     print(f"Aggregation Method: {diagnostics['aggregation_method']}")
#     print(f"Group Count: {diagnostics['group_count']}")
#     print(f"Total Indicator Types: {diagnostics['total_indicator_types']}")
    
#     print("\nProcessing Stats:")
#     for stat_key, stat_value in diagnostics['processing_stats'].items():
#         print(f"  {stat_key}: {stat_value}")
    
#     print("\nGroup Performance:")
#     for group_name, perf_stats in diagnostics['group_performance'].items():
#         print(f"  {group_name}:")
#         print(f"    Records: {perf_stats['total_records']}")
#         print(f"    Hit Rate: {perf_stats['recent_hit_rate']:.3f}")