"""
Phase 3: SystemEngine - System-Level Prediction Engine

This module combines group outputs from Phase 2 into final system-level predictions
using probabilistic fusion, Bayesian inference, and ensemble methods.

The SystemEngine produces:
- Final P(up) probability
- Expected return E[return]
- Return variance Var[return]
- Optional full predictive distribution
- Confidence-weighted system signal
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SystemPrediction:
    """Comprehensive system-level prediction output"""

    # Core predictions
    signal_value: float  # Final system signal [-1, 1]
    probability_up: float  # P(up) - probability of positive return
    expected_return: float  # E[return] - expected return
    return_variance: float  # Var[return] - return variance
    confidence: float  # Overall system confidence [0, 1]

    # Distribution parameters (optional)
    return_distribution: Optional[Dict[str, float]] = None  # mean, std, skew, kurtosis
    quantiles: Optional[Dict[str, float]] = None  # 5%, 25%, 50%, 75%, 95%

    # Component analysis
    group_contributions: Dict[str, float] = None  # How each group contributed
    group_weights: Dict[str, float] = None  # Final group weights used
    alignment_score: float = 0.0  # Agreement between groups
    dispersion_penalty: float = 0.0  # Penalty for high dispersion

    # Metadata
    fusion_method: str = "bayesian"  # Method used for fusion
    groups_used: int = 0  # Number of groups included
    total_indicators: int = 0  # Total indicators across groups
    explanation: str = ""  # Human-readable explanation
    metadata: Dict[str, Any] = None  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def __get__(self, item: str) -> Union[float, Dict[str, float], str]:
        """Get attribute by name"""
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' danda has no attribute '{item}'")

    def get_decision_metrics(self) -> Dict[str, float]:
        """Get key metrics for decision making"""
        return {
            "signal_strength": abs(self.signal_value),
            "directional_confidence": self.probability_up
            if self.signal_value > 0
            else (1 - self.probability_up),
            "risk_adjusted_signal": self.signal_value * self.confidence,
            "expected_sharpe": self.expected_return / np.sqrt(self.return_variance)
            if self.return_variance > 0
            else 0.0,
            "alignment_strength": self.alignment_score,
            "conviction_score": self.confidence * self.alignment_score,
        }


class SystemEngine:
    """
    System-level prediction engine that fuses group outputs into final predictions.

    Supports multiple fusion methods:
    - Bayesian: Combines predictions using Bayesian inference
    - Weighted: Confidence-weighted averaging with dispersion penalty
    - Ensemble: Multiple fusion methods with meta-learning
    - Bootstrap: Bootstrap aggregation for uncertainty estimation
    """

    def __init__(
        self,
        fusion_method: str = "bayesian",
        confidence_threshold: float = 0.2,
        alignment_weight: float = 0.3,
        dispersion_penalty: float = 0.5,
        history_length: int = 100,
        uncertainty_estimation: bool = True,
        calibration_enabled: bool = True,
    ):
        """
        Initialize SystemEngine

        Args:
            fusion_method: Primary method for combining groups ('bayesian', 'weighted', 'ensemble')
            confidence_threshold: Minimum confidence to include group in fusion
            alignment_weight: Weight given to group alignment in final confidence
            dispersion_penalty: Penalty factor for high dispersion between groups
            history_length: Length of performance history to maintain
            uncertainty_estimation: Whether to estimate full uncertainty distribution
            calibration_enabled: Whether to calibrate probabilities using historical data
        """
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold
        self.alignment_weight = alignment_weight
        self.dispersion_penalty = dispersion_penalty
        self.history_length = history_length
        self.uncertainty_estimation = uncertainty_estimation
        self.calibration_enabled = calibration_enabled

        # Performance tracking
        self.prediction_history = deque(maxlen=history_length)
        self.outcome_history = deque(maxlen=history_length)
        self.group_performance = defaultdict(
            lambda: {
                "predictions": deque(maxlen=history_length),
                "outcomes": deque(maxlen=history_length),
                "weights": deque(maxlen=history_length),
                "hit_rate": 0.5,
                "sharpe": 0.0,
                "reliability": 0.5,
            }
        )

        # Calibration mapping
        self.calibration_bins = 20
        self.calibration_map = {}

        # Fusion weights (learned over time)
        self.learned_group_weights = defaultdict(float)
        self.weight_decay = 0.95

        # Statistics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.processing_times = deque(maxlen=50)

        logger.info(f"SystemEngine initialized with fusion_method='{fusion_method}'")

    def fuse_group_predictions(self, group_results: Dict[str, Any]) -> SystemPrediction:
        """
        Main fusion method - combines group outputs into system prediction

        Args:
            group_results: Dictionary of group predictions from Phase 2

        Returns:
            SystemPrediction with fused results
        """
        import time

        start_time = time.time()

        try:
            # Input validation
            if not group_results:
                return self._create_null_prediction("No group results provided")

            # Filter groups by confidence threshold
            valid_groups = {}

            for name, results in group_results.items():
                if results['confidence'] >= self.confidence_threshold:
                    valid_groups[name] = results


            if not valid_groups:
                return self._create_null_prediction(
                    "No groups meet confidence threshold"
                )

            # Apply fusion method
            if self.fusion_method == "bayesian":
                prediction = self._bayesian_fusion(valid_groups)
            elif self.fusion_method == "weighted":
                prediction = self._weighted_fusion(valid_groups)
            elif self.fusion_method == "ensemble":
                prediction = self._ensemble_fusion(valid_groups)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            # Apply calibration if enabled
            if self.calibration_enabled:
                prediction = self._apply_calibration(prediction)

            # Estimate uncertainty distribution if enabled
            if self.uncertainty_estimation:
                prediction = self._estimate_uncertainty(prediction, valid_groups)

            # Generate explanation
            prediction.explanation = self._generate_explanation(
                prediction, valid_groups
            )

            # Update statistics
            self.total_predictions += 1
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            logger.debug(f"System prediction generated in {processing_time:.4f}s")
            
            return prediction

        except Exception as e:
            logger.error(f"Error in fusion: {str(e)}")
            return self._create_null_prediction(f"Fusion error: {str(e)}")

    def _bayesian_fusion(self, groups: Dict[str, Any]) -> SystemPrediction:
        """Bayesian fusion of group predictions"""

        # Extract group data
        signals = []
        confidences = []
        probabilities = []
        variances = []
        weights = []

        group_contributions = {}
        group_weights = {}

        for name, group in groups.items():
            # Get historical performance weight
            hist_weight = self._get_historical_weight(name)

            # Combine confidence with historical performance
            effective_weight = group['confidence'] * hist_weight

            signals.append(group['signal_value'])
            confidences.append(group['confidence'])
            probabilities.append(group['probability'])
            variances.append(getattr(group, "variance", 0.001))
            weights.append(effective_weight)

            group_weights[name] = effective_weight

        signals = np.array(signals)
        confidences = np.array(confidences)
        probabilities = np.array(probabilities)
        variances = np.array(variances)
        weights = np.array(weights)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Bayesian combination
        # Precision-weighted averaging (inverse variance weighting)
        precisions = 1.0 / (variances + 1e-8)
        precision_weights = precisions / precisions.sum()

        # Combined weights (confidence + precision)
        final_weights = 0.7 * weights + 0.3 * precision_weights
        final_weights = final_weights / final_weights.sum()

        # Weighted fusion
        fused_signal = np.sum(final_weights * signals)
        fused_probability = np.sum(final_weights * probabilities)

        # Bayesian variance combination
        fused_variance = np.sum(final_weights**2 * variances)

        # Expected return from signal and probability
        expected_return = fused_signal * 0.01  # Convert signal to return estimate

        # Calculate alignment and dispersion
        alignment_score = self._calculate_alignment(signals, weights)
        dispersion = np.sqrt(np.sum(final_weights * (signals - fused_signal) ** 2))
        dispersion_penalty = np.exp(-self.dispersion_penalty * dispersion)

        # System confidence
        base_confidence = np.sum(final_weights * confidences)
        alignment_bonus = self.alignment_weight * alignment_score
        system_confidence = np.clip(
            base_confidence * dispersion_penalty + alignment_bonus, 0.0, 1.0
        )

        # Store contributions
        for i, (name, group) in enumerate(groups.items()):
            group_contributions[name] = final_weights[i] * signals[i]

        return SystemPrediction(
            signal_value=np.clip(fused_signal, -1.0, 1.0),
            probability_up=np.clip(fused_probability, 0.0, 1.0),
            expected_return=expected_return,
            return_variance=fused_variance,
            confidence=system_confidence,
            group_contributions=group_contributions,
            group_weights={
                name: final_weights[i] for i, name in enumerate(groups.keys())
            },
            alignment_score=alignment_score,
            dispersion_penalty=dispersion_penalty,
            fusion_method="bayesian",
            groups_used=len(groups),
            total_indicators=sum(
                getattr(g, "component_count", 1) for g in groups.values()
            ),
        )

    def _weighted_fusion(self, groups: Dict[str, Any]) -> SystemPrediction:
        """Confidence-weighted fusion with dispersion penalty"""

        signals = []
        confidences = []
        probabilities = []
        group_contributions = {}
        group_weights = {}

        total_weight = 0.0

        for name, group in groups.items():
            # Historical performance adjustment
            hist_performance = self.group_performance[name]["reliability"]
            adjusted_confidence = group['confidence'] * hist_performance

            signals.append(group['signal_value'] * adjusted_confidence)
            confidences.append(adjusted_confidence)
            probabilities.append(group['probability'] * adjusted_confidence)

            group_weights[name] = adjusted_confidence
            total_weight += adjusted_confidence

        # Normalize
        if total_weight > 0:
            fused_signal = sum(signals) / total_weight
            fused_probability = sum(probabilities) / total_weight
            avg_confidence = sum(confidences) / len(confidences)
        else:
            fused_signal = 0.0
            fused_probability = 0.5
            avg_confidence = 0.0

        # Calculate dispersion penalty
        signal_values = [group['signal_value'] for group in groups.values()]
        dispersion = np.std(signal_values) if len(signal_values) > 1 else 0.0
        dispersion_penalty = np.exp(-self.dispersion_penalty * dispersion)

        # Calculate alignment
        weights_array = np.array(list(group_weights.values()))
        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        alignment_score = self._calculate_alignment(
            np.array(signal_values), weights_array
        )

        # Final confidence with dispersion penalty
        system_confidence = avg_confidence * dispersion_penalty

        # Calculate contributions
        for i, (name, group) in enumerate(groups.items()):
            group_contributions[name] = (
                group['signal_value'] * group_weights[name] / total_weight
                if total_weight > 0
                else 0.0
            )

        # Expected return and variance
        expected_return = fused_signal * 0.01
        return_variance = (dispersion**2 + 0.0001) * (1 - system_confidence)

        return SystemPrediction(
            signal_value=np.clip(fused_signal, -1.0, 1.0),
            probability_up=np.clip(fused_probability, 0.0, 1.0),
            expected_return=expected_return,
            return_variance=return_variance,
            confidence=np.clip(system_confidence, 0.0, 1.0),
            group_contributions=group_contributions,
            group_weights=group_weights,
            alignment_score=alignment_score,
            dispersion_penalty=dispersion_penalty,
            fusion_method="weighted",
            groups_used=len(groups),
            total_indicators=sum(
                getattr(g, "component_count", 1) for g in groups.values()
            ),
        )


    def _ensemble_fusion(self, groups: Dict[str, Any]) -> SystemPrediction:
        """Ensemble fusion using multiple methods"""

        # Get predictions from different methods
        bayesian_pred = self._bayesian_fusion(groups)
        weighted_pred = self._weighted_fusion(groups)

        # Ensemble weights based on historical performance
        bayesian_weight = 0.6  # Could be learned
        weighted_weight = 0.4

        # Combine predictions
        fused_signal = (
            bayesian_weight * bayesian_pred.signal_value
            + weighted_weight * weighted_pred.signal_value
        )

        fused_probability = (
            bayesian_weight * bayesian_pred.probability_up
            + weighted_weight * weighted_pred.probability_up
        )

        fused_confidence = (
            bayesian_weight * bayesian_pred.confidence
            + weighted_weight * weighted_pred.confidence
        )

        expected_return = (
            bayesian_weight * bayesian_pred.expected_return
            + weighted_weight * weighted_pred.expected_return
        )

        return_variance = (
            bayesian_weight * bayesian_pred.return_variance
            + weighted_weight * weighted_pred.return_variance
        )

        # Use Bayesian components for other metrics
        return SystemPrediction(
            signal_value=np.clip(fused_signal, -1.0, 1.0),
            probability_up=np.clip(fused_probability, 0.0, 1.0),
            expected_return=expected_return,
            return_variance=return_variance,
            confidence=np.clip(fused_confidence, 0.0, 1.0),
            group_contributions=bayesian_pred.group_contributions,
            group_weights=bayesian_pred.group_weights,
            alignment_score=bayesian_pred.alignment_score,
            dispersion_penalty=bayesian_pred.dispersion_penalty,
            fusion_method="ensemble",
            groups_used=len(groups),
            total_indicators=sum(
                getattr(g, "component_count", 1) for g in groups.values()
            ),
        )

    def _calculate_alignment(self, signals: np.ndarray, weights: np.ndarray) -> float:
        """Calculate alignment score between signals"""
        if len(signals) < 2:
            return 1.0

        # Weighted correlation with consensus
        weighted_mean = np.sum(weights * signals)
        deviations = signals - weighted_mean
        weighted_var = np.sum(weights * deviations**2)

        if weighted_var == 0:
            return 1.0

        # Convert to alignment score [0, 1]
        alignment = np.exp(-weighted_var)
        return np.clip(alignment, 0.0, 1.0)

    def _get_historical_weight(self, group_name: str) -> float:
        """Get historical performance weight for a group"""
        perf = self.group_performance[group_name]

        if len(perf["outcomes"]) < 5:  # Not enough history
            return 1.0

        # Combine hit rate and Sharpe ratio
        hit_rate_weight = 2 * perf["hit_rate"]  # 0-2 scale
        sharpe_weight = np.tanh(perf["sharpe"])  # -1 to 1, then scale
        reliability_weight = perf["reliability"]

        # Weighted combination
        historical_weight = (
            0.4 * hit_rate_weight + 0.3 * sharpe_weight + 0.3 * reliability_weight
        )

        return np.clip(
            historical_weight, 0.1, 2.0
        )  # Don't completely zero out any group

    def _apply_calibration(self, prediction: SystemPrediction) -> SystemPrediction:
        """Apply probability calibration using historical data"""
        if len(self.prediction_history) < 20:  # Not enough data for calibration
            return prediction

        # Find the calibration bin
        prob_bin = int(prediction.probability_up * self.calibration_bins)
        prob_bin = np.clip(prob_bin, 0, self.calibration_bins - 1)

        if prob_bin in self.calibration_map:
            calibrated_prob = self.calibration_map[prob_bin]
            prediction.probability_up = calibrated_prob

        return prediction

    def _estimate_uncertainty(
        self, prediction: SystemPrediction, groups: Dict[str, Any]
    ) -> SystemPrediction:
        """Estimate full uncertainty distribution"""

        # Bootstrap-like uncertainty estimation
        signals = [group['signal_value'] for group in groups.values()]
        confidences = [group['confidence'] for group in groups.values()]

        if len(signals) < 2:
            # Simple normal distribution
            prediction.return_distribution = {
                "mean": prediction.expected_return,
                "std": np.sqrt(prediction.return_variance),
                "skew": 0.0,
                "kurtosis": 3.0,
            }
        else:
            # More complex distribution based on signal dispersion
            mean_return = prediction.expected_return
            std_return = np.sqrt(prediction.return_variance)

            # Add skewness based on signal asymmetry
            signal_skew = stats.skew(signals) if len(signals) > 2 else 0.0

            prediction.return_distribution = {
                "mean": mean_return,
                "std": std_return,
                "skew": signal_skew * 0.5,  # Dampen skewness
                "kurtosis": 3.0 + abs(signal_skew),  # Slight leptokurtosis with skew
            }

        # Calculate quantiles
        mean = prediction.return_distribution["mean"]
        std = prediction.return_distribution["std"]

        prediction.quantiles = {
            "5%": mean + std * stats.norm.ppf(0.05),
            "25%": mean + std * stats.norm.ppf(0.25),
            "50%": mean,  # median = mean for normal
            "75%": mean + std * stats.norm.ppf(0.75),
            "95%": mean + std * stats.norm.ppf(0.95),
        }

        return prediction

    def _generate_explanation(
        self, prediction: SystemPrediction, groups: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of the prediction"""

        signal_strength = (
            "strong"
            if abs(prediction.signal_value) > 0.5
            else "moderate"
            if abs(prediction.signal_value) > 0.2
            else "weak"
        )
        direction = "bullish" if prediction.signal_value > 0 else "bearish"
        confidence_level = (
            "high"
            if prediction.confidence > 0.7
            else "medium"
            if prediction.confidence > 0.4
            else "low"
        )

        # Find top contributing groups
        contributions = prediction.group_contributions or {}
        top_groups = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )[:2]

        explanation = f"{signal_strength.title()} {direction} signal (confidence: {confidence_level}) "

        if top_groups:
            group_names = [name.replace("_", " ").title() for name, _ in top_groups]
            explanation += f"driven by {' and '.join(group_names)} groups. "

        explanation += f"Alignment score: {prediction.alignment_score:.2f}. "

        if prediction.probability_up > 0.6:
            explanation += f"High probability ({prediction.probability_up:.1%}) of positive return."
        elif prediction.probability_up < 0.4:
            explanation += f"High probability ({1 - prediction.probability_up:.1%}) of negative return."
        else:
            explanation += "Neutral probability assessment."

        return explanation

    def _create_null_prediction(self, reason: str) -> SystemPrediction:
        """Create a null/neutral prediction"""
        return SystemPrediction(
            signal_value=0.0,
            probability_up=0.5,
            expected_return=0.0,
            return_variance=0.01,
            confidence=0.0,
            group_contributions={},
            group_weights={},
            alignment_score=0.0,
            dispersion_penalty=1.0,
            fusion_method=self.fusion_method,
            groups_used=0,
            total_indicators=0,
            explanation=f"Null prediction: {reason}",
        )

    def update_performance(self, prediction: SystemPrediction, actual_return: float):
        """Update system performance with actual outcome"""

        # Store prediction and outcome
        self.prediction_history.append(
            {
                "signal": prediction.signal_value,
                "probability": prediction.probability_up,
                "expected_return": prediction.expected_return,
                "confidence": prediction.confidence,
            }
        )
        self.outcome_history.append(actual_return)

        # Update success tracking
        predicted_direction = 1 if prediction.signal_value > 0 else -1
        actual_direction = 1 if actual_return > 0 else -1

        if predicted_direction == actual_direction:
            self.successful_predictions += 1

        # Update group performance
        if prediction.group_contributions:
            for group_name, contribution in prediction.group_contributions.items():
                group_perf = self.group_performance[group_name]

                # Predict group direction
                group_direction = 1 if contribution > 0 else -1
                group_correct = group_direction == actual_direction

                group_perf["predictions"].append(contribution)
                group_perf["outcomes"].append(actual_return)

                # Update hit rate
                if len(group_perf["outcomes"]) > 0:
                    recent_predictions = list(group_perf["predictions"])[
                        -20:
                    ]  # Last 20
                    recent_outcomes = list(group_perf["outcomes"])[-20:]

                    hits = sum(
                        1
                        for p, o in zip(recent_predictions, recent_outcomes)
                        if (p > 0 and o > 0) or (p < 0 and o < 0)
                    )
                    group_perf["hit_rate"] = hits / len(recent_outcomes)

                # Update reliability (exponential moving average)
                alpha = 0.1
                if group_correct:
                    group_perf["reliability"] = (1 - alpha) * group_perf[
                        "reliability"
                    ] + alpha * 1.0
                else:
                    group_perf["reliability"] = (1 - alpha) * group_perf[
                        "reliability"
                    ] + alpha * 0.0

        # Update calibration mapping
        self._update_calibration(prediction.probability_up, actual_return > 0)

        logger.debug(
            f"Performance updated: prediction={prediction.signal_value:.3f}, "
            f"actual={actual_return:.3f}, success_rate={self.get_success_rate():.3f}"
        )

    def _update_calibration(self, predicted_prob: float, actual_positive: bool):
        """Update probability calibration mapping"""
        prob_bin = int(predicted_prob * self.calibration_bins)
        prob_bin = np.clip(prob_bin, 0, self.calibration_bins - 1)

        if prob_bin not in self.calibration_map:
            self.calibration_map[prob_bin] = {
                "predictions": [],
                "outcomes": [],
                "calibrated_prob": None  # optional, if you want default
            }

        self.calibration_map[prob_bin]["predictions"].append(predicted_prob)
        self.calibration_map[prob_bin]["outcomes"].append(
            1.0 if actual_positive else 0.0
        )

        # Update calibrated probability (moving average)
        outcomes = self.calibration_map[prob_bin]["outcomes"][-50:]  # Last 50
        if len(outcomes) >= 5:
            calibrated_prob = np.mean(outcomes)
            self.calibration_map[prob_bin]["calibrated_prob"] = calibrated_prob


    def get_success_rate(self) -> float:
        """Get overall success rate"""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic report"""

        avg_processing_time = (
            np.mean(self.processing_times) if self.processing_times else 0.0
        )

        # Recent performance
        recent_predictions = list(self.prediction_history)[-20:]
        recent_outcomes = list(self.outcome_history)[-20:]

        recent_hit_rate = 0.0
        recent_sharpe = 0.0

        if len(recent_outcomes) > 0:
            # Hit rate
            hits = sum(
                1
                for pred, outcome in zip(recent_predictions, recent_outcomes)
                if (pred["signal"] > 0 and outcome > 0)
                or (pred["signal"] < 0 and outcome < 0)
            )
            recent_hit_rate = hits / len(recent_outcomes)

            # Sharpe ratio
            returns = [pred["expected_return"] for pred in recent_predictions]
            if len(returns) > 1 and np.std(returns) > 0:
                recent_sharpe = np.mean(returns) / np.std(returns)

        return {
            "fusion_method": self.fusion_method,
            "total_predictions": self.total_predictions,
            "success_rate": self.get_success_rate(),
            "recent_hit_rate": recent_hit_rate,
            "recent_sharpe": recent_sharpe,
            "avg_processing_time": avg_processing_time,
            "calibration_bins": len(self.calibration_map),
            "group_performance": {
                name: {
                    "hit_rate": perf["hit_rate"],
                    "reliability": perf["reliability"],
                    "prediction_count": len(perf["predictions"]),
                }
                for name, perf in self.group_performance.items()
            },
            "prediction_history_length": len(self.prediction_history),
            "confidence_threshold": self.confidence_threshold,
            "alignment_weight": self.alignment_weight,
            "dispersion_penalty": self.dispersion_penalty,
        }
    

# # Example usage and testing
# if __name__ == "__main__":
#     # Import Phase 1 and 2 components
#     # from phase1_indicator_block import IndicatorBlock
#     # from phase2_group_block import GroupBlock

#     print("=== Phase 3: SystemEngine Test ===\n")

#     # Create mock GroupOutput class for testing (normally imported from Phase 2)
#     @dataclass
#     class MockGroupOutput:
#         signal_value: float
#         confidence: float
#         probability: float
#         variance: float
#         component_count: int
#         alignment_score: float
#         dispersion: float
#         explanation: str
#         weighted_contribution: Dict[str, float]

#     # Initialize SystemEngine
#     system_engine = SystemEngine(
#         fusion_method="bayesian",
#         confidence_threshold=0.2,
#         alignment_weight=0.3,
#         dispersion_penalty=0.5,
#         uncertainty_estimation=True,
#         calibration_enabled=True,
#     )

#     # Create mock group results (simulating Phase 2 output)
#     mock_group_results = {
#         "trend": MockGroupOutput(
#             signal_value=0.4,
#             confidence=0.8,
#             probability=0.7,
#             variance=0.001,
#             component_count=3,
#             alignment_score=0.85,
#             dispersion=0.1,
#             explanation="Strong trend alignment",
#             weighted_contribution={"sma": 0.15, "ema": 0.2, "macd": 0.25},
#         ),
#         "momentum": MockGroupOutput(
#             signal_value=0.2,
#             confidence=0.6,
#             probability=0.6,
#             variance=0.002,
#             component_count=2,
#             alignment_score=0.7,
#             dispersion=0.15,
#             explanation="Moderate momentum",
#             weighted_contribution={"rsi": -0.05, "stoch": 0.25},
#         ),
#         "volume": MockGroupOutput(
#             signal_value=0.3,
#             confidence=0.7,
#             probability=0.65,
#             variance=0.0015,
#             component_count=2,
#             alignment_score=0.8,
#             dispersion=0.08,
#             explanation="Volume confirmation",
#             weighted_contribution={"obv": 0.15, "vwap": 0.15},
#         ),
#         "price_action": MockGroupOutput(
#             signal_value=0.5,
#             confidence=0.9,
#             probability=0.75,
#             variance=0.0008,
#             component_count=2,
#             alignment_score=0.9,
#             dispersion=0.05,
#             explanation="Strong bullish pattern",
#             weighted_contribution={"engulfing": 0.3, "support": 0.2},
#         ),
#     }

#     print("Processing group results into system prediction...")
#     system_prediction = system_engine.fuse_group_predictions(mock_group_results)

#     print("\n=== System Prediction Results ===")
#     print(f"Signal Value:        {system_prediction.signal_value:+.3f}")
#     print(
#         f"Probability Up:      {system_prediction.probability_up:.3f} ({system_prediction.probability_up:.1%})"
#     )
#     print(
#         f"Expected Return:     {system_prediction.expected_return:+.4f} ({system_prediction.expected_return * 100:+.2f}%)"
#     )
#     print(f"Return Variance:     {system_prediction.return_variance:.6f}")
#     print(f"System Confidence:   {system_prediction.confidence:.3f}")
#     print(f"Alignment Score:     {system_prediction.alignment_score:.3f}")
#     print(f"Dispersion Penalty:  {system_prediction.dispersion_penalty:.3f}")
#     print(f"Groups Used:         {system_prediction.groups_used}")
#     print(f"Total Indicators:    {system_prediction.total_indicators}")
#     print(f"Fusion Method:       {system_prediction.fusion_method}")
#     print(f"Explanation:         {system_prediction.explanation}")

#     # Show group contributions
#     print("\n=== Group Contributions ===")
#     for group, contribution in system_prediction.group_contributions.items():
#         weight = system_prediction.group_weights[group]
#         print(
#             f"  {group.title()}: Contribution={contribution:+.3f}, Weight={weight:.3f}"
#         )

#     # Show decision metrics
#     print("\n=== Decision Metrics ===")
#     decision_metrics = system_prediction.get_decision_metrics()
#     for metric, value in decision_metrics.items():
#         print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")

#     # Show uncertainty distribution if available
#     if system_prediction.return_distribution:
#         print("\n=== Return Distribution ===")
#         dist = system_prediction.return_distribution
#         print(f"  Mean:     {dist['mean']:+.4f}")
#         print(f"  Std Dev:  {dist['std']:.4f}")
#         print(f"  Skewness: {dist['skew']:+.3f}")
#         print(f"  Kurtosis: {dist['kurtosis']:.3f}")

#         if system_prediction.quantiles:
#             print("\n  Quantiles:")
#             for q, value in system_prediction.quantiles.items():
#                 print(f"    {q:>3}: {value:+.4f} ({value * 100:+.2f}%)")

#     # Test different fusion methods
#     print("\n=== Testing Different Fusion Methods ===")

#     fusion_methods = ["bayesian", "weighted", "ensemble"]
#     results_comparison = {}

#     for method in fusion_methods:
#         test_engine = SystemEngine(fusion_method=method)
#         result = test_engine.fuse_group_predictions(mock_group_results)
#         results_comparison[method] = result

#         print(f"\n{method.upper()} METHOD:")
#         print(
#             f"  Signal: {result.signal_value:+.3f}, P(up): {result.probability_up:.3f}, "
#             f"Confidence: {result.confidence:.3f}"
#         )

#     # Test performance tracking
#     print("\n=== Testing Performance Updates ===")

#     # Simulate multiple predictions and outcomes
#     np.random.seed(42)
#     for i in range(10):
#         # Add some variation to group results
#         varied_groups = {}
#         for name, group in mock_group_results.items():
#             noise = np.random.normal(0, 0.1)
#             varied_groups[name] = MockGroupOutput(
#                 signal_value=np.clip(group.signal_value + noise, -1, 1),
#                 confidence=np.clip(group.confidence + noise * 0.1, 0, 1),
#                 probability=np.clip(group.probability + noise * 0.05, 0, 1),
#                 variance=group.variance,
#                 component_count=group.component_count,
#                 alignment_score=group.alignment_score,
#                 dispersion=group.dispersion,
#                 explanation=group.explanation,
#                 weighted_contribution=group.weighted_contribution,
#             )

#         # Get prediction
#         pred = system_engine.fuse_group_predictions(varied_groups)

#         # Simulate actual market outcome
#         actual_return = np.random.normal(
#             pred.expected_return, np.sqrt(pred.return_variance)
#         )

#         # Update performance
#         system_engine.update_performance(pred, actual_return)

#         print(
#             f"  Update {i + 1}: Predicted={pred.signal_value:+.3f}, "
#             f"Actual Return={actual_return:+.4f}, "
#             f"Success Rate={system_engine.get_success_rate():.3f}"
#         )

#     # Show diagnostic report
#     print("\n=== System Diagnostic Report ===")
#     diagnostics = system_engine.get_diagnostic_report()

#     print(f"Fusion Method:           {diagnostics['fusion_method']}")
#     print(f"Total Predictions:       {diagnostics['total_predictions']}")
#     print(f"Overall Success Rate:    {diagnostics['success_rate']:.3f}")
#     print(f"Recent Hit Rate:         {diagnostics['recent_hit_rate']:.3f}")
#     print(f"Recent Sharpe Ratio:     {diagnostics['recent_sharpe']:+.3f}")
#     print(f"Avg Processing Time:     {diagnostics['avg_processing_time']:.4f}s")
#     print(f"Calibration Bins:        {diagnostics['calibration_bins']}")
#     print(
#         f"Prediction History:      {diagnostics['prediction_history_length']} records"
#     )

#     print("\nGroup Performance:")
#     for group_name, perf in diagnostics["group_performance"].items():
#         print(f"  {group_name.title()}:")
#         print(f"    Hit Rate:         {perf['hit_rate']:.3f}")
#         print(f"    Reliability:      {perf['reliability']:.3f}")
#         print(f"    Predictions:      {perf['prediction_count']}")

#     # Test edge cases
#     print("\n=== Testing Edge Cases ===")

#     # Test with no groups
#     empty_prediction = system_engine.fuse_group_predictions({})
#     print(
#         f"Empty input: Signal={empty_prediction.signal_value}, "
#         f"Confidence={empty_prediction.confidence}"
#     )

#     # Test with low confidence groups
#     low_conf_groups = {
#         name: MockGroupOutput(
#             signal_value=group.signal_value,
#             confidence=0.1,  # Below threshold
#             probability=group.probability,
#             variance=group.variance,
#             component_count=group.component_count,
#             alignment_score=group.alignment_score,
#             dispersion=group.dispersion,
#             explanation=group.explanation,
#             weighted_contribution=group.weighted_contribution,
#         )
#         for name, group in mock_group_results.items()
#     }

#     low_conf_prediction = system_engine.fuse_group_predictions(low_conf_groups)
#     print(
#         f"Low confidence groups: Signal={low_conf_prediction.signal_value}, "
#         f"Confidence={low_conf_prediction.confidence}"
#     )

#     # Test signal validation
#     print("\n=== Signal Validation ===")
#     validation_passed = True

#     test_prediction = system_engine.fuse_group_predictions(mock_group_results)

#     # Check bounds
#     if not (-1 <= test_prediction.signal_value <= 1):
#         print(f"ERROR: Signal {test_prediction.signal_value} outside [-1, 1]")
#         validation_passed = False

#     if not (0 <= test_prediction.confidence <= 1):
#         print(f"ERROR: Confidence {test_prediction.confidence} outside [0, 1]")
#         validation_passed = False

#     if not (0 <= test_prediction.probability_up <= 1):
#         print(f"ERROR: Probability {test_prediction.probability_up} outside [0, 1]")
#         validation_passed = False

#     if test_prediction.return_variance < 0:
#         print(f"ERROR: Variance {test_prediction.return_variance} is negative")
#         validation_passed = False

#     if validation_passed:
#         print("✅ All predictions passed validation checks")
#     else:
#         print("❌ Some predictions failed validation")

#     # Test serialization
#     print("\n=== Testing Serialization ===")
#     prediction_dict = test_prediction.to_dict()
#     print(f"Serialized keys: {list(prediction_dict.keys())}")
#     print(f"Serialization successful: {'signal_value' in prediction_dict}")

#     # Performance benchmark
#     print("\n=== Performance Benchmark ===")
#     import time

#     start_time = time.time()
#     for _ in range(100):
#         benchmark_prediction = system_engine.fuse_group_predictions(mock_group_results)
#     end_time = time.time()

#     avg_time = (end_time - start_time) / 100
#     predictions_per_second = 1.0 / avg_time if avg_time > 0 else float("inf")

#     print(f"Average processing time:     {avg_time:.4f} seconds")
#     print(f"Predictions per second:      {predictions_per_second:.1f}")
#     print(f"Real-time capable:           {'✅' if avg_time < 0.01 else '⚠️'}")

#     # Export sample results for downstream testing
#     print("\n=== Export Sample Results ===")

#     export_data = {
#         "timestamp": pd.Timestamp.now().isoformat(),
#         "system_prediction": test_prediction.to_dict(),
#         "fusion_comparison": {
#             method: result.to_dict() for method, result in results_comparison.items()
#         },
#         "diagnostics": diagnostics,
#         "decision_metrics": decision_metrics,
#     }

#     print("Export structure:")
#     print(f"  - Timestamp: {export_data['timestamp']}")
#     print(f"  - Prediction keys: {len(export_data['system_prediction'])}")
#     print(
#         f"  - Fusion methods compared: {list(export_data['fusion_comparison'].keys())}"
#     )
#     print(f"  - Diagnostic categories: {len(export_data['diagnostics'])}")

#     print("\n🎯 Phase 3 Complete: SystemEngine Successfully Built")
#     print("   - Probabilistic fusion of group predictions")
#     print("   - Multiple fusion methods (Bayesian, Weighted, Ensemble)")
#     print("   - Uncertainty estimation and distribution modeling")
#     print("   - Performance tracking and adaptive learning")
#     print("   - Probability calibration")
#     print("   - Comprehensive diagnostics and explainability")
#     print("   - Real-time performance: {predictions_per_second:.1f} predictions/sec")

#     print("\n📋 Ready for Phase 4: Risk Controller and Decision Engine")
#     print("   - System provides final P(up), E[return], Var[return]")
#     print("   - Confidence-weighted signals available")
#     print("   - Decision metrics computed for position sizing")
#     print("   - Performance history available for risk adjustment")
    
#     print("\n" + "=" * 60)
#     print("PHASE 3 SYSTEM ENGINE: FULLY OPERATIONAL")
#     print("=" * 60)

