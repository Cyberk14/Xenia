import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
from typing import Dict, Any, List
from enum import Enum

# Configure page
st.set_page_config(
    page_title="XENIA: The Modular Trading System",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .module-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "system_trained" not in st.session_state:
    st.session_state.system_trained = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "live_signal" not in st.session_state:
    st.session_state.live_signal = None
if "system_config" not in st.session_state:
    st.session_state.system_config = {}

# Main title
st.markdown(
    '<h1 class="main-header">🧱 XENIA: The Modular Trading System</h1>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Sidebar - Module Selection
st.sidebar.header("🔧 System Configuration")

# Feature Extractors
st.sidebar.subheader("📊 Feature Extractors")
feature_extractors = st.sidebar.multiselect(
    "Select Feature Extractors:",
    ["TechnicalFeatureExtractor", "PriceActionFeatureExtractor"],
    default=["TechnicalFeatureExtractor"],
)

# Labeling Strategies
st.sidebar.subheader("🏷️ Labeling Strategy")
labeling_strategy = st.sidebar.selectbox(
    "Select Labeling Strategy:", ["ReturnBasedLabeling", "VolatilityAdjustedLabeling"]
)

# Labeling Strategy Parameters
if labeling_strategy == "ReturnBasedLabeling":
    lookforward_periods = st.sidebar.slider(
        "Lookforward Periods:", min_value=1, max_value=20, value=5
    )
    labeling_params = {"lookforward_periods": lookforward_periods}
else:  # VolatilityAdjustedLabeling
    vol_window = st.sidebar.slider(
        "Volatility Window:", min_value=5, max_value=50, value=20
    )
    vol_threshold = st.sidebar.slider(
        "Volatility Threshold:", min_value=0.1, max_value=2.0, value=0.5, step=0.1
    )
    labeling_params = {"window": vol_window, "threshold": vol_threshold}

# ML Models
st.sidebar.subheader("🤖 ML Models")
model_types = st.sidebar.multiselect(
    "Select Model Types:",
    ["Random Forest", "Gradient Boosting", "MLP"],
    default=["Random Forest"],
)

# Model Parameters
model_configs = {}
for model_type in model_types:
    st.sidebar.markdown(f"**{model_type} Parameters:**")
    if model_type == "Random Forest":
        n_estimators = st.sidebar.slider(
            f"RF - N Estimators:",
            min_value=10,
            max_value=200,
            value=100,
            key="rf_estimators",
        )
        model_configs["Random Forest"] = {"n_estimators": n_estimators}
    elif model_type == "Gradient Boosting":
        n_estimators = st.sidebar.slider(
            f"GB - N Estimators:",
            min_value=10,
            max_value=200,
            value=100,
            key="gb_estimators",
        )
        model_configs["Gradient Boosting"] = {"n_estimators": n_estimators}
    elif model_type == "MLP":
        layer1 = st.sidebar.slider(
            "MLP - Layer 1 Size:",
            min_value=10,
            max_value=200,
            value=100,
            key="mlp_layer1",
        )
        layer2 = st.sidebar.slider(
            "MLP - Layer 2 Size:",
            min_value=10,
            max_value=200,
            value=50,
            key="mlp_layer2",
        )
        model_configs["MLP"] = {"hidden_layer_sizes": (layer1, layer2)}

# Ensemble Option
use_ensemble = st.sidebar.checkbox("Use Ensemble Model")
if use_ensemble and len(model_types) > 1:
    st.sidebar.markdown("**Ensemble Weights:**")
    ensemble_weights = []
    for i, model_type in enumerate(model_types):
        weight = st.sidebar.slider(
            f"{model_type} Weight:",
            min_value=0.1,
            max_value=1.0,
            value=1.0 / len(model_types),
            step=0.1,
            key=f"weight_{i}",
        )
        ensemble_weights.append(weight)
    # Normalize weights
    total_weight = sum(ensemble_weights)
    ensemble_weights = [w / total_weight for w in ensemble_weights]

# Backtester
st.sidebar.subheader("📈 Backtester")
backtester_type = st.sidebar.selectbox(
    "Select Backtester:", ["WalkForwardBacktester", "MonteCarloBacktester"]
)

if backtester_type == "MonteCarloBacktester":
    n_simulations = st.sidebar.slider(
        "Number of Simulations:", min_value=100, max_value=1000, value=500
    )
    backtester_params = {"n_simulations": n_simulations}
else:
    backtester_params = {}

# Simulator
st.sidebar.subheader("⚡ Simulator")
simulator_type = st.sidebar.selectbox("Select Simulator:", ["EventBasedSimulator"])

initial_capital = st.sidebar.number_input(
    "Initial Capital ($):", min_value=1000, max_value=1000000, value=100000, step=1000
)

# Portfolio Manager
st.sidebar.subheader("💼 Portfolio Manager")
portfolio_manager = st.sidebar.selectbox(
    "Select Portfolio Manager:",
    ["FixedSizePortfolioManager", "RiskAdjustedPortfolioManager"],
)

# Trading Symbols
st.sidebar.subheader("📊 Trading Symbols")
symbols_input = st.sidebar.text_input(
    "Enter symbols (comma-separated):", value="AAPL,GOOGL,MSFT,TSLA,AMZN"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # System Configuration Summary
    st.header("🔧 System Configuration")

    config_data = {
        "Feature Extractors": feature_extractors,
        "Labeling Strategy": f"{labeling_strategy}({labeling_params})",
        "ML Models": [
            f"{model} ({model_configs.get(model, {})})" for model in model_types
        ],
        "Ensemble": f"Enabled (weights: {ensemble_weights})"
        if use_ensemble and len(model_types) > 1
        else "Disabled",
        "Backtester": f"{backtester_type}({backtester_params})",
        "Simulator": f"{simulator_type}(initial_capital=${initial_capital:,})",
        "Portfolio Manager": portfolio_manager,
        "Trading Symbols": symbols,
    }

    st.session_state.system_config = config_data

    # Display config in a nice format
    for key, value in config_data.items():
        st.markdown(f"**{key}:** {value}")

    # Sample Data Preview
    st.header("📊 Sample Data Preview")

    @st.cache_data
    def create_sample_data_preview():
        """Create sample data for preview"""
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)

        data = []
        for symbol in symbols[:3]:  # Limit to first 3 symbols for preview
            price = 100
            for date in dates:
                price += np.random.normal(0, 2)
                price = max(price, 10)  # Minimum price
                data.append(
                    {
                        "Date": date,
                        "Symbol": symbol,
                        "Open": price + np.random.normal(0, 1),
                        "High": price + abs(np.random.normal(0, 2)),
                        "Low": price - abs(np.random.normal(0, 2)),
                        "Close": price,
                        "Volume": np.random.randint(1000000, 10000000),
                    }
                )

        return pd.DataFrame(data)

    sample_data = create_sample_data_preview()
    st.dataframe(sample_data.tail(10), use_container_width=True)

with col2:
    # Control Panel
    st.header("🎮 Control Panel")

    # Training
    st.subheader("🏋️ Training")
    if st.button("🚀 Train System", type="primary", use_container_width=True):
        with st.spinner("Training system..."):
            # Simulate training process
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                # In real implementation, this would call:
                # system.train_system(data)

            st.session_state.system_trained = True
            st.success("✅ System trained successfully!")

    # Backtesting
    st.subheader("📊 Backtesting")
    if st.button(
        "📈 Run Backtest",
        use_container_width=True,
        disabled=not st.session_state.system_trained,
    ):
        if st.session_state.system_trained:
            with st.spinner("Running backtest..."):
                # Simulate backtest results
                # In real implementation, this would call the backtester
                backtest_results = {
                    "Total Return": "15.2%",
                    "Sharpe Ratio": "1.34",
                    "Max Drawdown": "-8.5%",
                    "Win Rate": "58.3%",
                    "Number of Trades": 142,
                    "Average Trade": "0.8%",
                }
                st.session_state.backtest_results = backtest_results
                st.success("✅ Backtest completed!")
        else:
            st.error("❌ Please train the system first!")

    # Live Signal
    st.subheader("📡 Live Signal")
    if st.button(
        "🔍 Generate Live Signal",
        use_container_width=True,
        disabled=not st.session_state.system_trained,
    ):
        if st.session_state.system_trained:
            with st.spinner("Generating live signal..."):
                # Simulate live signal generation
                # In real implementation, this would call generate_live_signal
                live_signals = {}
                for symbol in symbols:
                    signal = np.random.choice(
                        ["HOLD", "LONG", "SHORT"], p=[0.6, 0.2, 0.2]
                    )
                    confidence = np.random.uniform(0.5, 0.95)
                    live_signals[symbol] = {
                        "signal": signal,
                        "confidence": f"{confidence:.2f}",
                    }

                st.session_state.live_signal = live_signals
                st.success("✅ Live signals generated!")
        else:
            st.error("❌ Please train the system first!")

    # Live Trading
    st.subheader("🔴 Live Trading")
    if st.button(
        "🚨 Start Live Trading",
        use_container_width=True,
        disabled=not st.session_state.system_trained,
    ):
        if st.session_state.system_trained:
            st.warning("⚠️ Live trading would start here!")
            st.info("In production, this would call: `await main(symbols)`")
            # In real implementation:
            # asyncio.run(main(symbols))
        else:
            st.error("❌ Please train the system first!")

# Results Display
st.markdown("---")

# Backtest Results
if st.session_state.backtest_results:
    st.header("📊 Backtest Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Return", st.session_state.backtest_results["Total Return"])
        st.metric("Win Rate", st.session_state.backtest_results["Win Rate"])

    with col2:
        st.metric("Sharpe Ratio", st.session_state.backtest_results["Sharpe Ratio"])
        st.metric(
            "Number of Trades", st.session_state.backtest_results["Number of Trades"]
        )

    with col3:
        st.metric("Max Drawdown", st.session_state.backtest_results["Max Drawdown"])
        st.metric("Average Trade", st.session_state.backtest_results["Average Trade"])

# Live Signals
if st.session_state.live_signal:
    st.header("📡 Current Live Signals")

    signal_df = pd.DataFrame.from_dict(st.session_state.live_signal, orient="index")
    signal_df.index.name = "Symbol"
    signal_df = signal_df.reset_index()

    # Color code signals
    def color_signal(val):
        if val == "LONG":
            return "background-color: #d4edda; color: #155724"
        elif val == "SHORT":
            return "background-color: #f8d7da; color: #721c24"
        else:
            return "background-color: #fff3cd; color: #856404"

    styled_df = signal_df.style.applymap(color_signal, subset=["signal"])
    st.dataframe(styled_df, use_container_width=True)

# System Internals (Expandable)
with st.expander("🔍 System Internals & Architecture"):
    st.subheader("📋 Enums")

    col1, col2 = st.columns(2)

    with col1:
        st.code(
            """
# Signal Types
class Signal(Enum):
    HOLD = "HOLD"
    LONG = "LONG"
    SHORT = "SHORT"

# Model Types
class ModelType(Enum):
    RANDOM_FOREST = "RANDOM_FOREST"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"
    MLP = "MLP"
    CUSTOM = "CUSTOM"
        """,
            language="python",
        )

    with col2:
        st.code(
            """
# Backtest Types
class BacktestType(Enum):
    CLASSIC_CV = "CLASSIC_CV"
    WALK_FORWARD = "WALK_FORWARD"
    MONTE_CARLO = "MONTE_CARLO"

# Simulation Types
class SimulationType(Enum):
    EVENT_BASED = "EVENT_BASED"
    BAR_BASED = "BAR_BASED"
    HYBRID = "HYBRID"
        """,
            language="python",
        )

    st.subheader("🏗️ Abstract Base Classes")

    st.code(
        """
# Core abstract base classes that define the modular architecture
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class BaseLabelingStrategy(ABC):
    @abstractmethod
    def generate_labels(self, data: pd.DataFrame) -> pd.Series:
        pass

class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class BaseBacktester(ABC):
    @abstractmethod
    def run_backtest(self, system, data: pd.DataFrame) -> Dict[str, Any]:
        pass

class BaseSimulator(ABC):
    @abstractmethod
    def simulate(self, signals: pd.DataFrame) -> Dict[str, Any]:
        pass

class BasePortfolioManager(ABC):
    @abstractmethod
    def allocate_capital(self, signals: Dict[str, float]) -> Dict[str, float]:
        pass
    """,
        language="python",
    )

# Footer
st.markdown("---")
st.markdown("**XENIA Trading System** | Built with 🧱 modular architecture")

# Status indicator
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if st.session_state.system_trained:
        st.markdown(
            '<p class="status-success">✅ System Trained</p>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-warning">⏳ System Not Trained</p>',
            unsafe_allow_html=True,
        )

with status_col2:
    if st.session_state.backtest_results:
        st.markdown(
            '<p class="status-success">✅ Backtest Complete</p>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-warning">⏳ No Backtest Results</p>',
            unsafe_allow_html=True,
        )

with status_col3:
    if st.session_state.live_signal:
        st.markdown(
            '<p class="status-success">✅ Live Signals Ready</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="status-warning">⏳ No Live Signals</p>', unsafe_allow_html=True
        )
