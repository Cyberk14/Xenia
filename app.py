import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import numpy as np
from datetime import datetime, timedelta
import threading
from main import XeniaV2  # Import your XeniaV2 class

st.set_page_config(
    page_title="Xenia V2 Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .signal-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .trade-profit {
        color: #28a745;
        font-weight: bold;
    }
    .trade-loss {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "system" not in st.session_state:
    st.session_state.system = None
if "backtest_complete" not in st.session_state:
    st.session_state.backtest_complete = False
if "current_signals" not in st.session_state:
    st.session_state.current_signals = {}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_backtest_cached(
    symbols, initial_balance, transaction_cost, start_date, end_date
):
    """Cached backtest execution"""
    system = XeniaV2(symbols, initial_balance, transaction_cost)

    # Run backtest
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(system.run_backtest(start_date, end_date))
    finally:
        loop.close()

    return system


def main():
    st.title("Xenia V2 ML Trading System")
    st.markdown("Advanced Machine Learning Trading System with Technical Analysis")

    # Sidebar controls
    with st.sidebar:
        st.header("System Controls")

        # Symbol selection
        symbols = st.multiselect(
            "Select Trading Symbols",
            [
                "AAPL",
                "GOOGL",
                "MSFT",
                "TSLA",
                "AMZN",
                "NVDA",
                "META",
                "SPY",
                "QQQ",
                "NFLX",
                "AMD",
                "CRM",
            ],
            default=["AAPL", "GOOGL", "MSFT", "TSLA"],
        )

        if not symbols:
            st.error("Please select at least one symbol")
            return

        # System parameters
        st.subheader("Parameters")
        initial_balance = st.number_input(
            "Initial Balance ($)", value=100000, min_value=1000
        )
        transaction_cost = st.slider("Transaction Cost (%)", 0.001, 0.01, 0.002, 0.001)

        # Thresholds
        st.subheader("Trading Thresholds")
        min_accuracy = st.slider("Min Model Accuracy", 0.3, 0.7, 0.35, 0.05)
        signal_threshold = st.slider("Signal Threshold", 0.1, 0.5, 0.15, 0.05)
        confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.8, 0.4, 0.05)
        max_position_size = st.slider("Max Position Size (%)", 0.1, 0.5, 0.2, 0.05)

        # Backtest period
        st.subheader("Backtest Period")
        start_date = st.date_input("Start Date", value=datetime(2022, 6, 1))
        end_date = st.date_input("End Date", value=datetime(2024, 1, 1))

        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

        # Initialize and run system
        if st.button("Initialize & Run Backtest", type="primary"):
            with st.spinner(
                "Initializing system and running backtest... This may take several minutes."
            ):
                try:
                    # Create system with custom parameters
                    system = XeniaV2(symbols, initial_balance, transaction_cost)
                    system.min_accuracy_threshold = min_accuracy
                    system.signal_threshold = signal_threshold
                    system.confidence_threshold = confidence_threshold
                    system.max_position_size = max_position_size

                    # Run backtest
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            system.run_backtest(
                                start_date=start_date.strftime("%Y-%m-%d"),
                                end_date=end_date.strftime("%Y-%m-%d"),
                            )
                        )
                    finally:
                        loop.close()

                    st.session_state.system = system
                    st.session_state.backtest_complete = True
                    st.success("System initialized and backtest completed!")

                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
                    return

        # Get current signals button
        if st.session_state.system and st.button("Refresh Current Signals"):
            with st.spinner("Getting current market signals..."):
                try:
                    signals = st.session_state.system.get_current_signals()
                    st.session_state.current_signals = signals
                    st.success("Signals updated!")
                except Exception as e:
                    st.error(f"Error getting signals: {str(e)}")

    # Main content
    if st.session_state.system is None:
        st.info("Please initialize the system using the sidebar controls.")

        # Show system information
        st.subheader("About Xenia V2")
        st.markdown("""
        **Xenia V2** is an advanced machine learning trading system that combines:
        
        - **Random Forest & Gradient Boosting** models for price prediction
        - **Technical Analysis** indicators (RSI, MACD, Bollinger Bands, Moving Averages)
        - **Risk Management** with position sizing and transaction costs
        - **Time Series Cross-Validation** for robust model training
        - **Async Data Fetching** for improved performance
        
        The system uses **ensemble learning** by combining ML predictions with technical signals
        and includes comprehensive **backtesting** capabilities.
        """)
        return

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Portfolio Overview",
            "📈 Current Signals",
            "📉 Backtest Results",
            "🔍 Stock Analysis",
            "⚙️ Model Performance",
        ]
    )

    with tab1:
        display_portfolio_overview()

    with tab2:
        display_current_signals()

    with tab3:
        display_backtest_results()

    with tab4:
        display_stock_analysis()

    with tab5:
        display_model_performance()


def display_portfolio_overview():
    st.header("Portfolio Overview")

    system = st.session_state.system
    portfolio = system.get_portfolio_status()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Value",
            f"${portfolio['total_value']:,.2f}",
            f"{portfolio['total_return']:.2f}%",
        )

    with col2:
        st.metric("Cash Balance", f"${portfolio['cash']:,.2f}")

    with col3:
        st.metric("Active Positions", len(portfolio["positions"]))

    with col4:
        total_trades = len(system.trades)
        st.metric("Total Trades", total_trades)

    # Portfolio allocation
    if portfolio["positions"]:
        st.subheader("Current Positions")

        # Create portfolio dataframe
        positions_data = []
        for symbol, pos_data in portfolio["positions"].items():
            positions_data.append(
                {
                    "Symbol": symbol,
                    "Shares": pos_data["shares"],
                    "Current Price": pos_data["current_price"],
                    "Position Value": pos_data["value"],
                    "Weight": (pos_data["value"] / portfolio["total_value"]) * 100,
                }
            )

        if positions_data:
            pos_df = pd.DataFrame(positions_data)
            st.dataframe(pos_df, use_container_width=True)

            # Portfolio pie chart
            fig = px.pie(
                pos_df,
                values="Position Value",
                names="Symbol",
                title="Portfolio Allocation",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active positions. All capital is in cash.")


def display_current_signals():
    st.header("Current Trading Signals")

    system = st.session_state.system

    if not st.session_state.current_signals:
        st.info("Click 'Refresh Current Signals' in the sidebar to get latest signals.")
        return

    signals = st.session_state.current_signals

    if not signals:
        st.warning(
            "No signals available. Check if models are trained and data is available."
        )
        return

    # Display signals for each symbol
    for symbol, signal_data in signals.items():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

            with col1:
                st.metric(symbol, f"${signal_data['price']:.2f}")

            with col2:
                confidence = signal_data["confidence"]
                conf_color = (
                    "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                )
                st.metric("Confidence", f"{conf_color} {confidence:.3f}")

            with col3:
                # Signal gauge
                combined_signal = signal_data["combined_signal"]

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=combined_signal,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": f"{symbol} Signal"},
                        gauge={
                            "axis": {"range": [-1, 1]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [-1, -0.15], "color": "lightcoral"},
                                {"range": [-0.15, 0.15], "color": "lightyellow"},
                                {"range": [0.15, 1], "color": "lightgreen"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": system.signal_threshold,
                            },
                        },
                    )
                )
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                recommendation = signal_data["recommendation"]
                if recommendation == "BUY":
                    st.markdown(
                        f'<div class="signal-buy">🟢 {recommendation}</div>',
                        unsafe_allow_html=True,
                    )
                elif recommendation == "SELL":
                    st.markdown(
                        f'<div class="signal-sell">🔴 {recommendation}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="signal-hold">🟡 {recommendation}</div>',
                        unsafe_allow_html=True,
                    )

            with col5:
                st.metric("ML Signal", f"{signal_data['model_signal']:.3f}")
                st.metric("Tech Signal", f"{signal_data['tech_signal']:.3f}")

            st.divider()


def display_backtest_results():
    st.header("Backtest Results")

    if not st.session_state.backtest_complete:
        st.info("Run a backtest to see results here.")
        return

    system = st.session_state.system

    if not system.trades:
        st.warning("No trades were executed during the backtest period.")
        st.info(
            "Try adjusting the trading thresholds in the sidebar to generate more trades."
        )
        return

    # Calculate performance metrics
    portfolio = system.get_portfolio_status()
    total_return = portfolio["total_return"]

    # Trade statistics
    sell_trades = [t for t in system.trades if t["action"] == "SELL"]
    buy_trades = [t for t in system.trades if t["action"] == "BUY"]

    if sell_trades:
        profitable_trades = [t for t in sell_trades if t["pnl"] > 0]
        win_rate = len(profitable_trades) / len(sell_trades) * 100 if sell_trades else 0
        avg_profit = np.mean([t["pnl"] for t in sell_trades])
        avg_profit_pct = np.mean([t["pnl_pct"] for t in sell_trades])
    else:
        win_rate = 0
        avg_profit = 0
        avg_profit_pct = 0

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{total_return:.2f}%")

    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with col3:
        st.metric("Avg Trade P&L", f"${avg_profit:.2f}")

    with col4:
        st.metric("Total Trades", len(system.trades))

    # Equity curve
    if system.trades:
        st.subheader("Portfolio Performance Over Time")

        # Create equity curve from trades
        trade_dates = []
        portfolio_values = []
        running_balance = system.initial_balance

        for trade in system.trades:
            trade_dates.append(trade["date"])
            portfolio_values.append(trade["balance_after"])

        if trade_dates:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trade_dates,
                    y=portfolio_values,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="blue", width=2),
                )
            )

            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

    # Trade history
    st.subheader("Trade History")

    if system.trades:
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(system.trades)

        # Format columns
        if "date" in trades_df.columns:
            trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime(
                "%Y-%m-%d"
            )

        # Select relevant columns
        display_cols = ["date", "symbol", "action", "shares", "price"]
        if "pnl" in trades_df.columns:
            display_cols.extend(["pnl", "pnl_pct"])
        if "signal" in trades_df.columns:
            display_cols.extend(["signal", "confidence"])

        # Filter columns that exist
        display_cols = [col for col in display_cols if col in trades_df.columns]

        # Format numeric columns
        numeric_cols = ["shares", "price", "pnl", "pnl_pct", "signal", "confidence"]
        for col in numeric_cols:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].round(3)

        st.dataframe(trades_df[display_cols].tail(20), use_container_width=True)

    # Performance by symbol
    if sell_trades:
        st.subheader("Performance by Symbol")

        symbol_performance = {}
        for trade in sell_trades:
            symbol = trade["symbol"]
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {"trades": 0, "total_pnl": 0, "wins": 0}

            symbol_performance[symbol]["trades"] += 1
            symbol_performance[symbol]["total_pnl"] += trade["pnl"]
            if trade["pnl"] > 0:
                symbol_performance[symbol]["wins"] += 1

        perf_data = []
        for symbol, data in symbol_performance.items():
            win_rate = (
                (data["wins"] / data["trades"]) * 100 if data["trades"] > 0 else 0
            )
            perf_data.append(
                {
                    "Symbol": symbol,
                    "Trades": data["trades"],
                    "Total P&L": f"${data['total_pnl']:.2f}",
                    "Win Rate": f"{win_rate:.1f}%",
                }
            )

        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)


def display_stock_analysis():
    st.header("Individual Stock Analysis")

    system = st.session_state.system

    if not system.data:
        st.warning("No stock data available for analysis.")
        return

    # Stock selector
    available_symbols = list(system.data.keys())
    selected_stock = st.selectbox("Select Stock for Analysis", available_symbols)

    if selected_stock not in system.data:
        st.error(f"No data available for {selected_stock}")
        return

    # Get stock data
    stock_data = system.data[selected_stock]

    # Create features for analysis
    features_df = system.create_features(stock_data)

    if features_df is None:
        st.error(f"Unable to create features for {selected_stock}")
        return

    # Price chart with technical indicators
    st.subheader(f"{selected_stock} Price Analysis")

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.05,
        subplot_titles=(f"{selected_stock} Price Chart", "RSI", "MACD"),
        shared_xaxes=True,
    )

    # Price chart
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Add moving averages if available
    if "sma_20" in features_df.columns:
        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=features_df["sma_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="orange", dash="dash"),
            ),
            row=1,
            col=1,
        )

    if "sma_50" in features_df.columns:
        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=features_df["sma_50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Bollinger Bands if available
    if all(col in features_df.columns for col in ["bb_upper", "bb_lower"]):
        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=features_df["bb_upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="gray", dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=features_df["bb_lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="gray", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # RSI
    if "rsi" in features_df.columns:
        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=features_df["rsi"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if "macd" in features_df.columns:
        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=features_df["macd"],
                mode="lines",
                name="MACD",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        if "macd_signal" in features_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=features_df.index,
                    y=features_df["macd_signal"],
                    mode="lines",
                    name="MACD Signal",
                    line=dict(color="red"),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # Recent data
    st.subheader(f"Recent {selected_stock} Data")

    # Combine price and feature data
    recent_data = stock_data[["Open", "High", "Low", "Close", "Volume"]].tail(10)

    if "rsi" in features_df.columns:
        recent_data = recent_data.join(
            features_df[["rsi", "macd", "bb_position"]].tail(10)
        )

    st.dataframe(recent_data, use_container_width=True)


def display_model_performance():
    st.header("Model Performance Analytics")

    system = st.session_state.system

    if not system.models:
        st.warning("No models available. Please run the backtest first.")
        return

    # Model accuracy by symbol
    st.subheader("Model Accuracy by Symbol")

    model_data = []
    for symbol, model_info in system.models.items():
        model_data.append(
            {
                "Symbol": symbol,
                "Accuracy": model_info["accuracy"],
                "Feature Count": len(model_info["feature_cols"]),
            }
        )

    if model_data:
        model_df = pd.DataFrame(model_data)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                model_df,
                x="Symbol",
                y="Accuracy",
                title="Model Accuracy by Symbol",
                color="Accuracy",
                color_continuous_scale="RdYlGn",
            )
            fig.add_hline(
                y=system.min_accuracy_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Min Threshold ({system.min_accuracy_threshold})",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                model_df,
                x="Feature Count",
                y="Accuracy",
                text="Symbol",
                title="Accuracy vs Feature Count",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

        # Model details table
        st.subheader("Model Details")
        st.dataframe(model_df, use_container_width=True)

    # Feature importance (if available)
    st.subheader("Feature Analysis")

    selected_model_symbol = st.selectbox(
        "Select Symbol for Feature Analysis", list(system.models.keys())
    )

    if selected_model_symbol in system.models:
        model_info = system.models[selected_model_symbol]

        # Show feature columns used
        st.write(f"**Features used for {selected_model_symbol}:**")
        feature_cols = model_info["feature_cols"]

        # Group features by type
        feature_groups = {
            "Price Features": [
                f
                for f in feature_cols
                if any(x in f for x in ["returns", "momentum", "price_to"])
            ],
            "Technical Indicators": [
                f for f in feature_cols if any(x in f for x in ["rsi", "macd", "bb_"])
            ],
            "Moving Averages": [
                f for f in feature_cols if any(x in f for x in ["sma_", "ema_"])
            ],
            "Volume Features": [f for f in feature_cols if "volume" in f],
            "Volatility Features": [
                f for f in feature_cols if "volatility" in f or "std" in f
            ],
            "Lag Features": [f for f in feature_cols if "lag" in f],
            "Other": [
                f
                for f in feature_cols
                if not any(
                    group in f
                    for group in [
                        "returns",
                        "momentum",
                        "price_to",
                        "rsi",
                        "macd",
                        "bb_",
                        "sma_",
                        "ema_",
                        "volume",
                        "volatility",
                        "std",
                        "lag",
                    ]
                )
            ],
        }

        for group_name, features in feature_groups.items():
            if features:
                st.write(f"**{group_name}:** {', '.join(features)}")

        st.write(f"**Total Features:** {len(feature_cols)}")
        st.write(f"**Model Accuracy:** {model_info['accuracy']:.3f}")

    # System parameters
    st.subheader("Current System Parameters")

    params_data = {
        "Parameter": [
            "Min Accuracy Threshold",
            "Signal Threshold",
            "Confidence Threshold",
            "Max Position Size",
            "Transaction Cost",
        ],
        "Value": [
            f"{system.min_accuracy_threshold:.3f}",
            f"{system.signal_threshold:.3f}",
            f"{system.confidence_threshold:.3f}",
            f"{system.max_position_size:.1%}",
            f"{system.transaction_cost:.1%}",
        ],
    }

    params_df = pd.DataFrame(params_data)
    st.dataframe(params_df, use_container_width=True)


if __name__ == "__main__":
    main()
