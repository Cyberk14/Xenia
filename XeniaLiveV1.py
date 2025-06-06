import asyncio
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from main import XeniaV2, twelve_data_api_key
from datastream.twelvedata_ohlcv import TwelveDataOHLCVFetcher as DataFetcher


@dataclass
class SignalResult:
    """Clean data structure for signal results"""

    symbol: str
    signal: float
    confidence: float
    recommendation: str
    price: float
    price_change_pct: float
    timestamp: str
    volatility: float


class LiveStream:
    """Simple, focused live trading signals processor"""

    def __init__(
        self,
        fetcher: DataFetcher,
        model: XeniaV2,
        signal_threshold: float = 0.1,
        confidence_threshold: float = 0.5,
    ):
        self.fetcher = fetcher
        self.model = model
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold

    async def get_signals(
        self, symbols: List[str], interval: str = "5min", outputsize: int = 100
    ) -> Dict[str, SignalResult]:
        """Get live trading signals for symbols"""
        # Fetch data
        live_data = await self.fetcher.fetch_multiple(
            symbols, interval=interval, outputsize=outputsize
        )

        if not live_data:
            return {}

        # Process each symbol
        results = {}
        for symbol in symbols:
            if symbol not in live_data or live_data[symbol] is None:
                continue

            df = live_data[symbol]
            if df.empty or len(df) < 50:
                continue

            signal_result = self._process_symbol(symbol, df)
            if signal_result:
                results[symbol] = signal_result

        return results

    def _process_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[SignalResult]:
        """Process a single symbol's data into a signal"""
        try:
            df.rename(
                columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'},
                inplace=True)
            latest_idx = len(df) - 1

            # Get predictions
            model_signal, model_conf = self.model.get_prediction(symbol, latest_idx)
            tech_signal, tech_conf = self.model.get_technical_signal(symbol, latest_idx)

            # Combine signals (simple weighted average)
            combined_signal = model_signal * 0.6 + tech_signal * 0.4
            combined_confidence = model_conf * 0.6 + tech_conf * 0.4

            # Calculate metrics
            
            current_price = df["Close"].iloc[-1]
            price_change_pct = self._calculate_price_change_pct(df)
            volatility = self._calculate_volatility(df)
            recommendation = self._get_recommendation(
                combined_signal, combined_confidence
            )

            return SignalResult(
                symbol=symbol,
                signal=combined_signal,
                confidence=combined_confidence,
                recommendation=recommendation,
                price=current_price,
                price_change_pct=price_change_pct,
                timestamp=df.index[-1].isoformat(),
                volatility=volatility,
            )

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None

    def _calculate_price_change_pct(self, df: pd.DataFrame) -> float:
        """Calculate percentage price change"""
        if len(df) < 2:
            return 0.0
        current = df["Close"].iloc[-1]
        previous = df["Close"].iloc[-2]
        return ((current - previous) / previous) * 100

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate price volatility"""
        returns = df["Close"].pct_change().dropna()
        return returns.std() * 100 if len(returns) > 1 else 0.0

    def _get_recommendation(self, signal: float, confidence: float) -> str:
        """Get trading recommendation based on signal and confidence"""
        if confidence < self.confidence_threshold:
            return "HOLD"
        elif signal > self.signal_threshold:
            return "BUY"
        elif signal < -self.signal_threshold:
            return "SELL"
        else:
            return "HOLD"


# # Simple usage
# async def main():
#     fetcher = DataFetcher(twelve_data_api_key)
#     model = XeniaV2(["AAPL", "GOOGL", "MSFT"])
#     model.fetcher = "alphavantage"
#     await model.fetch_all_data()
#     await model.train_all_models()
#     if not model.models:
#         print("Model training failed. Exiting.")

#     print(f"{len(model.models)} are available")
#     processor = LiveStream(fetcher, model)

#     symbols = ["AAPL", "GOOGL", "MSFT"]
#     signals = await processor.get_signals(symbols)

#     for symbol, result in signals.items():
#         print(
#             f"{symbol}: {result.recommendation} | "
#             f"Signal: {result.signal:.3f} | "
#             f"Price: ${result.price:.2f} | "
#             f"Change: {result.price_change_pct:.2f}%"
#         )


# if __name__ == "__main__":
#     asyncio.run(main())
