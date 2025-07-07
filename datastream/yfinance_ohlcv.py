import yfinance as yf
import asyncio
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial


class YFinanceOHLCVFetcher:
    """
    Async YFinance OHLCV fetcher for recent and historical data.
    Outputs a DataFrame with valid datetime index.
    """

    def __init__(self):
        # YFinance doesn't require API key
        pass

    def _resolution_to_interval(self, resolution: str) -> str:
        """        Convert resolution string to yfinance interval format.
        resolution: '1', '5', '15', '30', '60', 'D', 'W', 'M'
        """
        resolution_map = {
            "1": "1m",  # 1 minute
            "5": "5m",  # 5 minutes
            "15": "15m",  # 15 minutes
            "30": "30m",  # 30 minutes
            "60": "1h",  # 1 hour
            "D": "1d",  # 1 day
            "W": "1wk",  # 1 week
            "M": "1mo",  # 1 month
        }
        return resolution_map.get(resolution, "1d")

    def _fetch_ohlcv_sync(
        self, symbol: str, resolution: str = "15", lookback_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Synchronous fetch method to be run in thread pool.
        """
        try:
            interval = self._resolution_to_interval(resolution)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True,
            )

            if data.empty:
                print(f"No data found for {symbol}")
                return None

            # Rename columns to match Finnhub format (lowercase)
            data.columns = data.columns.str.lower()

            # Ensure we have the required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in required_cols):
                print(f"Missing required columns for {symbol}")
                return None

            # Convert timezone-aware datetime to timezone-naive if needed
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            # Rename index to match Finnhub format
            data.index.name = "datetime"
            data = data.dropna()
            print(len(data), "records fetched for", symbol, resolution, "resolution")
            

            return data[required_cols]

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    async def fetch_ohlcv(
        self, symbol: str, resolution: str = "D", lookback_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol asynchronously.
        resolution: '1', '5', '15', '30', '60', 'D', 'W', 'M'
        lookback_days: number of days to look back from now
        Returns: DataFrame with columns [open, high, low, close, volume] and datetime index.
        """
        loop = asyncio.get_event_loop()

        # Run the synchronous yfinance call in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor,
                partial(self._fetch_ohlcv_sync, symbol, resolution, lookback_days),
            )
            return await future

    async def fetch_multiple(
        self, symbols: List[str], resolution: str = "D", lookback_days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.
        Returns: dict of symbol -> DataFrame
        """

        async def fetch_symbol(symbol):
            try:
                df = await self.fetch_ohlcv(symbol, resolution, lookback_days)
                return symbol, df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return symbol, None

        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {symbol: df for symbol, df in results}


# Example usage:


async def main():
    fetcher = YFinanceOHLCVFetcher()
    symbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]
    df = await fetcher.fetch_multiple(symbols, resolution="1", lookback_days=5)

    if df:
        # Print the last few rows of each DataFrame
        for symbol, data in df.items():
            print(f"Data for {symbol}:")
            if data is not None:
                print(f"Records: {len(data)}")
                print(data.tail(), "\n")
            else:
                print("No data.\n")
    else:
        print("Failed to fetch data.")


# Alternative example with daily data
async def main_daily():
    fetcher = YFinanceOHLCVFetcher()
    symbols = ["AAPL", "MSFT", "TSLA"]

    # Fetch daily data for the last 90 days
    df = await fetcher.fetch_multiple(symbols, resolution="1", lookback_days=90)

    for symbol, data in df.items():
        if data is not None:
            print(f"\n{symbol} - Last 5 days:")
            print(data.tail())
            print(f"Date range: {data.index.min()} to {data.index.max()}")
        else:
            print(f"\nNo data for {symbol}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Uncomment to run daily data example
    # asyncio.run(main_daily())
