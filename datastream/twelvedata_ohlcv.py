import aiohttp
import asyncio
import pandas as pd
from typing import List, Dict, Optional

class TwelveDataOHLCVFetcher:
    """
    Async Twelve Data OHLCV fetcher for recent and historical data.
    Outputs a DataFrame with valid datetime index.
    """
    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.
        interval: '1min', '5min', '15min', '30min', '1h', '4h', '1day', etc.
        outputsize: number of data points to retrieve (max 5000 for paid plans)
        Returns: DataFrame with columns [open, high, low, close, volume] and datetime index.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as resp:
                data = await resp.json()
                if "values" not in data:
                    print(f"Twelve Data error for {symbol}: {data.get('message', data)}")
                    return None
                df = pd.DataFrame(data["values"])
                # Ensure correct order and types
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime")
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df.set_index("datetime", inplace=True)
                return df[["open", "high", "low", "close", "volume"]]

    async def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "1day",
        outputsize: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.
        Returns: dict of symbol -> DataFrame
        """
        async def fetch_symbol(symbol):
            try:
                df = await self.fetch_ohlcv(symbol, interval, outputsize)
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                return symbol, df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return symbol, None

        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {symbol: df for symbol, df in results}

# Example usage:

import dotenv
dotenv.load_dotenv()
import os

twelve_data_api_key = os.getenv("TWELVE_DATA_API_KEY")
async def main():
    fetcher = TwelveDataOHLCVFetcher(twelve_data_api_key)
    symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA']
    df = await fetcher.fetch_multiple(symbols, interval="1min", outputsize=100)

    if df:
        # Print the last few rows of each DataFrame
        for symbol, data in df.items():
            print(f"Data for {symbol}:")
            if data is not None:
                print(len(data))
                print(data.tail(), "\n")
            else:
                print("No data.\n")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    asyncio.run(main())