import aiohttp
import asyncio
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class AlphaVantageOHLCVFetcher:
    """
    Async Alpha Vantage OHLCV fetcher for recent and historical data.
    Outputs a DataFrame with valid datetime index.
    """
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch_ohlcv(self, symbol: str, interval: str = "daily", outputsize: str = "compact") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.
        interval: '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'
        outputsize: 'compact' (latest 100) or 'full' (up to 20 years)
        Returns: DataFrame with columns [Open, High, Low, Close, Volume] and datetime index.
        """
        function_map = {
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY",
            "15min": "TIME_SERIES_INTRADAY",
            "30min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",
            "monthly": "TIME_SERIES_MONTHLY"
        }
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }

        if interval in ["1min", "5min", "15min", "30min", "60min"]:
            params["function"] = function_map[interval]
            params["interval"] = interval
            params["outputsize"] = outputsize
        elif interval in ["daily", "weekly", "monthly"]:
            params["function"] = function_map[interval]
            params["outputsize"] = outputsize
        else:
            raise ValueError("Invalid interval")

        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as resp:
                data = await resp.json()
                # Find the correct key for time series
                key = None
                for k in data.keys():
                    if "Time Series" in k:
                        key = k
                        break
                if not key or key not in data:
                    print(f"AlphaVantage error: {data.get('Note') or data.get('Error Message') or data}")
                    return None
                ts = data[key]
                df = pd.DataFrame.from_dict(ts, orient="index")
                df = df.rename(columns=lambda x: x.split(". ")[1].capitalize())
                # Convert index to datetime and sort
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                # Convert columns to numeric
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df[["Open", "High", "Low", "Close", "Volume"]]

    async def fetch_multiple(self, symbols: List[str], interval: str = "daily", outputsize: str = "compact") -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.
        Returns: dict of symbol -> DataFrame
        """
        async def fetch_symbol(symbol):
            try:
                df = await self.fetch_ohlcv(symbol, interval, outputsize)
                return symbol, df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return symbol, None

        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {symbol: df for symbol, df in results}

# Example usage:

# import dotenv
# dotenv.load_dotenv()
# import os

# alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
# async def main():
#     fetcher = AlphaVantageOHLCVFetcher(alpha_vantage_api_key)
#     symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA']
#     df = await fetcher.fetch_multiple(symbols, interval="5min", outputsize="full")

#     if df:
#         # Print the last few rows of each DataFrame
#         for symbol, data in df.items():
#             print(f"Data for {symbol}:")
#             print(len(data))
#             print(data.tail(), "\n")  
#     else:
#         print("Failed to fetch data.")

# if __name__ == "__main__":
#     asyncio.run(main())