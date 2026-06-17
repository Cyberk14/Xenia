import aiohttp
import asyncio
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta, date
import time
import sys

import finnhub

import dotenv
dotenv.load_dotenv()        

import os 
api_key = os.getenv('FINNHUB_API_KEY')


class FinnhubOHLCVFetcher:
    """
    Async Finnhub OHLCV fetcher for recent and historical data.
    Outputs a DataFrame with valid datetime index.
    """
    BASE_URL = "https://finnhub.io/api/v1/stock/candle"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch_ohlcv(
        self,
        symbol: str,
        resolution: str = "D",
        lookback_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.
        resolution: '1', '5', '15', '30', '60', 'D', 'W', 'M'
        lookback_days: number of days to look back from now
        Returns: DataFrame with columns [open, high, low, close, volume] and datetime index.
        """
        end = int(time.time())
        start = end - lookback_days * 24 * 60 * 60

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": start,
            "to": end,
            "token": self.api_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as resp:
                data = await resp.json()
                if data.get("s") != "ok":
                    print(f"Finnhub error for {symbol}: {data.get('s', data)}")
                    return None
                df = pd.DataFrame({
                    "datetime": [datetime.fromtimestamp(ts) for ts in data["t"]],
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"]
                })
                df.set_index("datetime", inplace=True)
                return df[["open", "high", "low", "close", "volume"]]

    async def fetch_multiple(
        self,
        symbols: List[str],
        resolution: str = "D",
        lookback_days: int = 30
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

    def get_news(self, symbol: str, look_back: int):

        newsClient = finnhub.Client(self.api_key)

        from_date = int(time.time())
        end_date = from_date - look_back *12 * 60 *60

        from_date = date.fromtimestamp(from_date)
        end_date = date.fromtimestamp(end_date)

        raw = newsClient.company_news(symbol, end_date, from_date)
        try:
            if raw != None:
                news_bucket = []
                for news in raw:
                    news_bucket.append(dict(
                        headline=news['headline'],
                        newsId=news["id"],
                        time=datetime.fromtimestamp(news["datetime"])
                    ))
                return news_bucket

        except Exception as e:
            print(f"ERR: {e}")
            return None

    def headline_line(self, symbol: str, lookback: int, line_len: int = 5):
        try:        
            news = self.get_news(symbol, lookback)

            headlines = []
            for i in news:
                headlines.append(i.get('headline', 'headline'))

            headline_line = "  ──  ".join(headlines[:line_len])

            return headline_line
        
        except Exception as e:
            print(f"NEWS ERROR: {e}")
             
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()        

    import os 
    api_key = os.getenv('FINNHUB_API_KEY')

    fetcher = FinnhubOHLCVFetcher(api_key=api_key)

    line = fetcher.headline_line('NVDA', 2)
    print(line)