from datastream.yfinance_ohlcv import YFinanceOHLCVFetcher as Fetcher
from main import XeniaV2

class LiveStream:
    def __init__(self, interval, resolution, lookback_period):
        self.system = XeniaV2()
        self.fetcher = Fetcher()
        
        # stream settings
        self.res = resolution
        self.inter = interval
        self.lookback = lookback_period

        # deep system componets.
        self.model_cache = []

    def run_one_sync(self, symbol):
        self.system.symbols = 
        self.system.fetch_all_data()
        if not self.model_cache:
            self.system.train_model_for_symbol(symbol)

