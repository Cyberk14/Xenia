import urllib.request
import pandas as pd
import time
from functools import wraps

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Create the request with a browser header to avoid security blocks
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

# Open the URL to download the HTML content
with urllib.request.urlopen(req) as response:
    # Read the downloaded HTML tables into pandas
    tables = pd.read_html(response)

# The first table on the page contains the current stock data
sp500_df = tables[0]

# Extract the 'Symbol' column into a Python list
sp500_tickers = sp500_df["Symbol"].tolist()

# Clean up symbols (change dots to hyphens for compatibility, e.g., BRK.B to BRK-B)
sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]

