"""
mt5_resolver.py
Fetches symbols directly from the connected MT5 terminal.
Handles broker suffix stripping for yfinance compatibility.
"""

from difflib import SequenceMatcher

# Broker-specific suffixes to strip when converting MT5 → yfinance ticker
# e.g. "AAPL.pro" → "AAPL",  "EURUSD_i" → "EURUSD"
MT5_SUFFIXES = [
    '.pro', '.ecn', '.raw', '.sp', '.fp', '.m', '.c', '.r', '.n', '.s',
    '_i', '_', '#', '.eu', '.uk', '.us', '.cent', '.micro', '.stp'
]


# ── FETCH ────────────────────────────────────────────────
def get_mt5_symbols(visible_only=False):
    """
    Fetch every symbol available in the connected MT5 terminal.
    visible_only=False returns all symbols including hidden ones.
    Returns list of dicts: {name, description, path, digits, visible}
    """
    try:
        import MetaTrader5 as mt5
        raw = mt5.symbols_get()
        if not raw:
            return []
        result = []
        for s in raw:
            if visible_only and not s.visible:
                continue
            result.append({
                'name':        s.name,
                'description': s.description,
                'path':        s.path,          # e.g. "Forex\Majors\EURUSD"
                'digits':      s.digits,
                'visible':     s.visible,
            })
        return result
    except Exception:
        return []


def symbols_to_options(mt5_symbols):
    """
    Convert MT5 symbol list to Dash Dropdown options list.
    Label: "AAPL  ·  Apple Inc"   Value: "AAPL"
    """
    options = []
    for s in mt5_symbols:
        label = (f"{s['name']}  ·  {s['description']}"
                 if s['description'] else s['name'])
        options.append({'label': label, 'value': s['name']})
    return options


def group_symbols_by_path(mt5_symbols):
    """
    Group MT5 symbols by their path category.
    Returns dict: { "Forex\\Majors": [...], "Stocks\\US": [...], ... }
    Useful for building grouped dropdowns.
    """
    groups = {}
    for s in mt5_symbols:
        key = s['path'] or 'Other'
        groups.setdefault(key, []).append(s)
    return groups


# ── CONVERSION ──────────────────────────────────────────
def mt5_to_yfinance(mt5_symbol):
    """
    Strip broker suffix to get a clean ticker for yfinance.
    "AAPL.pro" → "AAPL"
    "EURUSD_i" → "EURUSD"
    "TSLA#"    → "TSLA"
    """
    symbol = mt5_symbol.upper()
    for suffix in MT5_SUFFIXES:
        if symbol.endswith(suffix.upper()):
            return symbol[: -len(suffix)]
    return symbol


def find_mt5_symbol(yf_ticker, mt5_symbols):
    """
    Find the best matching MT5 symbol for a given yfinance-style ticker.

    Strategy order:
      1. Exact match           — "AAPL"     matches "AAPL"
      2. Suffix variants       — "AAPL"     matches "AAPL.pro"
      3. Description fuzzy     — "AAPL"     matches symbol described as "Apple Inc"
      4. Fallback              — returns original ticker unchanged

    Returns: (resolved_mt5_name, method, confidence_0_to_1)
    """
    if not mt5_symbols:
        return yf_ticker, 'fallback', 0.0

    ticker     = yf_ticker.upper()
    name_index = {s['name'].upper(): s['name'] for s in mt5_symbols}

    # 1. Exact
    if ticker in name_index:
        return name_index[ticker], 'exact', 1.0

    # 2. Suffix variants
    for sfx in MT5_SUFFIXES:
        candidate = ticker + sfx.upper()
        if candidate in name_index:
            return name_index[candidate], 'suffix', 0.95

    # 3. Description fuzzy
    best_sym, best_score = None, 0.0
    for s in mt5_symbols:
        desc  = s['description'].upper()
        score = SequenceMatcher(None, ticker, desc).ratio()
        if ticker in desc:
            score = max(score, 0.80)
        if ticker in s['name'].upper():
            score = max(score, 0.90)
        if score > best_score:
            best_score, best_sym = score, s['name']

    if best_sym and best_score > 0.50:
        return best_sym, 'description', best_score

    return yf_ticker, 'fallback', 0.0
