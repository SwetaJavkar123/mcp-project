import yfinance as yf
import pandas as pd


def get_data(symbol, start=None, end=None):
    """
    Fetch historical market data for a given symbol.
    :param symbol: Stock ticker (e.g., 'AAPL')
    :param start: Start date (YYYY-MM-DD)
    :param end: End date (YYYY-MM-DD)
    :return: DataFrame with historical prices
    """
    data = yf.download(symbol, start=start, end=end)
    return data


# ---------------------------------------------------------------------------
# Multi-asset helpers (crypto, forex, indices via yfinance)
# ---------------------------------------------------------------------------

# Common tickers for different asset classes
CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "XRP": "XRP-USD",
    "DOT": "DOT-USD",
    "AVAX": "AVAX-USD",
    "MATIC": "MATIC-USD",
    "LINK": "LINK-USD",
}

FOREX_SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
}

INDEX_SYMBOLS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "DAX": "^GDAXI",
}


def resolve_symbol(symbol: str) -> str:
    """
    Resolve a human-friendly name to a yfinance ticker.
    e.g. 'BTC' → 'BTC-USD', 'EUR/USD' → 'EURUSD=X'
    Passes through already-valid tickers unchanged.
    """
    upper = symbol.upper().strip()
    if upper in CRYPTO_SYMBOLS:
        return CRYPTO_SYMBOLS[upper]
    if upper in FOREX_SYMBOLS:
        return FOREX_SYMBOLS[upper]
    if upper in INDEX_SYMBOLS:
        return INDEX_SYMBOLS[upper]
    return symbol


def get_multi_asset_data(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch data for any asset class. Resolves friendly names automatically.
    Supports intraday intervals: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo.
    """
    ticker = resolve_symbol(symbol)
    data = yf.download(ticker, start=start, end=end, interval=interval)
    return data


def get_asset_class(symbol: str) -> str:
    """Determine the asset class of a symbol."""
    upper = symbol.upper().strip()
    if upper in CRYPTO_SYMBOLS or upper.endswith("-USD"):
        return "crypto"
    if upper in FOREX_SYMBOLS or "=X" in upper:
        return "forex"
    if upper in INDEX_SYMBOLS or upper.startswith("^"):
        return "index"
    return "equity"


def get_available_symbols() -> dict:
    """Return all known symbols grouped by asset class."""
    return {
        "crypto": CRYPTO_SYMBOLS,
        "forex": FOREX_SYMBOLS,
        "indices": INDEX_SYMBOLS,
    }
