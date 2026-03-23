import yfinance as yf

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
