from agents.market_data_agent import get_data

def run_marketdata_agent(symbol, start=None, end=None):
    print(f"Fetching data for {symbol} from {start} to {end}...")
    data = get_data(symbol, start, end)
    print(data.head())
    return data

if __name__ == "__main__":
    # Example usage: fetch Apple stock data for January 2023
    run_marketdata_agent("AAPL", start="2023-01-01", end="2023-01-31")
