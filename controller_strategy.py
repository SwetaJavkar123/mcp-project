"""
Controller for MarketDataAgent + StrategyAgent
Orchestrates the workflow: Fetch data -> Calculate indicators -> Generate trading signals
"""

from utils_marketdata import (
    fetch_and_process_data,
    calculate_moving_average,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_stochastic,
    calculate_atr,
    calculate_adx,
    get_basic_statistics,
)
from agents.strategy_agent import generate_signals, get_strategy_description


def run_trading_analysis(symbol, start_date, end_date, strategy_type="combined"):
    """
    Complete trading analysis pipeline: fetch data, calculate indicators, generate signals.
    
    :param symbol: Stock ticker symbol (e.g., 'AAPL')
    :param start_date: Start date (YYYY-MM-DD)
    :param end_date: End date (YYYY-MM-DD)
    :param strategy_type: Type of strategy ("combined", "momentum", "mean_reversion", "trend_following")
    :return: Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Trading Analysis for {symbol}")
    print(f"Strategy: {strategy_type.upper()}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch data
    print("📊 Step 1: Fetching market data...")
    df = fetch_and_process_data(symbol, start_date, end_date)
    if df is None:
        print("❌ Failed to fetch data!")
        return None
    print(f"✅ Fetched {len(df)} days of data\n")
    
    # Step 2: Calculate technical indicators
    print("📈 Step 2: Calculating technical indicators...")
    df = calculate_moving_average(df, window=20)
    df = calculate_rsi(df, window=14)
    df = calculate_bollinger_bands(df, window=20, num_std=2)
    df = calculate_macd(df, fast=12, slow=26, signal=9)
    df = calculate_stochastic(df, window=14)
    df = calculate_atr(df, window=14)
    df = calculate_adx(df, window=14)
    print("✅ All indicators calculated\n")
    
    # Step 3: Generate trading signals
    print("🎯 Step 3: Generating trading signals...")
    df = generate_signals(df, strategy_type=strategy_type)
    print("✅ Trading signals generated\n")
    
    # Step 4: Analyze results
    print("📊 Step 4: Analyzing results...\n")
    buy_signals = df[df["Signal"] == "BUY"]
    sell_signals = df[df["Signal"] == "SELL"]
    hold_signals = df[df["Signal"] == "HOLD"]
    
    print(f"Signal Summary:")
    print(f"  • Buy Signals:  {len(buy_signals)}")
    print(f"  • Sell Signals: {len(sell_signals)}")
    print(f"  • Hold Signals: {len(hold_signals)}\n")
    
    # Calculate statistics
    stats = get_basic_statistics(df)
    print(f"Price Statistics:")
    print(f"  • Mean Close:   ${stats['Mean Close']:.2f}")
    print(f"  • Min Close:    ${stats['Min Close']:.2f}")
    print(f"  • Max Close:    ${stats['Max Close']:.2f}")
    print(f"  • Volatility:   {stats['Volatility (std dev)']:.2f}\n")
    
    # Strategy-specific insights
    if len(buy_signals) > 0 or len(sell_signals) > 0:
        avg_confidence = df[df["Signal"] != "HOLD"]["Confidence"].mean()
        print(f"Trading Signals:")
        print(f"  • Average Confidence: {avg_confidence:.1f}%")
        
        if len(buy_signals) > 0:
            best_buy = buy_signals.loc[buy_signals["Confidence"].idxmax()]
            print(f"  • Best Buy:  ${best_buy['Close']:.2f} (Confidence: {best_buy['Confidence']:.1f}%)")
        
        if len(sell_signals) > 0:
            best_sell = sell_signals.loc[sell_signals["Confidence"].idxmax()]
            print(f"  • Best Sell: ${best_sell['Close']:.2f} (Confidence: {best_sell['Confidence']:.1f}%)\n")
    
    # Strategy description
    print(f"Strategy Description:")
    print(f"  {get_strategy_description(strategy_type)}\n")
    
    # Prepare return data
    result = {
        "symbol": symbol,
        "strategy": strategy_type,
        "period": {"start": start_date, "end": end_date},
        "data": df,
        "statistics": stats,
        "signals": {
            "buy": len(buy_signals),
            "sell": len(sell_signals),
            "hold": len(hold_signals),
        },
        "latest_data": {
            "price": df["Close"].iloc[-1],
            "signal": df["Signal"].iloc[-1],
            "confidence": df["Confidence"].iloc[-1],
            "rsi": df["RSI"].iloc[-1] if "RSI" in df.columns else None,
            "macd": df["MACD"].iloc[-1] if "MACD" in df.columns else None,
        }
    }
    
    print(f"Latest Data:")
    print(f"  • Price:      ${result['latest_data']['price']:.2f}")
    print(f"  • Signal:     {result['latest_data']['signal']}")
    print(f"  • Confidence: {result['latest_data']['confidence']:.1f}%")
    print(f"  • RSI:        {result['latest_data']['rsi']:.2f}" if result['latest_data']['rsi'] else "")
    print(f"  • MACD:       {result['latest_data']['macd']:.4f}" if result['latest_data']['macd'] else "")
    print(f"\n{'='*60}\n")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("\n🚀 Starting Trading Analysis Controller\n")
    
    # Run analysis with different strategies
    strategies = ["combined", "momentum", "mean_reversion", "trend_following"]
    
    for strategy in strategies:
        result = run_trading_analysis(
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            strategy_type=strategy
        )
        
        if result:
            # Export latest signals
            latest = result["latest_data"]
            print(f"Latest Signal for {result['symbol']} ({strategy}): {latest['signal']}\n")
