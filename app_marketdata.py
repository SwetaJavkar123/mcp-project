import streamlit as st
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
    get_popular_symbols,
)

st.set_page_config(page_title="Market Data Agent", layout="wide")
st.title("📈 Market Data Agent - Advanced Analysis")
st.subheader("Technical Analysis & Portfolio Comparison")

# Tabs for different features
tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Advanced Indicators", "Multi-Symbol Comparison"])

with tab1:
    st.markdown("### Technical Analysis (RSI + Bollinger Bands)")
    popular_symbols = get_popular_symbols()
    symbol = st.selectbox("Select stock symbol:", options=popular_symbols, index=0, key="symbol1")
    custom_symbol = st.text_input("Or enter another symbol:", key="custom1")
    if custom_symbol:
        symbol = custom_symbol.upper()

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", key="start1")
    with col2:
        end = st.date_input("End date", key="end1")

    ma_window = st.slider("Moving Average Window (days)", min_value=5, max_value=50, value=20, key="ma_window1")

    if st.button("Fetch Technical Analysis"):
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_and_process_data(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is not None:
                st.success(f"Showing technical analysis for {symbol}")
                # Calculate indicators
                df = calculate_moving_average(df, ma_window)
                df = calculate_rsi(df, window=14)
                df = calculate_bollinger_bands(df, window=20, num_std=2)
                
                # Price + Bollinger Bands
                st.markdown("#### Price with Bollinger Bands")
                st.line_chart(df[["Close", "BB_MA", "BB_Upper", "BB_Lower"]])
                
                # RSI Indicator
                st.markdown("#### Relative Strength Index (RSI)")
                rsi_chart = df[["RSI"]].copy()
                st.line_chart(rsi_chart)
                
                # Data table
                st.markdown("#### Data Table")
                st.dataframe(df[["Close", "MA", "RSI", "BB_Upper", "BB_MA", "BB_Lower"]])
                
                # Statistics
                st.markdown("#### Basic Statistics")
                stats = get_basic_statistics(df)
                st.json(stats)
            else:
                st.warning("No data found for the given symbol and date range.")

with tab2:
    st.markdown("### Advanced Technical Indicators (MACD, Stochastic, ATR, ADX)")
    popular_symbols = get_popular_symbols()
    symbol = st.selectbox("Select stock symbol:", options=popular_symbols, index=0, key="symbol2")
    custom_symbol = st.text_input("Or enter another symbol:", key="custom2")
    if custom_symbol:
        symbol = custom_symbol.upper()

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", key="start2")
    with col2:
        end = st.date_input("End date", key="end2")

    if st.button("Fetch Advanced Indicators"):
        with st.spinner(f"Fetching advanced indicators for {symbol}..."):
            df = fetch_and_process_data(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is not None:
                st.success(f"Advanced indicators for {symbol}")
                # Calculate all advanced indicators
                df = calculate_macd(df)
                df = calculate_stochastic(df)
                df = calculate_atr(df)
                df = calculate_adx(df)
                
                # MACD
                st.markdown("#### MACD (Moving Average Convergence Divergence)")
                st.line_chart(df[["MACD", "MACD_Signal", "MACD_Histogram"]])
                
                # Stochastic Oscillator
                st.markdown("#### Stochastic Oscillator (%K and %D)")
                st.line_chart(df[["Stoch_K", "Stoch_D"]])
                
                # ATR (Average True Range)
                st.markdown("#### ATR (Average True Range) - Volatility")
                st.line_chart(df[["ATR"]])
                
                # ADX (Trend Strength)
                st.markdown("#### ADX (Average Directional Index) - Trend Strength")
                st.line_chart(df[["ADX"]])
                
                # Data table with all indicators
                st.markdown("#### Full Data with All Indicators")
                display_cols = ["Close", "MACD", "MACD_Signal", "Stoch_K", "Stoch_D", "ATR", "ADX"]
                st.dataframe(df[display_cols].dropna())
            else:
                st.warning("No data found for the given symbol and date range.")

with tab3:
    st.markdown("### Multi-Symbol Comparison")
    symbols_to_compare = st.multiselect(
        "Select symbols to compare:",
        options=get_popular_symbols(),
        default=["AAPL", "MSFT"],
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_comp = st.date_input("Start date", key="start_comp")
    with col2:
        end_comp = st.date_input("End date", key="end_comp")

    if st.button("Compare Symbols"):
        if symbols_to_compare:
            with st.spinner("Fetching data..."):
                comparison_df = None
                for sym in symbols_to_compare:
                    df = fetch_and_process_data(sym, start=start_comp.strftime("%Y-%m-%d"), end=end_comp.strftime("%Y-%m-%d"))
                    if df is not None:
                        df = calculate_moving_average(df, 20)
                        if comparison_df is None:
                            comparison_df = df[["Close"]].copy()
                            comparison_df.columns = [sym]
                        else:
                            comparison_df[sym] = df["Close"]
                
                if comparison_df is not None:
                    st.success("Comparison data loaded")
                    st.line_chart(comparison_df)
                    st.dataframe(comparison_df)
                    
                    # Correlation matrix
                    st.markdown("#### Correlation Matrix")
                    corr = comparison_df.corr()
                    st.dataframe(corr)
                else:
                    st.warning("No data found for selected symbols.")
        else:
            st.warning("Please select at least one symbol to compare.")
