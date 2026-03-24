"""
Hedge Fund Dashboard — Streamlit UI
=====================================
Single-page app with tabs for every analysis stage.
Powered by controller_hedge_fund.run_full_analysis().
"""

import streamlit as st
import pandas as pd
import numpy as np

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
from agents.strategy_agent import generate_signals, get_strategy_description
from agents.company_research_agent import (
    get_fundamentals,
    get_recent_news,
    compare_peers,
    generate_research_report,
)
from agents.risk_agent import (
    Portfolio,
    TradeProposal,
    validate_trade,
    calculate_risk_metrics,
    calculate_position_size,
    calculate_portfolio_summary,
)
from agents.backtest_agent import run_backtest, BacktestConfig, trades_to_dataframe

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hedge Fund Multi-Agent Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — shared inputs
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Hedge Fund Platform")
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Parameters")

popular = get_popular_symbols()
symbol = st.sidebar.selectbox("Stock Symbol", options=popular, index=0)
custom = st.sidebar.text_input("Or enter ticker:")
if custom:
    symbol = custom.upper()

col_s, col_e = st.sidebar.columns(2)
with col_s:
    start = col_s.date_input("Start", value=pd.Timestamp("2024-01-01"))
with col_e:
    end = col_e.date_input("End", value=pd.Timestamp.today())

strategy_type = st.sidebar.selectbox(
    "Strategy",
    ["combined", "momentum", "mean_reversion", "trend_following"],
    format_func=lambda x: x.replace("_", " ").title(),
)
run_analysis = st.sidebar.button("🚀  Run Full Analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Multi-Agent Control Platform • v2.0")

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 Hedge Fund Multi-Agent Platform")
st.markdown("End-to-end trading analysis: **Market Data → Indicators → Strategy → Research → Risk → Backtest**")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_tech, tab_strategy, tab_research, tab_risk, tab_backtest, tab_compare = st.tabs([
    "📋 Overview",
    "📈 Technical Analysis",
    "🎯 Strategy Signals",
    "🔍 Company Research",
    "🛡️ Risk Management",
    "⏪ Backtest",
    "📊 Multi-Symbol",
])

# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

def _fmt(val, fmt=".2f", prefix="", suffix=""):
    if val is None:
        return "N/A"
    return f"{prefix}{val:{fmt}}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Run the pipeline
# ─────────────────────────────────────────────────────────────────────────────
if run_analysis:
    with st.spinner(f"Running full analysis for **{symbol}** …"):
        # 1. Data
        df = fetch_and_process_data(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            st.error("No data returned for this symbol / date range.")
            st.stop()

        # 2. Indicators
        df = calculate_moving_average(df, 20)
        df = calculate_rsi(df)
        df = calculate_bollinger_bands(df)
        df = calculate_macd(df)
        df = calculate_stochastic(df)
        df = calculate_atr(df)
        df = calculate_adx(df)
        stats = get_basic_statistics(df)

        # 3. Strategy signals
        df = generate_signals(df, strategy_type=strategy_type)

        # 4. Company research
        try:
            research = generate_research_report(symbol)
        except Exception:
            research = None

        # 5. Risk
        portfolio = Portfolio()
        latest_price = float(df["Close"].iloc[-1])
        latest_signal = df["Signal"].iloc[-1]
        latest_confidence = float(df["Confidence"].iloc[-1])
        returns = df["Close"].pct_change().dropna()
        risk_metrics = calculate_risk_metrics(returns, df["Close"])

        risk_validation = None
        if latest_signal in ("BUY", "SELL"):
            sizing = calculate_position_size(portfolio.total_value, latest_price)
            proposal = TradeProposal(
                symbol=symbol, action=latest_signal,
                shares=sizing["shares"], price=latest_price,
                confidence=latest_confidence, strategy=strategy_type,
            )
            risk_validation = validate_trade(proposal, portfolio, returns)

        # 6. Backtest
        try:
            bt = run_backtest(df, symbol=symbol)
        except Exception:
            bt = None

        st.session_state.result = {
            "symbol": symbol,
            "strategy": strategy_type,
            "df": df,
            "stats": stats,
            "research": research,
            "risk_metrics": risk_metrics,
            "risk_validation": risk_validation,
            "portfolio": portfolio,
            "backtest": bt,
            "latest_price": latest_price,
            "latest_signal": latest_signal,
            "latest_confidence": latest_confidence,
        }
    st.success(f"Analysis complete for **{symbol}**!")

res = st.session_state.result

# ─────────────────────────────────────────────────────────────────────────────
# TAB: Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    if res is None:
        st.info("👈 Configure parameters in the sidebar and click **Run Full Analysis**.")
    else:
        st.header(f"Overview — {res['symbol']}")

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", f"${res['latest_price']:.2f}")
        c2.metric("Signal", res["latest_signal"])
        c3.metric("Confidence", f"{res['latest_confidence']:.0f}%")
        rsi_val = res["df"]["RSI"].iloc[-1] if "RSI" in res["df"].columns else None
        c4.metric("RSI", _fmt(rsi_val, ".1f"))
        adx_val = res["df"]["ADX"].iloc[-1] if "ADX" in res["df"].columns else None
        c5.metric("ADX", _fmt(adx_val, ".1f"))

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Price & Moving Average")
            chart_cols = ["Close", "MA"]
            if "BB_Upper" in res["df"].columns:
                chart_cols += ["BB_Upper", "BB_Lower"]
            st.line_chart(res["df"][chart_cols])

        with col_right:
            st.subheader("Statistics")
            for k, v in res["stats"].items():
                st.write(f"**{k}:** {v}")
            st.markdown("---")
            buy_n = int((res["df"]["Signal"] == "BUY").sum())
            sell_n = int((res["df"]["Signal"] == "SELL").sum())
            st.write(f"**Buy signals:** {buy_n}")
            st.write(f"**Sell signals:** {sell_n}")
            st.write(f"**Strategy:** {res['strategy'].replace('_', ' ').title()}")

        # Research quick summary
        if res["research"]:
            st.markdown("---")
            r = res["research"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Fundamental Score", f"{r['score']}/100")
            c2.metric("Recommendation", r["recommendation"])
            sector = r["fundamentals"].get("sector", "N/A")
            c3.metric("Sector", sector)

        # Backtest quick summary
        if res["backtest"]:
            st.markdown("---")
            s = res["backtest"]["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Backtest Return", f"{s['total_return_pct']}%")
            c2.metric("Win Rate", f"{s['win_rate']}%")
            c3.metric("Sharpe Ratio", s["sharpe_ratio"])
            c4.metric("Max Drawdown", f"{s['max_drawdown']}%")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Technical Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_tech:
    if res is None:
        st.info("Run analysis first.")
    else:
        st.header("Technical Analysis")
        df = res["df"]

        st.subheader("Price with Bollinger Bands")
        bb_cols = [c for c in ["Close", "BB_MA", "BB_Upper", "BB_Lower"] if c in df.columns]
        st.line_chart(df[bb_cols])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RSI")
            if "RSI" in df.columns:
                st.line_chart(df[["RSI"]])
        with col2:
            st.subheader("MACD")
            macd_cols = [c for c in ["MACD", "MACD_Signal", "MACD_Histogram"] if c in df.columns]
            if macd_cols:
                st.line_chart(df[macd_cols])

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Stochastic Oscillator")
            if "Stoch_K" in df.columns:
                st.line_chart(df[["Stoch_K", "Stoch_D"]])
        with col4:
            st.subheader("ATR & ADX")
            atr_cols = [c for c in ["ATR", "ADX"] if c in df.columns]
            if atr_cols:
                st.line_chart(df[atr_cols])

        st.subheader("Full Data Table")
        display_cols = [c for c in ["Close", "MA", "RSI", "MACD", "BB_Upper", "BB_Lower", "Stoch_K", "ATR", "ADX"] if c in df.columns]
        st.dataframe(df[display_cols].dropna(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Strategy Signals
# ─────────────────────────────────────────────────────────────────────────────
with tab_strategy:
    if res is None:
        st.info("Run analysis first.")
    else:
        st.header("Strategy Signals")
        df = res["df"]
        st.info(get_strategy_description(res["strategy"]))

        # Signal counts
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BUY", int((df["Signal"] == "BUY").sum()))
        c2.metric("SELL", int((df["Signal"] == "SELL").sum()))
        c3.metric("HOLD", int((df["Signal"] == "HOLD").sum()))
        non_hold = df[df["Signal"] != "HOLD"]
        avg_conf = float(non_hold["Confidence"].mean()) if len(non_hold) else 0
        c4.metric("Avg Confidence", f"{avg_conf:.1f}%")

        st.subheader("Price with Signals")
        st.line_chart(df[["Close", "MA"]])

        buy_df = df[df["Signal"] == "BUY"][["Close", "Confidence"]]
        sell_df = df[df["Signal"] == "SELL"][["Close", "Confidence"]]

        col_b, col_s = st.columns(2)
        with col_b:
            st.markdown(f"#### 🟢 Buy Signals ({len(buy_df)})")
            if not buy_df.empty:
                st.dataframe(buy_df, use_container_width=True)
        with col_s:
            st.markdown(f"#### 🔴 Sell Signals ({len(sell_df)})")
            if not sell_df.empty:
                st.dataframe(sell_df, use_container_width=True)

        st.subheader("All Signals Table")
        sig_cols = [c for c in ["Close", "MA", "RSI", "MACD", "Signal", "Confidence"] if c in df.columns]
        st.dataframe(df[sig_cols].dropna(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Company Research
# ─────────────────────────────────────────────────────────────────────────────
with tab_research:
    if res is None:
        st.info("Run analysis first.")
    elif res["research"] is None:
        st.warning("Company research data not available.")
    else:
        r = res["research"]
        f = r["fundamentals"]

        st.header(f"Company Research — {f.get('company_name', symbol)}")
        st.caption(f"{f.get('sector', '')} · {f.get('industry', '')}")

        # Score & recommendation
        c1, c2, c3 = st.columns(3)
        c1.metric("Fundamental Score", f"{r['score']}/100")
        c2.metric("Recommendation", r["recommendation"])
        c3.metric("Analyst Consensus", f.get("analyst_recommendation", "N/A"))

        st.markdown("---")

        # Key metrics in two columns
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Valuation")
            st.write(f"**Market Cap:** ${f.get('market_cap', 0):,.0f}" if f.get('market_cap') else "**Market Cap:** N/A")
            st.write(f"**P/E Ratio:** {_fmt(f.get('pe_ratio'))}")
            st.write(f"**Forward P/E:** {_fmt(f.get('forward_pe'))}")
            st.write(f"**PEG Ratio:** {_fmt(f.get('peg_ratio'))}")
            st.write(f"**52w High:** ${_fmt(f.get('52w_high'))}")
            st.write(f"**52w Low:** ${_fmt(f.get('52w_low'))}")

        with col_r:
            st.subheader("Financials")
            st.write(f"**Revenue:** ${f.get('revenue', 0):,.0f}" if f.get('revenue') else "**Revenue:** N/A")
            st.write(f"**Net Income:** ${f.get('net_income', 0):,.0f}" if f.get('net_income') else "**Net Income:** N/A")
            st.write(f"**Profit Margin:** {f.get('profit_margin', 0):.1%}" if f.get('profit_margin') else "**Profit Margin:** N/A")
            st.write(f"**Debt/Equity:** {_fmt(f.get('debt_to_equity'), '.0f')}")
            st.write(f"**ROE:** {f.get('return_on_equity', 0):.1%}" if f.get('return_on_equity') else "**ROE:** N/A")
            st.write(f"**Beta:** {_fmt(f.get('beta'))}")

        # Strengths & Risks
        st.markdown("---")
        col_str, col_risk = st.columns(2)
        with col_str:
            st.subheader("💪 Strengths")
            for s_item in r.get("strengths", []):
                st.write(f"✅  {s_item}")
            if not r.get("strengths"):
                st.write("_No notable strengths identified._")
        with col_risk:
            st.subheader("⚠️ Risks")
            for r_item in r.get("risks", []):
                st.write(f"🔴  {r_item}")
            if not r.get("risks"):
                st.write("_No notable risks identified._")

        # Peer comparison
        st.markdown("---")
        st.subheader("Peer Comparison")
        peer_df = r.get("peer_comparison")
        if peer_df is not None and not peer_df.empty:
            st.dataframe(peer_df, use_container_width=True)
        else:
            st.write("_Peer data not available._")

        # News
        st.markdown("---")
        st.subheader("Recent News")
        for item in r.get("news", [])[:8]:
            title = item.get("title", "")
            link = item.get("link", "")
            pub = item.get("publisher", "")
            time_ = item.get("publish_time", "")
            st.markdown(f"- **{title}** — _{pub}_ ({time_}) [link]({link})")
        if not r.get("news"):
            st.write("_No recent news._")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Risk Management
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk:
    if res is None:
        st.info("Run analysis first.")
    else:
        st.header("Risk Management")

        # Portfolio-level risk metrics
        rm = res["risk_metrics"]
        st.subheader("Portfolio Risk Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Daily VaR (95%)", f"{rm['var_95']:.2%}")
        c2.metric("CVaR (95%)", f"{rm['cvar_95']:.2%}")
        c3.metric("Sharpe Ratio", f"{rm['sharpe_ratio']:.2f}")
        c4.metric("Sortino Ratio", f"{rm['sortino_ratio']:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Ann. Return", f"{rm['annualised_return']:.2%}")
        c6.metric("Ann. Volatility", f"{rm['annualised_volatility']:.2%}")
        c7.metric("Max Drawdown", f"{rm.get('max_drawdown', 0):.2%}")
        c8.metric("Daily Vol", f"{rm['daily_volatility']:.4f}")

        # Position sizing
        st.markdown("---")
        st.subheader("Position Sizing Calculator")
        sizing = calculate_position_size(
            res["portfolio"].total_value,
            res["latest_price"],
            stop_loss_pct=res["portfolio"].stop_loss_pct,
            max_position_pct=res["portfolio"].max_position_pct,
        )
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Recommended Shares", sizing["shares"])
        s2.metric("Dollar Amount", f"${sizing['dollar_amount']:,.2f}")
        s3.metric("Stop Loss", f"${sizing['stop_loss_price']:.2f}")
        s4.metric("Take Profit", f"${sizing['take_profit_price']:.2f}")

        # Trade validation
        st.markdown("---")
        st.subheader("Trade Validation")
        rv = res["risk_validation"]
        if rv is None:
            st.write("_Latest signal is HOLD — no trade to validate._")
        else:
            if rv["approved"]:
                st.success("✅  Trade APPROVED")
            else:
                st.error("❌  Trade REJECTED")
            if rv["reasons"]:
                st.write("**Rejection Reasons:**")
                for reason in rv["reasons"]:
                    st.write(f"  🔴  {reason}")
            if rv["warnings"]:
                st.write("**Warnings:**")
                for w in rv["warnings"]:
                    st.write(f"  ⚠️  {w}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Backtest
# ─────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    if res is None:
        st.info("Run analysis first.")
    elif res["backtest"] is None:
        st.warning("Backtest did not run successfully.")
    else:
        bt = res["backtest"]
        s = bt["summary"]

        st.header("Backtest Results")

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{s['total_return_pct']}%")
        c2.metric("Total P&L", f"${s['total_pnl']:,.2f}")
        c3.metric("Win Rate", f"{s['win_rate']}%")
        c4.metric("Sharpe Ratio", s["sharpe_ratio"])
        c5.metric("Max Drawdown", f"{s['max_drawdown']}%")

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Total Trades", s["total_trades"])
        c7.metric("Profit Factor", s["profit_factor"])
        c8.metric("Avg Win", f"${s['avg_win']:,.2f}")
        c9.metric("Avg Loss", f"${s['avg_loss']:,.2f}")

        st.markdown("---")

        # Equity curve
        st.subheader("Equity Curve")
        st.line_chart(bt["equity_curve"])

        # Monthly returns
        if not bt["monthly_returns"].empty:
            st.subheader("Monthly Returns")
            monthly = bt["monthly_returns"] * 100
            st.bar_chart(monthly)

        # Trade log
        st.subheader("Trade Log")
        trades_df = trades_to_dataframe(bt["trades"])
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.write("_No trades executed._")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Multi-Symbol Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.header("Multi-Symbol Comparison")

    symbols_to_compare = st.multiselect(
        "Select symbols to compare:",
        options=get_popular_symbols(),
        default=["AAPL", "MSFT"],
    )

    col1, col2 = st.columns(2)
    with col1:
        start_comp = st.date_input("Start", value=pd.Timestamp("2024-01-01"), key="start_comp")
    with col2:
        end_comp = st.date_input("End", value=pd.Timestamp.today(), key="end_comp")

    if st.button("Compare Symbols"):
        if not symbols_to_compare:
            st.warning("Select at least one symbol.")
        else:
            with st.spinner("Fetching comparison data …"):
                comparison_df = pd.DataFrame()
                for sym in symbols_to_compare:
                    df = fetch_and_process_data(sym, start_comp.strftime("%Y-%m-%d"), end_comp.strftime("%Y-%m-%d"))
                    if df is not None:
                        comparison_df[sym] = df["Close"]

                if comparison_df.empty:
                    st.warning("No data found.")
                else:
                    st.subheader("Price Comparison")
                    st.line_chart(comparison_df)

                    # Normalised returns
                    st.subheader("Normalised Returns (base = 100)")
                    normed = comparison_df / comparison_df.iloc[0] * 100
                    st.line_chart(normed)

                    # Correlation
                    st.subheader("Correlation Matrix")
                    st.dataframe(comparison_df.corr(), use_container_width=True)

                    # Stats table
                    st.subheader("Comparative Statistics")
                    stats_rows = []
                    for sym in comparison_df.columns:
                        series = comparison_df[sym].dropna()
                        ret = series.pct_change().dropna()
                        stats_rows.append({
                            "Symbol": sym,
                            "Mean Price": round(series.mean(), 2),
                            "Volatility": round(ret.std() * np.sqrt(252) * 100, 2),
                            "Total Return %": round((series.iloc[-1] / series.iloc[0] - 1) * 100, 2),
                            "Sharpe": round((ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0, 2),
                        })
                    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
