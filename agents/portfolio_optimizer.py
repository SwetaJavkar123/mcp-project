"""
Portfolio Optimizer Agent
==========================
Optimises portfolio allocation across multiple assets.

Supports:
  - Mean-Variance (Markowitz) optimisation
  - Risk-Parity allocation
  - Black-Litterman model
  - Rebalancing recommendations with transaction cost awareness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimiserConfig:
    risk_free_rate: float = 0.04          # annualised
    min_weight: float = 0.0               # no shorts by default
    max_weight: float = 0.40              # max 40 % in one asset
    target_return: float | None = None    # for min-variance @ target
    transaction_cost_bps: float = 10.0    # 10 bps per rebalance trade
    num_portfolios: int = 5_000           # Monte Carlo frontier samples
    trading_days: int = 252


# ---------------------------------------------------------------------------
# Mean-Variance (Markowitz)
# ---------------------------------------------------------------------------

def _annualised_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> tuple[float, float]:
    """Return (annualised_return, annualised_volatility) for a weight vector."""
    ret = np.dot(weights, mean_returns) * trading_days
    vol = np.sqrt(np.dot(weights, np.dot(cov_matrix * trading_days, weights)))
    return float(ret), float(vol)


def _random_weights(n: int, min_w: float = 0.0, max_w: float = 1.0) -> np.ndarray:
    """Generate random portfolio weights that sum to 1, within bounds."""
    for _ in range(200):
        w = np.random.dirichlet(np.ones(n))
        if np.all(w >= min_w) and np.all(w <= max_w):
            return w
    # Fallback: equal weight
    return np.ones(n) / n


def efficient_frontier(
    returns_df: pd.DataFrame,
    config: OptimiserConfig | None = None,
) -> dict:
    """
    Generate a Monte-Carlo efficient frontier and return the optimal portfolios.

    Parameters
    ----------
    returns_df : DataFrame of daily returns, columns = asset symbols.
    config     : OptimiserConfig (or use defaults).

    Returns
    -------
    {
        "frontier": DataFrame (return, volatility, sharpe, weights per row),
        "max_sharpe": {weights, return, volatility, sharpe},
        "min_volatility": {weights, return, volatility, sharpe},
        "symbols": list[str],
    }
    """
    if config is None:
        config = OptimiserConfig()

    symbols = list(returns_df.columns)
    n = len(symbols)
    mean_ret = returns_df.mean().values
    cov = returns_df.cov().values

    records: list[dict] = []

    for _ in range(config.num_portfolios):
        w = _random_weights(n, config.min_weight, config.max_weight)
        ret, vol = _annualised_stats(w, mean_ret, cov, config.trading_days)
        sharpe = (ret - config.risk_free_rate) / vol if vol > 0 else 0.0
        row = {"return": ret, "volatility": vol, "sharpe": sharpe}
        for sym, wi in zip(symbols, w):
            row[f"w_{sym}"] = float(wi)
        records.append(row)

    frontier = pd.DataFrame(records)

    # Best Sharpe
    best_idx = frontier["sharpe"].idxmax()
    best = frontier.loc[best_idx]
    max_sharpe = {
        "weights": {s: float(best[f"w_{s}"]) for s in symbols},
        "return": float(best["return"]),
        "volatility": float(best["volatility"]),
        "sharpe": float(best["sharpe"]),
    }

    # Minimum volatility
    min_idx = frontier["volatility"].idxmin()
    minv = frontier.loc[min_idx]
    min_vol = {
        "weights": {s: float(minv[f"w_{s}"]) for s in symbols},
        "return": float(minv["return"]),
        "volatility": float(minv["volatility"]),
        "sharpe": float(minv["sharpe"]),
    }

    return {
        "frontier": frontier,
        "max_sharpe": max_sharpe,
        "min_volatility": min_vol,
        "symbols": symbols,
    }


# ---------------------------------------------------------------------------
# Risk-Parity allocation
# ---------------------------------------------------------------------------

def risk_parity(
    returns_df: pd.DataFrame,
    config: OptimiserConfig | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """
    Equal risk contribution portfolio.

    Each asset contributes the same amount to total portfolio variance.

    Returns
    -------
    {weights: dict, return, volatility, sharpe, risk_contributions: dict}
    """
    if config is None:
        config = OptimiserConfig()

    symbols = list(returns_df.columns)
    cov = returns_df.cov().values * config.trading_days
    n = len(symbols)
    w = np.ones(n) / n

    for _ in range(max_iter):
        sigma = np.sqrt(np.dot(w, np.dot(cov, w)))
        marginal = np.dot(cov, w)
        risk_contrib = w * marginal / sigma
        target_rc = sigma / n
        w_new = w * (target_rc / risk_contrib)
        w_new = np.clip(w_new, config.min_weight + 1e-10, config.max_weight)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    mean_ret = returns_df.mean().values
    ret, vol = _annualised_stats(w, mean_ret, returns_df.cov().values, config.trading_days)
    sharpe = (ret - config.risk_free_rate) / vol if vol > 0 else 0.0

    sigma = np.sqrt(np.dot(w, np.dot(cov, w)))
    marginal = np.dot(cov, w)
    risk_contrib = w * marginal / sigma if sigma > 0 else np.zeros(n)

    return {
        "weights": {s: float(wi) for s, wi in zip(symbols, w)},
        "return": float(ret),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "risk_contributions": {s: float(rc) for s, rc in zip(symbols, risk_contrib)},
    }


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

def black_litterman(
    returns_df: pd.DataFrame,
    market_caps: dict[str, float],
    views: list[dict] | None = None,
    tau: float = 0.05,
    config: OptimiserConfig | None = None,
) -> dict:
    """
    Black-Litterman model.

    Parameters
    ----------
    returns_df  : Daily returns DataFrame.
    market_caps : {symbol: market_cap_usd} for each column in returns_df.
    views       : List of investor views, each:
                  {"assets": ["AAPL"], "weights": [1.0], "return": 0.10, "confidence": 0.5}
                  For relative views: {"assets": ["AAPL","MSFT"], "weights": [1,-1], ...}
    tau         : Scalar reflecting uncertainty in the prior.

    Returns
    -------
    {weights, return, volatility, sharpe, implied_returns, posterior_returns}
    """
    if config is None:
        config = OptimiserConfig()

    symbols = list(returns_df.columns)
    n = len(symbols)
    cov = returns_df.cov().values * config.trading_days

    # Market-cap weights (equilibrium)
    caps = np.array([market_caps.get(s, 1.0) for s in symbols])
    w_mkt = caps / caps.sum()

    # Risk aversion coefficient
    mkt_ret = float(np.dot(w_mkt, returns_df.mean().values)) * config.trading_days
    mkt_vol = float(np.sqrt(np.dot(w_mkt, np.dot(cov, w_mkt))))
    delta = (mkt_ret - config.risk_free_rate) / (mkt_vol ** 2) if mkt_vol > 0 else 2.5

    # Implied equilibrium returns
    pi = delta * np.dot(cov, w_mkt)

    if views and len(views) > 0:
        # Build P (pick matrix), Q (view returns), Omega (view uncertainty)
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, v in enumerate(views):
            for asset, weight in zip(v["assets"], v["weights"]):
                if asset in symbols:
                    P[i, symbols.index(asset)] = weight
            Q[i] = v["return"]
            confidence = v.get("confidence", 0.5)
            omega_diag[i] = (1 - confidence) / confidence * np.dot(
                P[i], np.dot(tau * cov, P[i])
            )

        Omega = np.diag(omega_diag)
        tau_cov = tau * cov
        tau_cov_inv = np.linalg.inv(tau_cov)
        Pt_Omega_inv = np.dot(P.T, np.linalg.inv(Omega))

        posterior_mean = np.linalg.inv(tau_cov_inv + np.dot(Pt_Omega_inv, P)).dot(
            np.dot(tau_cov_inv, pi) + np.dot(Pt_Omega_inv, Q)
        )
    else:
        posterior_mean = pi

    # Optimal weights from posterior
    posterior_cov = cov  # simplified
    try:
        w_bl = np.linalg.solve(delta * posterior_cov, posterior_mean)
    except np.linalg.LinAlgError:
        w_bl = w_mkt

    # Clip and normalise
    w_bl = np.clip(w_bl, config.min_weight, config.max_weight)
    if w_bl.sum() > 0:
        w_bl /= w_bl.sum()
    else:
        w_bl = np.ones(n) / n

    ret, vol = _annualised_stats(w_bl, returns_df.mean().values, returns_df.cov().values, config.trading_days)
    sharpe = (ret - config.risk_free_rate) / vol if vol > 0 else 0.0

    return {
        "weights": {s: float(wi) for s, wi in zip(symbols, w_bl)},
        "return": float(ret),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "implied_returns": {s: float(r) for s, r in zip(symbols, pi)},
        "posterior_returns": {s: float(r) for s, r in zip(symbols, posterior_mean)},
    }


# ---------------------------------------------------------------------------
# Rebalancing recommendations (with transaction cost awareness)
# ---------------------------------------------------------------------------

def rebalance_portfolio(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value: float,
    prices: dict[str, float],
    config: OptimiserConfig | None = None,
) -> dict:
    """
    Generate rebalancing trades from current → target allocation.

    Accounts for transaction costs: only recommends a trade if the benefit
    of rebalancing exceeds the round-trip transaction cost.

    Returns
    -------
    {
        trades: list of {symbol, action, shares, dollar_amount, weight_change},
        total_cost: estimated transaction cost,
        turnover: total absolute weight change / 2,
    }
    """
    if config is None:
        config = OptimiserConfig()

    all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
    trades = []
    total_turnover = 0.0
    total_cost = 0.0

    cost_rate = config.transaction_cost_bps / 10_000  # convert bps to decimal

    for sym in sorted(all_symbols):
        cur = current_weights.get(sym, 0.0)
        tgt = target_weights.get(sym, 0.0)
        delta = tgt - cur

        # Skip tiny rebalances (cost exceeds benefit)
        dollar_change = abs(delta) * portfolio_value
        trade_cost = dollar_change * cost_rate
        if abs(delta) < 0.005:  # less than 0.5% weight change → skip
            continue

        price = prices.get(sym, 0.0)
        shares = int(dollar_change / price) if price > 0 else 0

        if shares == 0:
            continue

        action = "BUY" if delta > 0 else "SELL"
        trades.append({
            "symbol": sym,
            "action": action,
            "shares": shares,
            "dollar_amount": round(shares * price, 2),
            "weight_change": round(delta * 100, 2),  # in %
            "estimated_cost": round(trade_cost, 2),
        })
        total_turnover += abs(delta)
        total_cost += trade_cost

    return {
        "trades": trades,
        "total_cost": round(total_cost, 2),
        "turnover": round(total_turnover / 2, 4),  # one-way turnover
        "num_trades": len(trades),
    }


# ---------------------------------------------------------------------------
# Convenience: run all optimisation methods at once
# ---------------------------------------------------------------------------

def optimise_portfolio(
    returns_df: pd.DataFrame,
    market_caps: dict[str, float] | None = None,
    views: list[dict] | None = None,
    current_weights: dict[str, float] | None = None,
    portfolio_value: float = 100_000.0,
    prices: dict[str, float] | None = None,
    config: OptimiserConfig | None = None,
) -> dict:
    """
    Run all optimisation models and return consolidated results.

    Returns
    -------
    {
        mean_variance: efficient_frontier result,
        risk_parity: risk_parity result,
        black_litterman: BL result (if market_caps provided),
        rebalance: rebalance recommendation (if current_weights provided),
    }
    """
    if config is None:
        config = OptimiserConfig()

    result: dict = {}

    # 1. Mean-Variance
    result["mean_variance"] = efficient_frontier(returns_df, config)

    # 2. Risk-Parity
    result["risk_parity"] = risk_parity(returns_df, config)

    # 3. Black-Litterman (needs market caps)
    if market_caps:
        result["black_litterman"] = black_litterman(
            returns_df, market_caps, views=views, config=config,
        )

    # 4. Rebalancing recommendations
    if current_weights and prices:
        # Default target = max-sharpe from mean-variance
        target = result["mean_variance"]["max_sharpe"]["weights"]
        result["rebalance"] = rebalance_portfolio(
            current_weights, target, portfolio_value, prices, config,
        )

    return result
