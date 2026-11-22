"""Frankel-Wei inference of the implicit S$NEER basket weights.

Idea: if MAS manages SGD against a basket, then, measured in an outside
numeraire (CHF here), SGD's returns should be a weighted average of the basket
currencies' returns plus a drift (the policy slope):

    dln(CHF per SGD) = c + sum_k w_k * dln(CHF per ccy_k) + e

The weights w_k are the implicit basket. We estimate them with:
- unconstrained OLS (with standard errors) on the full sample, and
- non-negative, sum-to-one constrained least squares on a rolling window,
  which gives a stable, interpretable weight path over time.

The intercept c, annualised, is a data-implied estimate of the band slope.

All crosses are pulled as USD-quoted Yahoo tickers ("XXX=X" = XXX per USD) so
the dependent variable is not mechanically built from the regressors.
Weekly returns are used because Asian and Western FX closes are asynchronous.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

NUMERAIRE = "CHF"
CANDIDATES = ["USD", "CNY", "MYR", "EUR", "JPY", "KRW", "TWD", "IDR", "THB", "AUD", "INR", "GBP"]
WEEKS_PER_YEAR = 52


def download_usd_quotes(currencies, start):
    """Units of each currency per USD, daily, forward-filled business days."""
    import yfinance as yf

    tickers = {f"{c}=X": c for c in currencies if c != "USD"}
    raw = yf.download(list(tickers), start=start, progress=False,
                      auto_adjust=False, threads=False, timeout=30)["Close"]
    raw = raw.rename(columns=tickers)
    raw = raw[raw.index.dayofweek < 5]
    missing = [c for c in tickers.values() if c not in raw or raw[c].isna().all()]
    if missing:
        raise RuntimeError(f"No Yahoo history for: {', '.join(missing)}")
    per_usd = raw.ffill().dropna()
    per_usd["USD"] = 1.0
    return per_usd


def numeraire_values(per_usd, numeraire=NUMERAIRE):
    """Value of each currency expressed in the numeraire (numeraire per unit)."""
    return per_usd[numeraire].to_numpy()[:, None] / per_usd.drop(columns=[numeraire])


def weekly_log_returns(values):
    weekly = values.resample("W-FRI").last().dropna()
    return np.log(weekly).diff().dropna()


def ols_with_se(y, X):
    """OLS with intercept. Returns DataFrame of coef/se/t plus R2."""
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy()])
    beta, *_ = np.linalg.lstsq(Xc, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - Xc @ beta
    dof = len(y) - Xc.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.diag(cov))
    names = ["const"] + list(X.columns)
    table = pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=names)
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return table, float(r2)


def constrained_weights(y, X):
    """Weights >= 0 summing to 1, plus a free intercept. Returns (weights, const, r2)."""
    Xv, yv = X.to_numpy(), y.to_numpy()
    k = Xv.shape[1]

    def sse(params):
        resid = yv - params[0] - Xv @ params[1:]
        return resid @ resid

    x0 = np.concatenate([[0.0], np.full(k, 1.0 / k)])
    res = minimize(
        sse, x0, method="SLSQP",
        bounds=[(None, None)] + [(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda p: p[1:].sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    params = res.x
    weights = pd.Series(np.clip(params[1:], 0, None), index=X.columns)
    weights /= weights.sum()
    resid = yv - params[0] - Xv @ weights.to_numpy()
    r2 = 1 - (resid @ resid) / np.sum((yv - yv.mean()) ** 2)
    return weights, float(params[0]), float(r2)


def rolling_constrained_weights(y, X, window=104, step=4):
    rows, consts, r2s = {}, {}, {}
    for end in range(window, len(y) + 1, step):
        sl = slice(end - window, end)
        w, c, r2 = constrained_weights(y.iloc[sl], X.iloc[sl])
        date = y.index[end - 1]
        rows[date], consts[date], r2s[date] = w, c, r2
    weights = pd.DataFrame(rows).T
    drift = pd.Series(consts) * WEEKS_PER_YEAR * 100
    return weights, drift.rename("implied_slope_pct"), pd.Series(r2s, name="r2")


def infer_basket(start="2018-01-01", currencies=CANDIDATES, window=104, step=4,
                 per_usd=None):
    """Run the full inference. ``per_usd`` can be injected for testing."""
    if per_usd is None:
        per_usd = download_usd_quotes(list(currencies) + ["SGD", NUMERAIRE], start)
    values = numeraire_values(per_usd)
    returns = weekly_log_returns(values)
    y = returns["SGD"]
    X = returns[list(currencies)]

    ols_table, ols_r2 = ols_with_se(y, X)
    full_w, full_c, full_r2 = constrained_weights(y, X)
    roll_w, roll_drift, roll_r2 = rolling_constrained_weights(y, X, window, step)

    return {
        "returns": returns,
        "ols": ols_table,
        "ols_r2": ols_r2,
        "full_weights": full_w,
        "full_implied_slope_pct": full_c * WEEKS_PER_YEAR * 100,
        "full_r2": full_r2,
        "rolling_weights": roll_w,
        "rolling_implied_slope_pct": roll_drift,
        "rolling_r2": roll_r2,
        "window_weeks": window,
    }
