"""Offline tests on synthetic data (no Yahoo calls)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basket_inference import constrained_weights, infer_basket, ols_with_se  # noqa: E402
from mps_calendar import MPSEvent  # noqa: E402
from policy_band import build_piecewise_center, implied_regime_slopes, slope_schedule  # noqa: E402
import SGD  # noqa: E402

TRUE_W = {"USD": 0.4, "EUR": 0.3, "CNY": 0.2, "JPY": 0.1}


def synthetic_basket(weeks=400, drift_pa=1.5, noise=0.0005, seed=0):
    """Business-day per-USD quotes where SGD tracks a known basket vs CHF."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=weeks * 5)
    n = len(dates)
    # Log value of each currency in CHF: random walks.
    log_chf = pd.DataFrame({c: np.cumsum(rng.normal(0, 0.004, n)) for c in TRUE_W}, index=dates)
    basket = sum(w * log_chf[c] for c, w in TRUE_W.items())
    t = np.arange(n) / 261.0
    log_sgd = basket + drift_pa / 100 * t + rng.normal(0, noise, n)
    # Convert to "units per USD": per_usd[c] = CHF/USD divided by CHF/c.
    chf_per_usd = np.exp(log_chf["USD"])
    per_usd = pd.DataFrame({c: chf_per_usd / np.exp(log_chf[c]) for c in TRUE_W if c != "USD"})
    per_usd["SGD"] = chf_per_usd / np.exp(log_sgd)
    per_usd["CHF"] = chf_per_usd
    per_usd["USD"] = 1.0
    return per_usd


def test_frankel_wei_recovers_known_weights():
    res = infer_basket(currencies=list(TRUE_W), window=104, step=13, per_usd=synthetic_basket())
    for ccy, w in TRUE_W.items():
        assert res["full_weights"][ccy] == pytest.approx(w, abs=0.02)
        assert res["ols"].loc[ccy, "coef"] == pytest.approx(w, abs=0.03)
    assert res["full_r2"] > 0.95
    assert res["full_implied_slope_pct"] == pytest.approx(1.5, abs=0.5)
    assert np.allclose(res["rolling_weights"].sum(axis=1), 1.0)
    assert (res["rolling_weights"] >= 0).all().all()


def test_constrained_weights_are_valid_simplex():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = X @ np.array([0.7, 0.3, -0.2]) + rng.normal(0, 0.1, 200)  # negative true weight
    w, _, _ = constrained_weights(pd.Series(y), X)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    table, r2 = ols_with_se(pd.Series(y), X)
    assert table.loc["c", "coef"] < 0  # unconstrained OLS keeps the sign
    assert 0 < r2 <= 1


def _events():
    return [
        MPSEvent(pd.Timestamp("2022-03-01"), "slope_up", 2.0, "up"),
        MPSEvent(pd.Timestamp("2022-07-01"), "recentre", 2.0, "recentre"),
        MPSEvent(pd.Timestamp("2022-09-01"), "hold", 2.0, "hold"),
    ]


def test_piecewise_center_follows_slopes_and_recentres():
    dates = pd.bdate_range("2022-01-03", "2022-12-30")
    t = (dates - dates[0]).days.to_numpy() / 365.25
    # Index grows 1%/yr, then jumps +3% at the re-centre date.
    jump = np.where(dates >= pd.Timestamp("2022-07-01"), 0.03, 0.0)
    index = pd.Series(100 * np.exp(0.01 * t + jump), index=dates)

    center, info = build_piecewise_center(index, _events(), initial_slope=1.0)

    # Re-centre resets the level to the prevailing index (geometric mean of last 5 obs).
    k = dates.searchsorted(pd.Timestamp("2022-07-01"))
    assert center.iloc[k] == pytest.approx(np.exp(np.log(index.iloc[k - 5:k]).mean()), rel=1e-9)
    # After the re-centre the centre grows at exactly 2% p.a.
    days = (dates[-1] - dates[k]).days
    assert np.log(center.iloc[-1] / center.iloc[k]) == pytest.approx(0.02 * days / 365.25, rel=1e-6)
    # "hold" events do not change the slope schedule.
    sched = slope_schedule(dates, _events(), 1.0)
    assert sched[pd.Timestamp("2022-02-01")] == 1.0
    assert sched[pd.Timestamp("2022-12-30")] == 2.0
    assert set(info["recentre_levels"]) == {pd.Timestamp("2022-07-01")}


def test_implied_regime_slopes_match_true_trend():
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    t = (dates - dates[0]).days.to_numpy() / 365.25
    index = pd.Series(100 * np.exp(0.015 * t), index=dates)
    table = implied_regime_slopes(index, _events(), 1.0)
    assert np.allclose(table["implied_slope_pct"].dropna(), 1.5, atol=1e-6)
    assert list(table["assumed_slope_pct"]) == [1.0, 2.0, 2.0]  # hold rows excluded


def test_chain_linked_index_equals_geometric_for_constant_weights():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2022-01-03", periods=50)
    data = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (50, 3)), axis=0)),
                        index=dates, columns=list("abc"))
    w = pd.Series([0.5, 0.3, 0.2], index=list("abc"))
    idx, _, _ = SGD.build_shadow_neer(data, w)
    geometric = np.exp((np.log(data) - np.log(data.iloc[0])).mul(w).sum(axis=1)) * 100
    assert np.allclose(idx.values, geometric.values)
    # Time-varying weights equal to the constant ones give the same index.
    path = pd.DataFrame([w.values] * len(dates), index=dates, columns=w.index)
    idx2, _, _ = SGD.build_shadow_neer(data, path)
    assert np.allclose(idx.values, idx2.values)


def test_daily_weight_path_has_no_lookahead():
    roll = pd.DataFrame({"a": [0.2, 0.8], "b": [0.8, 0.2]},
                        index=pd.to_datetime(["2022-01-07", "2022-01-14"]))
    dates = pd.bdate_range("2022-01-10", "2022-01-18")
    path = SGD.daily_weight_path(roll, dates, ["a", "b"])
    assert path.loc["2022-01-14", "a"] == 0.2  # Friday's own window not yet usable
    assert path.loc["2022-01-17", "a"] == 0.8
