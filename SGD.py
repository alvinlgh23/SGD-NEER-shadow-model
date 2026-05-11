import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ============================================================
# Shadow S$NEER proxy
# ============================================================
# MAS does not disclose the official S$NEER basket, weights, centre, slope, or
# band width. This script therefore builds a transparent proxy from observable
# SGD crosses and explicit approximate trade/competition weights.
#
# Yahoo Finance tickers below are quoted as foreign currency per SGD
# (for example SGDUSD=X ~= USD per SGD). A rise therefore means SGD strength.

START_DATE = "2022-01-01"
OUTPUT_FILE = "sgd_neer_dashboard.png"

POLICY_CHANGE = pd.Timestamp("2026-04-14")

# Approximate proxy weights. These are intentionally explicit and normalized
# again in code. Treat them as a modelling input, not an official MAS basket.
PROXY_BASKET = {
    "SGDUSD=X": ("USD", 0.170),
    "SGDCNY=X": ("CNY", 0.205),
    "SGDMYR=X": ("MYR", 0.125),
    "SGDEUR=X": ("EUR", 0.105),
    "SGDJPY=X": ("JPY", 0.070),
    "SGDKRW=X": ("KRW", 0.060),
    "SGDTWD=X": ("TWD", 0.055),
    "SGDIDR=X": ("IDR", 0.050),
    "SGDTHB=X": ("THB", 0.045),
    "SGDAUD=X": ("AUD", 0.040),
    "SGDINR=X": ("INR", 0.040),
    "SGDGBP=X": ("GBP", 0.035),
}

# Analyst-style policy slope assumptions for the illustrative centre line.
# MAS's public wording on 14 Apr 2026 was that it would increase the rate of
# appreciation slightly while keeping width and centre unchanged.
SLOPE_BEFORE_POLICY_CHANGE = 0.005  # 0.5% p.a. proxy
SLOPE_AFTER_POLICY_CHANGE = 0.010   # 1.0% p.a. proxy
FIXED_BAND = 0.02                   # ±2% proxy commonly used by market analysts


def r2_score_np(actual, fitted):
    actual = np.asarray(actual, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(fitted)
    actual = actual[mask]
    fitted = fitted[mask]
    if len(actual) < 2:
        return np.nan
    ss_res = np.sum((actual - fitted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else np.nan


def normalize_weights(basket):
    weights = pd.Series({ccy: weight for _, (ccy, weight) in basket.items()})
    return weights / weights.sum()


def download_sgd_crosses(basket):
    tickers = list(basket.keys())
    labels = {ticker: ccy for ticker, (ccy, _) in basket.items()}

    print("Downloading SGD FX crosses from Yahoo Finance...")
    downloaded = yf.download(
        tickers,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
        timeout=30,
    )

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise RuntimeError("Yahoo response did not include Close prices.")
        raw = downloaded["Close"].copy()
    else:
        raw = downloaded[["Close"]].copy()
        raw.columns = tickers

    raw = raw.rename(columns=labels)
    raw = raw.reindex(columns=list(labels.values()))

    missing = raw.columns[raw.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"No price history returned for: {', '.join(missing)}")

    gap_ratio = raw.isna().mean().sort_values(ascending=False)
    data = raw.ffill().dropna()

    if len(data) < 260:
        raise RuntimeError("Not enough overlapping data to build a stable index.")

    return data, gap_ratio


def build_shadow_neer(data, weights):
    log_levels = np.log(data)
    base = log_levels.iloc[0]
    weighted_log_index = (log_levels.subtract(base) * weights).sum(axis=1)
    index = np.exp(weighted_log_index) * 100
    index.name = "Shadow_NEER"

    component_returns = np.log(data / data.shift(1)).dropna()
    weighted_returns = component_returns.mul(weights, axis=1).sum(axis=1)
    weighted_returns.name = "Shadow_NEER_Log_Return"
    return index, weighted_returns, component_returns


def build_policy_center(index):
    dates = index.index

    def center_from_anchor(anchor_val):
        anchor = float(np.atleast_1d(anchor_val)[0])
        centers = np.empty(len(dates))
        centers[0] = anchor
        for i in range(1, len(dates)):
            slope = (
                SLOPE_AFTER_POLICY_CHANGE
                if dates[i] >= POLICY_CHANGE
                else SLOPE_BEFORE_POLICY_CHANGE
            )
            days = max((dates[i] - dates[i - 1]).days, 1)
            centers[i] = centers[i - 1] * np.exp(slope * days / 365.25)
        return centers

    opt = minimize(
        lambda anchor: np.sum((index.values - center_from_anchor(anchor)) ** 2),
        x0=[index.iloc[0]],
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    return pd.Series(center_from_anchor(opt.x), index=dates, name="Estimated_Center"), opt


def latest_snapshot(index, returns, center, weights, data, gap_ratio):
    latest_date = index.index[-1].date()
    latest_gap = ((index.iloc[-1] / center.iloc[-1]) - 1) * 100
    latest_vol = returns.rolling(21).std().iloc[-1] * np.sqrt(252) * 100

    print(f"\n{'=' * 62}")
    print("SHADOW S$NEER PROXY - LATEST SNAPSHOT")
    print("=" * 62)
    print(f"Date: {latest_date}")
    print(f"Index: {index.iloc[-1]:.3f}")
    print(f"Estimated centre: {center.iloc[-1]:.3f}")
    print(f"Distance from estimated centre: {latest_gap:+.2f}%")
    print(f"21d annualised volatility: {latest_vol:.2f}%")
    print(f"Latest USD per SGD: {data['USD'].iloc[-1]:.4f}")

    print(f"\n{'=' * 62}")
    print("PROXY BASKET WEIGHTS")
    print("=" * 62)
    for ccy, weight in weights.sort_values(ascending=False).items():
        bar = "#" * int(round(weight * 60))
        print(f"  {ccy:>4s}  {weight:6.2%}  {bar}")

    notable_gaps = gap_ratio[gap_ratio > 0.01]
    if not notable_gaps.empty:
        print("\nData gaps forward-filled before overlap trimming:")
        for ccy, ratio in notable_gaps.items():
            print(f"  {ccy:>4s}: {ratio:.1%}")


def plot_dashboard(data, index, returns, component_returns, center, weights):
    fixed_upper = center * (1 + FIXED_BAND)
    fixed_lower = center * (1 - FIXED_BAND)

    rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
    dynamic_half_width = (
        returns.rolling(30).std() * np.sqrt(30) * 1.5 * 100
    ).clip(lower=0.5)
    dynamic_upper = center * (1 + dynamic_half_width / 100)
    dynamic_lower = center * (1 - dynamic_half_width / 100)
    center_r2 = r2_score_np(index, center)

    contribution_63d = component_returns.mul(weights, axis=1).rolling(63).sum() * 100
    usd_per_sgd = data["USD"].reindex(index.index)
    usd_per_sgd_norm = usd_per_sgd / usd_per_sgd.iloc[0] * 100
    deviation = (index / center - 1) * 100
    latest_contrib = contribution_63d.iloc[-1].sort_values()
    latest_gap = deviation.iloc[-1]
    latest_vol = rolling_vol.iloc[-1]
    latest_date = index.index[-1].strftime("%d %b %Y")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1318")

    grid = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.35, 1.0, 1.0],
        hspace=0.46,
        wspace=0.22,
        left=0.055,
        right=0.965,
        top=0.79,
        bottom=0.075,
    )

    ax_neer = fig.add_subplot(grid[0, :])
    ax_weights = fig.add_subplot(grid[1, 0])
    ax_contrib = fig.add_subplot(grid[1, 1])
    ax_vol = fig.add_subplot(grid[2, 0])
    ax_compare = fig.add_subplot(grid[2, 1])

    teal = "#15c8b5"
    orange = "#ff9f43"
    red = "#f06565"
    green = "#62d394"
    gold = "#f4c95d"
    white = "#e7edf5"
    muted = "#9aa6b6"
    panel = "#171c24"
    grid_color = "#2b3340"

    def style_ax(ax, title):
        ax.set_facecolor(panel)
        ax.set_title(title, color=white, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#323b49")
        ax.grid(True, color=grid_color, linewidth=0.55, alpha=0.7)

    def add_kpi(x, label, value, subtext, accent):
        fig.text(x, 0.925, label.upper(), color=muted, fontsize=7.5, fontweight="bold")
        fig.text(x, 0.898, value, color=white, fontsize=15, fontweight="bold")
        fig.text(x, 0.875, subtext, color=accent, fontsize=8, fontweight="bold")

    fig.text(
        0.055,
        0.975,
        "S$NEER Shadow Proxy",
        color=white,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.948,
        "Trade-weighted proxy. MAS basket, centre and band are undisclosed.",
        color=muted,
        fontsize=9,
    )
    add_kpi(0.055, "Latest date", latest_date, "Yahoo FX closes", teal)
    add_kpi(0.22, "Proxy index", f"{index.iloc[-1]:.2f}", "Jan 2022 = 100", teal)
    add_kpi(
        0.385,
        "Vs centre",
        f"{latest_gap:+.2f}%",
        "above estimate" if latest_gap >= 0 else "below estimate",
        green if latest_gap >= 0 else red,
    )
    add_kpi(0.55, "21d vol", f"{latest_vol:.2f}%", "annualised", gold)

    style_ax(
        ax_neer,
        f"Proxy Path vs Estimated Policy Band | Centre Fit R2 = {center_r2:.3f}",
    )
    ax_neer.fill_between(
        index.index,
        fixed_lower,
        fixed_upper,
        color="#dbe4ee",
        alpha=0.05,
        label="Estimated fixed band (+/-2%)",
    )
    ax_neer.fill_between(
        index.index,
        dynamic_lower,
        dynamic_upper,
        color=teal,
        alpha=0.075,
        label="Vol-adjusted stress band",
    )
    ax_neer.plot(index, color=teal, lw=2.1, label="Shadow S$NEER proxy")
    ax_neer.plot(center, color=white, lw=1.25, ls=":", alpha=0.8, label="Estimated centre")
    ax_neer.plot(fixed_upper, color=red, lw=0.85, ls="--", alpha=0.55)
    ax_neer.plot(fixed_lower, color=green, lw=0.85, ls="--", alpha=0.55)
    ax_neer.axvline(POLICY_CHANGE, color=orange, lw=1.4, label="MAS slope increase (14 Apr 2026)")
    ax_neer.scatter(index.index[-1], index.iloc[-1], s=34, color=teal, edgecolor=white, linewidth=0.8, zorder=5)
    ax_neer.annotate(
        f"{index.iloc[-1]:.2f}",
        xy=(index.index[-1], index.iloc[-1]),
        xytext=(-48, 14),
        textcoords="offset points",
        color=white,
        fontsize=8.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=teal, lw=0.8),
    )
    ax_neer.set_ylabel("Index, Jan 2022 = 100", color=muted, fontsize=9)
    ax_neer.legend(loc="upper left", fontsize=7.5, framealpha=0.18, ncol=4)

    style_ax(ax_weights, "Explicit Proxy Basket Weights")
    sorted_weights = weights.sort_values()
    bar_colors = [teal if value < sorted_weights.max() else gold for value in sorted_weights.values]
    ax_weights.barh(sorted_weights.index, sorted_weights.values, color=bar_colors, alpha=0.9)
    for y, value in enumerate(sorted_weights.values):
        ax_weights.text(value + 0.002, y, f"{value:.1%}", color=muted, va="center", fontsize=8)
    ax_weights.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_weights.set_xlim(0, max(sorted_weights.max() * 1.22, 0.22))

    style_ax(ax_contrib, "Latest 63-Day Weighted Contributions")
    contrib_colors = [green if value >= 0 else red for value in latest_contrib.values]
    ax_contrib.barh(latest_contrib.index, latest_contrib.values, color=contrib_colors, alpha=0.88)
    ax_contrib.axvline(0, color="#7a8594", lw=0.9)
    for y, value in enumerate(latest_contrib.values):
        ha = "left" if value >= 0 else "right"
        offset = 0.015 if value >= 0 else -0.015
        ax_contrib.text(value + offset, y, f"{value:+.2f}%", color=muted, va="center", ha=ha, fontsize=8)
    contrib_limit = max(abs(latest_contrib.min()), abs(latest_contrib.max())) * 1.35
    ax_contrib.set_xlim(-contrib_limit, contrib_limit)
    ax_contrib.set_xlabel("Contribution to 63-day proxy return", color=muted, fontsize=8)

    style_ax(ax_vol, "Volatility and Centre Deviation")
    ax_vol.fill_between(rolling_vol.index, rolling_vol, color=teal, alpha=0.2)
    ax_vol.plot(rolling_vol, color=teal, lw=1.45, label="21d annualised volatility")
    ax_vol.plot(deviation.abs(), color=gold, lw=1.25, alpha=0.95, label="Absolute centre deviation")
    ax_vol.axvline(POLICY_CHANGE, color=orange, lw=1.2)
    ax_vol.set_ylabel("%", color=muted, fontsize=9)
    ax_vol.legend(loc="upper left", fontsize=7.5, framealpha=0.18)

    style_ax(ax_compare, "USD per SGD vs Trade-Weighted SGD Proxy")
    ax_compare.plot(usd_per_sgd_norm, color=gold, lw=1.35, label="USD per SGD, normalized")
    ax_compare.plot(index, color=teal, lw=1.8, label="Shadow S$NEER proxy")
    ax_compare.fill_between(index.index, usd_per_sgd_norm, index, color=white, alpha=0.035)
    ax_compare.axvline(POLICY_CHANGE, color=orange, lw=1.2)
    ax_compare.set_ylabel("Index, Jan 2022 = 100", color=muted, fontsize=9)
    ax_compare.legend(loc="upper left", fontsize=7.5, framealpha=0.18)

    fig.text(
        0.055,
        0.03,
        "Method: geometric weighted average of observable SGD crosses. "
        "MAS basket, weights, centre, slope and band width are undisclosed; policy band shown is an analyst-style estimate.",
        color=muted,
        fontsize=8,
    )

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved -> {OUTPUT_FILE}")
    plt.show()


def main():
    weights = normalize_weights(PROXY_BASKET)
    data, gap_ratio = download_sgd_crosses(PROXY_BASKET)
    index, returns, component_returns = build_shadow_neer(data, weights)
    center, opt = build_policy_center(index)

    print(f"Data loaded: {len(data)} observations, {len(data.columns)} SGD crosses")
    print(f"Date range: {data.index[0].date()} -> {data.index[-1].date()}")
    print(f"Policy-centre anchor optimisation success: {opt.success}")

    latest_snapshot(index, returns, center, weights, data, gap_ratio)
    plot_dashboard(data, index, returns, component_returns, center, weights)


if __name__ == "__main__":
    main()
