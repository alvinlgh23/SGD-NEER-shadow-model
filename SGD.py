import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from basket_inference import infer_basket
from mps_calendar import INITIAL_SLOPE, MPS_EVENTS
from policy_band import band_diagnostics, build_piecewise_center, implied_regime_slopes, slope_schedule

warnings.filterwarnings("ignore")

# ============================================================
# Shadow S$NEER proxy
# ============================================================
# MAS does not disclose the official S$NEER basket, weights, centre, slope, or
# band width. This script therefore:
#   1. infers the implicit basket weights from FX data (Frankel-Wei regression,
#      see basket_inference.py),
#   2. builds a geometric shadow index from observable SGD crosses, and
#   3. draws a piecewise policy band driven by the MAS policy-statement
#      calendar (see mps_calendar.py and policy_band.py).
#
# Yahoo Finance tickers below are quoted as foreign currency per SGD
# (for example SGDUSD=X ~= USD per SGD). A rise therefore means SGD strength.

START_DATE = "2022-01-01"
INFERENCE_START = "2018-01-01"  # longer history for the basket regression
OUTPUT_FILE = "sgd_neer_dashboard.png"
INFERENCE_FILE = "sgd_basket_inference.png"

# Which weights build the shadow index:
#   "inferred_rolling" - Frankel-Wei weights re-estimated on a rolling window,
#                        applied with a one-day lag (no look-ahead), chain-linked
#   "inferred_full"    - one set of Frankel-Wei weights from the whole sample
#   "assumed"          - the hand-set PROXY_BASKET weights below
WEIGHT_MODE = "inferred_rolling"
ROLLING_WINDOW_WEEKS = 104

# Hand-set reference weights (kept for comparison with the inferred basket).
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

FIXED_BAND = 0.02  # +/-2% half-width, common market estimate (undisclosed)


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


def daily_weight_path(rolling_weights, dates, columns):
    """Rolling regression weights mapped onto daily dates with a one-day lag.

    A window ending on Friday t is only usable from the next trading day, so
    the index never uses weights estimated with future data.
    """
    shifted = rolling_weights.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    path = shifted.reindex(shifted.index.union(dates)).ffill().reindex(dates)
    path = path[columns]
    if path.isna().any().any():
        raise RuntimeError("Rolling weights do not cover the index sample; extend INFERENCE_START.")
    return path


def build_shadow_neer(data, weights):
    """Chain-linked geometric index.

    ``weights`` is either a Series (constant basket) or a DataFrame of daily
    weights aligned to ``data``. With constant weights this equals the plain
    weighted geometric average of levels.
    """
    component_returns = np.log(data / data.shift(1)).dropna()
    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(component_returns.index)
        weighted_returns = (component_returns * w).sum(axis=1)
    else:
        weighted_returns = component_returns.mul(weights, axis=1).sum(axis=1)
    weighted_returns.name = "Shadow_NEER_Log_Return"

    log_index = pd.concat([pd.Series([0.0], index=data.index[:1]), weighted_returns.cumsum()])
    index = np.exp(log_index) * 100
    index.name = "Shadow_NEER"
    return index, weighted_returns, component_returns


def latest_weights(weights):
    return weights.iloc[-1] if isinstance(weights, pd.DataFrame) else weights


def print_section(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def latest_snapshot(index, returns, center, weights, assumed, inference, regimes, band, data, gap_ratio):
    latest_vol = returns.rolling(21).std().iloc[-1] * np.sqrt(252) * 100

    print_section(f"SHADOW S$NEER PROXY - LATEST SNAPSHOT ({WEIGHT_MODE} weights)")
    print(f"Date: {index.index[-1].date()}")
    print(f"Index: {index.iloc[-1]:.3f}   Estimated centre: {center.iloc[-1]:.3f}")
    print(f"Distance from estimated centre: {band['latest_deviation_pct']:+.2f}%  (band +/-{FIXED_BAND:.0%})")
    print(f"Share of days outside band: {band['share_outside_band']:.1%}   "
          f"Max |deviation|: {band['max_deviation_pct']:.2f}%")
    print(f"21d annualised volatility: {latest_vol:.2f}%   Latest USD per SGD: {data['USD'].iloc[-1]:.4f}")

    print_section("BASKET WEIGHTS: ASSUMED vs INFERRED (Frankel-Wei, CHF numeraire)")
    current = latest_weights(weights)
    table = pd.DataFrame({
        "assumed": assumed,
        "inferred_full": inference["full_weights"],
        "inferred_latest": inference["rolling_weights"].iloc[-1],
        "used_now": current,
    }).sort_values("inferred_latest", ascending=False)
    print(table.map(lambda v: f"{v:6.1%}").to_string())
    print(f"\nFull-sample fit R2 = {inference['full_r2']:.3f}; implied band slope = "
          f"{inference['full_implied_slope_pct']:.2f}% p.a.")
    print(f"Latest {inference['window_weeks']}-week window: R2 = {inference['rolling_r2'].iloc[-1]:.3f}, "
          f"implied slope = {inference['rolling_implied_slope_pct'].iloc[-1]:.2f}% p.a.")

    print_section("UNCONSTRAINED OLS (coef, standard error, t-stat)")
    print(inference["ols"].round(4).to_string())

    print_section("POLICY REGIMES: ASSUMED vs DATA-IMPLIED SLOPE (% p.a.)")
    print(regimes.round(2).to_string(index=False))
    unverified = [e for e in MPS_EVENTS if not e.verified]
    if unverified:
        print("\nUnverified calendar rows (check MAS statements):")
        for e in unverified:
            print(f"  {e.date.date()} {e.action}: {e.note}")

    notable_gaps = gap_ratio[gap_ratio > 0.01]
    if not notable_gaps.empty:
        print("\nData gaps forward-filled before overlap trimming:")
        for ccy, ratio in notable_gaps.items():
            print(f"  {ccy:>4s}: {ratio:.1%}")


PALETTE = {
    "teal": "#15c8b5", "orange": "#ff9f43", "red": "#f06565", "green": "#62d394",
    "gold": "#f4c95d", "white": "#e7edf5", "muted": "#9aa6b6", "panel": "#171c24",
    "grid": "#2b3340", "bg": "#0f1318",
}
SERIES_COLORS = ["#15c8b5", "#f4c95d", "#ff9f43", "#8fa8ff", "#f06565", "#62d394",
                 "#c58cf0", "#9aa6b6"]


def style_ax(ax, title):
    p = PALETTE
    ax.set_facecolor(p["panel"])
    ax.set_title(title, color=p["white"], fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=p["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#323b49")
    ax.grid(True, color=p["grid"], linewidth=0.55, alpha=0.7)


def band_changes_in(index):
    return [e for e in MPS_EVENTS
            if e.action != "hold" and index.index[0] <= e.date <= index.index[-1]]


def mark_policy_events(ax, index, label=True):
    events = band_changes_in(index)
    for i, ev in enumerate(events):
        color = PALETTE["gold"] if ev.action == "recentre" else PALETTE["orange"]
        ax.axvline(ev.date, color=color, lw=1.0, alpha=0.7, ls="-" if ev.verified else ":",
                   label="MAS band change (dotted = unverified)" if label and i == 0 else None)


def plot_dashboard(data, index, returns, component_returns, center, weights, assumed, band):
    p = PALETTE
    fixed_upper = center * (1 + FIXED_BAND)
    fixed_lower = center * (1 - FIXED_BAND)
    rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100

    current = latest_weights(weights)
    if isinstance(weights, pd.DataFrame):
        contribution_63d = (component_returns * weights.reindex(component_returns.index)).rolling(63).sum() * 100
    else:
        contribution_63d = component_returns.mul(weights, axis=1).rolling(63).sum() * 100
    usd_per_sgd = data["USD"].reindex(index.index)
    usd_per_sgd_norm = usd_per_sgd / usd_per_sgd.iloc[0] * 100
    deviation = band["deviation"]
    latest_contrib = contribution_63d.iloc[-1].sort_values()
    latest_gap = band["latest_deviation_pct"]
    latest_vol = rolling_vol.iloc[-1]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(p["bg"])
    grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.35, 1.0, 1.0], hspace=0.46,
                             wspace=0.22, left=0.055, right=0.965, top=0.79, bottom=0.075)
    ax_neer = fig.add_subplot(grid[0, :])
    ax_weights = fig.add_subplot(grid[1, 0])
    ax_contrib = fig.add_subplot(grid[1, 1])
    ax_dev = fig.add_subplot(grid[2, 0])
    ax_compare = fig.add_subplot(grid[2, 1])

    def add_kpi(x, label, value, subtext, accent):
        fig.text(x, 0.925, label.upper(), color=p["muted"], fontsize=7.5, fontweight="bold")
        fig.text(x, 0.898, value, color=p["white"], fontsize=15, fontweight="bold")
        fig.text(x, 0.875, subtext, color=accent, fontsize=8, fontweight="bold")

    fig.text(0.055, 0.975, "S$NEER Shadow Proxy", color=p["white"], fontsize=18, fontweight="bold")
    fig.text(0.055, 0.948,
             f"Basket weights inferred from FX data ({WEIGHT_MODE}); band follows the MAS policy-statement calendar. "
             "MAS basket, centre, slope and width are undisclosed.",
             color=p["muted"], fontsize=9)
    add_kpi(0.055, "Latest date", index.index[-1].strftime("%d %b %Y"), "Yahoo FX closes", p["teal"])
    add_kpi(0.22, "Proxy index", f"{index.iloc[-1]:.2f}", f"{index.index[0]:%b %Y} = 100", p["teal"])
    in_band = abs(latest_gap) <= FIXED_BAND * 100
    add_kpi(0.385, "Vs centre", f"{latest_gap:+.2f}%",
            ("upper half" if latest_gap >= 0 else "lower half") if in_band else "outside est. band",
            p["green"] if in_band else p["red"])
    add_kpi(0.55, "Days outside band", f"{band['share_outside_band']:.0%}", f"of sample, +/-{FIXED_BAND:.0%}", p["gold"])
    add_kpi(0.715, "21d vol", f"{latest_vol:.2f}%", "annualised", p["gold"])

    style_ax(ax_neer, "Proxy Path vs Piecewise Policy Band (re-centres and slope changes from MAS statements)")
    ax_neer.fill_between(index.index, fixed_lower, fixed_upper, color="#dbe4ee", alpha=0.06,
                         label=f"Estimated band (+/-{FIXED_BAND:.0%})")
    ax_neer.plot(index, color=p["teal"], lw=2.1, label="Shadow S$NEER proxy")
    ax_neer.plot(center, color=p["white"], lw=1.25, ls=":", alpha=0.8, label="Estimated centre")
    ax_neer.plot(fixed_upper, color=p["red"], lw=0.85, ls="--", alpha=0.55)
    ax_neer.plot(fixed_lower, color=p["green"], lw=0.85, ls="--", alpha=0.55)
    mark_policy_events(ax_neer, index)
    ax_neer.scatter(index.index[-1], index.iloc[-1], s=34, color=p["teal"], edgecolor=p["white"],
                    linewidth=0.8, zorder=5)
    ax_neer.annotate(f"{index.iloc[-1]:.2f}", xy=(index.index[-1], index.iloc[-1]), xytext=(-48, 14),
                     textcoords="offset points", color=p["white"], fontsize=8.5, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=p["teal"], lw=0.8))
    ax_neer.set_ylabel(f"Index, {index.index[0]:%b %Y} = 100", color=p["muted"], fontsize=9)
    ax_neer.legend(loc="upper left", fontsize=7.5, framealpha=0.18, ncol=5)

    style_ax(ax_weights, "Basket Weights: Inferred (latest) vs Hand-Set")
    order = current.sort_values().index
    y = np.arange(len(order))
    ax_weights.barh(y + 0.2, current[order].values, height=0.4, color=p["teal"], alpha=0.9, label="Inferred, used now")
    ax_weights.barh(y - 0.2, assumed[order].values, height=0.4, color=p["muted"], alpha=0.55, label="Hand-set")
    for yy, value in zip(y, current[order].values):
        ax_weights.text(value + 0.002, yy + 0.2, f"{value:.1%}", color=p["muted"], va="center", fontsize=7.5)
    ax_weights.set_yticks(y, order)
    ax_weights.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_weights.set_xlim(0, max(current.max(), assumed.max()) * 1.25)
    ax_weights.legend(loc="lower right", fontsize=7.5, framealpha=0.18)

    style_ax(ax_contrib, "Latest 63-Day Weighted Contributions")
    contrib_colors = [p["green"] if v >= 0 else p["red"] for v in latest_contrib.values]
    ax_contrib.barh(latest_contrib.index, latest_contrib.values, color=contrib_colors, alpha=0.88)
    ax_contrib.axvline(0, color="#7a8594", lw=0.9)
    for yy, value in enumerate(latest_contrib.values):
        ha, offset = ("left", 0.015) if value >= 0 else ("right", -0.015)
        ax_contrib.text(value + offset, yy, f"{value:+.2f}%", color=p["muted"], va="center", ha=ha, fontsize=8)
    contrib_limit = max(abs(latest_contrib.min()), abs(latest_contrib.max()), 0.05) * 1.35
    ax_contrib.set_xlim(-contrib_limit, contrib_limit)
    ax_contrib.set_xlabel("Contribution to 63-day proxy return", color=p["muted"], fontsize=8)

    style_ax(ax_dev, "Deviation from Estimated Centre and Volatility")
    ax_dev.axhspan(-FIXED_BAND * 100, FIXED_BAND * 100, color="#dbe4ee", alpha=0.05)
    ax_dev.axhline(0, color="#7a8594", lw=0.8)
    ax_dev.plot(deviation, color=p["gold"], lw=1.3, label="Deviation from centre, %")
    ax_dev.plot(rolling_vol, color=p["teal"], lw=1.0, alpha=0.75, label="21d annualised vol, %")
    mark_policy_events(ax_dev, index, label=False)
    ax_dev.set_ylabel("%", color=p["muted"], fontsize=9)
    ax_dev.legend(loc="upper left", fontsize=7.5, framealpha=0.18)

    style_ax(ax_compare, "USD per SGD vs Trade-Weighted SGD Proxy")
    ax_compare.plot(usd_per_sgd_norm, color=p["gold"], lw=1.35, label="USD per SGD, normalized")
    ax_compare.plot(index, color=p["teal"], lw=1.8, label="Shadow S$NEER proxy")
    ax_compare.fill_between(index.index, usd_per_sgd_norm, index, color=p["white"], alpha=0.035)
    ax_compare.set_ylabel(f"Index, {index.index[0]:%b %Y} = 100", color=p["muted"], fontsize=9)
    ax_compare.legend(loc="upper left", fontsize=7.5, framealpha=0.18)

    fig.text(0.055, 0.03,
             "Method: chain-linked geometric index of SGD crosses; weights from a Frankel-Wei regression (CHF numeraire, weekly returns). "
             "Band centre: re-centred at prevailing level on MAS re-centring dates, slopes are analyst-style estimates.",
             color=p["muted"], fontsize=8)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved -> {OUTPUT_FILE}")
    return fig


def plot_inference(inference, regimes):
    p = PALETTE
    roll = inference["rolling_weights"]
    top = roll.mean().sort_values(ascending=False).index[:7]
    stacked = roll[top].copy()
    stacked["Other"] = roll.drop(columns=top).sum(axis=1)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(p["bg"])
    grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1.0], hspace=0.4, wspace=0.2,
                             left=0.055, right=0.965, top=0.86, bottom=0.08)
    ax_roll = fig.add_subplot(grid[0, :])
    ax_coef = fig.add_subplot(grid[1, 0])
    ax_slope = fig.add_subplot(grid[1, 1])

    fig.text(0.055, 0.95, "Reverse-Engineering the S$NEER Basket", color=p["white"], fontsize=18, fontweight="bold")
    fig.text(0.055, 0.918,
             f"Frankel-Wei regression: weekly dln(CHF per SGD) on dln(CHF per currency). "
             f"Full-sample R2 = {inference['full_r2']:.3f}, implied slope = {inference['full_implied_slope_pct']:.2f}% p.a.",
             color=p["muted"], fontsize=9.5)

    style_ax(ax_roll, f"Rolling {inference['window_weeks']}-Week Implied Weights (non-negative, sum to 100%)")
    ax_roll.stackplot(stacked.index, stacked.T.values, labels=stacked.columns,
                      colors=SERIES_COLORS[: len(stacked.columns)], alpha=0.85)
    ax_roll.set_ylim(0, 1)
    ax_roll.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_roll.legend(loc="upper left", fontsize=8, framealpha=0.25, ncol=8)
    ax_roll.set_xlabel("Window end date", color=p["muted"], fontsize=8)

    style_ax(ax_coef, "Full-Sample OLS Weights with 95% Interval")
    ols = inference["ols"].drop(index="const").sort_values("coef")
    yy = np.arange(len(ols))
    ax_coef.barh(yy, ols["coef"], xerr=1.96 * ols["se"], color=p["teal"], alpha=0.85,
                 error_kw=dict(ecolor=p["white"], lw=0.8, capsize=2))
    ax_coef.set_yticks(yy, ols.index)
    ax_coef.axvline(0, color="#7a8594", lw=0.9)
    ax_coef.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    style_ax(ax_slope, "Band Slope: Calendar Assumption vs Data-Implied (% p.a.)")
    drift = inference["rolling_implied_slope_pct"]
    ax_slope.plot(drift, color=p["gold"], lw=1.4, label=f"Regression drift, rolling {inference['window_weeks']}w")
    sched = slope_schedule(pd.date_range(drift.index[0], drift.index[-1], freq="D"), MPS_EVENTS, INITIAL_SLOPE)
    ax_slope.step(sched.index, sched.values, where="post", color=p["white"], lw=1.2, ls="--",
                  label="Calendar slope assumption")
    for _, row in regimes.dropna().iterrows():
        start = pd.Timestamp(row["start"])
        if start >= drift.index[0]:
            ax_slope.scatter(start, row["implied_slope_pct"], color=p["teal"], s=22, zorder=5)
    ax_slope.scatter([], [], color=p["teal"], s=22, label="Index trend within regime")
    ax_slope.axhline(0, color="#7a8594", lw=0.8)
    ax_slope.legend(loc="upper left", fontsize=7.5, framealpha=0.18)

    fig.text(0.055, 0.025,
             "Rolling weights are backward-looking (window ends at the date shown). Drift = regression intercept, annualised; "
             "it mixes the policy slope with movement inside the band, so read it as a cross-check, not the slope itself.",
             color=p["muted"], fontsize=8)
    plt.savefig(INFERENCE_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved -> {INFERENCE_FILE}")
    return fig


def select_weights(mode, inference, data, assumed):
    if mode == "assumed":
        return assumed
    if mode == "inferred_full":
        return inference["full_weights"][data.columns]
    if mode == "inferred_rolling":
        return daily_weight_path(inference["rolling_weights"], data.index, data.columns)
    raise ValueError(f"Unknown WEIGHT_MODE: {mode}")


def main(show=True):
    assumed = normalize_weights(PROXY_BASKET)
    data, gap_ratio = download_sgd_crosses(PROXY_BASKET)
    print("Running Frankel-Wei basket inference...")
    inference = infer_basket(start=INFERENCE_START, currencies=list(data.columns),
                             window=ROLLING_WINDOW_WEEKS)
    weights = select_weights(WEIGHT_MODE, inference, data, assumed)

    index, returns, component_returns = build_shadow_neer(data, weights)
    center, _ = build_piecewise_center(index, MPS_EVENTS, INITIAL_SLOPE)
    regimes = implied_regime_slopes(index, MPS_EVENTS, INITIAL_SLOPE)
    band = band_diagnostics(index, center, FIXED_BAND)

    print(f"Data loaded: {len(data)} observations, {len(data.columns)} SGD crosses")
    print(f"Date range: {data.index[0].date()} -> {data.index[-1].date()}")

    latest_snapshot(index, returns, center, weights, assumed, inference, regimes, band, data, gap_ratio)
    plot_dashboard(data, index, returns, component_returns, center, weights, assumed, band)
    plot_inference(inference, regimes)
    if show:
        plt.show()


if __name__ == "__main__":
    main()
