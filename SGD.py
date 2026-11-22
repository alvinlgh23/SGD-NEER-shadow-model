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
from mps_calendar import INITIAL_SLOPE, MPS_EVENTS, NEXT_REVIEW_LABEL
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
    "grid": "#2b3340", "bg": "#0f1318", "edge": "#b8c4d4", "band": "#5b7fa6",
}
LINE_COLORS = ["#15c8b5", "#f4c95d", "#8fa8ff", "#ff9f43", "#c58cf0"]
EVENT_TAGS = {"slope_up": "↑ slope", "slope_down": "↓ slope", "recentre": "re-centre"}


def style_ax(ax, title, subtitle=None):
    p = PALETTE
    ax.set_facecolor(p["panel"])
    ax.set_title(title, color=p["white"], fontsize=12, fontweight="bold", pad=22 if subtitle else 10, loc="left")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=p["muted"], fontsize=9, va="bottom")
    ax.tick_params(colors=p["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#323b49")
    ax.grid(True, color=p["grid"], linewidth=0.55, alpha=0.7)


def band_changes_in(index):
    return [e for e in MPS_EVENTS
            if e.action != "hold" and index.index[0] <= e.date <= index.index[-1]]


def mark_policy_events(ax, index, tags=False):
    """Vertical lines at band changes; optional short labels along the top edge."""
    p = PALETTE
    for i, ev in enumerate(band_changes_in(index)):
        color = p["gold"] if ev.action == "recentre" else p["orange"]
        ax.axvline(ev.date, color=color, lw=1.0, alpha=0.55, ls="-" if ev.verified else ":", zorder=1)
        if tags:
            text = f"{ev.date:%b %y}\n{EVENT_TAGS[ev.action]}" + ("" if ev.verified else "\n(unverified)")
            ax.annotate(text, xy=(ev.date, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -6 - 26 * (i % 2)), textcoords="offset points",
                        color=color, fontsize=8, va="top", ha="left", linespacing=1.05)


def band_zone(gap_pct):
    half = FIXED_BAND * 100
    if gap_pct > half:
        return "above the estimated band"
    if gap_pct >= 0:
        return "in the upper half of the band"
    if gap_pct >= -half:
        return "in the lower half of the band"
    return "below the estimated band"


def current_slope(index):
    return slope_schedule(index.index[-1:], MPS_EVENTS, INITIAL_SLOPE).iloc[0]


def plot_dashboard(data, index, returns, component_returns, center, weights, assumed, band):
    p = PALETTE
    half = FIXED_BAND * 100
    upper = center * (1 + FIXED_BAND)
    lower = center * (1 - FIXED_BAND)
    deviation = band["deviation"]
    gap = band["latest_deviation_pct"]
    slope_now = current_slope(index)

    current = latest_weights(weights)
    if isinstance(weights, pd.DataFrame):
        contribution_63d = (component_returns * weights.reindex(component_returns.index)).rolling(63).sum() * 100
    else:
        contribution_63d = component_returns.mul(weights, axis=1).rolling(63).sum() * 100
    latest_contrib = contribution_63d.iloc[-1].sort_values()
    usd_per_sgd = data["USD"].reindex(index.index)
    usd_per_sgd_norm = usd_per_sgd / usd_per_sgd.iloc[0] * 100

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12.5))
    fig.patch.set_facecolor(p["bg"])
    grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.45, 1.0, 1.0], hspace=0.55,
                             wspace=0.2, left=0.055, right=0.965, top=0.79, bottom=0.07)
    ax_neer = fig.add_subplot(grid[0, :])
    ax_pos = fig.add_subplot(grid[1, 0])
    ax_weights = fig.add_subplot(grid[1, 1])
    ax_contrib = fig.add_subplot(grid[2, 0])
    ax_compare = fig.add_subplot(grid[2, 1])

    # Headline: the finding first, method second.
    fig.text(0.055, 0.972, f"The Singapore dollar is {band_zone(gap)} ({gap:+.1f}% vs estimated centre)",
             color=p["white"], fontsize=20, fontweight="bold")
    fig.text(0.055, 0.945,
             f"Shadow S$NEER from FX data as of {index.index[-1]:%d %b %Y}. MAS keeps its basket, band and slope "
             "undisclosed; everything shown is an estimate.", color=p["muted"], fontsize=10)

    in_band = abs(gap) <= half
    kpis = [
        ("Position vs centre", f"{gap:+.2f}%", f"band estimated at +/-{half:.0f}%", p["green"] if in_band else p["red"]),
        ("Room to upper edge", f"{half - gap:.2f}%", "before est. ceiling" if gap <= half else "already above",
         p["teal"] if gap <= half else p["red"]),
        ("Estimated slope", f"{slope_now:.2f}% p.a.", "pace of allowed appreciation", p["gold"]),
        ("Days outside band", f"{band['share_outside_band']:.0%}", f"since {index.index[0]:%b %Y}, model check", p["muted"]),
        ("Next MAS review", NEXT_REVIEW_LABEL, "date to confirm on mas.gov.sg", p["muted"]),
    ]
    for i, (label, value, sub, accent) in enumerate(kpis):
        x = 0.055 + i * 0.18
        fig.text(x, 0.905, label.upper(), color=p["muted"], fontsize=8.5, fontweight="bold")
        fig.text(x, 0.876, value, color=p["white"], fontsize=17, fontweight="bold")
        fig.text(x, 0.853, sub, color=accent, fontsize=8.5)

    # Main chart: corridor, centre, index, outside-band highlight, labelled events.
    style_ax(ax_neer, "Shadow S$NEER inside the estimated MAS policy band",
             "Band re-centred and re-sloped on the dates MAS announced; dotted line = unverified event")
    ax_neer.fill_between(index.index, lower, upper, color=p["band"], alpha=0.22, lw=0, label=f"Estimated band (+/-{half:.0f}%)")
    ax_neer.plot(upper, color=p["edge"], lw=0.8, alpha=0.6)
    ax_neer.plot(lower, color=p["edge"], lw=0.8, alpha=0.6)
    ax_neer.plot(center, color=p["edge"], lw=1.0, ls=":", alpha=0.8, label="Estimated centre")
    ax_neer.plot(index, color=p["teal"], lw=2.0, label="Shadow S$NEER")
    outside = deviation.abs() > half
    ax_neer.plot(index.where(outside), color=p["red"], lw=2.0, label="Outside estimated band")
    mark_policy_events(ax_neer, index, tags=True)
    ax_neer.scatter(index.index[-1], index.iloc[-1], s=40, color=p["teal"], edgecolor=p["white"], linewidth=0.9, zorder=5)
    lo, hi = min(lower.min(), index.min()), max(upper.max(), index.max())
    ax_neer.set_ylim(lo - 0.5, hi + (hi - lo) * 0.36)  # headroom for event tags
    ax_neer.set_ylabel(f"Index, {index.index[0]:%b %Y} = 100", color=p["muted"], fontsize=9)
    ax_neer.legend(loc="lower right", fontsize=8.5, framealpha=0.25, ncol=4)

    # Position within band: one unit, three zones.
    style_ax(ax_pos, "Where SGD sits inside the band", "Deviation from estimated centre, %")
    lim = max(3.0, np.ceil(deviation.abs().max()) + 0.5)
    ax_pos.axhspan(half, lim, color=p["red"], alpha=0.08)
    ax_pos.axhspan(-lim, -half, color=p["red"], alpha=0.08)
    ax_pos.axhspan(0, half, color=p["band"], alpha=0.18)
    ax_pos.axhspan(-half, 0, color=p["band"], alpha=0.08)
    for y, text in [(half + (lim - half) / 2, "above band"), (half / 2, "upper half"),
                    (-half / 2, "lower half"), (-half - (lim - half) / 2, "below band")]:
        ax_pos.text(1.005, y, text, transform=ax_pos.get_yaxis_transform(), color=p["muted"], fontsize=8, va="center")
    ax_pos.axhline(0, color=p["edge"], lw=0.8, ls=":")
    ax_pos.plot(deviation, color=p["gold"], lw=1.4)
    mark_policy_events(ax_pos, index)
    ax_pos.set_ylim(-lim, lim)
    ax_pos.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

    # Basket weights: dumbbell, hand-set vs inferred.
    style_ax(ax_weights, "What the basket looks like", "Weights inferred from FX data vs a typical hand-set guess")
    order = current.sort_values().index
    yy = np.arange(len(order))
    ax_weights.hlines(yy, assumed[order], current[order], color=p["muted"], lw=1.2, alpha=0.6)
    ax_weights.scatter(assumed[order], yy, color=p["muted"], s=28, label="Hand-set guess", zorder=3)
    ax_weights.scatter(current[order], yy, color=p["teal"], s=42, label="Inferred (latest window)", zorder=4)
    ax_weights.set_yticks(yy, order)
    ax_weights.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_weights.set_xlim(0, max(current.max(), assumed.max()) * 1.12)
    ax_weights.legend(loc="lower right", fontsize=8.5, framealpha=0.25)

    style_ax(ax_contrib, "What moved the index in the last 3 months", "Weighted contribution of each currency, %")
    colors = [p["green"] if v >= 0 else p["red"] for v in latest_contrib.values]
    ax_contrib.barh(latest_contrib.index, latest_contrib.values, color=colors, alpha=0.88)
    ax_contrib.axvline(0, color="#7a8594", lw=0.9)
    for y, v in enumerate(latest_contrib.values):
        ha, off = ("left", 0.015) if v >= 0 else ("right", -0.015)
        ax_contrib.text(v + off, y, f"{v:+.2f}%", color=p["muted"], va="center", ha=ha, fontsize=8.5)
    limit = max(abs(latest_contrib.min()), abs(latest_contrib.max()), 0.05) * 1.35
    ax_contrib.set_xlim(-limit, limit)
    ax_contrib.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}%"))
    ax_contrib.text(0.99, 0.02, "SGD stronger vs this currency → positive", transform=ax_contrib.transAxes,
                    color=p["muted"], fontsize=8, ha="right")

    style_ax(ax_compare, "Broad SGD strength vs USD/SGD alone",
             f"Both rebased to 100 in {index.index[0]:%b %Y}; the gap is strength against non-USD currencies")
    ax_compare.plot(usd_per_sgd_norm, color=p["gold"], lw=1.4, label="USD per SGD")
    ax_compare.plot(index, color=p["teal"], lw=1.9, label="Shadow S$NEER")
    ax_compare.legend(loc="upper left", fontsize=8.5, framealpha=0.25)

    fig.text(0.055, 0.025,
             "Method: chain-linked geometric index of 12 SGD crosses; weights from a rolling Frankel-Wei regression "
             "(CHF numeraire, weekly returns, lagged). Slopes are analyst-style estimates; event directions follow MAS statements.",
             color=p["muted"], fontsize=8.5)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved -> {OUTPUT_FILE}")
    return fig


def spread_labels(values, min_gap):
    """Nudge end-of-line label positions apart so none sit closer than min_gap."""
    ordered = values.sort_values()
    positions = ordered.to_numpy(dtype=float).copy()
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + min_gap)
    shift = (positions - ordered.to_numpy()).mean()  # re-centre the group on the data
    return pd.Series(positions - shift, index=ordered.index)


MIN_REGIME_OBS = 120  # shorter regimes give noisy trend estimates


def plot_inference(inference, regimes, assumed):
    p = PALETTE
    roll = inference["rolling_weights"]
    top = roll.iloc[-1].sort_values(ascending=False).index[:5]
    ols = inference["ols"].drop(index="const")
    full = inference["full_weights"]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10.5))
    fig.patch.set_facecolor(p["bg"])
    grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.5, wspace=0.2,
                             left=0.055, right=0.965, top=0.85, bottom=0.08)
    ax_roll = fig.add_subplot(grid[0, :])
    ax_coef = fig.add_subplot(grid[1, 0])
    ax_slope = fig.add_subplot(grid[1, 1])

    fig.text(0.055, 0.955,
             f"The S$NEER basket is led by USD and EUR. FX data explains {inference['full_r2']:.0%} of weekly SGD moves",
             color=p["white"], fontsize=20, fontweight="bold")
    fig.text(0.055, 0.925,
             "Frankel-Wei regression: weekly change in CHF per SGD on CHF per basket currency. "
             "The coefficients are the implicit basket weights MAS does not publish.",
             color=p["muted"], fontsize=10)

    style_ax(ax_roll, f"How the top-5 weights changed (rolling {inference['window_weeks']}-week window)",
             "Each point uses the two years of data ending on that date")
    label_y = spread_labels(roll[top].iloc[-1], min_gap=0.018)
    for color, ccy in zip(LINE_COLORS, top):
        series = roll[ccy]
        ax_roll.plot(series, color=color, lw=2.0)
        ax_roll.annotate(f"{ccy} {series.iloc[-1]:.0%}", xy=(series.index[-1], series.iloc[-1]),
                         xytext=(series.index[-1] + pd.Timedelta(days=12), label_y[ccy]),
                         textcoords="data", color=color, fontsize=9.5, fontweight="bold", va="center")
    ax_roll.set_ylim(0, max(0.35, roll[top].max().max() * 1.1))
    ax_roll.set_xlim(roll.index[0], roll.index[-1] + pd.Timedelta(days=110))
    ax_roll.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    style_ax(ax_coef, "Inferred weight vs hand-set guess (full sample)",
             "Teal = inferred with 95% interval, grey = hand-set guess")
    order = ols["coef"].sort_values().index
    yy = np.arange(len(order))
    ax_coef.errorbar(ols.loc[order, "coef"], yy, xerr=1.96 * ols.loc[order, "se"], fmt="none",
                     ecolor=p["teal"], elinewidth=1.4, alpha=0.6, capsize=0)
    ax_coef.scatter(full[order], yy, color=p["teal"], s=42, zorder=4, label="Inferred")
    ax_coef.scatter(assumed[order], yy, color=p["muted"], s=28, marker="D", zorder=3, label="Hand-set guess")
    ax_coef.set_yticks(yy, order)
    ax_coef.axvline(0, color="#7a8594", lw=0.9)
    ax_coef.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_coef.legend(loc="lower right", fontsize=8.5, framealpha=0.25)

    style_ax(ax_slope, "Band slope: assumption vs what the data implies",
             f"% p.a.; dots = index trend within regimes of {MIN_REGIME_OBS}+ trading days")
    drift = inference["rolling_implied_slope_pct"]
    sched = slope_schedule(pd.date_range(drift.index[0], drift.index[-1], freq="D"), MPS_EVENTS, INITIAL_SLOPE)
    ax_slope.step(sched.index, sched.values, where="post", color=p["white"], lw=1.4, ls="--",
                  label="Calendar slope assumption")
    ax_slope.plot(drift, color=p["gold"], lw=1.4, alpha=0.85, label="Regression drift (rolling)")
    long_regimes = regimes[regimes["obs"] >= MIN_REGIME_OBS].dropna()
    for _, row in long_regimes.iterrows():
        ax_slope.scatter(pd.Timestamp(row["start"]), row["implied_slope_pct"], color=p["teal"], s=40, zorder=5)
    ax_slope.scatter([], [], color=p["teal"], s=40, label="Index trend in regime")
    ax_slope.axhline(0, color="#7a8594", lw=0.8)
    ax_slope.set_ylim(min(-1.5, drift.min() - 0.3), max(5.0, drift.max() + 0.5))
    ax_slope.legend(loc="upper left", fontsize=8.5, framealpha=0.25)

    fig.text(0.055, 0.025,
             "Regression drift = annualised intercept; it mixes the policy slope with movement inside the band, "
             "so read it as a cross-check. Weights are non-negative and sum to 100% in the rolling estimate.",
             color=p["muted"], fontsize=8.5)
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
    plot_inference(inference, regimes, assumed)
    if show:
        plt.show()


if __name__ == "__main__":
    main()
