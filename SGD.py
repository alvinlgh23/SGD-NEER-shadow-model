import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA INGESTION — Extended BIS-Aligned Basket (10 pairs)
# ============================================================
# MAS's actual basket is undisclosed, but BIS data points to these
# top trading partners for Singapore. SGD is the NUMERAIRE,
# so we express all rates as SGD per foreign unit (inverted from X quotes).
pairs = {
    "SGDUSD=X": "USD",
    "SGDCNY=X": "CNY",
    "SGDMYR=X": "MYR",
    "SGDEUR=X": "EUR",
    "SGDJPY=X": "JPY",
    "SGDKRW=X": "KRW",
    "SGDGBP=X": "GBP",
    "SGDAUD=X": "AUD",
    "SGDTHB=X": "THB",
    "SGDINR=X": "INR",
}

print("Downloading FX data...")
raw = yf.download(list(pairs.keys()), start="2022-01-01", progress=False)['Close']
raw.columns = [pairs[col] for col in raw.columns]

# Forward-fill weekends/holidays, then drop any remaining NaNs
data = raw.ffill().dropna()

# Log returns: ln(P_t / P_{t-1})
# Under the NEER framework, a RISE in SGD/FCU means SGD WEAKENED against that currency.
# We invert so that a positive log-return = SGD appreciating vs that currency.
log_ret = -np.log(data / data.shift(1)).dropna()  # Negated: positive = SGD strength

print(f"Data loaded: {len(data)} trading days, {len(data.columns)} currency pairs")
print(f"Date range: {data.index[0].date()} → {data.index[-1].date()}")

# ============================================================
# 2. REVERSE-ENGINEERING ENGINE — Rolling Constrained OLS
# ============================================================
# Window = 252 trading days (1 year)
# Constraint: weights ≥ 0, sum to 1 (no short-selling, no leverage)
# Method: SLSQP (Sequential Least Squares Programming)

window = 252

base_currency = 'USD'
cols = log_ret.drop(columns=[base_currency]).columns.tolist()   # 9 currencies (excl. USD)
x_full = log_ret[cols].values
y_full = log_ret[base_currency].values   # y = SGD/USD; x = other 9 currencies
dates = log_ret.index[window:]

n_currencies = len(cols)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0.0, 1.0)] * n_currencies
w0 = np.array([1.0 / n_currencies] * n_currencies)

print(f"\nRunning rolling constrained OLS over {len(dates)} windows...")
rolling_weights = []

for i in range(len(dates)):
    x_win = x_full[i: i + window]
    y_win = y_full[i: i + window]
    res = minimize(
        lambda w: np.sum((y_win - x_win @ w) ** 2),
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )
    rolling_weights.append(res.x)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(dates)} windows done...")

weights_df = pd.DataFrame(rolling_weights, index=dates, columns=cols)
print("Rolling optimization complete.")

# ============================================================
# 3. SHADOW NEER INDEX CONSTRUCTION
# ============================================================
# Reconstruct the NEER as a weighted sum of log returns, then exponentiate
dynamic_ret = np.array([
    x_full[window + i] @ rolling_weights[i] for i in range(len(dates))
])
neer_series = pd.Series(
    np.exp(np.cumsum(dynamic_ret)) * 100,
    index=dates,
    name="Shadow_NEER"
)

# ============================================================
# 4. ROLLING VOLATILITY (Phase 3 — Missing from original)
# ============================================================
# 21-day rolling annualised vol of the NEER log-returns
neer_log_ret = np.log(neer_series / neer_series.shift(1)).dropna()
rolling_vol = neer_log_ret.rolling(21).std() * np.sqrt(252) * 100  # annualised, in %

# Dynamic band width: ±(1.5 × 30-day rolling vol), floored at ±0.5%
band_half_width = (neer_log_ret.rolling(30).std() * np.sqrt(30) * 1.5 * 100).clip(lower=0.5)

# ============================================================
# 5. MAS POLICY BAND TRACKER
# ============================================================
policy_change = pd.Timestamp('2026-04-14')

# MAS policy: the band's CENTRE appreciates at a controlled slope.
# Before Apr 14 2026: ~1.5% p.a. slope (pre-tightening regime)
# After Apr 14 2026: ~2.0% p.a. slope (tightened to combat imported inflation)
# We optimise the starting anchor so the centre best fits the market NEER.

def build_policy_center(anchor_val):
    anchor = float(np.atleast_1d(anchor_val)[0])
    centers = np.empty(len(dates))
    centers[0] = anchor
    for k in range(1, len(dates)):
        slope = 0.020 / 252 if dates[k] >= policy_change else 0.015 / 252
        centers[k] = centers[k - 1] * (1 + slope)
    return centers

opt_res = minimize(
    lambda a: np.sum((neer_series.values - build_policy_center(a)) ** 2),
    x0=[neer_series.iloc[0]],
    method='Nelder-Mead'
)
smooth_center = build_policy_center(opt_res.x)

# MAS uses a fixed ±2% statutory band; we also show a vol-dynamic band
fixed_band = 0.02
final_df = pd.DataFrame({
    'NEER': neer_series,
    'Center': smooth_center,
    'Upper_Fixed': smooth_center * (1 + fixed_band),
    'Lower_Fixed': smooth_center * (1 - fixed_band),
}, index=dates)

# Add dynamic (vol-adjusted) bands where vol data is available
final_df['Vol_BandWidth'] = band_half_width.reindex(dates)
final_df['Upper_Dynamic'] = final_df['Center'] * (1 + final_df['Vol_BandWidth'] / 100)
final_df['Lower_Dynamic'] = final_df['Center'] * (1 - final_df['Vol_BandWidth'] / 100)

r2 = r2_score(final_df['NEER'], final_df['Center'])

# ============================================================
# 6. KRW/SGD CORRELATION SPOTLIGHT (Korea-Singapore Trade Link)
# SGD/USD actual price series (reindexed to NEER dates for comparison)
sgdusd = data['USD'].reindex(dates)
# Normalise to same base as NEER (value at first date = 100)
sgdusd_norm = sgdusd / sgdusd.iloc[0] * 100

# Latest snapshot
latest_weights = weights_df.iloc[-1].sort_values(ascending=False)
print(f"\n{'='*55}")
print("SHADOW BASKET — LATEST WEIGHTS")
print('='*55)
for ccy, wt in latest_weights.items():
    bar = '█' * int(wt * 40)
    print(f"  {ccy:>4s}  {wt:6.2%}  {bar}")

print(f"\nOptimised Anchor:  {opt_res.x[0]:.4f}")
print(f"R² (Center vs Market NEER):  {r2:.4f}")
print(f"Current Rolling Vol (21d ann.):  {rolling_vol.iloc[-1]:.2f}%")
print(f"Latest SGD/USD: {sgdusd.iloc[-1]:.4f}")

# ============================================================
# 7. PROFESSIONAL 4-PANEL VISUALIZATION
# ============================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d0d1a')

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

ax_neer   = fig.add_subplot(gs[0, :])
ax_weight = fig.add_subplot(gs[1, :])
ax_vol    = fig.add_subplot(gs[2, 0])

gs_corr = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[2, 1],
    height_ratios=[0.68, 0.32], hspace=0.08
)
ax_corr = fig.add_subplot(gs_corr[0])
ax_dev  = fig.add_subplot(gs_corr[1], sharex=ax_corr)

TEAL   = '#00d4b4'
ORANGE = '#ff7043'
RED    = '#ef5350'
GREEN  = '#66bb6a'
GOLD   = '#ffd54f'
WHITE  = '#e8eaf6'
BG     = '#0d0d1a'
PANEL  = '#141428'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors='#8888aa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a2a4a')
    ax.grid(True, color='#1e1e3a', linewidth=0.5, alpha=0.8)

# --- Panel 1: Shadow NEER + Policy Band ---
style_ax(ax_neer, f"Shadow S$NEER Policy Tracker  |  R² = {r2:.4f}  |  Basket: {len(cols)+1} Currencies")

ax_neer.fill_between(dates, final_df['Lower_Fixed'], final_df['Upper_Fixed'],
                     color='#ffffff', alpha=0.03, label='Fixed ±2% MAS Band')
ax_neer.fill_between(dates, final_df['Lower_Dynamic'], final_df['Upper_Dynamic'],
                     color=TEAL, alpha=0.08, label='Vol-Dynamic Band (±1.5σ)')
ax_neer.plot(final_df['NEER'],    color=TEAL,   lw=1.8, label='Shadow S$NEER')
ax_neer.plot(final_df['Center'],  color=WHITE,  lw=1.2, ls=':', alpha=0.6, label='Policy Centre')
ax_neer.plot(final_df['Upper_Fixed'], color=RED,   lw=0.8, ls='--', alpha=0.5, label='Upper +2%')
ax_neer.plot(final_df['Lower_Fixed'], color=GREEN, lw=0.8, ls='--', alpha=0.5, label='Lower -2%')
ax_neer.axvline(policy_change, color=ORANGE, lw=1.8, ls='-',
                label='MAS Tightening (Apr 14 2026)', zorder=5)
ax_neer.annotate('MAS Slope\n+0.5pp', xy=(policy_change, ax_neer.get_ylim()[1] if ax_neer.get_ylim()[1] != 0 else 105),
                 xytext=(15, -30), textcoords='offset points',
                 color=ORANGE, fontsize=8, arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))
ax_neer.legend(loc='upper left', fontsize=7.5, framealpha=0.2, ncol=3)
ax_neer.set_ylabel('Index (Base = 100)', color='#8888aa', fontsize=9)

# --- Panel 2: Dynamic Basket Weights ---
style_ax(ax_weight, "Reverse-Engineered Shadow Basket Weights (Rolling 252-Day Constrained OLS)")

cmap = plt.cm.get_cmap('tab10', len(cols))
colors = [cmap(i) for i in range(len(cols))]
ax_weight.stackplot(dates, weights_df.T.values, labels=cols,
                    colors=colors, alpha=0.9)
ax_weight.axvline(policy_change, color=ORANGE, lw=1.5, ls='-')
ax_weight.set_ylabel('Weight', color='#8888aa', fontsize=9)
ax_weight.legend(loc='upper left', fontsize=7.5, framealpha=0.2, ncol=5)
ax_weight.set_ylim(0, 1)
ax_weight.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

# --- Panel 3: Rolling NEER Volatility ---
style_ax(ax_vol, "Rolling NEER Volatility (21-Day, Annualised)")
ax_vol.fill_between(rolling_vol.index, rolling_vol, color=TEAL, alpha=0.3)
ax_vol.plot(rolling_vol, color=TEAL, lw=1.2)
ax_vol.axvline(policy_change, color=ORANGE, lw=1.5, ls='-')
ax_vol.set_ylabel('Vol %', color='#8888aa', fontsize=9)
ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))

# --- Panel 4: SGD/USD vs Shadow NEER + Deviation ---
deviation = sgdusd_norm - final_df['NEER']   # +ve = market above policy implied (SGD overvalued)

style_ax(ax_corr, "SGD/USD vs Shadow NEER  |  Market vs Policy-Implied Level")
ax_corr.plot(sgdusd_norm,      color=GOLD, lw=1.4, label='SGD/USD (Normalised)')
ax_corr.plot(final_df['NEER'], color=TEAL, lw=1.4, alpha=0.85, label='Shadow NEER (Policy-Implied)')
# Shading: above = SGD stronger than policy implies (green), below = weaker (red)
ax_corr.fill_between(dates, sgdusd_norm, final_df['NEER'],
                     where=sgdusd_norm >= final_df['NEER'],
                     color=GREEN, alpha=0.15, label='Above implied (SGD strong)')
ax_corr.fill_between(dates, sgdusd_norm, final_df['NEER'],
                     where=sgdusd_norm < final_df['NEER'],
                     color=RED, alpha=0.15, label='Below implied (SGD weak)')
ax_corr.axvline(policy_change, color=ORANGE, lw=1.5, ls='-')
ax_corr.set_ylabel('Index (Base=100)', color='#8888aa', fontsize=8)
ax_corr.legend(loc='upper right', fontsize=6.5, framealpha=0.2, ncol=2)
ax_corr.tick_params(labelbottom=False)   # hide x-ticks, shared with ax_dev

# Deviation subplot
style_ax(ax_dev, "")
ax_dev.set_facecolor('#141428')
ax_dev.axhline(0, color='#555577', lw=0.8)
ax_dev.fill_between(dates, deviation, 0,
                    where=deviation >= 0, color=GREEN, alpha=0.5)
ax_dev.fill_between(dates, deviation, 0,
                    where=deviation < 0,  color=RED,   alpha=0.5)
ax_dev.plot(deviation, color=WHITE, lw=0.7, alpha=0.6)
ax_dev.axvline(policy_change, color=ORANGE, lw=1.5, ls='-')
ax_dev.set_ylabel('Deviation', color='#8888aa', fontsize=7)
ax_dev.tick_params(colors='#8888aa', labelsize=7)
for spine in ax_dev.spines.values():
    spine.set_color('#2a2a4a')
ax_dev.grid(True, color='#1e1e3a', linewidth=0.5, alpha=0.8)
# Annotate current deviation
cur_dev = deviation.iloc[-1]
ax_dev.annotate(f'{cur_dev:+.2f}', xy=(deviation.index[-1], cur_dev),
                xytext=(-35, 8 if cur_dev < 0 else -14), textcoords='offset points',
                color=GREEN if cur_dev >= 0 else RED, fontsize=7, fontweight='bold')

# --- Master title ---
fig.suptitle(
    "S$NEER Shadow Model  ·  Open-Source Replication of MAS Monetary Policy",
    color=WHITE, fontsize=15, fontweight='bold', y=0.97
)

plt.savefig('sgd_neer_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\nChart saved → sgd_neer_dashboard.png")
plt.show()