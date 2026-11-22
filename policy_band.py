"""Piecewise S$NEER policy-band centre driven by the MPS event calendar.

Rules:
- Between events the centre grows at the prevailing slope (log-linear).
- "slope_up" / "slope_down" change the slope from the event date onwards.
- "recentre" resets the centre to the prevailing index level (mean of the
  last few observations before the announcement), then applies the new slope.
- The only free parameter is the starting level before the first re-centre
  inside the sample; it is fitted by least squares on that first segment.
  After a re-centre the path is fully determined by the calendar, so any drift
  outside the band is a genuine test of the slope assumptions.
"""

import numpy as np
import pandas as pd

RECENTRE_LOOKBACK = 5  # observations averaged to define the "prevailing level"


def slope_schedule(dates, events, initial_slope=0.0):
    """Daily slope (% p.a.) in force on each date."""
    slopes = pd.Series(initial_slope, index=dates, dtype=float)
    for ev in sorted(events, key=lambda e: e.date):
        slopes[dates >= ev.date] = ev.slope_after
    return slopes


def _recentre_events(events, dates):
    return [e for e in sorted(events, key=lambda e: e.date)
            if e.action == "recentre" and dates[0] < e.date <= dates[-1]]


def build_piecewise_center(index, events, initial_slope=0.0, slope_overrides=None):
    """Return (centre series, info dict).

    ``slope_overrides`` optionally maps event dates to replacement slopes,
    used when comparing assumed vs data-implied slopes.
    """
    if slope_overrides:
        events = [
            type(e)(e.date, e.action, slope_overrides.get(e.date, e.slope_after),
                    e.note, e.verified, e.off_cycle)
            for e in events
        ]
    dates = index.index
    slopes = slope_schedule(dates, events, initial_slope).to_numpy()
    days = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    days = np.concatenate([[0.0], np.maximum(days, 0.0)])
    log_growth = np.cumsum(slopes / 100.0 * days / 365.25)

    log_index = np.log(index.to_numpy(dtype=float))
    log_center = np.empty_like(log_index)

    recentres = _recentre_events(events, dates)
    boundaries = [0] + [int(dates.searchsorted(e.date)) for e in recentres] + [len(dates)]

    # First segment: fit the unknown starting level (closed-form LS in logs).
    s0, s1 = boundaries[0], boundaries[1]
    offset = np.mean(log_index[s0:s1] - log_growth[s0:s1])
    log_center[s0:s1] = offset + log_growth[s0:s1]

    recentre_levels = {}
    for k, ev in enumerate(recentres, start=1):
        start, end = boundaries[k], boundaries[k + 1]
        lo = max(start - RECENTRE_LOOKBACK, 0)
        level = np.mean(log_index[lo:start]) if start > lo else log_index[start]
        log_center[start:end] = level + (log_growth[start:end] - log_growth[start])
        recentre_levels[ev.date] = float(np.exp(level))

    center = pd.Series(np.exp(log_center), index=dates, name="Estimated_Center")
    return center, {"segments": boundaries, "recentre_levels": recentre_levels}


def implied_regime_slopes(index, events, initial_slope=0.0):
    """Data-implied slope (% p.a.) of the index within each policy regime.

    Regimes are delimited by events that change the band. Slope comes from a
    log-linear OLS fit of the index on time inside each regime.
    """
    dates = index.index
    changes = [e for e in sorted(events, key=lambda e: e.date)
               if e.action != "hold" and dates[0] < e.date <= dates[-1]]
    starts = [dates[0]] + [e.date for e in changes]
    ends = starts[1:] + [dates[-1] + pd.Timedelta(days=1)]
    assumed = [slope_schedule(pd.DatetimeIndex([dates[0]]), events, initial_slope).iloc[0]]
    assumed += [e.slope_after for e in changes]
    labels = ["sample start"] + [f"{e.date.date()} {e.action}" for e in changes]

    rows = []
    for label, start, end, assumed_slope in zip(labels, starts, ends, assumed):
        seg = index[(dates >= start) & (dates < end)]
        if len(seg) < 20:
            implied = np.nan
        else:
            t = (seg.index - seg.index[0]).days.to_numpy(dtype=float) / 365.25
            implied = np.polyfit(t, np.log(seg.to_numpy(dtype=float)), 1)[0] * 100
        rows.append({
            "regime": label,
            "start": pd.Timestamp(start).date(),
            "obs": len(seg),
            "assumed_slope_pct": assumed_slope,
            "implied_slope_pct": implied,
        })
    return pd.DataFrame(rows)


def band_diagnostics(index, center, half_width=0.02):
    deviation = index / center - 1
    outside = deviation.abs() > half_width
    return {
        "share_outside_band": float(outside.mean()),
        "max_deviation_pct": float(deviation.abs().max() * 100),
        "latest_deviation_pct": float(deviation.iloc[-1] * 100),
        "deviation": deviation * 100,
    }
