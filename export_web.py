"""Export model results to docs/data.json for the interactive web dashboard.

Run:  python3 export_web.py
The page in docs/ recomputes the policy band in the browser, so this file
only ships the inputs: the shadow index for each weight mode, the MPS event
calendar, data-implied regime slopes and the basket inference results.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import SGD
from basket_inference import infer_basket
from mps_calendar import INITIAL_SLOPE, MPS_EVENTS, NEXT_REVIEW_LABEL
from policy_band import RECENTRE_LOOKBACK, build_piecewise_center, band_diagnostics, implied_regime_slopes

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "data.json")
MODE_LABELS = {
    "inferred_rolling": "Inferred, rolling 2-year window",
    "inferred_full": "Inferred, full sample",
    "assumed": "Hand-set guess",
}


def rounded(values, digits):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


def regime_key(row):
    """'start' for the opening regime, else the ISO date of the band change."""
    return "start" if row["regime"] == "sample start" else str(row["start"])


def build_payload():
    assumed = SGD.normalize_weights(SGD.PROXY_BASKET)
    data, _ = SGD.download_sgd_crosses(SGD.PROXY_BASKET)
    inference = infer_basket(start=SGD.INFERENCE_START, currencies=list(data.columns),
                             window=SGD.ROLLING_WINDOW_WEEKS)

    modes, checks = {}, {}
    for mode in MODE_LABELS:
        weights = SGD.select_weights(mode, inference, data, assumed)
        index, _, component_returns = SGD.build_shadow_neer(data, weights)
        regimes = implied_regime_slopes(index, MPS_EVENTS, INITIAL_SLOPE)
        if isinstance(weights, pd.DataFrame):
            contrib = (component_returns * weights.reindex(component_returns.index)).rolling(63).sum().iloc[-1]
        else:
            contrib = component_returns.mul(weights, axis=1).rolling(63).sum().iloc[-1]
        modes[mode] = {
            "label": MODE_LABELS[mode],
            "index": rounded(index.values, 4),
            "implied_slopes": {regime_key(r): (None if pd.isna(r["implied_slope_pct"]) else round(r["implied_slope_pct"], 2))
                               for _, r in regimes.iterrows()},
            "regime_obs": {regime_key(r): int(r["obs"]) for _, r in regimes.iterrows()},
            "contrib_63d_pct": {c: round(float(v) * 100, 3) for c, v in contrib.items()},
        }
        # Reference numbers so the browser implementation can be checked against Python.
        center, _ = build_piecewise_center(index, MPS_EVENTS, INITIAL_SLOPE)
        diag = band_diagnostics(index, center, SGD.FIXED_BAND)
        checks[mode] = {
            "latest_deviation_pct": round(diag["latest_deviation_pct"], 4),
            "share_outside_band": round(diag["share_outside_band"], 4),
        }

    ols = inference["ols"].drop(index="const")
    roll = inference["rolling_weights"]
    currencies = list(data.columns)
    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "first_date": str(data.index[0].date()),
            "last_date": str(data.index[-1].date()),
            "next_review": NEXT_REVIEW_LABEL,
            "default_band_pct": SGD.FIXED_BAND * 100,
            "initial_slope_pct": INITIAL_SLOPE,
            "recentre_lookback": RECENTRE_LOOKBACK,
            "default_mode": SGD.WEIGHT_MODE,
            "python_check": checks,
        },
        "dates": [d.strftime("%Y-%m-%d") for d in data.index],
        "usd_per_sgd": rounded(data["USD"].values, 5),
        "modes": modes,
        "events": [
            {"date": str(e.date.date()), "action": e.action, "slope_after": e.slope_after,
             "note": e.note, "verified": e.verified, "off_cycle": e.off_cycle}
            for e in MPS_EVENTS
        ],
        "weights": {
            "currencies": currencies,
            "assumed": {c: round(float(assumed[c]), 4) for c in currencies},
            "full": {c: round(float(inference["full_weights"][c]), 4) for c in currencies},
            "ols": {c: {"coef": round(float(ols.loc[c, "coef"]), 4), "se": round(float(ols.loc[c, "se"]), 4)}
                    for c in currencies},
            "latest": {c: round(float(roll[c].iloc[-1]), 4) for c in currencies},
            "rolling": {
                "dates": [d.strftime("%Y-%m-%d") for d in roll.index],
                "series": {c: rounded(roll[c].values, 4) for c in currencies},
            },
        },
        "fit": {
            "full_r2": round(inference["full_r2"], 4),
            "full_implied_slope_pct": round(inference["full_implied_slope_pct"], 3),
            "latest_r2": round(float(inference["rolling_r2"].iloc[-1]), 4),
            "latest_implied_slope_pct": round(float(inference["rolling_implied_slope_pct"].iloc[-1]), 3),
            "window_weeks": inference["window_weeks"],
            "rolling_drift": {
                "dates": [d.strftime("%Y-%m-%d") for d in inference["rolling_implied_slope_pct"].index],
                "values": rounded(inference["rolling_implied_slope_pct"].values, 3),
            },
        },
    }


def main():
    payload = build_payload()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.0f} KB), data to {payload['meta']['last_date']}")
    print("Python reference:", json.dumps(payload["meta"]["python_check"]))


if __name__ == "__main__":
    main()
