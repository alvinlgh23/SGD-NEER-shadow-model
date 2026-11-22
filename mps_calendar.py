"""MAS Monetary Policy Statement (MPS) event calendar.

MAS announces the *direction* of changes to the S$NEER policy band (slope,
width, centre) but never the numbers. Each event below records what MAS said,
plus an analyst-style estimate of the slope that applied afterwards.

- ``action`` is what MAS publicly said: "slope_up", "slope_down", "recentre"
  (re-centre at the prevailing S$NEER level) or "hold".
- ``slope_after`` is a modelling assumption in % p.a. (not disclosed by MAS).
  Market estimates differ between banks; edit these freely.
- ``verified`` marks whether the action was checked against MAS's own
  statement. Unverified rows rely on press coverage and should be confirmed at
  https://www.mas.gov.sg/monetary-policy before relying on them.

Only events that change the band are needed for the model; "hold" rows are kept
so charts can show every review date.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MPSEvent:
    date: pd.Timestamp
    action: str
    slope_after: float  # % p.a., assumption
    note: str
    verified: bool = True
    off_cycle: bool = False


def _e(date, action, slope_after, note, verified=True, off_cycle=False):
    return MPSEvent(pd.Timestamp(date), action, slope_after, note, verified, off_cycle)


# Slope estimates: rough consensus of public bank research at the time.
MPS_EVENTS = [
    _e("2021-10-14", "slope_up", 0.50, "Slope raised slightly from zero"),
    _e("2022-01-25", "slope_up", 1.00, "Off-cycle: slope raised slightly", off_cycle=True),
    _e("2022-04-14", "recentre", 1.50, "Re-centred at prevailing level and slope raised slightly"),
    _e("2022-07-14", "recentre", 1.50, "Off-cycle: re-centred at prevailing level, slope unchanged", off_cycle=True),
    _e("2022-10-14", "slope_up", 2.00, "Slope raised slightly"),
    _e("2023-04-14", "hold", 2.00, "Unchanged"),
    _e("2023-07-14", "hold", 2.00, "Unchanged"),
    _e("2023-10-13", "hold", 2.00, "Unchanged"),
    _e("2024-01-29", "hold", 2.00, "Unchanged"),
    _e("2024-04-12", "hold", 2.00, "Unchanged"),
    _e("2024-07-26", "hold", 2.00, "Unchanged"),
    _e("2024-10-14", "hold", 2.00, "Unchanged"),
    _e("2025-01-24", "slope_down", 1.50, "Slope reduced slightly"),
    _e("2025-04-14", "slope_down", 1.00, "Slope reduced slightly"),
    _e("2025-07-30", "hold", 1.00, "Unchanged"),
    _e("2025-10-14", "hold", 1.00, "Unchanged"),
    _e("2026-01-29", "hold", 1.00, "Unchanged"),
    _e("2026-04-14", "slope_up", 1.50, "Slope raised slightly (first tightening since 2022)"),
    _e("2026-07-27", "slope_up", 1.75, "Slope raised very slightly (per press reports)", verified=False),
    _e("2026-10-14", "hold", 1.75, "Unchanged"),
]

# Slope assumed before the first event in the calendar (flat band from 2020).
INITIAL_SLOPE = 0.0

# Next scheduled review, shown on the dashboard. MAS publishes its review
# calendar in advance; confirm the exact date on mas.gov.sg.
NEXT_REVIEW_LABEL = "Late Jan 2027"
