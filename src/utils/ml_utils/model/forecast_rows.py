"""Where does each forecast horizon land, in time and in step?

Two code paths project the forecast forward and they must agree. `predicted_alert_block` builds
rule-engine panel rows -- `base.copy()` with the vitals overwritten, because the engine reads
about eight axes and needs the rest carried. `chained_history` builds history READINGS with
everything unknown set to NaN, because a feature builder handed a carried-forward weight would
compute rolling moments over a series that never happened.

Different rows, deliberately. But the same arithmetic decides WHEN each horizon sits, and until
this module existed both computed it inline. They happened to agree; nothing enforced it, and a
one-line change to either would have made the engine's horizon-2 card and the symptom path's
horizon-2 row describe different sessions with no test failing.

This is the arithmetic, in one place. It returns the schedule; each caller still builds the row
its consumer needs.
"""

import pandas as pd


def forecast_schedule(base_ts, base_step, forecast: dict, signal: str = "sbp") -> list:
    """`[(key, node, ts, step_offset)]`, ordered by how far ahead each horizon sits.

    `step_offset` is `steps_ahead`, which is `h + 1`: horizon `h` is the target SHIFT, so the
    reading it describes is `h + 1` readings after the last observed one. Reporting `h` here
    understated every horizon by one gap, which is documented at the constants block as a bug
    that was fixed once already -- and it is the kind of thing that gets reintroduced by a
    second implementation, which is the reason this function exists.
    """
    per_h = (forecast or {}).get(signal) or {}
    if not per_h:
        return []

    out = []
    for key in sorted(per_h, key=lambda k: (per_h[k] or {}).get("steps_ahead", 0)):
        node = per_h[key]
        if not isinstance(node, dict) or node.get("point") is None:
            continue
        steps = int(node.get("steps_ahead") or 0)
        days = node.get("days_ahead_est")
        ts = (pd.Timestamp(base_ts) + pd.Timedelta(days=float(days))
              if days is not None else None)
        out.append((key, node, ts, steps))
    return out
