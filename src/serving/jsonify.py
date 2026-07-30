"""Make numpy/pandas output valid JSON.

Not optional, and not defensive programming. Two concrete failures without it:

1. `CausalFeatureBuilder.transform` casts every float column to float32. FastAPI's encoder
   raises `ValueError: Out of range float values are not JSON compliant` on np.float32.

2. `float("nan")` serialises to the bare token `NaN`, which is not valid JSON. The browser's
   `await res.json()` throws `SyntaxError: Unexpected token N`, the page shows nothing, and
   there is no error anywhere that names the cause. NaN is routine here: `lo80` on a horizon
   with no interval, `mae` on an empty backtest mask, every `*_lag14` on a short history.
"""

import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def to_jsonable(obj):
    """Recursively convert to something `json.dumps` accepts, mapping NaN/Inf to null."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if obj is pd.NaT or (isinstance(obj, float) and math.isnan(obj)):
        return None
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        # Keys are stringified: a tuple or numpy key is legal in a dict and illegal in JSON.
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)
