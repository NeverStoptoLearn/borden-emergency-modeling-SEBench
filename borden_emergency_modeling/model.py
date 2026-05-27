from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _base_plume(points: pd.DataFrame, times_days, config):
    # Intentionally weak no-pump starter: enough for format smoke tests, not enough for hidden pump-and-treat scoring.
    times = np.asarray(list(times_days), dtype=float)
    rows = []
    v = float(config["hydrogeological_parameters"]["velocity_m_per_day"])
    source_x, source_y = 260.0, 226.0
    for _, p in points.iterrows():
        x = float(p["x"]); y = float(p["y"])
        conc = []
        for t in times:
            center_x = source_x + v * max(t - 1600.0, 0.0)
            sx = 110.0 + 0.015 * max(t - 3000.0, 0.0)
            sy = 35.0 + 0.004 * max(t - 3000.0, 0.0)
            amp = 0.12 * (1.0 - math.exp(-max(t - 2200.0, 0.0) / 2400.0))
            c = amp * math.exp(-((x - center_x) / sx) ** 2 - ((y - source_y) / sy) ** 2)
            conc.append(max(0.0, c))
        for t, c in zip(times, conc):
            rows.append({"receptor_id": str(p.get("receptor_id", p.get("target_id", ""))), "x": x, "y": y, "z": float(p["z"]), "time_days": float(t), "concentration_mg_L": float(c)})
    return pd.DataFrame(rows)


def predict_remediation(plan: dict, receptors: pd.DataFrame, times_days, config: dict) -> pd.DataFrame:
    pred = _base_plume(receptors, times_days, config)
    return pred


def simulate_pump_treat(source: dict, plan: dict, receptors: pd.DataFrame, times_days, config: dict, remediation_wells=None) -> pd.DataFrame:
    return predict_remediation(plan, receptors, times_days, config)
