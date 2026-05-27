from __future__ import annotations
import math
import numpy as np
import pandas as pd

def staged_log_power_norm(value, low, high, alpha=8.0, p=1.6, knee=0.55, early_share=0.42):
    if high <= low:
        return 0.0
    x = max(0.0, min(1.0, (float(value) - float(low)) / (float(high) - float(low))))
    knee = max(1e-9, min(1.0, float(knee)))
    early_share = max(0.0, min(1.0, float(early_share)))
    if x <= knee:
        z = x / knee
        return early_share * math.log1p(float(alpha) * (z ** float(p))) / math.log1p(float(alpha))
    z = (x - knee) / (1.0 - knee)
    return early_share + (1.0 - early_share) * math.log1p(float(alpha) * (z ** float(p))) / math.log1p(float(alpha))

def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def plume_concentrations(source, points, times_days, config):
    hydro = config["hydrogeological_parameters"]
    v = float(hydro["velocity_m_per_day"])
    al = float(hydro["alpha_L_m"])
    at = float(hydro["alpha_TH_m"])
    av = float(hydro["alpha_TV_m"])
    q = float(hydro["q_source_m3_per_day"])
    scale = float(hydro["fallback_source_scale_factor"])
    times = np.asarray(list(times_days), dtype=float)
    offsets = np.linspace(0.0, float(source["duration"]), 9)
    xs = np.linspace(source["x_center"] - source["half_length_x"], source["x_center"] + source["half_length_x"], 5)
    ys = np.linspace(source["y_center"] - source["half_length_y"], source["y_center"] + source["half_length_y"], 5)
    zs = np.array([source["z_center"]], dtype=float)
    n = len(offsets) * len(xs) * len(ys) * len(zs)
    out = []
    for _, p in points.iterrows():
        total = np.zeros_like(times, dtype=float)
        for sx in xs:
            for sy in ys:
                for sz in zs:
                    for off in offsets:
                        age = times - (float(source["t_start"]) + off)
                        active = age > 0
                        if not np.any(active):
                            continue
                        t = np.maximum(age[active], 1e-6)
                        dl = max(al * abs(v), 1e-8)
                        dt = max(at * abs(v), 1e-8)
                        dv = max(av * abs(v), 1e-8)
                        dx = float(p["x"]) - sx - v * t
                        dy = float(p["y"]) - sy
                        dz = float(p["z"]) - sz
                        expo = -((dx * dx) / (4 * dl * t) + (dy * dy) / (4 * dt * t) + (dz * dz) / (4 * dv * t))
                        denom = (4 * math.pi * t) ** 1.5 * math.sqrt(dl * dt * dv)
                        total[active] += (float(source["C0"]) / n) * q * scale * np.exp(expo) / np.maximum(denom, 1e-30)
        for t, c in zip(times, np.maximum(total, 0.0)):
            rec = dict(p)
            rec["time_days"] = float(t)
            rec["concentration_mg_L"] = float(c)
            out.append(rec)
    return pd.DataFrame(out)

TREATMENT_TYPES = {
    "T1": {"efficiency": 0.65, "unit_cost": 0.70, "capacity": 120.0},
    "T2": {"efficiency": 0.80, "unit_cost": 1.00, "capacity": 80.0},
    "T3": {"efficiency": 0.92, "unit_cost": 1.60, "capacity": 40.0},
}

def expand_action_phases(action, wells=None):
    raw_phases = action.get("schedule")
    if isinstance(raw_phases, list) and raw_phases:
        phases = raw_phases
    else:
        phases = [{
            "start_day": action.get("start_day", 3300.0),
            "duration_days": action.get("duration_days", 0.0),
            "rate_m3_day": action.get("rate_m3_day", 0.0),
        }]
    treatment_type = str(action.get("treatment_type", "T2"))
    tech = TREATMENT_TYPES.get(treatment_type, TREATMENT_TYPES["T2"])
    eff = safe_float(action.get("treatment_efficiency"), tech["efficiency"])
    eff = max(0.0, min(0.95, eff))
    out = []
    for ph in phases:
        rate = max(0.0, safe_float(ph.get("rate_m3_day", action.get("rate_m3_day", 0.0))))
        start = safe_float(ph.get("start_day", action.get("start_day", 3300.0)))
        dur = max(0.0, safe_float(ph.get("duration_days", action.get("duration_days", 0.0))))
        out.append({"rate_m3_day": rate, "start_day": start, "duration_days": dur, "treatment_efficiency": eff, "treatment_type": treatment_type})
    return out

def pumping_response_fields(plan, wells, receptors, times_days, config=None, strength=1.0):
    times = np.asarray(list(times_days), dtype=float)
    n_r, n_t = len(receptors), len(times)
    removal_pressure = np.zeros((n_r, n_t), dtype=float)
    time_shift = np.zeros((n_r, n_t), dtype=float)
    y_shift = np.zeros((n_r, n_t), dtype=float)
    rebound = np.zeros((n_r, n_t), dtype=float)
    total_rate_by_t = np.zeros(n_t, dtype=float)
    actions = plan.get("actions", []) if isinstance(plan, dict) else []
    for action in actions:
        row = wells[wells["well_id"].astype(str) == str(action.get("well_id", ""))]
        if row.empty:
            continue
        w = row.iloc[0]
        max_rate = float(w.get("max_rate_m3_day", 120.0))
        capture_coeff = float(w.get("capture_coeff", 1.0))
        drawdown_coeff = float(w.get("drawdown_coeff", 0.006))
        screen_mid = 0.5 * (float(w.get("screen_top_m", w["z"])) + float(w.get("screen_bottom_m", w["z"])))
        for ph in expand_action_phases(action, wells):
            rate = max(0.0, min(max_rate, ph["rate_m3_day"]))
            start = ph["start_day"]
            duration = ph["duration_days"]
            eff = ph["treatment_efficiency"]
            active_time = np.maximum(0.0, np.minimum(times - start, duration))
            active_mask = active_time > 0
            post_time = np.maximum(0.0, times - (start + duration))
            total_rate_by_t[active_mask] += rate
            for i, (_, r) in enumerate(receptors.iterrows()):
                rx, ry, rz = float(r["x"]), float(r["y"]), float(r["z"])
                dx = rx - float(w["x"])
                dy = ry - float(w["y"])
                dz = rz - screen_mid
                downstream = 1.18 if dx >= -15.0 else 0.50
                lateral = math.exp(-(dy / 105.0) ** 2)
                longitudinal = math.exp(-(dx / 290.0) ** 2) * (1.0 + 0.22 * math.tanh(dx / 220.0))
                vertical = math.exp(-(dz / 1.25) ** 2)
                spatial = max(0.0, downstream * lateral * longitudinal * vertical * capture_coeff)
                buildup = 1.0 - np.exp(-rate * active_time / 42000.0)
                pressure = strength * eff * spatial * buildup
                removal_pressure[i, :] += pressure
                time_shift[i, :] += 165.0 * drawdown_coeff * rate * spatial * (1.0 - np.exp(-active_time / 180.0))
                bend_sign = -1.0 if dy > 0 else 1.0
                y_shift[i, :] += bend_sign * 9.0 * spatial * (rate / max(max_rate, 1e-9)) * (1.0 - np.exp(-active_time / 210.0))
                rebound[i, :] += 0.10 * pressure * np.exp(-post_time / 420.0) * (post_time > 0)
    interference = 1.0 / (1.0 + 0.0035 * np.maximum(total_rate_by_t - 160.0, 0.0))
    removal = 1.0 - np.exp(-removal_pressure * interference[None, :])
    removal = np.clip(removal, 0.0, 0.90)
    return {
        "removal": removal,
        "time_shift": np.clip(time_shift, -260.0, 420.0),
        "y_shift": np.clip(y_shift, -28.0, 28.0),
        "rebound": np.clip(rebound, 0.0, 0.22),
        "total_rate_by_time": total_rate_by_t,
    }

_BASE_PLUME_CACHE = {}

def _cached_base_matrix(source, receptors, eval_times, config):
    ids = tuple(receptors["receptor_id"].astype(str))
    coords = tuple((round(float(r.x), 3), round(float(r.y), 3), round(float(r.z), 3)) for r in receptors.itertuples())
    times_key = tuple(round(float(t), 3) for t in eval_times)
    source_key = tuple(round(float(source[k]), 6) for k in ["x_center", "y_center", "z_center", "half_length_x", "half_length_y", "half_length_z", "C0", "t_start", "duration"] if k in source)
    key = (source_key, coords, times_key)
    if key not in _BASE_PLUME_CACHE:
        base = plume_concentrations(source, receptors.reset_index(drop=True), eval_times, config)
        mat = base.pivot_table(index="receptor_id", columns="time_days", values="concentration_mg_L").reindex(index=ids, columns=list(eval_times)).to_numpy()
        _BASE_PLUME_CACHE[key] = mat
    return _BASE_PLUME_CACHE[key]

def reference_predict(plan, receptors, times_days, config, source, wells):
    ids = list(receptors["receptor_id"].astype(str))
    times = np.asarray(list(times_days), dtype=float)
    fields = pumping_response_fields(plan, wells, receptors, times, config)
    eval_times = np.linspace(max(0.0, float(np.min(times)) - 760.0), float(np.max(times)) + 5.0, 120)
    base_mat = _cached_base_matrix(source, receptors, eval_times, config)
    rows = []
    for i, rid in enumerate(ids):
        r0 = receptors.iloc[i]
        base_row = base_mat[i, :]
        for j, t in enumerate(times):
            t_eff = max(0.0, float(t) - float(fields["time_shift"][i, j]))
            tail_t = max(0.0, float(t) - 330.0)
            base_now = float(np.interp(t_eff, eval_times, base_row))
            base_tail = float(np.interp(tail_t, eval_times, base_row))
            bend_penalty = 1.0 + 0.10 * min(1.0, abs(float(fields["y_shift"][i, j])) / 24.0)
            c = bend_penalty * base_now * (1.0 - float(fields["removal"][i, j])) + base_tail * float(fields["rebound"][i, j])
            rows.append({"receptor_id": rid, "x": float(r0["x"]), "y": float(r0["y"]), "z": float(r0["z"]), "time_days": float(t), "concentration_mg_L": float(max(0.0, c))})
    return pd.DataFrame(rows)

def compute_metrics(obs_df, pred_df):
    obs = obs_df.copy()
    pred = pred_df.copy()
    obs["time_key"] = obs["time_days"].astype(float).round(6)
    pred["time_key"] = pred["time_days"].astype(float).round(6)
    merged = obs.merge(pred, on=["receptor_id", "time_key"], suffixes=("_obs", "_pred"))
    if merged.empty:
        return {"rrmse": float("inf"), "log_rmse": float("inf"), "peak_time_mae": float("inf"), "mass_rel_err": float("inf")}
    y = merged["concentration_mg_L_obs"].to_numpy(float)
    yp = np.maximum(np.nan_to_num(merged["concentration_mg_L_pred"].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    rmse = float(np.sqrt(np.mean((yp - y) ** 2)))
    rrmse = rmse / max(float(np.mean(np.abs(y))), 1e-9)
    log_rmse = float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(np.maximum(y, 0.0))) ** 2)))
    mass_rel = abs(float(np.sum(yp) - np.sum(y))) / max(float(np.sum(np.abs(y))), 1e-9)
    peak_errs = []
    for rid, g in merged.groupby("receptor_id"):
        go = g.sort_values("time_days_obs")
        to = go.iloc[int(np.argmax(go["concentration_mg_L_obs"].to_numpy(float)))]["time_days_obs"]
        tp = go.iloc[int(np.argmax(go["concentration_mg_L_pred"].to_numpy(float)))]["time_days_obs"]
        peak_errs.append(abs(float(tp) - float(to)))
    return {"rrmse": rrmse, "log_rmse": log_rmse, "peak_time_mae": float(np.mean(peak_errs)) if peak_errs else float("inf"), "mass_rel_err": mass_rel}

def score_error(err, good, bad):
    progress = max(0.0, min(1.0, (float(bad) - float(err)) / (float(bad) - float(good))))
    return staged_log_power_norm(progress, 0.0, 1.0)

def plan_model_quality(metrics):
    if not math.isfinite(metrics.get("rrmse", float("inf"))):
        return 0.0
    raw = (
        0.55 * score_error(metrics["rrmse"], 0.08, 0.62)
        + 0.18 * score_error(metrics["log_rmse"], 0.012, 0.070)
        + 0.15 * score_error(metrics["mass_rel_err"], 0.04, 0.32)
        + 0.12 * score_error(metrics["peak_time_mae"], 45.0, 320.0)
    )
    return max(0.0, min(1.0, raw))

def aggregate_quality(values):
    vals = sorted([float(v) for v in values if math.isfinite(float(v))])
    if not vals:
        return 0.0
    mean_v = float(np.mean(vals))
    n = max(1, int(math.ceil(0.25 * len(vals))))
    worst_q = float(np.mean(vals[:n]))
    return max(0.0, min(1.0, 0.65 * mean_v + 0.35 * worst_q))

def model_gate_from_score(model_score):
    return max(0.0, min(1.0, (float(model_score) - 8.0) / 22.0)) ** 1.5

def generate_benchmark_plans(wells, seed=20260526, n_random=40, n_stress=12):
    rng = np.random.default_rng(int(seed))
    ids = list(wells["well_id"].astype(str))
    central = list(wells[wells["zone_id"].astype(str) == "central"]["well_id"].astype(str))
    upstream = list(wells[wells["zone_id"].astype(str) == "upstream"]["well_id"].astype(str))
    downstream = list(wells[wells["zone_id"].astype(str) == "downstream"]["well_id"].astype(str))
    lateral = list(wells[np.abs(wells["y"].astype(float) - 245.0) >= 40.0]["well_id"].astype(str))

    def make_action(wid, family, phase_count=None):
        row = wells[wells["well_id"].astype(str) == str(wid)].iloc[0]
        max_rate = float(row["max_rate_m3_day"])
        treatment_type = str(rng.choice(["T1", "T2", "T3"], p=[0.32, 0.48, 0.20]))
        phase_count = int(phase_count or rng.choice([1, 2, 3], p=[0.55, 0.32, 0.13]))
        schedule = []
        base_start = float(rng.choice([2920.0, 3100.0, 3400.0, 3850.0, 4300.0, 4680.0]))
        if family == "late_response":
            base_start = float(rng.uniform(4100.0, 4920.0))
        if family == "pulse_pumping":
            phase_count = max(2, phase_count)
        for k in range(phase_count):
            start = base_start + k * float(rng.uniform(160.0, 420.0))
            duration = float(rng.uniform(70.0, 360.0)) if family == "pulse_pumping" else float(rng.uniform(180.0, 980.0))
            frac = float(rng.uniform(0.25, 0.98))
            if family == "overpump":
                frac = float(rng.uniform(0.78, 1.0))
            schedule.append({
                "start_day": round(max(float(row["availability_start_day"]), min(float(row["availability_end_day"]) - 30.0, start)), 3),
                "duration_days": round(max(30.0, min(1460.0, duration)), 3),
                "rate_m3_day": round(max(0.0, min(max_rate, frac * max_rate)), 3),
            })
        return {
            "well_id": str(wid),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "screen_depth_m": float(222.0 - row["z"]),
            "pump_power_kw": round(4.0 + 0.16 * max(p["rate_m3_day"] for p in schedule), 3),
            "treatment_type": treatment_type,
            "schedule": schedule,
            "rate_m3_day": schedule[0]["rate_m3_day"],
            "start_day": schedule[0]["start_day"],
            "duration_days": schedule[0]["duration_days"],
            "treatment_efficiency": TREATMENT_TYPES[treatment_type]["efficiency"],
        }

    plans = []
    families = ["single_well", "two_well_barrier", "staggered", "late_response", "overpump", "pulse_pumping", "off_axis_capture"]
    pools = {
        "single_well": ids,
        "two_well_barrier": central or ids,
        "staggered": ids,
        "late_response": downstream or ids,
        "overpump": upstream + central or ids,
        "pulse_pumping": ids,
        "off_axis_capture": lateral or ids,
    }
    for i in range(int(n_random)):
        fam = str(rng.choice(families))
        n_wells = int(rng.choice([0, 1, 2, 3, 4, 6], p=[0.04, 0.27, 0.28, 0.20, 0.14, 0.07]))
        if n_wells == 0:
            actions = []
        else:
            pool = pools.get(fam, ids)
            chosen = list(rng.choice(pool, size=min(n_wells, len(pool)), replace=False))
            actions = [make_action(wid, fam) for wid in chosen]
        plans.append({"name": f"random_{i:02d}_{fam}", "family": fam, "kind": "random", "actions": actions})
    stress_families = ["overpump", "pulse_pumping", "late_response", "off_axis_capture"]
    for i in range(int(n_stress)):
        fam = stress_families[i % len(stress_families)]
        pool = pools.get(fam, ids)
        n_wells = int(rng.choice([1, 2, 4, 6], p=[0.20, 0.30, 0.30, 0.20]))
        chosen = list(rng.choice(pool, size=min(n_wells, len(pool)), replace=False))
        actions = [make_action(wid, fam, phase_count=3 if fam == "pulse_pumping" else None) for wid in chosen]
        plans.append({"name": f"stress_{i:02d}_{fam}", "family": fam, "kind": "stress", "actions": actions})
    return plans

def scenario_sources(base_source, seed=20260527, n=8):
    rng = np.random.default_rng(int(seed))
    out = []
    for i in range(int(n)):
        s = dict(base_source)
        s["x_center"] = float(base_source["x_center"] + rng.normal(0.0, 22.0))
        s["y_center"] = float(base_source["y_center"] + rng.normal(0.0, 9.0))
        s["z_center"] = float(base_source["z_center"] + rng.normal(0.0, 0.18))
        s["C0"] = float(base_source["C0"] * rng.uniform(0.78, 1.22))
        s["duration"] = float(base_source["duration"] * rng.uniform(0.70, 1.35))
        s["t_start"] = float(base_source["t_start"] + rng.uniform(-220.0, 260.0))
        out.append({"name": f"scenario_{i+1:02d}", "source": s})
    return out

"""
Private Borden pump-and-treat reference solver for the judge.

This module wraps the Borden 3D AdePy reproduction kernel into the two
interfaces used by the benchmark. It is intentionally judge-only: public files
contain calibration observations and simplified config, not the hidden source,
hidden receptors, or hidden benchmark plans.
"""

def simulate_no_pump(source, receptors, times_days, config):
    return plume_concentrations(source, receptors, times_days, config)

def simulate_pump_treat(source, plan, receptors, times_days, config, remediation_wells):
    return reference_predict(plan, receptors, times_days, config, source, remediation_wells)
