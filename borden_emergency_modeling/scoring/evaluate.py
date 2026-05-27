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

import argparse
import importlib.util
import json
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def suspicious(submission_dir):
    toks = ["hidden_true_region_source", "hidden_receptors", "hidden_benchmark_plans", "hidden_pump_treat_observations", "hidden_reference_solver", "private_generation_record", "hidden_eval_config"]
    bad = []
    for p in Path(submission_dir).rglob("*"):
        if not p.is_file() or "scoring" in p.parts:
            continue
        if p.suffix.lower() in {".py", ".json", ".md", ".txt", ".csv", ".ipynb", ".sh"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:120000]
            for t in toks:
                if t in txt:
                    bad.append(f"{t} in {p.relative_to(submission_dir)}")
        if p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as zf:
                    if any(any(t in n for t in toks) for n in zf.namelist()):
                        bad.append(f"hidden member in {p.relative_to(submission_dir)}")
            except Exception:
                pass
    return bad

def load_agent_model(submission_dir):
    path = Path(submission_dir) / "model.py"
    if not path.exists():
        return None, "model.py not found"
    try:
        spec = importlib.util.spec_from_file_location("agent_model", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent_model"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "predict_remediation"):
            return None, "predict_remediation not found"
        if not hasattr(mod, "simulate_pump_treat"):
            setattr(mod, "simulate_pump_treat", getattr(mod, "predict_remediation"))
        return mod, ""
    except Exception:
        return None, traceback.format_exc(limit=3)

def validate_plan(plan, wells):
    actions = plan.get("actions", []) if isinstance(plan, dict) else []
    if not isinstance(actions, list):
        actions = []
    valid_ids = set(wells["well_id"].astype(str))
    valid = [a for a in actions if str(a.get("well_id", "")) in valid_ids]
    format_score = 0.0
    if valid:
        format_score += 0.8
    if 1 <= len(valid) <= 6:
        format_score += 0.5
    if valid and all(0 <= safe_float(a.get("rate_m3_day")) <= 140 and 2920 <= safe_float(a.get("start_day")) <= 5110 and 30 <= safe_float(a.get("duration_days")) <= 1460 for a in valid):
        format_score += 0.7
    if valid and all(safe_float(a.get("pump_power_kw"), 1.0) > 0 for a in valid):
        format_score += 0.4
    coord_ok = True
    for a in valid:
        row = wells[wells["well_id"].astype(str) == str(a.get("well_id", ""))]
        if row.empty:
            coord_ok = False
            break
        w = row.iloc[0]
        for k in ["x", "y", "z"]:
            if k in a and abs(safe_float(a.get(k)) - float(w[k])) > (2.0 if k != "z" else 0.5):
                coord_ok = False
    if valid and coord_ok:
        format_score += 0.6
    if valid and any(isinstance(a.get("schedule"), list) and len(a.get("schedule")) >= 2 for a in valid):
        format_score += 0.8
    if valid and all(str(a.get("treatment_type", "T2")) in TREATMENT_TYPES for a in valid):
        format_score += 0.7
    if valid and all(float(wells[wells["well_id"].astype(str) == str(a.get("well_id"))].iloc[0]["availability_start_day"]) <= safe_float(a.get("start_day")) <= float(wells[wells["well_id"].astype(str) == str(a.get("well_id"))].iloc[0]["availability_end_day"]) for a in valid):
        format_score += 0.8
    return valid, min(2.0, 0.4 * format_score)

def plan_cost(actions, wells):
    total = 0.0
    for a in actions:
        row = wells[wells["well_id"].astype(str) == str(a.get("well_id", ""))]
        if row.empty:
            continue
        w = row.iloc[0]
        treatment_type = str(a.get("treatment_type", "T2"))
        tech = TREATMENT_TYPES.get(treatment_type, TREATMENT_TYPES["T2"])
        phase_cost = 0.0
        for ph in expand_action_phases(a, wells):
            rate = max(0.0, min(float(w["max_rate_m3_day"]), safe_float(ph.get("rate_m3_day"))))
            dur = max(0.0, safe_float(ph.get("duration_days")))
            phase_cost += (float(w["variable_cost_usd_per_m3"]) * tech["unit_cost"] + float(w.get("energy_cost_usd_per_m3", 0.06))) * rate * dur
        total += float(w["fixed_cost_usd"]) + phase_cost + 48000.0 * tech["unit_cost"]
    return total

def extract_pareto_plans(answer):
    raw = answer.get("pareto_plans", []) if isinstance(answer, dict) else []
    plans = []
    if isinstance(raw, list):
        for i, p in enumerate(raw[:16]):
            if isinstance(p, dict):
                plans.append({
                    "plan_id": str(p.get("plan_id", f"p{i+1:02d}")),
                    "actions": p.get("actions", []),
                    "predicted_objectives": p.get("predicted_objectives", {}),
                })
    if not plans and isinstance(answer, dict) and "actions" in answer:
        plans = [{"plan_id": "single_plan", "actions": answer.get("actions", []), "predicted_objectives": {}}]
    selected = str(answer.get("selected_plan_id", plans[0]["plan_id"] if plans else "single_plan")) if isinstance(answer, dict) else "single_plan"
    if selected not in {p["plan_id"] for p in plans} and plans:
        selected = plans[0]["plan_id"]
    return plans, selected

def formulation_score(answer, plans):
    if not isinstance(answer, dict):
        return 0.0
    score = 0.0
    pf = answer.get("problem_formulation", {})
    dvs = str(pf.get("decision_variables", {})).lower()
    objs = str(pf.get("objectives", [])).lower()
    cons = str(pf.get("constraints", [])).lower()
    if all(k in dvs for k in ["well", "rate", "start", "duration"]):
        score += 2.0
    if ("treatment" in dvs) or ("schedule" in dvs):
        score += 1.0
    if all(k in objs for k in ["cost", "mass", "risk"]):
        score += 2.0
    if ("late" in objs) or ("compliance" in objs):
        score += 1.0
    if all(k in cons for k in ["budget", "rate", "duration", "capacity"]):
        score += 2.0
    if len(plans) >= 3:
        score += 1.0
    report = answer.get("optimization_report", {})
    if isinstance(report, dict) and any(k in str(report).lower() for k in ["nsga", "pareto", "constraint", "feasibility", "penalty"]):
        score += 1.0
    if len(plans) < 3:
        return min(2.0, score)
    if len(plans) < 5:
        return min(6.0, score)
    return min(10.0, score)

def normalized_plan_actions(actions, wells):
    valid_ids = set(wells["well_id"].astype(str))
    out = []
    for a in actions if isinstance(actions, list) else []:
        if str(a.get("well_id", "")) not in valid_ids:
            continue
        aa = dict(a)
        row = wells[wells["well_id"].astype(str) == str(aa.get("well_id"))].iloc[0]
        aa.setdefault("x", float(row["x"]))
        aa.setdefault("y", float(row["y"]))
        aa.setdefault("z", float(row["z"]))
        aa.setdefault("treatment_type", "T2")
        phases = expand_action_phases(aa, wells)
        if phases:
            aa["rate_m3_day"] = phases[0]["rate_m3_day"]
            aa["start_day"] = phases[0]["start_day"]
            aa["duration_days"] = phases[0]["duration_days"]
        out.append(aa)
    return out

def constraint_score(actions, wells, budget=1250000.0):
    actions = normalized_plan_actions(actions, wells)
    if not actions:
        return 0.0, {"feasible": False, "violations": ["no_valid_actions"], "cost": 0.0}
    violations = []
    score = 15.0
    if len({str(a.get("well_id")) for a in actions}) > 6:
        score -= 2.0; violations.append("max_active_wells")
    cost = plan_cost(actions, wells)
    if cost > budget:
        score -= min(3.0, 3.0 * (cost - budget) / max(1.0, budget * 0.4)); violations.append("budget")
    sample_days = np.linspace(2920.0, 5110.0, 16)
    for a in actions:
        row = wells[wells["well_id"].astype(str) == str(a.get("well_id"))].iloc[0]
        max_rate = float(row["max_rate_m3_day"])
        for ph in expand_action_phases(a, wells):
            if ph["rate_m3_day"] < -1e-9 or ph["rate_m3_day"] > max_rate + 1e-9:
                score -= 1.0; violations.append("well_rate_bounds")
            if ph["start_day"] < float(row["availability_start_day"]) - 1e-9 or ph["start_day"] > float(row["availability_end_day"]) + 1e-9:
                score -= 0.8; violations.append("availability_window")
            if ph["duration_days"] < 30.0 or ph["duration_days"] > 1460.0:
                score -= 0.8; violations.append("duration_bounds")
        if len(expand_action_phases(a, wells)) > 3:
            score -= 0.5; violations.append("too_many_switches")
        if str(a.get("treatment_type", "T2")) not in TREATMENT_TYPES:
            score -= 1.0; violations.append("treatment_type")
    for day in sample_days:
        total_q = 0.0
        tech_q = {k: 0.0 for k in TREATMENT_TYPES}
        for a in actions:
            tech = str(a.get("treatment_type", "T2"))
            for ph in expand_action_phases(a, wells):
                if ph["start_day"] <= day <= ph["start_day"] + ph["duration_days"]:
                    total_q += ph["rate_m3_day"]
                    tech_q[tech] = tech_q.get(tech, 0.0) + ph["rate_m3_day"]
        if total_q > 260.0 + 1e-9:
            score -= 1.2; violations.append("total_extraction_capacity")
        if total_q > 220.0 + 1e-9:
            score -= 0.8; violations.append("treatment_capacity")
        for tech, q in tech_q.items():
            if q > TREATMENT_TYPES.get(tech, TREATMENT_TYPES["T2"])["capacity"] + 1e-9:
                score -= 0.6; violations.append(f"{tech}_capacity")
    return max(0.0, min(15.0, score)), {"feasible": score >= 12.0 and not any(v in violations for v in ["budget", "total_extraction_capacity"]), "violations": sorted(set(violations)), "cost": cost}

def nondominated_flags(points):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    flags = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i] + 1e-12) and np.any(pts[j] < pts[i] - 1e-12):
                flags[i] = False
                break
    return flags

def pareto_quality(plan_rows):
    rows = [r for r in plan_rows if r.get("constraint_score", 0.0) >= 12.0 and r.get("effect_quality", 0.0) >= 0.18]
    if len(rows) < 4:
        return 0.0, {"nondominated_ratio": 0.0, "spread": 0.0, "hypervolume_like": 0.0, "knee_quality": 0.0}
    objs = np.array([[r["cost_norm"], r["residual_mass_norm"], r["risk_norm"], r["late_noncompliance_norm"]] for r in rows], dtype=float)
    nd = nondominated_flags(objs)
    nd_ratio = float(np.mean(nd))
    nd_objs = objs[nd]
    ideal = nd_objs.min(axis=0)
    nadir = nd_objs.max(axis=0)
    span = np.maximum(nadir - ideal, 1e-9)
    normalized = (nd_objs - ideal) / span
    spread = float(np.mean(np.std(normalized, axis=0)))
    closeness = 1.0 - np.mean(np.clip(nd_objs, 0.0, 1.0), axis=1)
    hv_like = float(np.mean(np.maximum(0.0, closeness) ** 1.4))
    knee = float(np.max(closeness))
    count_factor = min(1.0, len(rows) / 8.0)
    effect_factor = staged_log_power_norm(float(np.mean([r.get("effect_quality", 0.0) for r in rows])), 0.20, 0.62)
    objective_consistency = float(np.mean([r.get("objective_consistency", 0.0) for r in rows]))
    consistency_factor = staged_log_power_norm(objective_consistency, 0.35, 0.85)
    score = (5.0 * nd_ratio + 10.0 * staged_log_power_norm(hv_like, 0.14, 0.78) + 5.0 * staged_log_power_norm(spread, 0.08, 0.42) + 5.0 * staged_log_power_norm(knee, 0.24, 0.82)) * count_factor * effect_factor * consistency_factor
    return min(25.0, score), {"nondominated_ratio": nd_ratio, "spread": spread, "hypervolume_like": hv_like, "knee_quality": knee, "effect_factor": effect_factor, "objective_consistency": objective_consistency}

def pass_rate(score, max_score):
    max_score = float(max_score)
    if max_score <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(score) / max_score)), 3)

def task_result(score, max_score, pass_threshold=0.60):
    score = round(float(score), 3)
    max_score = round(float(max_score), 3)
    rate = pass_rate(score, max_score)
    return {
        "score": score,
        "max_score": max_score,
        "pass_rate": rate,
        "passed": bool(rate >= pass_threshold),
    }

def attach_task_results(detail):
    """Attach machine-readable per-task scores for judge feedback."""
    detail["task_results"] = {
        "problem_formulation": task_result(detail.get("format_score", 0.0), 10.0, 0.60),
        "hidden_pump_treat_model": task_result(detail.get("model_prediction_score", 0.0), 25.0, 0.50),
        "constraint_feasibility": task_result(detail.get("feasibility_score", 0.0), 15.0, 0.70),
        "pareto_front_quality": task_result(detail.get("pareto_score", 0.0), 25.0, 0.45),
        "selected_plan_effect": task_result(detail.get("selected_plan_score", 0.0), 20.0, 0.45),
        "workflow_and_report": task_result(detail.get("report_score", 0.0), 5.0, 0.40),
        "overall": task_result(detail.get("total_score", 0.0), 100.0, 0.50),
    }
    final_plan = detail.get("metrics", {}).get("final_plan", {})
    if final_plan:
        detail["task_results"]["final_plan_subtasks"] = {
            "residual_mass_reduction": task_result(final_plan.get("mass_score", 0.0), 7.0, 0.50),
            "risk_exceedance_reduction": task_result(final_plan.get("risk_score", 0.0), 6.0, 0.50),
            "late_time_compliance": task_result(final_plan.get("late_score", 0.0), 4.0, 0.50),
            "cost_efficiency": task_result(final_plan.get("cost_score", 0.0), 3.0, 0.50),
        }
    return detail

def feedback_payload(detail):
    metrics = detail.get("metrics", {})
    final_plan = metrics.get("final_plan", {})
    payload = {
        "total_score": detail.get("total_score", 0.0),
        "task_results": detail.get("task_results", {}),
        "model_score_breakdown": metrics.get("model_score_breakdown", {}),
        "pareto_quality": metrics.get("pareto_quality", {}),
        "feasibility_modifiers": metrics.get("feasibility_modifiers", {}),
        "selected_plan_summary": {
            "plan_id": final_plan.get("plan_id"),
            "total_cost_usd": final_plan.get("total_cost_usd"),
            "mass_reduction": final_plan.get("mass_reduction"),
            "risk_reduction": final_plan.get("risk_reduction"),
            "late_compliance": final_plan.get("late_compliance"),
            "effect_quality": final_plan.get("effect_quality"),
            "objective_consistency": final_plan.get("objective_consistency"),
            "violations": final_plan.get("violations", []),
        },
        "score_policy": metrics.get("score_policy", {}),
        "warnings": detail.get("warnings", [])[:5],
        "errors": detail.get("errors", [])[:3],
    }
    return payload

def print_feedback(detail):
    payload = feedback_payload(detail)
    print("SCORE_BREAKDOWN_JSON " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    model = payload.get("model_score_breakdown", {})
    if model:
        print(
            "MODEL_FEEDBACK "
            f"hidden_random={float(model.get('hidden_random_plan_score', 0.0)):.3f}/12 "
            f"hidden_stress={float(model.get('hidden_stress_ood_score', 0.0)):.3f}/8 "
            f"shape={float(model.get('peak_mass_log_shape_score', 0.0)):.3f}/5"
        )
    pareto = payload.get("pareto_quality", {})
    if pareto:
        print(
            "PARETO_FEEDBACK "
            f"nondominated_ratio={float(pareto.get('nondominated_ratio', 0.0)):.3f} "
            f"spread={float(pareto.get('spread', 0.0)):.3f} "
            f"hypervolume_like={float(pareto.get('hypervolume_like', 0.0)):.3f} "
            f"knee_quality={float(pareto.get('knee_quality', 0.0)):.3f}"
        )
    selected = payload.get("selected_plan_summary", {})
    if selected:
        print(
            "SELECTED_PLAN_FEEDBACK "
            f"plan_id={selected.get('plan_id')} "
            f"cost={safe_float(selected.get('total_cost_usd')):.3f} "
            f"mass_reduction={safe_float(selected.get('mass_reduction')):.3f} "
            f"risk_reduction={safe_float(selected.get('risk_reduction')):.3f} "
            f"late_compliance={safe_float(selected.get('late_compliance')):.3f} "
            f"effect_quality={safe_float(selected.get('effect_quality')):.3f} "
            f"objective_consistency={safe_float(selected.get('objective_consistency')):.3f} "
            f"violations={','.join(selected.get('violations', [])) if selected.get('violations') else 'none'}"
        )

def evaluate(submission_dir, case_dir, scoring_dir, output):
    detail = {
        "total_score": 0.0,
        "format_score": 0.0,
        "model_interface_score": 0.0,
        "model_prediction_score": 0.0,
        "feasibility_score": 0.0,
        "pareto_score": 0.0,
        "selected_plan_score": 0.0,
        "report_score": 0.0,
        "metrics": {},
        "warnings": [],
        "errors": [],
    }
    try:
        sub = Path(submission_dir)
        cfg = load_json(Path(case_dir) / "public_problem_config.json")
        wells = pd.read_csv(Path(case_dir) / "remediation_wells.csv")
        answer = load_json(sub / "answer.json") if (sub / "answer.json").exists() else {}
        truth = load_json(Path(scoring_dir) / "hidden_true_region_source.json")
        receptors = pd.read_csv(Path(scoring_dir) / "hidden_receptors.csv")
        hidden_cfg = load_json(Path(scoring_dir) / "hidden_eval_config.json")
        hidden_plans = load_json(Path(scoring_dir) / "hidden_benchmark_plans.json")
        hidden_obs = pd.read_csv(Path(scoring_dir) / "hidden_pump_treat_observations.csv")
        public_experiments = load_json(Path(case_dir) / "public_remediation_experiments.json") if (Path(case_dir) / "public_remediation_experiments.json").exists() else []
        public_obs = pd.read_csv(Path(case_dir) / "public_remediation_observations.csv") if (Path(case_dir) / "public_remediation_observations.csv").exists() else pd.DataFrame()
        times = np.linspace(3650.0, 7300.0, 20)

        detail["warnings"] = suspicious(sub)
        if detail["warnings"]:
            detail["total_score"] = 0.0
            attach_task_results(detail)
            save_json(detail, output)
            print("TOTAL_SCORE 0.000")
            for name, res in detail.get("task_results", {}).items():
                if isinstance(res, dict) and "score" in res:
                    print(f"TASK_RESULT {name} score={res['score']:.3f} max={res['max_score']:.3f} pass_rate={res['pass_rate']:.3f}")
            print_feedback(detail)
            return detail

        pareto_plans, selected_plan_id = extract_pareto_plans(answer)
        detail["format_score"] = formulation_score(answer, pareto_plans)
        selected_actions = []
        if pareto_plans:
            selected_actions = next((p["actions"] for p in pareto_plans if p["plan_id"] == selected_plan_id), pareto_plans[0]["actions"])
        selected_actions = normalized_plan_actions(selected_actions, wells)
        model, model_error = load_agent_model(submission_dir)
        if model is None:
            detail["errors"].append(model_error)
        else:
            # Interface score is awarded by being importable and returning valid rows on a smoke test.
            try:
                smoke_plan = {"actions": [selected_actions[0]]} if selected_actions else {"actions": []}
                pred = model.predict_remediation(smoke_plan, receptors.head(3).copy(), np.asarray(times[:4]), cfg)
                if isinstance(pred, pd.DataFrame) and {"receptor_id", "time_days", "concentration_mg_L"}.issubset(pred.columns):
                    vals = pred["concentration_mg_L"].to_numpy(float)
                    if len(pred) >= 6 and np.all(np.isfinite(vals)) and np.nanmin(vals) >= -1e-12:
                        detail["model_interface_score"] = 3.0
            except Exception:
                detail["errors"].append(traceback.format_exc(limit=3))

            # Hidden model benchmark plans: judge compares agent model to reference pumping-perturbed ADE.
            if detail["model_interface_score"] > 0:
                public_qualities = []
                public_rows = []
                public_receptors = pd.read_csv(Path(case_dir) / "public_risk_receptors.csv")
                public_times = np.linspace(3650.0, 6200.0, 12)
                for pe in public_experiments:
                    exp_id = pe.get("experiment_id", pe.get("name", "public_exp"))
                    ref = public_obs[public_obs["experiment_id"].astype(str) == str(exp_id)].copy()
                    try:
                        got = model.predict_remediation(pe, public_receptors.copy(), np.asarray(public_times), cfg)
                        if not isinstance(got, pd.DataFrame):
                            raise ValueError("predict_remediation did not return DataFrame")
                        m = compute_metrics(ref, got)
                    except Exception:
                        detail["errors"].append(traceback.format_exc(limit=2))
                        m = {"rrmse": float("inf"), "log_rmse": float("inf"), "peak_time_mae": float("inf"), "mass_rel_err": float("inf")}
                    q = plan_model_quality(m)
                    public_qualities.append(q)
                    m = dict(m)
                    m["experiment_id"] = exp_id
                    m["quality"] = round(float(q), 3)
                    public_rows.append(m)

                random_qualities = []
                stress_qualities = []
                all_qualities = []
                metric_rows = []
                for hp in hidden_plans:
                    ref = hidden_obs[hidden_obs["plan_name"].astype(str) == str(hp.get("name", ""))].copy()
                    try:
                        got = model.predict_remediation(hp, receptors.copy(), np.asarray(times), cfg)
                        if not isinstance(got, pd.DataFrame):
                            raise ValueError("predict_remediation did not return DataFrame")
                        m = compute_metrics(ref, got)
                    except Exception:
                        detail["errors"].append(traceback.format_exc(limit=2))
                        m = {"rrmse": float("inf"), "log_rmse": float("inf"), "peak_time_mae": float("inf"), "mass_rel_err": float("inf")}
                    q = plan_model_quality(m)
                    if hp.get("kind") == "stress":
                        stress_qualities.append(q)
                    else:
                        random_qualities.append(q)
                    all_qualities.append(q)
                    m = dict(m)
                    m["plan_name"] = hp.get("name", "hidden_plan")
                    m["family"] = hp.get("family", "")
                    m["kind"] = hp.get("kind", "random")
                    m["quality"] = round(float(q), 3)
                    m["score"] = round(float(q * 25.0), 3)
                    m["max_score"] = 25.0
                    m["pass_rate"] = pass_rate(q, 1.0)
                    metric_rows.append(m)
                public_score = 0.0
                random_score = 9.0 * aggregate_quality(random_qualities)
                stress_score = 11.0 * aggregate_quality(stress_qualities)
                shape_score = 5.0 * aggregate_quality(all_qualities)
                detail["model_prediction_score"] = random_score + stress_score + shape_score
                detail["metrics"]["public_model_validation"] = public_rows
                detail["metrics"]["hidden_model_tests"] = metric_rows
                detail["metrics"]["model_score_breakdown"] = {
                    "public_validation_score_ungraded": public_score,
                    "hidden_random_plan_score": random_score,
                    "hidden_stress_ood_score": stress_score,
                    "peak_mass_log_shape_score": shape_score,
                    "public_quality": aggregate_quality(public_qualities),
                    "hidden_random_quality": aggregate_quality(random_qualities),
                    "hidden_stress_quality": aggregate_quality(stress_qualities),
                }
                finite_rows = [m for m in metric_rows if math.isfinite(m.get("rrmse", float("inf")))]
                if finite_rows:
                    detail["metrics"]["hidden_model_summary"] = {
                        "mean_rrmse": float(np.mean([m["rrmse"] for m in finite_rows])),
                        "mean_log_rmse": float(np.mean([m["log_rmse"] for m in finite_rows])),
                        "mean_mass_rel_err": float(np.mean([m["mass_rel_err"] for m in finite_rows])),
                    }

        # Pareto-set and selected-plan scoring under hidden scenario ensemble.
        budget = 1_250_000.0
        scenarios = scenario_sources(truth, seed=hidden_cfg.get("scenario_seed", 20260527), n=hidden_cfg.get("n_hidden_scenarios", 4))
        no_action = {"actions": []}
        base_by_scenario = {
            sc["name"]: reference_predict(no_action, receptors, times, cfg, sc["source"], wells)
            for sc in scenarios
        }
        plan_rows = []
        for pinfo in pareto_plans:
            actions = normalized_plan_actions(pinfo.get("actions", []), wells)
            cscore, cmeta = constraint_score(actions, wells, budget=budget)
            scenario_metrics = []
            for sc in scenarios:
                ref_base = base_by_scenario[sc["name"]]
                ref_plan = reference_predict({"actions": actions}, receptors, times, cfg, sc["source"], wells)
                base_y = ref_base["concentration_mg_L"].to_numpy(float)
                plan_y = ref_plan["concentration_mg_L"].to_numpy(float)
                mass_reduction = max(0.0, min(1.0, (float(np.sum(base_y)) - float(np.sum(plan_y))) / max(float(np.sum(base_y)), 1e-9)))
                threshold = max(0.02, float(np.quantile(base_y, 0.72)))
                risk_reduction = 1.0 - float(np.sum(plan_y > threshold)) / max(float(np.sum(base_y > threshold)), 1.0)
                late = ref_plan[ref_plan["time_days"] >= times[-6]]
                compliance = float(np.mean(late["concentration_mg_L"].to_numpy(float) <= threshold)) if len(late) else 0.0
                scenario_metrics.append({"mass_reduction": mass_reduction, "risk_reduction": risk_reduction, "late_compliance": compliance})
            mean_mass = float(np.mean([m["mass_reduction"] for m in scenario_metrics])) if scenario_metrics else 0.0
            mean_risk = float(np.mean([m["risk_reduction"] for m in scenario_metrics])) if scenario_metrics else 0.0
            mean_late = float(np.mean([m["late_compliance"] for m in scenario_metrics])) if scenario_metrics else 0.0
            cost_norm = max(0.0, min(1.0, cmeta["cost"] / budget))
            mass_norm_for_quality = staged_log_power_norm(mean_mass, 0.24, 0.86)
            risk_norm_for_quality = staged_log_power_norm(mean_risk, 0.16, 0.72)
            late_norm_for_quality = staged_log_power_norm(mean_late, 0.82, 0.98)
            effect_quality = 0.44 * mass_norm_for_quality + 0.36 * risk_norm_for_quality + 0.20 * late_norm_for_quality
            pred = pinfo.get("predicted_objectives", {}) if isinstance(pinfo.get("predicted_objectives", {}), dict) else {}
            pred_vals = [
                safe_float(pred.get("total_cost_usd"), cmeta["cost"]) / max(budget, 1.0),
                safe_float(pred.get("residual_mass"), 1.0 - mean_mass),
                safe_float(pred.get("risk_exceedance"), 1.0 - mean_risk),
                safe_float(pred.get("late_noncompliance"), 1.0 - mean_late),
            ]
            true_vals = [cost_norm, 1.0 - mean_mass, 1.0 - mean_risk, 1.0 - mean_late]
            pred_err = float(np.mean([abs(float(a) - float(b)) for a, b in zip(pred_vals, true_vals)]))
            objective_consistency = max(0.0, min(1.0, 1.0 - pred_err / 0.55))
            row = {
                "plan_id": pinfo["plan_id"],
                "constraint_score": cscore,
                "feasible": cmeta["feasible"],
                "violations": cmeta["violations"],
                "total_cost_usd": cmeta["cost"],
                "mass_reduction": mean_mass,
                "risk_reduction": mean_risk,
                "late_compliance": mean_late,
                "effect_quality": effect_quality,
                "objective_consistency": objective_consistency,
                "cost_norm": cost_norm,
                "residual_mass_norm": 1.0 - mean_mass,
                "risk_norm": 1.0 - mean_risk,
                "late_noncompliance_norm": 1.0 - mean_late,
            }
            plan_rows.append(row)
        pareto_count_factor = min(1.0, len(plan_rows) / 6.0)
        meaningful_factor = staged_log_power_norm(float(np.mean([r.get("effect_quality", 0.0) for r in plan_rows])) if plan_rows else 0.0, 0.08, 0.42)
        consistency_factor = staged_log_power_norm(float(np.mean([r.get("objective_consistency", 0.0) for r in plan_rows])) if plan_rows else 0.0, 0.25, 0.80)
        detail["feasibility_score"] = (float(np.mean([r["constraint_score"] for r in plan_rows])) if plan_rows else 0.0) * pareto_count_factor * (0.30 + 0.70 * meaningful_factor) * (0.50 + 0.50 * consistency_factor)
        detail["pareto_score"], pareto_meta = pareto_quality(plan_rows)
        selected_row = next((r for r in plan_rows if r["plan_id"] == selected_plan_id), plan_rows[0] if plan_rows else None)
        if selected_row:
            mass_score = 7.0 * staged_log_power_norm(selected_row["mass_reduction"], 0.24, 0.86)
            risk_score = 6.0 * staged_log_power_norm(selected_row["risk_reduction"], 0.16, 0.72)
            late_score = 4.0 * staged_log_power_norm(selected_row["late_compliance"], 0.82, 0.98)
            cost_score = 3.0 * max(0.0, min(1.0, 1.0 - max(0.0, selected_row["total_cost_usd"] - 820000.0) / 620000.0))
            detail["selected_plan_score"] = (mass_score + risk_score + late_score + cost_score) * min(1.0, selected_row["constraint_score"] / 12.0) * pareto_count_factor
            detail["metrics"]["final_plan"] = {
                **selected_row,
                "mass_score": mass_score,
                "risk_score": risk_score,
                "late_score": late_score,
                "cost_score": cost_score,
            }
        detail["metrics"]["pareto_plans"] = plan_rows
        detail["metrics"]["pareto_quality"] = pareto_meta
        detail["metrics"]["feasibility_modifiers"] = {
            "pareto_count_factor": pareto_count_factor,
            "meaningful_effect_factor": meaningful_factor,
            "objective_consistency_factor": consistency_factor,
        }

        report = (sub / "report.md").read_text(encoding="utf-8", errors="ignore") if (sub / "report.md").exists() else ""
        lower = report.lower()
        if len(report.strip()) >= 500:
            detail["report_score"] += 1.0
        for group in [
            ["pumping", "capture", "velocity", "flow"],
            ["advection", "dispersion", "ade", "transport"],
            ["multi-objective", "pareto", "constraint", "budget"],
            ["uncertainty", "robust", "sensitivity", "tradeoff"],
            ["schedule", "phase", "switch", "capacity"],
            ["calibration", "public", "validation", "experiment"],
            ["rebound", "arrival", "time shift", "screen"],
        ]:
            if any(k in lower for k in group):
                detail["report_score"] += 0.7
        detail["report_score"] = min(5.0, detail["report_score"])
        if detail["model_prediction_score"] < 10.0:
            detail["report_score"] = min(2.0, detail["report_score"])

        # Structure-only evidence should not dominate early submissions.  The
        # formulation/report scores are released gradually as the submitted
        # pump-and-treat model starts to generalize on hidden intervention
        # plans, so writing Pareto/constraint keywords is not enough to cross
        # the 30-minute target.
        structure_gate = 0.30 + 0.70 * staged_log_power_norm(detail["model_prediction_score"], 6.0, 18.0)
        raw_format_score = detail["format_score"]
        raw_report_score = detail["report_score"]
        detail["format_score"] = detail["format_score"] * structure_gate
        detail["report_score"] = detail["report_score"] * structure_gate
        detail["metrics"]["structure_score_gate"] = {
            "gate": structure_gate,
            "raw_problem_formulation_score": raw_format_score,
            "raw_workflow_and_report_score": raw_report_score,
            "reason": "problem formulation and report credit are gated by hidden pump-and-treat model credibility",
        }

        raw = sum(detail[k] for k in ["format_score", "model_prediction_score", "feasibility_score", "pareto_score", "selected_plan_score", "report_score"])
        for k, v in list(detail.items()):
            if k.endswith("_score"):
                detail[k] = round(float(v), 3)
        detail["metrics"]["score_policy"] = {
            "raw_total_score": round(float(raw), 3),
            "structure": "10 gated formulation + 25 pump-and-treat model + 15 feasibility + 25 pareto quality + 20 selected plan + 5 gated report",
            "hard_caps_only_for": ["cheating", "missing interface", "invalid/unsafe outputs"],
        }
        detail["total_score"] = round(float(max(0.0, min(100.0, raw))), 3)
    except Exception:
        detail["errors"].append(traceback.format_exc(limit=5))
    attach_task_results(detail)
    save_json(detail, output)
    print(f"CASE borden_emergency_modeling OK score={detail['total_score']:.3f}")
    print(f"TOTAL_SCORE {detail['total_score']:.3f}")
    for name, res in detail.get("task_results", {}).items():
        if isinstance(res, dict) and "score" in res:
            print(f"TASK_RESULT {name} score={res['score']:.3f} max={res['max_score']:.3f} pass_rate={res['pass_rate']:.3f}")
    print_feedback(detail)
    return detail

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission_dir", required=True)
    p.add_argument("--case_dir", required=True)
    p.add_argument("--scoring_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    evaluate(args.submission_dir, args.case_dir, args.scoring_dir, args.output)

if __name__ == "__main__":
    main()
