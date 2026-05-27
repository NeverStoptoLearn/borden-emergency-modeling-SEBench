from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location("agent_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def metrics(obs_df, pred_df):
    obs = obs_df.copy()
    pred = pred_df.copy()
    obs["time_key"] = obs["time_days"].astype(float).round(6)
    pred["time_key"] = pred["time_days"].astype(float).round(6)
    merged = obs.merge(pred, on=["receptor_id", "time_key"], suffixes=("_obs", "_pred"))
    if merged.empty:
        return {"rrmse": float("inf"), "log_rmse": float("inf"), "mass_rel_err": float("inf")}
    y = merged["concentration_mg_L_obs"].to_numpy(float)
    yp = np.maximum(np.nan_to_num(merged["concentration_mg_L_pred"].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    rmse = float(np.sqrt(np.mean((yp - y) ** 2)))
    return {
        "rrmse": rmse / max(float(np.mean(np.abs(y))), 1e-9),
        "log_rmse": float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(np.maximum(y, 0.0))) ** 2))),
        "mass_rel_err": abs(float(np.sum(yp) - np.sum(y))) / max(float(np.sum(np.abs(y))), 1e-9),
    }


def main():
    root = Path.cwd()
    model = load_model(root / "model.py")
    config = json.loads((root / "public_problem_config.json").read_text(encoding="utf-8"))
    experiments = json.loads((root / "public_remediation_experiments.json").read_text(encoding="utf-8"))
    receptors = pd.read_csv(root / "public_risk_receptors.csv")
    obs = pd.read_csv(root / "public_remediation_observations.csv")
    times = sorted(obs["time_days"].astype(float).unique())
    rows = []
    for exp in experiments:
        exp_id = exp["experiment_id"]
        got = model.predict_remediation(exp, receptors.copy(), np.asarray(times), config)
        m = metrics(obs[obs["experiment_id"].astype(str) == exp_id], got)
        rows.append({"experiment_id": exp_id, **m})
    print(json.dumps({"public_validation": rows}, indent=2))


if __name__ == "__main__":
    main()
