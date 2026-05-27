import json
import pandas as pd

wells = pd.read_csv("remediation_wells.csv")
row = wells[wells["well_id"] == "P008"].iloc[0]
answer = {
    "schema_version": "borden_emergency_modeling.v1",
    "problem_formulation": {
        "decision_variables": {
            "well_selection": "binary choice over remediation_wells.csv",
            "rate_m3_day": "continuous per pumping phase",
            "start_day": "continuous day",
            "duration_days": "continuous day",
            "treatment_type": "discrete T1/T2/T3"
        },
        "objectives": [
            "minimize_total_cost",
            "minimize_residual_contaminant_mass",
            "minimize_receptor_exceedance_risk",
            "maximize_late_time_compliance"
        ],
        "constraints": [
            "budget",
            "well_rate_bounds",
            "time_window_bounds",
            "max_active_wells",
            "total_pumping_capacity",
            "treatment_capacity",
            "physical_validity"
        ]
    },
        "model_summary": {
            "transport_model": "very weak no-pump plume approximation",
            "pumping_perturbation": "not modeled in baseline; replace with pump-and-treat flow/transport simulator",
            "optimization_algorithm": "single feasible placeholder, not Pareto optimization"
    },
    "objective_weights": {
        "total_cost": 0.25,
        "plume_mass": 0.35,
        "receptor_exceedance": 0.25,
        "late_time_risk": 0.15
    },
    "constraints": {
        "budget_usd": 1250000.0,
        "max_active_wells": 6,
        "rate_bounds_m3_day": [0.0, 140.0],
        "start_day_bounds": [2920.0, 5110.0],
        "duration_day_bounds": [30.0, 1460.0]
    },
    "pareto_plans": [],
    "selected_plan_id": "p01_baseline",
    "actions": [],
    "method": "baseline: central downstream single-well pump-and-treat plan using a weak capture-zone approximation",
    "optimization_report": {
        "algorithm": "baseline single feasible placeholder, not a true NSGA-II run",
        "population_size": 1,
        "generations": 0,
        "constraint_handling": "manual feasibility",
        "pareto_selection_rule": "only candidate"
    },
    "optimization_summary": {
        "estimated_total_cost_usd": 565296.0,
        "notes": "Replace this baseline with constrained multi-objective search over well locations, depths, rates, pumping power, timing, and duration."
    }
}
action = {
    "well_id": "P008",
    "x": float(row["x"]),
    "y": float(row["y"]),
    "z": float(row["z"]),
    "screen_depth_m": float(222.0 - row["z"]),
    "treatment_type": "T2",
    "schedule": [{"start_day": 3285.0, "duration_days": 420.0, "rate_m3_day": 45.0}],
    "rate_m3_day": 45.0,
    "pump_power_kw": 18.0,
    "start_day": 3285.0,
    "duration_days": 420.0,
    "treatment_efficiency": 0.80,
}
answer["actions"] = [action]
answer["pareto_plans"] = [
    {
        "plan_id": "p01_baseline",
        "actions": [action],
        "predicted_objectives": {
            "total_cost_usd": 565296.0,
            "residual_mass": 1.0,
            "risk_exceedance": 1.0,
            "late_noncompliance": 1.0
        }
    }
]
with open("answer.json", "w", encoding="utf-8") as f:
    json.dump(answer, f, indent=2)
with open("results.json", "w", encoding="utf-8") as f:
    json.dump({"baseline": True, "selected_well": "P008"}, f, indent=2)
print("Wrote baseline answer.json and results.json")
