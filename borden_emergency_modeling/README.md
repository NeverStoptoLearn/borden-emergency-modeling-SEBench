# Borden Emergency Remediation Modeling and Constrained Multi-Objective Dispatch

## Role

You are a groundwater remediation modeling engineer. You must first build a simplified model for how emergency extraction wells perturb groundwater transport, then use that model to solve a constrained multi-objective pump-and-treat dispatch problem.

## Task

The public files describe a Borden-style 3D contaminant transport scene, early public monitoring observations, candidate extraction/treatment wells, and public risk receptors. Extraction wells change the local flow field and therefore change the contaminant transport equation. Do not treat remediation as a pure constant concentration discount. Build a model that approximates pumping-induced capture, velocity/arrival-time changes, and contaminant mass reduction, then optimize a feasible emergency dispatch plan.

## Required Outputs

Create these files in the task root:

- `model.py`
- `optimize.py` or another script used to produce the final plan
- `answer.json`
- `results.json`
- `report.md`

`model.py` must expose both names below. They may call the same implementation, but the code and report should explain the pump-and-treat physics being approximated:

```python
def simulate_pump_treat(source: dict, plan: dict, receptors, times_days, config: dict, remediation_wells):
    ...

def predict_remediation(plan: dict, receptors, times_days, config: dict):
    ...
```

It must return a pandas DataFrame with at least:

```text
receptor_id,time_days,concentration_mg_L
```

`answer.json` must describe the constrained multi-objective problem and submit a Pareto set, not only one action list:

```json
{
  "schema_version": "borden_emergency_modeling.v1",
  "problem_formulation": {
    "decision_variables": {
      "well_selection": "binary",
      "rate_m3_day": "continuous per phase",
      "start_day": "continuous/integer",
      "duration_days": "continuous/integer",
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
    "transport_model": "velocity-perturbed ADE or capture-zone approximation",
    "pumping_perturbation": "how pumping changes velocity, arrival time, and removal",
    "optimization_algorithm": "NSGA-II / weighted search / constrained differential evolution / other"
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
  "pareto_plans": [
    {
      "plan_id": "p01_low_cost",
      "actions": [
        {
          "well_id": "P001",
          "x": 360.0,
          "y": 245.0,
          "z": 221.0,
          "screen_depth_m": 1.0,
          "treatment_type": "T2",
          "schedule": [
            {"start_day": 3285.0, "duration_days": 180.0, "rate_m3_day": 60.0}
          ],
          "pump_power_kw": 20.0
        }
      ],
      "predicted_objectives": {
        "total_cost_usd": 0.0,
        "residual_mass": 0.0,
        "risk_exceedance": 0.0,
        "late_noncompliance": 0.0
      }
    }
  ],
  "selected_plan_id": "p01_low_cost",
  "method": "brief description of modeling and constrained multi-objective optimization method",
  "optimization_report": {
    "algorithm": "NSGA-II / MOEA-D / random+local search / custom",
    "population_size": 0,
    "generations": 0,
    "constraint_handling": "feasibility-first / penalty / repair",
    "pareto_selection_rule": "knee / robust / budget-aware"
  },
  "optimization_summary": {
    "estimated_total_cost_usd": 0.0,
    "pareto_or_search_notes": "how the selected plan was chosen"
  }
}
```

`well_id` should refer to a row in `remediation_wells.csv`; include `x`, `y`, and `z`
copied from that row so the selected location and screen elevation are explicit. `pump_power_kw`
is a required engineering field for the dispatch plan and should be consistent with the selected
rate, even though the public cost table is expressed in fixed and variable cost terms.

Submit at least three meaningfully different `pareto_plans` when possible: low-cost, balanced, and aggressive/risk-reduction candidates. Single-plan fallback is accepted but cannot score well on Pareto quality. Use the multi-phase schedule form:

```json
{
  "well_id": "P008",
  "treatment_type": "T2",
  "schedule": [
    {"start_day": 2920.0, "duration_days": 180.0, "rate_m3_day": 80.0},
    {"start_day": 3300.0, "duration_days": 240.0, "rate_m3_day": 40.0}
  ]
}
```

Allowed treatment types are listed in `public_remediation_config.json`. They represent
low-cost/high-capacity, medium, and high-efficiency/low-capacity treatment trains.

## Public Inputs

- `public_problem_config.json`: grid, hydrogeology, units, and source prior.
- `public_flow_config.json`: simplified groundwater-flow parameters extracted from the Borden 3D reproduction.
- `public_transport_config.json`: ADE transport parameters and source-history assumptions.
- `public_wells.csv`: public monitoring wells.
- `public_monitoring_data.csv`: early public concentration observations.
- `remediation_wells.csv`: candidate extraction/treatment wells with costs and max rates.
- `public_risk_receptors.csv`: public receptors for local validation.
- `public_remediation_experiments.json`: public pump-and-treat calibration experiments.
- `public_remediation_observations.csv`: public receptor responses for those experiments.
- `public_remediation_config.json`: budget, action bounds, output schema.
- `borden_grid.npz`: grid arrays.
- `tools/local_validate_remediation_model.py`: local public calibration evaluator.
- `baseline_solver.py`: weak baseline answer.
- `model.py`: weak baseline model you may replace.

## Optimization Goals

Minimize:

- total remediation cost
- future plume mass
- receptor exceedance count
- late-time residual risk

Subject to:

- total budget
- total extraction and treatment capacity
- per-well maximum rate
- well availability windows
- per-zone hydraulic limits and drawdown plausibility
- pumping power must be positive for every active well
- action start window
- duration bounds
- maximum practical number of active wells
- treatment type capacity and cost tradeoff
- limited switching / phase changes
- physically plausible nonnegative concentrations

## Pump-And-Treat Physics To Model

Your simulator should approximate the coupled pump-and-treat process:

- transient groundwater flow perturbation from extraction wells;
- local velocity and arrival-time shifts;
- advective-dispersive contaminant transport;
- contaminant mass removal at extraction wells;
- treatment train efficiency/capacity;
- well interference and diminishing returns;
- screen-depth/vertical mismatch effects;
- rebound after pumping stops.

The hidden reference solver is derived from the Borden 3D AdePy reproduction kernel, with additional pump-and-treat perturbation terms. The full hidden solver and hidden source/receptors are not visible to you.

## Scoring

The judge separately tests your model and your final dispatch plan. Public remediation experiments are available for calibration, but hidden benchmark plans and hidden scenarios are generated by the judge. A good final answer alone is not enough because `model.py::predict_remediation` is tested on many unseen schedules.

Total score:

- Problem formulation: 10 points.
- Hidden pump-and-treat model accuracy: 25 points.
- Constraint feasibility over the submitted Pareto set: 15 points.
- Pareto front quality: 25 points.
- Selected final plan effect: 20 points.
- Workflow and report: 5 points.

There is no hard 15-point platform for normal submissions. Hard caps are reserved for missing interfaces, invalid output, unsafe/constant predictions, or cheating.

The hidden reference model includes arrival-time shifts, lateral plume bending, screen-depth effects, nonlinear well interference, capacity limits, and rebound after pumping stops. A constant concentration discount is expected to fail on hidden stress cases.

Do not read or reference hidden scoring files. Do not hard-code hidden receptors, hidden plans, hidden source parameters, or hidden outputs. Do not use MODFLOW/MT3DMS/FloPy external executables.
