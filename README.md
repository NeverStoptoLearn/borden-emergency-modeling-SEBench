# Borden Emergency Modeling SE-Bench Task

This repository package defines one SE-Bench task:

```text
task_id: borden_emergency_modeling
name: Borden emergency remediation modeling and constrained multi-objective dispatch
```

The task is a Borden-style groundwater emergency remediation benchmark. The Agent must build a simplified pump-and-treat physical response model and then solve a constrained multi-objective emergency dispatch problem. The task is designed for SE-Bench iterative evaluation: the work container contains only public task files, while the judge container contains hidden scoring files and private benchmark scenarios.

---

## 1. Task Summary

The public workspace provides:

- a Borden-style 3D contaminant transport scene;
- public monitoring observations;
- candidate pump-and-treat wells;
- public risk receptors;
- public pump-and-treat calibration experiments;
- engineering constraints, cost parameters, treatment capacities, and action bounds.

The Agent must:

1. understand the groundwater flow and solute transport setting;
2. implement a remediation response model in `model.py`;
3. approximate how extraction wells perturb contaminant transport;
4. solve a constrained multi-objective pump-and-treat dispatch problem;
5. submit a Pareto set of feasible remediation plans in `answer.json`;
6. provide reproducible code and a concise report.

The key challenge is that pumping should not be modeled as a simple constant concentration discount. The submitted model should account for at least approximate capture-zone effects, arrival-time changes, mass removal, treatment capacity, well interference, screen-depth mismatch, and rebound behavior.

---

## 2. Package Structure

After unzipping `borden_emergency_modeling_task.zip`, the expected package layout is:

```text
borden_emergency_modeling_package/
├── README.md
├── task_settings.csv
├── borden_emergency_modeling_public_task_bundle.zip
├── borden_emergency_modeling_private_judge_bundle.zip
├── tasks/
│   └── borden_emergency_modeling.json
└── borden_emergency_modeling/
    ├── README.md
    ├── requirements.txt
    ├── baseline_solver.py
    ├── model.py
    ├── answer_template.json
    ├── borden_grid.npz
    ├── public_problem_config.json
    ├── public_flow_config.json
    ├── public_transport_config.json
    ├── public_wells.csv
    ├── public_monitoring_data.csv
    ├── remediation_wells.csv
    ├── public_risk_receptors.csv
    ├── public_remediation_config.json
    ├── public_remediation_experiments.json
    ├── public_remediation_observations.csv
    ├── tools/
    │   └── local_validate_remediation_model.py
    └── scoring/
        ├── evaluate.py
        ├── hidden_eval_config.json
        ├── hidden_true_region_source.json
        ├── hidden_benchmark_plans.json
        ├── hidden_pump_treat_observations.csv
        ├── hidden_receptors.csv
        ├── hidden_reference_solver.py
        └── private_generation_record_not_for_agent.json
```

The folder `borden_emergency_modeling/` is included for inspection and local development. During SE-Bench execution, the Agent should receive only the public task bundle. Hidden judge files must remain isolated in the judge container.

---

## 3. Important Files

### 3.1 Public files for the Agent

These files are available in the work container:

| File | Purpose |
|---|---|
| `README.md` | Task description, output requirements, public inputs, and scoring overview. |
| `requirements.txt` | Python dependencies for the work environment. |
| `public_problem_config.json` | Borden-style grid, source prior, units, hydrogeological parameters, and source-zone context. |
| `public_flow_config.json` | Regional flow description and expected pump-induced perturbation concepts. |
| `public_transport_config.json` | ADE transport parameters and source representation assumptions. |
| `public_wells.csv` | Public monitoring well locations. |
| `public_monitoring_data.csv` | Public concentration observations. |
| `remediation_wells.csv` | Candidate extraction/treatment wells, costs, rate limits, and engineering fields. |
| `public_risk_receptors.csv` | Public receptors for local validation and risk estimation. |
| `public_remediation_config.json` | Budget, action windows, treatment types, capacities, and dispatch constraints. |
| `public_remediation_experiments.json` | Public pump-and-treat calibration scenarios. |
| `public_remediation_observations.csv` | Public receptor responses for calibration scenarios. |
| `borden_grid.npz` | Grid arrays and scene geometry. |
| `tools/local_validate_remediation_model.py` | Public local validation script for calibration experiments. |
| `baseline_solver.py` | Weak baseline script. |
| `model.py` | Weak baseline model; the Agent is expected to improve or replace it. |

### 3.2 Hidden judge files

The following files are for the judge container only and must not be exposed to the Agent:

```text
scoring/evaluate.py
scoring/hidden_eval_config.json
scoring/hidden_true_region_source.json
scoring/hidden_benchmark_plans.json
scoring/hidden_pump_treat_observations.csv
scoring/hidden_receptors.csv
scoring/hidden_reference_solver.py
scoring/private_generation_record_not_for_agent.json
```

The task JSON explicitly excludes `scoring/`, hidden private records, and score artifacts from submission.

---

## 4. Required Agent Outputs

The Agent must create the following files in the task root:

```text
model.py
optimize.py or another optimization script
answer.json
results.json
report.md
```

`model.py` must expose both of these functions:

```python
def simulate_pump_treat(source: dict, plan: dict, receptors, times_days, config: dict, remediation_wells):
    ...

def predict_remediation(plan: dict, receptors, times_days, config: dict):
    ...
```

Both functions must return a pandas DataFrame with at least these columns:

```text
receptor_id,time_days,concentration_mg_L
```

The submitted `answer.json` must describe the constrained multi-objective problem and submit a Pareto set rather than only one action list.

A minimal high-level output structure is:

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
      "plan_id": "p01_balanced",
      "actions": [
        {
          "well_id": "P008",
          "x": 0.0,
          "y": 0.0,
          "z": 0.0,
          "screen_depth_m": 1.0,
          "treatment_type": "T2",
          "schedule": [
            {"start_day": 2920.0, "duration_days": 180.0, "rate_m3_day": 80.0}
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
  "selected_plan_id": "p01_balanced",
  "method": "brief description of modeling and constrained multi-objective optimization method",
  "optimization_report": {
    "algorithm": "NSGA-II / MOEA-D / random+local search / custom",
    "population_size": 0,
    "generations": 0,
    "constraint_handling": "feasibility-first / penalty / repair",
    "pareto_selection_rule": "knee / robust / budget-aware"
  }
}
```

---

## 5. Public Local Validation

Before running SE-Bench, the public model interface can be checked locally inside `borden_emergency_modeling/`:

```bash
cd borden_emergency_modeling
python -m pip install -r requirements.txt
python tools/local_validate_remediation_model.py
python baseline_solver.py
```

The local validator uses only public calibration experiments and public remediation observations. It is not equivalent to hidden scoring, but it is useful for checking whether `model.py` exposes the required interface and returns a valid concentration DataFrame.

---

## 6. SE-Bench Task Configuration

The task definition is:

```text
tasks/borden_emergency_modeling.json
```

Important fields:

```json
{
  "task_id": "borden_emergency_modeling",
  "name": "Borden emergency remediation modeling and constrained multi-objective dispatch",
  "language": "python310",
  "base_image": "sebench.base.python310:latest",
  "platform": "linux/amd64",
  "cwd": "/home/workspace/borden_emergency_modeling",
  "submit_paths": ["."],
  "internet": false,
  "status": "ready",
  "judge": {
    "parser": "score_sum",
    "score_direction": "maximize",
    "selection": "best_score"
  }
}
```

The work container downloads:

```text
borden_emergency_modeling_public_task_bundle.zip
```

The judge container downloads:

```text
borden_emergency_modeling_private_judge_bundle.zip
```

Both bundles are served from the package directory through a temporary HTTP server during SE-Bench build.

---

## 7. Setup Package HTTP Server

The task JSON expects the two task bundles to be reachable at:

```text
http://host.docker.internal:8000/borden_emergency_modeling_public_task_bundle.zip
http://host.docker.internal:8000/borden_emergency_modeling_private_judge_bundle.zip
```

Start an HTTP server from the package root:

```bash
cd /root/borden_emergency_pkg/borden_emergency_modeling_package
python3 -m http.server 8000 --bind 0.0.0.0
```

Recommended tmux workflow:

```bash
tmux new -s emergency-http
cd /root/borden_emergency_pkg/borden_emergency_modeling_package
python3 -m http.server 8000 --bind 0.0.0.0
```

Detach without stopping the server:

```text
Ctrl+B, then D
```

If the session already exists:

```bash
tmux attach -t emergency-http
```

If port 8000 is already occupied:

```bash
ss -ltnp | grep 8000
kill -9 <PID>
```

or use another port and update the bundle URLs in the task JSON accordingly.

---

## 8. Install SE-Bench

Example installation:

```bash
cd /root/borden_emergency_pkg
unzip SE-bench-v0.2.1.zip
cd SE-bench-v0.2.1

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e .
```

Check installation:

```bash
python -m sebench --help
docker ps
```

If Docker is not running:

```bash
systemctl start docker
systemctl enable docker
```

---

## 9. Register the Task

Copy the task JSON into the SE-Bench `tasks/` directory:

```bash
cp /root/borden_emergency_pkg/borden_emergency_modeling_package/tasks/borden_emergency_modeling.json \
   /root/borden_emergency_pkg/SE-bench-v0.2.1/tasks/borden_emergency_modeling.json
```

Validate the JSON:

```bash
cd /root/borden_emergency_pkg/SE-bench-v0.2.1
python -m json.tool tasks/borden_emergency_modeling.json > /tmp/check_borden_task.json
```

List tasks:

```bash
python -m sebench list --all
```

You should see:

```text
borden_emergency_modeling
```

---

## 10. Build the Task Images

Make sure the HTTP server is running before build.

```bash
cd /root/borden_emergency_pkg/SE-bench-v0.2.1
source .venv/bin/activate

python -m sebench build --task borden_emergency_modeling
```

A successful build means:

- the work image can download and unpack the public bundle;
- the judge image can download and unpack the private judge bundle;
- public files are visible in the work container;
- hidden judge files are isolated from the work container.

---

## 11. Start the Judge Server

Open a second SSH terminal:

```bash
cd /root/borden_emergency_pkg/SE-bench-v0.2.1
source .venv/bin/activate

python -m sebench serve
```

Keep this process running while doing smoke tests or Agent runs.

---

## 12. Stage 2: Smoke Test

Stage 2 checks task integration. It is not intended to produce a high score.

A simple smoke workflow can be:

```bash
cat > /root/borden_emergency_pkg/smoke_borden_emergency.sh <<'EOF'
#!/bin/bash
set -eux

pwd
find . -maxdepth 2 -type f | sort | head -100

python - <<'PY'
import json
import pandas as pd
from pathlib import Path

for name in [
    "public_problem_config.json",
    "public_flow_config.json",
    "public_transport_config.json",
    "public_remediation_config.json",
    "public_remediation_experiments.json",
    "public_remediation_observations.csv",
    "public_risk_receptors.csv",
    "remediation_wells.csv",
    "model.py",
    "baseline_solver.py",
]:
    print(name, Path(name).exists())

print(pd.read_csv("remediation_wells.csv").head())
print(pd.read_csv("public_risk_receptors.csv").head())
PY

python baseline_solver.py || true

sebench-submit
EOF

chmod +x /root/borden_emergency_pkg/smoke_borden_emergency.sh
```

Run smoke test:

```bash
cd /root/borden_emergency_pkg/SE-bench-v0.2.1
source .venv/bin/activate

python -m sebench run \
  --task borden_emergency_modeling \
  --custom-workflow /root/borden_emergency_pkg/smoke_borden_emergency.sh \
  --run-id borden-emergency-smoke-001 \
  --timeout 600
```

Expected judge output contains case scores and a total score, for example:

```text
CASE 0000_problem_formulation ...
CASE 0001_model_accuracy ...
CASE 0002_constraint_feasibility ...
CASE 0003_pareto_quality ...
CASE 0004_selected_plan_effect ...
CASE 0005_workflow_report ...
TOTAL_SCORE ...
```

If `TOTAL_SCORE` appears, the SE-Bench integration loop is working.

---

## 13. Stage 3: Run an Agent

Keep the judge server running, configure your Agent API keys, and run:

```bash
cd /root/borden_emergency_pkg/SE-bench-v0.2.1
source .venv/bin/activate

python -m sebench run \
  --task borden_emergency_modeling \
  --agent <agent_name> \
  --model <model_name> \
  --run-id borden-emergency-agent-001 \
  --timeout 7200 \
  --eval-interval 300
```

Replace:

```text
<agent_name>
<model_name>
```

with the values required by your SE-Bench deployment.

The Agent is expected to iteratively improve the model and dispatch plan based on judge feedback. The task uses:

```json
"selection": "best_score"
```

so the final reported result is the best score across all submissions in the run.

---

## 14. View Scores

After an Agent run, inspect:

```bash
cat logs/runs/borden-emergency-agent-001/borden_emergency_modeling/final_result.json
```

With `jq`:

```bash
jq '{best_score, best_round, total_rounds, timed_out, runtime_seconds}' \
logs/runs/borden-emergency-agent-001/borden_emergency_modeling/final_result.json
```

View submission history:

```bash
jq '.entries[] | select(.type=="submission") | {round, score, pass_rate, status}' \
logs/runs/borden-emergency-agent-001/borden_emergency_modeling/run_history.json
```

Quick grep:

```bash
grep -R "TOTAL_SCORE\|Best score" logs/runs/borden-emergency-agent-001 -n
```

Optional visualizer:

```bash
python -m sebench visualizer \
  --runs-dir logs/runs \
  --host 0.0.0.0 \
  --port 8000
```

Open:

```text
http://<server-ip>:8000
```

Use a different port if port 8000 is already used by the package HTTP server.

---

## 15. Scoring Rubric

The judge separately evaluates the submitted model and the final dispatch solution.

Total score: 100 points.

| Component | Points |
|---|---:|
| Problem formulation | 10 |
| Hidden pump-and-treat model accuracy | 25 |
| Constraint feasibility over the submitted Pareto set | 15 |
| Pareto front quality | 25 |
| Selected final plan effect | 20 |
| Workflow and report | 5 |

Hard caps are reserved for missing interfaces, invalid output, unsafe constant predictions, hidden-file access, or other cheating behavior.

The hidden benchmark evaluates unseen schedules, receptors, and scenarios. A constant concentration discount or a single unvalidated dispatch plan should not score well.

---

## 16. Debugging Checklist

### HTTP server

```bash
curl http://127.0.0.1:8000/
```

Expected: file listing including the public and private bundle zip files.

### Task JSON

```bash
python -m json.tool tasks/borden_emergency_modeling.json > /tmp/check.json
grep -n '"task_id"' tasks/borden_emergency_modeling.json
```

### Build problems

If build fails:

- confirm the HTTP server is running;
- confirm bundle filenames match the JSON URLs;
- confirm Docker is running;
- confirm the base image exists or can be pulled;
- inspect the failing `setup_cmds` block.

### Missing score

If no score is parsed, confirm `evaluate.py` prints:

```text
TOTAL_SCORE <number>
```

### Hidden leakage

The work container should not expose:

```text
scoring/
hidden_*
private_generation_record_not_for_agent.json
```

The task JSON removes these files from the work directory and excludes them from submissions.

---

## 17. Suggested Stage-2 Evidence

For task integration review, capture screenshots or logs showing:

1. `python -m sebench list --all` includes `borden_emergency_modeling`;
2. `python -m sebench build --task borden_emergency_modeling` succeeds;
3. package HTTP server is reachable;
4. smoke workflow reads public files;
5. smoke workflow submits successfully;
6. judge returns `TOTAL_SCORE`;
7. work container has no hidden judge files.

---

## 18. Suggested Stage-3 Evidence

For Agent/debug review, capture screenshots or logs showing:

1. the Agent run command;
2. the Agent creates or modifies `model.py`;
3. the Agent creates `answer.json`, `results.json`, and `report.md`;
4. at least one `sebench-submit` result;
5. `final_result.json` with `best_score`;
6. `run_history.json` showing submission score progression;
7. the final archive path.

---

## 19. Notes on Task Difficulty

This task is intentionally harder than a direct source-coordinate inversion problem. The Agent must model intervention physics and constrained dispatch jointly. Good solutions should:

- fit public remediation experiments without overfitting them;
- generalize to hidden pump-and-treat schedules;
- maintain physical feasibility and capacity constraints;
- submit diverse Pareto plans;
- choose a robust selected plan rather than a single aggressive schedule;
- explain the physical surrogate and optimization strategy.

---

## 20. Quick Command Summary

```bash
# Package HTTP server
cd /root/borden_emergency_pkg/borden_emergency_modeling_package
python3 -m http.server 8000 --bind 0.0.0.0

# SE-Bench
cd /root/borden_emergency_pkg/SE-bench-v0.2.1
source .venv/bin/activate
python -m sebench list --all
python -m sebench build --task borden_emergency_modeling
python -m sebench serve

# Smoke test
python -m sebench run \
  --task borden_emergency_modeling \
  --custom-workflow /root/borden_emergency_pkg/smoke_borden_emergency.sh \
  --run-id borden-emergency-smoke-001 \
  --timeout 600

# Agent run
python -m sebench run \
  --task borden_emergency_modeling \
  --agent <agent_name> \
  --model <model_name> \
  --run-id borden-emergency-agent-001 \
  --timeout 7200 \
  --eval-interval 300
```
