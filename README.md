# SHAP-GSD

Multi-granularity Shapley explanations for GNN-based Network Intrusion Detection.

Publication pending. Extends TE-G-SAGE with IP-level nodes,
temporally-faithful neighbour sampling, 15-dim node state, and three-granularity
SHAP coalitions: feature-group, temporal neighbourhood, and node novelty.

---

## Requirements

```bash
pip install -r requirements.txt
```

DGL requires a separate install matched to your CUDA version:

```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

---

## Dataset

Download **NF-UNSW-NB15-v3** from <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
and place it at:

```
data/NF-UNSW-NB15-v3.csv
```

---

## Pipeline

```
data/NF-UNSW-NB15-v3.csv
        │
        ▼
scripts/01_preprocess.py    →  feature_store/{train,val,test}/
                               split_indices.json
                               feature_groups.json
                               balanced_train_indices.npy
                               class_weights.npy
                               artifacts/transformers/
        │
        ▼
scripts/02_build_graph.py   →  graphs/{train,val,test}.bin
                               graphs/node_id_map.json
                               node_state_snapshots/
        │
        ▼
scripts/03_tune.py          →  artifacts/tuning/tuning_results.json
                               artifacts/tuning/best_params.json
        │
        ▼
scripts/04_train.py         →  artifacts/best_model.pt
                               artifacts/training_curves.json
        │
        ▼
scripts/05_evaluate.py      →  artifacts/evaluation/metrics.json
                               artifacts/evaluation/confusion_matrix.png
                               artifacts/evaluation/roc_curves.png
        │
        ▼
scripts/06_explain.py       →  outputs/explanations/
        │
        ▼
scripts/07_visualize.py     →  outputs/figures/
        │
        ▼
scripts/08_metrics.py       →  outputs/metrics/summary.json
                               outputs/metrics/table2.txt
        │
        ▼
scripts/09_w_ablation.py    →  outputs/w_ablation/gap_stats.json
                               outputs/w_ablation/summary.txt
        │
        ▼
scripts/10_baselines.py     →  outputs/baselines/summary.json
                               outputs/baselines/comparison_table.txt
```

All scripts read from `configs/experiment_unsw.yaml`, which inherits defaults
from `configs/default.yaml`.

---

## Step-by-step

### Phase 1 — Data pipeline

```bash
python scripts/01_preprocess.py --config configs/experiment_unsw.yaml
```

Chronological 60/30/10 split by `FLOW_START_MILLISECONDS`. Fits StandardScaler,
Spearman pruning mask, and OHE on training data only. Stores edge features as
memory-mapped arrays. Class weights use Effective Number of Samples
(Cui et al., CVPR 2019, β=0.9999).

**Outputs:** `feature_store/`, `split_indices.json`, `feature_groups.json`,
`balanced_train_indices.npy`, `class_weights.npy`, `artifacts/transformers/`

---

### Phase 2 — Graph construction

```bash
python scripts/02_build_graph.py --config configs/experiment_unsw.yaml
```

Builds IP-level DGL graphs (one node per unique IP address) and computes the
15-dim temporal node state for all nodes. Node state snapshots are written
at configurable intervals for use during inference.

**Outputs:** `graphs/`, `node_state_snapshots/`

---

### Phase 3 — Hyperparameter tuning

```bash
python scripts/03_tune.py --config configs/experiment_unsw.yaml
```

108-configuration grid search (3×3×4×3): fanouts × hidden size × dropout ×
batch size. Selection metric: val macro-F1. 20 epochs per trial, patience 5.

**Resume support:** results are flushed after every trial. Ctrl-C and re-run
to resume from where it stopped.

**Outputs:** `artifacts/tuning/tuning_results.json`, `artifacts/tuning/best_params.json`

---

### Phase 4 — Full training

```bash
python scripts/04_train.py --config configs/experiment_unsw.yaml
```

Trains with the best hyperparameters from Phase 3. 50 epochs max, patience 10.

**Outputs:** `artifacts/best_model.pt`, `artifacts/training_curves.json`

---

### Phase 5 — Evaluation

```bash
python scripts/05_evaluate.py --config configs/experiment_unsw.yaml
```

Runs inference on the test split using the same `TemporalNeighborSampler` as
training (no full-neighbourhood inflation). Prints per-class F1 alongside
TE-G-SAGE minority-class baselines — Backdoor F1=0.071,
DoS F1=0.26 — and warns if either is not improved.

**Outputs:** `artifacts/evaluation/metrics.json`, `confusion_matrix.png`, `roc_curves.png`

---

### Phase 6 — SHAP-GSD explanations

Read `specs/03_explainer.md` before running.

```bash
pytest tests/test_shap_axioms.py -v   # must pass first
python scripts/06_explain.py --config configs/experiment_unsw.yaml
```

Three-granularity Shapley attributions via KernelSHAP:
- **Feature-group** (`φ_F`) — 48 semantic groups (e.g. volume, timing, port service)
- **Temporal neighbourhood** (`φ_T`) — contribution of past flows to the prediction
- **Node novelty** (`φ_N`) — whether src/dst IP is new to the network

Each per-flow JSON now includes:

| Field group | Fields |
|-------------|--------|
| Efficiency baselines | `f_baseline_feature`, `f_logit_feature`, `f_baseline_temporal`, `f_logit_temporal`, `f_baseline_node`, `f_logit_node` |
| Per-layer timing | `runtime_feature_s`, `runtime_temporal_s`, `runtime_node_s`, `runtime_s` |

At the end of the run a runtime-by-layer summary is written:

**Outputs:** `outputs/explanations/<class>/<eid>.json`,
`outputs/explanations/summary.csv`,
`outputs/metrics/runtime_by_layer.json`

---

### Phase 11 — Shapley efficiency audit

Standalone post-hoc check: for each existing explanation JSON, computes the
Shapley efficiency error `|Σφ − (f_logit − f_baseline)|` without re-running
KernelSHAP.

```bash
python scripts/11_efficiency.py --config configs/experiment_unsw.yaml
```

**Finding (NF-UNSW-NB15-v3):** KernelSHAP satisfies the efficiency axiom
exactly by construction (errors ≈ 10⁻¹⁷, machine epsilon) because its
constrained WLS solver enforces `Σφ = f(x) − E[f(bg)]` algebraically.

**Outputs:** `outputs/metrics/efficiency.json`

---

### Phase 12 — Node novelty audit

Audits node IP topology and measures how often the node novelty coalition
(φ_N) produces non-zero SHAP values across all 1,764 explained flows.

```bash
# Fast pass — JSON scan only, no artifacts required:
python scripts/12_novelty_audit.py --config configs/experiment_unsw.yaml

# Full pass — adds NSM + test-graph dim-0/dim-1 sampling (run on SRCE):
python scripts/12_novelty_audit.py --config configs/experiment_unsw.yaml --full
```

Three passes:

1. **Node map** — classifies all unique node IPs (RFC1918, loopback, multicast, public)
2. **JSON scan** — counts flows with non-zero `src_novelty_shap` or `dst_novelty_shap`
3. **NSM sample** (`--full`) — samples test flows and measures dim-0 (is\_internal)
   and dim-1 (novelty) distributions from the NodeStateManager

**Finding (NF-UNSW-NB15-v3):** NF-UNSW-NB15-v3 has mixed IP topology — 44 nodes
total: 9 RFC1918/loopback, 1 multicast, 34 public. Node novelty attributions are
non-zero in **612/1,764 flows (34.69%)**. Attack classes show higher engagement
than benign: Backdoor 48.5%, Analysis 46.2%, Recon 42.5% vs Benign 25.0%.

**Outputs:** `outputs/metrics/novelty_audit.json`, `outputs/metrics/novelty_audit.txt`

---

### Phase 7 — Visualization

```bash
python scripts/07_visualize.py --config configs/experiment_unsw.yaml
```

**Outputs:** `figures/`

---

### Phase 8 — Quantitative SHAP-GSD metrics

```bash
python scripts/08_metrics.py --config configs/experiment_unsw.yaml
```

Computes Fidelity+, Fidelity−, and Stability on the 1,764-flow explanation
set from Phase 6. Fidelity+ measures sufficiency (removing top-k groups hurts
prediction); Fidelity− measures necessity (keeping only top-k groups maintains
prediction); Stability is mean per-group φ std across re-runs with different seeds.

**Outputs:** `outputs/metrics/fidelity.csv`, `outputs/metrics/stability.csv`,
`outputs/metrics/summary.json`, `outputs/metrics/table2.txt`

---

### Phase 9 — Temporal window (W) sensitivity ablation

```bash
python scripts/09_w_ablation.py --config configs/experiment_unsw.yaml
```

Step 1: for W ∈ {60, 300, 1800, 3600} s, reports the fraction of flows with
≥1 in-window neighbour and mean in-window neighbour count. Step 2: if any W
has >1% in-window flow rate, re-runs temporal SHAP on a 50-flow subset to
produce non-zero temporal attributions and measure Fidelity+ change.

On NF-UNSW-NB15-v3 all neighbour timestamps fall far outside W=60 s;
temporal φ values are near-zero (confirmed dataset property, reported as a
null result in the paper).

**Outputs:** `outputs/w_ablation/gap_stats.json`, `outputs/w_ablation/gap_stats.txt`,
`outputs/w_ablation/summary.txt`

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | All defaults — model, graph, compute, reproducibility |
| `configs/experiment_unsw.yaml` | Dataset-specific overrides for NF-UNSW-NB15-v3 |
| `configs/tuning_grid.yaml` | Hyperparameter search space |

Key fields in `default.yaml`:

```yaml
model:
  num_classes: 10
  hidden_size: 128
  fanouts: [25, 15]
  batch_size: 512
  max_epochs: 50
  patience: 10

balancer:
  class_weight_method: "effective_num"   # Cui et al. CVPR 2019
  effective_num_beta: 0.9999
```

---

## Generated artifacts

| Path | Contents |
|------|----------|
| `feature_store/{train,val,test}/` | Memory-mapped edge features, labels, timestamps |
| `split_indices.json` | Chronological split boundaries (τ_train, τ_val) |
| `feature_groups.json` | Semantic feature group definitions (K groups) |
| `class_weights.npy` | Per-class loss weights from original distribution |
| `graphs/*.bin` | DGL graphs for each split |
| `node_state_snapshots/` | Temporal node state at snapshot intervals |
| `artifacts/transformers/` | Fitted scaler, OHE, Spearman mask |
| `artifacts/tuning/` | Per-trial results + best hyperparameters |
| `artifacts/best_model.pt` | Best model checkpoint |
| `artifacts/training_curves.json` | Per-epoch loss, macro-F1, per-class F1 |
| `artifacts/evaluation/` | Test metrics, confusion matrix, ROC curves |
| `artifacts/label_map.json` | Class name → integer mapping |
| `outputs/explanations/` | Per-flow SHAP-GSD JSON results (1,764 flows) |
| `outputs/figures/case_studies/` | 4-panel case study figures, 9 attack classes |
| `outputs/metrics/summary.json` | Per-class Fidelity+/−, Stability for Table 2 |
| `outputs/metrics/runtime_by_layer.json` | Mean/std/p95 runtime per SHAP granularity |
| `outputs/metrics/efficiency.json` | Shapley efficiency audit results |
| `outputs/metrics/node_shap_convergence.json` | KernelSHAP convergence at nsamples 128–2048 |
| `outputs/metrics/novelty_audit.json` | Node IP topology + per-class novelty SHAP engagement |
| `outputs/metrics/novelty_audit.txt` | Human-readable novelty audit report |
| `outputs/w_ablation/` | Temporal W ablation gap stats and summary |
| `outputs/baselines/` | Baseline comparison results and Table 2 |

Runtime-generated directories (`feature_store/`, `graphs/`, `artifacts/`,
`outputs/`) are excluded from version control.

---

## Tests

```bash
# Gate tests — must pass before training
pytest tests/ -v --ignore=tests/test_shap_axioms.py

# SHAP axiom tests — run only after Phase 6 is implemented
pytest tests/test_shap_axioms.py -v
```

| Test file | What it checks |
|-----------|---------------|
| `test_temporal_sampler.py` | Zero temporal-leakage violations (hard gate) |
| `test_node_state.py` | 15-dim state correctness, novelty rollback |
| `test_feature_groups.py` | Group counts, DST_PORT 16-bin encoding |
| `test_balancer.py` | Oversampling ratios, class weight methods |
| `test_eid_alignment.py` | EID↔feature-store alignment (skips if graphs not built) |
| `test_shap_axioms.py` | Efficiency, dummy, symmetry axioms for SHAP-GSD |

---

## Baseline Explainer Comparison (Table 2)

Five baseline GNN explainers are benchmarked against SHAP-GSD.

### Additional dependencies

These are **not** in `requirements.txt` — install only when running the
baseline comparison:

```bash
# PGExplainer and GNNExplainer (PyG)
pip install torch_geometric

# GNNShap
pip install gnnshap

# EdgeSHAPer
pip install edgeshaper

# GraphSVX — install from source (PyPI version is stale)
git clone https://github.com/AlexDuvalinho/GraphSVX.git external/graphsvx
pip install -e external/graphsvx
```

GraphSVX ships its own `src/` package that conflicts with this repo's `src/`.
The wrapper (`src/baselines/graphsvx_wrapper.py`) isolates it automatically
via `sys.modules` save/restore — no manual path changes needed.

### How to run

**Full journal run** (all 5 baselines × all test flows, ~2–4 h):

```bash
python scripts/10_baselines.py --config configs/experiment_unsw.yaml
```

**Quick smoke-test** (5 flows per class, ~10 min):

```bash
python scripts/10_baselines.py \
    --config configs/experiment_unsw.yaml \
    --n-per-class 5 \
    --baselines all
```

**Single baseline:**

```bash
python scripts/10_baselines.py \
    --config configs/experiment_unsw.yaml \
    --baselines gnnexplainer          # or pgexplainer, gnnshap, graphsvx, edgeshaper
```

**Skip PGExplainer training** (reuse saved checkpoint from a previous run):

```bash
python scripts/10_baselines.py \
    --config configs/experiment_unsw.yaml \
    --skip-pg-train
```

PGExplainer trains a small MLP over 200 sampled flows × 30 epochs before
inference. The checkpoint is saved to `outputs/baselines/pgexplainer_ckpt.pt`
and reloaded automatically on subsequent runs when `--skip-pg-train` is passed.

### Outputs

| Path | Contents |
|------|----------|
| `outputs/baselines/summary.json` | Per-class Fidelity+/− and runtime for all explainers |
| `outputs/baselines/comparison_table.txt` | Full Table 2 — SHAP-GSD and all five baselines |
| `outputs/baselines/pgexplainer_ckpt.pt` | Trained PGExplainer MLP checkpoint |

### Upper-bound fanout reference

To reproduce the subsampling footnote (fanouts=[999,999] vs [25,15]):

```bash
python scripts/05_evaluate.py \
    --config configs/experiment_unsw_ub_fanout.yaml \
    --checkpoint artifacts/ub_fanout/best_model.pt
```


### Coalition spaces

| Method | Space | Fidelity mask |
|--------|-------|---------------|
| SHAP-GSD | Feature-group [F] | top-5 of 48 semantic groups |
| GNNExplainer | Feature-group [F] | top-5 of 48 semantic groups |
| PGExplainer | Node-coalition [N] | top-3 nodes in k-hop subgraph |
| GNNShap | Node-coalition [N] | top-3 nodes in k-hop subgraph |
| GraphSVX | Node-coalition [N] | top-3 nodes in k-hop subgraph |
| EdgeSHAPer | Node-coalition [N] | top-3 nodes in k-hop subgraph |

SHAP-GSD[F] and GNNExplainer[F] are directly comparable.
PGExplainer[N]/GNNShap[N]/GraphSVX[N]/EdgeSHAPer[N] are directly comparable.

---

## Dashboard

An interactive web app for exploring the SHAP-GSD results is included in `dashboard/`.
It cycles through all 9 attack-class case studies and renders the three-granularity
attribution panels (φ_F, φ_T, φ_N) side by side. Useful for conference demos and
as a companion to the paper figures.

See [`dashboard/README.md`](dashboard/README.md) for install, dev, Docker, and
Railway deployment instructions.

---

## Acknowledgements

Experiments performed using the Advanced Computing service provided by the
University of Zagreb University Computing Centre (SRCE).
