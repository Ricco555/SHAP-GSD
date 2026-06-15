# SHAP-GSD

**Version 2.1.0**

**SHAP-GSD** (SHapley Additive exPlanations on Graph-Structured Data) is a
temporally constrained Shapley explanation framework for GNN-based network
intrusion detection systems (NIDS).

Extends TE-G-SAGE with IP-level nodes, temporally-faithful neighbour sampling,
a 15-dim temporal node state, and three-granularity SHAP coalitions:
feature-group (φ_F), temporal neighbourhood (φ_T), and node novelty (φ_N).

---

## For reviewers

This repository supports two papers that share the same pipeline code.
Jump to the section that matches the paper you are reviewing:

- **[Paper 2 — SHAP-GSD](#paper-2--shap-gsd-peer-review-pending)** — single-dataset
  experiments on NF-UNSW-NB15-v3; all results reported in the paper come from
  this run.
- **[Paper 3 — Proxy Ground-Truth Evaluation Framework](#paper-3--proxy-ground-truth-evaluation-framework-working-title)** — multi-dataset
  cross-validation using the same pipeline across NF-CSE-CIC-IDS2018-v3,
  NF-ToN-IoT-v3, and NF-BoT-IoT-v3. The pipeline is ready; dataset runs
  are in progress.

Both papers use the `run_dataset.py` orchestrator. Each dataset run is
fully isolated under `runs/<run_id>/` — results never collide.

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

## Paper 2 — SHAP-GSD (peer-review pending)

SHAP-GSD introduces a temporally-faithful, multi-granularity Shapley explanation
framework evaluated on **NF-UNSW-NB15-v3**. The paper reports classification
performance (macro-F1 = 0.508, weighted-F1 = 0.961), SHAP-GSD fidelity and
stability metrics (Table 2), and baseline comparisons against five GNN explainers.
All results are reproducible with the single command below.

### Dataset

Download **NF-UNSW-NB15-v3** from <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
and place it at:

```
data/NF-UNSW-NB15-v3.csv
```

### Quick start

```bash
python scripts/run_dataset.py --csv data/NF-UNSW-NB15-v3.csv
```

Runs all 13 phases in order. Outputs are written to `runs/nf_unsw_nb15_v3/`.
The orchestrator sets `SHAP_GSD_CONFIG=configs/experiment_nf_unsw_nb15_v3.yaml`
for every sub-process automatically.

### Pipeline overview

```
data/NF-UNSW-NB15-v3.csv
        │
        ▼
01_preprocess.py    →  runs/nf_unsw_nb15_v3/feature_store/{train,val,test}/
                       split_indices.json  feature_groups.json
                       balanced_train_indices.npy  class_weights.npy
                       artifacts/transformers/
        │
        ▼
02_build_graph.py   →  runs/nf_unsw_nb15_v3/graphs/{train,val,test}.bin
                       graphs/node_id_map.json
                       node_state_snapshots/
        │
        ▼
03_tune.py          →  runs/nf_unsw_nb15_v3/artifacts/tuning/best_params.json
        │
        ▼
04_train.py         →  runs/nf_unsw_nb15_v3/artifacts/best_model.pt
        │
        ▼
05_evaluate.py      →  runs/nf_unsw_nb15_v3/artifacts/evaluation/metrics.json
                       confusion_matrix.png  roc_curves.png
        │
        ▼
06_explain.py       →  runs/nf_unsw_nb15_v3/outputs/explanations/
        │
        ▼
07_visualize.py     →  runs/nf_unsw_nb15_v3/outputs/figures/
        │
        ▼
08_metrics.py       →  runs/nf_unsw_nb15_v3/outputs/metrics/summary.json
                       outputs/metrics/table2.txt
        │
        ▼
09_w_ablation.py    →  runs/nf_unsw_nb15_v3/outputs/w_ablation/
        │
        ▼
10_baselines.py     →  runs/nf_unsw_nb15_v3/outputs/baselines/comparison_table.txt
        │
        ▼
11_efficiency.py    →  runs/nf_unsw_nb15_v3/outputs/metrics/efficiency.json
        │
        ▼
12_novelty_audit.py →  runs/nf_unsw_nb15_v3/outputs/metrics/novelty_audit.json
        │
        ▼
13_ablations.py     →  runs/nf_unsw_nb15_v3/outputs/metrics/ablation_results.json
```

### Phase notes

**Phase 1 — Data pipeline**

Chronological 60/30/10 split by `FLOW_START_MILLISECONDS`. Fits StandardScaler,
Spearman pruning mask, and OHE on training data only. Stores edge features as
memory-mapped arrays. Class weights use Effective Number of Samples
(Cui et al., CVPR 2019, β=0.9999).

**Phase 2 — Graph construction**

Builds IP-level DGL graphs (one node per unique IP address) and computes the
15-dim temporal node state for all nodes (11 behavioural + 4 seasonal).

**Phase 3 — Hyperparameter tuning**

108-configuration grid search: fanouts × hidden size × dropout × batch size.
Selection metric: val macro-F1. 20 epochs per trial, patience 5. Results flush
after every trial — Ctrl-C and re-run to resume.

Best params (NF-UNSW-NB15-v3): hidden=128, fanouts=[25,15], dropout=0.1,
lr=0.001, seed=90.

**Phase 6 — SHAP-GSD explanations**

Three-granularity Shapley attributions via KernelSHAP:

| Coalition | Symbol | What it measures |
|-----------|--------|-----------------|
| Feature-group | φ_F | 48 semantic groups (volume, timing, port service, …) |
| Temporal neighbourhood | φ_T | Contribution of in-window past flows |
| Node novelty | φ_N | Whether src/dst IP is new to the network |

1,764 flows explained (200/class × 9 attack + 200 Benign − 7 errors).

**Phase 11 — Efficiency audit**

KernelSHAP satisfies the efficiency axiom exactly by construction
(errors ≈ 6.8×10⁻¹⁷, machine epsilon). Runtime slope +0.71 (sub-linear).

**Phase 12 — Node novelty audit**

612/1,764 flows (34.69%) have non-zero φ_N. Attack classes show higher
engagement than benign: Backdoor 48.5%, Analysis 46.2%, Recon 42.5% vs
Benign 25.0%.

**Phase 13 — Ablation study**

| Component removed | Δ macro-F1 |
|-------------------|-----------|
| Node state        | −0.034    |
| Seasonal dims     | −0.010    |
| Balancing         | −0.034    |
| Tuning vs defaults| −0.001    |

### Exploration figures (`explore/`)

The `explore/` folder contains paper-figure scripts that read from pipeline
outputs and produce publication-quality charts. Run after phases 01–13 complete.

```bash
# Uses runs/nf_unsw_nb15_v3/ outputs automatically via SHAP_GSD_CONFIG
SHAP_GSD_CONFIG=configs/experiment_nf_unsw_nb15_v3.yaml python explore/table2_figure.py
```

All path resolution goes through `explore/_paths.py`, which reads
`cfg["run"]["dir"]` so outputs land in the correct `runs/<id>/` subdirectory.

| Script | Output | Status |
|--------|--------|--------|
| `explore/table2_figure.py` | `figures/explore/table2_fidelity_stability.*` | Dataset-agnostic |
| `explore/class_analysis.py` | `figures/explore/class_accuracy_fidelity.*` | Dataset-agnostic |
| `explore/fidelity_distributions.py` | `figures/explore/fidelity_violins.*` | Dataset-agnostic |
| `explore/method_comparison_figure.py` | `figures/explore/method_comparison_fg.*` | Dataset-agnostic |
| `explore/w_ablation_figure.py` | `figures/explore/w_ablation.*` | Dataset-agnostic |
| `explore/shap_scatter.py` | `figures/explore/shap_beeswarm_*.*` | Dataset-agnostic |
| `explore/node_shap_convergence.py` | `figures/explore/node_shap_convergence.*` | Dataset-agnostic |
| `explore/case_studies.py` | `figures/case_studies/<Class>/<Class>_<EID>.{pdf,png}` + 4 individual panels | UNSW-specific |
| `explore/graph/attribution_decomp.py` | `figures/graph/attribution_decomp.*` | Dataset-agnostic |
| `explore/graph/w_sensitivity_full.py` | `figures/graph/w_sensitivity_annotated.*` | Dataset-agnostic |
| `explore/graph/topology_panel.py` | `figures/graph/topology_<Class>_<EID>.*` | UNSW-specific |
| `explore/arguments/arg1_semantic_grouping.py` | `figures/arguments/arg1_*` | Borderline |
| `explore/arguments/arg2_absence_driven.py` | `figures/arguments/arg2_*` | UNSW-specific |
| `explore/arguments/arg3_mitre_port.py` | `figures/arguments/arg3_*` | UNSW-specific |
| `explore/arguments/arg5_node_state.py` | `figures/arguments/arg5_*` | UNSW-specific |
| `explore/arguments/arg8_temporal_faithfulness.py` | `figures/arguments/arg8_*` | Borderline |
| `explore/arguments/arg10_global_profiles.py` | `figures/arguments/arg10_*` | UNSW-specific |

**Dataset-agnostic** scripts work on any completed run.
**UNSW-specific** scripts contain hardcoded class names or flow EIDs (class-name
decoupling is deferred to EX-C).

`explore/case_studies.py` produces one subfolder per attack class under
`figures/case_studies/<Class>/`, each containing the composite 4-panel figure
(`<Class>_<EID>.{pdf,png}`) and four individual panel PNGs named after their
captions (`<Class>_<EID>_{a,b,c,d}_<description>.png`). Run a single class
with `--class <Name>` (e.g. `--class Shellcode`).

See `explore/AGENT.md`, `explore/arguments/AGENT.md`, and `explore/graph/AGENT.md`
for output location, reasoning-file format, and style constants.

### Baseline explainer comparison (Table 2)

Five baseline GNN explainers are benchmarked against SHAP-GSD. These
dependencies are **not** in `requirements.txt` — install only when running
the baseline comparison:

```bash
pip install torch_geometric
pip install gnnshap
pip install edgeshaper
git clone https://github.com/AlexDuvalinho/GraphSVX.git external/graphsvx
pip install -e external/graphsvx
```

```bash
# Full run (~2–4 h):
python scripts/10_baselines.py \
    --config runs/nf_unsw_nb15_v3/configs/experiment_nf_unsw_nb15_v3.yaml

# Quick smoke-test (~10 min, 5 flows per class):
python scripts/10_baselines.py \
    --config runs/nf_unsw_nb15_v3/configs/experiment_nf_unsw_nb15_v3.yaml \
    --n-per-class 5 --baselines all

# Single baseline:
python scripts/10_baselines.py \
    --config runs/nf_unsw_nb15_v3/configs/experiment_nf_unsw_nb15_v3.yaml \
    --baselines gnnexplainer
```

| Method | Coalition space | Fidelity mask |
|--------|----------------|---------------|
| SHAP-GSD | Feature-group [F] | top-5 of 48 semantic groups |
| GNNExplainer | Feature-group [F] | top-5 of 48 semantic groups |
| PGExplainer | Node-coalition [N] | top-3 nodes in k-hop subgraph |
| GNNShap | Node-coalition [N] | top-3 nodes in k-hop subgraph |
| GraphSVX | Node-coalition [N] | top-3 nodes in k-hop subgraph |
| EdgeSHAPer | Node-coalition [N] | top-3 nodes in k-hop subgraph |

SHAP-GSD[F] and GNNExplainer[F] are directly comparable.

### Dashboard

**Live:** <https://shap-gsd.i4s-consult.eu/>

An interactive web app for exploring the SHAP-GSD results is included in
`dashboard/`. It cycles through all 9 attack-class case studies and renders
the three-granularity attribution panels (φ_F, φ_T, φ_N) side by side.

See [`dashboard/README.md`](dashboard/README.md) for install, dev, Docker,
and deployment instructions.

---

## Paper 3 — Proxy Ground-Truth Evaluation Framework (working title)

Paper 3 validates the SHAP-GSD approach across multiple NetFlow datasets to
assess whether explanation quality and node novelty engagement generalise
beyond a single network environment. The pipeline is identical to Paper 2 —
the orchestrator handles per-dataset isolation automatically.

**Status:** pipeline ready (v2.0.0+). Dataset runs in progress.

### Datasets

Download any or all of the following from <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
and place them under `data/`:

```
data/NF-CSE-CIC-IDS2018-v3.csv
data/NF-ToN-IoT-v3.csv
data/NF-BoT-IoT-v3.csv
```

**Column contract:** the last two columns must be named `Label` (binary 0/1)
and `Attack` (string class name). All standard NF-* v3 files already follow
this convention. Class enumeration is automatic — `Benign=0`, all other classes
numbered alphabetically. Nothing is hardcoded.

### Running the pipeline

Each dataset is one command. Results are fully isolated under `runs/<run_id>/`:

```bash
python scripts/run_dataset.py --csv data/NF-CSE-CIC-IDS2018-v3.csv
python scripts/run_dataset.py --csv data/NF-ToN-IoT-v3.csv
python scripts/run_dataset.py --csv data/NF-BoT-IoT-v3.csv
```

Or run all datasets in one invocation (phases execute sequentially per dataset):

```bash
python scripts/run_dataset.py \
    --csv data/NF-CSE-CIC-IDS2018-v3.csv \
          data/NF-ToN-IoT-v3.csv \
          data/NF-BoT-IoT-v3.csv
```

Additional flags:

```bash
# Run only specific phases:
python scripts/run_dataset.py --csv data/NF-ToN-IoT-v3.csv --from-phase 1 --to-phase 2

# Skip the pytest gate after phase 02 (not recommended):
python scripts/run_dataset.py --csv data/NF-ToN-IoT-v3.csv --skip-tests

# Regenerate the per-dataset config (overwrites manual edits):
python scripts/run_dataset.py --csv data/NF-ToN-IoT-v3.csv --force-config
```

### Exploration figures across datasets

Set `SHAP_GSD_CONFIG` to select which dataset run's outputs an exploration
script reads from:

```bash
SHAP_GSD_CONFIG=configs/experiment_nf_ton_iot_v3.yaml python explore/table2_figure.py
SHAP_GSD_CONFIG=configs/experiment_nf_bot_iot_v3.yaml python explore/class_analysis.py
```

Dataset-agnostic scripts (see table above) work on any completed run without
modification.

### What v2.0.0 introduced (multi-dataset foundation)

- Dataset-agnostic label map: `Benign=0`, all other classes alphabetically
- Per-dataset `run.dir` path isolation under `runs/<run_id>/`
- `scripts/run_dataset.py` orchestrator
- Column contract enforced at load time

### What v2.1.0 introduced (explore/ alignment)

- `explore/_paths.py` reads `SHAP_GSD_CONFIG` and resolves all output paths
  to the correct `runs/<id>/` subdirectory
- All 12 exploration scripts and 3 `AGENT.md` files updated for multi-dataset use
- `SHAP_GSD_CONFIG` env var propagated through the orchestrator to every
  sub-process, test, and explore script

---

## Tests

```bash
# Gate tests — run before training
SHAP_GSD_CONFIG=configs/experiment_nf_unsw_nb15_v3.yaml pytest tests/ -v \
    --ignore=tests/test_shap_axioms.py

# SHAP axiom tests — run after Phase 6
SHAP_GSD_CONFIG=configs/experiment_nf_unsw_nb15_v3.yaml pytest tests/test_shap_axioms.py -v

# Against a different dataset run:
SHAP_GSD_CONFIG=configs/experiment_nf_ton_iot_v3.yaml pytest tests/ -v
```

| Test file | What it checks |
|-----------|---------------|
| `test_temporal_sampler.py` | Zero temporal-leakage violations (hard gate) |
| `test_node_state.py` | 15-dim state correctness, novelty rollback |
| `test_feature_groups.py` | Group counts, DST_PORT 16-bin encoding |
| `test_balancer.py` | Oversampling ratios, class weight methods |
| `test_eid_alignment.py` | EID↔feature-store alignment |
| `test_shap_axioms.py` | Efficiency, dummy, symmetry axioms for SHAP-GSD |

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | All defaults — model, graph, compute, reproducibility |
| `configs/experiment_<run_id>.yaml` | Per-dataset overrides, auto-generated by orchestrator |
| `configs/tuning_grid.yaml` | Hyperparameter search space |

Key fields in `default.yaml`:

```yaml
model:
  num_classes: 10       # overridden per dataset
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

All paths are relative to `runs/<run_id>/`.

| Path | Contents |
|------|----------|
| `feature_store/{train,val,test}/` | Memory-mapped edge features, labels, timestamps |
| `split_indices.json` | Chronological split boundaries (τ_train, τ_val) |
| `feature_groups.json` | Semantic feature group definitions |
| `class_weights.npy` | Per-class loss weights from original distribution |
| `graphs/*.bin` | DGL graphs for each split |
| `node_state_snapshots/` | Temporal node state at snapshot intervals |
| `artifacts/transformers/` | Fitted scaler, OHE, Spearman mask |
| `artifacts/tuning/` | Per-trial results + best hyperparameters |
| `artifacts/best_model.pt` | Best model checkpoint |
| `artifacts/label_map.json` | Class name → integer mapping |
| `artifacts/evaluation/` | Test metrics, confusion matrix, ROC curves |
| `outputs/explanations/` | Per-flow SHAP-GSD JSON results |
| `outputs/metrics/summary.json` | Per-class Fidelity+/−, Stability |
| `outputs/metrics/runtime_by_layer.json` | Mean/std/p95 runtime per SHAP granularity |
| `outputs/metrics/efficiency.json` | Shapley efficiency audit results |
| `outputs/metrics/novelty_audit.json` | Node IP topology + per-class novelty engagement |
| `outputs/w_ablation/` | Temporal W ablation gap stats and summary |
| `outputs/baselines/` | Baseline comparison results and Table 2 |
| `outputs/figures/` | All publication figures from explore/ scripts |

Runtime-generated directories (`runs/`, `feature_store/`, `graphs/`,
`artifacts/`, `outputs/`) are excluded from version control.

---

## Acknowledgments

This work has been supported by the European Union through the European
Regional Development Fund and the Cohesion Fund under the Competitiveness and
Cohesion Programme 2021–2027, project **GreenSecure360 — Convergent platform
for compliance, security, and sustainable operations** (PK.1.1.12.0210).

Computational experiments were carried out using the advanced computing
service provided by the University of Zagreb University Computing Centre
(SRCE).
