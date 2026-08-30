# SHAP-GSD

**Version 2.2.17**

**SHAP-GSD** (SHapley Additive exPlanations on Graph-Structured Data) is a
temporally constrained Shapley explanation framework for GNN-based network
intrusion detection systems (NIDS).

Extends TE-G-SAGE with IP-level nodes, temporally-faithful neighbour sampling,
a 15-dim temporal node state, and three-granularity SHAP coalitions:
feature-group (φ_F), temporal neighbourhood (φ_T), and node novelty (φ_N).

> **Paper:** *SHAP-GSD: Temporal Multi-granular Explanation Method for Graph Neural
> Networks in Network Intrusion Detection* — currently under peer review.
> Full citation to follow upon acceptance.

---

## For reviewers

This repository accompanies *SHAP-GSD: Temporal Multi-granular Explanation Method
for Graph Neural Networks in Network Intrusion Detection* (under peer review).
All results in the paper are from the single-dataset run on **NF-UNSW-NB15-v3**.

Jump to the **[SHAP-GSD](#shap-gsd-under-peer-review)** section for reproduction
instructions. Each dataset run is fully isolated under `runs/<run_id>/`.

### Version note for reviewers

**The current manuscript revision reflects `v2.2.16`** — check out that tag
to match what's currently under review:

```bash
git checkout v2.2.16
```

The original submission snapshot is tagged **`v2.1.0`** (commit `a024a6d`,
2026-06-16):

```bash
git checkout v2.1.0
```

As of `v2.2.17`, `arg10_global_profiles.py` (the global-profile-coherence
exploration figure) no longer scores an undefined literature comparison as
`ρ = +0.000 ("weak")`: any class with no valid comparison — because it has
no entry in the literature lookup table at all, or because its entry's
groups fall outside the figure's compared column set — is now reported as
an explicit exclusion, never folded into the reported mean or agreement
buckets. This does not affect model training, evaluation, or the underlying
SHAP values. It affects only this one exploration figure's own summary
statistic (a previously-reported mean correlation for `NF-UNSW-NB15-v3`
changed from 0.266 to 0.380 as a result).

Everything between those two tags is correctness fixes plus one addition —
`SHAP-GSD[raw]`, a raw per-feature KernelSHAP configuration with no semantic
grouping (see `jobs_draft/config_i_prep_notes.md` for its sample-budget
rationale if reproducing from scratch) — none of which changes the core
training/evaluation pipeline. Full detail for any intermediate tag is in
`git log` / the [Releases](https://github.com/Ricco555/SHAP-GSD/releases)
page. When comparing against a specific reported result,
check out the matching tag rather than assuming a later `main` reproduces
an earlier tag's numbers exactly. Regenerate explanation-derived metrics
and baseline-comparison output from whichever version you're actually
running.

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

## SHAP-GSD (under peer review)

*SHAP-GSD: Temporal Multi-granular Explanation Method for Graph Neural Networks
in Network Intrusion Detection* — full citation to follow upon acceptance.

SHAP-GSD introduces a temporally-faithful, multi-granularity Shapley explanation
framework evaluated on **NF-UNSW-NB15-v3** — classification performance,
SHAP-GSD explanation fidelity/stability, and comparisons against five GNN
explainers, all reproducible with the single command below.

### Dataset

Download **NF-UNSW-NB15-v3** from the UQ eSpace repository:
<https://espace.library.uq.edu.au/view/UQ:6e0eda1>

(The dataset was originally distributed at
`https://staff.itee.uq.edu.au/marius/NIDS_datasets/`, kept here for historical
reference — that link is no longer reachable. Use the eSpace link above.)

Place the downloaded CSV at:

```
data/NF-UNSW-NB15-v3.csv
```

### Quick start

```bash
python scripts/run_dataset.py --csv data/NF-UNSW-NB15-v3.csv
```

Runs all 14 phases in order. Outputs are written to `runs/nf_unsw_nb15_v3/`.
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
                       outputs/metrics/fidelity_temporal.csv (phi_T)
                       outputs/metrics/fidelity_novelty.csv (phi_N)
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
        │
        ▼
14_gateway_distance.py →  runs/nf_unsw_nb15_v3/outputs/topology/gateway_distance.json
                          (malicious-subgraph gateway distance; derives k*)
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
Trials are ranked by the same metric used for early stopping
(`model.early_stopping_metric`; `composite` for NF-UNSW-NB15-v3). 40 epochs per
trial, patience 20 — set in `configs/tuning_grid.yaml`'s `trial:` block. Results
flush after every trial — Ctrl-C and re-run to resume.

Best params (NF-UNSW-NB15-v3), selected under the previous 20-epoch /
patience-5 tuning budget: hidden=128, fanouts=[25,15], dropout=0.1,
lr=0.001, seed=90.

**Phase 6 — SHAP-GSD explanations**

Three-granularity Shapley attributions via KernelSHAP:

| Coalition | Symbol | What it measures |
|-----------|--------|-----------------|
| Feature-group | φ_F | 48 semantic groups (volume, timing, port service, …) |
| Temporal neighbourhood | φ_T | Contribution of in-window past flows |
| Node novelty | φ_N | Whether src/dst IP is new to the network |

1,764 flows explained (200/class × 9 attack + 200 Benign − 7 errors).

**Phase 8 — Metrics**

In addition to the feature-group (φ_F) Fidelity+/Fidelity−/Stability in
`table2.txt`, Phase 8 now computes independent Fidelity+/Fidelity− metrics for
the temporal-neighbourhood (φ_T) and node-novelty (φ_N) granularities
(`fidelity_temporal.csv`, `fidelity_novelty.csv`), each masking/rolling back
only that granularity's own coalition players — not a joint ranking across all
three. `--skip-temporal-fidelity` / `--skip-novelty-fidelity` disable either.

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

**Phase 14 — Gateway distance (topology diagnostic)**

Measures, on the malicious-only training subgraph, the reverse-hop distance
from each attacked host to the internal/external role boundary (`d_gw`), and
derives `k* = max d_gw + 1` — the neighbourhood depth justified by measured
topology rather than inherited as a default. Runs by default on every
pipeline execution; disable per-run with `--skip-gateway-distance` (CLI) or
`topology.gateway_distance.enabled: false` (config). Writes
`outputs/topology/gateway_distance.{json,txt}`; the headline `k_star` is
logged and stored in the JSON. On NF-UNSW-NB15-v3 the malicious subgraph is
degenerate (`d_gw ≡ 1` for every victim, `k* = 2`), which is consistent with
— though does not by itself prove optimal — the model's existing k=2 depth;
deriving model depth from `k*` on other datasets remains a separate, manual
step.

### Exploration figures (`explore/`)

The `explore/` folder contains paper-figure scripts that read from pipeline
outputs and produce publication-quality charts. Run after phases 01–14 complete.

```bash
# Uses runs/nf_unsw_nb15_v3/ outputs automatically via SHAP_GSD_CONFIG
SHAP_GSD_CONFIG=configs/experiment_nf_unsw_nb15_v3.yaml python explore/table2_figure.py
```

`explore/case_studies.py --stratified` draws one flow per class uniformly at
random from the full explained set, with no filtering on correctness,
confidence, or attribution, as a disclosed complement to the curated
case-study selection — writing figures, a `sampling_manifest.json`, and
per-flow diagnostics to `outputs/figures/case_studies_stratified/`.

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
| `explore/arguments/arg2_absence_driven.py` | `figures/arguments/arg2_*` | Dataset-agnostic |
| `explore/arguments/arg3_mitre_port.py` | `figures/arguments/arg3_*` | UNSW-specific |
| `explore/arguments/arg5_node_state.py` | `figures/arguments/arg5_*` | Borderline |
| `explore/arguments/arg8_temporal_faithfulness.py` | `figures/arguments/arg8_*` | Borderline |
| `explore/arguments/arg10_global_profiles.py` | `figures/arguments/arg10_*` | UNSW-specific |

**Dataset-agnostic** scripts work on any completed run.
**UNSW-specific** scripts contain hardcoded class names or flow EIDs.

`explore/case_studies.py` produces one subfolder per attack class under
`figures/case_studies/<Class>/`, each containing the composite 4-panel figure
(`<Class>_<EID>.{pdf,png}`) and four individual panel PNGs named after their
captions (`<Class>_<EID>_{a,b,c,d}_<description>.png`). Run a single class
with `--class <Name>` (e.g. `--class Shellcode`).

See [`explore/README.md`](explore/README.md) for output location, reasoning-file
format, and style constants. Subfolder conventions: [`explore/arguments/README.md`](explore/arguments/README.md)
and [`explore/graph/README.md`](explore/graph/README.md).

### Baseline explainer comparison

Five baseline GNN explainers are benchmarked against SHAP-GSD. These
dependencies are **not** in `requirements.txt` — install only when running
the baseline comparison:

```bash
pip install torch_geometric

git clone https://github.com/HipGraph/GNNShap.git external/gnnshap
pip install -r external/gnnshap/requirements.txt

git clone https://github.com/AlexDuvalinho/GraphSVX.git external/graphsvx
# Do NOT run `pip install -r external/graphsvx/requirements.txt` — it pins
# seaborn==0.10.1 exactly and would downgrade the seaborn already installed
# for src/visualization/metrics_plots.py. GraphSVX's actual import path
# (torch, torch_geometric, networkx, scipy, scikit-learn, seaborn) is already
# covered by SHAP-GSD's own dependencies plus the torch_geometric line above.

git clone https://github.com/AndMastro/EdgeSHAPer.git external/edgeshaper
pip install rdkit rdkit-heatmaps
```

Each wrapper resolves its clone directory as `external/<name>/` relative to
the repo root by default. To use a checkout kept elsewhere, set
`SHAP_GSD_GNNSHAP_DIR`, `SHAP_GSD_GRAPHSVX_DIR`, or `SHAP_GSD_EDGESHAPER_DIR`
to that checkout's root directory before running `scripts/10_baselines.py`.

GNNShap compiles a CUDA extension on first import — the first run can take
several minutes; this is expected, not a hang.

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

topology:
  gateway_distance:
    enabled: true   # phase 14 (gateway-distance diagnostic) runs by default
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
| `outputs/topology/gateway_distance.json` | Malicious-subgraph gateway-distance diagnostic (`k_star`, role assignment, per-class `d_gw`) |
| `outputs/w_ablation/` | Temporal W ablation gap stats and summary |
| `outputs/baselines/` | Baseline explainer comparison results |
| `outputs/figures/` | All publication figures from explore/ scripts |

Runtime-generated directories (`runs/`, `feature_store/`, `graphs/`,
`artifacts/`, `outputs/`) are excluded from version control.

---

## Acknowledgments

The work on SHAP-GSD has been supported by the European Union through the European
Regional Development Fund and the Cohesion Fund under the Competitiveness and
Cohesion Programme 2021–2027, project **GreenSecure360 — Convergent platform
for compliance, security, and sustainable operations** (PK.1.1.12.0210).

Computational experiments were carried out using the advanced computing
service provided by the University of Zagreb University Computing Centre
(SRCE).
