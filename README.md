# SHAP-GSD

Multi-granularity Shapley explanations for GNN-based Network Intrusion Detection.

Paper 2 of a PhD thesis. Builds on TE-G-SAGE (Paper 1) with IP-level nodes,
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
scripts/06_explain.py       →  artifacts/explanations/
                               artifacts/shap_summaries/
        │
        ▼
scripts/07_visualize.py     →  figures/
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
Paper 1 (TE-G-SAGE) minority-class baselines — Backdoor F1=0.071,
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
- **Feature-group** — ~48 semantic groups (e.g. volume, timing, port service)
- **Temporal neighbourhood** — contribution of past flows to the prediction
- **Node novelty** — whether src/dst IP is new to the network

**Outputs:** `artifacts/explanations/`, `artifacts/shap_summaries/`

---

### Phase 7 — Visualization

```bash
python scripts/07_visualize.py --config configs/experiment_unsw.yaml
```

**Outputs:** `figures/`

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

## Acknowledgements

Experiments performed using the Advanced Computing service provided by the
University of Zagreb University Computing Centre (SRCE).
