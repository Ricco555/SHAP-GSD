# SHAP-GSD Arguments Exploration — Agent Instructions

This subfolder (`explore/arguments/`) contains scripts that build the ten SHAP-level
methodology arguments for Paper 2. Six arguments are implemented here (Paper 2 scope);
four are deferred to Papers 3–4 (see `graph_arguments_plan.md`).
Scripts are **not committed to git** (`.gitignore` excludes `explore/`).
All figures are written to `outputs/figures/arguments/`.

---

## Output location
```python
OUT_DIR = ROOT / "outputs" / "figures" / "arguments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

## ROOT path
Scripts sit at `explore/arguments/`, three levels from repo root:
```python
ROOT = Path(__file__).resolve().parent.parent.parent
```

## Mandatory conventions (same as `explore/AGENT.md`)
- `matplotlib.use("Agg")` at the top of every script.
- No `fig.suptitle(...)`. Panel titles (ax-level `set_title`) are fine.
- Every figure: `<stem>.pdf` + `<stem>.png` + `<stem>.txt`.
- `bbox_inches="tight"`, `dpi=300` (PDF), `dpi=150` (PNG).
- No `plt.show()`; explicit `plt.close(fig)` after save.

## Style constants
```python
LABEL_FS = 9
_TEAL  = "#2a9d8f"   # presence-driven bars
_CORAL = "#e76f51"   # absence-driven bars
_AMBER = "#e9c46a"   # weak correlation / temporal
_GREEN = "#2ecc71"   # strong correlation (arg10)
_GRAY  = "#888888"
```

## Extended argument card `.txt` format
Every `.txt` must include these seven sections:
```
WHAT / KEY FINDINGS / PAPER FRAMING / CAPTION /
REVIEWER CHALLENGE / COUNTER-ARGUMENT / PAPER SECTION PLACEMENT
```

## Data sources
- `outputs/explanations/<class>/*.json` — raw SHAP-GSD records (glob `*.json` to catch `_fixed.json` variants)
- `outputs/metrics/fidelity.csv`        — per-flow fidelity_plus, top_k_groups (pipe-separated)
- `outputs/metrics/summary.json`        — per-class aggregated metrics
- `artifacts/feature_groups.json`       — group names, indices, K=48
- `graphs/test.bin`                     — DGL test graph (arg5, arg8)

## JSON fields (from `scripts/06_explain.py`)
`edge_id`, `true_label`, `predicted_label`, `predicted_proba`,
`feature_group_names` (K=48), `feature_group_shap` (K=48),
`node_ids`, `node_shap`, `src_novelty_shap`, `dst_novelty_shap`

## fidelity.csv columns (from `scripts/08_metrics.py`)
`class_name`, `edge_id`, `true_label`, `predicted_label`,
`p_full`, `p_masked`, `p_kept`, `fidelity_plus`, `fidelity_minus`,
`top_k_groups` (pipe-separated), `runtime_s`

## Scripts in this folder

| Script | Figure stem | Paper section | Key result |
|---|---|---|---|
| `arg1_semantic_grouping.py`     | `arg1_semantic_grouping`     | Methods §3.2    | 6.2 groups cover 80% |φ|; 2^48 vs 2^218 (×10^51) |
| `arg2_absence_driven.py`        | `arg2_absence_driven`        | Results §4.3    | 3 absence-dominant classes; Backdoor 70%, Analysis 66% |
| `arg3_mitre_port.py`            | `arg3_mitre_port`            | Results §4.4    | DST_PORT_GROUP top-5 in 89% Recon flows; MITRE tags |
| `arg5_node_state.py`            | `arg5_node_state`            | Results §4.5    | φ_N = 13–61% of total |φ|; GNN context non-trivial |
| `arg8_temporal_faithfulness.py` | `arg8_temporal_faithfulness` | Methods §3.3    | 0/405 violations; 99.2% leakage without causal filter |
| `arg10_global_profiles.py`      | `arg10_global_profiles`      | Evaluation §5.3 | Backdoor ρ=0.73; mean ρ≈0.37 vs Moustafa & Slay 2015 |

## Deferred arguments (Papers 3–4)

| # | Argument | Reason | Target |
|---|---|---|---|
| 4 | Class-conditional background | Needs explainer re-run | Paper 3 |
| 6 | Novel-node attribution | Near-zero on UNSW | Paper 3 |
| 7 | Lateral movement node arc | Needs LSTM multi-window | Paper 4 |
| 9 | Cross-granularity complementarity | φ_T null on UNSW | Paper 3 |
