# explore/arguments/

Scripts that build the SHAP-level methodology arguments for the current paper scope.

## Output location

```python
from explore._paths import paths
_P = paths()
OUT_DIR = _P["figures"] / "arguments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

ROOT path (three levels from repo root):
```python
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
```

## Script classification

| Script | Status |
|--------|--------|
| `arg1_semantic_grouping.py` | Borderline — logic generic, some colours hardcoded |
| `arg2_absence_driven.py` | UNSW-specific — hardcodes dominant-neg-group lookup per class |
| `arg3_mitre_port.py` | UNSW-specific — hardcodes MITRE port-service table |
| `arg5_node_state.py` | UNSW-specific — hardcodes class-specific expected patterns |
| `arg8_temporal_faithfulness.py` | Borderline — logic generic; W_S constant is configurable |
| `arg10_global_profiles.py` | UNSW-specific — hardcodes literature profile rankings |

## Script outputs

| Script | Figure stem | Paper section | Key result |
|--------|-------------|---------------|------------|
| `arg1_semantic_grouping.py` | `arg1_semantic_grouping` | Methods §3.2 | 6.2 groups cover 80% \|φ\|; 2^48 vs 2^218 (×10^51) |
| `arg2_absence_driven.py` | `arg2_absence_driven` | Results §4.3 | 3 absence-dominant classes; Backdoor 70%, Analysis 66% |
| `arg3_mitre_port.py` | `arg3_mitre_port` | Results §4.4 | DST_PORT_GROUP top-5 in 89% Recon flows |
| `arg5_node_state.py` | `arg5_node_state` | Results §4.5 | φ_N = 13–61% of total \|φ\| |
| `arg8_temporal_faithfulness.py` | `arg8_temporal_faithfulness` | Methods §3.3 | 0/405 violations; 99.2% leakage without causal filter |
| `arg10_global_profiles.py` | `arg10_global_profiles` | Evaluation §5.3 | Backdoor ρ=0.73; mean ρ≈0.37 vs Moustafa & Slay 2015 |

## Data sources

- `outputs/explanations/<class>/*.json` — raw SHAP-GSD records (glob `*.json`)
- `outputs/metrics/fidelity.csv` — per-flow fidelity_plus, top_k_groups (pipe-separated)
- `outputs/metrics/summary.json` — per-class aggregated metrics
- `artifacts/feature_groups.json` — group names, indices, K=48
- `graphs/test.bin` — DGL test graph (arg5, arg8)

## JSON explanation record fields

`edge_id`, `true_label`, `predicted_label`, `predicted_proba`,
`feature_group_names` (K=48), `feature_group_shap` (K=48),
`node_ids`, `node_shap`, `src_novelty_shap`, `dst_novelty_shap`

## fidelity.csv columns

`class_name`, `edge_id`, `true_label`, `predicted_label`,
`p_full`, `p_masked`, `p_kept`, `fidelity_plus`, `fidelity_minus`,
`top_k_groups` (pipe-separated), `runtime_s`

## Style constants

```python
LABEL_FS = 9
_TEAL  = "#2a9d8f"   # presence-driven bars
_CORAL = "#e76f51"   # absence-driven bars
_AMBER = "#e9c46a"   # weak correlation / temporal
_GREEN = "#2ecc71"   # strong correlation (arg10)
_GRAY  = "#888888"
```

