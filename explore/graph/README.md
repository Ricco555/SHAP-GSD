# explore/graph/

Scripts that visualise the three-granularity SHAP-GSD attribution decomposition
(φ_F, φ_T, φ_N) and the W-sensitivity null result.

## Output location

```python
from explore._paths import paths
_P = paths()
OUT_DIR = _P["figures"] / "graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

## Script classification

| Script | Status |
|--------|--------|
| `attribution_decomp.py` | Dataset-agnostic |
| `w_sensitivity_full.py` | Dataset-agnostic |
| `topology_panel.py` | Borderline — hardcodes candidate EIDs per class; logic generic |

## Script outputs

| Script | Figure stem | What it shows |
|--------|-------------|---------------|
| `attribution_decomp.py` | `attribution_decomp` | φ_F/φ_T/φ_N stacked fraction bars + φ_T violin per class |
| `w_sensitivity_full.py` | `w_sensitivity_annotated` | Dual-axis: % coverage + mean φ_T vs W, with saturation annotation |
| `topology_panel.py` | `topology_panel` | SHAP-GSD vs Flat KernelSHAP side-by-side for Recon + Backdoor |

## Data sources

- `outputs/explanations/<class>/*.json` — raw SHAP-GSD records
- `outputs/w_ablation/gap_stats.json` — coverage statistics per W
- `outputs/w_ablation/W{60,300,1800,3600}_temporal.csv` — per-flow W-sensitivity
- `graphs/test.bin` — DGL test graph (`topology_panel.py` only)
- `graphs/node_id_map.json` — IP ↔ int node-id map (`topology_panel.py` only)
- `node_state_snapshots/snapshots.pkl` — rolling node states (`topology_panel.py` only)

## DGL loading (`topology_panel.py`)

```python
import dgl
gs, _ = dgl.load_graphs("graphs/test.bin")
g = gs[0]
src_arr = g.edges()[0].numpy()
dst_arr = g.edges()[1].numpy()
eid_arr = g.edata[dgl.EID].numpy()
ts_arr  = g.edata["timestamp"].numpy()
```

## Snapshot loading (`topology_panel.py`)

```python
import pickle, bisect
snaps = pickle.load(open("node_state_snapshots/snapshots.pkl", "rb"))
# snaps = {"times": list[int], "states": list[dict[nid -> np.ndarray(15,)]]}
def get_node_state(nid, ts_ms):
    idx = bisect.bisect_right(snaps["times"], ts_ms) - 1
    return snaps["states"][idx].get(nid) if idx >= 0 else None
```

## Style constants

```python
LABEL_FS  = 9
_PHI_F    = "#2a9d8f"   # teal  — feature-group attribution
_PHI_T    = "#e9c46a"   # amber — temporal attribution
_PHI_N    = "#e76f51"   # coral — node/structural attribution
_GRAY     = "#888888"
_GREEN    = "#4caf50"   # dual-axis right axis
```
