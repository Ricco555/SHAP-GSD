# SHAP-GSD Graph Exploration — Agent Instructions

This subfolder (`explore/graph/`) contains scripts that visualise the three-granularity
SHAP-GSD attribution decomposition (φ_F, φ_T, φ_N) and the W-sensitivity null result.
Scripts are **not committed to git** (`.gitignore` excludes `explore/`).
All figures are written to `outputs/figures/graph/` (separate from `outputs/figures/explore/`).

---

## Output location
```python
OUT_DIR = ROOT / "outputs" / "figures" / "graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

## Mandatory conventions (same as `explore/AGENT.md`)
- `matplotlib.use("Agg")` at the top of every script.
- No `fig.suptitle(...)`. Panel titles (ax-level `set_title`) are fine.
- Every figure: `<stem>.pdf` + `<stem>.png` + `<stem>.txt` (reasoning + `SUGGESTED FIGURE CAPTION`).
- `bbox_inches="tight"`, `dpi=300` (PDF), `dpi=150` (PNG).
- No `plt.show()`; explicit `plt.close(fig)` after save.

## Style constants
```python
LABEL_FS  = 9
_PHI_F    = "#2a9d8f"   # teal  — feature-group attribution
_PHI_T    = "#e9c46a"   # amber — temporal attribution
_PHI_N    = "#e76f51"   # coral — node/structural attribution
_GRAY     = "#888888"
_GREEN    = "#4caf50"   # dual-axis right axis
```

## Key empirical facts (hardcoded targets — verify in .txt outputs)
| Quantity | Value |
|---|---|
| φ_T fraction | < 0.1% across all classes (temporal null result) |
| φ_N fraction | 28–59% of total \|φ\| per class |
| Recon flow EID | 2116715 — 12 temporal neighbors, node_shap = [+0.275, +0.045, ~0] |
| Backdoor flow EID | 2349890 — 0 temporal neighbors, 11 GNN computation nodes |
| All 200 Backdoor flows | 0 temporal neighbors (UNSW inter-arrival >> W=60s) |
| Median IAT | 5167.9 s |
| W-sensitivity | φ_T saturates W=1800→3600: Δ=+0.0005 |
| W=60s coverage | 1.25% of flows have in-window neighbors |
| W=3600s coverage | 39.4% of flows have in-window neighbors |

## DGL loading (topology_panel.py only)
```python
import dgl
gs, _ = dgl.load_graphs("graphs/test.bin")
g = gs[0]
src_arr = g.edges()[0].numpy()
dst_arr = g.edges()[1].numpy()
eid_arr = g.edata[dgl.EID].numpy()
ts_arr  = g.edata["timestamp"].numpy()
```

## Snapshot loading (topology_panel.py only)
```python
import pickle, bisect
snaps = pickle.load(open("node_state_snapshots/snapshots.pkl", "rb"))
# snaps = {"times": list[int], "states": list[dict[nid -> np.ndarray(15,)]]}
# state[4] = rolling_out_degree, state[8] = dst_port_entropy
def get_node_state(nid, ts_ms):
    idx = bisect.bisect_right(snaps["times"], ts_ms) - 1
    return snaps["states"][idx].get(nid) if idx >= 0 else None
```

## Data sources
- `outputs/explanations/<class>/*.json` — raw SHAP-GSD records
- `outputs/w_ablation/gap_stats.json` — coverage statistics per W
- `outputs/w_ablation/W{60,300,1800,3600}_temporal.csv` — per-flow W-sensitivity
- `graphs/test.bin` — DGL test graph (topology_panel only)
- `graphs/node_id_map.json` — IP ↔ int node-id map (topology_panel only)
- `node_state_snapshots/snapshots.pkl` — rolling node states (topology_panel only)

## Scripts in this folder
| Script | Figure stem | What it shows |
|---|---|---|
| `attribution_decomp.py` | `attribution_decomp` | φ_F/φ_T/φ_N stacked fraction bars + φ_T violin per class |
| `w_sensitivity_full.py` | `w_sensitivity_annotated` | Dual-axis: % coverage + mean φ_T vs W, with saturation annotation |
| `topology_panel.py` | `topology_panel` | SHAP-GSD vs Flat KernelSHAP side-by-side for Recon + Backdoor |
