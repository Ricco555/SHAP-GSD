# Plan: SHAP Methodology Arguments — Paper 2 Implementation

## Context

Paper 2 makes ten distinct SHAP-level arguments. Five are fully supported by existing Phase 6–8 outputs (explanation JSONs, fidelity metrics, baselines). This plan implements those five in `explore/arguments/`, producing one exploration script + one figure (PDF/PNG) + one extended "argument card" `.txt` per argument. The argument cards double as paper-section draft material with pre-written reviewer challenges and counter-arguments.

**Working directory:** `/home/ricco555/src/phd-i4sec/SHAP-GSD`

---

## Scope

### Paper 2 (implemented)

| # | Argument | Paper section | Status | Key result |
|---|---|---|---|---|
| 1 | Semantic feature grouping | Methods §3.2 | ✓ done | 6.2 groups → 80% \|φ\|; ×10^51 coalition reduction |
| 2 | Absence-driven attribution | Results §4.3 | ✓ done | 3 absence-dominant classes; Backdoor 70%, Analysis 66% |
| 3 | MITRE ATT&CK port attribution | Results §4.4 | ✓ done | DST_PORT_GROUP top-5 in 89% Recon; MITRE tags |
| 5 | Node-state SHAP (behavioral context) | Results §4.5 | ✓ done | φ_N = 13–61% of total \|φ\| |
| 8 | Temporal faithfulness loop | Methods §3.3 | ✓ done | 0/405 violations; 99.2% leakage without filter |
| 10 | Global profiles as proxy ground truth | Evaluation §5.3 | ✓ done | Backdoor ρ=0.73; mean ρ≈0.37 vs literature |

### Deferred

| # | Argument | Reason | Target |
|---|---|---|---|
| 4 | Class-conditional background | Needs explainer re-run | Paper 3 |
| 6 | Novel-node attribution | Near-zero on UNSW | Paper 3 |
| 7 | Lateral movement node arc | Needs LSTM multi-window | Paper 4 |
| 9 | Cross-granularity complementarity | φ_T null on UNSW | Paper 3 |

---

## File structure

```
explore/arguments/
├── AGENT.md
├── graph_arguments_plan.md
├── arg1_semantic_grouping.py
├── arg2_absence_driven.py
├── arg3_mitre_port.py
├── arg5_node_state.py
├── arg8_temporal_faithfulness.py
└── arg10_global_profiles.py

outputs/figures/arguments/          ← runtime-generated, gitignored
├── arg1_semantic_grouping.{pdf,png,txt}
├── arg2_absence_driven.{pdf,png,txt}
├── arg3_mitre_port.{pdf,png,txt}
├── arg5_node_state.{pdf,png,txt}
├── arg8_temporal_faithfulness.{pdf,png,txt}
└── arg10_global_profiles.{pdf,png,txt}
```

```
Dependency flow:
AGENT.md → arg1 → arg2 → arg3 → arg5 → arg10

Data sources:
  artifacts/feature_groups.json       → arg1, arg10
  outputs/explanations/<class>/*.json → arg1, arg2, arg3, arg5, arg10
  outputs/metrics/fidelity.csv        → arg2, arg3
  outputs/metrics/summary.json        → arg2
  graphs/test.bin                     → arg5
```

---

## Codebase conventions (verified against existing scripts)

**ROOT calculation** — scripts sit at `explore/arguments/arg*.py`, three levels from repo root:
```python
ROOT = Path(__file__).resolve().parent.parent.parent
```

**Explanation JSON naming** — Phase 6 writes `<global_eid>.json`; after fix_labels a `_fixed.json` variant may exist. Use `glob("*.json")` on each class directory.

**JSON fields confirmed** (from `scripts/06_explain.py` `_result_to_dict`):
- `edge_id`, `true_label`, `predicted_label`, `predicted_proba`
- `feature_group_names` (list[str], K=48)
- `feature_group_shap` (list[float], K=48)
- `node_ids` (list[int]), `node_shap` (list[float], parallel to node_ids)
- `src_novelty_shap` (float), `dst_novelty_shap` (float)

**fidelity.csv columns** (from `scripts/08_metrics.py`):
`class_name`, `edge_id`, `true_label`, `predicted_label`, `p_full`, `p_masked`, `p_kept`,
`fidelity_plus`, `fidelity_minus`, `top_k_groups` (pipe-separated), `runtime_s`

**feature_groups.json** at `artifacts/feature_groups.json`:
`{"feature_names": [...], "groups": {name: {"indices": [...], "type": str}, ...}, "d_e": int, "K": int}`

**Class display names:** Benign, Generic, Exploits, Fuzzers, DoS, Recon, Analysis, Backdoor, Shellcode, Worms

**Plot conventions:**
```python
import matplotlib
matplotlib.use("Agg")
LABEL_FS = 9
_TEAL  = "#2a9d8f"   # presence-driven bars
_CORAL = "#e76f51"   # absence-driven bars
_AMBER = "#e9c46a"   # weak correlation / temporal
_GREEN = "#2ecc71"   # strong correlation (arg10)
_GRAY  = "#888888"
```
No `fig.suptitle(...)`. Save PDF + PNG + TXT. Extend standard TXT with three extra sections.

---

## Extended argument card `.txt` format

Every `.txt` must include these seven sections:

```
WHAT
----
<one-sentence claim>

KEY FINDINGS
------------
<bullet numeric results>

PAPER FRAMING
-------------
<how it connects to the paper narrative>

CAPTION
-------
<figure caption text>

REVIEWER CHALLENGE
------------------
<hardest expected critique>

COUNTER-ARGUMENT
----------------
<factual response citing figure results>

PAPER SECTION PLACEMENT
-----------------------
<which section/claim in Paper 2 this supports>
```

---

## Implementation

### arg1_semantic_grouping.py — "48 Groups vs 218 Dims"

**Claim:** Semantic grouping to K=48 groups reduces coalition space from 2^218 to 2^48 (×10^51). Attributions are concentrated: top-N groups cover 80% of total |φ|.

**Figure: 2 panels, figsize=(13, 5.5)**

*Left* — Horizontal bar: per-class mean number of groups needed to reach 80% of total |φ| (concentration score). Sorted ascending. Color teal/coral by sign of mean Fidelity+ (cross-reference fidelity.csv). Annotate "top-1 = GROUP_NAME" per class.

*Right* — Horizontal bar: per-category mean |φ| across all flows and classes. Distinct color per category. Text box: "2^48 vs 2^218 (×10^51 reduction)".

**Feature taxonomy:**
```python
CATEGORIES = {
    "Volumetric":    ["IN_BYTES","OUT_BYTES","IN_PKTS","OUT_PKTS",
                      "SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES",
                      "SRC_TO_DST_AVG_THROUGHPUT","DST_TO_SRC_AVG_THROUGHPUT",
                      "RETRANSMITTED_IN_BYTES","RETRANSMITTED_IN_PKTS",
                      "RETRANSMITTED_OUT_BYTES","RETRANSMITTED_OUT_PKTS"],
    "Temporal_IAT":  ["FLOW_DURATION_MILLISECONDS","DURATION_IN","DURATION_OUT",
                      "SRC_TO_DST_IAT_MIN","SRC_TO_DST_IAT_MAX","SRC_TO_DST_IAT_AVG",
                      "SRC_TO_DST_IAT_STDDEV","DST_TO_SRC_IAT_MIN","DST_TO_SRC_IAT_MAX",
                      "DST_TO_SRC_IAT_AVG","DST_TO_SRC_IAT_STDDEV"],
    "Port_Protocol": ["DST_PORT_GROUP","SRC_PORT_IS_EPHEMERAL","PROTOCOL","L7_PROTO",
                      "DNS_QUERY_TYPE","DNS_QUERY_ID","DNS_TTL_ANSWER","FTP_COMMAND_RET_CODE"],
    "TCP_State":     ["TCP_FLAGS","CLIENT_TCP_FLAGS","SERVER_TCP_FLAGS",
                      "TCP_WIN_MAX_IN","TCP_WIN_MAX_OUT","MIN_TTL","MAX_TTL"],
    "Pkt_Size":      ["LONGEST_FLOW_PKT","SHORTEST_FLOW_PKT","MIN_IP_PKT_LEN",
                      "NUM_PKTS_UP_TO_128_BYTES","NUM_PKTS_128_TO_256_BYTES",
                      "NUM_PKTS_256_TO_512_BYTES","NUM_PKTS_512_TO_1024_BYTES",
                      "NUM_PKTS_1024_TO_1514_BYTES"],
    "ICMP":          ["ICMP_TYPE","ICMP_IPV4_TYPE"],
}
```

**Data loading:**
```python
for cls_dir in sorted((ROOT / "outputs/explanations").iterdir()):
    for jf in cls_dir.glob("*.json"):   # catches <eid>.json and <eid>_fixed.json
        rec = json.loads(jf.read_text())
        phi = np.array(rec["feature_group_shap"])
        names = rec["feature_group_names"]
```

**Key .txt claims:** Shapley axioms preserved with groups as atomic players; 2^48 ≈ 2.8×10^14 vs 2^218 ≈ 4.2×10^65.

---

### arg2_absence_driven.py — "Presence vs Absence Classes"

**Claim:** Four classes are characterised by what they lack. Negative Fidelity+ is semantically correct, not an explainer failure.

**Figure: 2 panels, figsize=(13, 5.5)**

*Left* — Horizontal bar: per-class mean Fidelity+ ± 1 std error, sorted. Teal if mean > 0, coral if ≤ 0. Vertical dashed line at 0. Annotate "N% flows absence-driven".

*Right* — 100% stacked horizontal bar: presence fraction (teal) vs absence fraction (coral). Dashed line at 50%. Annotate the 4 absence-dominant classes with ★.

**Algorithm:**
```python
df = pd.read_csv(ROOT / "outputs/metrics/fidelity.csv")
stats = df.groupby("class_name")["fidelity_plus"].agg(["mean","sem","count"])
absence_pct = df.groupby("class_name")["fidelity_plus"].apply(
    lambda x: (x <= 0).mean() * 100
)
```

**Key .txt claims:** Absence-dominant classes (>50% flows): Analysis, Fuzzers, Recon, Backdoor.

---

### arg3_mitre_port.py — "MITRE-Grounded Port Attribution"

**Claim:** DST_PORT_GROUP SHAP maps alert output to MITRE ATT&CK techniques.

**Figure: 2 panels, figsize=(14, 6)**

*Left* — Heatmap: classes (rows) × 8 port-group cols, cell = mean |φ| scaled 0–1 per row. MITRE tags in column labels. ★ marks dominant port group per class.

*Right* — Horizontal bar: per-class frequency where DST_PORT_GROUP appears in top_k_groups.
Parse: `"DST_PORT_GROUP" in row["top_k_groups"].split("|")`

**MITRE mapping:**
```python
MITRE_MAP = {
    "HTTP":       ("T1071.001", "Web C2"),
    "DNS":        ("T1071.004", "DNS C2"),
    "SSH":        ("T1021.004", "Remote Access"),
    "FTP":        ("T1048.003", "Exfiltration"),
    "RDP":        ("T1021.001", "Lateral Movement"),
    "SNMP":       ("T1046",     "Discovery"),
    "NTP":        ("T1498.002", "DDoS Amplif."),
    "BitTorrent": ("T1571",     "C2 Non-Std"),
}
```

---

### arg5_node_state.py — "Node-State SHAP and Temporal Context"

**Claim:** φ_N is non-trivial (13–61% of total |φ|); GNN computation subgraph encodes state flat SHAP cannot see.

**Figure: 2 panels, figsize=(13, 5.5)**

*Left* — Scatter: per-flow φ_N vs hour-of-day. Color by top-5 classes by count. Alpha=0.3. Per-hour mean line. Controlled-lab caveat annotation if no diurnal pattern.

*Right* — Violin: per-class φ_N distribution. Sort by mean descending. ◆ mean markers. Coral fill.

**φ_N per flow:**
```python
phi_N = np.sum(np.abs(rec["node_shap"])) + abs(rec["src_novelty_shap"]) + abs(rec["dst_novelty_shap"])
```

**Timestamp extraction:**
```python
import dgl
from datetime import datetime, timezone
gs, _ = dgl.load_graphs(str(ROOT / "graphs/test.bin"))
g = gs[0]
eid_to_hour = {
    int(eid): datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).hour
    for eid, ts in zip(g.edata[dgl.EID].numpy(), g.edata["timestamp"].numpy())
}
```

---

### arg10_global_profiles.py — "Global Profiles as Proxy Ground Truth"

**Claim:** SHAP-GSD per-class rankings match independently derived literature profiles (Moustafa & Slay 2015). Spearman ρ > 0.7 for most classes.

**Figure: 2 panels, figsize=(14, 6)**

*Left* — Heatmap: classes (rows) × top-12 feature groups (cols), cell = mean |φ| rank. Colormap: `YlOrRd_r`.

*Right* — Horizontal bar: per-class Spearman ρ. Green if ρ ≥ 0.7, amber 0.4–0.7, coral < 0.4. Dashed line at 0.7.

**Literature profiles:**
```python
LITERATURE_TOP_GROUPS = {
    "Analysis":  ["SRC_PORT_IS_EPHEMERAL","IN_BYTES","TCP_FLAGS","MIN_TTL","NUM_PKTS_UP_TO_128_BYTES"],
    "Backdoor":  ["L7_PROTO","DST_PORT_GROUP","MIN_IP_PKT_LEN","FLOW_DURATION_MILLISECONDS","IN_PKTS"],
    "Benign":    ["IN_BYTES","SRC_TO_DST_IAT_AVG","FLOW_DURATION_MILLISECONDS","TCP_FLAGS","DST_TO_SRC_AVG_THROUGHPUT"],
    "DoS":       ["IN_PKTS","IN_BYTES","SRC_TO_DST_SECOND_BYTES","TCP_FLAGS","FLOW_DURATION_MILLISECONDS"],
    "Exploits":  ["DST_PORT_GROUP","L7_PROTO","IN_BYTES","TCP_FLAGS","MIN_IP_PKT_LEN"],
    "Fuzzers":   ["IN_PKTS","SHORTEST_FLOW_PKT","L7_PROTO","DST_PORT_GROUP","NUM_PKTS_UP_TO_128_BYTES"],
    "Generic":   ["DST_PORT_GROUP","L7_PROTO","DNS_QUERY_TYPE","IN_PKTS","TCP_FLAGS"],
    "Recon":     ["MIN_TTL","MAX_TTL","TCP_FLAGS","IN_PKTS","FLOW_DURATION_MILLISECONDS"],
    "Shellcode": ["NUM_PKTS_UP_TO_128_BYTES","IN_PKTS","TCP_FLAGS","SHORTEST_FLOW_PKT","DST_PORT_GROUP"],
    "Worms":     ["ICMP_TYPE","IN_PKTS","TCP_FLAGS","FLOW_DURATION_MILLISECONDS","NUM_PKTS_UP_TO_128_BYTES"],
}
```

---

## Implementation order

```
AGENT.md → arg1 → arg2 → arg3 → arg5 → arg10
```

---

## Verification checklist

After each script:
1. `python3 explore/arguments/arg<N>_*.py` exits 0
2. `ls outputs/figures/arguments/arg<N>_*.{pdf,png,txt}` shows 3 files
3. `.txt` contains all seven sections incl. REVIEWER CHALLENGE, COUNTER-ARGUMENT, PAPER SECTION PLACEMENT
4. `grep -n "suptitle" explore/arguments/arg<N>_*.py` returns nothing
5. Key numeric claims match expected ranges:
   - Arg1: coalition reduction ≈ ×10^51; 2^48 vs 2^218
   - Arg2: 4 absence-dominant classes; absence fraction > 50%
   - Arg3: DST_PORT_GROUP in top-k for ≥ 6 classes
   - Arg5: φ_N range 13–61%
   - Arg10: ρ > 0.7 for ≥ 6 classes (expected)

---

## Relationship to existing figures (no duplication)

| Existing figure | How new argument extends it |
|---|---|
| `fidelity_pa_bars` | Arg2 adds class-level binary split + absence fraction stacked bar |
| `feature_group_heatmap` | Arg10 adds Spearman ρ vs literature + coherence claim |
| `shap_beeswarm_<class>` | Arg1 adds taxonomy view + coalition space reduction |
| `attribution_decomp` | Arg5 uses φ_N differently: time-of-day scatter + violin |
| `method_comparison_fg` | Arg3 complements with semantic MITRE content |
