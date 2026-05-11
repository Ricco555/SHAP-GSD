# SHAP-GSD Explore Folder — Agent Instructions

This folder contains exploratory scripts that are **not committed to git** (`.gitignore` excludes `explore/`).
All figures and reasoning files are written to `outputs/figures/explore/`.

---

## Purpose

The `explore/` folder is a scratchpad for analysing experimental results during the research process.
Scripts here are not part of the reproducible pipeline; they read from committed `outputs/` directories
and produce charts and textual reasoning for the paper authors.

---

## Conventions (follow these in every new exploration script)

### 1. Output location
```python
OUT_DIR = ROOT / "outputs" / "figures" / "explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

### 2. No supertitles in images
Never use `fig.suptitle(...)`. The supertitle text becomes the figure caption in the
paper and must live in the `.txt` reasoning file under `SUGGESTED FIGURE CAPTION`, not
embedded in the image. Panel titles (axes-level `set_title`) are fine.

### 2. File naming
Every exploration produces at minimum:
- `<figure_stem>.pdf`  — print-quality
- `<figure_stem>.png`  — web/draft
- `<figure_stem>.txt`  — reasoning and interpretation (see §3)

Use a short, descriptive `<figure_stem>`, e.g. `class_accuracy_fidelity`, `w_ablation`, `method_comparison_fg`.

### 3. Reasoning .txt file (mandatory)
After saving figures, write a companion `.txt` file to `OUT_DIR` with the **same stem**.
Every `.txt` file must end with a `SUGGESTED FIGURE CAPTION` section — this text is
copied verbatim into the paper as the figure caption.

```python
reasoning = """
Figure reasoning — <figure_stem>
=================================

WHAT THE FIGURE SHOWS
----------------------
<One paragraph.>

KEY FINDINGS
------------
<Bullet-point observations for the paper author.>

PAPER FRAMING
-------------
<Draft paragraph for the paper body.>

SUGGESTED FIGURE CAPTION
-------------------------
<Caption text, written as it will appear under the figure in the paper.
 No supertitle text appears in the image — this is the sole source of
 the figure caption.>
"""
txt_path = OUT_DIR / f"{STEM}.txt"
txt_path.write_text(reasoning)
```

If a script produces multiple figures, write one `.txt` per figure (matching stems).

### 4. Script header
Every script begins with a module-level docstring listing:
- What it shows
- What panels it contains
- What files it reads (source paths)
- What files it outputs

### 5. Data sources (never use val/test raw data directly)
Approved sources for exploration scripts:
- `outputs/metrics/fidelity.csv`         — per-flow SHAP-GSD fidelity
- `outputs/metrics/stability.csv`        — per-flow phi std across seeds
- `outputs/metrics/summary.json`         — per-class and overall metrics
- `outputs/explanations/<class>/*.json`  — raw SHAP-GSD explanation records
- `outputs/baselines/*.csv`              — baseline explainer results
- `outputs/baselines/summary.json`       — aggregated baseline metrics
- `outputs/w_ablation/*.csv`             — W-sensitivity ablation results
- `outputs/w_ablation/gap_stats.json`    — temporal gap statistics

### 6. Style constants (use these for visual consistency)
```python
LABEL_FS  = 9          # all axis labels, tick labels, legend text
_PRESENCE = "#2a9d8f"  # teal  — presence-driven (φ ≥ 0)
_ABSENCE  = "#e76f51"  # coral — absence-driven  (φ < 0)
_SHAP_GSD = "#2a9d8f"  # teal  — SHAP-GSD in comparison plots
_GNN_EXP  = "#e9c46a"  # amber — GNNExplainer
_GRAY     = "#888888"  # reference lines, annotations
```

---

## Existing exploration scripts and their outputs

| Script | Figures | What it shows |
|---|---|---|
| `table2_figure.py` | `table2_fidelity_stability.{pdf,png,txt}` | Fidelity+ diverging bar + Stability colour-banded bar |
| `class_analysis.py` | `class_accuracy_fidelity.{pdf,png}`, `feature_group_heatmap.{pdf,png}` | Per-class test accuracy, Fidelity+ split by correctness, feature group heatmap |
| `w_ablation_figure.py` | `w_ablation.{pdf,png,txt}` | In-window coverage and temporal SHAP magnitude vs W |
| `fidelity_distributions.py` | `fidelity_violins.{pdf,png}`, `fidelity_pa_bars.{pdf,png}` | Per-class Fidelity+ violin plots, presence/absence × correctness breakdown |
| `method_comparison_figure.py` | `method_comparison_fg.{pdf,png,txt}` | SHAP-GSD vs GNNExplainer[F] per-class and Δ Fidelity+ |
| `shap_scatter.py` | `shap_beeswarm_<class>.{pdf,png}`, `shap_absence_drivers.{pdf,png}`, `shap_beeswarm_grid.{pdf,png}`, `shap_scatter.txt` | Per-class SHAP beeswarm (top-15 groups, colour=feature value); 2×2 absence-driver scatter; all-class 2×5 grid |

**Subfolder `graph/`** — outputs to `outputs/figures/graph/` (see `graph/AGENT.md`):

| Script | Figures | What it shows |
|---|---|---|
| `graph/attribution_decomp.py` | `attribution_decomp.{pdf,png,txt}` | φ_F/φ_T/φ_N stacked fraction bars + φ_T violin per class |
| `graph/w_sensitivity_full.py` | `w_sensitivity_annotated.{pdf,png,txt}` | Dual-axis W-sensitivity: % coverage + mean φ_T, saturation annotation |
| `graph/topology_panel.py` | `topology_panel.{pdf,png,txt}` | SHAP-GSD vs Flat KernelSHAP topology panels for Recon + Backdoor |

---

## Key analytical findings so far (for the paper)

### Feature-group Fidelity+ (Table 2 coalition space)
- **SHAP-GSD overall**: 0.107 ± 0.337 | **GNNExplainer overall**: 0.156
- SHAP-GSD wins on Backdoor (+0.014), DoS (+0.075), Exploits (+0.045)
- GNNExplainer leads on Recon (Δ=−0.241), Shellcode (Δ=−0.146), Fuzzers (Δ=−0.129)
- The ~30% Fidelity+ gap is the cost of semantic grouping (48 groups vs 218 raw dims)

### Absence-driven classes
Four classes have mean Fidelity+ < 0 (absence-driven):
Analysis (−0.044), Backdoor (+0.008 overall but 70% neg flows), Fuzzers (−0.009), Recon (−0.063)

Dominant negative-phi groups:
- **Analysis**: SRC_PORT_IS_EPHEMERAL (−0.047, 74% neg), NUM_PKTS_256-1024B
- **Backdoor**: L7_PROTO (−0.171, 62% neg), DNS groups (Backdoor doesn't use DNS)
- **Fuzzers**: L7_PROTO (−0.099), DST_PORT_GROUP — absence of normal protocol/port structure
- **Recon**: MIN_TTL/MAX_TTL (−0.150/−0.128) — absence of normal TTL variation; TCP flags absent

Validated absence-driver scatter features for `shap_scatter.py`:
| Class    | Feature              | corr(feat,φ) | lo-val % φ<0 | Notes |
|----------|----------------------|-------------|--------------|-------|
| Analysis | SRC_PORT_IS_EPHEMERAL | +0.668       | 86%          | Binary 0/1; low=server port → negative φ |
| Backdoor | MIN_IP_PKT_LEN       | +0.753       | 77%          | L7_PROTO was constant (all=1), degenerate |
| Fuzzers  | SHORTEST_FLOW_PKT    | +0.752       | 69%          | DST_PORT_GROUP was constant (all=1); MIN_IP_PKT_LEN corr=-0.515 (wrong direction) |
| Recon    | MIN_TTL              | +0.957       | 100%         | 28 low-TTL flows all have φ<0; 172 high-TTL flows mostly positive |

Key rule: for absence-driver scatter, pick features with **corr > 0** (low value → negative φ) AND **variance > 0**. Use `lo_mask = x < x.mean()` for annotation ("% of low-value flows with φ < 0").

### Classification accuracy (from explanation sample n≤200)
- Backdoor: 3% (6/200) — confusion among attack classes (not with Benign)
  - Confused as: Fuzzers(56), Exploits(44), Generic(38), DoS(30), Recon(21)
  - The 6 correct flows all use DST_PORT_GROUP + L7_PROTO with Fidelity+ 0.32–0.79
- Benign: 97.5% — near-zero feature attribution (median Fid+=0.000)
- Generic: 74% — strongly presence-driven (98% of correct flows have positive Fid+)

### Stability
- Overall: 0.0188 (mean phi std across 3 coalition seeds)
- Most stable: Analysis (0.001) — consistently absence-driven
- Least stable: Benign (0.043) — heterogeneous traffic, diffuse attribution

### Temporal null result (W-ablation)
- Median inter-flow gap: 5,168 s
- At W=60s: 1.25% of flows have in-window neighbors; mean top φ_T = 0.010
- At W=3600s: 39.4% of flows; mean top φ_T = 0.032 — still near-zero
- SHAP-GSD runtime: mean=1.287 s/flow, median=1.402 s/flow (measured after full re-run; excludes 51 WSL2 clock-jitter flows with rt<0; research_plan placeholder "0.1 / 3.9" is outdated)

---

## Next exploration ideas

- [ ] Confusion matrix annotated with SHAP-GSD dominant feature groups per predicted class
- [ ] Node novelty (src_novelty_shap / dst_novelty_shap) distribution per class
- [ ] Per-flow runtime distribution (from updated JSONs; note: 51/1764 flows have negative WSL2 clock jitter — filter rt >= 0; mean ~1.24s, median ~1.40s)
- [ ] Baseline comparison figure for node-coalition methods (PGExplainer/GNNShap/GraphSVX/EdgeSHAPer) — wait for EdgeSHAPer to complete
- [x] SHAP-GSD runtime: mean=1.287s, median=1.402s (now in outputs/metrics/summary.json)
