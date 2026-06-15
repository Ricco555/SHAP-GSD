# explore/

Exploratory scripts for analysing pipeline outputs and producing publication figures.
All scripts read from the active dataset run's `outputs/` directory, resolved via
`SHAP_GSD_CONFIG`.

## Selecting a dataset run

```bash
# Default — reads from outputs/ (UNSW reference run)
python explore/table2_figure.py

# Alternate run
SHAP_GSD_CONFIG=configs/experiment_nf_ton_iot_v3.yaml python explore/table2_figure.py
```

## Output location

Every script writes under `<outputs_dir>/figures/explore/`:

```python
from explore._paths import paths
_P = paths()
OUT_DIR = _P["figures"] / "explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

Subfolder scripts use `"arguments"` or `"graph"` in place of `"explore"`.

## Mandatory conventions

- `matplotlib.use("Agg")` at the top of every script.
- No `fig.suptitle(...)`. Panel titles (`ax.set_title`) are fine.
- Every figure produces three files with a shared stem:
  - `<stem>.pdf` — print quality (`dpi=300`, `bbox_inches="tight"`)
  - `<stem>.png` — web/draft (`dpi=150`, `bbox_inches="tight"`)
  - `<stem>.txt` — reasoning and figure caption (see below)
- No `plt.show()`; explicit `plt.close(fig)` after each save.

## Reasoning `.txt` file format

```
Figure reasoning — <stem>
=========================

WHAT THE FIGURE SHOWS
----------------------
<One paragraph.>

KEY FINDINGS
------------
<Bullet-point observations.>

PAPER FRAMING
-------------
<Draft paragraph for the paper body.>

SUGGESTED FIGURE CAPTION
-------------------------
<Caption text as it will appear in the paper.>
```

For `explore/arguments/` scripts, also include:
`REVIEWER CHALLENGE / COUNTER-ARGUMENT / PAPER SECTION PLACEMENT`

## Style constants

```python
LABEL_FS  = 9           # axis labels, tick labels, legend
_PRESENCE = "#2a9d8f"   # teal  — presence-driven (φ ≥ 0)
_ABSENCE  = "#e76f51"   # coral — absence-driven  (φ < 0)
_SHAP_GSD = "#2a9d8f"   # teal  — SHAP-GSD in comparison plots
_GNN_EXP  = "#e9c46a"   # amber — GNNExplainer
_GRAY     = "#888888"   # reference lines, annotations
```

## Script classification

| Script | Status |
|--------|--------|
| `table2_figure.py` | Dataset-agnostic |
| `class_analysis.py` | Dataset-agnostic |
| `fidelity_distributions.py` | Dataset-agnostic |
| `method_comparison_figure.py` | Dataset-agnostic |
| `w_ablation_figure.py` | Dataset-agnostic |
| `shap_scatter.py` | Dataset-agnostic |
| `node_shap_convergence.py` | Dataset-agnostic |
| `case_studies.py` | UNSW-specific — hardcoded exemplar EIDs |

See [`arguments/README.md`](arguments/README.md) and [`graph/README.md`](graph/README.md)
for subfolder scripts.

## Script outputs

| Script | Figures |
|--------|---------|
| `table2_figure.py` | `table2_fidelity_stability.{pdf,png,txt}` |
| `class_analysis.py` | `class_accuracy_fidelity.{pdf,png}`, `feature_group_heatmap.{pdf,png}` |
| `w_ablation_figure.py` | `w_ablation.{pdf,png,txt}` |
| `fidelity_distributions.py` | `fidelity_violins.{pdf,png}`, `fidelity_pa_bars.{pdf,png}` |
| `method_comparison_figure.py` | `method_comparison_fg.{pdf,png,txt}` |
| `shap_scatter.py` | `shap_beeswarm_<class>.{pdf,png}`, `shap_absence_drivers.{pdf,png}`, `shap_beeswarm_grid.{pdf,png}`, `shap_scatter.txt` |
