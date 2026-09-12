# explore/evaluation/

Cross-dataset evaluation-methodology and comparison library: fidelity aggregation,
missing-class-aware macro statistics, global coherence against a proxy ground
truth with actual statistical testing, and stability beyond the pipeline's
intra-run measure.

Its subject throughout is **SHAP-GSD's own metrics compared across datasets** —
does explanation quality generalise beyond one network environment? Comparing
SHAP-GSD against other explanation methods is out of scope here: the SHAP-GSD
paper already does that once, on UNSW, and it is not repeated per-dataset
(specs/64, DESCOPED 2026-08-26).

Specified by `specs/64_functional_spec_evaluation_methodology_2026-08-26.md`
(Part I functional, Part II implementation — Part II governs).

Everything here reads `runs/**` **read-only** and writes only into
`outputs/figures/evaluation/`.

## Output location

```python
from explore.evaluation._discover import DEFAULT_OUT_DIR   # outputs/figures/evaluation/
```

Hardcoded, **not** `SHAP_GSD_CONFIG`-resolved — every experiment config sets a
non-empty `run.dir`, so a config-driven path would file a cross-dataset artifact
inside one dataset's `runs/` subtree. See `AGENT.md`.

## Choosing datasets

`SHAP_GSD_CONFIG` does **not** apply to this folder. Name the dataset per
invocation:

| flag | meaning |
|---|---|
| `--dataset NAME` | Analyse this dataset, by short name (repeatable). Case-insensitive; a unique prefix works (`--dataset unsw` → `unsw_nb15`). |
| `--all-datasets` | Analyse every dataset that resolves to exactly one canonical run. Mutually exclusive with `--dataset`. |
| `--run-dir NAME=PATH` | Pin a dataset to an explicit run directory. The only way to analyse a variant/sweep/archived run deliberately, and the required disambiguator when resolution reports an ambiguity. |
| `--config NAME=PATH` | Alternative to `--run-dir`: resolve through an experiment YAML's `run.dir`. |
| `--label NAME=TEXT` | Paper-facing display label (`--label unsw=UNSW`); lands in every output's `dataset` column. |
| `--runs-root PATH` | Directory scanned for candidate runs (default `runs`). |
| `--out-dir PATH` | Override the output directory (used by tests). |
| `--alpha FLOAT` | Significance threshold on **corrected** p-values (default 0.05). |
| `--top-k INT` | k for top-k rank/overlap agreement (default 5, matching `scripts/08_metrics.py`). |
| `--log-level STR` | Default `INFO`. |

```bash
python explore/evaluation/eval01_long_metrics.py --all-datasets \
    --label unsw_nb15=UNSW --label bot_iot=BoT-IoT --label ton_iot=ToN-IoT
```

## Why the dataset list is resolved, not enumerated

`runs/` holds real dataset runs, an incomplete stub, a differently-labelled
experiment and a dozen balancer-sweep cells side by side — and **each of those
non-datasets has its own git-tracked `configs/experiment_*.yaml` with a
`run.dir`**. So neither a config glob nor a directory listing yields "the
datasets": both return ~17 entries, most of which are not datasets. This is the
live layout (probed 2026-08-26):

| `runs/` entry | what it is | 4 required artifacts |
|---|---|:--:|
| `nf_unsw_nb15_v3` | canonical UNSW run | 4/4 |
| `nf_bot_iot_v3` | canonical BoT-IoT run | 4/4 |
| `nf_ton_iot_v3` | canonical ToN-IoT run | 4/4 |
| `nf_cicids2018_v3` | canonical CICIDS2018 run | 4/4 |
| `nf_unsw_nb15_v3_stub` | incomplete stub — **not** real UNSW data | 0/4 |
| `nf_ton_iot_v3_binary` | binary-label experiment | 1/4 |
| `nf_ton_iot_v3_r1_*` (12 dirs) | balancer/split sweep cells | 1/4 each |
| `archive/` | tarballs + a pre-rebuild copy | never a candidate |

The resolution rule (`_discover.py`, specs/64 §13) has three layers:

1. **name → base run id.** A directory name splits as `nf_<...>_v<N>` + optional
   variant suffix, matching the run-id grammar `scripts/run_dataset.py::derive_run_id`
   already establishes. The base with `nf_` and `_v<N>` stripped is the *dataset
   key* (`nf_unsw_nb15_v3` → `unsw_nb15`); the caller's token matches it
   case-insensitively, exact first, then by unique prefix.
2. **base → canonical run, by completeness.** Among the directories sharing a
   base, only those carrying **all four** of
   `artifacts/evaluation/metrics.json`, `artifacts/label_map.json`,
   `outputs/metrics/summary.json` and `outputs/explanations/summary.csv` survive.
   All four conjuncts are load-bearing: a 1-of-4 or 2-of-4 predicate lets the
   sweep cells through, and `outputs/explanations/summary.csv` is the single
   strongest discriminator.
3. **ambiguity is a hard error.** More than one survivor raises, listing every
   candidate and telling you to pass `--run-dir`. There is deliberately **no**
   tiebreak — not "prefer bare", not "prefer newest", not "prefer longest
   suffix". "Prefer bare" would have resolved UNSW to an empty stub as recently
   as 2026-08-26, when the canonical run lived at `nf_unsw_nb15_v3_r3_s2` and the
   bare path held the stub. Failing loudly costs one CLI flag; a silent wrong
   pick costs a paper number computed from the wrong run.

Zero survivors likewise raises, naming every directory examined and the first
artifact each was missing — that is the normal state for a dataset whose pipeline
has not finished, and `--all-datasets` simply omits it. No dataset-specific code
exists anywhere.

## Run order

`eval01` → `eval02` → `eval05` → `eval04` → `eval07b` → `eval08`.
`eval04` depends on `eval05`'s CSV; `eval08` reads every other module's CSV and
never recomputes a number.

## Modules

| Script | What it produces |
|---|---|
| `_discover.py` | dataset name → canonical run resolution, per-run path accessors, shared CLI surface |
| `_load.py` | per-dataset metrics ingestion + schema normalisation (the single ingestion point) |
| `_stats.py` | BH-FDR, Wilcoxon/Friedman wrappers, Spearman ρ / Kendall τ / Jaccard |
| `eval01_long_metrics.py` | `eval_long_metrics.csv` (one row per dataset × class × coalition space × metric) + `eval_long_metrics_coverage.md` |
| `eval02_macro_denominators.py` | `macro_stats_by_denominator.csv` — macro stats under all three denominators |
| `eval04_global_coherence.py` | `global_coherence_rank_correlation.csv`, `global_coherence_structural.csv`, per-dataset summary figures, `source3_consistency_cells.md` |
| `eval05_per_class_explanations.py` | `per_class_explanation_summary.csv` |
| `eval07b_stability_windows.py` | `stability_temporal_windows.csv`, per-dataset stability comparison figures |
| `eval08_cross_dataset_report.py` | `cross_dataset_comparison_report.md` |

## What can and cannot be built from `runs/` alone (specs/64 §17)

- **A — implementable now, from `runs/` artifacts alone:** everything in the table
  above except the two items below.
- **B — gated on an owner-authored input:** `eval04`'s §4.4a rank-correlation half
  needs `proxy_gt_feature_expectation_mapping.csv`, which is **manually authored
  by the researcher and never generated** (Part I §8). It ships git-tracked at
  `explore/evaluation/data/` (the default `eval04` reads from) so it is present
  in a fresh clone. Until it exists, `eval04` logs a WARNING naming the path,
  writes the structural half only, and exits 0. `eval04 --source3-locked`
  additionally needs the researcher's explicit non-circularity assertion that
  proxy-GT sources 1/2/4 are frozen.
- **C — spec-only, no code:** inter-seed stability (Part I §4.7a) needs 5
  independently-seeded training runs per dataset plus a compute-budget sign-off;
  the Nemenyi post-hoc needs a new dependency (`scikit-posthocs`/`statsmodels`)
  this repo does not carry, so a significant Friedman omnibus is reported with
  `posthoc = "not_computed_no_dependency"` rather than dropped. `eval08` renders
  both as explicit "not available" stubs so their absence is visible in the
  deliverable.

## Two terms that must never be conflated

**φ_T fidelity** (a SHAP-GSD coalition space — mask a flow's temporal neighbours)
and **temporal stability** (a PROXEVAL stability dimension — does an explanation
hold as traffic drifts) share only the word. Identifiers here are always
`fidelity_temporal_*` or `stability_temporal_*`; a bare `temporal_*` name is a
defect. See `AGENT.md`.

## Data-source notes worth knowing before reading a number

- `outputs/metrics/table2*.txt` is **never parsed** — it is a fixed-width human
  rendering with an em-dash sentinel for undefined cells. The `summary*.json` /
  `fidelity*.csv` siblings carry the same numbers plus diagnostics.
- Intra-run stability exists for the **feature** coalition space only; temporal
  and novelty rows are emitted as `NaN` with a note, never omitted.
- `summary_temporal.json` carries JSON `null` for undefined fidelity cells; every
  numeric read goes through `_load.as_float`, which maps it to `NaN`.
- `feature_store/train/` can be genuinely absent for a run (it is, for BoT-IoT, as
  of 2026-08-26), which makes the closed-set denominator not computable for that
  dataset. That is reported explicitly, never silently replaced by another
  denominator.
- `outputs/baselines/**` is **never read** by anything in this package. Those are
  the SHAP-GSD paper's own baseline-comparison numbers, not a PROXEVAL input.
