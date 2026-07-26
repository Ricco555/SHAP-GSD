# SHAP-GSD Result Regeneration — Writer Instructions (2026-07-26)

**Status: pre-peer-review.** This document exists because a data-pipeline bug was
found and fixed, and every downstream figure/table touching the **temporal-neighborhood
SHAP granularity (φ_T)** has been regenerated. The corrected numbers **change the
paper's temporal-signal narrative**, not just the values. Read this in full before
editing any figure, table, or discussion text.

The corresponding-author/PI is handling the publisher/journal notification personally,
given the paper has not yet entered peer review. This note is for the article writer:
what changed, where, and what needs rewriting.

---

## 1. What was actually wrong

Three independent bugs were found and fixed, all in code paths that had been
silently producing near-empty or crashing-but-never-triggered results:

1. **`src/model/temporal_sampler.py` — EID composition bug (the root cause).**
   DGL's `edge_subgraph` → `sample_neighbors` → `to_block` chain does not compose
   `block.edata[dgl.EID]` back to the true position in the split graph across more
   than one hop — `to_block` unconditionally overwrites it with a small,
   frontier-local `arange(0, 1, 2, ...)`. Every downstream reader that indexed the
   split graph directly with `block.edata[dgl.EID]` (the temporal-neighborhood SHAP
   explainer and four figure/report scripts) was silently looking up the **wrong
   edges** — typically the split's earliest-timestamped edges, which almost always
   fail the 60-second temporal window and get filtered out. **This bug was present
   in the exact code (`v2.1.0`, tag `a024a6d`) that produced every number currently
   in the paper** — confirmed via `git diff v2.1.0 main -- src/model/temporal_sampler.py`
   (zero diff).
2. **`scripts/06_explain.py` — `num_classes` never propagated to `cfg["model"]`.**
   A dormant bug unrelated to the EID issue; `_load_model()` computed it locally but
   never wrote it back to the shared config dict, so any explanation rerun after the
   codebase's hardcoded-value cleanup would crash before producing anything. Fixed by
   propagating it the same way the sibling scripts (`04_train.py`, `05_evaluate.py`,
   `13_ablations.py`) already do.
3. **`scripts/09_w_ablation.py` Step 2 — tuple-unpacking bug.** `TemporalNeighborhoodSHAP.explain()`
   returns `(results_list, f_baseline, f_logit)`, but Step 2 assigned the whole
   3-tuple to `temp_results` and indexed into it as if it were the results list.
   This was dead code before tonight — Step 1's flow-selection filter almost never
   passed under the old (fabricated) gap data, so Step 2 essentially never ran. It
   started running for the first time once the EID fix made Step 1 find real
   in-window flows, which is when this crashed and was fixed.

None of these bugs touch model training, `macro-F1`, feature-group SHAP (φ_F), or
node-novelty SHAP (φ_N) — verified directly by reading `src/model/sage_model.py`:
the GNN forward pass never reads per-neighbor edge features off `block.edata`, only
graph structure plus the target edge's own features. **Classification performance
and the two non-temporal SHAP granularities are unaffected.**

---

## 2. The headline number

| | Before (bug) | After (fixed) |
|---|---|---|
| Flows with ≥1 non-empty temporal neighbor (n=1,764 test flows) | **22 (1.2%)** | **1,053 (59.7%)** |
| `attribution_decomp.py`: mean φ_T fraction of total \|φ\| across classes | reported as **< 0.1%** | **1.56%** (range 0.24-4.93% per class) |
| `w_ablation`: % flows with in-window neighbor at W=60s | ~1% | **60.3%** |
| `w_ablation`: % flows with in-window neighbor at W=3600s | not meaningfully different | **99.8%** (58.03% of edges) |
| `arg8_temporal_faithfulness`: total verified neighbor assignments | 405 | **4,446** |

**The old ~1% figure was not a property of the dataset — it was the bug.** The
"22/1764" number even appears hardcoded into `scripts/11_efficiency.py`'s output
JSON (`"n_flows_with_temporal_nbrs": 22`) from the original run; that script's
efficiency numbers themselves are fine (they only ever covered φ_F/φ_N), but the note
text around it needs updating.

---

## 3. Important: numbers refreshed, but auto-generated *narrative text* did not update itself

Several figure-generation scripts write a hardcoded prose interpretation into their
`.txt` companion file, and that prose was written under the (wrong) assumption that
coverage would always be near-zero. **The regenerated numbers are correct, but the
scripts' own written conclusions now directly contradict those numbers** — this is
the single most important thing for the writer to catch, because it is not visually
obvious from a glance at the figure:

- **`outputs/figures/explore/w_ablation.txt`** still says: *"This is a correct null
  result, not an explainer failure... there simply are no temporally-proximate
  neighbors on NF-UNSW-NB15-v3 at deployment timescales."* — while reporting in the
  same document that **60.3% of flows have an in-window neighbor at W=60s**, rising to
  99.8%/58.03%-of-edges at W=3600s. A majority is not a null result. The "motivates
  Paper 3" framing (IoT datasets will show the *contrast* case) needs to be dropped
  or substantially rewritten — UNSW-NB15 itself already shows non-trivial temporal
  coverage.
- **`outputs/figures/graph/attribution_decomp.txt`** still says: *"phi_T genuine null
  result: mean across classes = 1.5637% < 0.1%"* — the sentence's own stated number
  (1.5637%) is not less than 0.1%; the threshold text is stale template language, not
  a recomputed check. The figure's textbox/caption claiming "φ_T < 0.1% across all
  attack classes... a dataset property rather than a model failure" needs to be
  removed or rewritten against the real per-class range (0.24-4.93%, see table above).
- **`outputs/figures/arguments/arg8_temporal_faithfulness.txt`**'s "REVIEWER CHALLENGE
  / COUNTER-ARGUMENT" section argues from the premise "φ_T < 0.1% on UNSW-NB15" —
  that premise is no longer true (now ~1.56% mean, higher per-class). The rest of
  arg8's causal-ordering verification claims (0 violations, leakage-rate-without-filter
  math) are unaffected and remain valid — only the φ_T-is-negligible framing needs
  updating.
- **`outputs/figures/graph/topology_panel.txt`** and the 3 of 9 per-class topology
  captions with `n_nbr=0` (Backdoor, Generic, Worms — the specific hardcoded example
  EIDs for those classes still happen to have zero neighbors post-fix) auto-generate
  text like *"0 temporal neighbours: temporal window contained no prior flows"* and
  the panel-level summary says *"confirms the temporal null result extends"* —
  that generalization from 3 individually-zero examples to a dataset-wide "null
  result" is no longer supportable given 59.7% overall coverage. The other 6 classes'
  example flows now show real, non-trivial neighbor counts (11-41 neighbors) with
  correctly-drawn temporal edges.
- **`explore/case_studies.py`**'s case-selection criteria (documented in its
  docstring as specific in-window-neighbor counts per class) were very likely chosen
  using the old corrupted counts. All 9 case-study figures regenerated successfully
  and now show real (mostly non-zero) neighbor/gap data, but **whether these are still
  the most illustrative examples for the paper is a curation question the writer
  should revisit**, not something the code can decide.

---

## 4. What needs writer attention, concretely

1. **Rewrite the temporal-neighborhood-granularity narrative wherever it currently
   asserts UNSW-NB15 has "no temporal signal."** It has a real, if modest, temporal
   signal (φ_T ≈ 1.56% mean, up to ~5% per class) — smaller than φ_F (majority) and
   φ_N (12-60%), but not absent, and not "structurally" absent at any tested window.
2. **Re-examine the "Paper 3 motivation" framing** that used UNSW's temporal null
   result as the contrast case against IoT datasets' expected dense temporal
   clustering — this specific rationale is no longer supported by UNSW's own data.
3. **Regenerate/re-check all figure captions** listed in §3 before they go in the
   paper or a rebuttal — the underlying plots and data are correct, only the
   auto-written prose text needs a human rewrite.
4. **Re-verify case-study selection** in `explore/case_studies.py` against the new
   per-flow neighbor counts if the paper cites specific example flows by class.
5. Table 2 / macro-F1 / weighted-F1 / Fidelity+/−/Stability are **unaffected** —
   confirmed these metrics never read `block.edata[dgl.EID]` or the temporal SHAP
   output. No change needed there.

---

## 5. Regenerated artifacts (all under `outputs/`, this run)

- `outputs/explanations/<class>/<eid>.json` + `summary.csv` — 1,764 flows, `06_explain.py`
- `outputs/figures/case_studies/` — 9 case-study figures + quality report, `07_visualize.py` / `explore/case_studies.py`
- `outputs/figures/feature_groups/` — `07_visualize.py`
- `outputs/w_ablation/gap_stats.{json,txt}`, `outputs/w_ablation/summary.txt`, `W<s>_temporal.csv` — `09_w_ablation.py`
- `outputs/metrics/efficiency.json` — `11_efficiency.py` (unaffected numbers, refreshed for consistency)
- `outputs/metrics/runtime_by_layer.json` — `06_explain.py`
- `outputs/figures/explore/w_ablation.{pdf,png,txt}` — `explore/w_ablation_figure.py`
- `outputs/figures/arguments/arg8_temporal_faithfulness.{pdf,png,txt}` — `explore/arguments/arg8_temporal_faithfulness.py`
- `outputs/figures/graph/attribution_decomp.{pdf,png,txt}` — `explore/graph/attribution_decomp.py`
- `outputs/figures/graph/topology_<Class>_<EID>.{pdf,png,txt}` (9 classes) — `explore/graph/topology_panel.py`

**Removed:** `outputs/figures/graph/topology_panel.{pdf,png,txt}` — a stale, hand-written
leftover from before the script was refactored (commit `9fb9ed2`) to produce one figure
per class instead of a single combined 2-flow panel. It described a figure structure
that no longer exists and used outdated EIDs/values; deleted rather than "corrected"
since there is no current combined-figure equivalent to update it against.

A full backup of the pre-fix `outputs/` directory is at
`backups/outputs_pre_eid_fix_20260725_233151.tar.gz` for before/after comparison.

---

## 6. Code changes backing this regeneration

- `src/model/temporal_sampler.py` — EID composition fix (root cause), plus a new
  regression test `tests/test_temporal_sampler.py::test_block_eid_resolves_to_true_split_graph_edge`
  locking in the contract that was previously untested.
- `scripts/06_explain.py` — `num_classes` propagation fix.
- `scripts/09_w_ablation.py` — tuple-unpacking fix in Step 2.

Full test suite: 120/120 passing after all fixes.
