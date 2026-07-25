# README_PROX.md — Post-submission additions (Paper 3 / PROXEVAL groundwork)

This file tracks everything added to this codebase **after** the SHAP-GSD paper
was submitted (`v2.1.0`, commit `a024a6d`, 2026-06-16) that is *not* part of
that paper — mostly infrastructure and groundwork for a separate, forthcoming
paper (working name **PROXEVAL** — a proxy ground-truth evaluation framework
for GNN-based NIDS explanations, evaluated across multiple NetFlow datasets).

See [`README.md`](README.md) for the SHAP-GSD paper itself — reproduction
instructions, dataset, results. This file is the "everything since then" log,
kept separate so the paper's own README stays accurate to what was actually
submitted and reviewed.

---

## Why this file exists

Between `v2.1.0` and the current version, real new features landed alongside
bug fixes — enough that calling the whole delta "bug fixes only" would be
inaccurate to a reviewer who diffs the repository. This file draws the line
explicitly: infrastructure fixes are covered in `README.md`'s version note;
everything below is new, additive, and belongs to the follow-up paper's
groundwork, not to SHAP-GSD.

## Current status

**Not yet a separate paper submission.** This is pipeline groundwork,
dataset-support work, and diagnostic tooling being built ahead of PROXEVAL's
own experiments. Nothing here has independent results yet.

## What's here so far

### Phase 14 — Gateway-distance diagnostic

A new, read-only pipeline phase (`scripts/14_gateway_distance.py` +
`src/data/gateway_distance.py`) that measures, on each dataset's malicious-only
training subgraph, the reverse-hop distance from each attacked host to the
internal/external role boundary (`d_gw`), and derives `k*` — a topology-derived
justification for the neighborhood-sampling depth, rather than an inherited
architecture default. Runs by default in the pipeline (config-gated,
`topology.gateway_distance.enabled`); see `README.md`'s Phase 14 note for the
mechanics.

**Why this isn't SHAP-GSD's:** it never feeds `model.num_layers`/fanouts back
into training — it's a diagnostic, not an architecture change. On
NF-UNSW-NB15-v3 (the SHAP-GSD paper's only dataset) the measurement is
degenerate (`d_gw ≡ 1` for every victim), which retroactively validates the
existing k=2 depth rather than changing anything the paper reports. It becomes
genuinely informative once run against the additional datasets below.

**Design record:** `specs/05_graph_node_distance.md` (approved, ratified —
local-only, not in this git history) and `specs/07_phase14_gateway_distance_impl.md`.

### Multi-dataset support (Paper 3's four-dataset benchmark)

Config and orchestrator support for three additional NetFlow v3 datasets,
alongside NF-UNSW-NB15-v3:

| Dataset | Config | Status |
|---|---|---|
| NF-CSE-CIC-IDS2018-v3 | `configs/experiment_nf_cicids2018_v3.yaml` | Phase 1–2 validated on HPC |
| NF-ToN-IoT-v3 | `configs/experiment_nf_ton_iot_v3.yaml` | Phase 1–2 validated on HPC |
| NF-BoT-IoT-v3 | `configs/experiment_nf_bot_iot_v3.yaml` | Not yet run |

**A known open item:** Phase 14's exact-diameter computation is a likely
walltime risk on NF-CSE-CIC-IDS2018-v3's node count (~205,801 nodes) — a
component-size cap or approximate fallback is designed but not yet built. See
`specs/05_graph_node_distance.md` §7 risk 8 and
`specs/10_phase14_diameter_optimization.md`. **Do not run the full pipeline
through phase 14 on CICIDS2018 yet** — use `--skip-gateway-distance` until this
lands.

### Infrastructure fixes that enabled the above

These are also mentioned in `README.md`'s version note (they don't change the
SHAP-GSD paper's results), listed here with more detail since they were
necessary to even attempt the larger datasets:

- **Phase 2 OOM fix** — a dead-weight node-state snapshot cache (never read by
  any runtime code path) was disabled by default. Confirmed on HPC:
  NF-CSE-CIC-IDS2018-v3 Phase 2 went from an OOM kill after ~2h wall-clock at
  ~95.8 GiB peak to a clean run in 00:03:55 at ~9.5 GiB peak. Design record:
  `specs/02_graph_model.md` § SNAPSHOTS erratum.
- **Baseline-explainer wrapper portability** — GNNShap/GraphSVX/EdgeSHAPer no
  longer hardcode a developer-machine path; see `README.md`'s Baseline
  explainer comparison section for the corrected install instructions.
- **FeatureStore EID lookup** — O(n) dict replaced with O(log n) binary
  search over the same sorted data; identical results, lower memory at scale.

### Known open items, not yet built

- **Phase 3 (tuner) walltime risk at large-dataset scale** — the temporal
  sampler's per-batch subgraph rebuild is a plausible multi-day-per-epoch
  problem at 20M+ edges across the 108-configuration grid search. A redesign
  (using DGL GraphBolt's native temporal-sampling primitive instead of a
  hand-rolled rebuild) has been spec'd and independently re-verified, but
  **not implemented** — this needs a new differential/reproducibility test
  before implementation is authorized, since it touches the temporal-leakage
  correctness gate directly. See `specs/11_phase3_tuner_sampler_optimization.md`.
- **Phase 14 diameter computation on CICIDS2018** — see above.

### Related research notes (not committed to any implementation)

- `specs/08_topology_ground_truth_paper4.md` — a candidate *future* paper idea
  (deliberately deferred, not PROXEVAL, not authorized scope): cross-referencing
  each NetFlow dataset's documented original-testbed architecture (from its own
  published paper) against what a NetFlow re-export can/can't structurally
  recover.
- `specs/15_temporal_sampling_sota_for_proxeval.md` — a literature-relevance
  check for PROXEVAL specifically (temporal-neighbor-sampling SOTA: TGL, TGN,
  pyg-lib/GraphBolt) — already folded into PROXEVAL's own `research_plan.md`,
  `sources/`, and `paper3/literature_search/` in the separate PROXEVAL project
  folder; kept here too as the SHAP-GSD-side copy of record.

## What is NOT here

Anything that changes the SHAP-GSD paper's model, training procedure, or
reported results. If a future change does touch those, it does not belong in
this file — it would need to go through the paper's own revision process
instead.
