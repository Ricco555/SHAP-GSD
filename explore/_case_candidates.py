"""Single source of truth for the curated case-study / topology-panel flows.

`explore/case_studies.py` and `explore/graph/topology_panel.py` both render
one flow per attack class. Before this module existed they sampled
independently and disagreed on four classes (Analysis, Generic,
Reconnaissance, Worms) -- a reviewer comparing the case-study figure to the
topology figure for the "same" class would see two different flows. See
`final/review01/coder_instructions_figure_determinism.md` S4.

Each entry is (class_name, edge_id) only. Neighbour counts, correctness,
p(y_hat) and node-SHAP magnitudes are NOT stored here -- every consumer
reads those live from the run's own `outputs/explanations/<class>/<eid>.json`
at render time, so a rerun against a different run directory produces
genuinely different narrative text instead of a stale literal repeating
itself (S0 "definition of done").

Backdoor is deliberately absent from CANDIDATES: 0 of 200 explained Backdoor
flows in this run are correctly classified (max p(Backdoor) = 0.0520, zero
flows predicted Backdoor anywhere), so no attribution case study can satisfy
the "correctly classified" selection rule (S3). `BACKDOOR_TOPOLOGY_EID` is
its structural-only entry: the 2-hop topology panel does not depend on
prediction correctness, so Backdoor gets that figure and no others.
"""

CANDIDATES: list[tuple[str, int]] = [
    ("Fuzzers", 1931492),
    ("Shellcode", 1929451),
    ("Exploits", 1945754),
    ("Analysis", 2099604),        # flagship: feature-group, temporal, AND node-novelty all nonzero and reinforcing
    ("DoS", 1942014),
    ("Reconnaissance", 1929497),  # 30 in-window temporal neighbours, p=0.994
    ("Generic", 2250911),
    ("Worms", 2310720),
]

BACKDOOR_TOPOLOGY_EID = 2174974
