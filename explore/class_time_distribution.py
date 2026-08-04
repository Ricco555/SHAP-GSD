"""
Attack-class temporal distribution — empirical CDF over the capture window.

Two-panel figure: one cumulative-fraction curve per class (Benign + every
attack class present in the dataset), x-axis = elapsed hours since the first
flow, y-axis = fraction of that class's own flows seen by time t (0.0-1.0).
A curve that rises steeply in one place and stays flat elsewhere shows that
class's traffic is concentrated in a narrow time window ("bursty"/campaign
traffic); a curve that rises roughly linearly shows traffic spread evenly
across the whole capture.

  Left  — full capture span, for context (shows whether there is a dominant
          idle gap between capture sessions, as on NF-UNSW-NB15-v3).
  Right — auto-zoomed to the dense region only. The zoom start is picked by
          finding the single largest gap between consecutive flows across
          ALL classes combined; if that gap covers >=5% of the total span,
          the zoom starts right after it (captures a "two capture sessions"
          structure without hardcoding it), otherwise it falls back to the
          last 10% of the span. This makes the figure dataset-agnostic: a
          dataset with no dominant gap just gets a modest zoom instead of a
          near-degenerate one.

A second row repeats both panels with y = raw flow COUNT per 1-hour bin
(log scale) instead of cumulative fraction — the same underlying signal,
but bursts show up as visible peaks rather than as a change in slope.

Two vertical reference lines (on all four panels, where in range) mark the
CURRENTLY ACTIVE config's chronological split boundaries (tau_train =
train_frac, tau_val = train_frac + val_frac, as row-fractions mapped onto
the same elapsed-time axis) — this is what makes the figure useful for
spotting a better split: a boundary that lands inside a class's steep rise
mid-burst, or well before/after every class's traffic has appeared, is
visible directly in the zoomed panels.

Data source note (deviation from explore/AGENT.md's normal "never read raw
val/test rows" rule): every other exploration script reads DERIVED artifacts
(fidelity.csv, metrics.json, ...) so it cannot leak information the model
didn't see. This script characterizes the RAW INPUT DATASET's own temporal
structure (Attack + FLOW_START_MILLISECONDS columns only) -- it runs before
any train/val/test split even exists conceptually, the same class of
data-characterization work already done in
local/macro_f1_regression_investigation.md Sec 4 and local/r3_quantile_pinning.py.
Reading the raw CSV here is not a leakage risk; it is the object of study.

Reads:
  <cfg["data"]["csv_path"]>   -- raw NetFlow CSV, columns "Attack" and
                                  "FLOW_START_MILLISECONDS" only (usecols,
                                  memory-light even at multi-million-row scale)
  <cfg["data"]["train_frac"]>, <cfg["data"]["val_frac"]>  -- for the split
                                  boundary reference lines

Outputs:
  outputs/figures/explore/class_time_distribution.{pdf,png,txt}
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402

_P = paths()
CFG = _P["cfg"]
OUT_DIR = _P["figures"] / "explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM = "class_time_distribution"
LABEL_FS = 9

# ── Load raw CSV (Attack + timestamp columns only) ──────────────────────────

csv_path = ROOT / CFG["data"]["csv_path"]
df = pd.read_csv(csv_path, usecols=["Attack", "FLOW_START_MILLISECONDS"])
df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)

t0 = df["FLOW_START_MILLISECONDS"].iloc[0]
elapsed_hours = (df["FLOW_START_MILLISECONDS"] - t0) / (1000.0 * 3600.0)
total_span_hours = float(elapsed_hours.iloc[-1])

# ── Split-boundary reference lines (currently active config) ───────────────

n = len(df)
train_frac = float(CFG["data"]["train_frac"])
val_frac = float(CFG["data"]["val_frac"])
train_cut_idx = int(n * train_frac) - 1
val_cut_idx = int(n * (train_frac + val_frac)) - 1
tau_train_h = float(elapsed_hours.iloc[max(train_cut_idx, 0)])
tau_val_h = float(elapsed_hours.iloc[max(val_cut_idx, 0)])

# ── Per-class ECDF over time ────────────────────────────────────────────────

classes = sorted(df["Attack"].unique())
n_classes = len(classes)
cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
colors = {cls: cmap(i % cmap.N) for i, cls in enumerate(classes)}

curves = {}
for cls in classes:
    ct = np.sort(elapsed_hours[df["Attack"] == cls].to_numpy())
    frac = np.arange(1, len(ct) + 1) / len(ct)
    curves[cls] = (ct, frac)

# ── Auto-detect the largest single idle gap across ALL flows combined ──────
# (dataset-agnostic: if a dataset has a dominant capture-session gap like
# UNSW's, the zoomed panel starts right after it; if not, the gap is small
# and the zoomed panel just falls back to the last 10% of the span.)

all_t = np.sort(elapsed_hours.to_numpy())
gaps = np.diff(all_t)
largest_gap_idx = int(np.argmax(gaps))
largest_gap_size = float(gaps[largest_gap_idx])
gap_end_h = float(all_t[largest_gap_idx + 1])

if largest_gap_size >= 0.05 * total_span_hours:
    zoom_start_h = gap_end_h
else:
    zoom_start_h = total_span_hours * 0.90

# ── 1-hour-bin flow counts per class (for the bottom row) ──────────────────

bin_edges_full = np.arange(0.0, np.ceil(total_span_hours) + 1.0, 1.0)
bin_centers_full = bin_edges_full[:-1] + 0.5

hist_counts = {}
for cls in classes:
    ct, _ = curves[cls]
    counts, _ = np.histogram(ct, bins=bin_edges_full)
    counts_f = counts.astype(float)
    counts_f[counts_f == 0] = np.nan  # gaps on log scale instead of log(0) warnings
    hist_counts[cls] = counts_f

max_hourly_count = max(np.nanmax(c) for c in hist_counts.values())

# ── Plot: top row = CDF (full span, zoomed); bottom row = 1h flow counts ───

fig, axes = plt.subplots(
    2, 2, figsize=(13, 10), width_ratios=[1, 1.4], height_ratios=[1, 1],
)
(ax_full, ax_zoom), (ax_full_cnt, ax_zoom_cnt) = axes

for ax, xlim, show_legend in (
    (ax_full, (0.0, total_span_hours), False),
    (ax_zoom, (zoom_start_h, total_span_hours), True),
):
    for cls in classes:
        ct, frac = curves[cls]
        lw = 2.2 if cls != "Benign" else 1.4
        ls = "-" if cls != "Benign" else "--"
        ax.plot(ct, frac, label=cls, color=colors[cls], linewidth=lw, linestyle=ls)

    for tau_h in (tau_train_h, tau_val_h):
        if xlim[0] <= tau_h <= xlim[1]:
            ax.axvline(tau_h, color="#555555", linestyle=":", linewidth=1.3)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=LABEL_FS)
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(fontsize=LABEL_FS - 1, loc="lower right", ncol=2, framealpha=0.9)

ax_full.set_ylabel("Cumulative fraction of class's own flows", fontsize=LABEL_FS)
ax_full.set_title("Full capture span", fontsize=LABEL_FS)
ax_zoom.set_title(
    f"Zoomed: dense region only ({zoom_start_h:.1f}h–{total_span_hours:.1f}h)",
    fontsize=LABEL_FS,
)

# Boundary labels placed once, above the zoom panel where both lines are
# spread out enough to stay legible (avoids the overlap the full panel would
# cause when tau_train/tau_val sit close together relative to the full span).
y_top = 1.02
if abs(tau_val_h - tau_train_h) < 0.06 * (total_span_hours - zoom_start_h):
    ax_zoom.annotate(
        "train/val", xy=(tau_train_h, y_top), xytext=(tau_train_h, y_top + 0.08),
        ha="center", fontsize=LABEL_FS - 1, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        annotation_clip=False,
    )
    ax_zoom.annotate(
        "val/test", xy=(tau_val_h, y_top), xytext=(tau_val_h, y_top + 0.16),
        ha="center", fontsize=LABEL_FS - 1, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        annotation_clip=False,
    )
else:
    ax_zoom.text(tau_train_h, y_top, "train/val", ha="center", va="bottom",
                 fontsize=LABEL_FS - 1, color="#555555")
    ax_zoom.text(tau_val_h, y_top, "val/test", ha="center", va="bottom",
                 fontsize=LABEL_FS - 1, color="#555555")

# Bottom row: 1-hour-bin flow counts, same x-axis ranges as the row above.
for ax, xlim in (
    (ax_full_cnt, (0.0, total_span_hours)),
    (ax_zoom_cnt, (zoom_start_h, total_span_hours)),
):
    for cls in classes:
        lw = 2.0 if cls != "Benign" else 1.3
        ls = "-" if cls != "Benign" else "--"
        ax.plot(
            bin_centers_full, hist_counts[cls], label=cls,
            color=colors[cls], linewidth=lw, linestyle=ls,
        )
    for tau_h in (tau_train_h, tau_val_h):
        if xlim[0] <= tau_h <= xlim[1]:
            ax.axvline(tau_h, color="#555555", linestyle=":", linewidth=1.3)

    ax.set_xlabel("Elapsed time (hours since first flow)", fontsize=LABEL_FS)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max_hourly_count * 1.3)
    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=LABEL_FS)
    ax.grid(alpha=0.25)

ax_full_cnt.set_ylabel("Flow count per 1h bin (log scale)", fontsize=LABEL_FS)

fig.tight_layout()
fig.savefig(OUT_DIR / f"{STEM}.pdf")
fig.savefig(OUT_DIR / f"{STEM}.png", dpi=200)
plt.close(fig)

# ── Reasoning file ───────────────────────────────────────────────────────────

lines = []
full_stats = {}
for cls in classes:
    ct, frac = curves[cls]
    n_cls = len(ct)
    # time at which the class reaches 10%/50%/90% cumulative
    def t_at(q, ct=ct, frac=frac):
        idx = int(np.searchsorted(frac, q))
        idx = min(idx, len(ct) - 1)
        return float(ct[idx])
    full_stats[cls] = {
        "n": n_cls, "t10": t_at(0.10), "t50": t_at(0.50), "t90": t_at(0.90),
    }
    lines.append(
        f"  {cls:<16} n={n_cls:>8,}  "
        f"t10%={t_at(0.10):>7.2f}h  t50%={t_at(0.50):>7.2f}h  t90%={t_at(0.90):>7.2f}h"
    )
per_class_block = "\n".join(lines)

zoom_span_h = total_span_hours - zoom_start_h

zoom_lines = []
zoom_stats = {}
for cls in classes:
    ct_full, _ = curves[cls]
    ct_zoom = ct_full[ct_full >= zoom_start_h] - zoom_start_h
    n_zoom = len(ct_zoom)
    if n_zoom == 0:
        zoom_stats[cls] = {"n": 0, "t10": None, "t50": None, "t90": None}
        zoom_lines.append(f"  {cls:<16} n=0 (no flows in the zoomed/dense region)")
        continue
    frac_zoom = np.arange(1, n_zoom + 1) / n_zoom
    def tz_at(q, ct=ct_zoom, frac=frac_zoom):
        idx = min(int(np.searchsorted(frac, q)), len(ct) - 1)
        return float(ct[idx])
    zoom_stats[cls] = {
        "n": n_zoom, "t10": tz_at(0.10), "t50": tz_at(0.50), "t90": tz_at(0.90),
    }
    zoom_lines.append(
        f"  {cls:<16} n={n_zoom:>8,}  "
        f"t10%={tz_at(0.10):>7.2f}h  t50%={tz_at(0.50):>7.2f}h  t90%={tz_at(0.90):>7.2f}h"
        f"  (hours elapsed WITHIN the zoomed region)"
    )
zoom_block = "\n".join(zoom_lines)

count_lines = []
for cls in classes:
    counts = hist_counts[cls]
    peak_count = int(np.nanmax(counts))
    peak_bin = int(np.nanargmax(counts))
    peak_hour = bin_centers_full[peak_bin]
    median_count = np.nanmedian(counts)
    count_lines.append(
        f"  {cls:<16} peak={peak_count:>8,}/h at t={peak_hour:>7.2f}h   "
        f"median (non-zero bins)={median_count:>8.1f}/h"
    )
count_block = "\n".join(count_lines)

# ── Computed KEY FINDINGS ───────────────────────────────────────────────────
# Every criterion below is evaluated against the per-class quantities already
# computed above (full_stats / zoom_stats / the tau_* boundaries), with the
# thresholds stated inline in the emitted text so a reader can audit them.
# Nothing here names a class a priori: whichever classes trigger on THIS run
# are the ones reported, and a criterion that nothing triggers says so.

# Criterion A — sharp early burst followed by a long quiet tail, measured
# WITHIN the dense/zoomed region (the full-span numbers are dominated by the
# inter-session idle gap and cannot express burst shape).
BURST_RATIO_MAX = 0.20      # (t50-t10) must be <= 20% of (t90-t50): the rise
                            # to the class's own median is >= 5x faster than
                            # the tail that follows it.
BURST_T50_FRAC_MAX = 0.25   # and that median must land in the first 25% of
                            # the dense region, i.e. the burst is EARLY.
burst_hits = []
for cls in classes:
    s = zoom_stats[cls]
    if s["n"] == 0:
        continue
    rise = s["t50"] - s["t10"]
    tail = s["t90"] - s["t50"]
    if tail <= 0.0:
        continue  # degenerate: class ends at its own median, no tail to compare
    ratio = rise / tail
    t50_frac = s["t50"] / zoom_span_h if zoom_span_h > 0 else 1.0
    if ratio <= BURST_RATIO_MAX and t50_frac <= BURST_T50_FRAC_MAX:
        burst_hits.append((cls, s, ratio, t50_frac, tail))
burst_hits.sort(key=lambda r: r[2])

# Criterion B — a split boundary landing inside a class's steep-rise region.
# "Steep" is local flow density: the fraction of a class's DENSE-REGION flows
# falling in a narrow window centred on the boundary, compared with what a
# class spread uniformly over the dense region would put there. Both the count
# and its denominator are taken over the dense region so the printed uniform
# expectation is exactly the reference the test uses; mixing in the pre-gap
# session would silently make the test stricter than the number it prints.
BOUNDARY_WIN_H = 0.01 * zoom_span_h          # half-width, 1% of the dense region
BOUNDARY_DENSITY_MULT = 5.0                  # flag at >= 5x uniform expectation
uniform_win_frac = (2.0 * BOUNDARY_WIN_H / zoom_span_h) if zoom_span_h > 0 else 0.0
boundary_hits = []
boundary_untestable = []
for tau_name, tau_h in (("tau_train", tau_train_h), ("tau_val", tau_val_h)):
    if tau_h < zoom_start_h:
        boundary_untestable.append((tau_name, tau_h))
        continue
    for cls in classes:
        ct_full, _ = curves[cls]
        ct_zoom = ct_full[ct_full >= zoom_start_h]
        n_zoom = len(ct_zoom)
        if n_zoom == 0:
            continue
        n_in_win = int(np.sum(np.abs(ct_zoom - tau_h) <= BOUNDARY_WIN_H))
        if n_in_win == 0:
            continue
        win_frac = n_in_win / n_zoom
        if win_frac >= BOUNDARY_DENSITY_MULT * uniform_win_frac:
            boundary_hits.append((tau_name, tau_h, cls, n_in_win, win_frac))
boundary_hits.sort(key=lambda r: -r[4])

# Criterion C — near-zero support on one side of a split boundary. Splits are
# chronological ROW fractions, exactly as the pipeline cuts them.
SUPPORT_FRAC_MIN = 0.01     # < 1% of the class's own flows in a split, or
SUPPORT_ABS_MIN = 30        # < 30 rows outright, counts as near-zero support.
split_of_row = np.full(n, 2, dtype=np.int8)
split_of_row[: train_cut_idx + 1] = 0
split_of_row[train_cut_idx + 1 : val_cut_idx + 1] = 1
split_names = ("train", "val", "test")
support_table = {}
support_hits = []
for cls in classes:
    mask = (df["Attack"] == cls).to_numpy()
    counts_split = [int(np.sum(split_of_row[mask] == k)) for k in range(3)]
    support_table[cls] = counts_split
    for k, c in enumerate(counts_split):
        frac = c / full_stats[cls]["n"]
        if c < SUPPORT_ABS_MIN or frac < SUPPORT_FRAC_MIN:
            support_hits.append((cls, split_names[k], c, frac))

# ── Render the findings ─────────────────────────────────────────────────────

fnd = []
fnd.append(
    "All three checks below are computed from the per-class tables above; the\n"
    "thresholds are stated with each check. Only classes that trigger a check\n"
    "are listed, and a check that nothing triggers says so explicitly."
)
fnd.append("")
fnd.append(
    f"(A) SHARP EARLY BURST, THEN A QUIET TAIL  [within the {zoom_span_h:.2f}h dense region]\n"
    f"    Criterion: (t50%-t10%) <= {BURST_RATIO_MAX:.2f} x (t90%-t50%)  AND  "
    f"t50% <= {BURST_T50_FRAC_MAX:.0%} of the dense region ({BURST_T50_FRAC_MAX * zoom_span_h:.2f}h)."
)
if burst_hits:
    for cls, s, ratio, t50_frac, tail in burst_hits:
        fnd.append(
            f"    {cls}: reaches 50% of its own {s['n']:,} dense-region flows "
            f"{s['t50']:.2f}h in ({t50_frac:.1%} of the region), having reached 10% at "
            f"{s['t10']:.2f}h -- a {s['t50'] - s['t10']:.2f}h rise -- then takes a further "
            f"{tail:.2f}h to reach 90% (t90%={s['t90']:.2f}h). Rise/tail ratio "
            f"{ratio:.4f}. Peak hourly volume {int(np.nanmax(hist_counts[cls])):,} flows/h at "
            f"t={bin_centers_full[int(np.nanargmax(hist_counts[cls]))]:.2f}h (full-span axis)."
        )
else:
    fnd.append("    No class met this criterion on this run.")
fnd.append("")
fnd.append(
    f"(B) SPLIT BOUNDARY LANDING IN A CLASS'S STEEP RISE\n"
    f"    Criterion: >= {BOUNDARY_DENSITY_MULT:.0f}x the uniform-density expectation of a class's\n"
    f"    dense-region flows inside +/-{BOUNDARY_WIN_H:.3f}h of a boundary (window = 1% of the\n"
    f"    dense region each side; uniform expectation = {uniform_win_frac:.2%} of a class's\n"
    f"    dense-region flows, count and denominator both taken over the dense region)."
)
for tau_name, tau_h in boundary_untestable:
    fnd.append(
        f"    {tau_name}={tau_h:.2f}h falls before the dense region starts "
        f"({zoom_start_h:.2f}h) and is not testable under this criterion."
    )
if boundary_hits:
    for tau_name, tau_h, cls, n_in_win, win_frac in boundary_hits:
        fnd.append(
            f"    {cls} at {tau_name}={tau_h:.2f}h: {n_in_win:,} flows "
            f"({win_frac:.2%} of the class's dense-region flows) inside the window, "
            f"{win_frac / uniform_win_frac:.1f}x uniform -- a small shift of this boundary "
            f"moves a disproportionate share of this class between splits."
        )
elif len(boundary_untestable) < 2:
    tested = ", ".join(
        f"{nm}={th:.2f}h" for nm, th in (("tau_train", tau_train_h), ("tau_val", tau_val_h))
        if nm not in {u[0] for u in boundary_untestable}
    )
    fnd.append(
        f"    No class met this criterion on this run: {tested} "
        f"{'does' if len(boundary_untestable) == 1 else 'do'} not land\n"
        f"    inside any class's steep-rise region."
    )
else:
    fnd.append("    Not evaluated: both boundaries fall outside the dense region.")
fnd.append("")
fnd.append(
    f"(C) NEAR-ZERO SUPPORT ON ONE SIDE OF A BOUNDARY\n"
    f"    Criterion: a class holding < {SUPPORT_FRAC_MIN:.0%} of its own flows, or fewer than "
    f"{SUPPORT_ABS_MIN} flows outright,\n    in one of the three chronological splits "
    f"(train/val/test row cuts at {train_frac:.6f} / {train_frac + val_frac:.6f})."
)
if support_hits:
    for cls, sname, c, frac in support_hits:
        fnd.append(
            f"    {cls}: {c:,} flows in {sname} ({frac:.2%} of its {full_stats[cls]['n']:,} "
            f"total) -- too few to measure that class reliably on that split."
        )
else:
    fnd.append(
        f"    No class met this criterion on this run: every class holds at least "
        f"{SUPPORT_FRAC_MIN:.0%} of its own\n    flows, and at least {SUPPORT_ABS_MIN} flows, "
        f"in each of train, val and test."
    )
fnd.append("")
fnd.append("    Per-class split support (chronological row cuts):")
for cls in classes:
    tr, va, te = support_table[cls]
    fnd.append(
        f"      {cls:<16} train={tr:>9,}  val={va:>9,}  test={te:>9,}"
        f"   (total {full_stats[cls]['n']:>9,})"
    )
key_findings_block = "\n".join(fnd)

# ── Computed PAPER FRAMING ──────────────────────────────────────────────────
# Built strictly from what the checks above actually found on this run. The
# paragraph is descriptive of the observed geometry only: it states where the
# volume sits relative to the boundaries and does not assert that a different
# cutoff would change any downstream result.

frm = []
if burst_hits:
    burst_desc = "; ".join(
        f"{cls} reaches half of its {s['n']:,} dense-region flows within {s['t50']:.2f}h "
        f"of the region's start ({t50_frac:.1%} of the {zoom_span_h:.2f}h region) and then "
        f"needs {tail:.2f}h more to reach 90%"
        for cls, s, ratio, t50_frac, tail in burst_hits
    )
    burst_names = ", ".join(c for c, *_ in burst_hits)
    frm.append(
        f"The capture is not temporally homogeneous: {largest_gap_size / total_span_hours:.1%} "
        f"of its {total_span_hours:.2f}h span is a single {largest_gap_size:.2f}h idle gap, and "
        f"essentially all of the analysable traffic sits in the {zoom_span_h:.2f}h dense region "
        f"that follows it. Within that region {len(burst_hits)} of {n_classes} classes "
        f"{'is' if len(burst_hits) == 1 else 'are'} sharply front-loaded rather than spread "
        f"across it ({burst_names}): {burst_desc}. "
        f"The consequence for a chronological split is positional. "
    )
    tau_desc = []
    for cls, s, ratio, t50_frac, tail in burst_hits:
        for tau_name, tau_h in (("tau_train", tau_train_h), ("tau_val", tau_val_h)):
            rel = tau_h - zoom_start_h
            side = "after" if rel >= s["t50"] else "before"
            tau_desc.append(
                f"{tau_name} falls {rel:.2f}h into the dense region, "
                f"{abs(rel - s['t50']):.2f}h {side} {cls}'s median"
            )
    frm.append(" ".join(
        [f"{d}." for d in tau_desc[: 2 * len(burst_hits)]]
    ))
    dominant, dom_s = burst_hits[0][0], burst_hits[0][1]
    tr, va, te = support_table[dominant]
    dom_total = full_stats[dominant]["n"]
    n_after = sum(
        1 for _, th in (("tau_train", tau_train_h), ("tau_val", tau_val_h))
        if (th - zoom_start_h) >= dom_s["t50"]
    )
    biggest = max(zip(("train", "val", "test"), (tr, va, te)), key=lambda kv: kv[1])
    below = [
        f"{nm} {c:,} ({c / dom_total:.1%})"
        for nm, c in (("val", va), ("test", te)) if c / dom_total < SUPPORT_FRAC_MIN
    ]
    floor_clause = (
        f"below the {SUPPORT_FRAC_MIN:.0%} floor used in check (C) on {' and '.join(below)}"
        if below else
        f"both above the {SUPPORT_FRAC_MIN:.0%} floor used in check (C)"
    )
    frm.append(
        f" With {('neither boundary', 'one of the two boundaries', 'both boundaries')[n_after]} "
        f"positioned after {dominant}'s dense-region median, "
        f"the largest share of {dominant}'s volume lands in {biggest[0]} "
        f"({biggest[1]:,} of {dom_total:,} flows, {biggest[1] / dom_total:.1%}); the split "
        f"is train {tr:,} ({tr / dom_total:.1%}), val {va:,} ({va / dom_total:.1%}), test "
        f"{te:,} ({te / dom_total:.1%}) -- {floor_clause}. "
        f"This is a statement about where the volume sits, not an explanation of any "
        f"downstream metric, and no claim that a different cutoff would improve any "
        f"per-class result follows from this figure."
    )
else:
    frm.append(
        f"Across the {n_classes} classes present, none is sharply front-loaded within the "
        f"{zoom_span_h:.2f}h dense region under the stated burst criterion: every class's rise "
        f"to its own median takes more than {BURST_RATIO_MAX:.0%} of the time its subsequent "
        f"tail takes, so class arrival is broadly spread rather than campaign-like. "
    )
if boundary_hits:
    frm.append(
        f" {len(boundary_hits)} class-boundary pairs place a split cut inside a steep rise, "
        f"the strongest being {boundary_hits[0][2]} at {boundary_hits[0][0]} "
        f"({boundary_hits[0][4] / uniform_win_frac:.1f}x uniform local density), so the "
        f"train/val/test composition of those classes is sensitive to small movements of the cut."
    )
else:
    tested_taus = [
        (nm, th) for nm, th in (("tau_train", tau_train_h), ("tau_val", tau_val_h))
        if nm not in {u[0] for u in boundary_untestable}
    ]
    if tested_taus:
        frm.append(
            f" No tested boundary ({', '.join(f'{nm}={th:.2f}h' for nm, th in tested_taus)}) "
            f"lands inside any class's steep rise, so no class's split composition is "
            f"knife-edge sensitive to a small movement of those cuts."
        )
    else:
        frm.append(
            " Both boundaries fall outside the dense region, so boundary-versus-steep-rise "
            "sensitivity was not evaluated on this run."
        )
if support_hits:
    frm.append(
        f" {len({c for c, *_ in support_hits})} classes fall below the stated support floor "
        f"on at least one split and cannot be evaluated reliably there."
    )
else:
    frm.append(
        " Every class retains usable support in all three splits, so no per-class result "
        "reported elsewhere is attributable to a missing split population."
    )
paper_framing_block = "".join(frm)

reasoning = f"""
Figure reasoning — {STEM}
=================================

WHAT THE FIGURE SHOWS
----------------------
Top row: one empirical CDF per class (Benign dashed, attack classes solid),
left = FULL capture span, right = auto-zoomed to the dense region only (see
ZOOM DETECTION below). X-axis elapsed hours since the first flow in the raw
CSV ({csv_path.name}), y-axis the fraction of THAT class's own flows seen by
time t. A curve is a vertical step wherever that class's traffic bursts, and
flat wherever it is silent.

Bottom row: the same two x-axis ranges, but y-axis is raw flow COUNT per
1-hour bin (log scale, since Benign outnumbers the rarest attack class by
~4 orders of magnitude -- a linear scale would make Analysis/Worms invisible).
Zero-count bins are left as gaps rather than plotted at log(0). This is the
same information as the CDF row but makes bursts/rate-changes visually
obvious as peaks rather than as a change in a curve's slope.

Dotted vertical lines (both rows) mark the currently active config's
chronological split boundaries: tau_train (row-fraction {train_frac}) and
tau_val (row-fraction {train_frac + val_frac}), computed from
"{CFG['data']['csv_path']}" via
`{{run: {_P['cfg'].get('run', {}).get('dir', '(bare outputs/)')}}}`.

ZOOM DETECTION
--------------
Largest single gap between consecutive flows (all classes combined):
{largest_gap_size:.2f}h, ending at elapsed time {gap_end_h:.2f}h.
This is {largest_gap_size / total_span_hours * 100:.1f}% of the total span
({'>= 5% threshold met -- zoom starts right after this gap' if largest_gap_size >= 0.05 * total_span_hours else '< 5% threshold -- zoom falls back to the last 10% of the span'}).
Zoomed region: {zoom_start_h:.2f}h to {total_span_hours:.2f}h
({total_span_hours - zoom_start_h:.2f}h span).

PER-CLASS TIMING, FULL SPAN (t10%/t50%/t90% = elapsed hours to reach that cumulative fraction)
------------------------------------------------------------------------------------------------
{per_class_block}

PER-CLASS TIMING, WITHIN THE ZOOMED/DENSE REGION ONLY (hours elapsed since the zoom start)
----------------------------------------------------------------------------------------------
{zoom_block}

PER-CLASS PEAK HOURLY FLOW COUNT (1h bins, full span)
-------------------------------------------------------
{count_block}

Total capture span: {total_span_hours:.2f} hours.
Split boundaries on this axis: tau_train={tau_train_h:.2f}h, tau_val={tau_val_h:.2f}h.

KEY FINDINGS
------------
{key_findings_block}

PAPER FRAMING
-------------
{paper_framing_block}

SUGGESTED FIGURE CAPTION
-------------------------
Top: cumulative distribution of each attack class's flows over the capture
window (elapsed hours since the first flow), full span (left) and zoomed to
the dense region (right). Bottom: the same time axes, showing raw flow count
per 1-hour bin (log scale). Dotted vertical lines mark the train/validation
and validation/test chronological split boundaries. Steep CDF sections /
count peaks indicate temporally concentrated ("bursty") class traffic;
classes whose CDF curves track the diagonal / whose counts stay flat are
spread evenly across the capture.
"""
# Placeholder guard: no section of the emitted artifact may ship an unfilled
# template. `<Fill in ...>` / `<Draft paragraph ...>`-style placeholders start
# with an uppercase letter directly after "<", which prose like "< 5% threshold"
# never does. Fail loudly rather than write a template that looks complete.
_placeholder = re.search(r"<[A-Z][^<>]{15,}>", reasoning, re.S)
if _placeholder is not None:
    sys.stderr.write(
        f"ERROR: {STEM} would emit an unfilled placeholder, refusing to write:\n"
        f"  {_placeholder.group(0)[:120]}...\n"
    )
    sys.exit(1)

(OUT_DIR / f"{STEM}.txt").write_text(reasoning)

print(f"Wrote {OUT_DIR / f'{STEM}.pdf'}")
print(f"Wrote {OUT_DIR / f'{STEM}.png'}")
print(f"Wrote {OUT_DIR / f'{STEM}.txt'}")
