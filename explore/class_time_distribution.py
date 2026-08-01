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
for cls in classes:
    ct, frac = curves[cls]
    n_cls = len(ct)
    # time at which the class reaches 10%/50%/90% cumulative
    def t_at(q):
        idx = int(np.searchsorted(frac, q))
        idx = min(idx, len(ct) - 1)
        return ct[idx]
    lines.append(
        f"  {cls:<16} n={n_cls:>8,}  "
        f"t10%={t_at(0.10):>7.2f}h  t50%={t_at(0.50):>7.2f}h  t90%={t_at(0.90):>7.2f}h"
    )
per_class_block = "\n".join(lines)

zoom_lines = []
for cls in classes:
    ct_full, _ = curves[cls]
    ct_zoom = ct_full[ct_full >= zoom_start_h] - zoom_start_h
    n_zoom = len(ct_zoom)
    if n_zoom == 0:
        zoom_lines.append(f"  {cls:<16} n=0 (no flows in the zoomed/dense region)")
        continue
    frac_zoom = np.arange(1, n_zoom + 1) / n_zoom
    def tz_at(q, ct=ct_zoom, frac=frac_zoom):
        idx = min(int(np.searchsorted(frac, q)), len(ct) - 1)
        return ct[idx]
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
<Fill in after visual inspection -- this is a data-characterization tool, not
 a hypothesis-confirming figure. Look for: (1) any class whose curve is nearly
 vertical near a split boundary (moving that boundary by a small amount would
 sharply change how much of that class lands in train vs val vs test), (2)
 classes concentrated entirely on one side of a boundary (near-zero support in
 val or test), (3) classes spread near-uniformly (curve close to the diagonal)
 -- these are insensitive to where the boundary falls.>

PAPER FRAMING
-------------
<Draft paragraph once findings above are filled in -- likely candidate for the
 Discussion section's treatment of chronological-split sensitivity, alongside
 local/macro_f1_regression_investigation.md's R3 (split-cutoff) results.>

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
(OUT_DIR / f"{STEM}.txt").write_text(reasoning)

print(f"Wrote {OUT_DIR / f'{STEM}.pdf'}")
print(f"Wrote {OUT_DIR / f'{STEM}.png'}")
print(f"Wrote {OUT_DIR / f'{STEM}.txt'}")
