"""
Four-panel case study figure for SHAP-GSD paper.

Panel layout (2×2):
  (a) top-left  — Feature-group SHAP bar chart (top 20 by |φ|)
  (b) top-right — 2-hop neighbourhood topology (role-coloured, IP-labelled)
  (c) bot-left  — Node SHAP bar chart (src/dst novelty + non-target nodes)
  (d) bot-right — Temporal gap distribution (shows WHY temporal SHAP = 0)

Usage::
    fig = make_case_study_figure(explanation, topo, id2ip, class_name, cfg)
    fig.savefig('figure.pdf', bbox_inches='tight', dpi=300)
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ── colour palette (colourblind-safe) ─────────────────────────────────────────
_POS = "#d62728"   # red — positive SHAP
_NEG = "#1f77b4"   # blue — negative SHAP
_ZERO = "#aec7e8"  # light blue — near-zero

# Node roles in the topology panel
_ROLE_COLOURS = {
    "target_src": "#d62728",
    "target_dst": "#ff7f0e",
    "hop1":       "#2ca02c",
    "hop2":       "#9467bd",
}
_ROLE_LABELS = {
    "target_src": "Target src",
    "target_dst": "Target dst",
    "hop1":       "1-hop neighbour",
    "hop2":       "2-hop neighbour",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _shorten_ip(ip: str) -> str:
    """Return last two octets for compact node labels."""
    parts = ip.split(".")
    return ".".join(parts[-2:]) if len(parts) == 4 else ip


def _bar_colours(values: np.ndarray) -> list[str]:
    return [_POS if v >= 0 else _NEG for v in values]


# ── Panel A: feature-group SHAP ───────────────────────────────────────────────

def plot_feature_shap(
    ax: plt.Axes,
    group_names: list[str],
    shap_values: np.ndarray,
    top_n: int = 20,
    class_name: str = "",
) -> None:
    """Horizontal bar chart of the top_n feature groups by |φ|."""
    idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    names  = [group_names[i] for i in idx]
    values = shap_values[idx]
    # Show most important at top
    names  = names[::-1]
    values = values[::-1]

    colours = _bar_colours(values)
    y = np.arange(len(names))
    ax.barh(y, values, color=colours, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("SHAP value φ", fontsize=8)
    ax.set_title(f"(a) Feature-group SHAP — {class_name}", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)

    # Annotation: sum(φ)
    ax.text(0.98, 0.02, f"Σφ = {values.sum():.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="dimgray")

    pos_patch = mpatches.Patch(color=_POS, label="Positive")
    neg_patch = mpatches.Patch(color=_NEG, label="Negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=6, loc="lower right")


# ── Panel B: 2-hop neighbourhood topology ────────────────────────────────────

def plot_topology(
    ax: plt.Axes,
    src_nid: int,
    dst_nid: int,
    hop1_nodes: list[int],
    hop2_nodes: list[int],
    id2ip: dict[int, str],
    class_name: str = "",
) -> None:
    """NetworkX spring-layout graph of the 2-hop neighbourhood."""
    import networkx as nx

    G = nx.DiGraph()
    role: dict[int, str] = {}

    all_nodes = [src_nid, dst_nid] + hop1_nodes + hop2_nodes
    for n in dict.fromkeys(all_nodes):
        G.add_node(n)
        if n == src_nid:
            role[n] = "target_src"
        elif n == dst_nid:
            role[n] = "target_dst"
        elif n in hop1_nodes:
            role[n] = "hop1"
        else:
            role[n] = "hop2"

    # Add target edge
    G.add_edge(src_nid, dst_nid)
    # Add 1-hop edges (hop1 → src or dst)
    for n in hop1_nodes:
        if n != src_nid and n != dst_nid:
            G.add_edge(n, src_nid)
            G.add_edge(n, dst_nid)
    # Add 2-hop edges (hop2 → hop1)
    for n in hop2_nodes:
        if n not in (src_nid, dst_nid) and n not in hop1_nodes:
            for h in hop1_nodes:
                G.add_edge(n, h)

    pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colours = [_ROLE_COLOURS[role.get(n, "hop2")] for n in G.nodes()]
    node_sizes   = [600 if role.get(n, "hop2") in ("target_src", "target_dst") else 300
                    for n in G.nodes()]
    labels = {n: _shorten_ip(id2ip.get(n, str(n))) for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color="gray", arrows=True,
                           arrowsize=10, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)

    ax.set_title(f"(b) 2-hop neighbourhood topology — {class_name}",
                 fontsize=9, fontweight="bold")
    ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=_ROLE_COLOURS[r], label=_ROLE_LABELS[r])
        for r in ["target_src", "target_dst", "hop1", "hop2"]
        if any(role.get(n) == r for n in G.nodes())
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="lower left")


# ── Panel C: node SHAP ────────────────────────────────────────────────────────

def plot_node_shap(
    ax: plt.Axes,
    node_shap_dict: dict[str, float],
    src_novelty_shap: float,
    dst_novelty_shap: float,
    id2ip: dict[int, str],
    class_name: str = "",
    top_n: int = 12,
) -> None:
    """Bar chart of node SHAP values (novelty + top non-target nodes)."""
    names: list[str] = []
    values: list[float] = []

    # Src and dst novelty flags always shown first
    names.append("src novelty")
    values.append(src_novelty_shap)
    names.append("dst novelty")
    values.append(dst_novelty_shap)

    # Non-target node states, sorted by |φ|
    sorted_nodes = sorted(node_shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for nid_str, phi in sorted_nodes[:top_n]:
        ip = id2ip.get(int(nid_str), nid_str)
        names.append(_shorten_ip(ip))
        values.append(phi)

    names_arr  = np.array(names[::-1])
    values_arr = np.array(values[::-1])

    y = np.arange(len(names_arr))
    colours = _bar_colours(values_arr)
    ax.barh(y, values_arr, color=colours, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names_arr, fontsize=7)
    ax.set_xlabel("SHAP value φ", fontsize=8)
    ax.set_title(f"(c) Node SHAP — {class_name}", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)

    # Shade the novelty rows
    ax.axhspan(len(names_arr) - 2 - 0.4, len(names_arr) - 1 + 0.4,
               alpha=0.08, color="gold", zorder=0)
    ax.text(0.98, 0.98, "gold = novelty flags",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="goldenrod")


# ── Panel D: temporal gap distribution ────────────────────────────────────────

def plot_temporal_gaps(
    ax: plt.Axes,
    neighbor_gap_seconds: np.ndarray,
    W_seconds: float,
    class_name: str = "",
) -> None:
    """Histogram of (target_ts − neighbor_ts) with W-second cutoff marked."""
    if len(neighbor_gap_seconds) == 0:
        ax.text(0.5, 0.5, "No neighbour edges sampled",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(f"(d) Temporal neighbour gap — {class_name}",
                     fontsize=9, fontweight="bold")
        return

    in_window = neighbor_gap_seconds[neighbor_gap_seconds <= W_seconds]
    out_window = neighbor_gap_seconds[neighbor_gap_seconds > W_seconds]

    bins = np.logspace(
        np.log10(max(neighbor_gap_seconds.min(), 1)),
        np.log10(neighbor_gap_seconds.max() + 1),
        40
    )
    ax.hist(out_window / 60, bins=bins / 60, color=_NEG, alpha=0.7,
            label=f"Outside W ({len(out_window)})")
    if len(in_window) > 0:
        ax.hist(in_window / 60, bins=bins / 60, color=_POS, alpha=0.9,
                label=f"Inside W ({len(in_window)})")

    ax.axvline(W_seconds / 60, color="red", linestyle="--", linewidth=1.2,
               label=f"W = {W_seconds:.0f} s")
    ax.set_xscale("log")
    ax.set_xlabel("Gap to target edge (minutes, log scale)", fontsize=8)
    ax.set_ylabel("Edge count", fontsize=8)
    ax.set_title(f"(d) Temporal neighbour gap — {class_name}\n"
                 f"φ_T ≈ 0: {len(in_window)}/{len(neighbor_gap_seconds)} "
                 f"neighbours within W={W_seconds:.0f} s",
                 fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)


# ── Composite figure ──────────────────────────────────────────────────────────

def make_case_study_figure(
    explanation: dict,
    topo: dict,
    id2ip: dict[int, str],
    class_name: str,
    W_seconds: float = 60.0,
    top_feat: int = 20,
    global_eid: int | None = None,
) -> plt.Figure:
    """Build the 2×2 four-panel case study figure.

    Args:
        explanation:  parsed _fixed.json dict (from re-explained flows).
        topo:         dict with keys 'hop1_nodes', 'hop2_nodes',
                      'all_neighbor_gaps_s' (array of seconds).
        id2ip:        node_id (int) → IP string.
        class_name:   attack class name for titles.
        W_seconds:    node-state window in seconds.
        top_feat:     number of feature groups to show in panel (a).
        global_eid:   global edge ID for figure suptitle (optional).

    Returns:
        matplotlib Figure (not yet saved).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_feat, ax_topo), (ax_node, ax_temp) = axes

    fig.suptitle(
        f"SHAP-GSD Case Study — {class_name}"
        + (f"  (EID {global_eid})" if global_eid else ""),
        fontsize=11, fontweight="bold", y=1.01,
    )

    # Panel (a): feature SHAP
    plot_feature_shap(
        ax_feat,
        group_names=explanation["feature_group_names"],
        shap_values=np.array(explanation["feature_group_shap"]),
        top_n=top_feat,
        class_name=class_name,
    )

    # Panel (b): topology
    plot_topology(
        ax_topo,
        src_nid=explanation["src_nid"],
        dst_nid=explanation["dst_nid"],
        hop1_nodes=topo.get("hop1_nodes", []),
        hop2_nodes=topo.get("hop2_nodes", []),
        id2ip=id2ip,
        class_name=class_name,
    )

    # Panel (c): node SHAP
    node_dict = explanation.get("node_shap_dict", {})
    plot_node_shap(
        ax_node,
        node_shap_dict=node_dict if isinstance(node_dict, dict) else {},
        src_novelty_shap=float(explanation.get("src_novelty_shap", 0.0)),
        dst_novelty_shap=float(explanation.get("dst_novelty_shap", 0.0)),
        id2ip=id2ip,
        class_name=class_name,
    )

    # Panel (d): temporal gap histogram
    gaps = np.array(topo.get("all_neighbor_gaps_s", []))
    plot_temporal_gaps(ax_temp, gaps, W_seconds=W_seconds, class_name=class_name)

    plt.tight_layout()
    return fig


# ── Per-class feature-group summary ───────────────────────────────────────────

def plot_class_feature_summary(
    ax: plt.Axes,
    group_names: list[str],
    shap_matrix: np.ndarray,
    class_name: str,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of mean signed φ ± 1 std across all flows in a class.

    Args:
        ax:           Matplotlib axes to draw on.
        group_names:  Length-K list of feature group names.
        shap_matrix:  (N, K) array of signed SHAP values (N flows, K groups).
        class_name:   Display name for the title.
        top_n:        Number of top groups to show (ranked by mean |φ|).
    """
    mean_phi = shap_matrix.mean(axis=0)
    std_phi  = shap_matrix.std(axis=0)

    # Rank by mean |φ| — preserves sign in the bar
    idx = np.argsort(np.abs(mean_phi))[::-1][:top_n]
    names  = [group_names[i] for i in idx]
    means  = mean_phi[idx]
    stds   = std_phi[idx]

    # Most important at top
    names = names[::-1]
    means = means[::-1]
    stds  = stds[::-1]

    colours = _bar_colours(means)
    y = np.arange(len(names))

    ax.barh(y, means, xerr=stds, color=colours,
            error_kw=dict(elinewidth=0.8, ecolor="black", capsize=2),
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Mean SHAP value φ  (±1 std)", fontsize=8)
    ax.set_title(
        f"{class_name}  (n={shap_matrix.shape[0]})",
        fontsize=9, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.text(0.98, 0.02, f"Σ|μφ| = {np.abs(means).sum():.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="dimgray")


def make_all_classes_figure(
    class_shap: dict[str, np.ndarray],
    group_names: list[str],
    top_n: int = 15,
    ncols: int = 5,
) -> plt.Figure:
    """2×5 grid of per-class mean-φ bar charts for all 10 classes.

    Args:
        class_shap:  dict mapping class_name → (N, K) shap_matrix.
        group_names: Length-K list of feature group names.
        top_n:       Groups per subplot.
        ncols:       Columns in the grid (rows computed automatically).

    Returns:
        matplotlib Figure.
    """
    classes = sorted(class_shap.keys())
    nrows = -(-len(classes) // ncols)   # ceiling division
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * (top_n * 0.35 + 1.2)),
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, cls in enumerate(classes):
        plot_class_feature_summary(
            axes_flat[i], group_names, class_shap[cls], cls, top_n=top_n
        )

    # Hide unused subplots
    for j in range(len(classes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "SHAP-GSD — Feature-group attributions by class (mean ± 1 std, signed φ)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig
