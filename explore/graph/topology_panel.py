"""
topology_panel.py — per-class topology figures (Option C)
==========================================================
What it shows:
  One 1×2 figure per attack class contrasting what SHAP-GSD sees (left) vs
  what Flat KernelSHAP sees (right) for the best representative flow of that
  class (selected by: maximise in-window temporal neighbours, break ties by
  max |node_shap|).

  Left (SHAP-GSD): full subgraph — temporal neighbour edges with width ∝ |φ_T|
  and alpha fading by age; computation nodes sized by |φ_N|; node-state
  annotations (deg, port-entropy) on src/dst.
  Right (Flat KernelSHAP): same node positions; temporal edges ghosted;
  uniform node sizes; amber warning box listing what flat SHAP cannot see.

Candidates (n_nbr = in-window temporal neighbours):
  Fuzzers   2117155  n_nbr=28
  Shellcode 2117054  n_nbr=25
  Exploits  2117278  n_nbr=25
  Analysis  2116629  n_nbr=25
  DoS       2117597  n_nbr=13
  Recon     2116715  n_nbr=12
  Backdoor  2348527  n_nbr=0   max|φ_N|=1.438
  Generic   2280979  n_nbr=0   max|φ_N|=1.293
  Worms     2267785  n_nbr=0   max|φ_N|=1.614

Files read (all resolved via explore._paths.paths(), run.dir-aware):
  outputs/explanations/<Class>/<EID>.json
  graphs/test.bin
  graphs/node_id_map.json
  node_state_snapshots/snapshots.pkl  (optional — empty when snapshots disabled)

Files output:
  outputs/figures/graph/topology_<Class>_<EID>.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import bisect
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
_P = paths()
OUT_DIR = _P["figures"] / "graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FS = 10

_TEAL  = "#2a9d8f"
_AMBER = "#e9c46a"
_GRAY  = "#888888"
_GHOST = "#cccccc"

# Best EID per class: (class_name, eid, n_nbr, selection_note)
CANDIDATES = [
    ("Fuzzers",   2117155, 28, "richest temporal context"),
    ("Shellcode", 2117054, 25, "temporal + strong node shap"),
    ("Exploits",  2117278, 25, "temporal (1 computation node)"),
    ("Analysis",  2116629, 25, "temporal (1 computation node)"),
    ("DoS",       2117597, 13, "moderate temporal context"),
    ("Recon",     2116715, 12, "temporal + balanced node shap"),
    ("Backdoor",  2348527,  0, "no temporal; max |phi_N|=1.438"),
    ("Generic",   2280979,  0, "no temporal; max |phi_N|=1.293"),
    ("Worms",     2267785,  0, "no temporal; max |phi_N|=1.614"),
]

# Distinct colour per class (IEEE-safe)
_CLASS_COLORS = {
    "Analysis":  "#6a4c93",
    "Backdoor":  "#fb8500",
    "DoS":       "#e63946",
    "Exploits":  "#457b9d",
    "Fuzzers":   "#2d6a4f",
    "Generic":   "#52b788",
    "Recon":     "#8ecae6",
    "Shellcode": "#c77dff",
    "Worms":     "#f4a261",
}


# ─────────────────────────── helpers ────────────────────────────────────────

def load_graph() -> tuple[dict, dict]:
    """Return (eid_to_edge, inv_map int→IP)."""
    graphs_dir = _P["graphs"]
    gs, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g = gs[0]
    src_arr = g.edges()[0].numpy()
    dst_arr = g.edges()[1].numpy()
    eid_arr = g.edata[dgl.EID].numpy()
    ts_arr  = g.edata["timestamp"].numpy()
    eid_to_edge = {
        int(eid_arr[i]): (int(src_arr[i]), int(dst_arr[i]), int(ts_arr[i]))
        for i in range(g.num_edges())
    }
    nmap    = json.loads((graphs_dir / "node_id_map.json").read_text())
    inv_map = {v: k for k, v in nmap.items()}
    return eid_to_edge, inv_map


def load_snapshots() -> dict:
    """Load node-state snapshot pickle (optional — empty when snapshots disabled)."""
    p = _P["node_state"] / "snapshots.pkl"
    if not p.exists():
        return {"times": [], "states": []}
    with open(p, "rb") as f:
        return pickle.load(f)


def get_node_state(snaps: dict, nid: int, ts_ms: int) -> Optional[np.ndarray]:
    """Return the 15-dim state vector for nid at ts_ms, or None."""
    idx = bisect.bisect_right(snaps["times"], ts_ms) - 1
    if idx < 0:
        return None
    return snaps["states"][idx].get(nid)


def short_ip(ip: str) -> str:
    """Collapse '10.40.182.3' → '.182.3'."""
    parts = ip.split(".")
    return "." + ".".join(parts[-2:]) if len(parts) == 4 else ip


# ─────────────────────────── panel drawing ──────────────────────────────────

def draw_shap_gsd_panel(
    ax,
    d: dict,
    eid_to_edge: dict,
    inv_map: dict,
    snaps: dict,
    class_color: str,
) -> dict:
    """
    Draw the SHAP-GSD view on ax.
    Returns the NetworkX position dict (reused by draw_flat_shap_panel).
    """
    target_eid  = d["edge_id"]
    target_edge = eid_to_edge.get(target_eid, (0, 1, 0))
    src_nid, dst_nid, target_ts = target_edge

    node_ids  = d.get("node_ids", [])
    node_shap = d.get("node_shap", [])
    nbr_eids  = d.get("neighbor_edge_ids", [])
    nbr_shap  = d.get("neighbor_shap", [])
    nbr_ts    = d.get("neighbor_timestamps", [])

    # Collect neighbour nodes
    nbr_nodes: set = set()
    for eid in nbr_eids:
        e = eid_to_edge.get(eid)
        if e:
            nbr_nodes.update([e[0], e[1]])

    comp_nodes = [n for n in node_ids if n not in {src_nid, dst_nid}]

    # Shell layout: [src,dst] inner; neighbour nodes middle; computation outer
    shell0 = [src_nid, dst_nid]
    shell1 = sorted(nbr_nodes - {src_nid, dst_nid})
    shell2 = sorted(set(comp_nodes) - {src_nid, dst_nid} - nbr_nodes)
    shells = [s for s in [shell0, shell1, shell2] if s]

    G = nx.DiGraph()
    all_nodes = list({src_nid, dst_nid} | nbr_nodes | set(comp_nodes))
    G.add_nodes_from(all_nodes)

    pos = (nx.shell_layout(G, nlist=shells) if len(shells) >= 2
           else nx.spring_layout(G, seed=42, k=1.5))

    # Node sizes and colours
    ns_map     = dict(zip(node_ids, node_shap))
    node_size  = []
    node_color = []
    for n in G.nodes():
        if n in {src_nid, dst_nid}:
            node_size.append(600)
            node_color.append(class_color)
        else:
            ns = abs(float(ns_map.get(n, 0.0)))
            node_size.append(ns * 200 + 150)
            node_color.append(_GRAY)

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes()),
                           node_size=node_size, node_color=node_color, alpha=0.85)

    label_map = {n: (short_ip(inv_map[n]) if n in inv_map else str(n))
                 for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=label_map,
                            font_size=7, font_color="black")

    # Node-state annotation on src / dst
    for nid in [src_nid, dst_nid]:
        sv = get_node_state(snaps, nid, target_ts)
        if sv is not None:
            out_deg   = sv[4]
            port_entr = sv[8]
            px, py = pos.get(nid, (0.0, 0.0))
            ax.text(px, py + 0.12,
                    f"deg={out_deg:.0f}\nH={port_entr:.2f}",
                    ha="center", va="bottom", fontsize=6, color="#333333")

    # Temporal neighbour edges: width ∝ |φ_T|, alpha fades by age
    if nbr_eids:
        ts_arr   = np.array(nbr_ts, dtype=float)
        ts_range = (ts_arr.max() - ts_arr.min()) or 1.0
        max_shap = max(abs(s) for s in nbr_shap) if nbr_shap else 1.0
        if max_shap == 0:
            max_shap = 1.0

        # Deduplicate parallel edges: keep max |φ_T|
        edge_dict: dict = {}
        for eid, shap, ts_val in zip(nbr_eids, nbr_shap, nbr_ts):
            e = eid_to_edge.get(eid)
            if not e:
                continue
            key = (e[0], e[1])
            if key not in edge_dict or abs(shap) > abs(edge_dict[key][0]):
                edge_dict[key] = (shap, ts_val)

        for (u, v), (shap, ts_val) in edge_dict.items():
            if u not in pos or v not in pos:
                continue
            age_norm = (ts_val - ts_arr.min()) / ts_range
            alpha    = 0.15 + 0.65 * age_norm
            lw       = 0.5 + 3.0 * abs(shap) / max_shap
            ax.annotate("",
                xy=pos[v], xytext=pos[u],
                arrowprops=dict(arrowstyle="-|>", color=_TEAL, lw=lw, alpha=alpha,
                                connectionstyle="arc3,rad=0.2"))

    # Target edge
    if src_nid in pos and dst_nid in pos:
        ax.annotate("",
            xy=pos[dst_nid], xytext=pos[src_nid],
            arrowprops=dict(arrowstyle="-|>", color=class_color, lw=2.5))

    ax.axis("off")
    return pos


def draw_flat_shap_panel(
    ax,
    d: dict,
    eid_to_edge: dict,
    inv_map: dict,
    pos: dict,
    class_color: str,
) -> None:
    """
    Draw the Flat KernelSHAP view on ax.
    Uses identical node positions from draw_shap_gsd_panel.
    """
    target_eid  = d["edge_id"]
    target_edge = eid_to_edge.get(target_eid, (0, 1, 0))
    src_nid, dst_nid, _ = target_edge

    node_ids  = d.get("node_ids", [])
    nbr_eids  = d.get("neighbor_edge_ids", [])
    node_shap = d.get("node_shap", [])

    all_nodes  = list(pos.keys())
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)

    node_color = [class_color if n in {src_nid, dst_nid} else _GRAY
                  for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=all_nodes,
                           node_size=200, node_color=node_color, alpha=0.8)
    label_map = {n: (short_ip(inv_map[n]) if n in inv_map else str(n))
                 for n in all_nodes}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=label_map,
                            font_size=7, font_color="black")

    # Ghost temporal neighbour edges
    seen: set = set()
    for eid in nbr_eids:
        e = eid_to_edge.get(eid)
        if not e:
            continue
        u, v = e[0], e[1]
        if (u, v) in seen or u not in pos or v not in pos:
            continue
        seen.add((u, v))
        ax.annotate("",
            xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-", color=_GHOST, lw=0.8,
                            alpha=0.20, linestyle="dashed",
                            connectionstyle="arc3,rad=0.2"))

    # Target edge
    if src_nid in pos and dst_nid in pos:
        ax.annotate("",
            xy=pos[dst_nid], xytext=pos[src_nid],
            arrowprops=dict(arrowstyle="-|>", color=class_color, lw=2.5))

    # Amber warning box
    n_nbr  = len(nbr_eids)
    n_nids = len(node_ids)
    ns_str = ", ".join(f"{v:+.3f}" for v in node_shap[:3])
    if len(node_shap) > 3:
        ns_str += ", ..."
    warn = (
        "Flat KernelSHAP sees:\n"
        "  ✓ 218 edge features\n"
        f"  ✗ {n_nbr} temporal neighbors (ghosted)\n"
        f"  ✗ {n_nids} node states (φ_N=[{ns_str}])\n"
        "  ✗ Rolling behaviour context"
    )
    ax.text(0.97, 0.05, warn,
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec=_AMBER, alpha=0.92),
            family="monospace")

    ax.axis("off")


# ─────────────────────────── per-class figure ───────────────────────────────

def make_class_figure(
    cls_name: str,
    eid: int,
    n_nbr_expected: int,
    note: str,
    eid_to_edge: dict,
    inv_map: dict,
    snaps: dict,
) -> None:
    """Produce topology_<cls_name>_<eid>.{pdf,png,txt} for one class."""
    json_path = _P["explanations"] / cls_name / f"{eid}.json"
    if not json_path.exists():
        print(f"  SKIP {cls_name} — {json_path} not found")
        return

    d          = json.loads(json_path.read_text())
    cls_color  = _CLASS_COLORS[cls_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.08)

    axes[0].set_title(f"SHAP-GSD  |  {cls_name}",
                      fontsize=LABEL_FS + 1, pad=6)
    axes[1].set_title(f"Flat KernelSHAP  |  {cls_name}",
                      fontsize=LABEL_FS + 1, pad=6)

    pos = draw_shap_gsd_panel(axes[0], d, eid_to_edge, inv_map, snaps, cls_color)
    draw_flat_shap_panel(axes[1], d, eid_to_edge, inv_map, pos, cls_color)

    # Below-panel captions (IEEE style)
    axes[0].text(0.5, -0.04, "(a) SHAP-GSD view",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=LABEL_FS, fontweight="bold")
    axes[1].text(0.5, -0.04, "(b) Flat KernelSHAP view",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=LABEL_FS, fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=cls_color, label=f"{cls_name} src/dst"),
        mpatches.Patch(color=_GRAY,     label="Computation node (size ∝ |φ_N|)"),
        mlines.Line2D([0], [0], color=_TEAL,  lw=2.0,
                      label="Temporal edge (width ∝ |φ_T|, alpha ∝ age)"),
        mlines.Line2D([0], [0], color=_GHOST, lw=0.8, ls="--",
                      label="Temporal edge (ghosted — invisible to flat SHAP)"),
    ]
    fig.legend(handles=handles, fontsize=LABEL_FS - 1,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10))

    stem = f"topology_{cls_name}_{eid}"
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}.{{pdf,png}}")

    # ── txt reasoning ────────────────────────────────────────────────────────
    n_nbr  = len(d.get("neighbor_edge_ids", []))
    n_nids = len(d.get("node_ids", []))
    ns     = d.get("node_shap", [])
    max_ns = max(abs(float(v)) for v in ns) if ns else 0.0
    te     = eid_to_edge.get(eid, (0, 1, 0))

    lines = [
        f"Figure reasoning — topology_{cls_name}_{eid}",
        "=" * 50, "",
        "SELECTION",
        f"  Class   : {cls_name}",
        f"  EID     : {eid}",
        f"  Reason  : {note}",
        f"  src→dst : {te[0]} → {te[1]}",
        f"  n_nbr   : {n_nbr}  (in-window temporal neighbours)",
        f"  n_nids  : {n_nids} (GNN computation nodes)",
        f"  max|φ_N|: {max_ns:.4f}",
        "",
        "KEY OBSERVATIONS",
    ]
    if n_nbr > 0:
        lines += [
            f"  - {n_nbr} temporal neighbour edges drawn with width ∝ |φ_T| and",
            "    alpha fading by age (left panel).",
            "  - Right panel ghosts the same edges — flat SHAP cannot attribute them.",
        ]
    else:
        lines += [
            "  - 0 temporal neighbours: temporal window contained no prior flows.",
            f"  - {n_nids} GNN computation nodes sized by |φ_N| (max={max_ns:.4f}).",
            "  - Structural attribution dominates; flat SHAP misses it entirely.",
        ]
    lines += [
        "",
        "SUGGESTED CAPTION",
        f"  Topology contrast for {cls_name} (EID {eid}).",
        "  (a) SHAP-GSD: src/dst nodes coloured; computation nodes sized by |φ_N|;",
        "  temporal edges width ∝ |φ_T|, alpha ∝ recency.",
        "  (b) Flat KernelSHAP: temporal edges ghosted; uniform node sizes.",
        "  Amber box itemises the three attribution components invisible to flat SHAP.",
    ]
    (OUT_DIR / f"{stem}.txt").write_text("\n".join(lines))
    print(f"  Saved {stem}.txt")


# ─────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    print("Loading graph and snapshots …")
    eid_to_edge, inv_map = load_graph()
    snaps = load_snapshots()

    for cls_name, eid, n_nbr, note in CANDIDATES:
        print(f"Processing {cls_name} EID={eid} …")
        make_class_figure(cls_name, eid, n_nbr, note, eid_to_edge, inv_map, snaps)

    print(f"\nAll topology panels saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
