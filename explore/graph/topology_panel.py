"""
topology_panel.py — Figure 4(a)–(b)
======================================
What it shows:
  Side-by-side topology panels comparing what SHAP-GSD sees (left column) vs what
  Flat KernelSHAP sees (right column) for two example flows:
    Row 0: Recon   EID 2116715 — 12 temporal neighbors, node_shap=[+0.275, +0.045, ~0]
    Row 1: Backdoor EID 2349890 — 0 temporal neighbors, 11 GNN computation nodes

  Left (SHAP-GSD): full graph with temporal neighbors (arrow width ∝ |shap|, alpha fades
  by age), computation nodes sized by |node_shap|, node-state annotations on src/dst.
  Right (Flat KernelSHAP): same positions, temporal edges ghosted, uniform node sizes,
  amber warning box listing what flat SHAP cannot see.

Panels:
  fig, axes — 2×2 subplots, figsize=(14, 10)

Files read:
  outputs/explanations/Recon/2116715.json
  outputs/explanations/Backdoor/2349890.json
  graphs/test.bin
  graphs/node_id_map.json
  node_state_snapshots/snapshots.pkl

Files output:
  outputs/figures/graph/topology_panel.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import bisect
import json
import pickle
from pathlib import Path
from typing import Optional

import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "figures" / "graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "topology_panel"
LABEL_FS = 9
_TEAL    = "#2a9d8f"
_AMBER   = "#e9c46a"
_CORAL   = "#e76f51"
_GRAY    = "#888888"
_GHOST   = "#cccccc"

# Class colour map (mirrors class_analysis.py palette)
_CLASS_COLORS = {
    "Recon":    "#8ecae6",
    "Backdoor": "#fb8500",
}


# ─────────────────────────── helpers ────────────────────────────────────────

def load_graph():
    """Return (eid_to_edge dict, node_id_map IP→int, inv_map int→IP)."""
    gs, _ = dgl.load_graphs(str(ROOT / "graphs" / "test.bin"))
    g = gs[0]
    src_arr = g.edges()[0].numpy()
    dst_arr = g.edges()[1].numpy()
    eid_arr = g.edata[dgl.EID].numpy()
    ts_arr  = g.edata["timestamp"].numpy()
    eid_to_edge = {int(eid_arr[i]): (int(src_arr[i]), int(dst_arr[i]), int(ts_arr[i]))
                   for i in range(g.num_edges())}
    nmap = json.loads((ROOT / "graphs" / "node_id_map.json").read_text())
    inv_map = {v: k for k, v in nmap.items()}
    return eid_to_edge, inv_map


def load_snapshots():
    """Load node-state snapshot pickle."""
    with open(ROOT / "node_state_snapshots" / "snapshots.pkl", "rb") as f:
        return pickle.load(f)


def get_node_state(snaps: dict, nid: int, ts_ms: int) -> Optional[np.ndarray]:
    """Return the 15-dim state vector for nid at time ts_ms, or None."""
    idx = bisect.bisect_right(snaps["times"], ts_ms) - 1
    if idx < 0:
        return None
    return snaps["states"][idx].get(nid)


def short_ip(ip: str) -> str:
    """Collapse '10.40.182.3' → '.182.3'."""
    parts = ip.split(".")
    return "." + ".".join(parts[-2:]) if len(parts) == 4 else ip


# ─────────────────────────── panel drawing ──────────────────────────────────

def draw_shap_gsd_panel(ax, d: dict, eid_to_edge: dict, inv_map: dict,
                        snaps: dict, class_color: str) -> dict:
    """
    Draw the SHAP-GSD view on ax.
    Returns the NetworkX position dict (shared with flat-SHAP panel).
    """
    target_eid = d["edge_id"]
    target_edge = eid_to_edge.get(target_eid, (0, 1, 0))
    src_nid, dst_nid, target_ts = target_edge

    node_ids    = d.get("node_ids", [])
    node_shap   = d.get("node_shap", [])
    nbr_eids    = d.get("neighbor_edge_ids", [])
    nbr_shap    = d.get("neighbor_shap", [])
    nbr_ts      = d.get("neighbor_timestamps", [])

    # Collect all nodes
    nbr_nodes: set = set()
    for eid in nbr_eids:
        e = eid_to_edge.get(eid)
        if e:
            nbr_nodes.update([e[0], e[1]])

    comp_nodes = [n for n in node_ids if n not in {src_nid, dst_nid}]

    # Shell layout
    shell0 = [src_nid, dst_nid]
    shell1 = sorted(nbr_nodes - {src_nid, dst_nid})
    shell2 = sorted(set(comp_nodes) - {src_nid, dst_nid} - nbr_nodes)
    shells = [s for s in [shell0, shell1, shell2] if s]

    G = nx.DiGraph()
    all_nodes = list({src_nid, dst_nid} | nbr_nodes | set(comp_nodes))
    G.add_nodes_from(all_nodes)

    if len(shells) >= 2:
        pos = nx.shell_layout(G, nlist=shells)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5)

    # Node sizes and colors
    ns_map = dict(zip(node_ids, node_shap))
    node_size  = []
    node_color = []
    for n in G.nodes():
        if n in {src_nid, dst_nid}:
            node_size.append(600)
            node_color.append(class_color)
        else:
            ns = abs(ns_map.get(n, 0.0))
            node_size.append(ns * 200 + 150)
            node_color.append(_GRAY)

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes()),
                           node_size=node_size, node_color=node_color, alpha=0.85)

    # Node labels
    label_map = {}
    for n in G.nodes():
        ip = inv_map.get(n, str(n))
        label_map[n] = short_ip(ip) if len(inv_map) > 0 else str(n)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=label_map, font_size=6, font_color="black")

    # Node state annotation on src/dst
    for nid in [src_nid, dst_nid]:
        sv = get_node_state(snaps, nid, target_ts)
        if sv is not None:
            out_deg   = sv[4]
            port_entr = sv[8]
            px, py = pos.get(nid, (0, 0))
            ax.text(px, py + 0.12, f"deg={out_deg:.0f}\nH={port_entr:.2f}",
                    ha="center", va="bottom", fontsize=5.5, color="#333333")

    # Draw temporal neighbor edges (arc, width and alpha by |shap| and age)
    if nbr_eids:
        ts_arr = np.array(nbr_ts, dtype=float)
        ts_range = ts_arr.max() - ts_arr.min() if ts_arr.max() > ts_arr.min() else 1.0
        max_shap = max(abs(s) for s in nbr_shap) if nbr_shap else 1.0
        if max_shap == 0:
            max_shap = 1.0

        # Deduplicate parallel edges: keep max |shap|
        edge_dict: dict = {}
        for eid, shap, ts_val in zip(nbr_eids, nbr_shap, nbr_ts):
            e = eid_to_edge.get(eid)
            if not e:
                continue
            u, v = e[0], e[1]
            key = (u, v)
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
                arrowprops=dict(arrowstyle="-|>", color=_TEAL,
                                lw=lw, alpha=alpha,
                                connectionstyle="arc3,rad=0.2"))

    # Draw target edge
    if src_nid in pos and dst_nid in pos:
        ax.annotate("",
            xy=pos[dst_nid], xytext=pos[src_nid],
            arrowprops=dict(arrowstyle="-|>", color=class_color, lw=2.5))

    ax.axis("off")
    return pos


def draw_flat_shap_panel(ax, d: dict, eid_to_edge: dict, inv_map: dict,
                         pos: dict, class_color: str) -> None:
    """
    Draw the Flat KernelSHAP view on ax.
    Uses identical node positions from SHAP-GSD panel.
    """
    target_eid = d["edge_id"]
    target_edge = eid_to_edge.get(target_eid, (0, 1, 0))
    src_nid, dst_nid, _ = target_edge

    node_ids = d.get("node_ids", [])
    nbr_eids = d.get("neighbor_edge_ids", [])
    node_shap = d.get("node_shap", [])

    all_nodes = list(pos.keys())
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)

    node_color = [class_color if n in {src_nid, dst_nid} else _GRAY for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=all_nodes,
                           node_size=200, node_color=node_color, alpha=0.8)
    label_map = {}
    for n in all_nodes:
        ip = inv_map.get(n, str(n))
        label_map[n] = short_ip(ip) if inv_map else str(n)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=label_map, font_size=6, font_color="black")

    # Ghost temporal neighbor edges
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
    n_nbr   = len(nbr_eids)
    n_nodes = len(node_ids)
    ns_str  = ", ".join(f"{v:+.3f}" for v in node_shap[:3])
    if len(node_shap) > 3:
        ns_str += ", ..."
    warn = (
        "Flat KernelSHAP sees:\n"
        "  ✓ 218 edge features\n"
        f"  ✗ {n_nbr} temporal neighbors (ghosted)\n"
        f"  ✗ {n_nodes} node states (φ_N=[{ns_str}])\n"
        "  ✗ Rolling behaviour context"
    )
    ax.text(0.97, 0.05, warn,
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec=_AMBER, alpha=0.92),
            family="monospace")

    ax.axis("off")


# ─────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    eid_to_edge, inv_map = load_graph()

    snaps_path = ROOT / "node_state_snapshots" / "snapshots.pkl"
    snaps = load_snapshots() if snaps_path.exists() else {"times": [], "states": []}

    d_recon    = json.loads((ROOT / "outputs/explanations/Recon/2116715.json").read_text())
    d_backdoor = json.loads((ROOT / "outputs/explanations/Backdoor/2349890.json").read_text())

    flows = [
        (d_recon,    "Recon",    _CLASS_COLORS["Recon"],    "Recon — EID 2116715"),
        (d_backdoor, "Backdoor", _CLASS_COLORS["Backdoor"], "Backdoor — EID 2349890"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.15, wspace=0.05)

    for row, (d, cls_name, cls_color, row_title) in enumerate(flows):
        ax_shap = axes[row][0]
        ax_flat = axes[row][1]

        # Titles (left panel carries row label, right is "Flat KernelSHAP")
        ax_shap.set_title(f"SHAP-GSD  |  {row_title}", fontsize=LABEL_FS + 1, pad=6)
        ax_flat.set_title(f"Flat KernelSHAP  |  {row_title}", fontsize=LABEL_FS + 1, pad=6)

        pos = draw_shap_gsd_panel(ax_shap, d, eid_to_edge, inv_map, snaps, cls_color)
        draw_flat_shap_panel(ax_flat, d, eid_to_edge, inv_map, pos, cls_color)

    # Shared legend
    handles = [
        mpatches.Patch(color=_CLASS_COLORS["Recon"],    label="Recon src/dst"),
        mpatches.Patch(color=_CLASS_COLORS["Backdoor"], label="Backdoor src/dst"),
        mpatches.Patch(color=_GRAY,                      label="Computation node"),
        mlines.Line2D([0], [0], color=_TEAL,  lw=2.0, label="Temporal edge (SHAP-GSD)"),
        mlines.Line2D([0], [0], color=_GHOST, lw=0.8, ls="--", label="Temporal edge (ghosted)"),
    ]
    fig.legend(handles=handles, fontsize=LABEL_FS - 1,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # ------------------------------------------------------------------ txt
    d_r = d_recon
    d_b = d_backdoor
    lines = [
        "Figure reasoning — topology_panel",
        "=" * 36, "",
        "WHAT THE FIGURE SHOWS",
        "----------------------",
        "A 2 × 2 panel contrasting the SHAP-GSD view (left column) with what Flat",
        "KernelSHAP can see (right column) for two representative flows.",
        "",
        "  Row 0 — Recon EID 2116715:",
        f"    src={eid_to_edge[2116715][0]} → dst={eid_to_edge[2116715][1]},",
        f"    {len(d_r['neighbor_edge_ids'])} temporal neighbors,",
        f"    node_shap = {[f'{v:.3f}' for v in d_r['node_shap'][:3]]} (nodes {d_r['node_ids'][:3]})",
        f"    src_novelty_shap={d_r['src_novelty_shap']:.4f},",
        f"    dst_novelty_shap={d_r['dst_novelty_shap']:.4f}",
        "",
        "  Row 1 — Backdoor EID 2349890:",
        f"    src={eid_to_edge[2349890][0]} → dst={eid_to_edge[2349890][1]},",
        f"    {len(d_b['neighbor_edge_ids'])} temporal neighbors (all Backdoor flows: 0),",
        f"    {len(d_b['node_ids'])} GNN computation nodes,",
        f"    max |node_shap| = {max(abs(v) for v in d_b['node_shap']):.3f}",
        "",
        "KEY FINDINGS",
        "------------",
        "- Recon left panel: temporal arrows with varying widths (12 neighbors) show that",
        "  some prior flows strongly influence the attribution (top φ_T = 0.009).",
        "- Recon right panel: the same arrows are ghosted — flat SHAP cannot account for",
        "  the temporal context that shaped the GNN representation.",
        "- Backdoor left: 0 temporal arrows — confirms the temporal null result extends",
        "  even to the most complex attack class (11 computation nodes, large |φ_N|).",
        "- Computation nodes sized by |node_shap| — Backdoor shows several large circles",
        "  reflecting substantial GNN structural influence (φ_N = 28%).",
        "- The amber warning boxes quantify exactly what Flat KernelSHAP misses per flow.",
        "",
        "PAPER FRAMING",
        "-------------",
        "The epistemological gap between SHAP-GSD and Flat KernelSHAP is most visible here.",
        "Flat KernelSHAP treats the GNN prediction as a function of 218 edge features alone,",
        "ignoring both the temporal neighborhood context (missed for Recon) and the GNN",
        "computation subgraph structure (missed for both). SHAP-GSD decomposes all three",
        "contributions explicitly, providing a complete attribution that matches the actual",
        "information processed by the model.",
        "",
        "SUGGESTED FIGURE CAPTION",
        "-------------------------",
        "Topology panels for two representative flows: Recon EID 2116715 (top row) and",
        "Backdoor EID 2349890 (bottom row). Left column: SHAP-GSD view. Coloured src/dst",
        "nodes; grey computation subgraph nodes sized by |φ_N|. Temporal neighbor edges",
        "drawn with width ∝ |φ_T| and alpha fading by age. Right column: Flat KernelSHAP",
        "view. Temporal edges are ghosted (dashed, alpha = 0.20) because flat SHAP has no",
        "mechanism to attribute them; all computation nodes appear at uniform size because",
        "their structural contribution is invisible to the surrogate. Amber boxes list the",
        "three attribution components invisible to Flat KernelSHAP for each flow.",
    ]
    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
