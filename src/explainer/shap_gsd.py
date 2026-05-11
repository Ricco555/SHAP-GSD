"""
SHAP-GSD orchestrator. Runs all three granularities per target edge.

For each target edge e:
  1. Build 2-hop temporal subgraph (sampler)
  2. Run feature_shap   → φ_F ∈ R^K
  3. Run temporal_shap  → φ_T ∈ R^|E_sub|
  4. Run node_shap      → φ_N ∈ R^(|V_sub|+2)
  5. Extract top-K explanatory subgraph
  6. Return ExplanationResult

Batch mode:      explain_batch(edge_ids)       → list[ExplanationResult]
Stratified mode: explain_stratified(n_per_class) → dict[int, list[ExplanationResult]]
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import dgl
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.model.node_state import NodeStateManager
    from src.data.feature_store import FeatureStore
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """All three SHAP-GSD granularities for one target edge."""

    edge_id: int
    true_label: int
    predicted_label: int
    predicted_proba: np.ndarray          # shape (num_classes,)

    # Feature-level SHAP
    feature_group_names: list[str]       # length K
    feature_group_shap: np.ndarray       # shape (K,), signed

    # Temporal neighborhood SHAP
    neighbor_edge_ids: list[int]
    neighbor_timestamps: list[float]
    neighbor_shap: np.ndarray            # shape (|E_sub|,), signed

    # Node novelty SHAP
    node_ids: list[int]                  # non-target nodes
    node_shap: np.ndarray                # shape (|V_sub|,), signed
    src_novelty_shap: float
    dst_novelty_shap: float

    # Explanatory subgraph
    subgraph_edge_ids: list[int]         # top-K by |φ_T|
    subgraph_shap_weights: list[float]

    # Shapley efficiency baselines (logit space, for true_class)
    # f_baseline = E[f(background)]; f_logit = f(all-present foreground)
    # efficiency_error = |sum(phi) - (f_logit - f_baseline)|
    f_baseline_feature: float = 0.0
    f_logit_feature: float = 0.0
    f_baseline_temporal: float = 0.0
    f_logit_temporal: float = 0.0
    f_baseline_node: float = 0.0
    f_logit_node: float = 0.0

    # Per-layer timing (wall-clock seconds)
    runtime_feature_s: float = 0.0
    runtime_temporal_s: float = 0.0
    runtime_node_s: float = 0.0
    runtime_s: float = 0.0              # sum of the three layers


class SHAPGSDExplainer:
    """Run all three SHAP-GSD granularities for test-split edges."""

    def __init__(
        self,
        model: "EdgeAwareGraphSAGE",
        g_split: "dgl.DGLGraph",
        fs: "FeatureStore",
        nsm: "NodeStateManager",
        background: "BackgroundDistributions",
        feature_groups: dict,
        cfg: dict,
        device: torch.device,
    ) -> None:
        """
        Args:
            model:          trained EdgeAwareGraphSAGE (will be set to eval mode).
            g_split:        DGL graph for the split being explained (test).
            fs:             FeatureStore for the same split.
            nsm:            NodeStateManager (loaded from training snapshots).
            background:     BackgroundDistributions (training split).
            feature_groups: parsed feature_groups.json dict.
            cfg:            full merged config dict.
            device:         torch device.
        """
        import dgl as _dgl
        from src.model.temporal_sampler import TemporalNeighborSampler
        from src.explainer.feature_shap import FeatureGroupSHAP
        from src.explainer.temporal_shap import TemporalNeighborhoodSHAP
        from src.explainer.node_shap import NodeNoveltySHAP

        self.model = model
        self.g_split = g_split
        self.fs = fs
        self.nsm = nsm
        self.background = background
        self.cfg = cfg
        self.device = device
        self._dgl = _dgl

        model.eval()

        m = cfg["model"]
        self.fanouts: list[int] = m["fanouts"]
        self.num_classes: int = m["num_classes"]
        self.subgraph_k: int = cfg.get("explainer", {}).get("subgraph_k", 10)

        self.sampler = TemporalNeighborSampler(fanouts=self.fanouts)
        self.feat_shap = FeatureGroupSHAP(feature_groups, background, device)
        self.temp_shap = TemporalNeighborhoodSHAP(background, nsm, g_split, device)
        self.node_shap = NodeNoveltySHAP(background, device)

        feat_nsamples = cfg.get("explainer", {}).get("feature_nsamples", 512)
        temp_nsamples = cfg.get("explainer", {}).get("temporal_nsamples", 1024)
        node_nsamples = cfg.get("explainer", {}).get("node_nsamples", 512)
        self._feat_nsamples = feat_nsamples
        self._temp_nsamples = temp_nsamples
        self._node_nsamples = node_nsamples

    def explain_edge(self, local_eid: int) -> ExplanationResult:
        """Explain one edge identified by its local EID in g_split.

        Args:
            local_eid: local edge ID (0-indexed in g_split).

        Returns:
            ExplanationResult with all three SHAP granularities filled.
        """
        _t0 = time.time()
        dgl = self._dgl
        seed_eid_t = torch.tensor([local_eid], dtype=torch.long)

        # --- Sample computation blocks ---
        input_nodes, seed_eids, blocks = self.sampler.sample_blocks(
            self.g_split, seed_eid_t
        )
        blocks = [b.to(self.device) for b in blocks]
        input_nodes = input_nodes.to(self.device)

        # --- Look up target edge metadata ---
        global_eid = int(self.g_split.edata[dgl.EID][local_eid])
        target_ts = float(self.g_split.edata["timestamp"][local_eid])
        src_t, dst_t = self.g_split.find_edges(seed_eid_t)
        target_src = int(src_t[0])
        target_dst = int(dst_t[0])

        x_e = self.fs[global_eid].copy()  # (d_e,)
        true_label = int(self.fs.labels[self.fs._eid_to_pos[global_eid]])

        # --- Build base node features (all neighbors present) ---
        input_node_ids = input_nodes.cpu().numpy()
        base_node_feats = np.stack([
            self.nsm.get_state_at_time(int(nid), target_ts)
            for nid in input_node_ids
        ])

        node_feats_t = torch.tensor(
            base_node_feats, dtype=torch.float32, device=self.device
        )
        x_e_t = torch.tensor(x_e, dtype=torch.float32, device=self.device).unsqueeze(0)

        # --- Compute predicted label and probabilities ---
        from src.model.sage_model import build_src_dst_pos
        seed_nodes_final = blocks[-1].dstdata[dgl.NID]
        src_pos, dst_pos = build_src_dst_pos(
            self.g_split, seed_eid_t, seed_nodes_final
        )
        src_pos = src_pos.to(self.device)
        dst_pos = dst_pos.to(self.device)

        with torch.no_grad():
            logits = self.model(blocks, node_feats_t, x_e_t, src_pos, dst_pos)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label = int(np.argmax(proba))

        # --- Feature-group SHAP ---
        _t_feat = time.time()
        feat_phi_dict, f_baseline_feat, f_logit_feat = self.feat_shap.explain(
            true_class=true_label,
            model=self.model,
            blocks=blocks,
            node_feats=node_feats_t,
            x_e=x_e,
            src_pos=src_pos,
            dst_pos=dst_pos,
            nsamples=self._feat_nsamples,
        )
        runtime_feature_s = time.time() - _t_feat
        group_names = list(feat_phi_dict.keys())
        feat_phi_arr = np.array([feat_phi_dict[n] for n in group_names])

        # --- Temporal neighborhood SHAP ---
        _t_temp = time.time()
        temp_results, f_baseline_temp, f_logit_temp = self.temp_shap.explain(
            target_local_eid=local_eid,
            true_class=true_label,
            model=self.model,
            blocks=blocks,
            input_nodes=input_nodes.cpu(),
            target_ts_ms=target_ts,
            src_pos=src_pos,
            dst_pos=dst_pos,
            x_e=x_e,
            base_node_feats=base_node_feats,
            nsamples=self._temp_nsamples,
        )
        runtime_temporal_s = time.time() - _t_temp
        neighbor_eids = [t[0] for t in temp_results]
        neighbor_ts = [t[1] for t in temp_results]
        neighbor_phi = np.array([t[2] for t in temp_results])

        # --- Node novelty SHAP ---
        _t_node = time.time()
        node_result = self.node_shap.explain(
            true_class=true_label,
            src_nid=target_src,
            dst_nid=target_dst,
            model=self.model,
            blocks=blocks,
            input_nodes=input_nodes.cpu(),
            base_node_feats=base_node_feats,
            src_pos=src_pos,
            dst_pos=dst_pos,
            x_e=x_e,
            nsamples=self._node_nsamples,
        )
        runtime_node_s = time.time() - _t_node
        non_target_node_ids = [int(k) for k in node_result["node_shap"].keys()]
        node_phi_arr = np.array(list(node_result["node_shap"].values()))
        f_baseline_node = node_result["f_baseline"]
        f_logit_node = node_result["f_logit"]

        # --- Top-K explanatory subgraph ---
        from src.explainer.subgraph_extractor import extract_top_k_subgraph
        # Build endpoint dict for all neighbor edges
        edge_endpoints: dict[int, tuple[int, int]] = {}
        for geid, _ts, _phi in temp_results:
            # Retrieve src/dst from g_split via global EID
            # global_eid → local_eid for g_split: inverse lookup
            local_nbr = int(torch.where(
                self.g_split.edata[dgl.EID] == geid
            )[0][0])
            s, d = self.g_split.find_edges(torch.tensor([local_nbr]))
            edge_endpoints[geid] = (int(s[0]), int(d[0]))

        subgraph = extract_top_k_subgraph(
            neighbor_shap=temp_results,
            target_edge_id=global_eid,
            target_src=target_src,
            target_dst=target_dst,
            edge_endpoints=edge_endpoints,
            k=self.subgraph_k,
        )
        subgraph_eids = [t[0] for t in subgraph]
        subgraph_weights = [t[1] for t in subgraph]

        total_s = time.time() - _t0
        return ExplanationResult(
            edge_id=global_eid,
            true_label=true_label,
            predicted_label=predicted_label,
            predicted_proba=proba,
            feature_group_names=group_names,
            feature_group_shap=feat_phi_arr,
            neighbor_edge_ids=neighbor_eids,
            neighbor_timestamps=neighbor_ts,
            neighbor_shap=neighbor_phi,
            node_ids=non_target_node_ids,
            node_shap=node_phi_arr,
            src_novelty_shap=node_result["src_novelty_shap"],
            dst_novelty_shap=node_result["dst_novelty_shap"],
            subgraph_edge_ids=subgraph_eids,
            subgraph_shap_weights=subgraph_weights,
            f_baseline_feature=f_baseline_feat,
            f_logit_feature=f_logit_feat,
            f_baseline_temporal=f_baseline_temp,
            f_logit_temporal=f_logit_temp,
            f_baseline_node=f_baseline_node,
            f_logit_node=f_logit_node,
            runtime_feature_s=round(runtime_feature_s, 4),
            runtime_temporal_s=round(runtime_temporal_s, 4),
            runtime_node_s=round(runtime_node_s, 4),
            runtime_s=round(total_s, 4),
        )

    def explain_batch(self, local_eids: list[int]) -> list[ExplanationResult]:
        """Explain a list of edges by their local EIDs.

        Args:
            local_eids: local edge IDs in g_split.

        Returns:
            list of ExplanationResult (same order as input).
        """
        results = []
        for i, leid in enumerate(local_eids):
            logger.info(f"Explaining edge {i+1}/{len(local_eids)} (local_eid={leid})")
            try:
                results.append(self.explain_edge(leid))
            except Exception:
                logger.exception(f"Failed to explain local_eid={leid}, skipping")
        return results

    def explain_stratified(
        self,
        n_per_class: int = 200,
        rng_seed: int = 123,
    ) -> dict[int, list[ExplanationResult]]:
        """Explain n_per_class edges per class from g_split.

        Args:
            n_per_class: max edges to explain per class.
            rng_seed:    numpy RNG seed for reproducible sampling.

        Returns:
            dict mapping class_int → list[ExplanationResult].
        """
        dgl = self._dgl
        # Labels are in fs; EIDs are local positions in g_split
        n_split = self.g_split.num_edges()
        rng = np.random.default_rng(rng_seed)

        all_results: dict[int, list[ExplanationResult]] = {}

        for c in range(self.num_classes):
            # Get local EIDs for this class: local EID i maps to global EID
            # via g_split.edata[dgl.EID][i]
            global_eids = self.g_split.edata[dgl.EID].numpy()
            labels = np.array([
                self.fs.labels[self.fs._eid_to_pos[int(geid)]]
                for geid in global_eids
            ])
            class_local_eids = np.where(labels == c)[0]

            if len(class_local_eids) == 0:
                logger.info(f"Class {c}: no edges in split, skipping")
                all_results[c] = []
                continue

            sample = rng.choice(
                class_local_eids,
                size=min(n_per_class, len(class_local_eids)),
                replace=False,
            )
            sample = np.sort(sample)  # temporal order for sampler assertion

            logger.info(
                f"Class {c}: explaining {len(sample)}/{len(class_local_eids)} edges"
            )
            all_results[c] = self.explain_batch(sample.tolist())

        return all_results
