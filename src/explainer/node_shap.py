"""
Node novelty coalition SHAP. Third SHAP-GSD granularity.

Answers: "Did endpoint history/novelty drive the classification?"

Coalition space: z_N ∈ {0,1}^(|V_sub| + 2) where:
  z_N[0]: source novelty flag  (1=keep actual, 0=zero out state dim 1)
  z_N[1]: dest novelty flag    (1=keep actual, 0=zero out state dim 1)
  z_N[2:]: non-target node presence (0=replace full 15-dim state with background)

Masking rules:
  Non-target node absent: replace full 15-dim state with background_node_state[c]
  Target novelty absent (z_N[0/1]=0): zero state[1] (novelty dim), keep rest
  time_sin (dim 11), time_cos (dim 12): never masked — time context not identity
  volume_deviation (dim 13), iat_regularity (dim 14): masked with non-target nodes

Output: dict with 'src_novelty', 'dst_novelty', and per-node φ values.

Two dummy-player guards drop a target-novelty column from the KernelSHAP
regression before it is fit, so a player that provably cannot move the
model receives an exact 0.0 rather than solver noise:

  * input-degeneracy (specs/63): the player's masked and unmasked *inputs*
    are bit-identical.
  * output-dummy (specs/65): the input genuinely varies, but the model's
    *output* does not respond. Strictly stronger than the first, and the
    only one that catches a trained-away (weight-decayed) input column.
"""

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    import dgl
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)

_NOVELTY_DIM = 1
_NEVER_MASK_DIMS = (11, 12)  # time_sin, time_cos
# See feature_shap.py's _L1_REG for the shared rationale (specs/45 sec 1-2,
# 3.2).
_L1_REG: bool = False


def build_non_target_ids(
    input_node_ids: np.ndarray,
    src_nid: int,
    dst_nid: int,
) -> list[int]:
    """Non-target node IDs in coalition-column order (indices 2..2+M-1).

    Single source of truth for the node-novelty player layout: both
    ``NodeNoveltySHAP.explain`` and the fidelity pass in
    ``src/explainer/node_fidelity.py`` derive the coalition columns from
    this function, so a stored per-node phi can never be attached to the
    wrong player.

    Args:
        input_node_ids: int array (N_in,) of global node IDs for blocks[0]
                        input, in block row order.
        src_nid:        global node ID of the target edge's source.
        dst_nid:        global node ID of the target edge's destination.

    Returns:
        The input node IDs with both target endpoints removed, in the
        original block row order.
    """
    return [
        int(nid) for nid in input_node_ids
        if int(nid) != src_nid and int(nid) != dst_nid
    ]


def build_non_target_pos(non_target_ids: list[int]) -> dict[int, int]:
    """{global node id -> coalition column}, columns starting at 2.

    Args:
        non_target_ids: output of :func:`build_non_target_ids`.

    Returns:
        Mapping from global node ID to its coalition column, where column
        0 is the target source novelty flag and column 1 the target
        destination novelty flag.
    """
    return {nid: 2 + i for i, nid in enumerate(non_target_ids)}


class NodeNoveltySHAP:
    """KernelSHAP over node novelty and non-target node states."""

    def __init__(
        self,
        background: "BackgroundDistributions",
        device: torch.device,
    ) -> None:
        """
        Args:
            background: BackgroundDistributions (training split).
            device:     torch device.
        """
        self.background = background
        self.device = device

    def build_masked_node_states(
        self,
        row: np.ndarray,
        base_node_feats: np.ndarray,
        input_node_ids: np.ndarray,
        src_nid: int,
        dst_nid: int,
        non_target_pos: dict[int, int],
        true_class: int,
    ) -> np.ndarray:
        """Node-state matrix for one node-novelty coalition (Option C masking).

        Two masking mechanisms, dispatched per input-node row — never
        interchangeable:

          * target endpoint (src_nid / dst_nid) with its novelty bit absent
            (row[0] / row[1] == 0): ONLY dim _NOVELTY_DIM is zeroed; every
            other dim of the node's actual state is preserved.
          * non-target node with its own bit absent: the full state row is
            replaced by background_node_state[true_class], with dims
            _NEVER_MASK_DIMS copied back from the node's actual state.

        Nothing is rebuilt in DGL: the caller's `blocks` object is reused
        unchanged (CLAUDE.md Option C).

        Unlike the temporal granularity's masking helper, this takes the FULL
        binary coalition row rather than an absent-index list, so that the
        per-player mechanism dispatch lives here and cannot be duplicated or
        conflated by callers.

        Args:
            row:             (2 + M,) binary coalition vector. Index 0 = target
                             src novelty, index 1 = target dst novelty,
                             indices 2.. = non-target nodes in non_target_pos
                             order. 1 = present (keep actual), 0 = absent.
            base_node_feats: float32 (N_in, node_state_dim) actual node states;
                             row j corresponds to input_node_ids[j].
            input_node_ids:  int array (N_in,) of global node IDs for blocks[0]
                             input, in block row order.
            src_nid:         global node ID of the target edge's source.
            dst_nid:         global node ID of the target edge's destination.
            non_target_pos:  {global node id -> coalition column}, as built by
                             build_non_target_pos().
            true_class:      class index selecting the background node state.

        Returns:
            float32 array (N_in, node_state_dim) — always a fresh copy, never
            `base_node_feats` itself.
        """
        assert base_node_feats.shape[0] == len(input_node_ids), (
            f"row misalignment: base_node_feats has {base_node_feats.shape[0]} rows "
            f"but input_node_ids has {len(input_node_ids)}"
        )
        assert len(row) == 2 + len(non_target_pos), (
            f"coalition width {len(row)} != 2 + {len(non_target_pos)} players"
        )

        bg_ns = self.background.background_node_state[true_class]  # (15,)
        modified = base_node_feats.copy()
        src_novelty_present = bool(row[0])
        dst_novelty_present = bool(row[1])

        for j, nid in enumerate(input_node_ids):
            nid_int = int(nid)
            if nid_int == src_nid and not src_novelty_present:
                # Zero out novelty dimension; preserve all other dims
                modified[j, _NOVELTY_DIM] = 0.0
            elif nid_int == dst_nid and not dst_novelty_present:
                modified[j, _NOVELTY_DIM] = 0.0
            elif nid_int in non_target_pos:
                if not row[non_target_pos[nid_int]]:
                    # Replace full state with class-conditional background
                    bg = bg_ns.copy()
                    # Preserve time context (never masked)
                    for d in _NEVER_MASK_DIMS:
                        bg[d] = base_node_feats[j, d]
                    modified[j] = bg
        return modified

    def screen_novelty_players(
        self,
        forward_full: "Callable[[np.ndarray], float]",
        base_node_feats: np.ndarray,
        input_node_ids: np.ndarray,
        src_nid: int,
        dst_nid: int,
        coalition_size: int,
    ) -> tuple[bool, bool, bool, bool]:
        """Decide which of the two target-novelty players are provably null.

        Two independent guards, applied in order. Each identifies a player
        whose contribution the solver cannot legitimately estimate, so the
        caller drops its column from the KernelSHAP regression and reports
        an exact 0.0 instead of letting the solver fit noise onto it.

        The two differ in evidential strength, and the difference is
        deliberate. Guard 1 is a *proof*: bit-identical inputs mean the
        player's TRUE Shapley value is exactly 0 for every coalition.
        Guard 2 is a *necessary condition*, checked at the two extreme
        coalitions — it is exact for the case that motivated it (a globally
        dead weight column, inert at every coalition), and it always
        preserves the efficiency axiom (see the two-anchor note below), but
        a hypothetical player inert at both anchors yet live in the interior
        would have its (non-zero) contribution absorbed by the other
        players. No such player has been observed; the alternative — leaving
        a provably dead column in an under-sampled regression — is the
        failure this guard exists to remove.

        **Guard 1 — input-degeneracy (specs/63).** A target endpoint's
        novelty toggle is degenerate iff its native dim-1 value already
        equals the value :meth:`build_masked_node_states` forces onto the
        "absent" state (always 0.0) — i.e. iff the native value is 0.0.
        "Present" and "absent" coalition inputs are then bit-identical.
        Costs zero forward passes.

        **Guard 2 — output-dummy (specs/65).** Strictly stronger, and
        catches a class guard 1 cannot by construction: a novelty toggle
        whose *input* genuinely varies (the endpoint really is novel, dim 1
        really flips 1 -> 0) but which the *model* cannot respond to at all.
        specs/65 sec 1 records the measurement on all four trained
        checkpoints — Adam's ``weight_decay`` drives column 1 of
        ``convs.0.fc_self/fc_neigh`` to 1e-24..1e-40 because dim 1 is
        constant-zero throughout training (specs/59 sec 4.4), leaving the
        trained model bit-invariant to it. The player is then a dummy player
        in the game-theoretic sense, but KernelSHAP still fits regression
        noise onto its column whenever the coalition is too wide to
        enumerate (``2**P > nsamples``).

        Only players that survive guard 1 are screened by guard 2, so the
        two counts the caller reports stay disjoint and separately
        auditable: an input-degenerate player is trivially output-dummy
        too, and folding it into the guard-2 count would destroy the
        meaning of ``2 - n_degenerate_novelty_players`` as this flow's
        novel-endpoint count.

        Two probe anchors, not one: a player is declared dummy only if
        toggling it moves the output at NEITHER the all-absent nor the
        all-present coalition. A single-anchor screen would be cheaper by
        one forward pass per screened player but could drop a player that
        merely happens to be inert at that one anchor. Requiring both
        anchors can only ever drop FEWER players, so it cannot over-broaden
        the guard, and it remains exact for the case that motivated it — a
        globally dead weight column, inert everywhere.

        These two anchors are not an arbitrary conservative choice: they are
        exactly the two coalitions that determine the caller's efficiency
        baselines. ``_expand_row`` pins a dropped column to 1, so
        ``f_logit`` is read at the all-present coalition and ``f_baseline``
        at the all-absent one with the dropped column held present. Probing
        the all-present anchor is what makes ``f_logit`` exact; probing the
        all-absent anchor is what makes ``f_baseline`` exact. So
        ``sum(phi) == f_logit - f_baseline`` survives a guard-2 drop even if
        the player were live somewhere in the coalition interior — only the
        OTHER players' individual phi would then be perturbed, never the
        sum. A single-anchor screen at all-present would leave
        ``f_baseline`` unjustified.

        One precision, since each probe is a SINGLE-column flip. ``f_logit``
        is exact unconditionally: ``_expand_row``'s fill of 1 makes the
        foreground row literally the all-present anchor. ``f_baseline`` is
        established exactly by the all-absent probe whenever at most one
        column is dropped by guard 2 — a guard-1 column is input-inert in
        every combination, so it composes freely. If BOTH columns are dropped
        by guard 2 and ``M > 0``, the probes pin each flip singly but not the
        pair jointly, so ``f_baseline``'s exactness there rests on the same
        globally-inert premise as the drop itself. At ``M == 0`` the pair is
        pinned as well, because the anchors and probes then cover the whole
        4-point coalition space (see the ``reduced_size == 0`` branch).
        Either way ``sum(phi) == f_logit - f_baseline`` holds against the
        REPORTED baselines: KernelSHAP's efficiency constraint is enforced on
        the reduced game the solver was actually handed.

        That is not merely defensive. A concrete case exists in this very
        masking function: a **self-loop** flow, ``src_nid == dst_nid``.
        :meth:`build_masked_node_states` dispatches on ``if``/``elif``, so
        the source branch shadows the destination branch on the shared row.
        At the all-absent anchor, flipping EITHER novelty column leaves the
        input untouched (the src branch has already zeroed dim 1), making a
        genuinely live player look inert. Only the all-present anchor
        distinguishes them. A single-anchor screen at all-absent would
        silently null out both novelty players on every self-loop flow.

        Exact equality, deliberately: the two probes are the same model in
        eval mode on the same device, differing in one input scalar, so a
        genuinely dead column reproduces bit-identical logits. A small but
        genuinely live response is left in the solver, matching specs/63's
        ``== 0.0`` exactness philosophy — never drop a player on a
        tolerance.

        Cost: 2 anchor passes plus up to 2 probe passes per screened
        player, so 0 forward passes when both players are input-degenerate,
        4 when one is screened, and at most 6 when both are — against a
        node-granularity budget of ``nsamples`` (2048 in the shipped
        config; this method's caller defaults to 512), i.e. at most 0.3%.
        This runs no second KernelSHAP solve.

        Args:
            forward_full:    callable mapping a FULL-width (coalition_size,)
                             binary coalition row to the model's logit for
                             the true class.
            base_node_feats: float32 (N_in, node_state_dim) actual node
                             states; row j corresponds to input_node_ids[j].
            input_node_ids:  int array (N_in,) of global node IDs.
            src_nid:         global node ID of the target edge's source.
            dst_nid:         global node ID of the target edge's destination.
            coalition_size:  full conceptual coalition width, 2 + M.

        Returns:
            ``(src_degenerate, dst_degenerate, src_dummy, dst_dummy)``.
            ``*_dummy`` is False whenever the matching ``*_degenerate`` is
            True — the two are disjoint by construction.
        """
        src_row_idx = next(
            j for j, nid in enumerate(input_node_ids) if int(nid) == src_nid
        )
        dst_row_idx = next(
            j for j, nid in enumerate(input_node_ids) if int(nid) == dst_nid
        )
        src_degenerate = bool(base_node_feats[src_row_idx, _NOVELTY_DIM] == 0.0)
        dst_degenerate = bool(base_node_feats[dst_row_idx, _NOVELTY_DIM] == 0.0)

        probe_cols: list[int] = []
        if not src_degenerate:
            probe_cols.append(0)
        if not dst_degenerate:
            probe_cols.append(1)

        src_dummy = False
        dst_dummy = False
        if probe_cols:
            anchors = (
                np.ones(coalition_size, dtype=np.float32),
                np.zeros(coalition_size, dtype=np.float32),
            )
            anchor_refs = [forward_full(a) for a in anchors]
            for col in probe_cols:
                inert_everywhere = True
                for anchor, ref in zip(anchors, anchor_refs):
                    probe = anchor.copy()
                    probe[col] = 1.0 - probe[col]
                    if forward_full(probe) != ref:
                        inert_everywhere = False
                        break
                if inert_everywhere:
                    if col == 0:
                        src_dummy = True
                    else:
                        dst_dummy = True

        return src_degenerate, dst_degenerate, src_dummy, dst_dummy

    def explain(
        self,
        true_class: int,
        src_nid: int,
        dst_nid: int,
        model: "EdgeAwareGraphSAGE",
        blocks: list,
        input_nodes: torch.Tensor,
        base_node_feats: np.ndarray,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        x_e: np.ndarray,
        nsamples: int = 512,
    ) -> dict:
        """Run KernelSHAP over node novelty and non-target node states.

        Args:
            true_class:      integer class label of the target edge.
            src_nid:         global node ID of the target edge's source.
            dst_nid:         global node ID of the target edge's destination.
            model:           trained EdgeAwareGraphSAGE (eval mode expected).
            blocks:          DGL computation blocks from the sampler.
            input_nodes:     global node IDs for blocks[0], shape (N_in,).
            base_node_feats: float32 array (N_in, 15) — actual node states.
            src_pos:         int64 tensor — seed edge source positions into h.
            dst_pos:         int64 tensor — seed edge destination positions into h.
            x_e:             float32 array (d_e,) — target edge features.
            nsamples:        KernelSHAP coalition samples (default 512).

        Returns:
            dict with keys:
              'src_novelty_shap': float
              'dst_novelty_shap': float
              'node_shap': dict {node_id_str: φ} for non-target nodes
              'coalition_size': int (|V_sub| + 2), full conceptual width —
                  unaffected by either guard below
              'n_degenerate_novelty_players': int, 0-2, count of target
                  novelty toggles dropped from the solver's coalition matrix
                  because their "absent" and "present" *inputs* are bit-
                  identical (specs/63). Meaning unchanged since specs/63:
                  ``2 - n_degenerate_novelty_players`` remains the count of
                  novel endpoints on this flow.
              'n_dummy_novelty_players': int, 0-2, count of target novelty
                  toggles dropped because their input genuinely varies but
                  the MODEL OUTPUT does not respond (specs/65). Disjoint
                  from n_degenerate_novelty_players by construction — a
                  player is screened for output-dummyness only if it passed
                  the input-degeneracy check. Total dropped is the sum.
        """
        import shap

        model.eval()
        input_node_ids = input_nodes.cpu().numpy()
        x_e_t = torch.tensor(x_e, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Coalition layout:
        #   index 0 → src novelty (target src)
        #   index 1 → dst novelty (target dst)
        #   index 2..2+M-1 → non-target node states
        non_target_ids: list[int] = build_non_target_ids(input_node_ids, src_nid, dst_nid)
        M = len(non_target_ids)
        coalition_size = 2 + M

        non_target_pos: dict[int, int] = build_non_target_pos(non_target_ids)

        def _forward_full(full_row: np.ndarray) -> float:
            """Model logit for `true_class` under one FULL-width coalition row."""
            modified = self.build_masked_node_states(
                full_row, base_node_feats, input_node_ids,
                src_nid, dst_nid, non_target_pos, true_class,
            )
            nf_t = torch.tensor(modified, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logit = model(blocks, nf_t, x_e_t, src_pos, dst_pos)
            return float(logit[0, true_class].item())

        # Both dummy-player guards, in one place (specs/63 + specs/65).
        (
            src_degenerate,
            dst_degenerate,
            src_dummy,
            dst_dummy,
        ) = self.screen_novelty_players(
            forward_full=_forward_full,
            base_node_feats=base_node_feats,
            input_node_ids=input_node_ids,
            src_nid=src_nid,
            dst_nid=dst_nid,
            coalition_size=coalition_size,
        )
        n_degenerate_novelty_players = int(src_degenerate) + int(dst_degenerate)
        n_dummy_novelty_players = int(src_dummy) + int(dst_dummy)
        src_dropped = src_degenerate or src_dummy
        dst_dropped = dst_degenerate or dst_dummy

        # Full (conceptual, width coalition_size) -> reduced (solver-facing)
        # column index mapping. Dropped columns (either guard) are simply
        # absent from this mapping.
        full_to_reduced: dict[int, int] = {}
        reduced_col = 0
        if not src_dropped:
            full_to_reduced[0] = reduced_col
            reduced_col += 1
        if not dst_dropped:
            full_to_reduced[1] = reduced_col
            reduced_col += 1
        for i in range(M):
            full_to_reduced[2 + i] = reduced_col
            reduced_col += 1
        # == coalition_size - n_degenerate_novelty_players
        #                   - n_dummy_novelty_players
        reduced_size = reduced_col

        def _expand_row(reduced_row: np.ndarray) -> np.ndarray:
            """Reduced-width solver row -> full-width coalition row.

            Dropped columns are filled with a constant 1 (arbitrary but
            fixed): build_masked_node_states reads this as "present", which
            is a no-op for a dropped player by definition — an
            input-degenerate player's present and absent states produce the
            same masked input, and an output-dummy player's produce the same
            model output at both probe anchors, which is all `_predict_fn`
            returns. The fill value is what makes `f_logit` and `f_baseline`
            exactly the two coalitions the guard probed; see
            `screen_novelty_players` for why that keeps efficiency intact.
            """
            full_row = np.empty(coalition_size, dtype=np.float32)
            full_row[0] = 1.0 if src_dropped else reduced_row[full_to_reduced[0]]
            full_row[1] = 1.0 if dst_dropped else reduced_row[full_to_reduced[1]]
            for i in range(M):
                full_row[2 + i] = reduced_row[full_to_reduced[2 + i]]
            return full_row

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            return np.array(
                [_forward_full(_expand_row(r)) for r in coalition_matrix],
                dtype=np.float64,
            )

        if reduced_size == 0:
            # No non-target nodes, and both target novelty players dropped by
            # one guard or the other (degenerate, dummy, or one of each). The
            # single achievable coalition state is the only one there is, so
            # baseline and foreground logits coincide by construction: a
            # degenerate player's present/absent INPUTS are identical, and a
            # dummy player's present/absent OUTPUTS are, which is all
            # _predict_fn reports. Note this holds here without any
            # interior-inertness assumption: M == 0 makes coalition_size 2,
            # so the guard's 2 anchors + up to 4 probes already cover the
            # whole 4-point coalition space, and f(1,1) == f(0,0) chains out
            # of them directly.
            phi_reduced = np.array([])
            f_baseline = f_logit = float(
                _predict_fn(np.ones((1, 0), dtype=np.float32))[0]
            )
        else:
            background_data = np.zeros((1, reduced_size), dtype=np.float32)
            foreground_data = np.ones((1, reduced_size), dtype=np.float32)
            explainer = shap.KernelExplainer(_predict_fn, background_data)
            f_baseline = float(np.squeeze(explainer.expected_value))
            f_logit = float(_predict_fn(foreground_data)[0])
            phi_raw = explainer.shap_values(
                foreground_data,
                nsamples=nsamples,
                l1_reg=_L1_REG,
                silent=True,
            )
            if reduced_size == 1:
                phi_reduced = np.array([float(np.squeeze(phi_raw))])
            else:
                phi_reduced = np.array(phi_raw).squeeze()

        # Reassemble the full-width phi vector. Dropped columns (either
        # guard) are left at their np.zeros initialization — an exact 0.0,
        # not estimated.
        phi = np.zeros(coalition_size, dtype=np.float64)
        for full_idx, reduced_idx in full_to_reduced.items():
            phi[full_idx] = phi_reduced[reduced_idx]

        src_novelty_phi = float(phi[0])
        dst_novelty_phi = float(phi[1])
        node_phi = {str(nid): float(phi[2 + i]) for i, nid in enumerate(non_target_ids)}

        logger.debug(
            f"Node SHAP: class={true_class}, coalition_size={coalition_size}, "
            f"n_degenerate_novelty_players={n_degenerate_novelty_players}, "
            f"n_dummy_novelty_players={n_dummy_novelty_players}, "
            f"src_novelty_φ={src_novelty_phi:.4f}, dst_novelty_φ={dst_novelty_phi:.4f}, "
            f"efficiency_err={abs(phi.sum() - (f_logit - f_baseline)):.4f}"
        )

        return {
            "src_novelty_shap": src_novelty_phi,
            "dst_novelty_shap": dst_novelty_phi,
            "node_shap": node_phi,
            "coalition_size": coalition_size,
            "n_degenerate_novelty_players": n_degenerate_novelty_players,
            "n_dummy_novelty_players": n_dummy_novelty_players,
            "f_baseline": f_baseline,
            "f_logit": f_logit,
        }
