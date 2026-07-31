"""Unit tests for the two opt-in Phase-3 training perf flags:
``compute.fast_index_loader`` and ``compute.cpu_src_dst_pos``.

Both flags are default-off pure performance changes — they must never
change what gets computed, only how/where an equivalent computation
happens. These tests check exactly that equivalence property, using small
synthetic fixtures (no full Trainer construction, no real dataset), in the
same spirit as ``test_batch_timer.py``:

- ``_index_batches``: with the flag off, behavior must be byte-for-byte
  identical to the pre-existing ``DataLoader(TensorDataset(...))`` path
  (regression safety). With the flag on, batch contents/order/drop_last
  must be identical to the flag-off path (the actual correctness property).
- ``build_src_dst_pos``: computing it from CPU-resident node IDs (as
  ``cpu_src_dst_pos=True`` does, before ``blocks.to(device)``) must give
  identical ``(src_pos, dst_pos)`` tensors to computing it from the same
  node IDs after a device round-trip (the ``cpu_src_dst_pos=False`` path).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dgl

from src.model.sage_model import build_src_dst_pos
from src.model.temporal_sampler import TemporalNeighborSampler
from src.model.trainer import Trainer


# ---------------------------------------------------------------------------
# _index_batches
# ---------------------------------------------------------------------------

def _make_fake_trainer(
    fast_index_loader: bool,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> SimpleNamespace:
    """Build a minimal stand-in exposing exactly what ``_index_batches`` reads.

    ``_index_batches`` only touches ``self.fast_index_loader``,
    ``self.batch_size``, ``self.num_workers`` and ``self.pin_memory`` — so a
    full ``Trainer`` (model + graphs + feature stores + node-state manager)
    is unnecessary to exercise it in isolation.
    """
    return SimpleNamespace(
        fast_index_loader=fast_index_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _reference_dataloader_batches(local_eids: np.ndarray, batch_size: int) -> list[torch.Tensor]:
    """The pre-existing DataLoader(TensorDataset(...)) path, called directly."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(local_eids)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    return [batch_local_t.long() for (batch_local_t,) in loader]


@pytest.mark.parametrize("n_eids,batch_size", [
    (0, 8),      # empty input
    (1, 8),      # single element, smaller than one batch
    (8, 8),      # exactly one full batch
    (17, 8),     # multiple full batches + a short final batch (drop_last=False)
    (100, 32),   # larger, non-divisible
])
def test_index_batches_off_matches_dataloader_exactly(n_eids: int, batch_size: int) -> None:
    """flag off: _index_batches must reproduce the DataLoader path exactly."""
    local_eids = np.arange(n_eids, dtype=np.int64)
    fake_self = _make_fake_trainer(fast_index_loader=False, batch_size=batch_size)

    got = list(Trainer._index_batches(fake_self, local_eids))
    want = _reference_dataloader_batches(local_eids, batch_size)

    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype == torch.int64
        assert torch.equal(g, w)


@pytest.mark.parametrize("n_eids,batch_size", [
    (0, 8),
    (1, 8),
    (8, 8),
    (17, 8),
    (100, 32),
])
def test_index_batches_on_matches_off(n_eids: int, batch_size: int) -> None:
    """flag on: batch contents/order/drop_last must match the flag-off path."""
    local_eids = np.arange(n_eids, dtype=np.int64)

    fake_self_off = _make_fake_trainer(fast_index_loader=False, batch_size=batch_size)
    fake_self_on = _make_fake_trainer(fast_index_loader=True, batch_size=batch_size)

    batches_off = list(Trainer._index_batches(fake_self_off, local_eids))
    batches_on = list(Trainer._index_batches(fake_self_on, local_eids))

    assert len(batches_off) == len(batches_on)
    for off, on in zip(batches_off, batches_on):
        assert off.dtype == on.dtype == torch.int64
        assert torch.equal(off, on)


def test_index_batches_on_preserves_original_eid_values() -> None:
    """The fast path must yield the exact EIDs in local_eids, not their positions."""
    # Non-contiguous, non-trivial EID values (as balanced/sorted EIDs would be).
    local_eids = np.array([3, 7, 11, 42, 100, 101, 999], dtype=np.int64)
    fake_self = _make_fake_trainer(fast_index_loader=True, batch_size=3)

    got = torch.cat(list(Trainer._index_batches(fake_self, local_eids)))
    assert torch.equal(got, torch.from_numpy(local_eids).long())


def test_index_batches_drop_last_is_always_false() -> None:
    """The final short batch must never be dropped, for either implementation."""
    local_eids = np.arange(10, dtype=np.int64)  # batch_size=4 -> batches of 4,4,2
    for flag in (False, True):
        fake_self = _make_fake_trainer(fast_index_loader=flag, batch_size=4)
        batches = list(Trainer._index_batches(fake_self, local_eids))
        sizes = [b.shape[0] for b in batches]
        assert sizes == [4, 4, 2], f"flag={flag}: got batch sizes {sizes}"
        assert torch.equal(torch.cat(batches), torch.from_numpy(local_eids).long())


def test_index_batches_fast_path_does_not_alias_source_array() -> None:
    """Yielded fast-path batches must be independent copies, not views.

    ``train()`` computes ``balanced_sorted`` once and passes the SAME numpy
    array into ``_run_epoch`` on every epoch. A bare slice of
    ``torch.from_numpy(local_eids)`` shares storage with that array; if a
    yielded batch were ever mutated in place downstream, every subsequent
    epoch's batching would silently corrupt. This is the same failure class
    (buffer-reuse aliasing) that made the rejected pinned-memory variant
    silently produce wrong losses — verify the fast path is immune by
    construction, not by absence-of-evidence in sample_blocks today.
    """
    local_eids = np.arange(20, dtype=np.int64)
    original = local_eids.copy()
    fake_self = _make_fake_trainer(fast_index_loader=True, batch_size=6)

    batches = list(Trainer._index_batches(fake_self, local_eids))
    for b in batches:
        b += 999_999  # mutate the yielded tensor in place

    # The source array must be untouched by mutating any yielded batch.
    assert np.array_equal(local_eids, original)
    # And a second, independent pass over the same source must be unaffected.
    fresh_batches = list(Trainer._index_batches(fake_self, local_eids))
    assert torch.equal(torch.cat(fresh_batches), torch.from_numpy(original).long())


# ---------------------------------------------------------------------------
# cpu_src_dst_pos: build_src_dst_pos CPU-before-transfer vs post-transfer
# ---------------------------------------------------------------------------

def _make_synthetic_block_graph(seed: int = 0) -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """Small synthetic graph + a one-hop block, mirroring the trainer's usage.

    Returns (g, seed_eids, seed_nodes) where seed_nodes matches what
    ``blocks[-1].dstdata[dgl.NID]`` would contain in the real training loop.
    """
    rng = np.random.default_rng(seed)
    n_nodes = 20
    n_edges = 60

    src = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    dst = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    same = src == dst
    dst[same] = (dst[same] + 1) % n_nodes

    g = dgl.graph((src, dst), num_nodes=n_nodes)

    # A handful of seed edges. seed_nodes mirrors blocks[-1].dstdata[dgl.NID]
    # in the real training loop, which (as an input-nodes union produced by
    # dgl.to_block) contains both endpoints of every seed edge, not only the
    # destinations — reproduce that here via the union of src and dst ids.
    seed_eids = torch.tensor([0, 5, 12, 30, 45], dtype=torch.long)
    endpoint_ids = np.concatenate([src[seed_eids.numpy()], dst[seed_eids.numpy()]])
    seed_nodes = torch.tensor(np.unique(endpoint_ids), dtype=torch.long)
    return g, seed_eids, seed_nodes


def test_build_src_dst_pos_cpu_matches_post_transfer_cpu_only() -> None:
    """On CPU: computing before vs after an (identity) transfer is identical."""
    g, seed_eids, seed_nodes = _make_synthetic_block_graph()

    # "cpu_src_dst_pos=True" path: compute directly from the pre-transfer tensor.
    src_pos_pre, dst_pos_pre = build_src_dst_pos(g, seed_eids, seed_nodes)

    # "cpu_src_dst_pos=False" path: simulate the round trip via .to("cpu") (a
    # no-op device-wise here, but exercises the same code path/tensor identity
    # semantics as blocks[-1].dstdata[dgl.NID] after blocks[i].to(device)).
    seed_nodes_after = seed_nodes.to("cpu")
    src_pos_post, dst_pos_post = build_src_dst_pos(g, seed_eids, seed_nodes_after)

    assert torch.equal(src_pos_pre, src_pos_post)
    assert torch.equal(dst_pos_pre, dst_pos_post)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_build_src_dst_pos_cpu_before_matches_gpu_after_transfer() -> None:
    """Compute pre-transfer (CPU) vs post-transfer (GPU round trip): identical result."""
    g, seed_eids, seed_nodes = _make_synthetic_block_graph()

    # cpu_src_dst_pos=True: computed from the CPU tensor before any device move.
    src_pos_cpu, dst_pos_cpu = build_src_dst_pos(g, seed_eids, seed_nodes)

    # cpu_src_dst_pos=False: node IDs travel to the GPU and back (as
    # blocks[-1].dstdata[dgl.NID] would after `blocks = [b.to(device) ...]`),
    # then build_src_dst_pos is called on the GPU-resident tensor.
    seed_nodes_gpu = seed_nodes.to("cuda")
    src_pos_gpu, dst_pos_gpu = build_src_dst_pos(g, seed_eids, seed_nodes_gpu)

    assert torch.equal(src_pos_cpu, src_pos_gpu.cpu())
    assert torch.equal(dst_pos_cpu, dst_pos_gpu.cpu())


# ---------------------------------------------------------------------------
# End-to-end: real TemporalNeighborSampler blocks through both flags together
# ---------------------------------------------------------------------------

def _make_synthetic_timestamped_graph(n_nodes: int = 60, n_edges: int = 500,
                                       seed: int = 0) -> dgl.DGLGraph:
    """Chronologically-ordered synthetic graph (mirrors test_temporal_sampler.py)."""
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(1000, 1_000_000, size=n_edges)).astype(np.int64)
    src = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    dst = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    same = src == dst
    dst[same] = (dst[same] + 1) % n_nodes

    g = dgl.graph((src, dst), num_nodes=n_nodes)
    g.edata["timestamp"] = torch.tensor(ts, dtype=torch.int64)
    return g


@pytest.mark.parametrize("fast_index_loader", [False, True])
def test_cpu_src_dst_pos_matches_real_blocks_from_sample_blocks(fast_index_loader: bool) -> None:
    """End-to-end: batches from _index_batches feed real sample_blocks output
    into build_src_dst_pos; the CPU-before-transfer result must match the
    post-transfer (device round-trip) result, batch for batch — this is the
    actual mechanism cpu_src_dst_pos changes, exercised together with
    fast_index_loader in both settings (they are independent flags but are
    used together in practice).
    """
    g = _make_synthetic_timestamped_graph(n_nodes=60, n_edges=500, seed=1)
    local_eids = np.arange(g.num_edges(), dtype=np.int64)
    sampler = TemporalNeighborSampler(fanouts=[10, 5])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fake_self = _make_fake_trainer(fast_index_loader=fast_index_loader, batch_size=64)

    n_batches_checked = 0
    for batch_local in Trainer._index_batches(fake_self, local_eids):
        _, seed_local, blocks = sampler.sample_blocks(g, batch_local)

        # cpu_src_dst_pos=True: computed from the still-CPU block, before
        # any device transfer.
        src_pos_pre, dst_pos_pre = build_src_dst_pos(
            g, seed_local, blocks[-1].dstdata[dgl.NID]
        )

        # cpu_src_dst_pos=False: blocks move to device first (as
        # `blocks = [b.to(self.device) for b in blocks]` does in the real
        # loop), then build_src_dst_pos runs on the (possibly GPU-resident)
        # post-transfer node IDs.
        blocks_on_device = [b.to(device) for b in blocks]
        src_pos_post, dst_pos_post = build_src_dst_pos(
            g, seed_local, blocks_on_device[-1].dstdata[dgl.NID]
        )

        assert torch.equal(src_pos_pre, src_pos_post.cpu())
        assert torch.equal(dst_pos_pre, dst_pos_post.cpu())
        n_batches_checked += 1

    assert n_batches_checked > 0
