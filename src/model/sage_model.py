"""
Parameterized EdgeAware GraphSAGE for edge classification.

Input:
  node_features: (num_input_nodes, node_state_dim=15)
  edge_features: (num_seed_edges, d_e)

Architecture:
  num_layers SAGEConv blocks: SAGEConv(in→hidden) + BatchNorm1d + ReLU + Dropout
  Edge MLP: Linear(2*hidden + d_e, hidden) → ReLU → Dropout → Linear(hidden, num_classes)

Node features are passed as tensors to forward(); they are NOT stored in g.ndata.
This is required for SHAP coalition swapping (Phase 6).
"""

import logging
from typing import Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

logger = logging.getLogger(__name__)


class EdgeAwareGraphSAGE(nn.Module):
    """Two-level GraphSAGE with an edge MLP for edge classification.

    After running SAGEConv blocks to obtain per-node embeddings, the edge MLP
    concatenates [h_src, h_dst, edge_feats] and produces per-class logits.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.4,
        aggregator: str = "mean",
    ) -> None:
        """
        Args:
            node_in_dim:  node feature dimensionality (15 for SHAP-GSD).
            edge_in_dim:  edge feature dimensionality (d_e, e.g. 218).
            hidden_size:  width of all hidden layers.
            num_classes:  number of output classes.
            num_layers:   number of SAGEConv + BN + ReLU blocks (default 2).
            dropout:      dropout probability used in both GNN and edge MLP.
            aggregator:   SAGEConv aggregator type ("mean", "gcn", "pool", "lstm").
        """
        super().__init__()
        self.node_in_dim  = node_in_dim
        self.edge_in_dim  = edge_in_dim
        self.hidden_size  = hidden_size
        self.num_classes  = num_classes
        self.num_layers   = num_layers
        self.dropout_p    = dropout

        # GNN layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_in_dim if i == 0 else hidden_size
            self.convs.append(SAGEConv(in_dim, hidden_size, aggregator))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        # Edge classification head
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + edge_in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self._init_weights()
        logger.info(
            f"EdgeAwareGraphSAGE: {num_layers} layers, hidden={hidden_size}, "
            f"d_e={edge_in_dim}, classes={num_classes}, dropout={dropout}, "
            f"aggregator={aggregator}"
        )

    def _init_weights(self) -> None:
        """Xavier-uniform init for linear layers; BN default (weight=1, bias=0)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def encode(
        self,
        blocks: list[dgl.DGLGraph],
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Run GNN layers to obtain per-node embeddings for seed nodes.

        Args:
            blocks:     list of DGL bipartite blocks (one per GNN layer),
                        ordered from deepest hop (blocks[0]) to shallowest (blocks[-1]).
            node_feats: float32 tensor (num_input_nodes, node_in_dim).

        Returns:
            h: float32 tensor (num_seed_nodes, hidden_size).
               Seed nodes are blocks[-1].dstdata[dgl.NID].
        """
        h = node_feats
        for conv, bn, block in zip(self.convs, self.bns, blocks):
            h = conv(block, h)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h

    def classify(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Apply edge MLP to produce per-class logits.

        Args:
            h:          node embeddings (num_seed_nodes, hidden_size).
            edge_feats: float32 tensor (num_seed_edges, edge_in_dim).
            src_pos:    int64 indices into h for seed edge sources (num_seed_edges,).
            dst_pos:    int64 indices into h for seed edge destinations (num_seed_edges,).

        Returns:
            logits: float32 tensor (num_seed_edges, num_classes).
        """
        h_src = h[src_pos]
        h_dst = h[dst_pos]
        x = torch.cat([h_src, h_dst, edge_feats], dim=1)
        return self.edge_mlp(x)

    def forward(
        self,
        blocks: list[dgl.DGLGraph],
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end: encode nodes then classify edges.

        Args:
            blocks:     computation blocks from TemporalNeighborSampler.
            node_feats: (num_input_nodes, node_in_dim).
            edge_feats: (num_seed_edges, edge_in_dim).
            src_pos:    int64 indices into final h for each seed edge's source.
            dst_pos:    int64 indices into final h for each seed edge's destination.

        Returns:
            logits: (num_seed_edges, num_classes).
        """
        h = self.encode(blocks, node_feats)
        return self.classify(h, edge_feats, src_pos, dst_pos)


def build_src_dst_pos(
    g: dgl.DGLGraph,
    seed_eids: torch.Tensor,
    seed_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute src_pos and dst_pos for seed edges into the seed_nodes array.

    Args:
        g:          the split graph (used to resolve edge endpoints).
        seed_eids:  global edge IDs for the current mini-batch.
        seed_nodes: global node IDs of seed nodes, i.e. blocks[-1].dstdata[dgl.NID].

    Returns:
        (src_pos, dst_pos): int64 tensors of shape (len(seed_eids),) giving the
        position of each edge's source and destination in seed_nodes.
    """
    src_ids, dst_ids = g.find_edges(seed_eids)

    # Build mapping from global NID to position in seed_nodes
    nid_to_pos: dict[int, int] = {
        int(nid): pos for pos, nid in enumerate(seed_nodes.tolist())
    }

    src_pos = torch.tensor(
        [nid_to_pos[int(s)] for s in src_ids.tolist()], dtype=torch.long
    )
    dst_pos = torch.tensor(
        [nid_to_pos[int(d)] for d in dst_ids.tolist()], dtype=torch.long
    )
    return src_pos, dst_pos
