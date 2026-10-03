"""Reliability-Adaptive Tanner Graph -- architecture figure block 4.

    Dynamic Graph Construction
        A_uv = Softmax_s(U U^T / sqrt(d))          B x 128 x 128
    Variable Nodes                                 B x 256 x 768
    Check Nodes                                    B x  16 x 128
    Identity Map                                   B x 128

Shape bookkeeping that ties this module to the figure
-----------------------------------------------------
* 256 variable nodes come from the target candidate selector (256 = 16x16 search grid).
* The selector keeps the **best 128** of them as *graph nodes*; their pairwise
  affinity ``A_uv`` is therefore ``B x 128 x 128`` -- exactly the figure's shape.
* The 16 parity tokens of the codebook become the 16 check nodes; each check
  aggregates its neighbourhood and is embedded to 128 dims (``B x 16 x 128``).
* ``Identity Map`` is the per-graph-node "is this the target" score derived from the
  codebook, used later by the error locator.

``Softmax_s`` denotes a **sparsified softmax**: only the ``top_k`` affinities per row
survive before renormalisation.  Sparsity is what turns a syndrome into an
*address*: a handful of failing checks points at a specific node.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsified_softmax(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Softmax that keeps only the ``top_k`` logits per row (``Softmax_s``)."""
    if top_k >= logits.shape[-1]:
        return logits.softmax(dim=-1)
    kth = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
    masked = logits.masked_fill(logits < kth, float("-inf"))
    return masked.softmax(dim=-1)


class AdaptiveTannerGraph(nn.Module):
    """Build the dynamic graph and the variable/check node representations."""

    def __init__(self, dim: int = 768, num_identity: int = 16, num_parity: int = 16,
                 num_variables: int = 256, num_graph_nodes: int = 128,
                 check_dim: int = 128, top_k: int = 8, heads: int = 4,
                 code_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.num_parity = num_parity
        self.num_variables = num_variables
        self.num_graph_nodes = min(num_graph_nodes, num_variables)
        self.check_dim = check_dim
        self.top_k = top_k

        # variable -> graph node embedding, fed by the identity map
        self.node_embed = nn.Linear(dim, check_dim)
        self.identity_map = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1),
        )

        # the graph is built from the codebook (U) expanded to graph-node resolution
        self.code_to_node = nn.Linear(code_dim, check_dim)
        self.heads = heads
        self.q_proj = nn.Linear(check_dim, check_dim)
        self.k_proj = nn.Linear(check_dim, check_dim)

        # check node embedding: codebook parity -> check space
        self.parity_to_check = nn.Linear(code_dim, check_dim)
        self.check_norm = nn.LayerNorm(check_dim)

        # 16 check slots pool the graph-node features (soft, learnable topology)
        self.slot_weight = nn.Parameter(torch.rand(num_parity, self.num_graph_nodes))
        # parity observation (code space) -> check space
        self.obs_proj = nn.Linear(code_dim, check_dim)

    def forward(self, variables: torch.Tensor, identity: torch.Tensor,
                parity: torch.Tensor, priority: Optional[torch.Tensor] = None,
                variables_tir: Optional[torch.Tensor] = None,
                H: Optional[torch.Tensor] = None,
                observation: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """``variables``: ``B x 256 x 768``; ``identity``/``parity``: ``B x 16 x 256``.

        ``H`` is the **shared parity-check matrix** ``M x N`` owned by the decoder, and
        ``observation`` is the code-space view of the variables (``B x N x code_dim``).
        Together they decide the check nodes: the check is what ``H`` says the codeword
        should look like over its neighbourhood, and the syndrome is how far the actual
        observation is from it.

        ``priority`` (``B x 256``, from the target candidate selector) is added to the
        internal identity score when choosing which nodes become graph nodes.
        ``variables_tir`` makes the graph react to corruption in **either** modality.
        """
        b, n, d = variables.shape

        # ---- 1. top-k graph nodes + identity map -----------------------------------
        id_logits = self.identity_map(variables).squeeze(-1)            # B x 256
        if variables_tir is not None:
            id_logits = 0.5 * (id_logits + self.identity_map(variables_tir).squeeze(-1))
        if priority is not None:
            id_logits = id_logits + priority
        k = min(self.num_graph_nodes, n)
        _, node_index = torch.topk(id_logits, k=k, dim=1)               # B x k
        gather_idx = node_index.unsqueeze(-1).expand(-1, -1, d)
        nodes = variables.gather(1, gather_idx)
        if variables_tir is not None:
            nodes = 0.5 * (nodes + variables_tir.gather(1, gather_idx))

        node_feat = self.node_embed(nodes)                             # B x 128 x check_dim
        # ranking uses the logit, but everything downstream treats this as a probability
        identity_map = torch.sigmoid(id_logits.gather(1, node_index))   # B x 128, in (0,1)

        # ---- 2. variable-variable semantic affinity A_vv (NOT the parity-check graph) --
        # Kept because it decides which tokens are worth watching, but the Tanner
        # structure below is defined by H alone.
        code_prior = self.code_to_node(parity.mean(dim=1, keepdim=True))       # B x 1 x c
        code_prior = code_prior + self.code_to_node(identity.mean(dim=1, keepdim=True))

        q = self.q_proj(node_feat)
        kk = self.k_proj(node_feat + code_prior)
        scale = (self.check_dim // self.heads) ** -0.5
        logits = (q @ kk.transpose(-2, -1)) * scale                    # B x 128 x 128
        a_vv = sparsified_softmax(logits, self.top_k)                   # B x 128 x 128

        # ---- 3. check nodes -------------------------------------------------------
        # Primary term: the parity observation aggregated through the SHARED H, i.e. what
        # the code says the checks should see.  The A_vv term adds semantic context only;
        # it is an auxiliary variable-variable graph, not a second check matrix.
        if H is not None and observation is not None:
            obs = self.obs_proj(torch.einsum("mn,bnd->bmd", H, observation))  # B x 16 x c
        else:
            obs = torch.zeros(b, self.num_parity, self.check_dim,
                              device=variables.device, dtype=node_feat.dtype)
        sem = self._check_slots(a_vv @ node_feat, b)                          # B x 16 x c
        checks = self.check_norm(obs + sem + self.parity_to_check(parity))    # B x 16 x 128

        return {
            "A_uv": a_vv,                     # B x 128 x 128 (auxiliary semantic graph)
            "A_vv": a_vv,
            "graph_nodes": node_feat,        # B x 128 x 128
            "node_index": node_index,        # B x 128
            "identity_map": identity_map,    # B x 128, in (0,1)
            "checks": checks,                # B x 16 x 128
        }

    def _check_slots(self, agg: torch.Tensor, batch: int) -> torch.Tensor:
        """Pool ``B x 128 x c`` graph features into ``B x 16 x c`` check slots."""
        w = F.softmax(self.slot_weight, dim=-1)                        # 16 x 128
        return w.unsqueeze(0).to(agg.dtype) @ agg                      # B x 16 x c
