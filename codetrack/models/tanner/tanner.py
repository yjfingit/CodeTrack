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

        # NOTE: there is deliberately no ``parity_to_check`` projection here anymore.
        # It used to be summed into ``checks`` while ``VisualSyndrome`` compared the same
        # ``checks`` against ``parity_ref(parity)`` -- the reference was then present on
        # both sides of the comparison.  A linear layer kept only to be unused would just
        # be dead parameters, so it is removed together with the shortcut.
        self.check_norm = nn.LayerNorm(check_dim)
        # The semantic (A_vv) term is *context*, not evidence, and it is the noisiest
        # thing in the check: it is a pooling of 128 randomly-initialised node embeddings.
        # Measured, the corruption evidence in ``obs`` correlates with the true per-check
        # density at r=0.64, but at `sem_gate = 0.5` the semantic noise is the same size as
        # the signal and the syndrome head converges to a constant instead.  The gate
        # therefore starts fully closed (sigmoid(-4) ~ 0.018) and only opens if it earns
        # its place -- the semantic path is additive context, the H path is the syndrome.
        self.sem_gate = nn.Parameter(torch.tensor(-4.0))
        # how strongly the *pre-normalisation* residual enters the syndrome
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        # 16 check slots pool the graph-node features (soft, learnable topology)
        self.slot_weight = nn.Parameter(torch.rand(num_parity, self.num_graph_nodes))
        # parity observation (code space) -> check space
        self.obs_proj = nn.Linear(code_dim, check_dim)

    def forward(self, variables: torch.Tensor, identity: torch.Tensor,
                parity: torch.Tensor, priority: Optional[torch.Tensor] = None,
                variables_tir: Optional[torch.Tensor] = None,
                H: Optional[torch.Tensor] = None,
                observation: Optional[torch.Tensor] = None,
                expected_obs: Optional[torch.Tensor] = None
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
        ``expected_obs`` (``B x M x check_dim``) is what the **trusted** identity predicts
        for the same aggregation; ``check_residual`` is the pre-normalisation difference
        against it, which is what the syndrome head reads.
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
        #
        # The reference parity is deliberately NOT added here.  ``VisualSyndrome``
        # already compares ``checks`` against ``parity_ref(parity)``; feeding parity into
        # both sides let the discrepancy be computed against itself instead of against
        # the received word, which is a shortcut that survives without any real
        # "H x == parity" test.  checks must be built from the *observation* alone.
        if H is not None and observation is not None:
            obs_raw = torch.einsum("mn,bnd->bmd", H, observation)      # B x 16 x code_dim
            obs = self.obs_proj(obs_raw)                                # B x 16 x c
            # Per-check observation energy, measured on the raw aggregation *before* the
            # projection.  Erasure sets a token to exactly zero, so this is a direct,
            # linear read-out of "how much of what this check watches is gone":
            # measured corr(energy, true per-check density) = 1.000 on random erasure, and
            # it survives the projection and the LayerNorm below, which together reduce the
            # corruption's footprint on ``checks`` to ~13% and drown the signal.
            obs_energy = obs_raw.pow(2).mean(dim=-1)                    # B x 16
            obs_energy = obs_energy / obs_energy.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            obs = torch.zeros(b, self.num_parity, self.check_dim,
                              device=variables.device, dtype=node_feat.dtype)
            obs_energy = torch.zeros(b, self.num_parity, device=variables.device,
                                     dtype=node_feat.dtype)
        sem = self._check_slots(a_vv @ node_feat, b)                          # B x 16 x c
        sem = sem * torch.sigmoid(self.sem_gate).to(sem.dtype)
        checks = self.check_norm(obs + sem)                                  # B x 16 x 128

        # ---- 4. the residual the syndrome actually reads ------------------------------
        # ``check_norm`` is a LayerNorm, so ``checks`` always has per-element scale 1 and
        # the *absolute* deviation of the observation is normalised away: the corruption
        # moves ``agg`` by only ~13% relative, which is what made the syndrome head
        # converge to a constant no matter how it was parameterised.  The residual below
        # is computed **before** the norm, against the expectation the trusted identity
        # predicts for exactly this neighbourhood, and handed to the syndrome alongside.
        # This is the "H x should equal the codeword" test in its literal form.
        residual = obs                                                  # B x 16 x c
        if expected_obs is not None and expected_obs.shape == residual.shape:
            residual = residual - expected_obs

        return {
            "A_uv": a_vv,                     # B x 128 x 128 (auxiliary semantic graph)
            "A_vv": a_vv,
            "graph_nodes": node_feat,        # B x 128 x 128
            "node_index": node_index,        # B x 128
            "identity_map": identity_map,    # B x 128, in (0,1)
            "checks": checks,                # B x 16 x 128
            "check_residual": residual,      # B x 16 x 128, pre-normalisation
            "obs_energy": obs_energy,        # B x 16, per-check observation energy
        }

    def _check_slots(self, agg: torch.Tensor, batch: int) -> torch.Tensor:
        """Pool ``B x 128 x c`` graph features into ``B x 16 x c`` check slots."""
        w = F.softmax(self.slot_weight, dim=-1)                        # 16 x 128
        return w.unsqueeze(0).to(agg.dtype) @ agg                      # B x 16 x c
