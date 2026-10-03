"""CodeTrack -- full model assembly.

One forward pass follows the architecture figure stage by stage:

    Target Coding        Target Codebook Encoder  -> identity U, parity P
    Syndrome Checking    Reliability-Adaptive Tanner Graph -> checks, A_uv
                         Visual Syndrome -> S
    Error Localization   Error Locator -> locator, severity
    Iterative Correction Neural BP Decoder x n -> corrected / residual
                         Fusion + FPN -> feature map -> Tracking Head

All shapes are those annotated on the figure; see ``docs/architecture.md`` for the
module-by-module table.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import build_backbone, load_ostrack_pretrained
from .codebook import TargetCodebookEncoder
from .decoder import NeuralBPDecoder
from .fusion import CodeTrackFusion
from .head import build_head, load_head_weights
from .reliability import ReliabilityEstimator, TargetCandidateSelector
from .syndrome import ErrorLocator, ReliabilityAwareGating, VisualSyndrome
from .tanner import AdaptiveTannerGraph


class CodeTrack(nn.Module):
    """Structured redundancy and syndrome-guided correction for robust RGB-T tracking."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg
        backbone_section = model.get("backbone", {})
        backbone_cfg = backbone_section if isinstance(backbone_section, dict) else {}

        def get(key: str, default):
            if key in model:
                return model[key]
            return backbone_cfg.get(key, default)

        self.img_size = int(get("img_size", 256))
        self.template_size = int(get("template_size", 128))
        self.patch_size = int(get("patch_size", 16))
        self.grid = self.img_size // self.patch_size
        self.dim = int(get("embed_dim", 768))
        self.code_dim = int(get("code_dim", 256))
        self.check_dim = int(get("check_dim", 128))
        self.num_identity = int(get("num_identity_tokens", 16))
        self.num_parity = int(get("num_parity_tokens", 16))
        self.num_variables = int(get("num_variable_nodes", self.grid * self.grid))
        self.num_graph_nodes = int(get("num_graph_nodes", 128))
        self.return_stages = tuple(get("return_stages", (2, 5)))

        self.backbone = build_backbone({"backbone": {
            "img_size": self.img_size,
            "template_size": self.template_size,
            "patch_size": self.patch_size,
            "embed_dim": self.dim,
            "depth": int(get("depth", 12)),
            "num_heads": int(get("num_heads", 12)),
            "return_stages": self.return_stages,
            "freeze": bool(get("freeze_backbone", True)),
            "tir_from_rgb": bool(get("tir_from_rgb", True)),
        }})

        self.codebook = TargetCodebookEncoder(
            dim=self.dim, code_dim=self.code_dim,
            proj_dim=int(get("codebook_proj_dim", 512)),
            num_identity=self.num_identity, num_parity=self.num_parity,
            num_heads=int(get("codebook_heads", 8)),
            links_per_row=int(get("parity_links", 4)),
            dropout=float(get("dropout", 0.0)),
        )
        self.reliability = ReliabilityEstimator(dim=self.dim, num_identity=self.num_identity,
                                                code_dim=self.code_dim)
        self.selector = TargetCandidateSelector(dim=self.dim, num_identity=self.num_identity,
                                                topk=self.num_variables, code_dim=self.code_dim)
        self.tanner = AdaptiveTannerGraph(
            dim=self.dim, num_identity=self.num_identity, num_parity=self.num_parity,
            num_variables=self.num_variables, num_graph_nodes=self.num_graph_nodes,
            check_dim=self.check_dim, top_k=int(get("graph_top_k", 8)),
            code_dim=self.code_dim,
        )
        self.syndrome = VisualSyndrome(check_dim=self.check_dim, num_parity=self.num_parity,
                                       code_dim=self.code_dim,
                                       discrepancy=str(get("discrepancy", "learned")))
        self.locator = ErrorLocator(num_parity=self.num_parity,
                                    num_variables=self.num_variables,
                                    num_graph_nodes=self.num_graph_nodes)
        self.gating = ReliabilityAwareGating(num_graph_nodes=self.num_graph_nodes)
        self.decoder = NeuralBPDecoder(dim=self.dim, code_dim=self.code_dim,
                                       num_parity=self.num_parity,
                                       num_variables=self.num_variables,
                                       iterations=int(get("bp_iterations", 2)),
                                       links_per_check=int(get("h_links_per_check", 32)))
        self.fusion = CodeTrackFusion(dim=self.dim, fpn_dim=int(get("fpn_dim", 256)),
                                      head_dim=self.dim, grid=self.grid,
                                      taps=self.return_stages)
        self.head = build_head({
            "type": get("head_type", "CENTER"),
            "inplanes": self.dim,
            "channel": int(get("head_channel", 256)),
            "feat_sz": self.grid,
            "stride": self.patch_size,
        })

    # ------------------------------------------------------------------ weights
    def load_ostrack(self, ckpt_path: str, load_head: bool = True, verbose: bool = True
                     ) -> Dict[str, Any]:
        """Initialise the backbone (and optionally the head) from an OSTrack checkpoint."""
        report = {"backbone": load_ostrack_pretrained(self.backbone, ckpt_path, verbose=verbose)}
        if load_head:
            report["head"] = load_head_weights(self.head, ckpt_path, verbose=verbose)
        return report

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self) -> Dict[str, Tuple[int, int]]:
        """``{module: (trainable, total)}`` parameter counts."""
        out: Dict[str, Tuple[int, int]] = {}
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            out[name] = (trainable, total)
        return out

    # ------------------------------------------------------------------ forward
    def _code_path(self, z_r: torch.Tensor, x_r: torch.Tensor, z_t: torch.Tensor,
                   x_t: torch.Tensor, feats: Dict[str, torch.Tensor],
                   memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Everything after the backbone: coding -> checking -> correction -> head."""
        identity, parity, _query = self.codebook(z_r, z_t, memory)
        rel = self.reliability(x_r, x_t, identity, memory)
        sel = self.selector(x_r, x_t, identity)

        graph = self.tanner(x_r, identity, parity, priority=sel["priority"])
        syn = self.syndrome(graph["checks"], parity, graph["A_uv"])
        loc = self.locator(syn["syndrome"], graph["identity_map"],
                           graph["node_index"], self.num_variables)
        gate = self.gating(loc["severity"], graph["identity_map"])

        dec = self.decoder(x_r, x_t, parity, syn["syndrome"],
                           rel["r_r"], rel["r_t"], gate=gate)

        fus = self.fusion(dec["corrected_rgb"], dec["corrected_tir"],
                          dec["residual_rgb"], dec["residual_tir"],
                          feats["inter_r"], feats["inter_t"])
        score_map_ctr, bbox, size_map, offset_map = self.head(fus["feature_map"])

        return {
            "bbox": bbox,
            "score_map_ctr": score_map_ctr,
            "size_map": size_map,
            "offset_map": offset_map,
            "feature_map": fus["feature_map"],
            "syndrome": syn["syndrome"],
            "syndrome_raw": syn["syndrome_raw"],
            "reliability_rgb": rel["r_r"],
            "reliability_tir": rel["r_t"],
            "locator": loc["locator"],
            "severity": loc["severity"],
            "gate": gate,
            "corrected_rgb": dec["corrected_rgb"],
            "corrected_tir": dec["corrected_tir"],
            "residual_rgb": dec["residual_rgb"],
            "residual_tir": dec["residual_tir"],
            "A_uv": graph["A_uv"],
            "identity_map": graph["identity_map"],
            "identity_tokens": identity,
            "parity_tokens": parity,
            "fpn_rgb": fus["fpn_rgb"],
            "fpn_tir": fus["fpn_tir"],
        }

    def _maybe_corrupt(self, x_r: torch.Tensor, x_t: torch.Tensor,
                       corruption: Optional[Dict[str, Any]]):
        """Apply token-level corruption and return ``(x_r, x_t, mask)``.

        The mask (``B x N``, 1 = corrupted) is what supervises ``L_detect``.
        """
        from ..data.corruption.corruption import corrupt_tokens

        mask = torch.zeros(x_r.shape[:2], device=x_r.device)
        if not corruption or not corruption.get("enabled", True):
            return x_r, x_t, mask

        kinds = corruption.get("token") or corruption.get("kinds") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        if not kinds:
            kinds = ["tok_random_erase"]

        ratio = float(corruption.get("ratio", 0.2))
        severity = float(corruption.get("severity", 0.4))
        both = bool(corruption.get("both_modalities", True))

        for kind in kinds:
            x_r, m = corrupt_tokens(x_r, kind, ratio, severity)
            mask = torch.clamp(mask + m, max=1.0)
            if both:
                x_t, m2 = corrupt_tokens(x_t, kind, ratio, severity)
                mask = torch.clamp(mask + m2, max=1.0)
        return x_r, x_t, mask

    def forward(self, template_rgb: torch.Tensor, search_rgb: torch.Tensor,
                template_tir: torch.Tensor, search_tir: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                corruption: Optional[Dict[str, Any]] = None,
                clean_teacher: bool = False) -> Dict[str, torch.Tensor]:
        """One forward pass.

        ``corruption`` enables token-level corruption (``L_detect`` supervision);
        ``clean_teacher`` additionally runs the corruption-free path under
        ``no_grad`` and exposes its tokens/features as the target of ``L_correct``
        and ``L_identity``.  The frozen backbone is executed only once.
        """
        feats = self.backbone((template_rgb, search_rgb), (template_tir, search_tir),
                              return_inter=True)
        z_r, z_t = feats["z_r"], feats["z_t"]
        x_r_clean, x_t_clean = feats["x_r"], feats["x_t"]

        x_r, x_t, token_mask = self._maybe_corrupt(x_r_clean, x_t_clean, corruption)

        out = self._code_path(z_r, x_r, z_t, x_t, feats, memory=memory)
        out["token_mask"] = token_mask

        if clean_teacher:
            out["clean_tokens"] = {"rgb": x_r_clean.detach(), "tir": x_t_clean.detach()}
            with torch.no_grad():
                clean = self._code_path(z_r, x_r_clean, z_t, x_t_clean, feats, memory=memory)
            out["clean"] = {k: clean[k] for k in
                            ("feature_map", "corrected_rgb", "corrected_tir",
                             "reliability_rgb", "reliability_tir", "syndrome",
                             "bbox", "score_map_ctr", "size_map", "offset_map")}
        return out
