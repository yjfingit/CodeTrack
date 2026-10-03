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
            temperature=float(get("assign_temperature", 0.1)),
            tau_min=float(get("tau_min", 0.02)),
            tau_max=float(get("tau_max", 0.5)),
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
        # ``syndrome_use_obs_energy`` is OFF by default.  Under zero-erasure the per-check
        # observation energy correlates with the corruption density at r = 1.0000 -- the
        # label leaks through it -- so any claim that "the Tanner structure produced the
        # syndrome" has to be made on a run without it.  See VisualSyndrome's docstring.
        self.syndrome = VisualSyndrome(
            check_dim=self.check_dim, num_parity=self.num_parity, code_dim=self.code_dim,
            discrepancy=str(get("discrepancy", "learned")),
            density_prior=float(get("corruption_ratio", 0.2)),
            use_obs_energy=bool(get("syndrome_use_obs_energy", False)))
        self.locator = ErrorLocator(num_parity=self.num_parity,
                                    num_variables=self.num_variables,
                                    num_graph_nodes=self.num_graph_nodes)
        self.gating = ReliabilityAwareGating(num_graph_nodes=self.num_graph_nodes)
        self.decoder = NeuralBPDecoder(dim=self.dim, code_dim=self.code_dim,
                                       num_parity=self.num_parity,
                                       num_variables=self.num_variables,
                                       iterations=int(get("bp_iterations", 2)),
                                       links_per_check=int(get("h_links_per_check", 32)),
                                       mode=str(get("decoder_mode", "bp")),
                                       locality_window=int(get("h_locality_window", 0)),
                                       free_edge_frac=float(get("h_free_edge_frac", 0.25)),
                                       locality_wrap=bool(get("h_locality_wrap", True)))
        self.fusion = CodeTrackFusion(dim=self.dim, fpn_dim=int(get("fpn_dim", 256)),
                                      head_dim=self.dim, grid=self.grid,
                                      taps=self.return_stages,
                                      use_fpn=bool(get("use_fpn", True)))
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
        # ---- 1. Target Coding -------------------------------------------------
        # identity comes from the trusted template only
        identity, _query = self.codebook.encode_identity(z_r, z_t, memory)
        rel = self.reliability(x_r, x_t, identity, memory)
        sel = self.selector(x_r, x_t, identity)

        # one incidence H for check aggregation, syndrome, localization and BP
        H = self.decoder.matrix()

        # the "received word": both modalities projected into the codebook space and
        # averaged, so corruption in either one shows up in the parity observation
        observation = 0.5 * (self.codebook.to_code(x_r) + self.codebook.to_code(x_t))
        # the "expected codeword": same H, reconstructed from the trusted identity.
        # ``w`` is gated by the per-token reliability so that already-suspect tokens cannot
        # drag the mixture coefficients around -- otherwise the "redundancy" would be
        # computed from the very word it is supposed to correct.
        reliability_code = 0.5 * (rel["r_r"] + rel["r_t"])
        parity = self.codebook.parity_from_incidence(identity, observation, H,
                                                     reliability=reliability_code)

        # ---- 2. Syndrome Checking ----------------------------------------------
        # What the *trusted* identity predicts the same aggregation should look like.
        # ``parity`` already is exactly that (H @ w @ U), so re-projecting it through the
        # Tanner's own ``obs_proj`` gives an ``expected_obs`` on the same scale as
        # ``obs_proj(H @ observation)``.  Their difference is the literal
        # "H x == codeword" residual the syndrome reports -- and it is computed *before*
        # the check LayerNorm, which is the only place the magnitude survives.
        expected_obs = self.tanner.obs_proj(parity)
        graph = self.tanner(x_r, identity, parity, priority=sel["priority"],
                            variables_tir=x_t, H=H, observation=observation,
                            expected_obs=expected_obs)
        syn = self.syndrome(graph["checks"], parity,
                            residual=graph["check_residual"],
                            obs_energy=graph["obs_energy"])
        loc = self.locator(syn["syndrome"], graph["identity_map"],
                           graph["node_index"], self.num_variables, H=H)

        # reliability gathered onto the selected graph nodes, one vector per modality
        idx = graph["node_index"]
        gate_rgb, gate_tir = self.gating(loc["severity"],
                                         rel["r_r"].gather(1, idx),
                                         rel["r_t"].gather(1, idx))

        dec = self.decoder(x_r, x_t, parity, syn["syndrome"],
                           rel["r_r"], rel["r_t"],
                           gate_rgb=gate_rgb, gate_tir=gate_tir)

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
            "check_agg": syn["check_agg"],
            "check_ref": syn["check_ref"],
            "check_delta": syn["check_delta"],
            "check_residual": graph["check_residual"],
            "reliability_rgb": rel["r_r"],
            "reliability_tir": rel["r_t"],
            "locator": loc["locator"],
            "locator_scattered": loc["locator_scattered"],
            "severity": loc["severity"],
            "gate_rgb": gate_rgb,
            "gate_tir": gate_tir,
            "H": H,
            "H_support": self.decoder.connectivity(),
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
            "fpn_conf": fus["fpn_conf"],
        }

    def _maybe_corrupt(self, x_r: torch.Tensor, x_t: torch.Tensor,
                       corruption: Optional[Dict[str, Any]]):
        """Apply token-level corruption; returns ``(x_r, x_t, mask_rgb, mask_tir)``.

        The two masks are kept **separate** on purpose: supervising both reliabilities
        with their union would force a healthy modality's reliability down whenever the
        other one is damaged, which is exactly the opposite of "repair the bad modality
        using the good one".
        """
        from ..data.corruption.corruption import corrupt_tokens

        mask_r = torch.zeros(x_r.shape[:2], device=x_r.device)
        mask_t = torch.zeros(x_t.shape[:2], device=x_t.device)
        if not corruption or not corruption.get("enabled", True):
            return x_r, x_t, mask_r, mask_t

        kinds = corruption.get("token") or corruption.get("kinds") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        if not kinds:
            kinds = ["tok_random_erase"]

        ratio = float(corruption.get("ratio", 0.2))
        severity = float(corruption.get("severity", 0.4))
        # "rgb" / "tir" / "both" -- single-modality corruption is what exposes whether
        # the tracker repairs the damaged modality using the healthy one
        target = str(corruption.get("target", "both")).lower()
        if not bool(corruption.get("both_modalities", True)):
            target = "both" if target == "both" else target

        for kind in kinds:
            if target in ("both", "rgb"):
                x_r, m = corrupt_tokens(x_r, kind, ratio, severity)
                mask_r = torch.clamp(mask_r + m, max=1.0)
            if target in ("both", "tir"):
                x_t, m2 = corrupt_tokens(x_t, kind, ratio, severity)
                mask_t = torch.clamp(mask_t + m2, max=1.0)
        return x_r, x_t, mask_r, mask_t

    def _apply_tap_corruption(self, feats: Dict[str, torch.Tensor], mask_rgb: torch.Tensor,
                              mask_tir: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Erase the FPN taps at exactly the corrupted search positions.

        ``feats["inter_r"]`` / ``feats["inter_t"]`` hold ``block2`` / ``block5`` taps of
        shape ``B x (Lz + Lx) x C``, where ``Lx`` tokens correspond 1:1 to the corrupted
        search tokens.  Applying the same mask keeps the two paths honest: the FPN cannot
        carry information the corrupted token no longer has.

        Only the search part is touched; the template tokens are trusted.
        """
        lens_z = int(self.backbone.pos_embed_z.shape[1])
        out = dict(feats)
        for key, mask in (("inter_r", mask_rgb), ("inter_t", mask_tir)):
            taps = feats.get(key)
            if not taps or float(mask.sum()) == 0:
                continue
            corrupted = {}
            for name, feat in taps.items():
                if float(mask.sum()) == 0:
                    corrupted[name] = feat
                    continue
                # split template / search, erase the search positions the mask points at
                z_part, x_part = feat[:, :lens_z], feat[:, lens_z:]
                keep = (1.0 - mask.to(feat.dtype)).unsqueeze(-1)
                corrupted[name] = torch.cat([z_part, x_part * keep], dim=1)
            out[key] = corrupted
        return out

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

        x_r, x_t, mask_r, mask_t = self._maybe_corrupt(x_r_clean, x_t_clean, corruption)

        # The FPN taps are read from *inside* the backbone, i.e. before token corruption.
        # Left untouched they hand the tracker a clean shortcut: L_track can open the gate
        # and ignore the whole correction branch, because the answer is already in block2
        # and block5.  A soft gate cannot fix this -- it is an optimisation brake, not a
        # structural one.  So the taps get the *same* mask at the *same* 16x16 positions:
        # whatever the decoder does not repair is also missing from the FPN path.
        feats = self._apply_tap_corruption(feats, mask_r, mask_t)

        out = self._code_path(z_r, x_r, z_t, x_t, feats, memory=memory)
        out["token_mask_rgb"] = mask_r
        out["token_mask_tir"] = mask_t
        out["token_mask"] = torch.clamp(mask_r + mask_t, max=1.0)   # union, diagnostics only
        if corruption is not None:
            # the corrupted view is kept so the evaluation can measure
            # "error before correction" against the clean tokens
            out["corrupted_rgb"] = x_r
            out["corrupted_tir"] = x_t

        if clean_teacher:
            # The teacher is the *uncorrupted token set* -- running a second full
            # _code_path here would cost ~30% of the training step and nothing consumed
            # its output, so the branch is intentionally kept to the backbone features.
            out["clean_tokens"] = {"rgb": x_r_clean.detach(), "tir": x_t_clean.detach()}
        return out
