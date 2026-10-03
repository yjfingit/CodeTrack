"""Training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import build_dataset
from ..data.corruption.corruption import CorruptionConfig, apply_image_corruption
from ..metrics import normalized_precision, precision_at, success_auc, summarize
from .evaluator import summarize_detection, summarize_recovery, summarize_hinge
from ..models.codetrack import CodeTrack
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .losses import CodeTrackLoss


class Trainer:
    """Owns the model, the data, the optimizer and the train / evaluate loops."""

    def __init__(self, cfg: Dict[str, Any], output_dir: str | Path = "outputs/run"):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("codetrack", self.output_dir / "train.log")

        self.seed = int(cfg.get("seed", 0))
        set_seed(self.seed)

        requested = str(cfg.get("device", "cuda"))
        self.device = torch.device(requested if (requested == "cpu" or torch.cuda.is_available())
                                   else "cpu")
        self.logger.info("device: %s", self.device)

        # ---- model ------------------------------------------------------------
        model_cfg = cfg.get("model", {})
        self.model = CodeTrack(cfg).to(self.device)
        pretrained = model_cfg.get("pretrained") or cfg.get("pretrained")
        if pretrained:
            if Path(pretrained).is_file():
                self.model.load_ostrack(str(pretrained), verbose=True)
            else:
                self.logger.warning("pretrained checkpoint not found: %s", pretrained)

        train_cfg = cfg.get("train", {})
        self.epochs = int(train_cfg.get("epochs", 10))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.log_every = int(train_cfg.get("log_every", 10))
        self.amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"

        loss_cfg = cfg.get("loss", {})
        self.loss_fn = CodeTrackLoss(
            cls_weight=float(loss_cfg.get("cls", 1.0)),
            l1_weight=float(loss_cfg.get("l1", 5.0)),
            giou_weight=float(loss_cfg.get("giou", 2.0)),
            lambda_detect=float(loss_cfg.get("lambda_detect", 1.0)),
            lambda_correct=float(loss_cfg.get("lambda_correct", 2.0)),
            lambda_identity=float(loss_cfg.get("lambda_identity", 0.1)),
            lambda_preserve=float(loss_cfg.get("lambda_preserve", 0.5)),
            lambda_gain=float(loss_cfg.get("lambda_gain", 1.0)),
            gain_beta=float(loss_cfg.get("gain_beta", 0.9)),
            detect_reliability=float(loss_cfg.get("detect_reliability", 1.0)),
            detect_syndrome=float(loss_cfg.get("detect_syndrome", 1.0)),
            detect_localize=float(loss_cfg.get("detect_localize", 1.0)),
            num_parity=self.model.num_parity,
            dim=self.model.dim,
            code_dim=self.model.code_dim,
        ).to(self.device)

        # built after the loss module: the optimizer owns both parameter sets
        self.optimizer = torch.optim.AdamW(
            self._optim_parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        # ---- corruption + staged training schedule -----------------------------
        corr = dict(cfg.get("corruption", {}) or {})
        self.use_corruption = bool(corr.get("enabled", False))
        self.use_clean_teacher = bool(loss_cfg.get("clean_teacher", self.use_corruption))
        self.warmup_epochs = int(train_cfg.get("codec_warmup_epochs", 0))
        self.grad_diag_every = int(train_cfg.get("grad_diag_every", 0))
        self.stage = ""

        self.dataset = None
        self.loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------- setup
    def build_loader(self) -> None:
        data_cfg = self.cfg.get("data", {})
        self.dataset = build_dataset(self.cfg)
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(data_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=True,
            persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
        )
        self.logger.info("dataset: %d sequences | %d samples/epoch | batch %d",
                         len(self.dataset.sequences), len(self.dataset),
                         int(data_cfg.get("batch_size", 8)))

    def _to_device(self, batch: Dict[str, Any]):
        keys = ("template_rgb", "search_rgb", "template_tir", "search_tir")
        return [batch[k].to(self.device, non_blocking=True) for k in keys]

    def _optim_parameters(self):
        """Trainable model parameters **plus** the loss module's own parameters.

        The identity projection used by ``L_identity`` lives inside the loss module; if
        it is not handed to the optimizer it silently stays at its random init.
        """
        return list(self.model.trainable_parameters()) + list(self.loss_fn.parameters())

    def set_trainable(self, modules: Optional[set] = None) -> None:
        """Freeze every CodeTrack block outside ``modules`` (``None`` = all of them).

        The backbone keeps its own (frozen) state.  The optimizer is rebuilt afterwards
        because the parameter set changed.
        """
        for name, module in self.model.named_children():
            if name == "backbone":
                continue
            allow = modules is None or name in modules
            for p in module.parameters():
                p.requires_grad = allow
        train_cfg = self.cfg.get("train", {})
        self.optimizer = torch.optim.AdamW(
            self._optim_parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

    def _corruption_cfg(self) -> Optional[Dict[str, Any]]:
        if not self.use_corruption:
            return None
        c = self.cfg.get("corruption", {}) or {}
        return {
            "enabled": True,
            "token": c.get("token") or ["tok_random_erase"],
            "ratio": float(c.get("ratio", 0.2)),
            "severity": float(c.get("severity", 0.4)),
            "both_modalities": bool(c.get("both_modalities", True)),
        }

    @torch.enable_grad()
    def _gradient_conflict(self, inputs, target, corruption) -> Optional[Dict[str, float]]:
        """Cosine between the tracking and the correction gradients.

        ``L_correct`` pulls the features back towards the clean codeword while ``L_track``
        only needs the box to be right; if the two gradients are consistently opposed the
        correction branch is being fought by the tracking objective rather than helped.
        Measured on the blocks the two losses actually share.
        """
        watched = ("decoder", "codebook", "reliability")
        params = [p for n, p in self.model.named_parameters()
                  if p.requires_grad and n.split(".")[0] in watched]
        if not params:
            return None

        def grads(loss_key: str) -> Optional[torch.Tensor]:
            out = self.model(*inputs, corruption=corruption, clean_teacher=True)
            parts = self.loss_fn(out, target)
            loss = parts.get(f"{loss_key}_graph", parts[loss_key])
            if not torch.isfinite(loss):
                return None
            # In decoder_mode="off" the correction terms are constants with no grad_fn (the
            # "corrected" tokens are literally the input), and torch.autograd.grad raises on
            # them.  isfinite() alone does not catch it.  The conflict diagnostic is
            # meaningless for that arm anyway -- there is no correction gradient to conflict.
            if not loss.requires_grad:
                return None
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            flat = [x.reshape(-1) for x in g if x is not None]
            return torch.cat(flat) if flat else None

        if self.model.decoder.mode == "off":
            # no correction path exists, so there is no conflict to report
            return None

        g_track = grads("track")
        g_correct = grads("correct")
        if g_track is None or g_correct is None:
            return None
        cos = torch.nn.functional.cosine_similarity(g_track, g_correct, dim=0)
        return {"grad_cos_track_correct": float(cos),
                "grad_norm_track": float(g_track.norm()),
                "grad_norm_correct": float(g_correct.norm())}

    # ------------------------------------------------------------------- train
    def train(self, max_iters: Optional[int] = None) -> Dict[str, Any]:
        if self.loader is None:
            self.build_loader()
        self.model.train()

        corruption = self._corruption_cfg()
        if corruption:
            self.logger.info("token corruption ON: %s ratio=%.2f severity=%.2f",
                             corruption["token"], corruption["ratio"], corruption["severity"])

        step = 0
        t_start = time.time()

        for epoch in range(self.epochs):
            # staged schedule: first learn to detect+repair (codebook + BP decoder only),
            # then unfreeze everything for joint fine-tuning.
            stage = "codec" if epoch < self.warmup_epochs else "joint"
            if stage != self.stage:
                # warm up the whole detect+repair chain, not just the codebook and the
                # decoder -- otherwise the syndrome and the locator stay random while the
                # loss still asks them to localise corruption
                warmup_modules = {"codebook", "reliability", "selector", "tanner",
                                  "syndrome", "locator", "gating", "decoder"}
                self.set_trainable(warmup_modules if stage == "codec" else None)
                self.stage = stage
                n_train = sum(p.numel() for p in self._optim_parameters())
                self.logger.info("stage -> %s | trainable %.3f M", stage, n_train / 1e6)

            if hasattr(self.dataset, "seed"):
                self.dataset.seed = self.seed + epoch * 9973
            epoch_start = time.time()

            for batch in self.loader:
                inputs = self._to_device(batch)
                target = batch["gt_box"].to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    outputs = self.model(*inputs, corruption=corruption,
                                         clean_teacher=self.use_clean_teacher)
                # the objective is evaluated in fp32: focal / BCE are unsafe under autocast
                losses = self.loss_fn(outputs, target)

                self.scaler.scale(losses["loss"]).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                step += 1
                if self.grad_diag_every and step % self.grad_diag_every == 0:
                    diag = self._gradient_conflict(inputs, target, corruption)
                    if diag:
                        self.logger.info(
                            "[grad] cos(L_track, L_correct) = %+.3f | |g_track| %.3f "
                            "| |g_correct| %.3f",
                            diag["grad_cos_track_correct"], diag["grad_norm_track"],
                            diag["grad_norm_correct"])
                if step % self.log_every == 0 or step == 1:
                    mem = (torch.cuda.max_memory_allocated() / 2**20
                           if self.device.type == "cuda" else 0)
                    self.logger.info(
                        "[%s] it %d | loss %.4f | track %.4f (cls %.3f l1 %.3f giou %.3f) "
                        "| detect %.4f (rel %.3f syn %.3f loc %.3f) "
                        "| correct %.4f (gain %.4f act %.2f p95 %.3f pres %.4f) "
                        "| identity %.4f | %.2fs/it | %.0f MiB",
                        stage, step, losses["loss"].item(), losses["track"].item(),
                        losses["cls"].item(), losses["l1"].item(), losses["giou"].item(),
                        losses["detect"].item(), losses["detect_reliability"].item(),
                        losses["detect_syndrome"].item(), losses["detect_localize"].item(),
                        losses["correct"].item(), losses["gain"].item(),
                        losses["gain_active_fraction"].item(),
                        losses["e_ratio_p95"].item(), losses["preserve"].item(),
                        losses["identity"].item(), (time.time() - t_start) / step, mem)

                if max_iters and step >= max_iters:
                    break

            path = save_checkpoint(self.output_dir / "last.pth", self.model, self.optimizer,
                                   epoch=epoch, cfg=self.cfg)
            self.logger.info("epoch %d done in %.1fs -> %s", epoch, time.time() - epoch_start,
                             path.name)
            if max_iters and step >= max_iters:
                break

        final = save_checkpoint(self.output_dir / "final.pth", self.model, self.optimizer,
                                epoch=self.epochs - 1, cfg=self.cfg)
        self.logger.info("training finished: %d steps, checkpoint %s", step, final)
        return {"steps": step, "checkpoint": str(final)}

    # ---------------------------------------------------------------- evaluate
    def _sequence_frames(self, root: Path, subset: str, seq: str, modality: str):
        d = root / subset / seq / modality
        frames = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
        return frames

    @staticmethod
    def _crop_resize(img: np.ndarray, cx: float, cy: float, side: float, out_size: int) -> np.ndarray:
        from ..data.transforms.sample import _crop_square, _resize
        return _resize(_crop_square(img, cx, cy, side), out_size)

    @torch.no_grad()
    def infer_sequence(self, root: Path, subset: str, seq: str,
                       max_frames: Optional[int] = None,
                       corruption: Optional[Dict[str, Any]] = None,
                       collect: bool = False) -> Dict[str, Any]:
        """Run the tracker over one sequence.

        With ``corruption`` the frames are degraded (image level + token level) exactly as
        during training, and with ``collect`` the per-frame error-correction diagnostics
        are accumulated alongside the boxes.
        """
        rgb_frames = self._sequence_frames(root, subset, seq, "visible")
        tir_frames = self._sequence_frames(root, subset, seq, "infrared")
        anno_path = root / "annos" / f"{seq}.txt"
        annos = np.array([[float(v) for v in line.replace("\t", ",").split(",")[:4]]
                          for line in anno_path.read_text().splitlines() if line.strip()],
                         dtype=np.float32)

        n = min(len(rgb_frames), len(tir_frames), len(annos))
        if max_frames:
            n = min(n, max_frames)
        if n < 2:
            return {"pred": np.zeros((0, 4)), "gt": np.zeros((0, 4))}

        sf = self.model.cfg.get("model", {}).get("search_factor", 4.0)
        tf = self.model.cfg.get("model", {}).get("template_factor", 2.0)
        search_size = self.model.img_size
        template_size = self.model.template_size

        rgb0 = cv2.cvtColor(cv2.imread(str(rgb_frames[0])), cv2.COLOR_BGR2RGB)
        tir0 = cv2.imread(str(tir_frames[0]), cv2.IMREAD_GRAYSCALE)
        g0 = annos[0]
        t_cx, t_cy = g0[0] + g0[2] / 2, g0[1] + g0[3] / 2
        t_side = float(np.sqrt(max(g0[2], 1) * max(g0[3], 1))) * tf
        template_rgb = self._crop_resize(rgb0, t_cx, t_cy, t_side, template_size)
        template_tir = self._crop_resize(tir0, t_cx, t_cy, t_side, template_size)

        from ..data.transforms.sample import to_tensor
        tpl_rgb_t = to_tensor(template_rgb, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        tpl_tir_t = to_tensor(template_tir, 1, (0.449,), (0.226,))

        prev = g0.copy()
        preds, gts = [], []

        img_cfg = None
        token_cfg = None
        rng = None
        if corruption:
            merged = {**self.cfg.get("corruption", {}), **corruption}
            img_cfg = CorruptionConfig.from_dict(merged)
            rng = np.random.default_rng(int(img_cfg.seed or 0))
            token_cfg = {"enabled": True,
                         "token": img_cfg.token or ["tok_random_erase"],
                         "ratio": img_cfg.ratio, "severity": img_cfg.severity,
                         "target": merged.get("target", "both")}

        diag = {"syndrome": [], "syndrome_y": [], "syndrome_density": [], "syndrome_clean": [],
                "locator": [], "locator_y": [],
                "e_before": [], "e_after": [], "e_clean": [],
                "rec_e_before": [], "rec_e_after": [], "rec_hinge": [], "rec_active": []}

        for f in range(n):
            gt = annos[f]
            rgb = cv2.cvtColor(cv2.imread(str(rgb_frames[f])), cv2.COLOR_BGR2RGB)
            tir = cv2.imread(str(tir_frames[f]), cv2.IMREAD_GRAYSCALE)
            if img_cfg is not None:
                rgb, tir = apply_image_corruption(rgb, tir, img_cfg, rng)

            cx, cy = prev[0] + prev[2] / 2, prev[1] + prev[3] / 2
            side = float(np.sqrt(max(prev[2], 1) * max(prev[3], 1))) * sf
            search_rgb = self._crop_resize(rgb, cx, cy, side, search_size)
            search_tir = self._crop_resize(tir, cx, cy, side, search_size)

            s_rgb = to_tensor(search_rgb, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            s_tir = to_tensor(search_tir, 1, (0.449,), (0.226,))

            out = self.model(tpl_rgb_t.unsqueeze(0).to(self.device),
                             s_rgb.unsqueeze(0).to(self.device),
                             tpl_tir_t.unsqueeze(0).to(self.device),
                             s_tir.unsqueeze(0).to(self.device),
                             corruption=token_cfg, clean_teacher=bool(collect))
            box = out["bbox"][0].float().cpu().numpy()          # cx, cy, w, h in [0, 1]

            if collect:
                self._collect_diagnostics(out, diag)
                # Clean reference pass.  A healthy codeword must produce a low syndrome;
                # without these negatives the per-check AUROC has a single class and is
                # undefined (a check watching 32/256 tokens fires on a 20%-corrupted
                # frame with probability ~0.999).
                clean_out = self.model(tpl_rgb_t.unsqueeze(0).to(self.device),
                                       s_rgb.unsqueeze(0).to(self.device),
                                       tpl_tir_t.unsqueeze(0).to(self.device),
                                       s_tir.unsqueeze(0).to(self.device),
                                       corruption=None)
                diag["syndrome_clean"].append(
                    clean_out["syndrome"].flatten(1)[0].float().cpu().numpy())

            px = cx + (box[0] - 0.5) * side
            py = cy + (box[1] - 0.5) * side
            pw, ph = box[2] * side, box[3] * side
            prev = np.array([px - pw / 2, py - ph / 2, pw, ph], dtype=np.float32)

            preds.append(prev.copy())
            gts.append(gt.copy())

        result: Dict[str, Any] = {"pred": np.asarray(preds), "gt": np.asarray(gts)}
        if collect:
            result["diagnostics"] = diag
        return result

    @torch.no_grad()
    def _collect_diagnostics(self, out: Dict[str, torch.Tensor],
                             diag: Dict[str, list]) -> None:
        """Accumulate one frame of error-correction diagnostics.

        Two per-check labels are recorded because they answer different questions:

        * ``syndrome_y``      -- ``1[(H @ M) > 0]``, "does this check watch anything bad".
          Under 20% erasure with >=32 edges per check this is ~always 1, so it is kept
          only as a coarse reference.
        * ``syndrome_density`` -- the fraction of each check's neighbourhood that is
          corrupted.  This is what training actually regresses, and what a ranking metric
          must be computed against.
        """
        m_r = out["token_mask_rgb"][0].float()
        m_t = out["token_mask_tir"][0].float()
        mask_any = (m_r + m_t).clamp(max=1.0)

        diag["syndrome"].append(out["syndrome"].flatten(1)[0].float().cpu().numpy())
        diag["syndrome_y"].append(((out["H"] @ mask_any) > 0).float().cpu().numpy())
        support = out["H_support"].float()                                  # M x N
        degree = support.sum(dim=1).clamp(min=1.0)
        diag["syndrome_density"].append(
            ((support @ mask_any) / degree).float().cpu().numpy())
        diag["locator"].append(out["locator_scattered"][0].float().cpu().numpy())
        diag["locator_y"].append(mask_any.cpu().numpy())

        if "clean_tokens" in out:
            clean = out["clean_tokens"]
            for mod in ("rgb", "tir"):
                key = f"corrected_{mod}"
                if key not in out:
                    continue
                ref = clean[mod][0].float()
                after = (out[key][0].float() - ref).abs().mean(-1)
                before = (out[f"corrupted_{mod}"][0].float() - ref).abs().mean(-1)
                m = (m_r if mod == "rgb" else m_t)
                diag["e_before"].append(before[m > 0.5].cpu().numpy())
                diag["e_after"].append(after[m > 0.5].cpu().numpy())
                diag["e_clean"].append(after[m < 0.5].cpu().numpy())

        # The authoritative recovery numbers: computed by the SAME helper training uses, so
        # "L_gain = 0.0000" and "recovery_gain = -0.7" cannot both be true.  The per-modality
        # arrays above are kept for the damage breakdown, where per-modality granularity is
        # what you actually want; this block is for the pass/fail question.
        #
        # Inference runs batch 1, so these are single-element tensors.  ``.mean()`` rather
        # than ``[0]``: indexing a 0-d tensor raises, and that used to be swallowed by
        # evaluate()'s except-and-skip, which would have turned a crash into "N/A".
        rec = self.loss_fn.masked_recovery(out)
        diag["rec_e_before"].append(float(rec["e_before"].mean()))
        diag["rec_e_after"].append(float(rec["e_after"].mean()))
        diag["rec_hinge"].append(float(rec["hinge"].mean()))
        diag["rec_active"].append(float(rec["active"].mean()))

    @torch.no_grad()
    def evaluate(self, checkpoint: Optional[str] = None, subset: str = "testingset",
                 max_sequences: Optional[int] = None, max_frames: Optional[int] = None,
                 root: Optional[str] = None,
                 corruption: Optional[Dict[str, Any]] = None,
                 topk: int = 5) -> Dict[str, float]:
        """Sequence-level PR / SR / NPR, plus the error-correction diagnostics.

        With ``corruption`` set, the run also reports syndrome AUROC, localization
        recall@k (against its chance level) and the masked recovery gain / damage --
        the numbers that actually speak to the error-correction claim.
        """
        if checkpoint:
            report = load_checkpoint(checkpoint, self.model, map_location=str(self.device))
            self.logger.info("loaded checkpoint %s (epoch %s)", checkpoint, report["epoch"])
        self.model.eval()

        data_root = Path(root or self.cfg.get("data", {}).get("root", "data/LasHeR"))
        list_name = "testingsetList.txt" if subset.startswith("test") else "trainingsetList.txt"
        seqs = [s.strip() for s in (data_root / list_name).read_text().splitlines() if s.strip()]
        if max_sequences:
            seqs = seqs[:max_sequences]

        results = []
        diags: list = []
        skipped: List[str] = []
        for i, seq in enumerate(seqs):
            try:
                r = self.infer_sequence(data_root, subset, seq, max_frames=max_frames,
                                        corruption=corruption, collect=bool(corruption))
            except Exception as exc:                                   # noqa: BLE001
                self.logger.warning("skip %s: %s", seq, exc)
                skipped.append(f"{seq}: {exc}")
                continue
            if len(r["pred"]) == 0:
                skipped.append(f"{seq}: empty prediction")
                continue
            results.append({
                "sr": success_auc(r["pred"], r["gt"]),
                "pr": precision_at(r["pred"], r["gt"], 20.0),
                "npr": normalized_precision(r["pred"], r["gt"], 0.2),
            })
            if "diagnostics" in r:
                diags.append(r["diagnostics"])
            if (i + 1) % 20 == 0:
                self.logger.info("evaluated %d/%d sequences", i + 1, len(seqs))

        summary = summarize(results)
        # A silently shortened evaluation is worse than a failed one: metrics computed over
        # half the set look fine and mean nothing.  The valid count is reported explicitly and
        # a large skip rate is surfaced rather than left in the log.
        summary["n_sequences"] = len(results)
        summary["n_requested"] = len(seqs)
        summary["n_skipped"] = len(skipped)
        self.logger.info("evaluation on %d sequences: PR %.2f | SR(AUC) %.2f | NPR %.2f",
                         len(results), summary.get("pr", 0) * 100,
                         summary.get("sr", 0) * 100, summary.get("npr", 0) * 100)
        if skipped:
            self.logger.warning("%d/%d sequences skipped -- metrics are over the valid "
                                "subset only; first few: %s",
                                len(skipped), len(seqs), skipped[:3])

        if diags:
            merged = {k: [x for d in diags for x in d[k]] for k in diags[0]}
            summary.update(summarize_detection(merged["syndrome"], merged["syndrome_y"],
                                               merged["locator"], merged["locator_y"],
                                               merged.get("syndrome_clean"),
                                               merged.get("syndrome_density"), topk=topk))
            summary.update(summarize_recovery(merged["e_before"], merged["e_after"],
                                              merged["e_clean"]))
            if merged.get("rec_hinge"):
                # The per-modality arrays above average every corrupted token of both
                # modalities together; the helper normalises each modality by its own
                # corrupted-token count and then adds.  With different corruption counts per
                # modality the two weightings differ, which is why the headline number was
                # computed from a different estimator than the one training logs.  Both are
                # reported, and ``recovery_gain`` now comes from the helper.
                summary.update(summarize_hinge(merged["rec_e_before"], merged["rec_e_after"],
                                                merged["rec_hinge"], merged["rec_active"],
                                                self.loss_fn.gain_beta))
                eb = float(np.mean(merged["rec_e_before"])) if merged["rec_e_before"] else 0.0
                ea = float(np.mean(merged["rec_e_after"])) if merged["rec_e_after"] else 0.0
                summary["recovery_gain_unpooled"] = summary.get("recovery_gain")
                summary["recovery_gain"] = (1.0 - ea / eb) if eb > 0 else float("nan")
            self.logger.info(
                "error-correction | syndrome AUROC %.3f | density rho %.3f (mae %.3f) "
                "| locator P@%d %.3f (chance %.3f) | recovery %.3f | damage %.4f",
                summary["syndrome_auroc"],
                summary.get("syndrome_density_spearman", float("nan")),
                summary.get("syndrome_density_mae", float("nan")),
                topk, summary[f"locator_precision_at_{topk}"],
                summary.get("locator_precision_chance", float("nan")),
                summary["recovery_gain"], summary["damage_clean"])
        return summary
