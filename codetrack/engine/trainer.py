"""Training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.runtime import CPU_THREAD_SETTINGS

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import build_dataset
from ..data.corruption.corruption import CorruptionConfig, apply_image_corruption
from ..metrics import (_box_iou, normalized_precision, precision_at, success_auc,
                       summarize)
from .evaluator import summarize_detection, summarize_recovery, summarize_hinge
from ..models.codetrack import CodeTrack
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .losses import CodeTrackLoss


def plan_corruption(cfg_corruption: Dict[str, Any], corruption: Optional[Dict[str, Any]]
                    ) -> tuple:
    """Resolve one evaluation condition into ``(img_cfg, token_cfg, rng, identity_mode)``.

    ``corruption["mode"] == "identity"`` is the **strict no-op control**: diagnostics stay
    enabled but no image-level degradation, no token mask and -- crucially -- **no RNG draw**
    happens, so the frame stream, the masks and every later random number match the clean run
    bit for bit.  ``ratio = 0`` is *not* a no-op: ``corrupt_tokens`` clamps to one erased
    token per frame, which can still move a response peak and therefore the whole closed-loop
    trajectory.

    Extracted from ``infer_sequence`` so the plan is testable without a model or a dataset.
    """
    if not corruption:
        return None, None, None, False
    if str(corruption.get("mode", "")).lower() == "identity":
        return None, None, None, True
    merged = {**cfg_corruption, **corruption}
    img_cfg = CorruptionConfig.from_dict(merged)
    rng = np.random.default_rng(int(img_cfg.seed or 0))
    token_cfg = ({"enabled": True, "token": img_cfg.token,
                  "ratio": img_cfg.ratio, "severity": img_cfg.severity,
                  "target": merged.get("target", "both")}
                 if img_cfg.token else None)
    return img_cfg, token_cfg, rng, False


# A tracked box larger than this multiple of the frame means the loop has diverged; letting it
# through is what turns one bad frame into an unbounded crop (see MAX_CROP_SIDE_FACTOR).
MAX_BOX_SCALE = 4.0


#: Why a prediction had to be pulled back onto the frame.  A single "clamped" counter mixed four
#: very different events: an unbounded *scale* explosion (the closed loop diverging), a
#: centre that walked off the frame, a non-positive size, and a non-finite prediction.  Reporting
#: "87 % of frames were clamped" without this breakdown reads as "87 % diverged", which is not
#: what the counter measured (docs/results.md 6.15).
CLAMP_SCALE = "scale"
CLAMP_CENTER = "center"
CLAMP_NONPOSITIVE = "nonpositive"
CLAMP_NONFINITE = "nonfinite"
CLAMP_REASONS = (CLAMP_SCALE, CLAMP_CENTER, CLAMP_NONPOSITIVE, CLAMP_NONFINITE)


def clamp_box(box: np.ndarray, width: int, height: int,
              max_scale: float = MAX_BOX_SCALE) -> tuple:
    """Keep a predicted box finite, positive and at most ``max_scale`` times the frame.

    Returns ``(box, reasons)`` where ``reasons`` is the set (possibly empty) of
    :data:`CLAMP_REASONS` that fired.  ``bool(reasons)`` is the old ``clamped`` flag, so callers
    that only need "was it touched" are unchanged; callers that report divergence should use the
    breakdown, because only ``CLAMP_SCALE`` means the loop grew without bound.
    """
    reasons: set = set()
    value = np.asarray(box, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        reasons.add(CLAMP_NONFINITE)
        return (np.array([width * 0.25, height * 0.25, width * 0.5, height * 0.5],
                         dtype=np.float32), reasons)
    x, y, w, h = (float(v) for v in value)
    if w <= 0 or h <= 0:
        w, h = max(w, 2.0), max(h, 2.0)
        reasons.add(CLAMP_NONPOSITIVE)
    if w > max_scale * width:
        w = max_scale * width
        reasons.add(CLAMP_SCALE)
    if h > max_scale * height:
        h = max_scale * height
        reasons.add(CLAMP_SCALE)
    cx, cy = x + w / 2.0, y + h / 2.0
    ccx, ccy = min(max(cx, 0.0), float(width)), min(max(cy, 0.0), float(height))
    if ccx != cx or ccy != cy:
        reasons.add(CLAMP_CENTER)
    return np.array([ccx - w / 2.0, ccy - h / 2.0, w, h], dtype=np.float32), reasons


def crop_box_for_frame(predicted: np.ndarray, fixed_boxes: Optional[np.ndarray],
                       index: int) -> np.ndarray:
    """Which box drives the search crop of frame ``index``.

    ``fixed_boxes`` (a reference trajectory) makes every condition see identical crops, which
    is what separates "this frame was harder" from "the closed loop diverged earlier".

    The reference is applied **lagged by one frame**, because the free loop crops frame ``f``
    around the box produced for frame ``f - 1`` (and around the frame-0 annotation for ``f == 0``)
    -- never around its own output for frame ``f``.  Passing ``reference[f]`` instead, as the
    first version of this function did, gave the replay a crop the free run never saw and handed
    it the reference's answer one frame early; frame 0 in particular was cropped around the model's
    *prediction* rather than the annotation, so the runs did not even start from the same crop.
    ``tools/trajectory_replay.py``'s ``crop_identical`` check was writtten to catch exactly that
    and could not, because its condition was ``or len(gt) > 0``.
    """
    if fixed_boxes is None:
        return predicted
    if index <= 0:
        # frame 0 is driven by the initial box in every arm, fixed schedule or not
        return predicted
    return np.asarray(fixed_boxes[index - 1], dtype=np.float32)


class Trainer:
    """Owns the model, the data, the optimizer and the train / evaluate loops."""

    def __init__(self, cfg: Dict[str, Any], output_dir: str | Path = "outputs/run"):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("codetrack", self.output_dir / "train.log")
        self.logger.info("CPU thread limits: %s", CPU_THREAD_SETTINGS)

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
            # "mask" (shipped) supervises the detection heads with the injected corruption
            # mask; "deviation" supervises the continuous feature deviation from the frozen
            # teacher instead, which is definable for degradation families that have no mask.
            detect_target=str(loss_cfg.get("detect_target", "mask")),
            detect_deviation_cm=float(loss_cfg.get("detect_deviation_cm", 0.2)),
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
            "target": str(c.get("target", "both")),
        }

    @torch.enable_grad()
    def _gradient_conflict(self, inputs, target, corruption) -> Optional[Dict[str, float]]:
        """Is the tracking objective fighting the *weighted* auxiliary objective?

        ``L_correct`` pulls the features back towards the clean codeword while ``L_track``
        only needs the box to be right; if the two gradients are consistently opposed the
        correction branch is being fought by the tracking objective rather than helped.

        The first version reported ``cos(L_track, L_correct)`` only -- one unweighted pair, on
        three modules, with no notion of how hard the auxiliary terms actually push.  But the
        optimiser sees

            g = g_track + lambda_detect * g_detect + lambda_correct * g_correct
                       + lambda_preserve * g_preserve + lambda_gain * g_gain
                       + lambda_identity * g_identity

        so a positive cosine between two *unweighted* terms says little about whether the
        combined auxiliary gradient opposes tracking, and nothing about its magnitude.  With
        ``lambda_correct = 2.0`` and ``lambda_preserve = 0.5`` the auxiliary side can dominate
        ``g_track`` while ``L_correct`` alone looks harmless, which is exactly the
        "repair is harmful rather than merely ineffective" hypothesis that has to be
        distinguishable (docs/results.md 6.27).  Both readings are therefore reported.
        """
        watched = ("decoder", "codebook", "reliability")
        params = [p for n, p in self.model.named_parameters()
                  if p.requires_grad and n.split(".")[0] in watched]
        if not params:
            return None

        # Both objectives share one forward and one sampled corruption.  Re-running the
        # model here compared gradients from different erased tokens and different dropout
        # draws, which is not a gradient-conflict measurement.
        out = self.model(*inputs, corruption=corruption, clean_teacher=True)
        parts = self.loss_fn(out, target)

        def grads(loss_key: str, retain_graph: bool) -> Optional[torch.Tensor]:
            loss = parts.get(f"{loss_key}_graph", parts[loss_key])
            if not torch.isfinite(loss):
                return None
            # In decoder_mode="off" the correction terms are constants with no grad_fn (the
            # "corrected" tokens are literally the input), and torch.autograd.grad raises on
            # them.  isfinite() alone does not catch it.  The conflict diagnostic is
            # meaningless for that arm anyway -- there is no correction gradient to conflict.
            if not loss.requires_grad:
                return None
            g = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                    allow_unused=True)
            # Preserve parameter-vector alignment: an unused parameter contributes a zero
            # segment in its original position instead of disappearing from the vector.
            flat = [(torch.zeros_like(param) if grad is None else grad).reshape(-1)
                    for param, grad in zip(params, g)]
            return torch.cat(flat) if flat else None

        if self.model.decoder.mode == "off":
            # no correction path exists, so there is no conflict to report
            return None

        # The auxiliary terms, weighted exactly as the optimiser sees them.
        weighted = (("detect", self.loss_fn.lambda_detect),
                    ("correct", self.loss_fn.lambda_correct),
                    ("preserve", self.loss_fn.lambda_preserve),
                    ("gain", self.loss_fn.lambda_gain),
                    ("identity", self.loss_fn.lambda_identity))
        active = [(key, weight) for key, weight in weighted if float(weight) != 0.0]

        g_track = grads("track", retain_graph=True)
        if g_track is None:
            return None
        terms: Dict[str, torch.Tensor] = {}
        for position, (key, _) in enumerate(active):
            last = position == len(active) - 1
            value = grads(key, retain_graph=not last)
            if value is not None:
                terms[key] = value
        if not terms:
            return None
        g_aux = torch.zeros_like(g_track)
        for key, weight in active:
            if key in terms:
                g_aux = g_aux + float(weight) * terms[key]

        diag: Dict[str, float] = {}
        # Unweighted pair, kept so the existing log parser and every archived log stay readable.
        if "correct" in terms:
            diag["grad_cos_track_correct"] = float(
                torch.nn.functional.cosine_similarity(g_track, terms["correct"], dim=0))
            diag["grad_norm_track"] = float(g_track.norm())
            diag["grad_norm_correct"] = float(terms["correct"].norm())
        # The quantity that actually drives the shared weights.
        diag["grad_cos_track_aux_weighted"] = float(
            torch.nn.functional.cosine_similarity(g_track, g_aux, dim=0))
        diag["grad_norm_aux_weighted"] = float(g_aux.norm())
        diag["grad_aux_over_track"] = float(g_aux.norm() / g_track.norm().clamp(min=1e-12))
        diag["grad_cos_weighted_total"] = float(torch.nn.functional.cosine_similarity(
            g_track, g_track + g_aux, dim=0))
        return diag

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
                        # The legacy prefix is kept verbatim so archived logs stay parseable; the
                        # weighted reading is appended, because the unweighted pair alone cannot
                        # distinguish "repair is ineffective" from "repair is harmful".
                        message = ("[grad] cos(L_track, L_correct) = %+.3f | |g_track| %.3f "
                                   "| |g_correct| %.3f")
                        values = [diag["grad_cos_track_correct"], diag["grad_norm_track"],
                                  diag["grad_norm_correct"]]
                        if "grad_cos_track_aux_weighted" in diag:
                            message += (" | cos(L_track, g_aux) = %+.3f | |g_aux| %.3f "
                                        "| aux/track %.3f | cos(L_track, g_total) = %+.3f")
                            values += [diag["grad_cos_track_aux_weighted"],
                                       diag["grad_norm_aux_weighted"],
                                       diag["grad_aux_over_track"],
                                       diag["grad_cos_weighted_total"]]
                        self.logger.info(message, *values)
                nonfinite = int(outputs.get("decoder_nonfinite_steps", 0) or 0)
                if nonfinite and nonfinite != getattr(self, "_nonfinite_reported", 0):
                    self._nonfinite_reported = nonfinite
                    self.logger.warning(
                        "decoder replaced %d non-finite update(s) with zero so far "
                        "(fp16 overflow in the un-normalised residual path?)", nonfinite)
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
                       collect: bool = False,
                       fixed_boxes: Optional[np.ndarray] = None,
                       forward_fn=None) -> Dict[str, Any]:
        """Run the tracker over one sequence.

        With ``corruption`` the frames are degraded (image level + token level) exactly as
        during training, and with ``collect`` the per-frame error-correction diagnostics
        are accumulated alongside the boxes.

        ``fixed_boxes`` replays a *reference* trajectory: frame ``f`` is cropped around
        ``fixed_boxes[f]`` instead of around this run's own previous prediction, so every
        condition sees exactly the same crops and a per-frame difference is the model's
        response to that frame rather than the closed loop having diverged earlier.  The
        returned ``pred`` still holds this run's own predictions.

        ``forward_fn`` replaces the model call of the *corrupted* pass with a callable of the
        same signature.  That is how ``tools/trajectory_replay.py`` injects path substitutions
        (clean reliability and gates, or clean TIR tokens into the fusion) into the tracker loop
        without duplicating the loop; ``None`` keeps the plain model call.
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
        # The box that actually drove each frame's crop.  Returned so a fixed-crop replay can
        # *verify* that it saw the same crops instead of asserting it in a comment.
        crops_used: List[np.ndarray] = []
        box_clamps = 0
        box_clamps_by_reason = {reason: 0 for reason in CLAMP_REASONS}
        frame_h, frame_w = rgb0.shape[:2]

        img_cfg, token_cfg, rng, identity_mode = plan_corruption(
            self.cfg.get("corruption", {}) or {}, corruption)
        if identity_mode:
            # strict no-op control: diagnostics on, corruption off, RNG untouched
            self.logger.info("%s: identity corruption (diagnostics=%s, no RNG draw)",
                             seq, collect)

        diag = {"syndrome": [], "syndrome_y": [], "syndrome_density": [], "syndrome_clean": [],
                "locator": [], "locator_y": [],
                "reliability_rgb": [], "reliability_y_rgb": [],
                "reliability_tir": [], "reliability_y_tir": [],
                "e_before": [], "e_after": [], "e_clean": [],
                "rec_e_before": [], "rec_e_after": [], "rec_hinge": [], "rec_active": []}

        for f in range(n):
            gt = annos[f]
            rgb = cv2.cvtColor(cv2.imread(str(rgb_frames[f])), cv2.COLOR_BGR2RGB)
            tir = cv2.imread(str(tir_frames[f]), cv2.IMREAD_GRAYSCALE)
            # Keep the pristine pair: with image-level corruption the "clean reference"
            # pass below must see an undamaged image, otherwise corrupted and clean
            # syndrome come from the same degraded frame and the AUROC is 0.5 by
            # construction.
            rgb_pristine, tir_pristine = rgb, tir
            if img_cfg is not None:
                rgb, tir = apply_image_corruption(rgb, tir, img_cfg, rng)

            crop = crop_box_for_frame(prev, fixed_boxes, f)
            crops_used.append(crop.astype(np.float32).copy())
            cx, cy = crop[0] + crop[2] / 2, crop[1] + crop[3] / 2
            side = float(np.sqrt(max(crop[2], 1) * max(crop[3], 1))) * sf
            search_rgb = self._crop_resize(rgb, cx, cy, side, search_size)
            search_tir = self._crop_resize(tir, cx, cy, side, search_size)

            s_rgb = to_tensor(search_rgb, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            s_tir = to_tensor(search_tir, 1, (0.449,), (0.226,))

            call = forward_fn or self.model
            out = call(tpl_rgb_t.unsqueeze(0).to(self.device),
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
                if img_cfg is not None:
                    clean_rgb = self._crop_resize(rgb_pristine, cx, cy, side, search_size)
                    clean_tir = self._crop_resize(tir_pristine, cx, cy, side, search_size)
                    clean_s_rgb = to_tensor(clean_rgb, 3, (0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))
                    clean_s_tir = to_tensor(clean_tir, 1, (0.449,), (0.226,))
                else:
                    clean_s_rgb, clean_s_tir = s_rgb, s_tir
                clean_out = self.model(tpl_rgb_t.unsqueeze(0).to(self.device),
                                       clean_s_rgb.unsqueeze(0).to(self.device),
                                       tpl_tir_t.unsqueeze(0).to(self.device),
                                       clean_s_tir.unsqueeze(0).to(self.device),
                                       corruption=None)
                diag["syndrome_clean"].append(
                    clean_out["syndrome"].flatten(1)[0].float().cpu().numpy())

            px = cx + (box[0] - 0.5) * side
            py = cy + (box[1] - 0.5) * side
            pw, ph = box[2] * side, box[3] * side
            prediction = np.array([px - pw / 2, py - ph / 2, pw, ph], dtype=np.float32)
            prediction, clamp_reasons = clamp_box(prediction, frame_w, frame_h)
            if clamp_reasons:
                box_clamps += 1
                for reason in clamp_reasons:
                    box_clamps_by_reason[reason] += 1
            if fixed_boxes is None:
                # free-running: this run's own prediction drives the next crop
                prev = prediction

            preds.append(prediction.copy())
            gts.append(gt.copy())

        result: Dict[str, Any] = {"pred": np.asarray(preds), "gt": np.asarray(gts),
                                  "crops": (np.asarray(crops_used) if crops_used
                                            else np.zeros((0, 4), dtype=np.float32)),
                                  "box_clamps": box_clamps,
                                  "box_clamps_by_reason": box_clamps_by_reason}
        if box_clamps:
            # Visible, not silent: a frame whose box had to be pulled back is a frame where the
            # closed loop diverged, and that is part of reading the metrics.  The breakdown
            # matters: only CLAMP_SCALE means the prediction grew without bound, while a centre
            # clamp is a much weaker event that the old single counter reported identically.
            detail = ", ".join(f"{reason}={count}"
                               for reason, count in box_clamps_by_reason.items() if count)
            self.logger.info("%s: %d/%d frames had the predicted box clamped (%s)",
                             seq, box_clamps, len(preds), detail)
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
        # 1 - r is the coefficient the decoder multiplies, so its AUROC against the *token*
        # mask is the honest version of "does the reliability head know what is damaged".
        # Per modality, with that modality's own mask as the label.
        if "reliability_rgb" in out:
            diag["reliability_rgb"].append(
                (1.0 - out["reliability_rgb"][0].float()).cpu().numpy())
            diag["reliability_y_rgb"].append(m_r.cpu().numpy())
        if "reliability_tir" in out:
            diag["reliability_tir"].append(
                (1.0 - out["reliability_tir"][0].float()).cpu().numpy())
            diag["reliability_y_tir"].append(m_t.cpu().numpy())

        if "clean_tokens" in out:
            clean = out["clean_tokens"]
            for mod in ("rgb", "tir"):
                key = f"corrected_{mod}"
                corrupted_key = f"corrupted_{mod}"
                # With image-level corruption and no token corruption there is no
                # ``corrupted_*`` token view at all (the model is called with
                # ``corruption=None``), and the teacher would be the degraded image's own
                # features.  Indexing the key used to raise KeyError, which evaluate()
                # swallowed as "skip <sequence>", so every image-level condition silently
                # scored zero sequences.
                if key not in out or corrupted_key not in out:
                    continue
                ref = clean[mod][0].float()
                after = (out[key][0].float() - ref).abs().mean(-1)
                before = (out[corrupted_key][0].float() - ref).abs().mean(-1)
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
                 topk: int = 5,
                 sequence_list: Optional[str] = None,
                 collect_diagnostics: bool = False) -> Dict[str, float]:
        """Sequence-level PR / SR / NPR, plus the error-correction diagnostics.

        With ``corruption`` set, the run also reports syndrome AUROC, localization
        recall@k (against its chance level) and the masked recovery gain / damage --
        the numbers that actually speak to the error-correction claim.

        ``sequence_list`` names a file with one sequence per line and replaces the subset
        listing.  It defaults to ``None``, so the historical "first N of the subset" path
        (and every number already computed from it) stays bit-identical.

        ``collect_diagnostics`` decouples the per-frame diagnostics pass (and its extra
        clean-reference forward) from the presence of corruption, which is what makes a
        strict no-op control possible at all.
        """
        if checkpoint:
            report = load_checkpoint(checkpoint, self.model, map_location=str(self.device))
            self.logger.info("loaded checkpoint %s (epoch %s)", checkpoint, report["epoch"])
            if report["missing"] or report["unexpected"]:
                # A silently partial load is how a randomly initialised decoder gets
                # evaluated: load_checkpoint's mode guard catches the arm-level case, this
                # catches any other key drift.
                self.logger.warning(
                    "checkpoint key mismatch: %d missing, %d unexpected (first missing: %s)",
                    len(report["missing"]), len(report["unexpected"]),
                    report["missing"][:3] or "none")
        self.model.eval()

        data_root = Path(root or self.cfg.get("data", {}).get("root", "data/LasHeR"))
        if sequence_list:
            seqs = [line.strip() for line in Path(sequence_list).read_text().splitlines()
                    if line.strip()]
            self.logger.info("explicit sequence list %s (%d sequences)", sequence_list,
                             len(seqs))
        else:
            list_name = ("testingsetList.txt" if subset.startswith("test")
                         else "trainingsetList.txt")
            seqs = [s.strip() for s in
                    (data_root / list_name).read_text().splitlines() if s.strip()]
        if max_sequences:
            seqs = seqs[:max_sequences]

        results = []
        sequence_metrics = []
        diags: list = []
        skipped: List[str] = []
        for i, seq in enumerate(seqs):
            try:
                r = self.infer_sequence(data_root, subset, seq, max_frames=max_frames,
                                        corruption=corruption,
                                        collect=bool(corruption) or collect_diagnostics)
            except Exception as exc:                                   # noqa: BLE001
                self.logger.warning("skip %s: %s", seq, exc)
                skipped.append(f"{seq}: {exc}")
                continue
            if len(r["pred"]) == 0:
                skipped.append(f"{seq}: empty prediction")
                continue
            seq_metrics = {
                # frames and clamped frames make the *divergence rate* of a condition
                # reportable next to its metrics: with both modalities erased at 40 % some
                # sequences spend most frames with the box clipped at the frame bound, and an
                # SR number there describes a broken loop rather than a repair failure.
                "frames": int(len(r["pred"])),
                "box_clamps": int(r.get("box_clamps", 0)),
                "box_clamps_by_reason": {reason: int(count) for reason, count
                                         in (r.get("box_clamps_by_reason") or {}).items()},
                "sr": success_auc(r["pred"], r["gt"]),
                "pr": precision_at(r["pred"], r["gt"], 20.0),
                "npr": normalized_precision(r["pred"], r["gt"], 0.2),
                # Mean frame IoU is stored per sequence on purpose: SR is a thresholded
                # statistic whose per-sequence variance is large (see docs/results.md 6.13), and
                # a paired analysis needs a lower-variance endpoint to be powered at all.  It is
                # an additional endpoint, not a replacement for the pre-registered SR.
                "iou_mean": float(_box_iou(r["pred"], r["gt"]).mean())
                if len(r["pred"]) else float("nan"),
            }
            results.append(seq_metrics)
            sequence_metrics.append({"sequence": seq, **seq_metrics})
            if "diagnostics" in r:
                diags.append(r["diagnostics"])
            if (i + 1) % 20 == 0:
                self.logger.info("evaluated %d/%d sequences", i + 1, len(seqs))

        summary = summarize(results)
        summary["per_sequence"] = sequence_metrics
        iou_values = [m["iou_mean"] for m in sequence_metrics
                      if np.isfinite(m.get("iou_mean", float("nan")))]
        if iou_values:
            # Additional endpoint next to the pre-registered SR: mean frame IoU is not
            # thresholded, so its per-sequence variance is smaller and a paired claim can be
            # powered at a realistic sample size (docs/results.md 6.13).
            summary["iou"] = float(np.mean(iou_values))
        # A silently shortened evaluation is worse than a failed one: metrics computed over
        # half the set look fine and mean nothing.  The valid count is reported explicitly and
        # a large skip rate is surfaced rather than left in the log.
        summary["n_sequences"] = len(results)
        summary["n_box_clamped_frames"] = int(sum(m.get("box_clamps", 0)
                                                  for m in sequence_metrics))
        # The breakdown, so "N frames clamped" cannot be read as "N scaled-up divergences": a
        # centre clamp is a different (much weaker) event than a scale clamp, and the old single
        # counter reported both identically (docs/results.md 6.15).
        summary["n_box_clamps_by_reason"] = {
            reason: int(sum((m.get("box_clamps_by_reason") or {}).get(reason, 0)
                            for m in sequence_metrics))
            for reason in CLAMP_REASONS}
        total_frames = int(sum(m.get("frames", 0) for m in sequence_metrics))
        summary["n_frames"] = total_frames
        summary["divergence_rate"] = (float(summary["n_box_clamped_frames"] / total_frames)
                                      if total_frames else float("nan"))
        if summary["n_box_clamped_frames"]:
            detail = ", ".join(f"{reason}={count}" for reason, count
                               in summary["n_box_clamps_by_reason"].items() if count)
            self.logger.warning(
                "%d frames had their predicted box clamped to the frame bound (%s) -- only the "
                "scale count means the closed loop grew without bound (MAX_BOX_SCALE); see "
                "docs/results.md 6.15",
                summary["n_box_clamped_frames"], detail)
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
                                               merged.get("syndrome_density"),
                                               merged.get("reliability_rgb"),
                                               merged.get("reliability_y_rgb"),
                                               merged.get("reliability_tir"),
                                               merged.get("reliability_y_tir"),
                                               topk=topk))
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
