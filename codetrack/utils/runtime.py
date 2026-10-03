"""Runtime limits shared by training, evaluation, and diagnostic processes."""

from __future__ import annotations

import os


def configure_cpu_threads() -> dict[str, int]:
    """Cap library thread pools so concurrent experiment processes stay predictable."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

    import cv2
    import torch

    torch_threads = int(os.environ.get("CODETRACK_TORCH_THREADS", "1"))
    interop_threads = int(os.environ.get("CODETRACK_TORCH_INTEROP_THREADS", "1"))
    cv2_threads = int(os.environ.get("CODETRACK_OPENCV_THREADS", "0"))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # This can only happen if another import already started PyTorch work. Report the
        # actual setting so parallel-run logs remain auditable.
        pass
    cv2.setNumThreads(cv2_threads)
    return {
        "torch": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
        "opencv": cv2.getNumThreads(),
    }


CPU_THREAD_SETTINGS = configure_cpu_threads()
