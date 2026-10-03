"""Tracking head: bounding box prediction from the fused feature map.

See docs/architecture.md section 7.
"""

from .build import build_head, load_head_weights
from .center import CenterPredictor
from .corner import CornerPredictor

__all__ = ["CenterPredictor", "CornerPredictor", "build_head", "load_head_weights"]
