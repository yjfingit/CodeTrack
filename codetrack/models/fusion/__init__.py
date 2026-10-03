"""Residual add, feature-map reshape and FPN fusion across backbone blocks.

See docs/architecture.md section 7.
"""

from .fusion import CodeTrackFusion

__all__ = ["CodeTrackFusion"]
