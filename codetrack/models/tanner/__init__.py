"""Reliability-Adaptive Tanner Graph: dynamic sparse variable/check adjacency.

See docs/architecture.md section 4.
"""

from .tanner import AdaptiveTannerGraph, sparsified_softmax

__all__ = ["AdaptiveTannerGraph", "sparsified_softmax"]
