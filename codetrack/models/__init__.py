"""Network modules.

One sub-package per architecture block; the block order mirrors docs/architecture.md:
backbone -> codebook -> reliability -> tanner -> syndrome -> decoder -> fusion -> head.
"""

from .codetrack import CodeTrack

__all__ = ["CodeTrack"]
