"""Target Codebook Encoder: identity tokens U and sparse parity tokens P (ECC projection).

P is generated from trusted sources only (initial template + reliable history),
never from the current search frame. See docs/architecture.md section 2.
"""

from .codebook import SparseParityGenerator, TargetCodebookEncoder

__all__ = ["SparseParityGenerator", "TargetCodebookEncoder"]
