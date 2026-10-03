"""Visual Syndrome Computation (S) and Error Locator / severity gating.

See docs/architecture.md section 5.
"""

from .syndrome import ErrorLocator, ReliabilityAwareGating, VisualSyndrome

__all__ = ["ErrorLocator", "ReliabilityAwareGating", "VisualSyndrome"]
