"""Neural belief-propagation decoder: configurable number of message-passing iterations.

v_i^{(l+1)} = v_i^{(l)} + (1 - r_i) * dv_i^{(l)}. See docs/architecture.md section 6.
"""

from .bp import NeuralBPDecoder

__all__ = ["NeuralBPDecoder"]
