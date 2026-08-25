"""tensorforge: a from-scratch reverse-mode autodiff engine over NumPy arrays."""

from .tensor import Tensor, gradcheck

__all__ = ["Tensor", "gradcheck"]
__version__ = "0.1.0"
