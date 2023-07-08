import torch
from typing import Optional

from .core import LPC

__all__ = ["sample_wise_lpc"]


def sample_wise_lpc(
    x: torch.Tensor, a: torch.Tensor, zi: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute LPC filtering sample-wise.

    Args:
        x (torch.Tensor): Input signal.
        a (torch.Tensor): LPC coefficients.
        zi (torch.Tensor): Initial conditions.

    Shape:
        - x: :math:`(B, T)`
        - a: :math:`(B, T, order)`
        - zi: :math:`(B, order)`

    Returns:
        torch.Tensor: Filtered signal with the same shape as x.
    """
    assert x.shape[0] == a.shape[0]
    assert x.shape[1] == a.shape[1]
    assert x.ndim == 2
    assert a.ndim == 3

    B, T, order = a.shape
    if zi is None:
        zi = a.new_zeros(B, order)
    else:
        assert zi.shape == (B, order)

    return LPC.apply(x, a, zi)
