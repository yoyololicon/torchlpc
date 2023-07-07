import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torchlpc.core import LPC


@pytest.mark.parametrize(
    "x_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "a_requires_grad",
    [True],
)
@pytest.mark.parametrize(
    "zi_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "samples",
    [32],
)
def test_low_order(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
):
    start_coeffs = [-0.9, 0.0]
    end_coeffs = [0.0, 1]

    A = (
        torch.stack(
            [torch.linspace(start_coeffs[i], end_coeffs[i], samples) for i in range(2)]
        )
        .T.unsqueeze(0)
        .double()
    )
    x = torch.randn(1, samples).double()
    zi = torch.randn(1, 2).double()

    A.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LPC.apply, (x, A, zi))
    assert gradgradcheck(LPC.apply, (x, A, zi))
