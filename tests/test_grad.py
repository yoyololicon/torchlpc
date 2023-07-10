import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torchlpc.core import LPC


def create_test_inputs(batch_size, samples):
    start_coeffs = [-0.9, 0.0]
    end_coeffs = [0.0, 1]

    A = (
        torch.stack(
            [torch.linspace(start_coeffs[i], end_coeffs[i], samples) for i in range(2)]
        )
        .T.unsqueeze(0)
        .double()
        .repeat(batch_size, 1, 1)
    )
    x = torch.randn(batch_size, samples).double()
    zi = torch.randn(batch_size, 2).double()
    return x, A, zi


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
def test_low_order_cpu(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
):
    batch_size = 4
    x, A, zi = create_test_inputs(batch_size, samples)
    A.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LPC.apply, (x, A, zi))
    assert gradgradcheck(LPC.apply, (x, A, zi))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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
def test_low_order_cuda(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
):
    batch_size = 4
    x, A, zi = create_test_inputs(batch_size, samples)
    x = x.cuda()
    A = A.cuda()
    zi = zi.cuda()

    A.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LPC.apply, (x, A, zi))
    assert gradgradcheck(LPC.apply, (x, A, zi))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_float64_vs_32_cuda():
    batch_size = 4
    samples = 32
    x, A, zi = create_test_inputs(batch_size, samples)
    x = x.cuda()
    A = A.cuda()
    zi = zi.cuda()

    x32 = x.float()
    A32 = A.float()
    zi32 = zi.float()

    y64 = LPC.apply(x, A, zi)
    y32 = LPC.apply(x32, A32, zi32)

    assert torch.allclose(y64, y32.double(), atol=1e-6), torch.max(
        torch.abs(y64 - y32.double())
    )
