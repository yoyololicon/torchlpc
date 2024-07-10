import torch
import torch.nn.functional as F
from torch.func import hessian
import pytest
from torchlpc.core import LPC


from .test_grad import create_test_inputs


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "cuda",
    ],
)
def test_hessian(device: str):
    batch_size = 1
    samples = 10
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, False)
    )
    y = torch.randn_like(x)

    A = A[:, 0, :].clone()

    A.requires_grad = True
    zi.requires_grad = True

    args = (A, zi)

    def func(A, zi):
        return F.mse_loss(LPC.apply(x, A[:, None, :].expand(-1, samples, -1), zi), y)

    h = hessian(func, 0)(*args)
    assert torch.any(h != 0)

    h_inv = torch.linalg.inv(h.squeeze())
