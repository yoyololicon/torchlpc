import torch
import torch.nn.functional as F
from torch.func import hessian, jacfwd
import pytest
from torchlpc.core import LPC


from .test_grad import create_test_inputs


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_vmap(device: str):
    batch_size = 4
    samples = 40
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, False)
    )
    y = torch.randn_like(x)

    A = A[:, 0, :].clone()

    A.requires_grad = True
    zi.requires_grad = True
    x.requires_grad = True

    args = (x, A, zi)

    def func(x, A, zi):
        return F.mse_loss(LPC.apply(x, A[:, None, :].expand(-1, samples, -1), zi), y)

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)
