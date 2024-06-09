import logging
import os
import time
from typing import Optional

import torch as tr
import torch.utils.cpp_extension
from torch import Tensor as T

from torchlpc import sample_wise_lpc

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def sample_wise_lpc_scriptable(x: T, a: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert a.ndim == 3
    assert x.size(0) == a.size(0)
    assert x.size(1) == a.size(1)

    B, T, order = a.shape
    if zi is None:
        zi = a.new_zeros(B, order)
    else:
        assert zi.shape == (B, order)

    zi = tr.flip(zi, dims=[1])
    a_flip = tr.flip(a, dims=[2])
    padded_y = tr.cat([zi, x], dim=1)

    for t in range(T):
        prod = a_flip[:, t: t + 1] @ padded_y[:, t: t + order, None]
        prod = prod[:, 0, 0]
        padded_y[:, t + order] -= prod

    return padded_y[:, order:]


if __name__ == "__main__":
    log.info(f"torch number of threads: {torch.get_num_threads()}")
    os.environ["OMP_NUM_THREADS"] = f"{torch.get_num_threads()}"
    torch.set_num_threads(1)
    print(torch.__config__.parallel_info())

    torch.utils.cpp_extension.load(
        name="forward",
        sources=["torchlpc.cpp"],
        is_python_module=False,
        verbose=True
    )

    # torch.ops.load_library("build/torchlpc.so")

    # T = 10
    T = 2000
    # T = 10000
    # T = 200000
    bs = 4
    order = 3

    x = torch.randn(bs, T, dtype=torch.double)
    a = torch.randn(bs, T, order, dtype=torch.double)
    zi = torch.randn(bs, order, dtype=torch.double)

    log.info("Testing")
    start = time.time()
    og = sample_wise_lpc(x, a, zi)
    end = time.time()
    log.info(f"Testing sample_wise_lpc: {end - start:.4f}s")
    # start = time.time()
    # scriptable = sample_wise_lpc_scriptable(x, a, zi)
    # end = time.time()
    # log.info(f"Testing sample_wise_lpc_scriptable: {end - start:.4f}s")
    start = time.time()
    cpp = torch.ops.torchlpc.forward(x, a, zi)
    end = time.time()
    log.info(f"Testing sample_wise_lpc_cpp: {end - start:.4f}s")

    eps = 1e-8
    # log.info(torch.allclose(og, scriptable, atol=eps))
    log.info(torch.allclose(og, cpp, atol=eps))
