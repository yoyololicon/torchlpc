import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Tuple, Optional
from numba import jit, njit, prange, cuda, float32, float64


@cuda.jit
def lpc_cuda_kernel_float64(padded_y, A, B, T, order) -> None:
    sm = cuda.shared.array(shape=(1024,), dtype=float64)

    batch_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    i = tid
    b = batch_idx

    if b >= B or i >= order:
        return

    circular_idx = 0
    sm[i] = padded_y[b, i]

    for t in range(T):
        circular_idx = t % order
        if i == (order - 1):
            sm[circular_idx] *= -A[b, t, i]
        cuda.syncthreads()

        if i == (order - 1):
            v = padded_y[b, t + order]
        elif i > circular_idx - 1:
            v = -A[b, t, i] * sm[circular_idx - i - 1 + order]
        else:
            v = -A[b, t, i] * sm[circular_idx - i - 1]
        cuda.atomic.add(sm, circular_idx, v)
        cuda.syncthreads()

        if i == (order - 1):
            padded_y[b, t + order] = sm[circular_idx]


@cuda.jit
def lpc_cuda_kernel_float32(padded_y, A, B, T, order) -> None:
    sm = cuda.shared.array(shape=(1024,), dtype=float32)

    batch_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    i = tid
    b = batch_idx

    if b >= B or i >= order:
        return

    circular_idx = 0
    sm[i] = padded_y[b, i]

    for t in range(T):
        circular_idx = t % order
        if i == (order - 1):
            sm[circular_idx] *= -A[b, t, i]
        cuda.syncthreads()

        if i == (order - 1):
            v = padded_y[b, t + order]
        elif i > circular_idx - 1:
            v = -A[b, t, i] * sm[circular_idx - i - 1 + order]
        else:
            v = -A[b, t, i] * sm[circular_idx - i - 1]
        cuda.atomic.add(sm, circular_idx, v)
        cuda.syncthreads()

        if i == (order - 1):
            padded_y[b, t + order] = sm[circular_idx]


def lpc_cuda(x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
    B, T, order = A.shape
    assert order <= 1024
    padded_y = torch.empty((B, T + order), dtype=x.dtype, device=x.device)
    padded_y[:, :order] = zi.flip(1)
    padded_y[:, order:] = x

    threads_per_block = order
    blocks_per_grid = B

    if x.dtype == torch.float64:
        lpc_cuda_kernel_float64[blocks_per_grid, threads_per_block](
            cuda.as_cuda_array(padded_y), cuda.as_cuda_array(A), B, T, order
        )
    elif x.dtype == torch.float32:
        lpc_cuda_kernel_float32[blocks_per_grid, threads_per_block](
            cuda.as_cuda_array(padded_y), cuda.as_cuda_array(A), B, T, order
        )
    else:
        raise NotImplementedError

    return padded_y[:, order:]


@njit(parallel=True)
def lpc_np(x: np.ndarray, A: np.ndarray, zi: np.ndarray) -> None:
    B, T = x.shape
    order = zi.shape[1]
    padded_y = np.empty((B, T + order), dtype=x.dtype)
    padded_y[:, :order] = zi[:, ::-1]
    padded_y[:, order:] = x

    for b in prange(B):
        for t in range(T):
            ref = padded_y[b, t + order]
            for i in prange(order):
                ref -= A[b, t, i] * padded_y[b, t + order - i - 1]
            padded_y[b, t + order] = ref

    return padded_y[:, order:]


class LPC(Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor
    ) -> torch.Tensor:
        if x.is_cuda:
            y = lpc_cuda(x.detach(), A.detach(), zi.detach())
        else:
            y = lpc_np(
                x.detach().cpu().numpy(),
                A.detach().cpu().numpy(),
                zi.detach().cpu().numpy(),
            )
            y = torch.from_numpy(y).to(x.device, x.dtype)
        ctx.save_for_backward(A, zi, y)
        return y

    @staticmethod
    def backward(
        ctx, grad_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None
        B, T, order = A.shape

        flipped_A = A.flip(2)
        padded_flipped_A = F.pad(flipped_A.transpose(1, 2), (0, order + 1))
        shifted_A = (
            padded_flipped_A.reshape(B, T + order + 1, order)[:, :-1, :]
            .reshape(B, order, T + order)
            .transpose(1, 2)
            .flip(2)
        )

        if not ctx.needs_input_grad[2]:
            shifted_A = shifted_A[:, order:, :]
            padded_grad_y = grad_y
        else:
            padded_grad_y = F.pad(grad_y.unsqueeze(1), (order, 0)).squeeze(1)

        flipped_grad_x = LPC.apply(
            padded_grad_y.flip(1), shifted_A.flip(1), torch.zeros_like(zi)
        )

        if ctx.needs_input_grad[2]:
            grad_zi = flipped_grad_x[:, -order:]
            flipped_grad_x = flipped_grad_x[:, :-order]

        if ctx.needs_input_grad[0]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[1]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.flip(1), valid_y], dim=1)

            unfolded_y = padded_y.unfold(1, order, 1).flip(2)
            grad_A = unfolded_y * -flipped_grad_x.flip(1).unsqueeze(2)

        return grad_x, grad_A, grad_zi
