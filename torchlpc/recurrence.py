import torch
import torch.nn.functional as F
from torch.autograd import Function
from numba import cuda

from .parallel_scan import compute_linear_recurrence


class RecurrenceCUDA(Function):
    @staticmethod
    def forward(
        ctx, decay: torch.Tensor, impulse: torch.Tensor, initial_state: torch.Tensor
    ) -> torch.Tensor:
        n_dims, n_steps = decay.shape
        out = torch.empty_like(impulse)
        compute_linear_recurrence(
            cuda.as_cuda_array(decay.detach()),
            cuda.as_cuda_array(impulse.detach()),
            cuda.as_cuda_array(initial_state.detach()),
            cuda.as_cuda_array(out),
            n_dims,
            n_steps,
        )
        ctx.save_for_backward(decay, initial_state, out)
        return out

    @staticmethod
    def backward(ctx: torch.Any, grad_out: torch.Tensor) -> torch.Tensor:
        decay, initial_state, out = ctx.saved_tensors
        grad_decay = grad_impulse = grad_initial_state = None
        n_dims, _ = decay.shape

        padded_decay = F.pad(decay.unsqueeze(1), (0, 1)).squeeze(1)
        if ctx.needs_input_grad[2]:
            padded_grad_out = F.pad(grad_out.unsqueeze(1), (1, 0)).squeeze(1)
        else:
            padded_grad_out = grad_out
            padded_decay = padded_decay[:, 1:]

        init = padded_grad_out.new_zeros(n_dims)
        flipped_grad_impulse = RecurrenceCUDA.apply(
            padded_decay.flip(1).conj_physical(),
            padded_grad_out.flip(1),
            init,
        )

        if ctx.needs_input_grad[2]:
            grad_initial_state = flipped_grad_impulse[:, -1]
            flipped_grad_impulse = flipped_grad_impulse[:, :-1]

        if ctx.needs_input_grad[1]:
            grad_impulse = flipped_grad_impulse.flip(1)

        if ctx.needs_input_grad[0]:
            valid_out = out[:, :-1]
            padded_out = torch.cat([initial_state.unsqueeze(1), valid_out], dim=1)
            grad_decay = padded_out.conj_physical() * flipped_grad_impulse.flip(1)

        return grad_decay, grad_impulse, grad_initial_state
