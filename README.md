# TorchLPC

`torchlpc` provides a PyTorch implementation of the Linear Predictive Coding (LPC) filter, also known as IIR filtering.
It's fast, differentiable, and supports batched inputs with time-varying filter coefficients.
The computation is done as follows:

Given an input signal $`\mathbf{x} \in \mathbb{R}^T`$ and time-varying LPC coefficients $`\mathbf{A} \in \mathbb{R}^{T \times N}`$ with an order of $`N`$, the LPC filter is defined as:

$$
y_t = x_t - \sum_{i=1}^N A_{t,i} y_{t-i}.
$$

## Usage

```python

import torch
from torchlpc import sample_wise_lpc

# Create a batch of 10 signals, each with 100 time steps
x = torch.randn(10, 100)

# Create a batch of 10 sets of LPC coefficients, each with 100 time steps and an order of 3
A = torch.randn(10, 100, 3)

# Apply LPC filtering
y = sample_wise_lpc(x, A)

# Optionally, you can provide initial values for the output signal (default is 0)
zi = torch.randn(10, 3)
y = sample_wise_lpc(x, A, zi=zi)
```


## Installation

```bash
pip install torchlpc
```

or from source

```bash
pip install git+https://github.com/yoyololicon/torchlpc.git
```

## Derivation of the gradients of the LPC filter

The details of the derivation can be found in our preprint **Differentiable All-pole Filters for Time-varying Audio Systems**[^1].
We show that, given the instataneous gradient $\frac{\partial \mathcal{L}}{\partial y_t}$ where $\mathcal{L}$ is the loss function, the gradients of the LPC filter with respect to the input signal $x_t$ and the filter coefficients $A_{t, :}$ can be computed as follows:

```math
\frac{\partial \mathcal{L}}{\partial x_t}
= \frac{\partial \mathcal{L}}{\partial y_t}
- \sum_{i=1}^{N} A_{t+i,i} \frac{\partial \mathcal{L}}{\partial x_{t+i}}
```

$$
\frac{\partial \mathcal{L}}{\partial A_{t,i}}
= -\frac{\partial \mathcal{L}}{\partial x_t} y_{t-i}.
$$

### Gradients for the initial condition $`y_t|_{t \leq 0}`$

The algorithm could be extended for modelling initial conditions based on the same idea from the previous [section](#propagating-gradients-to-the-coefficients).
The initial conditions are the inputs to the system when $t \leq 0$, so their gradients equal $`\frac{\partial \mathcal{L}}{\partial x_t}|_{-N < t \leq 0}`$. 
You can imaginate that $`x_t|_{1 \leq t \leq T}`$ just represent a segment of the whole signal $x_t$ and $y_t|_{t \leq 0}$ are the system outputs based on $`x_t|_{t \leq 0}`$.
The [initial rest condition](#derivation-of-the-gradients-of-the-lpc-filtering-operation) still holds but happens somewhere $t \leq -N$.
In practice, we get the gradients by running the backward filter for $N$ more steps at the end.

### Time-invariant filtering

In the time-invariant setting, $`A_{t', i} = A_{t, i} \forall t, t' \in [1, T]`$ and the filter is simplified to

```math
y_t = x_t - \sum_{i=1}^N a_i y_{t-i}, \mathbf{a} = A_{1,:}.
```

The gradients $`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}`$ are filtering $`\frac{\partial \mathcal{L}}{\partial \mathbf{y}}`$ with $\mathbf{a}$ backwards in time, same as in the time-varying case.
For $`\frac{\partial \mathcal{L}}{\partial \mathbf{a}}`$, instead of matrices multiplication, we do a vecotr-matrix multiplication $`-\frac{\partial \mathcal{L}}{\partial \mathbf{x}} \mathbf{Y}`$.
You can think of the difference as summarising the gradients for $a_i$ at all the time steps, eliminating the time axis.
This algorithm is more efficient than [^1] because it only needs one pass of filtering to get the two gradients while the latter needs two.

[^1]: [Differentiable All-pole Filters for Time-varying Audio Systems](https://arxiv.org/abs/2404.07970)

## TODO

- [ ] Use PyTorch C++ extension for faster computation.
- [ ] Use native CUDA kernels for GPU computation.
- [ ] Add examples.

## Citation

If you find this repository useful in your research, please cite the repository with the following BibTex entry:

```bibtex
 @misc{ycy2024diffapf,
      title={Differentiable All-pole Filters for Time-varying Audio Systems},
      author={Chin-Yun Yu and Christopher Mitcheltree and Alistair Carson and Stefan Bilbao and Joshua D. Reiss and György Fazekas},
      year={2024},
      eprint={2404.07970},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
  }
```
