# TorchLPC

`torchlpc` provides a PyTorch implementation of the Linear Predictive Coding (LPC) filtering operation, also known as IIR filtering.
It's fast, differentiable, and supports batched inputs with time-varying filter coefficients.
The computation is done as follows:

Given an input signal $\mathbf{x} \in \mathbb{R}^T$ and time-varying LPC coefficients $\mathbf{A} \in \mathbb{R}^{T \times N}$ with an order of $N$, the LPC filtering operation is defined as:

```math
\mathbf{y}_t = \mathbf{x}_t - \sum_{i=1}^N \mathbf{A}_{t,i} \mathbf{y}_{t-i}.
```

It's still in early development, so please open an issue if you find any bugs.

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

## Derivation of the gradients of the LPC filtering operation

Will (not) be added soon... I'm not good at math :sweat_smile:.
But the implementation passed both `gradcheck` and `gradgradcheck` tests, so I think it's 99.99% correct and workable :laughing:.
The algorithm is extended from my recent paper **GOLF**[^1].

[^1]: [Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables](https://arxiv.org/abs/2306.17252).

## TODO

- [ ] Use PyTorch C++ extension for faster computation.
- [ ] Use native CUDA kernels for GPU computation.
- [ ] Add examples.

## Citation

If you find this repository useful in your research, please cite the repository with the following BibTex entry:

```bibtex
@software{torchlpc,
  author = {Chin-Yun Yu},
  title = {{TorchLPC}: fast, efficient, and differentiable time-varying {LPC} filtering in {PyTorch}},
  year = {2023},
  version = {0.1.0},
  url = {https://github.com/yoyololicon/torchlpc},
}
```
