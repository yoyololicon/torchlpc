# TorchLPC

`torchlpc` provides a PyTorch implementation of the Linear Predictive Coding (LPC) filtering operation, also known as IIR filtering.
It's fast, differentiable, and supports batched inputs with time-varying filter coefficients.
The computation is done as follows:

Given an input signal $`\mathbf{x} \in \mathbb{R}^T`$ and time-varying LPC coefficients $`\mathbf{A} \in \mathbb{R}^{T \times N}`$ with an order of $`N`$, the LPC filtering operation is defined as:

$$
y_t = x_t - \sum_{i=1}^N A_{t,i} y_{t-i}.
$$

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

~~Will (not) be added soon... I'm not good at math :sweat_smile:.
But the implementation passed both `gradcheck` and `gradgradcheck` tests, so I think it's 99.99% correct and workable :laughing:.~~

To make the filter be differentiable and efficient at the same time, I derived the close formulation of backpropagating gradients through a time-varying IIR filter and used non-differentiable fast IIR filters for both forward and backward computation.
The algorithm is extended from my recent paper **GOLF**[^1].


In the following derivations, I'll assume $x_t$, $y_t$, and $A_{t, :}$ are zeros for $t \leq 0, t > T$ cuz we're dealing with finite signal.
$\mathcal{L}$ represents the loss evaluated with a chosen function.


### Propagating gradients to the input $x_t$

Firstly, let me introduce $`\hat{A}_{t,i}  = -A_{t,i}`$ so we can get rid of the (a bit annoying) minus sign and write the filtering as (equation 1):
```math
y_t = x_t + \sum_{i=1}^N \hat{A}_{t,i} y_{t-i}.
```
The time-varying IIR filtering is equivalent to the following time-varying FIR filtering (equation 2):
```math
y_t = x_t + \sum_{i=1}^{t-1} B_{t,i} x_{t-i},
```
```math
B_{t,i} 
= \sum_{j=1}^i 
\sum_{\{\mathbf{\alpha}: \mathbf{\alpha} \in {\mathbb{Z}}^{j+1}, \alpha_1 = 0, i \geq \alpha_k|_{k > 1} \geq 1, \sum_{k=1}^j \alpha_k = i\}}
\prod_{k=1}^j \hat{A}_{t - \sum_{l=1}^k\alpha_{l}, \alpha_{k+1}}.
```
(The exact value of $`B_{t,i}`$ is just for completeness and doesn't matter for the following proof.)

It's clear that $`\frac{\partial y_t}{\partial x_l}|_{l < t} = B_{t, t-l}`$ and $\frac{\partial y_t}{\partial x_t} = 1$.
Our target, $\frac{\partial \mathcal{L}}{\partial x_t}$, depends on all future outputs $y_{t+i}|_{i \geq 1}$, thus, equals to (equation 3)
```math
\frac{\partial \mathcal{L}}{\partial x_t} 
= \frac{\partial \mathcal{L}}{\partial y_t}
+ \sum_{i=1}^{T - t} \frac{\partial \mathcal{L}}{\partial y_{t+i}} \frac{\partial y_{t+i}}{\partial x_t} \\
= \frac{\partial \mathcal{L}}{\partial y_t}
+ \sum_{i = 1}^{T - t} \frac{\partial \mathcal{L}}{\partial y_{t+i}} B_{t+i,i}.
```
Interestingly, the above equation equals and behaves the same as the time-varying FIR form (Eq. 2) if $x_t := \frac{\partial \mathcal{L}}{\partial y_{T - t + 1}}$ and $B_{t, i} := B_{t + i, i}$, implies that 
```math
\frac{\partial \mathcal{L}}{\partial x_t} 
= \frac{\partial \mathcal{L}}{\partial y_t}
+ \sum_{i=1}^{T - t} \hat{A}_{t+i,i} \frac{\partial \mathcal{L}}{\partial x_{t+i}}.
```

In summary, getting the gradients for the time-varying IIR filter input is easy as filtering the backpropagated gradients backwards with the coefficient matrix shifted column-wised.

### Propagating gradients to the coefficients $\mathbf{A}$

My explanation of this is based on a high-level view of backpropagation.

In each step $t$, we feed two types of inputs to the system.
One is $x_t$, the others are $`\hat{A}_{t,1}y_{t-1}, \hat{A}_{t,2}y_{t-2} \dots`$.
Clearly, the gradients arrived at $t$ are the same for all inputs ($` \frac{\partial \mathcal{L}}{\partial \hat{A}_{t,i}y_{t-i}}|_{1 \leq i \leq N} = \frac{\partial \mathcal{L}}{\partial x_t}`$).
Thus, 

```math
\frac{\partial \mathcal{L}}{\partial A_{t,i}}
= \frac{\partial \mathcal{L}}{\partial \hat{A}_{t,i}y_{t-i}}
\frac{\partial \hat{A}_{t,i}y_{t-i}}{\partial \hat{A}_{t,i}}
\frac{\partial \hat{A}_{t,i}}{\partial A_{t,i}}
= -\frac{\partial \mathcal{L}}{\partial x_t} y_{t-i}.
```

We don't need to evaluate $`\frac{\partial y_{t-i}}{\partial \hat{A}_{t,i}}`$ because of causality.
In summary, the whole backpropagation runs as the following.
It uses the same filter coefficients to get $`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}`$ first, and then $`\frac{\partial \mathcal{L}}{\partial \mathbf{A}}`$ is simply doing matrices multiplication $`-\mathbf{D}_{\frac{\partial \mathcal{L}}{\partial \mathbf{x}}} \mathbf{Y} `$ where

```math
\mathbf{D}_{\frac{\partial \mathcal{L}}{\partial \mathbf{x}}} = 
\begin{vmatrix}
\frac{\partial \mathcal{L}}{\partial x_1} & 0 & \dots & 0 \\
0 & \frac{\partial \mathcal{L}}{\partial x_2} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \frac{\partial \mathcal{L}}{\partial x_t}
\end{vmatrix}
,
\mathbf{Y} = 
\begin{vmatrix}
y_1 & y_0 & \dots & y_{-N + 1} \\
y_2 & y_1 & \dots & y_{-N + 2} \\
\vdots & \vdots & \ddots & \vdots \\
y_T & y_{T - 1} & \dots & y_{T - N}
\end{vmatrix}
.
```

### Gradients for the initial condition $y_t|_{t \leq 0}$

The algorithm could be extended for modelling initial conditions based on the same idea from the previous [section](#propagating-gradients-to-the-coefficients).
The initial conditions are the inputs to the system when $t \leq 0$, so their gradients equal $`\frac{\partial \mathcal{L}}{\partial x_t}|_{-N < t \leq 0}`$. 
You can imaginate that $`x_t|_{1 \leq t \leq T}`$ just represent a segment of the whole signal $x_t$ and $y_t|_{t \leq 0}$ are the system outputs based on $`x_t|_{t \leq 0}`$.
The [initial rest condition](#derivation-of-the-gradients-of-the-lpc-filtering-operation) still holds but happens somewhere $t \leq -N$.
In practice, running the backward filter for $N$ more steps at the end, then we get the gradients.

### Time-invariant filtering

In the time-invariant setting, $`A_{t', i} = A_{t, i} \forall t, t' \in [1, T]`$ and the filtering operation is simplified to

```math
y_t = x_t - \sum_{i=1}^N a_i y_{t-i}, \mathbf{a} = A_{1,:}.
```

The gradients $`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}`$ are filtering $`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}`$ with $\mathbf{a}$ backwards in time, same as in the time-varying case.
For $`\frac{\partial \mathcal{L}}{\partial \mathbf{a}}`$, instead of matrices multiplication, we do a vecotr-matrix multiplication $`-\frac{\partial \mathcal{L}}{\partial \mathbf{x}} \mathbf{Y}`$.
You can think of the difference as summarising the gradients for $a_i$ at all the time steps thus eliminating the time axis.
This algorithm is more efficient than [^1] because it only needs one pass of filtering to get the two gradients while the latter needs two.



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
