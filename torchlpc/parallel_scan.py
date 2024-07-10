from numba import cuda

WARPSIZE = 32

# implementation was translated from https://github.com/eamartin/parallelizing_linear_rnns/blob/master/linear_recurrent_net/linear_recurrence.cu


@cuda.jit(device=True)
def divide_work(n_jobs, n_workers, worker_idx) -> tuple:
    cd = (n_jobs + n_workers - 1) // n_workers
    d, doing_cd = divmod(n_jobs, n_workers)
    if worker_idx < doing_cd:
        x = cd * worker_idx
        y = x + cd
    else:
        x = cd * doing_cd + d * (worker_idx - doing_cd)
        y = x + d
    return x, y


@cuda.jit(device=True)
def compute_warp_start_stop(blockIdx, warp_idx, n_blocks, n_steps):
    block_start, block_stop = divide_work(n_steps, n_blocks, blockIdx)
    block_jobs = block_stop - block_start

    warp_start, warp_stop = divide_work(block_jobs, WARPSIZE, warp_idx)
    warp_start += block_start
    warp_stop += block_start

    return warp_start, warp_stop


@cuda.jit
def reduction_kernel(
    decay, impulses, initial_state, decay_storage, h_storage, n_dims, n_steps
):
    warp, lane = divmod(cuda.threadIdx.x, WARPSIZE)

    storage_offset = cuda.blockIdx.x * (WARPSIZE + 1)

    warp_start, warp_stop = compute_warp_start_stop(
        cuda.blockIdx.x, lane, cuda.gridDim.x, n_steps
    )

    # reduce within warp
    for i in range(warp, n_dims, (cuda.blockDim.x + WARPSIZE - 1) // WARPSIZE):
        cum_decay = 1.0
        h = 0.0
        if (cuda.blockIdx.x == 0) and (lane == 0):
            h = initial_state[i]

        for t in range(warp_start, warp_stop):
            cum_decay *= decay[i, t]
            h = decay[i, t] * h + impulses[i, t]

        decay_storage[lane + storage_offset, i] = cum_decay
        h_storage[lane + storage_offset, i] = h

    cuda.syncthreads()

    # reduce within block
    for i in range(cuda.threadIdx.x, n_dims, cuda.blockDim.x):
        cum_decay = 1.0
        h = 0.0
        for t in range(storage_offset, storage_offset + WARPSIZE):
            cum_decay *= decay_storage[t, i]
            h = decay_storage[t, i] * h + h_storage[t, i]

        decay_storage[WARPSIZE + storage_offset, i] = cum_decay
        h_storage[WARPSIZE + storage_offset, i] = h


@cuda.jit
def block_scan_kernel(decay_storage, h_storage, n_dims, n_blocks):
    for i in range(
        cuda.grid(1),
        n_dims,
        cuda.gridsize(1),
    ):
        for t in range(1, n_blocks):
            cur_idx = t * (WARPSIZE + 1) + WARPSIZE
            prev_idx = (t - 1) * (WARPSIZE + 1) + WARPSIZE
            h_storage[cur_idx, i] += h_storage[prev_idx, i] * decay_storage[cur_idx, i]
            decay_storage[cur_idx, i] *= decay_storage[prev_idx, i]


@cuda.jit
def warp_scan_kernel(
    decay, impulses, initial_state, out, decay_storage, h_storage, n_dims, n_steps
):
    warp, lane = divmod(cuda.threadIdx.x, WARPSIZE)

    for i in range(cuda.threadIdx.x, n_dims, cuda.blockDim.x):
        offset = cuda.blockIdx.x * (WARPSIZE + 1)
        for cur_idx in range(offset, offset + WARPSIZE):
            if cur_idx == 0:
                continue
            prev_idx = cur_idx - 1
            h_storage[cur_idx, i] = (
                h_storage[prev_idx, i] * decay_storage[cur_idx, i]
                + h_storage[cur_idx, i]
            )
            decay_storage[cur_idx, i] *= decay_storage[prev_idx, i]

    cuda.syncthreads()

    warp_start, warp_stop = compute_warp_start_stop(
        cuda.blockIdx.x, lane, cuda.gridDim.x, n_steps
    )

    # scan within warp
    for i in range(warp, n_dims, (cuda.blockDim.x + WARPSIZE - 1) // WARPSIZE):
        if (cuda.blockIdx.x == 0) and (lane == 0):
            h = initial_state[i]
        else:
            h = h_storage[lane - 1 + cuda.blockIdx.x * (WARPSIZE + 1), i]

        for t in range(warp_start, warp_stop):
            h = decay[i, t] * h + impulses[i, t]
            out[i, t] = h


def compute_linear_recurrence(
    decays, impulses, init_states, out, n_dims: int, n_steps: int
):
    n_blocks = min((n_steps + WARPSIZE - 1) // WARPSIZE, 128)

    reduction_mem_shape = (n_blocks * (WARPSIZE + 1), n_dims)
    decay_storage = cuda.device_array(reduction_mem_shape, dtype=decays.dtype)
    h_storage = cuda.device_array(reduction_mem_shape, dtype=impulses.dtype)

    reduction_kernel[n_blocks, 512](
        decays, impulses, init_states, decay_storage, h_storage, n_dims, n_steps
    )

    block_scan_kernel[n_blocks, 512](decay_storage, h_storage, n_dims, n_blocks)

    warp_scan_kernel[n_blocks, 512](
        decays, impulses, init_states, out, decay_storage, h_storage, n_dims, n_steps
    )


if __name__ == "__main__":
    import numpy as np

    n_dims = 16
    n_steps = 20480
    decays = np.full((n_dims, n_steps), 0.9, dtype=np.float32)
    impulses = np.full((n_dims, n_steps), 0.0, dtype=np.float32)
    impulses[:, 0] = 1.0
    init_states = np.full(n_dims, 0.0, dtype=np.float32)

    decays = cuda.to_device(decays)
    impulses = cuda.to_device(impulses)
    init_states = cuda.to_device(init_states)
    out = cuda.device_array((n_dims, n_steps), dtype=np.float32)

    compute_linear_recurrence(decays, impulses, init_states, out, n_dims, n_steps)

    print(out.copy_to_host())
