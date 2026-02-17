import time

import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


CUDA_SRC = r"""
__global__ void vector_add(int n, const float *a, const float *b, float *c) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

__global__ void matrix_add(int rows, int cols, const float *a, const float *b, float *c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void mandelbrot_kernel(
    int width,
    int height,
    int max_iter,
    float x_min,
    float x_max,
    float y_min,
    float y_max,
    int *out
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float cr = x_min + (x_max - x_min) * ((float)x / (float)(width - 1));
    float ci = y_min + (y_max - y_min) * ((float)y / (float)(height - 1));

    float zr = 0.0f;
    float zi = 0.0f;
    int iter = 0;

    while (iter < max_iter) {
        float zr2 = zr * zr - zi * zi + cr;
        float zi2 = 2.0f * zr * zi + ci;
        zr = zr2;
        zi = zi2;
        if (zr * zr + zi * zi > 4.0f) break;
        iter++;
    }

    int idx = y * width + x;
    if (iter == max_iter) {
        out[idx] = max_iter + 5;
    } else {
        out[idx] = (100 * iter) / max_iter;
    }
}
"""


def build_kernels():
    mod = SourceModule(CUDA_SRC)
    return (
        mod.get_function("vector_add"),
        mod.get_function("matrix_add"),
        mod.get_function("mandelbrot_kernel"),
    )


def run_vector_add(vector_add_fn):
    n = 1_000_000
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.empty_like(a)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    block = (256, 1, 1)
    grid = ((n + block[0] - 1) // block[0], 1, 1)
    vector_add_fn(np.int32(n), a_gpu, b_gpu, c_gpu, block=block, grid=grid)
    cuda.memcpy_dtoh(c, c_gpu)

    np.testing.assert_allclose(c, a + b, rtol=1e-5, atol=1e-6)
    print(f"[OK] vector_add n={n}, grid={grid[0]}, block={block[0]}")


def run_matrix_add(matrix_add_fn):
    rows, cols = 1024, 1024
    a = np.random.randn(rows, cols).astype(np.float32)
    b = np.random.randn(rows, cols).astype(np.float32)
    c = np.empty_like(a)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    block = (16, 16, 1)
    grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1], 1)
    matrix_add_fn(
        np.int32(rows),
        np.int32(cols),
        a_gpu,
        b_gpu,
        c_gpu,
        block=block,
        grid=grid,
    )
    cuda.memcpy_dtoh(c, c_gpu)

    np.testing.assert_allclose(c, a + b, rtol=1e-5, atol=1e-6)
    print(f"[OK] matrix_add shape={rows}x{cols}, grid={grid[:2]}, block={block[:2]}")


def run_mandelbrot(mandelbrot_fn):
    width, height = 1500, 1500
    max_iter = 1000
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -1.5, 1.5

    out = np.empty((height, width), dtype=np.int32)
    out_gpu = cuda.mem_alloc(out.nbytes)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1], 1)

    start = time.time()
    mandelbrot_fn(
        np.int32(width),
        np.int32(height),
        np.int32(max_iter),
        np.float32(x_min),
        np.float32(x_max),
        np.float32(y_min),
        np.float32(y_max),
        out_gpu,
        block=block,
        grid=grid,
    )
    cuda.memcpy_dtoh(out, out_gpu)
    end = time.time()

    print(f"[OK] mandelbrot GPU time: {end - start:.4f} s")

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    plt.rcParams["figure.figsize"] = [12, 7.5]
    plt.contourf(c.real, c.imag, out)
    plt.xlabel("Real($c$)")
    plt.ylabel("Imag($c$)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig("tp5/plot_mandelbrot_gpu.png")
    plt.show()


def main():
    np.random.seed(1729)
    vector_add_fn, matrix_add_fn, mandelbrot_fn = build_kernels()
    run_vector_add(vector_add_fn)
    run_matrix_add(matrix_add_fn)
    run_mandelbrot(mandelbrot_fn)


if __name__ == "__main__":
    main()
