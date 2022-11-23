#include "matrix.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

__global__ void cuda_kernel_rand_matrix(float *v, int length, int offset) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= length) return;
    curandState curand_state;
    curand_init(0, id, offset, &curand_state);
    v[id] = curand_uniform(&curand_state);
}

void Matrix::Random() {
    int length = n_ * m_;
    float *dev_v;
    cudaMalloc(&dev_v, sizeof(float) * n_ * m_);
    static int offset = 0;
    cuda_kernel_rand_matrix<<<(length + 127) / 128, 128>>>(dev_v, length, offset);
    cudaMemcpy(v_, dev_v, sizeof(float) * n_ * m_, cudaMemcpyDeviceToHost);
    cudaFree(dev_v);
    offset++;
}

Matrix Matrix::operator*(const Matrix &mat) const {
    auto result = Matrix(n_, mat.m_);
    auto *buffer = new float[m_];
    for (int i = 0; i < n_; i++) {
        for (int k = 0; k < m_; k++) {
            buffer[k] = operator()(k, i);
        }
        for (int j = 0; j < mat.m_; j++) {
            float ans = 0.0f;
            auto *col = mat.GetBuffer() + j * m_;
            for (int k = 0; k < m_; k++) {
                ans += buffer[k] * col[k];
            }
            result(j, i) = ans;
        }
    }
    delete[] buffer;
    return result;
}

__global__ void cuda_kernel_matrix_multiplication(float *dev_res, float *dev_a, float *dev_b, int n, int m, int l) {
    uint32_t idy = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t id = idx * n + idy;

    if (idy >= n || idx >= l) {
        return;
    }
    float ans = 0.0f;
    for (int k = 0; k < m; k++) {
        ans += dev_a[k * n + idy] * dev_b[idx * m + k];
    }
    dev_res[id] = ans;
}

Matrix MatrixMultiplication(const Matrix &m1, const Matrix &m2) {
    auto result = Matrix(m1.n_, m2.m_);
    float *dev_res;
    float *dev_a;
    float *dev_b;
    cudaMalloc(&dev_res, sizeof(float) * m1.n_ * m2.m_);
    cudaMalloc(&dev_a, sizeof(float) * m1.n_ * m1.m_);
    cudaMalloc(&dev_b, sizeof(float) * m2.n_ * m2.m_);
    cudaMemcpy(dev_a, m1.GetBuffer(), sizeof(float) * m1.n_ * m1.m_, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, m2.GetBuffer(), sizeof(float) * m2.n_ * m2.m_, cudaMemcpyHostToDevice);
    cuda_kernel_matrix_multiplication<<<dim3((m1.n_ + 15) / 16, (m2.m_ + 15) / 16, 1), dim3(16, 16, 1)>>>(dev_res,
                                                                                                          dev_a, dev_b,
                                                                                                          m1.n_, m1.m_,
                                                                                                          m2.m_);
    cudaMemcpy(result.GetBuffer(), dev_res, sizeof(float) * m1.n_ * m2.m_, cudaMemcpyDeviceToHost);
    cudaFree(dev_res);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return result;
}

#define BUFFER_LENGTH 352

__global__ void
cuda_kernel_matrix_multiplication_shared(float *dev_res, float *dev_a, float *dev_b, int n, int m, int l) {
    uint32_t idy = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float a_shared_buffer[BUFFER_LENGTH][17];
    __shared__ float b_shared_buffer[BUFFER_LENGTH][17];
    float ans;
    for (int k = 0; k < m; k += BUFFER_LENGTH) {
        for (int i = 0; i < BUFFER_LENGTH; i += 16) {
            if (k + i + threadIdx.y < m && idy < n) {
                a_shared_buffer[i + threadIdx.y][threadIdx.x] = dev_a[(k + i + threadIdx.y) * n + idy];
            } else {
                a_shared_buffer[i + threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        for (int i = 0; i < l; i += 16) {
            int idx = i + threadIdx.y;
            for (int j = 0; j < BUFFER_LENGTH; j += 16) {
                if (k + j + threadIdx.x < m && idx < l) {
                    b_shared_buffer[j + threadIdx.x][threadIdx.y] = dev_b[idx * m + (k + j + threadIdx.x)];
                } else {
                    b_shared_buffer[j + threadIdx.x][threadIdx.y] = 0.0f;
                }
            }
            __syncthreads();
            if (idx < l && idy < n) {
                ans = 0.0f;
                for (int j = 0; j < BUFFER_LENGTH && k + j < m; j++) {
                    ans += a_shared_buffer[j][threadIdx.x] * b_shared_buffer[j][threadIdx.y];
                }
                __syncthreads();
                dev_res[idx * n + idy] += ans;
            }
        }
    }
}

Matrix MatrixMultiplicationShared(const Matrix &m1, const Matrix &m2) {
    auto result = Matrix(m1.n_, m2.m_);
    float *dev_res;
    float *dev_a;
    float *dev_b;
    cudaMalloc(&dev_res, sizeof(float) * m1.n_ * m2.m_);
    cudaMalloc(&dev_a, sizeof(float) * m1.n_ * m1.m_);
    cudaMalloc(&dev_b, sizeof(float) * m2.n_ * m2.m_);
    cudaMemcpy(dev_a, m1.GetBuffer(), sizeof(float) * m1.n_ * m1.m_, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, m2.GetBuffer(), sizeof(float) * m2.n_ * m2.m_, cudaMemcpyHostToDevice);
    cudaMemset(dev_res, 0, sizeof(float) * m1.n_ * m2.m_);
    cuda_kernel_matrix_multiplication_shared<<<dim3((m1.n_ + 15) / 16, 1, 1), dim3(16, 16, 1)>>>(dev_res, dev_a, dev_b,
                                                                                                 m1.n_, m1.m_, m2.m_);
    cudaMemcpy(result.GetBuffer(), dev_res, sizeof(float) * m1.n_ * m2.m_, cudaMemcpyDeviceToHost);
    cudaFree(dev_res);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return result;
}

float MaxDifference(const Matrix &m1, const Matrix &m2) {
    float res = 0.0f;
    for (int y = 0; y < m1.n_; y++) {
        for (int x = 0; x < m1.m_; x++) {
            res = std::max(res, std::abs(m1(x, y) - m2(x, y)));
        }
    }
    return res;
}

Matrix::Matrix() {
    n_ = 0;
    m_ = 0;
    v_ = nullptr;
}

Matrix MatrixMultiplicationCUBLAS(const cublasHandle_t &handle, const Matrix &m1, const Matrix &m2) {
    auto result = Matrix(m1.n_, m2.m_);
    float *dev_res;
    float *dev_a;
    float *dev_b;
    cudaMalloc(&dev_res, sizeof(float) * m1.n_ * m2.m_);
    cudaMalloc(&dev_a, sizeof(float) * m1.n_ * m1.m_);
    cudaMalloc(&dev_b, sizeof(float) * m2.n_ * m2.m_);
    cudaMemcpy(dev_a, m1.GetBuffer(), sizeof(float) * m1.n_ * m1.m_, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, m2.GetBuffer(), sizeof(float) * m2.n_ * m2.m_, cudaMemcpyHostToDevice);
    cudaMemset(dev_res, 0, sizeof(float) * m1.n_ * m2.m_);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1.n_, m2.m_, m1.m_, &alpha, dev_a, m1.n_, dev_b, m2.n_, &beta,
                dev_res, m1.n_);
    cudaMemcpy(result.GetBuffer(), dev_res, sizeof(float) * m1.n_ * m2.m_, cudaMemcpyDeviceToHost);
    cudaFree(dev_res);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return result;
}

__global__ void add_kernel(const int *A, const int *B, int *C) {
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main() {
    int A[] = {1, 2, 3, 4, 5};
    int B[] = {10, 20, 30, 40, 50};
    int C[5];
    int *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, sizeof(float) * 5);
    cudaMalloc(&dev_B, sizeof(float) * 5);
    cudaMalloc(&dev_C, sizeof(float) * 5);
    cudaMemcpy(dev_A, A, sizeof(float) * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(float) * 5, cudaMemcpyHostToDevice);
    add_kernel<<<1, 5>>>(dev_A, dev_B, dev_C);
    cudaMemcpy(C, dev_C, sizeof(float) * 5, cudaMemcpyDeviceToHost);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    for (int i = 0; i < 5; i++) {
        printf("%d, ", C[i]);
    }
}