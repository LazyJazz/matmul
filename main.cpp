#include "iostream"
#include "matrix.h"
#include "Timer.h"

#define N 8192
#define M 8192
#define L 8192

int main() {
    Matrix m1(N, M), m2(M, L), m3, m4, m5, m6;
    m1.Random();
    m2.Random();
//    {
//        Timer timer("CPU Brute Force");
//        m3 = m1 * m2;
//    }
    {
        Timer timer("CUDA Brute Force");
        m4 = MatrixMultiplication(m1, m2);
    }
    {
        Timer timer("CUDA Shared Memory");
        m5 = MatrixMultiplicationShared(m1, m2);
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    {
        Timer timer("CUBLAS");
        m6 = MatrixMultiplicationCUBLAS(handle, m1, m2);
    }
    cublasDestroy(handle);
    std::cout << MaxDifference(m6, m4) << std::endl;
    std::cout << MaxDifference(m6, m5) << std::endl;
}