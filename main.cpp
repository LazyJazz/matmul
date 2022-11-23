#include "iostream"
#include "matrix.h"
#include "timer.h"
#include "thread"

#define N 8192
#define M 8192
#define L 8192

int main() {

    Matrix m1(N, M), m2(M, L), m3, m4, m5, m6;
    m1.Random();
    m2.Random();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    {
        Timer timer("CPU Brute Force");
        m3 = m1 * m2;
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    {
        Timer timer("CUDA Brute Force");
        m4 = MatrixMultiplication(m1, m2);
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    {
        Timer timer("CUDA Shared Memory");
        m5 = MatrixMultiplicationShared(m1, m2);
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    cublasHandle_t handle;
    cublasCreate(&handle);
    {
        Timer timer("CUBLAS");
        m6 = MatrixMultiplicationCUBLAS(handle, m1, m2);
    }
    cublasDestroy(handle);
    std::cout << "Maximum difference (CUBLAS, Brute Force): " << MaxDifference(m6, m4) << std::endl;
    std::cout << "Maximum difference (CUBLAS, Shared Memory): " << MaxDifference(m6, m5) << std::endl;
}