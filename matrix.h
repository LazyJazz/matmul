#pragma once
#include "iostream"
#include "string"

#include "cublas.h"
#include "cublas_v2.h"
#include "cublas_api.h"

class Matrix {
public:
    Matrix();
    Matrix(int n, int m);
    Matrix(Matrix &&) noexcept ;
    Matrix(const Matrix& mat);
    ~Matrix();
    float& operator()(int x, int y);
    const float& operator()(int x, int y) const;
    float* GetBuffer();
    [[nodiscard]] const float* GetBuffer() const;
    Matrix& operator = (const Matrix& mat);
    Matrix& operator = (Matrix&& mat) noexcept;
    friend std::ostream& operator << (std::ostream& os, const Matrix& mat);
    void Random();
    Matrix operator* (const Matrix& mat) const;
    friend Matrix MatrixMultiplication(const Matrix& m1, const Matrix& m2);
    friend Matrix MatrixMultiplicationShared(const Matrix& m1, const Matrix& m2);
    friend Matrix MatrixMultiplicationCUBLAS(const cublasHandle_t &handle, const Matrix &m1, const Matrix &m2);
    friend float MaxDifference(const Matrix& m1, const Matrix& m2);
private:
    void Clear();
    int n_{0};
    int m_{0};
    float *v_{nullptr};
};

Matrix MatrixMultiplication(const Matrix& m1, const Matrix& m2);
Matrix MatrixMultiplicationShared(const Matrix& m1, const Matrix& m2);
float MaxDifference(const Matrix& m1, const Matrix& m2);