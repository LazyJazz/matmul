#pragma once
#include "matrix.h"
#include "cstring"

Matrix::Matrix(int n, int m) {
    n_ = n;
    m_ = m;
    v_ = new float [n * m];
}

Matrix::~Matrix() {
    delete v_;
}

Matrix::Matrix(const Matrix &mat) {
    Clear();
    n_ = mat.n_;
    m_ = mat.m_;
    v_ = new float [n_ * m_];
    std::memcpy(v_, mat.v_, sizeof (float) * n_ * m_);
}

Matrix::Matrix(Matrix && mat) noexcept {
    Clear();
    n_ = mat.n_;
    m_ = mat.m_;
    v_ = mat.v_;
    mat.v_ = nullptr;
}

float &Matrix::operator()(int x, int y) {
    return v_[x * n_ + y];
}

const float &Matrix::operator()(int x, int y) const {
    return v_[x * n_ + y];
}

float *Matrix::GetBuffer() {
    return v_;
}

const float *Matrix::GetBuffer() const {
    return v_;
}

void Matrix::Clear() {
    delete v_;
    v_ = nullptr;
}

Matrix &Matrix::operator=(const Matrix &mat) {
    if (this == &mat) {
        return *this;
    }
    n_ = mat.n_;
    m_ = mat.m_;
    auto new_v_ = new float [n_ * m_];
    std::memcpy(new_v_, mat.v_, sizeof (float) * n_ * m_);
    Clear();
    v_ = new_v_;
    return *this;
}

Matrix &Matrix::operator=(Matrix &&mat)  noexcept {
    Clear();
    n_ = mat.n_;
    m_ = mat.m_;
    v_ = mat.v_;
    mat.v_ = nullptr;
    return *this;
}

std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
    for (int i = 0; i < mat.n_; i++) {
        if (i) {
            os << " [";
        } else {
            os << "[[";
        }
        for (int j = 0; j < mat.m_; j++) {
            if (j) {
                os << ", ";
            }
            os << std::to_string(mat(j, i));
        }
        if (i < mat.n_ - 1) {
            os << "] \n";
        } else {
            os << "]]\n";
        }
    }
    return os;
}