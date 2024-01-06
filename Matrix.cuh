#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <stdexcept>
#include <math.h>

int blockSize = 256;

class Matrix
{
public:
    int rows, cols;
    float* data;

    // init types
    enum class InitType
    {
        Identity,
        Random,
        Xavier,
        He
    };

    // init methods
    Matrix(int r, int c);
    Matrix(int r, int c, float val);
    Matrix(int r, int c, const float* input_arr);
    Matrix(int r, int c, InitType type);

    // rule of 5
    ~Matrix();
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // member functions
    void print();
    Matrix T() const;
    Matrix relu() const;
    Matrix exp() const;
    Matrix log() const;

    // operator overloads
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(const float scalar) const;
    Matrix& operator*=(const float scalar);
    Matrix operator/(const float scalar) const;
    Matrix& operator/=(const float scalar);
};

// non-member functions
Matrix operator*(const float scalar, const Matrix mat);
Matrix operator/(const float scalar, const Matrix mat);
Matrix had(const Matrix mat1, const Matrix mat2);
Matrix avgToColumn(const Matrix mat);
Matrix fromCSV(std::string path);

// CUDA kernels
__global__ void fill(float* data, float val, int rows, int cols);
__global__ void diagfill(float* data, int n, float val);
__global__ void matrixAdd(float* first, float* second, float* result, int rows, int cols);
__global__ void matrixSub(float* first, float* second, float* result, int rows, int cols);
__global__ void matrixDot(float* first, float* second, float* result, int cols1, int cols2, int rows);
__global__ void matrixScalarMult(float* mat, float scalar, float* result, int rows, int cols);
__global__ void matrixScalarDiv(float* mat, float scalar, float* result, int rows, int cols);
__global__ void matrixTranspose(float* src, float* dest, int rows, int cols);
__global__ void matrixRELU(float* src, float* dest, int size);
__global__ void matrixExp(float* src, float* dest, int rows, int cols);
__global__ void matrixLog(float* src, float* dest, int rows, int cols);
__global__ void matrixHad(float* src1, float* src2, float* dest, int rows, int cols);
__global__ void gradRELU(float* grad, float* data, float* dest);
__global__ void matrixScalarReciprocal(float scalar, float* data, float* dest, int rows, int cols);
__global__ void avgToColumn(float* src, float* dest, int cols);
