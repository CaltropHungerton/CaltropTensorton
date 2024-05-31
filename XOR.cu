#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <stdexcept>
#include <math.h>
#include <vector>
#include <set>
#include <functional>
#include <memory>

int blockSize = 256; // TODO experiment with other sizes like 512, etc.

/*
get shape/dims?
matrix norms
*/

__global__ void fill(float* data, float val, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols)
    {
        data[idx] = val;
    }
}

__global__ void diagfill(float* data, int rows, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[(idx * rows) + idx] = val;
}

__global__ void matrixAdd(float* first, float* second, float* result, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols)
    {
        result[idx] = first[idx] + second[idx];
    }
}

__global__ void matrixSub(float* first, float* second, float* result, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols)
    {
        result[idx] = first[idx] - second[idx];
    }
}

__global__ void broadcastAdd(float* first, float* second, float* result, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols)
    {
        result[idx] = first[idx] + second[idx / cols];
    }
}

__global__ void broadcastSub(float* first, float* second, float* result, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols)
    {
        result[idx] = first[idx] - second[idx / cols];
    }
}

__global__ void matrixDot(float* first, float* second, float* result, int cols1, int cols2, int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols2)
    {
        for (int i = 0; i < cols1; i++)
        {
            result[idx] += first[((idx / cols2) * cols1) + i] * second[(i * cols2) + (idx % cols2)]; // TODO optimize with tiling/whatever for efficiency/cache usage
        }
    }
}

__global__ void matrixScalarMult(float* mat, float scalar, float* result, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        result[idx] = mat[idx] * scalar;
    }
}

__global__ void matrixScalarDiv(float* mat, float scalar, float* result, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        result[idx] = mat[idx] / scalar;
    }
}

__global__ void matrixTranspose(float* src, float* dest, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        int x = idx / cols;
        int y = idx % cols;
        dest[(y * rows) + x] = src[idx];
    }
}

__global__ void matrixRELU(float* src, float* dest, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (src[idx] < 0)
        {
            dest[idx] = 0;
        }
        else
        {
            dest[idx] = src[idx];
        }
    }
}

__global__ void matrixExp(float* src, float* dest, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        dest[idx] = exp(src[idx]);
    }
}

__global__ void matrixLog(float* src, float* dest, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        dest[idx] = log2(src[idx]);
    }
}

__global__ void matrixHad(float* src1, float* src2, float* dest, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        dest[idx] = src1[idx] * src2[idx];
    }
}

__global__ void gradRELU(float* grad, float* data, float* dest) // bespoke function for relu backprop
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data[idx] >= 0)
    {
        dest[idx] += grad[idx];
    }
}

__global__ void matrixScalarReciprocal(float scalar, float* data, float* dest, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        dest[idx] = scalar / data[idx];
    }
}

__global__ void avgToColumn(float* src, float* dest, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < cols; i++)
    {
        dest[idx] += src[idx * cols + i];
    }
    dest[idx] /= cols;
}

// newaddition
__global__ void sumToColumn(float* src, float* dest, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < cols; i++)
    {
        dest[idx] += src[idx * cols + i];
    }
}

class Matrix
{
public:
    int rows, cols;
    float* data;

    enum class InitType
    {
        Identity,
        Random,
        Xavier,
        He
    };

    Matrix(int r, int c) : rows(r), cols(c), data(0) {}

    Matrix(int r, int c, float val) : rows(r), cols(c) // fill with single value
    {
        cudaMalloc(&data, r * c * sizeof(float));
        int numBlocks = ((r * c) + blockSize - 1) / blockSize;

        fill <<<numBlocks, blockSize>>> (data, val, rows, cols);
        cudaDeviceSynchronize();
    }

    Matrix(int r, int c, const float* input_arr) : rows(r), cols(c)
    {
        cudaMalloc(&data, r * c * sizeof(float));
        cudaMemcpy(data, input_arr, r * c * sizeof(float), cudaMemcpyHostToDevice);
    }

    // arg constructor
    Matrix(int r, int c, InitType type) : rows(r), cols(c)
    {
        if (r < 1 || c < 1)
        {
            throw std::invalid_argument("Matrix dimensions cannot be smaller than 1.");
        }
        if (type == Matrix::InitType::Identity)
        {
            if (r == c)
            {
                cudaMalloc(&data, r * c * sizeof(float));

                int numBlocks = ((r * c) + blockSize - 1) / blockSize;

                fill << <numBlocks, blockSize >> > (data, 0, rows, cols);
                cudaDeviceSynchronize();
                diagfill << <1, rows >> > (data, rows, 1); // TODO possibly extend for matrices with more than 1024 rows (larger than block size)
                cudaDeviceSynchronize();
            }
            else
            {
                throw std::invalid_argument("Matrix must be square to be an identity matrix.");
            }
        }

        else if (type == Matrix::InitType::Random)
        {
            float* hostData = new float[r * c];

            std::default_random_engine generator;
            std::normal_distribution<float> distribution(0, sqrt(.01));

            for (int i = 0; i < r * c; i++)
            {
                hostData[i] = distribution(generator);
            }

            cudaMalloc(&data, r * c * sizeof(float));
            cudaMemcpy(data, hostData, r * c * sizeof(float), cudaMemcpyHostToDevice);

            delete[] hostData;
        }

        else if (type == Matrix::InitType::Xavier)
        {
            float* hostData = new float[r * c];

            std::default_random_engine generator;
            std::normal_distribution<float> distribution(0, sqrt(2 / float(r + c)));

            for (int i = 0; i < r * c; i++)
            {
                hostData[i] = distribution(generator);
            }

            cudaMalloc(&data, r * c * sizeof(float));
            cudaMemcpy(data, hostData, r * c * sizeof(float), cudaMemcpyHostToDevice);

            delete[] hostData;
        }

        else if (type == Matrix::InitType::He)
        {
            float* hostData = new float[r * c];

            std::default_random_engine generator;
            std::normal_distribution<float> distribution(0, sqrt(2 / float(c)));

            for (int i = 0; i < r * c; i++)
            {
                hostData[i] = distribution(generator);
            }

            cudaMalloc(&data, r * c * sizeof(float));
            cudaMemcpy(data, hostData, r * c * sizeof(float), cudaMemcpyHostToDevice);

            delete[] hostData;
        }
    }

    // destructor
    ~Matrix()
    {
        cudaFree(data);
    }

    // move constructor
    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data)
    {
        other.data = nullptr;
    }

    // move assignment operator
    Matrix& operator=(Matrix&& other) noexcept
    {
        if (this != &other)
        {
            cudaFree(data);

            rows = other.rows;
            cols = other.cols;
            data = other.data;

            other.data = nullptr;
        }
        return *this;
    }

    // copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols)
    {
        cudaMalloc(&data, rows * cols * sizeof(float));
        cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // copy assignment operator
    Matrix& operator=(const Matrix& other)
    {
        if (this != &other)
        {
            cudaFree(data);

            rows = other.rows;
            cols = other.cols;

            cudaMalloc(&data, rows * cols * sizeof(float));
            cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    void print() const
    {
        float* dup = (float*)malloc(rows * cols * sizeof(float));
        cudaMemcpy(dup, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "////////////////////////////////////////\n";

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << dup[(i * cols) + j] << "  ";
            }
            std::cout << "\n";
        }

        std::cout << "////////////////////////////////////////\n";

        free(dup);
    }

    Matrix operator+(const Matrix& other) const
    {
        if (this->cols != other.cols || this->rows != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        Matrix result(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixAdd << <numBlocks, blockSize >> > (this->data, other.data, result.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return result;
    }

    Matrix operator-(const Matrix& other) const
    {
        if (this->cols != other.cols || this->rows != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }

        Matrix result(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixSub << <numBlocks, blockSize >> > (this->data, other.data, result.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return result;
    }

    Matrix& operator+=(const Matrix& other)
    {
        if (this->cols != other.cols || this->rows != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixAdd << <numBlocks, blockSize >> > (this->data, other.data, this->data, this->rows, this->cols); // calling regular add kernel but w/ left matrix as result
        cudaDeviceSynchronize();

        return *this;
    }

    Matrix& operator-=(const Matrix& other)
    {
        if (this->cols != other.cols || this->rows != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixSub << <numBlocks, blockSize >> > (this->data, other.data, this->data, this->rows, this->cols); // calling regular sub kernel but w/ left matrix as result
        cudaDeviceSynchronize();

        return *this;
    }

    Matrix operator*(const Matrix& other) const
    {
        if (this->cols != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix result(this->rows, other.cols);
        cudaMalloc(&result.data, this->rows * other.cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * other.cols * sizeof(float));

        int numBlocks = ((this->rows * other.cols) + blockSize - 1) / blockSize;

        matrixDot << <numBlocks, blockSize >> > (this->data, other.data, result.data, this->cols, other.cols, this->rows);
        cudaDeviceSynchronize();

        return result;
    }

    // Matrix scalar multiplication!!
    Matrix operator*(const float scalar) const
    {
        Matrix result(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixScalarMult <<<numBlocks, blockSize>>> (this->data, scalar, result.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return result;
    }

    Matrix& operator*=(const float scalar)
    {
        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixScalarMult << <numBlocks, blockSize >> > (this->data, scalar, this->data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return *this;
    }

    Matrix operator/(const float scalar) const
    {
        Matrix result(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixScalarDiv << <numBlocks, blockSize >> > (this->data, scalar, result.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return result;
    }

    Matrix& operator/=(const float scalar)
    {
        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixScalarDiv << <numBlocks, blockSize >> > (this->data, scalar, this->data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return *this;
    }

    Matrix T() const
    {
        Matrix transposed = Matrix(this->cols, this->rows);
        cudaMalloc(&transposed.data, this->cols * this->rows * sizeof(float));
        cudaMemset(transposed.data, 0, this->cols * this->rows * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixTranspose << <numBlocks, blockSize >> > (this->data, transposed.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return transposed;
    }

    Matrix relu() const
    {
        Matrix result = Matrix(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixRELU << <numBlocks, blockSize >> > (this->data, result.data, this->rows * this->cols);
        cudaDeviceSynchronize();

        return result;
    }

    Matrix exp() const
    {
        Matrix result = Matrix(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixExp << <numBlocks, blockSize >> > (this->data, result.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return result;
    }

    Matrix log() const
    {
        Matrix result = Matrix(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(float));
        cudaMemset(result.data, 0, this->rows * this->cols * sizeof(float));

        int numBlocks = ((this->rows * this->cols) + blockSize - 1) / blockSize;

        matrixLog << <numBlocks, blockSize >> > (this->data, result.data, this->rows, this->cols);
        cudaDeviceSynchronize();

        return result;
    }
};

// global non-member function for making matrix-scalar multiplication commutative
Matrix operator*(const float scalar, const Matrix mat)
{
    return mat * scalar;
}

Matrix operator/(const float scalar, const Matrix mat)
{
    Matrix result(mat.rows, mat.cols);
    cudaMalloc(&result.data, mat.rows * mat.cols * sizeof(float));
    cudaMemset(result.data, 0, mat.rows * mat.cols * sizeof(float));

    int numBlocks = ((mat.rows * mat.cols) + blockSize - 1) / blockSize;

    matrixScalarReciprocal << <numBlocks, blockSize >> > (scalar, mat.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix broadcastAdd(const Matrix mat, const Matrix vec)
{
    if (mat.cols != vec.rows && vec.cols != 1)
    {
        throw std::invalid_argument("Matrix dimensions do not match for broadcast addition.");
    }

    Matrix result = Matrix(mat.rows, mat.cols);
    cudaMalloc(&result.data, mat.rows * mat.cols * sizeof(float));
    cudaMemset(result.data, 0, mat.rows * mat.cols * sizeof(float));

    int numBlocks = ((mat.rows * mat.cols) + blockSize - 1) / blockSize;

    broadcastAdd << <numBlocks, blockSize >> > (mat.data, vec.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix broadcastSub(const Matrix mat, const Matrix vec)
{
    if (mat.cols != vec.rows && vec.cols != 1)
    {
        throw std::invalid_argument("Matrix dimensions do not match for broadcast subtraction.");
    }

    Matrix result = Matrix(mat.rows, mat.cols);
    cudaMalloc(&result.data, mat.rows * mat.cols * sizeof(float));
    cudaMemset(result.data, 0, mat.rows * mat.cols * sizeof(float));

    int numBlocks = ((mat.rows * mat.cols) + blockSize - 1) / blockSize;

    broadcastSub << <numBlocks, blockSize >> > (mat.data, vec.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();

    return result;
}

// hadamard product (element-wise matrix multiplication)
Matrix had(const Matrix mat1, const Matrix mat2)
{
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for hadamard product.");
    }
    Matrix result = Matrix(mat1.rows, mat1.cols);
    cudaMalloc(&result.data, mat1.rows * mat1.cols * sizeof(float));
    cudaMemset(result.data, 0, mat1.rows * mat1.cols * sizeof(float));

    int numBlocks = ((mat1.rows * mat1.cols) + blockSize - 1) / blockSize;

    matrixHad << <numBlocks, blockSize >> > (mat1.data, mat2.data, result.data, mat1.rows, mat1.cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix avgToColumn(const Matrix mat) // TODO fix for matrices larger than blocksize
{
    Matrix result = Matrix(mat.rows, 1, 0.0f);
    cudaMalloc(&result.data, mat.rows * sizeof(float));
    cudaMemset(result.data, 0, mat.rows * sizeof(float));

    avgToColumn << < 1, mat.rows >> > (mat.data, result.data, mat.cols);
    cudaDeviceSynchronize();

    return result;
}

// newaddition
Matrix sumToColumn(const Matrix mat) // TODO fix for matrices larger than blocksize
{
    Matrix result = Matrix(mat.rows, 1, 0.0f);
    cudaMalloc(&result.data, mat.rows * sizeof(float));
    cudaMemset(result.data, 0, mat.rows * sizeof(float));

    sumToColumn << <1, mat.rows >> > (mat.data, result.data, mat.cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix fromCSV(std::string path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::invalid_argument("Invalid path or file does not exist: " + path);
    }
    // assumes the csv file is all digits, check if it's rectangular, then put everything into float array, call constructor, return
    std::string line;
    std::vector<float> values;
    int rows = 0;
    int cols = -1;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> rowValues;

        while (std::getline(ss, cell, ','))
        {
            rowValues.push_back(std::stof(cell));
        }

        if (cols == -1)
        {
            cols = rowValues.size();
        }
        else if (rowValues.size() != cols)
        {
            throw std::invalid_argument("Inconsistent number of columns at line " + std::to_string(rows + 1));
        }

        values.insert(values.end(), rowValues.begin(), rowValues.end());
        rows++;
    }

    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Empty CSV file or invalid content in file: " + path);
    }
    return Matrix(rows, cols, values.data());
}

// TODO applying (atomic) math functions, saving matrices, loading matrices, more inplace functions (for example, distinguish using f(), f_() maybe)
// probably not necessary given that we already have move semantics in place
// long term: literally make AUTODIFF for backpropagation, infrastructure for mini-batch inference/gradient descent, other stuff necessary for
// moderately-fledged NN library, make OOP stuff more encapsulated
//

// i can do stack arrays of whatever dimension just fine, just need to pass pointer to array into the constructor
// heap arrays: should be allocated as 1d. i can make that happen with the dataloader helper functions. can't think of any other instance where that would
// actually be used.

// already did: exp, relu
// list of math functions to implement (sensible derivatives): pow, root, exp, log (just base e), sin, cos, relu, loss function derivatives
// matrix norms (need to take derivatives of these too!)
// hadamard matrix multiplication
// sums, axis sums
// data loaders, whatever would be convenient as an interface for the dataloading functions in the neural network class


// this is where we declare the tensor stuff



class Tensor;
void backwardsAdd(Tensor* tensor);
void backwardsSub(Tensor* tensor);
void backwardsBroadcastAdd(Tensor* tensor);
void backwardsDot(Tensor* tensor);
void backwardsScalarMult(Tensor* tensor);
void backwardsHad(Tensor* tensor);
void backwardsExp(Tensor* tensor);
void backwardsTranspose(Tensor* tensor);
void backwardsRecip(Tensor* tensor);
void backwardsSigmoid(Tensor* tensor);
void backwardsSoftmax(Tensor* tensor);
void backwardsNull(Tensor* tensor);

class Tensor
{
public:
    Matrix value;
    Matrix gradients;
    std::vector<Tensor*> parents;
    void(*backprop)(Tensor*); // initialize this with a default function
    std::string op;
    bool canUpdate;
    float scalar; // for use in gradients where you're just multiplying things by scalars

    Tensor(int r, int c, bool canUpdate) : value(Matrix(r, c, 0.0f)), gradients(Matrix(r, c, 0.0f)), backprop(&backwardsNull), op("n/a"), canUpdate(canUpdate) {}

    Tensor(Matrix m, bool canUpdate) : value(m), gradients(Matrix(m.rows, m.cols, 0.0f)), backprop(&backwardsNull), op("n/a"), canUpdate(canUpdate) {}

    Tensor operator+(Tensor& other)
    {
        Tensor newtensor = Tensor(this->value + other.value, true);
        newtensor.parents.push_back(this);
        newtensor.parents.push_back(&other);
        newtensor.backprop = &backwardsAdd;
        newtensor.op = "add";
        return newtensor;
    }

    Tensor operator-(Tensor& other)
    {
        Tensor newtensor(this->value - other.value, true);
        newtensor.parents.push_back(this);
        newtensor.parents.push_back(&other);
        newtensor.backprop = &backwardsSub;
        newtensor.op = "sub";
        return newtensor;
    }

    Tensor operator*(Tensor& other)
    {
        Tensor newtensor(this->value * other.value, true);
        newtensor.parents.push_back(this);
        newtensor.parents.push_back(&other);
        newtensor.backprop = &backwardsDot;
        newtensor.op = "dot";
        return newtensor;
    }

    Tensor operator*(float scalar)
    {
        Tensor newtensor(this->value * scalar, true);
        newtensor.parents.push_back(this);
        newtensor.backprop = &backwardsScalarMult;
        newtensor.op = "scalarmult";
        newtensor.scalar = scalar;
        return newtensor;
    }

    Tensor operator/(float scalar)
    {
        Tensor newtensor(this->value / scalar, true);
        newtensor.parents.push_back(this);
        newtensor.backprop = &backwardsScalarMult;
        newtensor.op = "scalardiv";
        newtensor.scalar = 1 / scalar;
        return newtensor;
    }

    Tensor exp() 
    {
        Tensor newtensor(this->value.exp(), true);
        newtensor.parents.push_back(this);
        newtensor.backprop = &backwardsExp;
        newtensor.op = "exp";
        return newtensor;
    }

    Tensor T() 
    {
        Tensor newtensor(this->value.T(), true);
        newtensor.parents.push_back(this);
        newtensor.backprop = &backwardsTranspose;
        newtensor.op = "transpose";
        return newtensor;
    }

    Tensor relu() 
    {
        Tensor newtensor(this->value.relu(), true);
        newtensor.parents.push_back(this);
        // newtensor.backprop = &backwardsRelu;
        newtensor.op = "relu";
        return newtensor;
    }

    Tensor sigmoid() 
    {
        Tensor newtensor(1 / (Matrix(this->value.rows, this->value.cols, 1) + (-1 * this->value).exp()), true);
        newtensor.parents.push_back(this);
        newtensor.backprop = &backwardsSigmoid;
        newtensor.op = "sigmoid";
        return newtensor;
    }

    // TODO check all rule of 5 stuff

    ~Tensor() {}

    Tensor(Tensor&& other) noexcept
        : value(std::move(other.value)),
        gradients(std::move(other.gradients)),
        parents(std::move(other.parents)),
        backprop(other.backprop),
        op(std::move(other.op)),
        canUpdate(other.canUpdate),
        scalar(other.scalar)
    {}

    Tensor(const Tensor& other)
        : value(other.value),
        gradients(other.gradients),
        parents(other.parents),
        backprop(other.backprop),
        op(other.op),
        canUpdate(other.canUpdate),
        scalar(other.scalar)
    {}

    Tensor& operator=(Tensor&& other) noexcept
    {
        if (this != &other)
        {
            value = other.value;
            gradients = Matrix(value.rows, value.cols, 0.0f);
            // don't need to change parent vector or anything else really, i know this will only be reinitialized with the same thing
            // should i optimize so that it doesn't remake the parent tensors unnecessarily? that's negligible i guess. inelegant, but ok
        }
        return *this;
    }

    Tensor& operator=(const Tensor& other)
    {
        if (this != &other)
        {
            value = other.value;
            gradients = other.gradients;
            parents = other.parents;
            backprop = other.backprop;
            op = other.op;
            canUpdate = other.canUpdate;
            scalar = other.scalar;
        }
        return *this;
    }

    void print()
    {
        std::cout << "/////////////////////////////////////////////////////\n";
        std::cout << "data:\n";
        value.print();
        std::cout << "gradients:\n";
        gradients.print();
        std::cout << "operation: " << op << "\n";
        std::cout << "/////////////////////////////////////////////////////\n";
    }

    void update(float epsilon)
    {
        if (canUpdate)
        {
            value -= epsilon * gradients;
        }
    }

    void zeroGrads()
    {
        gradients = Matrix(gradients.rows, gradients.cols, 0.0f);
    }
};

void backwardsAdd(Tensor* tensor)
{
    tensor->parents[0]->gradients += tensor->gradients;
    tensor->parents[1]->gradients += tensor->gradients;
}

void backwardsSub(Tensor* tensor)
{
    tensor->parents[0]->gradients += tensor->gradients;
    tensor->parents[1]->gradients -= tensor->gradients;
}

// newaddition
void backwardsBroadcastAdd(Tensor* tensor)
{
    tensor->parents[0]->gradients += tensor->gradients;
    tensor->parents[1]->gradients += sumToColumn(tensor->gradients);
}

void backwardsDot(Tensor* tensor)
{
    tensor->parents[0]->gradients += tensor->gradients * tensor->parents[1]->value.T();
    tensor->parents[1]->gradients += tensor->parents[0]->value.T() * tensor->gradients;
}

void backwardsScalarMult(Tensor* tensor)
{
    tensor->parents[0]->gradients += tensor->gradients * tensor->scalar; // TODO CHECK THIS (probably right)
}

void backwardsHad(Tensor* tensor)
{
    tensor->parents[0]->gradients += had(tensor->gradients, tensor->parents[1]->value);
    tensor->parents[1]->gradients += had(tensor->parents[0]->value, tensor->gradients);
}

void backwardsExp(Tensor* tensor)
{
    tensor->parents[0]->gradients += had(tensor->gradients, tensor->value);
}

void backwardsTranspose(Tensor* tensor) // TODO CHECK THIS (probably right)
{
    tensor->parents[0]->gradients += tensor->gradients.T();
}

/*
void backwardsRelu(Tensor* tensor)
{
    gradRELU <<< 1, tensor->value->rows * tensor->value->cols >>> (tensor->gradients, tensor->value, tensor->parents[0]->gradients); // TODO: make blocks/threads better
    cudaDeviceSynchronize();
}
*/

void backwardsRecip(Tensor* tensor)
{
    tensor->parents[0]->gradients -= had(tensor->gradients, tensor->scalar / had(tensor->parents[0]->value, tensor->parents[0]->value)); // check this
}

void backwardsSigmoid(Tensor* tensor)
{
    Matrix sigmoid_derivative = had(tensor->value, (Matrix(tensor->value.rows, tensor->value.cols, 1) - tensor->value));
    Matrix propagated_gradients = had(tensor->gradients, sigmoid_derivative);
    tensor->parents[0]->gradients += propagated_gradients;
}

// there isn't a backwards crossEntropy function because we did it when we calculated the tensor

void backwardsSoftmax(Tensor* tensor)
{

}

void backwardsNull(Tensor* tensor) {}

/*
Tensor operator*(const float scalar, const Tensor t)
{
    return t * scalar;
}
*/

Tensor crossEntropy(Tensor& prob, Tensor& y)
{
    Tensor newtensor(-1 * (had((broadcastAdd(prob.value, Matrix(1, 1, 1e-6))).log(), y.value)), false);
    prob.gradients = (prob.value - y.value) / float(prob.value.cols);
    //std::cout << "CHECKING PROB GRADIENTS\n";
    //prob.gradients.print();
    newtensor.parents.push_back(&prob);
    newtensor.op = "crossEntropy";
    return newtensor;
}

Tensor broadcastAdd(Tensor& t1, Tensor& t2)
{
    Tensor newtensor(broadcastAdd(t1.value, t2.value), false);
    newtensor.parents.push_back(&t1);
    newtensor.parents.push_back(&t2);
    newtensor.backprop = &backwardsBroadcastAdd;
    newtensor.op = "broadcastAdd";
    return newtensor;
}

void buildTopoSort(Tensor* node, std::set<Tensor*>& visited, std::vector<Tensor*>& topo)
{
    if (node && visited.find(node) == visited.end())
    {
        visited.insert(node);
        for (Tensor* parent : node->parents)
        {
            buildTopoSort(parent, visited, topo);
        }
        topo.push_back(node);
    }
}

std::vector<Tensor*> topologicalSort(Tensor& root)
{
    std::vector<Tensor*> topo;
    std::set<Tensor*> visited;
    buildTopoSort(&root, visited, topo);
    return topo;
}

// i'm going to see whether changing the inputs to just be simple 00, 11, 01, 10 will make things easier. 
// that and changing the initialization of the weights. 

std::pair<std::vector<float>, std::vector<float>> generateXORDataSimple(int n)
{
    std::vector<float> inputs(n * 2);
    std::vector<float> outputs(n);

    for (int i = 0; i < n; ++i)
    {
        inputs[i] = float(i % 2);
        inputs[i + n] = float((i % 4) / 2);

        if (inputs[i] == inputs[i + n])
        {
            outputs[i] = 0.0f;
        } 
        else
        {
            outputs[i] = 1.0f;
        }
    }
    return { inputs, outputs };
}

std::pair<std::vector<float>, std::vector<float>> generateXORData(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);

    std::vector<float> inputs(n * 2);
    std::vector<float> outputs(n);

    for (int i = 0; i < n; ++i) {
        float x = dis(gen);
        float y = dis(gen);
        inputs[i] = x;
        inputs[i + n] = y;

        if ((x < 0 && y >= 0) || (x >= 0 && y < 0)) {
            outputs[i] = 1.0;
        }
        else 
        {
            outputs[i] = 0.0;
        }
    }

    return { inputs, outputs };
}

int main()
{   
    std::cout << "\nhere begins the autodiff tests with xor\n";

    int iters = 10000;
    float lr = 1;

    // Tensor X = Tensor(Matrix(2, 50, Matrix::InitType::Random), false); // (2 x n)
    // Tensor Y = Tensor(Matrix(1, 50), false); // (1 x n)

    auto points = generateXORDataSimple(4);
    Tensor X = Tensor(Matrix(2, 4, points.first.data()), false);
    Tensor Y = Tensor(Matrix(1, 4, points.second.data()), false);
    
    X.print();
    Y.print();
    
    Tensor W1 = Tensor(Matrix(2, 2, Matrix::InitType::Xavier), true);
    Tensor b1 = Tensor(2, 1, true);
    Tensor W2 = Tensor(Matrix(1, 2, Matrix::InitType::Xavier), true);
    Tensor b2 = Tensor(1, 1, true);

    W1.print();
    b1.print();
    W2.print();
    b2.print();
    
    // creating the intermediate tensors
    std::cout << "creating the intermediate tensors\n";
    Tensor J1 = W1 * X;
    W1.print();
    X.print();
    J1.print();

    std::cout << "first matmul\n";
    b1.print();
    Tensor K1 = broadcastAdd(J1, b1);
    K1.print();
    Tensor A1 = K1.sigmoid();
    
    Tensor J2 = W2 * A1;
    Tensor K2 = broadcastAdd(J2, b2);
    Tensor A2 = K2.sigmoid();
    A2.print();

    Tensor loss = crossEntropy(A2, Y);

    loss.print();
    std::cout << "jeff2" << std::endl;

    std::vector<Tensor*> sorted = topologicalSort(loss);

    for (int i = 0; i < iters; i++)
    {
        J1 = W1 * X;
        K1 = broadcastAdd(J1, b1);
        A1 = K1.sigmoid();

        J2 = W2 * A1;
        K2 = broadcastAdd(J2, b2);
        A2 = K2.sigmoid();

        loss = crossEntropy(A2, Y);
        //A2.print();

        for (int j = sorted.size() - 1; j >= 0; j--)
        {
            sorted[j]->backprop(sorted[j]);
            sorted[j]->update(lr);
            sorted[j]->zeroGrads();
        }

        if (i % 100 == 0)
        {
            std::cout << i << "\n";
        }
    }

    A2.print();
    Y.print();

    std::cout << "printing the values of all of the weights\n";

    W1.print();
    b1.print();
    W2.print();
    b2.print();
    std::cout << "\n\n\n\n\n\n";
    
    while (true)
    {
        float x, y;
        std::cout << "Please input XOR values.\n";
        std::cin >> x;
        std::cin >> y;
        float input[2] = { x, y };
        Tensor Xin = Tensor(Matrix(2, 1, input), false);

        Tensor J1 = W1 * Xin;
        Tensor K1 = broadcastAdd(J1, b1);
        Tensor A1 = K1.sigmoid();
        Tensor J2 = W2 * A1;
        Tensor K2 = broadcastAdd(J2, b2);
        Tensor A2 = K2.sigmoid();
        A2.print();
        
        int answer;
        if (x != y)
        {
            answer = 1;
        }
        else
        {
            answer = 0;
        }
        std::cout << answer << " is the answer\n\n";
    }
}
