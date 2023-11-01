#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <stdexcept>

/*
make 2d matrix of various dimensions (generalize later, this is just a first attempt)
initializations with data/arrays, from file
get shape/dims
matrix norms
*/

__global__ void fill(float* data, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = val;
}

__global__ void diagfill(float* data, int n, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[(idx * n) + idx] = val;
}

__global__ void matrixAdd(float* first, float* second, float* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = first[idx] + second[idx];
}

__global__ void matrixSub(float* first, float* second, float* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = first[idx] - second[idx];
}

// k is the number of cols of the second matrix, sry for obscurity, i just wanted compactness
__global__ void matrixDot(float* first, float* second, float* result, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < n; i++)
    {
        result[idx] += first[((idx / k) * n) + i] * second[(i * k) + (idx % k)]; // TODO optimize with tiling/whatever for efficiency/cache usage
    }
}

__global__ void matrixTranspose(float* src, float* dest, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx / cols;
    int y = idx % cols;
    dest[(y * rows) + x] = src[idx];
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
        fill <<<1, rows * cols>>> (data, val); // TODO make this more sophisticated
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
        if (type == Matrix::InitType::Identity) // TODO make this whole thing a switch statement
        {
            if (r == c)
            {
                cudaMalloc(&data, r * c * sizeof(float));
                fill <<<1, rows * cols>>> (data, 0); // TODO make this more sophisticated
                cudaDeviceSynchronize();
                diagfill <<<1, rows>>> (data, rows, 1); // this as well
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

    // TODO make matrices from array literals
    
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

    void print()
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
        
        matrixAdd <<< 1, this->rows * this->cols >>> (this->data, other.data, result.data); // TODO make more sophisticated for the love of god
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

        matrixSub <<< 1, this->rows * this->cols >>> (this->data, other.data, result.data); // TODO make more sophisticated for the love of god
        cudaDeviceSynchronize();

        return result;
    }

    Matrix operator*(const Matrix& other) const
    {
        if (this->cols != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix result(this->rows, other.cols);
        cudaMalloc(&result.data, this->rows * other.cols * sizeof(float));

        matrixDot <<<1, this->rows * other.cols>>> (this->data, other.data, result.data, this->cols, other.cols);

        cudaDeviceSynchronize();
        
        return result;
    }

    Matrix T() const
    {
        Matrix transposed = Matrix(this->cols, this->rows);
        cudaMalloc(&transposed.data, this->cols * this->rows * sizeof(float));

        matrixTranspose <<< 1, this->rows * this->cols >>> (this->data, transposed.data, this->rows, this->cols);
        cudaDeviceSynchronize();
        
        return transposed;
    }

    // operator overload for double addition, subtraction, multiplication, (division?? integer division? modulo???) on matrix, hadamard (mat1.had(mat2);)
    // +=, -=, does operation in place
    // 
    // probably just want to do multiplication for now
    // switch everything to floats?
    
};

// TODO scalar multiplication, applying (atomic) math functions, saving matrices, loading matrices
// long term: literally make AUTODIFF for backpropagation, infrastructure for mini-batch inference/gradient descent, other stuff necessary for
// moderately-fledged NN library, make OOP stuff more encapsulated
// 

// i can do stack arrays of whatever dimension just fine, just need to pass pointer to array into the constructor
// heap arrays: should be allocated as 1d. i can make that happen with the dataloader helper functions. can't think of any other instance where that would
// actually be used.

// list of math functions to implement (sensible derivatives): pow, root, exp, log, sin, cos

/**/

int main() 
{
    
    std::cout << "testing out matrix creation\n";
    Matrix a = Matrix(5, 5, Matrix::InitType::Random);
    a.print();
    Matrix b = Matrix(5, 5, 3);
    b.print();
    Matrix c = Matrix(5, 5, Matrix::InitType::Identity);
    c.print();
    std::cout << "testing out matrix addition/subtraction\n";
    Matrix d = a + b + c;
    d.print();

    Matrix e = a - c;
    e.print();
    std::cout << "testing out matrix multiplication\n";
    Matrix f = Matrix(2, 2, 5);
    Matrix g = Matrix(2, 3, 5);
    Matrix h = f * g;

    f.print();
    g.print();
    h.print();

    Matrix i = Matrix(4, 4, Matrix::InitType::Identity);
    Matrix j = Matrix(4, 4, Matrix::InitType::Random);
    Matrix k = i * j;

    i.print();
    j.print();
    k.print();

    Matrix l = Matrix(10, 10, Matrix::InitType::Random);
    Matrix m = Matrix(10, 1, 0.1);
    Matrix n = l * m;

    l.print();
    m.print();
    n.print();
    
    float stackMatrix[4][3] = { {1,2,3},{4,5,6},{7,8,9},{10,11,12} };
    float* heapMatrix = new float[9] {4,2,3,-8,2.5,6,1,0,1};
    Matrix mat1 = Matrix(4, 3, *stackMatrix);
    Matrix mat2 = Matrix(3, 3, heapMatrix);
    mat1.print();
    mat2.print();
    Matrix mat3 = mat1 * mat2;
    mat3.print();
    
    std::cout << "testing transposes:\n";
    float thearr[5][4] = { {1,2,3,4},{1,0,0,0},{1,1,0,0},{1,1,1,0},{1,1,1,1} };
    Matrix transposeTest = Matrix(5, 4, *thearr);
    transposeTest.print();
    Matrix max = transposeTest.T();
    max.print();
}
