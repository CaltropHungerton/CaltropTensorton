
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <stdexcept>


/*
std::default_random_engine generator;
std::normal_distribution<double> distribution(1.0, 1.0);
*/

/*
plans:
i am going to create a linear algebra library

matrix scalar multiplication: parallelize over every single element in strided matrix


it is tempting to have each of these operations be their own little cuda kernel things (agglomerate in order to do mv or mm mult)
but that is probably a bad idea

i should also practice writing good test harnesses. that would look EPICCC to an employer




think about what matrix object/struct looks like

*/


/*
pointer to strided array
int rows, cols


matrix object/struct
probably have cuda malloc managed

make 2d matrix of various dimensions (generalize later, this is just a first attempt)

initializations as different constructors? remind myself what the c++ rule of # bullshit is for this
initializations with data/arrays, from file
ARGS: (int rows, int cols, float val) #fill with that value
ARGS: (int rows, int cols, std::string arg) # could be identity, types of random initialization,
ARGS: (int rows, int cols, float* arr) # will need to change this, because we're moving posession of memory here

get dims

print

operator overloading
matrix matrix +: need to have same dimensions
matrix matrix dot product *: need to have same inner dimensions, allocate new matrix of right dimentions, return new constructed matrix

transpose makes more sense as just a method i think. either M.transpose() or M.T().

Matrix T() -- think about how this will work.

matrix norms

des

i think i will have
*/

__global__ void fill(double* data, double val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = val;
}

__global__ void diagfill(double* data, int n, double val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[(idx * n) + idx] = val;
}

__global__ void matrixAdd(double* first, double* second, double* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = first[idx] + second[idx];
}

__global__ void matrixSub(double* first, double* second, double* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = first[idx] - second[idx];
}


// k is the number of cols of the second matrix, sry for obscurity, i just wanted compactness
__global__ void matrixDot(double* first, double* second, double* result, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < n; i++)
    {
        result[idx] += first[((idx / k) * n) + i] * second[(i * k) + (idx % k)];
    }
}

//matrixDot <<<1, this->rows * other.cols>>> (this->data, other.data, result.data, this->cols, other.rows);

class Matrix
{
public:
    int rows, cols; // TODO ensure these can't be made negative
    double* data;

    enum class InitType
    {
        Identity,
        Random,
        Xavier,
        He
    };

    Matrix(int r, int c) : rows(r), cols(c), data(0) {}

    Matrix(int r, int c, double val) : rows(r), cols(c) // fill with single value
    {
        cudaMalloc(&data, r * c * sizeof(double));
        fill <<<1, rows * cols>>> (data, val); // TODO make this more sophisticated
        cudaDeviceSynchronize();
    }

    // arg constructor
    Matrix(int r, int c, InitType type) : rows(r), cols(c)
    {
        if (type == Matrix::InitType::Identity)
        {
            if (r == c)
            {
                cudaMalloc(&data, r * c * sizeof(double));
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
            double* hostData = new double[r * c];

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0, sqrt(.01));

            for (int i = 0; i < r * c; i++)
            {
                hostData[i] = distribution(generator);
            }

            cudaMalloc(&data, r * c * sizeof(double));
            cudaMemcpy(data, hostData, r * c * sizeof(double), cudaMemcpyHostToDevice);

            delete[] hostData;
        }

        else if (type == Matrix::InitType::Xavier)
        {
            double* hostData = new double[r * c];

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0, sqrt(2 / double(r + c)));

            for (int i = 0; i < r * c; i++)
            {
                hostData[i] = distribution(generator);
            }

            cudaMalloc(&data, r * c * sizeof(double));
            cudaMemcpy(data, hostData, r * c * sizeof(double), cudaMemcpyHostToDevice);

            delete[] hostData;
        }

        else if (type == Matrix::InitType::He) // TODO check to make sure i have input dims correct
        {
            double* hostData = new double[r * c];

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0, sqrt(2 / double(c)));

            for (int i = 0; i < r * c; i++)
            {
                hostData[i] = distribution(generator);
            }

            cudaMalloc(&data, r * c * sizeof(double));
            cudaMemcpy(data, hostData, r * c * sizeof(double), cudaMemcpyHostToDevice);

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
        cudaMalloc(&data, rows * cols * sizeof(double));
        cudaMemcpy(data, other.data, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    // copy assignment operator
    Matrix& operator=(const Matrix& other)
    {
        if (this != &other)
        {
            cudaFree(data);

            rows = other.rows;
            cols = other.cols;

            cudaMalloc(&data, rows * cols * sizeof(double));
            cudaMemcpy(data, other.data, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    void print()
    {
        double* dup = (double*)malloc(rows * cols * sizeof(double));
        cudaMemcpy(dup, data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

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

        cudaFree(dup);
    }

    Matrix operator+(const Matrix& other) const
    {
        if (this->cols != other.cols || this->rows != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        Matrix result(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(double));
        
        matrixAdd <<< 1, this->rows * this->cols >>> (this->data, other.data, result.data); // TODO make more sophisticated for the love of god
        cudaDeviceSynchronize();

        return result;
    }

    Matrix operator-(const Matrix& other) const
    {
        if (this->cols != other.cols || this->rows != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        Matrix result(this->rows, this->cols);
        cudaMalloc(&result.data, this->rows * this->cols * sizeof(double));

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
        cudaMalloc(&result.data, this->rows * other.cols * sizeof(double));

        matrixDot <<<1, this->rows * other.cols>>> (this->data, other.data, result.data, this->cols, other.cols);

        /*
        think about what this function needs:
        send in dimensions of arrays, rx(c r)xc
        */

        cudaDeviceSynchronize();
        
        return result;
    }

    // TODO matrix dot product, transpose, scalar multiplication, applying math functions, etc etc.
};

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
}