Working GPU parallelized linear algebra library in CUDA C++!

The Linear Algebra library that powers all of this (Matrix.cu) is fully working. Please check it out!

XOR.cu is a demonstration of the autodiff library successfully solving the XOR problem. MNIST.cu will be implemented soon, with a demonstration of a simple feed-forward network on the MNIST dataset

TODOs:
-figure out the cuda linker errors so that i don't need to copy the matrix and autodiff library components into new main files for each demonstration
-add mnist
-add cache tiling to the matmuls, get rid of the superfluous matrix fill kernel
