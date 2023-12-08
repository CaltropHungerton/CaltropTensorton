


/*
the way the training loop goes;

we declare our params first

and training data/batch (not a parameter!)
and test data/batch (not a parameter!)

and we build up the computational graph: do ops with these params and arrive at result (loss function)

how should i handle backprop traversal? creating a vector would have to happen after definition of first forward pass, which seems inelegant

i think it's actually totally fine to create the list on the first pass, then set a boolean so it doesn't happen again. all we need to
specify is the end of the computational graph.
*/


/*
I'm going to think about what a forward pass for mnist might look like


input: 784xN matrix
hidden layer 1: 256, RELU
hidden layer 2: 256, RELU
output layer 3: 10, softmax
cross entropy loss

so hidden layer 1 will have 256x784 matrix (need to get cuda to handle larger dimensions soon)
then we relu each activation (we have 256xN matrix now, a1)
hidden layer 2: we have 10x256 matrix, 10x256 * 256xN, we get 10xN matrix
apply softmax to each column, calculate loss function, calculate gradients (maybe transpose for sake of cache)

need to create a bespoke function for this that returns a matrix of same dimensions, but softmaxes down columns

soft max is just entire matrix plugged into e^X eltwise divided by sum

actually i WILL need to worry about broadcasting and gradients thereof since i will be dividing elementwise by compacted sum of e^x'd activations

wait actually this is profoundly trivial! we just broadcast the entire gradient vector evenly over the child matrix that we summed

but after that i will still have rectangular matrix of predictions. Then i cross entropy loss



BROADCASTING RULES:
(maybe do this with try catch for matrices?)

one of the matrices will be <= the other.

i shouldn't worry about parallelizing vectorwise, should just create new temporary matrix depending
so i should make cuda kernels for broadcasting (parallelize over every element of new matrix, assign different number depending on...)
i'll think through the algorithm for this in a bit
will i have to create l-values for this is the question...this is hard



we could ofc just create a new broadcast matrix from the tensor, do at matrix level


*/