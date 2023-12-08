#include "Matrix.cuh"

#include <vector>
#include <set>

class Tensor
{
public:
	Matrix value;
	Matrix gradients;
	std::vector<Tensor*> parents;
	void(*backprop)(Tensor*); // initialize this with a default function
	std::string op;
	bool param;
	bool initialized; // already backpropped, modifies operator overloading behavior
	float scalar; // for use in gradients where you're just multiplying things by scalars

	Tensor(int r, int c, bool param) : value(Matrix(r, c, 0.0f)), gradients(Matrix(r, c, 0.0f)), backprop(&backwardsNull), op("n/a"), param(param) {}

	Tensor(Matrix m, bool param) : value(m), gradients(Matrix(m.rows, m.cols, 0.0f)), backprop(&backwardsNull), op("n/a"), param(param) {}

	Tensor operator+(const Tensor& other) const
	{
		Tensor newtensor = Tensor(this->value + other.value, false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.parents.push_back(const_cast<Tensor*>(&other));
		newtensor.backprop = &backwardsAdd;
		newtensor.op = "add";
		return newtensor;
	}

	Tensor operator-(const Tensor& other) const
	{
		Tensor newtensor = Tensor(this->value - other.value, false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.parents.push_back(const_cast<Tensor*>(&other));
		newtensor.backprop = &backwardsSub;
		newtensor.op = "sub";
		return newtensor;
	}

	Tensor operator*(const Tensor& other) const
	{
		Tensor newtensor = Tensor(this->value * other.value, false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.parents.push_back(const_cast<Tensor*>(&other));
		newtensor.backprop = &backwardsDot;
		newtensor.op = "dot";
		return newtensor;
	}

	Tensor operator*(const float scalar) const
	{
		Tensor newtensor = Tensor(this->value * scalar, false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.backprop = &backwardsScalarMult;
		newtensor.op = "scalarmult";
		newtensor.scalar = scalar;
		return newtensor;
	}

	Tensor operator/(const float scalar) const
	{
		Tensor newtensor = Tensor(this->value / scalar, false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.backprop = &backwardsScalarMult;
		newtensor.op = "scalardiv";
		newtensor.scalar = 1 / scalar;
		return newtensor;
	}

	Tensor exp() const
	{
		Tensor newtensor = Tensor(this->value.exp(), false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.backprop = &backwardsExp;
		newtensor.op = "exp";
		return newtensor;
	}

	Tensor T() const
	{
		Tensor newtensor = Tensor(this->value.T(), false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.backprop = &backwardsTranspose;
		newtensor.op = "transpose";
		return newtensor;
	}

	Tensor relu() const
	{
		Tensor newtensor = Tensor(this->value.relu(), false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.backprop = &backwardsRelu;
		newtensor.op = "relu";
		return newtensor;
	}

	Tensor sigmoid() const
	{
		Tensor newtensor = Tensor(1 / (Matrix(this->value.rows, this->value.cols, 1) + (-1 * this->value).exp()), false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.backprop = &backwardsSigmoid;
		newtensor.op = "sigmoid";
		return newtensor;
	}

	/*
	rule of 5
	functions: what do i do for transpose? just transpose the gradients as well?
	do i even need rule of 5 for the time being? 

	do i need to worry about comp graph with duplicate
	*/

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
		if (param)
		{
			value -= epsilon * gradients;
			gradients = Matrix(gradients.rows, gradients.cols, 0.0f); // do i even need? everything might get overwritten anyway
		}
	}
};

/*
we presumably have something that allows us to create parameter tensors and also new tensors from various primitive ops (i think we have all of them now)
we can call backwards on each of them to propagate the gradients to the parent tensors
we can update them once the gradients are in place

how do i deal with temporary objects being a part of the computational graph? i guess i just have to create nodes one at a time as l-values

and broadcasting
and averaging

and how this will interop with the nn class/training loops/inference
*/

Tensor had(const Tensor t1, const Tensor t2)
{
	Tensor newtensor = Tensor(had(t1.value, t2.value), false);
	newtensor.parents.push_back(const_cast<Tensor*>(&t1));
	newtensor.parents.push_back(const_cast<Tensor*>(&t2));
	newtensor.backprop = &backwardsHad;
	newtensor.op = "had";
	return newtensor;
}

Tensor operator/(float scalar, const Tensor t)
{
	Tensor newtensor = Tensor(scalar / t.value, false);
	newtensor.parents.push_back(const_cast<Tensor*>(&t));
	newtensor.backprop = &backwardsRecip;
	newtensor.op = "recip";
	newtensor.scalar = scalar;
	return newtensor;
}

Tensor operator*(const float scalar, const Tensor t)
{
	return t * scalar;
}

Tensor crossEntropy(Tensor probs, Tensor y) // TODO
{

}

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

void backwardsRelu(Tensor* tensor)
{
	gradRELU <<< 1, tensor->value->rows * tensor->value->cols >>> (tensor->gradients, tensor->value, tensor->parents[0]->gradients); // TODO: make blocks/threads better
	cudaDeviceSynchronize(); // wait i can't do this from the c++ file. i have to add this to my cuda c++ file TODO make function in other file
}

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

void backwardsNull(Tensor* tensor) {}

void buildTopoSort(Tensor* node, std::set<Tensor*>& visited, std::vector<Tensor*>& topo)
{
	if (node != nullptr && visited.find(node) == visited.end())
	{
		visited.insert(node);
		for (Tensor* parent : node->parents)
		{
			buildTopoSort(parent, visited, topo);
		}
		topo.push_back(node);
	}
}

std::vector<Tensor*> topologicalSort(Tensor* root)
{
	std::vector<Tensor*> topo;
	std::set<Tensor*> visited;
	buildTopoSort(root, visited, topo);
	return topo;
}

/*
I think that the way i'll handle broadcasting: i'll just add another field that lets you set something as a bias and i'll call a bespoke add/sub/whatever method when i'm
creating the value for the new tensor. After that I'll 
*/

class MLP
{
	
};

/*
thinking about how broadcasting will work:

Although i have the standard safeguards in place in my matrix library, i will assume that if the dimensions ever differ between 
*/

// need to think about what to add in order to allow the weights to broadcast efficiently when doing minibatches
// my program assumes that the user knows what their doing--if they are adding two incompatible vectors with one w/ single col

/*
i need to figure out how i'm going to do my loss function. that's where all of this begins. that's from where the first gradients get passed down to the first nodes
and so on and so forth. 

presumably, it's just a tensor class that outputs stuff, but i call a function on it to calculate the loss whenever i want to print out my outputs.
the scalar value we end up with after that has no influence on the gradient calculation of this stuff.

i will need to compare things against the given training example, which i'll need to reference somehow.

all of this is happening within one forward pass, and i'm not equipped to deal with the gradient averaging involved in doing minibatch gradient descent. 

there will be a lot of parents for the loss function tensor. i will have to go through each of them and propagate the gradient

*/

/*
need to write my backprop function first. (maybe i do this in NN class)

probably want to do a backprop init function where i create a vector of the topological ordering of all nodes in the computational graph which i can reverse iterate
thru and calculate each gradient in order.

i zero out every gradient in the list first
then i reverse iterate, calling backprop function on each tensor pointer i encounter

there will be a seperate update function that performs the gradient update given the specific learning rate (maybe i make bool for param or not? would have to
edit my constructors)

still have to think about how the end function will be handled, what the grad for that will be initialized to, how to handle loss functions

how will inference work? I haven't even thought about how the shape of the inputs will work, how that will affect the shape of the gradient

*/



int main()
{
	// create some tensors, do ops, see the computational graph, debug gradient flows.
}

/*
ok draft for REAL this time
we have a backward function that we call at beginning which does DFS and creates reverse topological ordering. then we can iterate through and call backprop function in order
during each round of backprop given the activations in each forward pass.
how to think about logic of turning this process on or off should work, since this will only be happening in the backprop method. different methods for solely forwardprop
and forward + backprop? who knows. probably best to have bool since we're operator overloading. but still, how will this look? do we even need to turn off? it's not
like we even have to call backprop at the end when we're doing inference. All we are doing is appending parent pointers to the vectors, which isn't crazy at all.
WE CAN ALSO CHECK WHETHER THE PARENT VECTOR IS EMPTY, it's not like we're going to change it after we append stuff the first time.

ADD: 
matrix = t1.matrix + t2.matrix (operator overloaded matrix addition)
gradients = 0 matrix (i think)
parents.push_back(t1);
parents.push_back(t2);
backprop = [] { parents[0].gradients += this->gradients; parents[1].gradients += this->gradients; }; make this a defined function instead of a lambda?

SUB:
matrix = t1.matrix - t2.matrix (operator overloaded matrix subtraction)
gradients =  // check what you need to do for subtraction
...

scalar mult:
matrix = t1.matrix * scalar

backprop = [] { parents[0].gradients += this->gradients * scalar; }
*/


/*
multiply current gradient level by

let me think about the simple case for something like the overriden plus operator
i know what the parents will be (the things i'm adding)
the gradient of the sum is the sum of the gradients

things to drill down on more:
i don't fully understand the nature of the differential roles the parameters/inputs play/how i take derivatives of these sorts of things. this should change

if we have a matrix multiplication, given that the things is linear, say WX, the derivative with respect to W is X, and vice versa (i think)
if we have that thing created in the computational graph, then is that what we are returning and if not what are we returning instead?

it seems that we are just returning the outer gradient multiplied by the transpose of the other matrix multiplied on the correct side.
Literally just replace the thing you're taking the gradient with respect to by the gradient and then transpose the other thing.

and then for the hadamard product it's the same but you don't transpose (and you're doing the hadamard product instead of the dot product).

the only difference between these things that the hadamard product doesn't involve the transpose of the other, and it does the hadamard product instead of the dot product

something like biases weren't created by any atomic ops and are just added, so the gradient is just the gradient that was brought down to them.

in my nn class, i'll just have a vector of all of the tensors that can be updated, and they will have their gradients stored. i can just do an individual update w/ learningrate

so the goal is to get the dL/dP calculated for each parameter and store it so that it can be used.

*/


/*
this is my tensor wrapper class.
tensor has a matrix
it has operator overloading for updating the computational graph
it has bool for whether autograd has been turned on or not (define in constructor, obscure in NN libarary)
whatever the necessary bits for the computational graph representation that i will end up using

how am i to think about the stored values? is this something that will interop with the forward pass/backward pass (probably)
so i'll end up needing a method for storing the value that was input into the function so you can feed it into derivative instead
but that means that this will turn into a matrix, since we will be feeding a bunch of inputs simultaneously. no worries!
or will it... clearly i need to think about this more, but this is good because it's exposing the places where i need to think more
*/

/*
Tensor operator+(const Tensor& other) const {
	Tensor newtensor;
	if (!newtensor.is_initialized) {
		newtensor = Tensor(this->value + other.value, false);
		newtensor.parents.push_back(const_cast<Tensor*>(this));
		newtensor.parents.push_back(const_cast<Tensor*>(&other));
		newtensor.backprop = &backwardsAdd;
		newtensor.op = "add";
		newtensor.is_initialized = true;
	} else {
		newtensor.value = this->value + other.value;
		newtensor.gradients = Matrix(this->value.rows, this->value.cols, 0.0f);
	}
	return newtensor;
}

THIS WOULD NOT WORK but i could check whether one or more of the previous tensors were already BACKPROPPED or CHILD OF BACKPROPPED

need to make sure i can just replace the matrix without memory leaks...

this is where i'll need move semantics
*/