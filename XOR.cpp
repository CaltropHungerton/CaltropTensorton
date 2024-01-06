// figure out what all we need to import

/*
proving ground for minibatching/broadcasting


Tensor W1 = (initialize as param) (2, 2)
Tensor b1 = (initialize as param) (2, 1)

Tensor W2 = (initalize as param) (1, 2)
Tensor b2 = (initialize as param) (1, 1)

Tensor X = ... (2, n)

Tensor J1 = W1 * X;
Tensor K1 = J1 + b1;

// now we do the sigmoid.
I am going to create a bespoke function that does all of the matrix operations and has the simple gradient formula as well

Tensor A1 = K1.sigmoid();

Tensor J2 = W2 * A1;
Tensor K2 = J2 + b2;

Tensor A2 = K2.sigmoid();

Tensor loss = crossEntropy(A2, Y)


for crossEntropy:
Y is some matrix where cols are training examples (num cols C is minibatch size) (RxC)
X same

-1 * (had(Y, log(A2)) + had(1-Y, log(1-A2))) // figure out best way to calculate logs


gradient for crossEntropy(probs, Y) is just probs - Y propagated to probs tensor. store that in the loss gradient matrix
(do we scale this down by batch size? calculations for weights will get scaled down)
backprop function just adds that to the probs tensor


*/



/*
move semantics will probably be important for forward passes to avoid inefficiencies when reassigning
i'll only need to update values, gradients


*/



/*
MLP drafting:

maybe i just do it imperatively as a test for the XOR

first: declare every single variable/parameter, via primitive ops (X val is something defined by pointer arithmetic each time)
declare vector, traverse entire graph, reverse topological order

second: redo trace thru graph, move assignment to previous variables (only change values, reset gradients to 0)
call backwards function using reference to vector of pointers, if param, update weights, if not, backprop

then, confirm to move on to test set ()
*/

/*
i have to think about what the input/output pairs should look like

generate two digits, either 0 or 1
if sum is 1: 1. else: 0

X will be vertical 2 columns of first two digits

we can just create a csv file of training data

i should go back and add constructor for creating matrices from csv file to the Matrix.cu
*/



float eps = .001;
int numEpochs = 10;
int batchesPerEpoch;
bool firstPass = true;
std::vector<Tensor*> traversal;

// make a function for initializing matrix from X! call it input

for (int i = 0; i < numEpochs; i++)
{
	for (int j = 0; j < batchesPerEpoch; j++)
	{
		if (firstPass)
		{
			Tensor W1 = Tensor(Matrix(2, 2, Matrix::InitType::Xavier), true); // is xavier initialization a good idea for such small dims...
			Tensor b1 = Tensor(Matrix(2, 1, Matrix::InitType::Xavier), true);
			Tensor W2 = Tensor(Matrix(1, 2, Matrix::InitType::Xavier), true);
			Tensor b2 = Tensor(Matrix(1, 1, Matrix::InitType::Xavier), true);

			Tensor X = Tensor(fromCSV("XORin.csv"), false);
			Tensor Y = Tensor(fromCSV("XORout.csv"), false);

			Tensor J1 = W1 * X;
			Tensor K1 = J1 + b1;

			Tensor A1 = K1.sigmoid();

			Tensor J2 = W2 * A1;
			Tensor K2 = J2 + b2;

			Tensor A2 = K2.sigmoid();
			
			Tensor Loss = CrossEntropy(A2, Y); // add cross entropy calculation, figure out how this works for minibatches

			// maybe print Loss scalar if j == batchesPerEpoch - 1?
			//

			// next, we go from the Loss tensor, traversing the entire computational graph. we append each tensor pointer
			// to traversal in reverse topologically sorted order

			std::vector<Tensor*> nodelist = topologicalSort(&Loss);

			for (auto x = nodeList.rbegin(); x != nodeList.rend(); ++x)
			{
				(*x)->backprop(*x); // is this correct?
				(*x)->update();
			}

			firstPass = false;
		}

		else
		{
			X = Tensor(input..., false);
			Y = Tensor(input..., false);

			J1 = W1 * X;
			K1 = J1 + b1;

			A1 = K1.sigmoid();

			J2 = W2 * A1;
			K2 = J2 + b2;

			A2 = K2.sigmoid();

			Loss = CrossEntropy(A2, Y);

			for (auto x = nodeList.rbegin(); x != nodeList.rend(); ++x)
			{
				(*x)->backprop(*x); // is this correct?
				(*x)->update();
			}
		}
	}
}
