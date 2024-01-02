import numpy as np

# generate a 2xn numpy array of 0,1
# create another one which is 1 if the col sums to 1, 0 if not

n = 100 # number of training examples

X = np.random.randint(2, size=(2, n))
Y = np.where(np.sum(X, axis=0) == 1, 1, 0)

np.savetxt("XORin.csv", X, delimiter=",", fmt="%d")
np.savetxt("XORout.csv", Y.T, delimiter=",", fmt="%d")