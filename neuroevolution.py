import numpy as np
from random import randint, random
import random
import pdb
#from numba import jit, prange, njit
from scipy.special import expit
from sklearn.datasets import make_classification, make_regression, make_moons, make_blobs
import scipy.linalg.blas as FD
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation



#X, y = make_moons(n_samples=1000, random_state=43, noise=0.2)
X, y = make_blobs(n_samples=100, centers=2, n_features=2,random_state=42, cluster_std=3.5)
#X, y = make_regression(n_samples=100, n_features=2, n_informative=1, random_state=43)
org_y = y
y = np.asarray([[i] for i in np.absolute(y)])
# X = np.absolute(X) * 8  # make it a reasonable
######################################

# X = (hours sleeping, hours studying), y = score on test
#X = np.array(([2, 9], [1, 5], [3, 6], [8, 2], [9, 1], [10, 0]), dtype=np.int8)
#y = np.array(([0], [0], [1], [1], [1], [0]), dtype=np.int8)

# scale units
# X = X/np.amax(X, axis=0)  # maximum of X array
# y = y  # max test score is 100
np.set_printoptions(suppress=True)


class Neural_Network(object):
    def __init__(self, num_population, number_iterations, save_weights_loc, load_weights_loc, graph):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 20
        self.load_weights_loc = load_weights_loc
        self.save_weights_loc = save_weights_loc
        self.num_population = num_population
        self.number_iterations = number_iterations
        self.current_iteration = 0
        self.graph = graph
        self.best_solution = 0
        self.w1_population_list = num_population * [None]
        self.w2_population_list = num_population * [None]
       # plt.axis([0, self.number_iterations, 0, 1])
        # weights
        if self.load_weights_loc:
            h5f = h5py.File(self.load_weights_loc, 'r')
            self.W1 = h5f['weights1'][:]
            self.W2 = h5f['weights2'][:]
        else:
            self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
            self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        #self.z = np.dot(X, self.W1)
        self.z = FD.dgemm(alpha=1.0, a=X.T, b=self.W1.T,
                          trans_a=True, trans_b=True)  # MKL fast dot product
        self.z2 = NN.Elu((self.z))  # ACTIVATION FUNCTION
        # dot product of hidden layer (z2) and second set of 3x1 weights
        #self.z3 = np.dot(self.z2, self.W2)
        self.z3 = FD.dgemm(alpha=1.0, a=self.z2.T,
                           b=self.W2.T, trans_a=True, trans_b=True)
        o = expit((self.z3))  # final activation function
        return o

    def gen_w1_population(self):
        for i in range(len(self.w1_population_list)):
            if self.load_weights_loc:
                self.w1_population_list[i] = self.W1
            else:
                self.w1_population_list[i] = np.random.randn(
                    self.inputSize, self.hiddenSize)
        return self.w1_population_list

    def gen_w2_population(self):
        for i in range(len(self.w2_population_list)):
            if self.load_weights_loc:
                self.w2_population_list[i] = self.W2
            else:
                self.w2_population_list[i] = np.random.randn(
                    self.hiddenSize, self.outputSize)
        return self.w2_population_list

    def crosover(self, parent1, parent2):
        # parent 1 and parent 2 shape should be the same, but checking for this is
        # expensive...
        rand_list = np.random.randint(0, 2, size=parent1.shape)
        rand_list_2 = np.random.randint(0, 2, size=parent1.shape)
        child1 = np.multiply(rand_list, parent1) + \
            np.multiply(1-rand_list, parent2)
        child2 = np.multiply(rand_list_2, parent1) + \
            np.multiply(1-rand_list_2, parent2)
        return(child1, child2)

    def mutate(self, arr, probability):
        temp = np.asarray(arr)   # Cast to numpy array
        shape = temp.shape       # Store original shape
        temp = temp.flatten()    # Flatten to 1D
        num_to_change = int(len(temp) * probability)
        inds = np.random.choice(
            temp.size, size=num_to_change)   # Get random indices
        # multiply weights by random # from -2 to 2)
        temp[inds] = temp[inds] + np.random.uniform(-2, 2, size=num_to_change)
        temp = temp.reshape(shape)                     # Restore original shape
        return temp

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def compute_fitness(self, w1_canidates, w2_canidates):
        values_list = 2 * self.num_population * [None]
        for i in range(0, 2 * self.num_population):
            self.W1 = w1_canidates[i]
            self.W2 = w2_canidates[i]
            values_list[i] = NN.rmse(NN.forward(X), y)
        # arg partition retuns array of same len as input, need to take
        ind = np.argpartition(values_list, self.num_population)[
            0:self.num_population]
        # pdb.set_trace()
        best_canidate = np.asarray(values_list)[ind][0]
        if self.graph:
            plt.scatter(self.current_iteration, best_canidate, c="g", marker=".")
            plt.pause(0.001)
        else:
            print("iteration num " + str(self.current_iteration) + " RMSE " + str(best_canidate))
        w1_canidates = np.asarray(w1_canidates)
        w2_canidates = np.asarray(w2_canidates)
        # print(w1_canidates[1])
        return w1_canidates[ind], w2_canidates[ind]

    def neuroevolve(self, mutation_probability):
        num_iterations = self.number_iterations
        self.w1_parents = self.gen_w1_population()
        self.w2_parents = self.gen_w2_population()
        self.w1_children = self.num_population * [None]
        self.w2_children = self.num_population * [None]
        for num in range(0, num_iterations):
            #self.w1_children = self.num_population * [None]
            #self.w2_children = self.num_population * [None]
            self.current_iteration = num
            iteration = 1
            for i in range(iteration, self.num_population, 2):
                self.w1_children[i-1], self.w1_children[i] = self.crosover(
                    self.w1_parents[i-1], self.w1_parents[i])

                self.w2_children[i-1], self.w2_children[i] = self.crosover(
                    self.w2_parents[i-1], self.w2_parents[i])
            for j in range(0, self.num_population):
                if random.random() < mutation_probability:
                    self.w1_children[j] = self.mutate(
                        self.w1_children[j], random.random())
                    self.w2_children[j] = self.mutate(
                        self.w2_children[j], random.random())
            w1_canidates = self.w1_parents + self.w1_children
            w2_canidates = self.w2_parents + self.w2_children
            new_w1_parents, new_w2_parents = self.compute_fitness(
                w1_canidates, w2_canidates)
            self.w1_parents = list(new_w1_parents)
            self.w2_parents = list(new_w2_parents)
            self.W1 = self.w1_parents[0]
            self.W2 = self.w2_parents[0]
            if num == 0 or num % 10 == 0:
                self.plot_decision_boundary()
                plt.pause(0.001)
        best = self.forward(X)
        self.best_solution = best
        if self.save_weights_loc:
            self.save_weights(self.save_weights_loc)
        print("Best solution" + str(best))
        self.best_rmse = str(self.rmse(best, y))
        print("RMSE of best solution" + self.best_rmse)
        print()
        print("W1 Weights of best solution" + str(self.W1))
        print("W2 Weights of best solution" + str(self.W2))
        return best
        # return 0

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def ReLU(self, arr):
        arr[arr < 0] = 0
        return arr

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def Elu(self, x, a=2):
        """exponential linear unit, from this paper  https://arxiv.org/abs/1511.07289... seems to work quite well"""
        return np.where(x <= 0, a * (np.exp(x) - 1), x)

    def gaussian(self, x):
        sq = np.square(x)
        neg = np.negative(sq)
        return np.exp(neg)

    def save_weights(self, save_weights_loc):
        h5f = h5py.File(save_weights_loc, 'w')
        h5f.create_dataset('weights1', data=self.W1)
        h5f.create_dataset('weights2', data=self.W2)
        h5f.close()

    def plot_decision_boundary(self):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.5
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()] ##I'm not good enough at numpy to say that I just "knew" that this is what I wanted.. thank you stack overflow
        Z = NN.forward(Z)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        cp = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r)
        if self.current_iteration == 0:
            plt.colorbar(cp)
        plt.scatter(X[:, 0], X[:, 1], c=org_y, cmap=plt.cm.Spectral)

NN = Neural_Network(100, 1000, "100pt.hdf5", None, False)
plt.show()
a = NN.neuroevolve(0.5)

def plot_decision_boundary():
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()] ##I'm not good enough at numpy to say that I just "knew" that this is what I wanted.. thank you stack overflow
    Z = NN.forward(Z)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    cp = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r)
    plt.colorbar(cp)
    plt.scatter(X[:, 0], X[:, 1], c=org_y, cmap=plt.cm.Spectral)

#plot_decision_boundary()



#print(gph_x)

#print(X)

#print("Actual Output: \n" + str(y))
