import numpy as np
from random import randint, random
import random
import pdb

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6], [6, 2]), dtype=float)
y = np.array(([92], [86], [89], [99]), dtype=float)

# scale units
X = X/np.amax(X, axis=0)  # maximum of X array
y = y/100  # max test score is 100


class Neural_Network(object):
    def __init__(self, num_population):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.num_population = num_population
        self.w1_population_list = num_population * [None]
        self.w2_population_list = num_population * [None]
        # weights
        # (3x2) weight matrix from input to hidden layer
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        # (3x1) weight matrix from hidden to output layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)  # activation function
        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def gen_w1_population(self):
        for i in range(len(self.w1_population_list)):
            self.w1_population_list[i] = np.random.randn(
                self.inputSize, self.hiddenSize)
        return self.w1_population_list

    def gen_w2_population(self):
        for i in range(len(self.w2_population_list)):
            self.w2_population_list[i] = np.random.randn(
                self.hiddenSize, self.outputSize)
        return self.w2_population_list

    def crosover(self, parent1, parent2):
        # parent 1 and parent 2 shape should be the same, but checking for this is
        # expensive...
        rand_list = np.random.randint(0, 2, size=parent1.shape)
        rand_list_2 = np.random.randint(0, 2, size=parent1.shape)
        child1 = np.multiply(rand_list, parent1) + np.multiply(1-rand_list, parent2)
        child2 = np.multiply(rand_list_2, parent1) + np.multiply(1-rand_list_2, parent2)
        return(child1, child2)

    def mutate(self, arr, probability):
        temp = np.asarray(arr)   # Cast to numpy array
        shape = temp.shape       # Store original shape
        temp = temp.flatten()    # Flatten to 1D
        num_to_change = int(len(temp) * probability)
        inds = np.random.choice(temp.size, size=num_to_change)   # Get random indices
        temp[inds] = temp[inds] + np.random.uniform(-2, 2, size=num_to_change) #multiply weights by random # from -2 to 2)
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
        ind = np.argpartition(values_list, self.num_population)[0:self.num_population] #arg partition retuns array of same len as input, need to take  
        #pdb.set_trace()
        print(np.asarray(values_list)[ind])
        w1_canidates = np.asarray(w1_canidates)
        w2_canidates = np.asarray(w2_canidates)
        #print(w1_canidates[1])
        return w1_canidates[ind], w2_canidates[ind]

    def neuroevolve(self, mutation_probability):
        num_iterations = 50
        self.w1_parents = self.gen_w1_population()
        self.w2_parents = self.gen_w2_population()
        self.w1_children = self.num_population * [None]
        self.w2_children = self.num_population * [None]
        for num in range(0, num_iterations):
            #self.w1_children = self.num_population * [None]
            #self.w2_children = self.num_population * [None]
            iteration = 1
            for i in range(iteration, self.num_population, 2):
                self.w1_children[i-1], self.w1_children[i] = self.crosover(self.w1_parents[i-1], self.w1_parents[i])

                self.w2_children[i-1], self.w2_children[i] = self.crosover(self.w2_parents[i-1], self.w2_parents[i])
            for j in range(0, self.num_population):
                if random.random() < mutation_probability:
                    self.w1_children[j] = self.mutate(self.w1_children[j], 0.5)
                    self.w2_children[j] = self.mutate(self.w2_children[j], 0.5)
            w1_canidates = self.w1_parents + self.w1_children
            w2_canidates = self.w2_parents + self.w2_children
            new_w1_parents, new_w2_parents = self.compute_fitness(w1_canidates, w2_canidates)
            self.w1_parents = list(new_w1_parents)
            self.w2_parents = list(new_w2_parents)
            self.W1 = self.w1_parents[0]
            self.W2 = self.w2_parents[0]
        best = self.forward(X)
        print("Best solution" + str(best))
        print("RMSE of best solution" + str(self.rmse(best, y)))
        print()
        print("W1 Weights of best solution" + str(self.W1))
        print("W2 Weights of best solution" + str(self.W2))
        return best
        #return 0



    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))


NN = Neural_Network(100)
#o = NN.forward(X)
a = NN.neuroevolve(0.5)
# defining our output
#population = 1000 * [None]
# for i in range(0, 1000):
#    NN = Neural_Network()
#    o = NN.forward(X)
#    population[i] = o
# print(population[0:10])


print("Actual Output: \n" + str(y))
