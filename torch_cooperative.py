import torch
from sklearn.datasets import make_classification, make_regression, make_moons, make_blobs
import numpy as np
from random import randint, random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import random

X, y = make_moons(n_samples=100, random_state=42, noise=0.1)
#X, y = make_blobs(n_samples=1000, centers=2, n_features=2,random_state=42, cluster_std=2.0)
#X, y = make_regression(n_samples=100, n_features=2, n_informative=1, random_state=43)
#X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=43, class_sep=0.8, flip_y=0.08)
org_y = y
y = np.asarray([[i] for i in np.absolute(y)])
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


class Neural_Network():
    def __init__(self, num_population, number_iterations, mutation_probability):
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 10
        self.num_population = num_population
        self.number_iterations = number_iterations
        self.w1_size = self.inputSize * self.hiddenSize
        self.w2_size = self.hiddenSize * self.outputSize
        self.total_size = self.w1_size + self.w2_size
        self.pop_matrix = self.initialize_population()
        self.mutation_probability = mutation_probability
        self.iterator = 0

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 3 X 2 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 3 X 1 tensor

    def forward(self, X):
        # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z = torch.matmul(X, self.W1)
        self.z2 = self.ReLU(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def ReLU(self, arr):
        arr[arr < 0] = 0
        return arr

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def initialize_population(self):
        pop_arr = torch.randn(self.num_population, self.total_size, 2)
        for elem in pop_arr:
            for neuron in elem:
                neuron[1] = 1
        #pop_arr = torch.from_numpy(pop_arr).float()
        return pop_arr

    def rmse(self, predictions, targets):
        return torch.sqrt(((predictions - targets) ** 2).mean())

    def crossover(self, parent1, parent2):
        # parent 1 and parent 2 shape should be the same, but checking for this is
        # expensive...
        rand_list = torch.randint(0, 2, size=parent1.shape)
        rand_list_2 = torch.randint(0, 2, size=parent1.shape)
        child1 = (rand_list *  parent1) +  (1-rand_list * parent2)
        child2 = (rand_list_2 *  parent1) + (1-rand_list_2 * parent2)
        return(child1, child2)


    def mutate(self, arr, probability):
        temp = arr.flatten()
        num_to_change = int(len(temp) * probability)
        perm = torch.randperm(len(temp))
        to_change = perm[:num_to_change]
        # multiply weights by random # from -2 to 2)
        temp[to_change] = temp[to_change] + \
            torch.FloatTensor(len(to_change)).uniform_(-5, 5)
        #temp[inds] = temp[inds] + np.random.uniform(-2, 2, size=num_to_change)
        # Restore original shape
        ret = temp.reshape(arr.shape)
        return ret


    def plot_decision_boundary(self):
        # Set min and max values and give it some padding
        new_x = X.numpy()
        new_y = y.numpy()
        x_min, x_max = new_x[:, 0].min() - .5, new_x[:, 0].max() + .5
        y_min, y_max = new_x[:, 1].min() - .5, new_x[:, 1].max() + .5
        h = 0.1
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()] ##I'm not good enough at numpy to say that I just "knew" that this is what I wanted.. thank you stack overflow
        Z = torch.from_numpy(Z).float()
        Z = NN.forward(Z)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        cp = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r)
        if self.iterator == 1:
            plt.colorbar(cp)
        plt.scatter(X[:, 0], X[:, 1], c=org_y, cmap=plt.cm.Spectral)


    def convert_to_weights(self, i, update):
        weights = self.pop_matrix[i, :, 0]
        fitnes_values = self.pop_matrix[i, :, 1]
        #w1_fitness_slice = fitnes_values[:self.w1_size]
        #w2_fitness_slice = fitnes_values[self.w1_size:self.total_size]
        w1_slice = weights[:self.w1_size]
        w2_slice = weights[self.w1_size:self.total_size]
        self.W1 = torch.reshape(w1_slice, (self.inputSize, self.hiddenSize))
        self.W2 = torch.reshape(w2_slice, (self.hiddenSize, self.outputSize))
#        print(self.W1)
#        print(self.W2)
        if update:
            prediction = self.forward(X)
            error = self.rmse(prediction, y)
            fitnes_values[fitnes_values != 0] = error

    def cooperative_neuroevolution(self):
        fitness_vals = self.pop_matrix[:, :, 1]
        weight_vals = self.pop_matrix[:, :, 0]
        # print(a)
        # the average fitness of each network
        best_fitness = 1
        while self.iterator < 100000000000:
            network_avg_fitness = fitness_vals.sum(dim=1) / self.total_size
            #print(network_avg_fitness)
            most_fit_networks, most_fit_networks_i = network_avg_fitness.topk(2, largest=False)
            least_fit_nets, worst_network_i = network_avg_fitness.topk(1, largest=True)

            for i in range(1, len(most_fit_networks_i)):
                self.convert_to_weights(most_fit_networks_i[i-1], False)
                w1_first_parent = self.W1
                w2_first_parent = self.W2
                best_pred = self.forward(X)
                check_fitness = self.rmse(best_pred, y)
                if best_fitness > check_fitness:
                    best_fitness = check_fitness
                    #self.plot_decision_boundary()
                    #plt.pause(0.001)
                    print("best fitness " + str(best_fitness) + " generation: " + str(self.iterator))
                self.convert_to_weights(most_fit_networks_i[i], False)
                w1_second_parent = self.W1
                w2_second_parent = self.W2
                #print(w2_second_parent)
                w1_child1, w1_child2 = self.crossover(
                    w1_first_parent, w1_second_parent)
                w2_child1, w2_child2 = self.crossover(
                    w2_first_parent, w2_second_parent)
                #print(w2_child1)
                if random.random() < self.mutation_probability:
                    w1_child1 = self.mutate(w1_child1, 0.8)
    #                w1_child2 = self.mutate(w1_child2, 0.3)
                    w2_child1 = self.mutate(w2_child1, 0.5)
    #                w2_child2 = self.mutate(w2_child2, 0.3)
            self.W1 = w1_child1
            self.W2 = w2_child1
            new_prediction_1 = self.forward(X)
            new_fitness_1 = self.rmse(new_prediction_1, y)
            new_weights_1 = torch.cat((w1_child1.flatten(), w2_child1.flatten()))

            weight_vals[worst_network_i] = new_weights_1 #new weight replacement!
            filled_weights = torch.full((1,1), new_fitness_1)
            to_avg = torch.cat((filled_weights, fitness_vals[worst_network_i]), dim=1)
            fitness_vals[worst_network_i] = torch.mean(to_avg)
            #print(fitness_vals[worst_network_i])
            #print(filled_weights)
            #fitness_vals[worst_network_i] = torch.mean([fitness_vals[worst_network_i], filled_weights]), dim=0) #fitness replacement
            #print(fitness_vals)
            #print(worst_network_i)
            #print(self.pop_matrix[:,0:1,0])
            for j in range(0, self.num_population, 2):
                self.pop_matrix[:,j,:] = self.pop_matrix[:,j,:][torch.randperm(self.num_population)] #permutate all values

#            for i in range(0, self.num_population): #finally, score each network after permutation
#                self.convert_to_weights(i, False)
#                best_pred = self.forward(X)
#                net_fitness = self.rmse(best_pred, y)
#                filled_net_fitness = torch.full((self.total_size,1), net_fitness)
#                to_avg_perm = torch.cat((filled_net_fitness.flatten(), fitness_vals[i].flatten()), dim=-1)
#                fitness_vals[i] = torch.mean(to_avg_perm)

            self.iterator += 1



NN = Neural_Network(10, 100, 0.5)

for i in range(0, NN.num_population):
    NN.convert_to_weights(i, True)
NN.cooperative_neuroevolution()
