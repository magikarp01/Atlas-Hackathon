# A neural network architecture that can arbitrarily learn a new task on new data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y, size=10):
    one_hot_Y = np.zeros((Y.size, size))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


class BasicNN():
    def __init__(self, num_neurons_l1, num_neurons_l2):
        self.num_neurons_l1 = num_neurons_l1
        self.num_neurons_l2 = num_neurons_l2

        # initializing weights and bias
        self.W1 = np.random.rand(num_neurons_l1, 784) - 0.5
        self.b1 = np.random.rand(num_neurons_l1, 1) - 0.5
        self.W2 = np.random.rand(num_neurons_l2, 10) - 0.5
        self.b2 = np.random.rand(num_neurons_l2, 1) - 0.5
        
    def save_weights(self, name):
        np.savez(file=name, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_weights(self, name):
        loaded_weights = np.load(name)
        self.W1 = loaded_weights['W1']
        self.b1 = loaded_weights['b1']
        self.W2 = loaded_weights['W2']
        self.b2 = loaded_weights['b2']

    # get values of all the neurons
    def forward_prop(self, X):
        # Save layer outputs for backpropagation
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(Z2)

        # A1, A2 are nx1 arrays of neurons 
        return Z1, A1, Z2, A2
    
    # get gradients of each weight/bias array
    def backward_prop(self, X, Y):
        # m is number of datapoints
        m = X.shape[0]
        Z1, A1, Z2, A2 = self.forward_prop(X)
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2
    
    # train on data
    def train_data(self, alpha, X, Y):
        dW1, db1, dW2, db2 = self.backward_prop(X, Y)
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2



class ModularNN(BasicNN):
    def __init__(self, num_neurons_l1, num_neurons_l2):
        super().__init__(num_neurons_l1, num_neurons_l2)

        # array of NNHeads
        self.heads = []

    # new head of the modular neural network
    class NNHead(BasicNN):
        # have to access outer ModularNN 
        # selected_neurons is list of neurons (a, b) where a is layer 
        # (0 for inputs, 1 for first layer, or 2 for second layer)
        # and b is position in array
        def __init__(self, outer, num_neurons_l1, num_neurons_l2, selected_neurons):
            super().__init__(num_neurons_l1, num_neurons_l2)
            self.neuron_input = selected_neurons
            self.outer = outer
        
        def forward_prop(self, X):
            Z1, A1, Z2, A2 = self.outer.forward_prop(X)
            inputs = []

            # iterate through all selected neurons
            for i in self.selected_neurons:
                if i[0] == 0:
                    inputs.append(X[i[1]])
                elif i[0] == 1:
                    inputs.append(A1[i][1])
                else:
                    inputs.append(A2[i][1])

            # recalculate layer outputs, redo forward
            new_X = np.array(inputs)
            Z1 = self.W1.dot(new_X) + self.b1
            A1 = ReLU(Z1)
            Z2 = self.W2.dot(A1) + self.b2
            A2 = softmax(Z2)

            # now, return new forward outputs
            return Z1, A1, Z2, A2


    def make_new_head(self, selected_neurons, num_neurons_l1, num_neurons_l2, ):
        # initialize a new head and add it to the heads list
        self.heads.append(self.NNHead(self, num_neurons_l1, num_neurons_l2, selected_neurons))

    


    