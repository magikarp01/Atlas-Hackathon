# A neural network architecture that can arbitrarily learn a new task on new data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from learning_architecture import BasicNN, ModularNN

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# class BasicNN():
#     def __init__(self, num_neurons_l1, num_neurons_l2):
#         self.num_neurons_l1 = num_neurons_l1
#         self.num_neurons_l2 = num_neurons_l2

#         # initializing weights and bias
#         self.W1 = np.random.rand(num_neurons_l1, 784) - 0.5
#         self.b1 = np.random.rand(num_neurons_l1, 1) - 0.5
#         self.W2 = np.random.rand(num_neurons_l2, num_neurons_l1) - 0.5
#         self.b2 = np.random.rand(num_neurons_l2, 1) - 0.5
    
#     def save_weights(self, name):
#         np.savez(file=name, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

#     def load_weights(self, name):
#         loaded_weights = np.load(name)
#         self.W1 = loaded_weights['W1']
#         self.b1 = loaded_weights['b1']
#         self.W2 = loaded_weights['W2']
#         self.b2 = loaded_weights['b2']

#     # get values of all the neurons
#     def forward_prop(self, X):
#         # Save layer outputs for backpropagation
#         Z1 = self.W1.dot(X) + self.b1
#         A1 = ReLU(Z1)
#         Z2 = self.W2.dot(A1) + self.b2
#         A2 = softmax(Z2)

#         # A1, A2 are nx1 arrays of neurons 
#         return Z1, A1, Z2, A2
    
#     # get gradients of each weight/bias array
#     def backward_prop(self, X, Y):
#         # m is number of datapoints
#         m = X.shape[0]
#         Z1, A1, Z2, A2 = self.forward_prop(X)
#         one_hot_Y = one_hot(Y)
#         dZ2 = A2 - one_hot_Y
#         dW2 = 1 / m * dZ2.dot(A1.T)
#         db2 = 1 / m * np.sum(dZ2)
#         dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(Z1)
#         dW1 = 1 / m * dZ1.dot(X.T)
#         db1 = 1 / m * np.sum(dZ1)
#         return dW1, db1, dW2, db2
    
#     # train on data
#     def train_data(self, alpha, X, Y):
#         dW1, db1, dW2, db2 = self.backward_prop(X, Y)
#         self.W1 -= alpha * dW1
#         self.b1 -= alpha * db1
#         self.W2 -= alpha * dW2
#         self.b2 -= alpha * db2


# new head of the modular neural network
# class NNHead():


# class ModularNN(BasicNN):
#     def __init__(self):
#         super().__init__()
    
#     def make_new_head(self):


# Pass in an array of arrays, each array is two items, the first item is a class for one of N ground-truth states for the variable that's being compared, and the second element is an array containing every neural net layer, with the values of each neuron in the layer. For each ground-truth state, all of the datapoints are processed to form a kernel-density estimation calculation for every neuron, and the indicies of the 100 highest correlation neurons (layer index and index within layer) are outputted as a list. The top-100 lists for each of the N ground-truth states are appended into a large list returning the most relevant correlated neuron datapoints.
import numpy as np
from sklearn.neighbors import KernelDensity

def find_correlated_neurons(data, num_neurons=100):
    # Initialize lists to store results
    results = []
    for state in set([d[0] for d in data]):
        state_data = [d[1] for d in data if d[0] == state]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(state_data)
        state_results = []
        for i, layer in enumerate(state_data[0]):
            layer_results = []
            for j, neuron in enumerate(layer):
                score = kde.score_samples([neuron])[0]
                layer_results.append((i, j, score))
            layer_results = sorted(layer_results, key=lambda x: x[2], reverse=True)[:num_neurons]
            state_results += layer_results
        results.append(state_results)
    return results

# format the neuron values after inputting X (1x784 np array of 1 picture)
def get_neurons(X, label, nn):
    Z1, A1, Z2, A2 = nn.forward_prop(X)
    return A1, A2


# xList is list of all pictures, so mx784 for m pictures
# labels is m array of labels
def format_neurons(xList, labels, nn, sample_size=10):
    neuron_values = []
    for i in range(sample_size):
        A1, A2 = get_neurons(xList[i], labels[i], nn)
        # have to reshape xList to be 1x784 instead of just 784 array
        neurons = [np.expand_dims(xList[i],axis=0), A1, A2]
        neuron_values.append(neurons)

    return [labels, neuron_values]

# for run 100 times:
#     var z = []
#     z.push(format_neurons(x(N), label(N), nn))

# nn = BasicNN(10, 10)
nn = ModularNN(10, 10)
print("loading data")
data = pd.read_csv('data/train.csv')
print("done loading")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# print(X_train.shape)
# print(Y_train.shape)

for e in range(10):
    for i in range(40000):
        X_input = np.expand_dims(X_train[:, i], axis=1)
        label = Y_train[i]
        # print(nn.forward_prop(X_input))
        hot = one_hot(label)
        nn.train_data(.1, X_input, label)

nn.save_weights('modular_nn.npz')
nn.load_weights('modular_nn.npz')

for i in range(10):
    X_input = np.expand_dims(X_train[:, i], axis=1)
    label = Y_train[i]
    # print(nn.forward_prop(X_input))
    hot = one_hot(label)
    print(nn.forward_prop(X_input)[3][label])
    # print(hot)