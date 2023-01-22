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

# format the neuron values after inputting X (784x1 np array of 1 picture)
def get_neurons(X, label, nn):
    Z1, A1, Z2, A2 = nn.forward_prop(X)
    return A1, A2


# xList is list of all pictures, so 784xm for m pictures
# labels is m array of labels
def format_neurons(xList, labels, nn, sample_size=10):
    neuron_values = []
    for i in range(sample_size):
        A1, A2 = get_neurons(xList[i], labels[i], nn)
        # have to reshape xList to be 784x1 instead of just 784 array
        neurons = [np.expand_dims(xList[i],axis=1), A1, A2]
        neuron_values.append(neurons)

    return [labels, neuron_values]

# for run 100 times:
#     var z = []
#     z.push(format_neurons(x(N), label(N), nn))