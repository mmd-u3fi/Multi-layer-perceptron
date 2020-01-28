import math
from perceptron_structs import Layer
from dataset_utils import convert_to_row

def sigmoid(value):
        return (1 / (1 + math.exp((-1) * value)))

class MLP(object):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.015):
        self.learning_rate = learning_rate
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.layers = [Layer(input_dim), Layer(hidden_dim), Layer(output_dim)]
        self.initialize_weights()
    def initialize_weights(self):
        for index, layer in enumerate(self.layers[:-1]):
            for perceptron in layer:
                perceptron.initiliaze_weights(len(self.layers[index + 1]))
    def feedforward(self, feature_vector):
        self.set_input(feature_vector)
        for layer_number, layer in enumerate(self.layers[1:], start=1):
            for perceptron_number, perceptron in enumerate(layer):
                net = 0
                for prior_perceptron in self.layers[layer_number - 1]:
                    net += prior_perceptron.output * prior_perceptron.output_weights[perceptron_number]
                net += perceptron.bias
                perceptron.output = sigmoid(net)
    def first_layer_delta(self, targets):
        deltas = []
        for perceptron in self.layers[0]:
            perceptron_deltas = []
            for weight_number, weight in enumerate(perceptron.output_weights):
                delta = 0
                hidden_perceptron = self.layers[1][weight_number]
                for hidden_weight_num, hidden_weight in enumerate(hidden_perceptron.output_weights):
                    temp = self.output_perceptron_delta(hidden_weight_num, targets[hidden_weight_num]) * hidden_weight
                    delta += temp
                out = hidden_perceptron.output
                delta *= (out * (1 - out) * perceptron.output)
                perceptron_deltas.append(delta)
            deltas.append(perceptron_deltas)
        return deltas
    def second_layer_delta(self, targets):
        deltas = []
        for perceptron_number, perceptron in enumerate(self.layers[1]):
            perceptron_deltas = []
            for weight_number, weight in enumerate(perceptron.output_weights):
                delta = self.output_perceptron_delta(weight_number, targets[weight_number])
                perceptron_deltas.append(delta * perceptron.output)
            deltas.append(perceptron_deltas)
        return deltas
    def output_perceptron_delta(self, perceptron_number, target):
        out = self.layers[-1][perceptron_number].output
        result = (out - target) * out * (1 - out)
        return result
    def total_error(self, labels):
        error = 0
        for perceptron, label in zip(self.layers[-1], labels):
            perceptron_error = 0.5 * (label - perceptron.output) ** 2
            error += perceptron_error
        return error
    def set_input(self, feature_vector):
        if self.input_dim != len(feature_vector):
            print('feature vector size is not equal to network dimension')
            return
        for perceptron, value in zip(self.layers[0], feature_vector):
            perceptron.output = value
    def update_weights(self, layer1_delta, layer2_delta):
        for perceptron, delta in zip(self.layers[0], layer1_delta):
            for index, (weight, update) in enumerate(zip(perceptron.output_weights, delta)):
                perceptron.output_weights[index] -= (self.learning_rate * update)
        for perceptron, delta in zip(self.layers[1], layer2_delta):
            for index, (weight, update) in enumerate(zip(perceptron.output_weights, delta)):
                perceptron.output_weights[index] -= (self.learning_rate * update)
    def train(self, X_train, y_train, epochs):
        layer_1_deltas = []
        layer_2_deltas = []
        size = len(y_train[list(y_train.keys())[0]])
        for epoch in range(epochs):
            index = 0
            for data, label in zip(convert_to_row(X_train), convert_to_row(y_train)):
                self.feedforward(data)
                layer_2_deltas = self.second_layer_delta(label)
                layer_1_deltas = self.first_layer_delta(label)
                self.update_weights(layer_1_deltas, layer_2_deltas)
                index += 1
                print(f'Epoch {epoch}/{epochs}: {index} out of {size} rows\r', end='')
    def dump_values(self):
        for layer_number, layer in enumerate(self.layers):
            print(f'layer {layer_number}:')
            for neuron_number, perceptron in enumerate(layer):
                print(f'\tneuron {neuron_number}:\n\t\t{perceptron}')
    def test(self, X_test, y_test):
        size = len(y_test[list(y_test.keys())[0]])
        correct_guesses = 0
        for data, label in zip(convert_to_row(X_test), convert_to_row(y_test)):
            self.feedforward(data)
            expected_response = label.index(1)
            check = True
            for perceptron in self.layers[-1]:
                if self.layers[-1][expected_response].output < perceptron.output:
                    check = False
            if check:
                correct_guesses += 1
        return correct_guesses / size