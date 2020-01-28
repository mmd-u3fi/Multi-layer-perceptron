import random

class Perceptron(object):
    def __init__(self):
        self.output_weights = []
        self.bias = random.random()
        self.output = 0
    def set_weight(self, weight, index):
        self.output_weights[index] = weight
    def initiliaze_weights(self, dimension):
        self.output_weights = [random.random() for i in range(dimension)]
    def number_of_parameters(self):
        return len(self.output_weights)
    def __str__(self):
        return f'bias: {self.bias}\n\t\tweights: {self.output_weights}\n\t\toutput: {self.output}'
    def __repr__(self):
        return str(self.output)

class Layer(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.perceptrons = [Perceptron() for i in range(dimension)]
    def __iter__(self): 
        return iter(self.perceptrons)
    def __getitem__(self, item):
         return self.perceptrons[item]
    def __len__(self):
        return self.dimension
    def __str__(self):
        return f'{self.perceptrons}'
    def number_of_parameters(self):
        return self.dimension * self.perceptrons[0].number_of_parameters()
