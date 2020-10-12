import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


net = Network([2, 3, 1])
