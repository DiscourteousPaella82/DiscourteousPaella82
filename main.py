import numpy as np
import matplotlib as mpl

def initialize(self, inputs: int, outputs: int):

    standard_deviation = np.sqrt(2.0/inputs)
    weights = np.random.normal(loc=0.0, scale=standard_deviation, size=(inputs,outputs))

    biases = np.zeros(outputs)
    return self(weights,biases)

class InputData:

    features:np.ndarray
    labels:np.ndarray
    numFeatures:int

def load_data(filename:str):
    with open(filename,'r') as f:
        pass    #load labels

def load_labels(filename:str):
    with open(filename,'r') as f:
        pass    #load labels

def normalize_data():
    features = np.max(np.absolute(features), axis=0)    #normalizes data between -1 and 1

def relu(x: np.ndarray):
    return np.maximum(0, x) #sets any value less than 0 to 0

def softmax(x:np.ndarray):
    exponential_x = np.exp(x -np.max(x, axis = 1, keepdims=True))
    return exponential_x / np.sum(exponential_x, axis = 1, keepdims=True)


def main():
    pass

if __name__ == '__main__':
 main()