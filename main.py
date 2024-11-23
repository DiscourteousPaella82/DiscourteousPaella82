import numpy as np
import matplotlib as mpl

def initialize(self, inputs: int, outputs: int):

    standardDeviation = np.sqrt(2.0/inputs)
    weights = np.random.normal(loc=0.0, scale=standardDeviation, size=(inputs,outputs))

    biases = np.zeros(outputs)
    return self(weights,biases)

class InputData:

    features:np.ndarray
    labels:np.ndarray
    numFeatures:int

def main():
    pass

if __name__ == '__main__':
 main()