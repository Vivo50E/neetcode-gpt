import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        def linear(
            x: NDArray[np.float64],
            w: NDArray[np.float64],
            b: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return x @ w + b

        def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.maximum(x, 0.0)
        
        output = x

        for i, (weight, bias) in enumerate(zip(weights, biases)):
            output = linear(output, weight, bias)

            # ReLU on hidden layers only
            if i < len(weights) - 1:
                output = relu(output)
    
        return np.round(output, 5)