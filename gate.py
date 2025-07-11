import numpy as np

class MultiplyGate:
    def forward(self, Weight, x):
        return np.dot(Weight, x)

    def backward(self, Weight, x, dz):
        """
            Derive using formula downstream gradient = local gradient x upstream gradient, where
                dz - upstream gradient
                dW - downstream gradient for Weight
                dx - downstream gradient for x
                Local gradients: If z = xy, then dz/dx = y and dz/dy =  x. Therefore for MultiplyGate,
                    Weight - local gradient for x
                    x      - local gradient for Weight
            See https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2 for full explanation
        """
        # dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        # dW = np.dot(np.asmatrix(dz), np.transpose(np.asmatrix(x)))
        dW = np.dot(dz, np.transpose(x))
        dx = np.dot(np.transpose(Weight), dz)
        return dW, dx

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, x1, x2, dz):
        """
            Derive using formula downstream gradient = local gradient x upstream gradient, where
                dz - upstream gradient
                dW - downstream gradient for Weight
                dx - downstream gradient for x
                Local gradients: If z = x + y, then dz/dx = 1 and dz/dy =  1.
                    1 - local gradient for x1
                    1 - local gradient for x2
            Therefore for AddGate, downstream gradient for both x1 and x2 are same and is equal to dz
            See https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2 for full explanation
        """
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2
