import numpy as np

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        """
        https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/

        # https://medium.com/udacity/shannon-entropy-information-gain-and-picking-balls-from-buckets-5810d35d54b4
        # https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
        # https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2
        # loss = -log(Sm)
        The tricky part here is the dependence of loss on a single element of the vector S. So, l = -log(Sm) 
        and ∂l/∂Sm = -1/Sm where Sm represents the mth element of S where m is the ground truth label. 
        """
        return -np.log(probs[y])

    def loss_derivative(self, x, y):
        """The expression -log(Sm) represents the cross-entropy loss for a single data point when 
        using a softmax output layer and a one-hot encoded target. Here's why: [1, 2, 3]  
            Explanation: 

            1. Softmax Output: The softmax function converts a vector of real numbers into a probability 
               distribution, where each element represents the probability of belonging to a specific 
               class. In this case, Sm is the probability output by the softmax for the correct 
               class 'm'. [2, 4, 4]  
            2. Cross-Entropy Loss: Cross-entropy loss is a common loss function for classification tasks.
               It measures the difference between the predicted probability distribution and the true 
               distribution (which is a one-hot encoded vector in this scenario, with a 1 for the 
               correct class and 0 for others). [1, 4, 5, 6, 7, 8, 9, 10, 11]  
            3. Negative Log Likelihood: When the target is one-hot encoded (e.g., for the third class), 
               the cross-entropy loss simplifies to -log(Sm). This is because the log of the 
               probabilities of all incorrect classes will be negative infinity, and only the log of the
               probability of the correct class (Sm) contributes to the loss. Taking the negative makes 
               the loss a positive value that increases as the predicted probability (Sm) decreases, 
               which is the desired behavior. [2, 2, 4, 4, 5, 5, 12, 13]  
            4. Maximizing Likelihood: Minimizing cross-entropy loss is equivalent to maximizing the 
               likelihood of the observed data given the model. By minimizing -log(Sm), the model is 
               essentially being trained to make the predicted probability of the correct class as high 
               as possible. [2, 2, 12, 12, 14]  

            In summary:  The -log(Sm) expression is a convenient way to calculate the cross-entropy loss
            in a classification task with a softmax output and a one-hot encoded target, as it directly 
            relates to the negative log-likelihood of the correct class. [2, 5, 12]  


            [1]https://github.com/chen-hao-chao/dlsm-toy/blob/master/losses.py
            [2]https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2
            [3]https://developer.apple.com/documentation/accelerate/bnnslossfunctionsoftmaxcrossentropy?changes=_1&language=objc
            [4]https://ravimashru.dev/blog/2021-07-18-understanding-cross-entropy-loss/
            [5]https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
            [6]https://medium.com/biased-algorithms/log-loss-vs-cross-entropy-740df12d7526
            [7]https://machine-learning.paperspace.com/wiki/accuracy-and-loss
            [8]https://soulpageit.com/ai-glossary/cross-entropy-loss-explained/
            [9]https://caisplusplus.usc.edu/curriculum/neural-networks/optimization
            [10]https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/
            [11]https://stats.stackexchange.com/questions/392681/cross-entropy-loss-max-value
            [12]https://medium.com/intro-to-artificial-intelligence/the-link-between-maximum-likelihood-estimation-mle-and-cross-entropy-599cc1414753
            [13]https://medium.com/@devanshipratiher/understanding-loss-functions-for-deep-learning-segmentation-models-30187836b30a
            [14]https://arxiv.org/pdf/2102.11887

        Returns:
            _type_: _description_
        """
        # dl/dyj = Sm - 1 if j = m else Sm
        # https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2 

        probs = self.predict(x)
        probs[y] -= 1.0
        return probs
