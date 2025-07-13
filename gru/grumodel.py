from datetime import datetime
import numpy as np
import sys
from gru import GRULayer
from output import Softmax

# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))


class Model:
    def __init__(self, word_dim, hidden_dim=100, model_path="./gru_model.npz", toload=False, bptt_truncate=4):
        self.model_path = model_path
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate # TODO: Future use
        if not toload:
            self.Ur = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wr = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.Uz = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wz = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.Ug = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wg = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.V = initWeights(hidden_dim, word_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
            self.br = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.bz = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.bg = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.by = np.zeros((self.word_dim, 1)) # output bias
        else:
            # Load the .npz file
            loaded_data = np.load(self.model_path)
            self.Ur = loaded_data['Ur']
            self.Wr = loaded_data['Wr']
            self.Uz = loaded_data['Uz']
            self.Wz = loaded_data['Wz']
            self.Ug = loaded_data['Ug']
            self.Wg = loaded_data['Wg']
            self.V  = loaded_data['V ']
            self.br = loaded_data['br']
            self.bz = loaded_data['bz']
            self.bg = loaded_data['bg']
            self.by = loaded_data['by']

    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
    # For one sentence
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        layers = []
        next_h = np.zeros((self.hidden_dim, 1))
        next_c = np.zeros((self.hidden_dim, 1))
        # For each time step...
        for t in range(T):
            layer = GRULayer()
            input = np.zeros((self.word_dim, 1))
            input[x[t]] = 1
            layer.forward(input, next_h, self.Ur, self.Uz, self.Ug, \
                          self.Wr, self.Wz, self.Wg, self.V, \
                            self.br, self.bz, self.bg, self.by)
            next_h = layer.h
            layers.append(layer)
        return layers

    # For one sentence
    def bptt(self, inputs, targets): #, hnext):
        layers = self.forward_propagation(inputs)

        # backward pass: compute gradients going backwards
        dUr, dUz, dUg, dWr, dWz, dWg, dV, dbr, dbz, dbg, dby = \
            np.zeros_like(self.Ur), np.zeros_like(self.Uz), np.zeros_like(self.Ug), \
            np.zeros_like(self.Wr), np.zeros_like(self.Wz), np.zeros_like(self.Wg), \
            np.zeros_like(self.V), \
            np.zeros_like(self.br), np.zeros_like(self.bz), np.zeros_like(self.bg), \
            np.zeros_like(self.by)
 
        dnext_h_t = np.zeros((self.hidden_dim, 1))
        output = Softmax()
        T = len(inputs)
        for t in reversed(range(T)):
            input = np.zeros((self.word_dim,1))
            input[inputs[t]] = 1
            prev_h = np.zeros((self.hidden_dim,1)) if t == 0 else layers[t-1].h
            
            dldy = output.loss_derivative(layers[t].y, targets[t])
            dnext_h_t, dUr_t, dUz_t, dUg_t, dWr_t, dWz_t, dWg_t, \
                dV_t, dbr_t, dbz_t, dbg_t, dby_t = \
                layers[t].backward(input, prev_h, dnext_h_t, self.Ur, self.Uz, self.Ug, \
                                   self.Wr, self.Wz, self.Wg, self.V, self.br, self.bz, \
                                    self.bg, self.by, dldy)

            dUr += dUr_t 
            dUz += dUz_t
            dUg += dUg_t
            dWr += dWr_t
            dWz += dWz_t
            dWg += dWg_t
            dbr += dbr_t
            dbz += dbz_t
            dbg += dbg_t

            dV += dV_t
            dby += dby_t

        for dparam in [dUr, dUz, dUg, dWr, dWz, dWg, dV, dbr, dbz, dbg, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dUr, dUz, dUg, dWr, dWz, dWg, dV, dbr, dbz, dbg, dby

    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.y)) for layer in layers]

    def calculate_loss_for_sentence(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.y, y[i])
        return loss / float(len(y))

    def calculate_total_loss_for_batch(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss_for_sentence(X[i], Y[i])
        return loss / float(len(Y))

    def sgd_step(self, x, y, learning_rate):
        dUr, dUz, dUg, dWr, dWz, dWg, dV, dbr, dbz, dbg, dby = self.bptt(x, y)
        self.Ur -= learning_rate * dUr 
        self.Uz -= learning_rate * dUz 
        self.Ug -= learning_rate * dUg
        self.Wr -= learning_rate * dWr
        self.Wz -= learning_rate * dWz
        self.Wg -= learning_rate * dWg
        self.V -= learning_rate * dV
        self.br -= learning_rate * dbr 
        self.bz -= learning_rate * dbz
        self.bg -= learning_rate * dbg
        self.by -= learning_rate * dby

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        """Main training def

        Args:
            X (int): Batch of training examples. Each example is a list of integers.
            Y (int): Batch of targets. One each for X. Each target is a list of intergers.
            learning_rate (float, optional): Defaults to 0.005.
            nepoch (int, optional): Defaults to 100.
            evaluate_loss_after (int, optional): Defaults to 5.

        Returns:
            losses (float):
        """    
        num_examples_seen = 0
        losses = []
        for epoch in range(nepoch):
            print(f"Started epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss_for_batch(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{time}: Loss after num_examples_seen={num_examples_seen} epoch={epoch}: {loss}")
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print(f"Setting learning rate to {learning_rate}")
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
            print(f"After {num_examples_seen} examples, loss is {losses}")

        np.savez(self.model_path, \
            Ur = self.Ur, \
            Wr = self.Wr, \
            Uz = self.Uz, \
            Wz = self.Wz, \
            Ug = self.Ug, \
            Wg = self.Wg, \
            V  = self.V, \
            br = self.br, \
            bz = self.bz, \
            bg = self.bg, \
            by = self.by
        )
        return losses
