from datetime import datetime
import sys
import numpy as np
from rnn import RNNLayer
from output import Softmax

# https://www.youtube.com/watch?v=4wuIOcD1LLI

class Model:
    def __init__(self, word_dim, hidden_dim=100, model_path="./rnn_model.npz", toload=False, bptt_truncate=4):
        self.model_path = model_path
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate # TODO: Future use
        if not toload:
            self.Wxh = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Whh = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.Wyh = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
            self.bh = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.by = np.zeros((self.word_dim, 1)) # output bias
        else:
            # Load the .npz file
            loaded_data = np.load(self.model_path)
            self.Wxh = loaded_data['Wxh']
            self.Whh = loaded_data['Whh']
            self.Wyh = loaded_data['Wyh']
            self.bh = loaded_data['bh']
            self.by = loaded_data['by']
            loaded_data.close()

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
        prev_h = np.zeros((self.hidden_dim, 1))
        # For each time step...
        for t in range(T):
            layer = RNNLayer()
            input = np.zeros((self.word_dim, 1))
            input[x[t]] = 1
            layer.forward(input, prev_h, self.Wxh, self.Whh, self.Wyh, self.bh, self.by)
            prev_h = layer.h
            layers.append(layer)
        return layers

    # For one sentence
    def bptt(self, inputs, targets): #, hprev):       
        layers = self.forward_propagation(inputs)

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWyh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Wyh)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dnext_h = np.zeros((self.hidden_dim, 1))
        output = Softmax()
        T = len(inputs)
        for t in reversed(range(T)):
            input = np.zeros((self.word_dim,1))
            input[inputs[t]] = 1
            prev_h = np.zeros((self.hidden_dim,1)) if t == 0 else layers[t-1].h
            
            dldy = output.loss_derivative(layers[t].z, targets[t])
            dnext_h, dWxh_t, dWhh_t, dWyh_t, dbh_t, dby_t = \
                layers[t].backward(input, prev_h, dnext_h, self.Wxh, self.Whh, self.Wyh, \
                                   self.bh, self.by, dldy)

            dWxh += dWxh_t
            dWhh += dWhh_t
            dWyh += dWyh_t
            dbh += dbh_t
            dby += dby_t

        for dparam in [dWxh, dWhh, dWyh, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dWxh, dWhh, dWyh, dbh, dby

    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.z)) for layer in layers]

    def calculate_loss_for_sentence(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.z, y[i])
        return loss / float(len(y))

    def calculate_total_loss_for_batch(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss_for_sentence(X[i], Y[i])
        return loss / float(len(Y))

    def sgd_step(self, x, y, learning_rate):
        dWxh, dWhh, dWyh, dbh, dby = self.bptt(x, y)
        self.Wxh -= learning_rate * dWxh
        self.Wyh -= learning_rate * dWyh
        self.Whh -= learning_rate * dWhh
        self.bh -= learning_rate * dbh
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

        np.savez(self.model_path, Wxh=self.Wxh, Whh=self.Whh, Wyh=self.Wyh, bh=self.bh, by=self.by)

        return losses
