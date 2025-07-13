from datetime import datetime
import numpy as np
import sys
from lstm import LSTMLayer
from output import Softmax

# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))


class Model:
    def __init__(self, word_dim, hidden_dim=100, model_path="./lstm_model.npz", toload=False, bptt_truncate=4):
        self.model_path = model_path
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate # TODO: Future use
        if not toload:
            self.Ui = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wi = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.Uf = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wf = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.Ug = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wg = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.Uo = initWeights(word_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.Wo = initWeights(hidden_dim, hidden_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
            self.V = initWeights(hidden_dim, word_dim) # np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
            self.bi = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.bf = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.bg = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.bo = np.zeros((self.hidden_dim, 1)) # hidden bias
            self.by = np.zeros((self.word_dim, 1)) # output bias
        else:
            # Load the .npz file
            loaded_data = np.load(self.model_path)
            self.Ui = loaded_data['Ui']
            self.Wi = loaded_data['Wi']
            self.Uf = loaded_data['Uf']
            self.Wf = loaded_data['Wf']
            self.Ug = loaded_data['Ug']
            self.Wg = loaded_data['Wg']
            self.Uo = loaded_data['Uo']
            self.Wo = loaded_data['Wo']
            self.V  = loaded_data['V ']
            self.bi = loaded_data['bi']
            self.bf = loaded_data['bf']
            self.bg = loaded_data['bg']
            self.bo = loaded_data['bo']
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
            layer = LSTMLayer()
            input = np.zeros((self.word_dim, 1))
            input[x[t]] = 1
            layer.forward(input, next_h, next_c, self.Ui, self.Uf, self.Uo, self.Ug, \
                          self.Wi, self.Wf, self.Wo, self.Wg, self.V, \
                            self.bi, self.bf, self.bo, self.bg, self.by)
            next_h = layer.h
            layers.append(layer)
        return layers

    # For one sentence
    def bptt(self, inputs, targets): #, hnext):
        layers = self.forward_propagation(inputs)

        # backward pass: compute gradients going backwards
        dUi, dUf, dUo, dUg, dWi, dWf, dWo, dWg, dV, dbi, dbf, dbo, dbg, dby = \
            np.zeros_like(self.Ui), np.zeros_like(self.Uf), np.zeros_like(self.Uo), \
            np.zeros_like(self.Ug), np.zeros_like(self.Wi), np.zeros_like(self.Wf), \
            np.zeros_like(self.Wo), np.zeros_like(self.Wg), np.zeros_like(self.V), \
            np.zeros_like(self.bi), np.zeros_like(self.bf), np.zeros_like(self.bo), \
            np.zeros_like(self.bg), np.zeros_like(self.by)
 
        dnext_h_t = np.zeros((self.hidden_dim, 1))
        dnext_c_t = np.zeros((self.hidden_dim, 1))
        output = Softmax()
        T = len(inputs)
        for t in reversed(range(T)):
            input = np.zeros((self.word_dim,1))
            input[inputs[t]] = 1
            prev_h = np.zeros((self.hidden_dim,1)) if t == 0 else layers[t-1].h
            prev_c = np.zeros((self.hidden_dim,1)) if t == 0 else layers[t-1].c
            
            dldy = output.loss_derivative(layers[t].z, targets[t])
            dnext_h_t, dnext_c_t, dUi_t, dUf_t, dUo_t, dUg_t, dWi_t, dWf_t, dWo_t, dWg_t, \
                dV_t, dbi_t, dbf_t, dbo_t, dbg_t, dby_t = \
                layers[t].backward(input, prev_h, prev_c, dnext_h_t, dnext_c_t, self.Ui, self.Uf, self.Uo, self.Ug, \
                                   self.Wi, self.Wf, self.Wo, self.Wg, self.V, self.bi, self.bf, self.bo, \
                                    self.bg, self.by, dldy)

            dUi += dUi_t 
            dUf += dUf_t
            dUo += dUo_t
            dUg += dUg_t
            dWi += dWi_t
            dWf += dWf_t
            dWo += dWo_t
            dWg += dWg_t
            dbi += dbi_t
            dbf += dbf_t
            dbo += dbo_t
            dbg += dbg_t

            dV += dV_t
            dby += dby_t

        for dparam in [dUi, dUf, dUo, dUg, dWi, dWf, dWo, dWg, dV, dbi, dbf, dbo, dbg, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dUi, dUf, dUo, dUg, dWi, dWf, dWo, dWg, dV, dbi, dbf, dbo, dbg, dby

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
        dUi, dUf, dUo, dUg, dWi, dWf, dWo, dWg, dV, dbi, dbf, dbo, dbg, dby = self.bptt(x, y)
        self.Ui -= learning_rate * dUi 
        self.Uf -= learning_rate * dUf 
        self.Uo -= learning_rate * dUo
        self.Ug -= learning_rate * dUg
        self.Wi -= learning_rate * dWi
        self.Wf -= learning_rate * dWf
        self.Wo -= learning_rate * dWo
        self.Wg -= learning_rate * dWg
        self.V -= learning_rate * dV
        self.bi -= learning_rate * dbi 
        self.bf -= learning_rate * dbf
        self.bo -= learning_rate * dbo
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
                    Ui = self.Ui, \
                    Wi = self.Wi, \
                    Uf = self.Uf, \
                    Wf = self.Wf, \
                    Ug = self.Ug, \
                    Wg = self.Wg, \
                    Uo = self.Uo, \
                    Wo = self.Wo, \
                    V  = self.V, \
                    bi = self.bi, \
                    bf = self.bf, \
                    bg = self.bg, \
                    bo = self.bo, \
                    by = self.by
        )
        return losses
