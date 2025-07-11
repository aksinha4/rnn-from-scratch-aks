import numpy as np

import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from preprocessing import getSentenceData
from grumodel import Model

word_dim = 8000
hidden_dim = 100
X_train, y_train = getSentenceData('../data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
gru = Model(word_dim, hidden_dim, model_path="./gru_model.npz", toload=False)

losses = gru.train(X_train[:100], y_train[:100], learning_rate=0.005, nepoch=10, evaluate_loss_after=1)