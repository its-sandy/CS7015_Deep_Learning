"""
Programming assignment 4 for CS7015 (Jan-May 2019)
"""

import numpy as np
import pandas as pd
import math
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import pickle

class Model:
    def __init__(self):
        self.n_visible = 784
        self.n_hidden = 500
        self.k = 1
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')

        self.X_train_true = self.train_data.loc[:,'feat0':'feat783'].to_numpy()
        self.X_train = np.copy(self.X_train_true)
        self.X_train[self.X_train_true<127] = 0.0
        self.X_train[self.X_train_true>=127] = 1.0
        self.Y_train = self.train_data.loc[:,'label'].to_numpy()

        print('Train dataset read')
        
        self.X_test_true = self.test_data.loc[:,'feat0':'feat783'].to_numpy()
        self.X_test = np.copy(self.X_test_true)
        self.X_test[self.X_test_true<127] = 0.0
        self.X_test[self.X_test_true>=127] = 1.0
        self.Y_test = self.test_data.loc[:,'label'].to_numpy()

        print('Test dataset read')

        self.W = np.random.randn(self.n_visible, self.n_hidden)*np.sqrt(2.0/(self.n_visible + self.n_hidden))
        self.b = np.random.randn(self.n_visible)
        self.c = np.random.randn(self.n_hidden)

        self.eta = 0.005

    def sigmoid(self, inp):
        return 1/(1+np.exp(-inp))

    def sample(self, prob):
        """Samples from the given probability distribution""" 
        return (np.random.random_sample(prob.shape) < prob).astype(np.float32)

    def forward(self, inp):
        """Calculates P(H|V)"""
        return self.sigmoid(inp @ self.W + self.c)

    def backward(self, inp):
        """Calculates P(V|H)"""        
        return self.sigmoid(self.W @ inp + self.b)

    def train(self):
        XY_train = list(zip(self.X_train, self.X_train_true, self.Y_train))
        random.shuffle(XY_train)
        self.X_train, self.X_train_true, self.Y_train = zip(*XY_train)

        #Use for plotting
        #fig, axs = plt.subplots(8, 8, sharex=True, sharey=True)
        #fig.subplots_adjust(hspace=0.5)
        #fig_tilde, axs_tilde = plt.subplots(8, 8, sharex=True, sharey=True)
        #fig_tilde.subplots_adjust(hspace=0.5)
        #num_iterations = 6400

        num_iterations = len(self.X_train)

        for i in range(num_iterations):
            print('Example', i+1, 'out of', num_iterations)
            v = self.X_train[i]
            v_tilde = v
            for j in range(self.k):
                h = self.sample(self.forward(v_tilde))
                v_tilde = self.sample(self.backward(h))
            self.W += self.eta * (np.outer(v, self.forward(v)) - np.outer(v_tilde, self.forward(v_tilde)))
            self.b += self.eta * (v - v_tilde)
            self.c += self.eta * (self.forward(v) - self.forward(v_tilde))
            #Use for plotting
            #if i%100 == 99:
            #    axs_tilde[i//800, (i//100)%8].set_title('Step {}'.format(i+1))
            #    axs_tilde[i//800, (i//100)%8].imshow(v_tilde.reshape((28, 28)))
            #    axs[i//800, (i//100)%8].set_title('Step {}'.format(i+1))
            #    axs[i//800, (i//100)%8].imshow(self.X_train[i].reshape((28, 28)))
        #plt.show()

    def save_weights(self):
        with open('weights_{}.pkl'.format(self.n_hidden), 'wb') as f:
            pickle.dump((self.W, self.b, self.c), f)

    def load_weights(self):
        with open('weights_{}.pkl'.format(self.n_hidden), 'rb') as f:
            self.W, self.b, self.c = pickle.load(f)

    def plot_tsne(self):
        tsne = TSNE(verbose=1, n_iter=1000, learning_rate = 50)
        #using expected value of hidden variables
        X_hidden = self.forward(self.X_test)
        X_2D = tsne.fit_transform(X_hidden)
        np.savetxt("X_2D.csv", X_2D, delimiter=",")

        #to load saved 2D representation, if needed
        #X_2D = np.genfromtxt('X_2D.csv', delimiter=',')

        plt.scatter(X_2D[:, 0], X_2D[:, 1], c = self.Y_test, cmap = "magma")
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)

    model = Model()
    model.train()
    model.save_weights()
    #model.load_weights()
    model.plot_tsne()