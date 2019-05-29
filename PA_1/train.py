"""
Programming assignment 1 for CS7015 (Jan-May 2019)
"""

##TODO: sq error loss, anneal, opt, save model, log files

import numpy as np
import pandas as pd
import math
import argparse
import pickle
import os
from sklearn import decomposition

class Model:
	def __init__(self, args):
		self.n_output_features = 10
		self.args = args
		if args.num_hidden>0:
			self.args.sizes = [self.args.num_input] + list(map(int, args.sizes.split(','))) + [self.n_output_features]
		else:
			self.args.sizes = [self.args.num_input, self.n_output_features]
		self.train_data = pd.read_csv(self.args.train)
		self.valid_data = pd.read_csv(self.args.validation)
		self.test_data = pd.read_csv(self.args.test)

		self.X_train = self.train_data.loc[:,'feat0':'feat783'].to_numpy()/255.0
		mn = self.X_train.mean(axis=0)
		stddev = self.X_train.std(axis=0)
		self.X_train = (self.X_train - mn)/stddev
		pca = decomposition.PCA(n_components = self.args.num_input)
		pca.fit(self.X_train)
		self.X_train = pca.transform(self.X_train)
		self.Y_train = self.train_data.loc[:,'label'].to_numpy()
		self.X_valid = self.valid_data.loc[:,'feat0':'feat783'].to_numpy()/255.0
		self.X_valid = (self.X_valid - mn)/stddev
		self.X_valid = pca.transform(self.X_valid)
		self.Y_valid = self.valid_data.loc[:,'label'].to_numpy()
		self.X_test = self.test_data.loc[:,'feat0':'feat783'].to_numpy()/255.0
		self.X_test = (self.X_test - mn)/stddev
		self.X_test = pca.transform(self.X_test)
		self.one_hot_outputs = np.eye(10)[self.Y_train]

		if self.args.pretrain:
			self.load_weights(self.args.state)
		else:
			self.W = [(np.random.randn(self.args.sizes[i], self.args.sizes[i+1])*np.sqrt(1/self.args.sizes[i])) for i in range(self.args.num_hidden+1)]
			self.b = [(np.random.randn(self.args.sizes[i+1])*np.sqrt(1/self.args.sizes[i+1])) for i in range(self.args.num_hidden+1)]

		self.beta_1 = 0.9
		self.beta_2 = 0.999
		self.epsilon = 1e-8

		if self.args.write_log:
			try:
				os.remove(self.args.expt_dir +'log_train.txt')
			except:
				pass
			try:
				os.remove(self.args.expt_dir +'log_val.txt')
			except:
				pass

	def softmax(self, inp):
		inp = inp - np.amax(inp)	#subtract max value from all elements to prevent overflow while taking exp
		inp = np.exp(inp)
		return inp/np.sum(inp, axis=1)[:, None]

	def activation(self, inp, func):
		if func == "tanh":
			return np.tanh(inp)
		elif func == "relu":
			inp[inp<0] = 0
			return inp
		else: #sigmoid
			return 1/(1+np.exp(-inp))

	def deriv(self, inp, func):
		if func == "tanh":
			return 1-inp**2
		elif func == "relu":
			inp[inp<=0] = 0
			inp[inp>0] = 1
			return inp
		else: #sigmoid
			return inp*(1-inp)

	def eval(self, inp):
		"""
		Returns output of each layer for input "inp" ("inp" may be a collection of examples)
		"""
		h = [np.zeros(i) for i in self.args.sizes]
		h[0] = inp
		for i, (W_i, b_i) in enumerate(zip(self.W, self.b)):
			h[i+1] = np.matmul(h[i], W_i) + b_i
			if i<self.args.num_hidden:
				h[i+1] = self.activation(h[i+1][:], self.args.activation)
			else:
				h[i+1] = self.softmax(h[i+1])
		return h
		
	def error_fn(self, target, output):
		if self.args.loss == "ce":
			return -np.tensordot(target, np.log(output+1e-10))/np.shape(target)[0]
		elif self.args.loss == "sq":
    		#should we divide by number of samples?	
			return np.sum(np.square(target-output))/(2.0*np.shape(target)[0])

	def acc_fn(self, target, output):
		"""
		Takes the predicted and actual outputs in one-hot form and returns fraction of correctly predicted samples
		"""
		output = np.eye(10)[np.argmax(output, axis=1)]
		return np.tensordot(target, output)/target.shape[0]

	def train(self):
		timestep = 1
		temp_W, temp_b = [np.zeros_like(w) for w in self.W] , [np.zeros_like(w) for w in self.b]
		dW, db, dW_prev, db_prev = [np.zeros_like(w) for w in self.W] , [np.zeros_like(w) for w in self.b], [np.zeros_like(w) for w in self.W], [np.zeros_like(w) for w in self.b]
		v_W, v_b, v_W_prev, v_b_prev = [np.zeros_like(w) for w in self.W] , [np.zeros_like(w) for w in self.b], [np.zeros_like(w) for w in self.W], [np.zeros_like(w) for w in self.b]
		
		da_curr = [np.zeros((self.args.batch_size, self.args.sizes[i])) for i in range(1, self.args.num_hidden+2)]
		prev_valid_acc, valid_acc = 0, -1

		for epoch in range(self.args.state+1, self.args.state+self.args.epochs+1):
			#change order in self.X_train itself
			perm = np.random.permutation(len(self.X_train))
			self.X_train = self.X_train[perm]
			self.Y_train = self.Y_train[perm]
			self.one_hot_outputs = self.one_hot_outputs[perm]

			while True:
				if self.args.anneal:
					temp_W[:] = self.W
					temp_b[:] = self.b
					temp_timestep = timestep
				batch_inds = list(range(0, self.X_train.shape[0], self.args.batch_size))
				np.random.shuffle(batch_inds)

				for step, i in enumerate(batch_inds):
					curr_batch_size = min(self.args.batch_size, self.X_train.shape[0]-i)
					if self.args.opt == "nag":
						for k in range(self.args.num_hidden+1):
							self.W[k] += self.args.momentum*dW_prev[k]
							self.b[k] += self.args.momentum*db_prev[k]
					h_curr = self.eval(self.X_train[i:i+curr_batch_size, :])

					if self.args.loss == "ce":
						da_curr[-1][0:curr_batch_size, :] = (h_curr[-1][:,:] - self.one_hot_outputs[i:i+curr_batch_size, :])
					elif self.args.loss == "sq":
						#shape(M3) = N*10*10
						#shape(M2) = N*10*1
						M3_1 = np.zeros((curr_batch_size,10,10))
						M3_1[:,np.arange(10),np.arange(10)] = h_curr[-1][:,:]
						M3_2 = np.matmul((h_curr[-1][:,:]).reshape((-1,10,1)), (h_curr[-1][:,:]).reshape((-1,1,10)))
						M3 = M3_1 - M3_2
						M2 = (h_curr[-1][:,:] - self.one_hot_outputs[i:i+curr_batch_size, :])[:,:,None]	
						da_curr[-1][0:curr_batch_size, :] = np.matmul(M3,M2)[:,:,0]	
					
					for j in range(self.args.num_hidden-1, -1, -1):
						da_curr[j][0:curr_batch_size, :] = np.matmul(da_curr[j+1][0:curr_batch_size, :], self.W[j+1].T)*self.deriv(h_curr[j+1][:], self.args.activation)
					
					for k in range(self.args.num_hidden+1):
						if self.args.opt == "adam":
							dW[k] = self.beta_1 * dW_prev[k] + (1.0 - self.beta_1) * (np.matmul(h_curr[k].T, da_curr[k][0:curr_batch_size, :]) + self.args.lamb*self.W[k])
							db[k] = self.beta_1 * db_prev[k] + (1.0 - self.beta_1) * (np.sum(da_curr[k][0:curr_batch_size, :], axis=0) + self.args.lamb*self.b[k])
							v_W[k] = self.beta_2 * v_W_prev[k] + (1.0 - self.beta_2) * (np.matmul(h_curr[k].T, da_curr[k][0:curr_batch_size, :]) + self.args.lamb*self.W[k])**2
							v_b[k] = self.beta_2 * v_b_prev[k] + (1.0 - self.beta_2) * (np.sum(da_curr[k][0:curr_batch_size, :], axis=0) + self.args.lamb*self.b[k])**2
							dW_prev[k][:] = dW[k]
							db_prev[k][:] = db[k]
							v_W_prev[k][:] = v_W[k]
							v_b_prev[k][:] = v_b[k]
							dW[k] /= 1.0-math.pow(self.beta_1, timestep)
							db[k] /= 1.0-math.pow(self.beta_1, timestep)
							v_W[k] /= 1.0-math.pow(self.beta_2, timestep)
							v_b[k] /= 1.0-math.pow(self.beta_2, timestep)
							self.W[k] -= (self.args.lr/np.sqrt(self.epsilon+v_W[k]))*dW[k]
							self.b[k] -= (self.args.lr/np.sqrt(self.epsilon+v_b[k]))*db[k]
						else:
							dW[k] = -self.args.lr*(np.matmul(h_curr[k].T, da_curr[k][0:curr_batch_size, :]) + self.args.lamb*self.W[k])
							db[k] = -self.args.lr*(np.sum(da_curr[k][0:curr_batch_size, :], axis=0) + self.args.lamb*self.b[k])
							if self.args.opt == "momentum":
								dW[k] += self.args.momentum*dW_prev[k] 
								db[k] += self.args.momentum*db_prev[k]
							self.W[k] += dW[k]
							self.b[k] += db[k]
							if self.args.opt == "nag":
								dW[k] += self.args.momentum*dW_prev[k] 
								db[k] += self.args.momentum*db_prev[k]
							dW_prev[k][:] = dW[k]
							db_prev[k][:] = db[k]
					timestep+=1

					if self.args.write_log and (step+1)%100 == 0:
						self.update_log(epoch, step+1)

				if not self.args.anneal:
					break
				valid_acc = self.acc_fn(np.eye(10)[self.Y_valid], self.eval(self.X_valid)[-1])
				if prev_valid_acc>valid_acc and epoch>0:
					self.args.lr /= 2.0
					self.W = temp_W
					self.b = temp_b
					timestep = temp_timestep
				else:
					prev_valid_acc = valid_acc
					break
			print('\nEpoch', epoch)
			print('Training Loss:', self.error_fn(self.one_hot_outputs, self.eval(self.X_train)[-1]))
			print('Validation Loss:', self.error_fn(np.eye(10)[self.Y_valid], self.eval(self.X_valid)[-1]))
			print('Training accuracy:', self.acc_fn(self.one_hot_outputs, self.eval(self.X_train)[-1])*100)
			print('Validation accuracy:', self.acc_fn(np.eye(10)[self.Y_valid], self.eval(self.X_valid)[-1])*100)
			print('Learning Rate:', self.args.lr)

			if self.args.save_models:
				self.save_weights(epoch)

	def write_test_output(self):
		output = np.argmax(self.eval(self.X_test)[-1], axis=1)
		df = pd.DataFrame(output, columns = ['label'])
		print(df)
		df.to_csv(self.args.expt_dir +'predictions_{}.csv'.format(self.args.state + self.args.epochs), index_label = 'id')

	def save_weights(self, epoch):
		with open(self.args.save_dir +'weights_{}.pkl'.format(epoch), 'wb') as f:
			pickle.dump((self.W, self.b), f)

	def load_weights(self, state):
		with open(self.args.save_dir +'weights_{}.pkl'.format(state), 'rb') as f:
			self.W, self.b = pickle.load(f)

	def update_log(self, epoch, step):
		with open(self.args.expt_dir +'log_train.txt', 'a') as f:
			print('Epoch %d, Step %d, Loss: %f, Error: %0.2f%%, lr: %f' % (epoch, step, self.error_fn(self.one_hot_outputs, self.eval(self.X_train)[-1]), (1-self.acc_fn(self.one_hot_outputs, self.eval(self.X_train)[-1]))*100, self.args.lr), file = f)
		with open(self.args.expt_dir +'log_val.txt', 'a') as f:
			print('Epoch %d, Step %d, Loss: %f, Error: %0.2f%%, lr: %f' % (epoch, step, self.error_fn(np.eye(10)[self.Y_valid], self.eval(self.X_valid)[-1]), (1-self.acc_fn(np.eye(10)[self.Y_valid], self.eval(self.X_valid)[-1]))*100, self.args.lr), file = f)


if __name__ == '__main__':
	np.random.seed(1234)
	#using argparse to get parameters according to the problem statement
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, help="learning rate", default = 0.001)
	parser.add_argument("--lamb", type=float, help="L2 regularization parameter", default = 0.0)
	parser.add_argument("--momentum", type=float, help="momentum", default = 0.9)
	parser.add_argument("--num_hidden", type=int, help="number of hidden layers", default=0)
	parser.add_argument("--num_input", type=int, help="dimensionality after PCA reduction", default=784)
	parser.add_argument("--sizes", type=str, help="comma-separated list for size of each hidden layer", default="0")
	parser.add_argument("--activation", type=str, choices = ["tanh", "sigmoid", "relu"], help="activation function", default="sigmoid")
	parser.add_argument("--loss", type=str, choices = ["sq", "ce"], help="loss function", default="ce")
	parser.add_argument("--opt", type=str, choices = ["gd", "momentum", "nag", "adam"], help="optimization algorithm", default="gd")
	parser.add_argument("--batch_size", type=int, help="batch size", default=100)
	parser.add_argument("--epochs", type=int, help="number of epochs", default=0)
	parser.add_argument("--anneal", action="store_true", help="flag to anneal learning rate")
	parser.add_argument("--save_dir", type=str, help="directory to save pickled model in", default = "")
	parser.add_argument("--expt_dir", type=str, help="directory to save log files in", default = "")
	parser.add_argument("--train", type=str, help="path to training dataset", default="train.csv")
	parser.add_argument("--validation", type=str, help="path to validation dataset", default="valid.csv")
	parser.add_argument("--test", type=str, help="path to test dataset", default="test.csv")
	parser.add_argument("--pretrain", action="store_true", help="use pretrained model")
	parser.add_argument("--state", type=int, help="state of pretrained model to be used", default = 0)
	parser.add_argument("--testing", action="store_true", help="use pretrained model", default = False)
	parser.add_argument("--save_models", action="store_true", help="save model after epoch of training")
	parser.add_argument("--write_log", action="store_true", help="write log after every 100 steps ")
	args = parser.parse_args()

	model = Model(args)
	if not model.args.testing:
		model.train()
	model.write_test_output()