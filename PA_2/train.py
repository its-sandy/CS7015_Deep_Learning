import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from scipy import ndimage
from skimage import transform

def conv2d(x, W, b, padding, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
	#x = tf.nn.bias_add(x, b)
	x = tf.nn.leaky_relu(x, alpha=0.1)
	#x = tf.layers.batch_normalization(x, training = training)
	return tf.nn.dropout(x, keep_prob = dropout_keep_prob_conv) 

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):  

	conv1 = conv2d(x, weights['wc1'], biases['bc1'], 'SAME')
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 'SAME')
	conv2 = maxpool2d(conv2, k=2)

	conv2 = tf.layers.batch_normalization(conv2, training = training)

	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 'SAME')
	conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 'SAME')
	conv4 = maxpool2d(conv4, k=2)

	conv4 = tf.layers.batch_normalization(conv4, training = training)

	conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], 'SAME')
	conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], 'VALID')
	conv6 = maxpool2d(conv6, k=2)

	fc1 = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.layers.batch_normalization(fc1, training = training)
	fc1 = tf.nn.dropout(fc1, keep_prob = dropout_keep_prob)
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc2 = tf.nn.leaky_relu(fc1, alpha = 0.1)
	fc2 = tf.layers.batch_normalization(fc2, training = training)
	fc2 = tf.nn.dropout(fc2, keep_prob = dropout_keep_prob)
	out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
	out = tf.layers.batch_normalization(out, training = training)
	return out

if __name__ == '__main__':
	np.random.seed(1234)
	#using argparse to get parameters according to the problem statement
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, help="learning rate", default = 0.001)
	parser.add_argument("--batch_size", type=int, help="batch size", default=20)
	parser.add_argument("--init", type=int, choices = [1, 2], help="weight initialization: 1. Xavier, 2. He", default = 1)
	parser.add_argument("--save_dir", type=str, help="directory to save model in", default = "models")
	parser.add_argument("--epochs", type=int, help="number of epochs", default=0)
	parser.add_argument("--dataAugment", type=int, choices = [0, 1], help="data augmentation", default = 1)
	parser.add_argument("--train", type=str, help="path to training dataset", default="train.csv")
	parser.add_argument("--val", type=str, help="path to validation dataset", default="valid.csv")
	parser.add_argument("--test", type=str, help="path to test dataset", default="test.csv")
	args = parser.parse_args()

	train_data = pd.read_csv(args.train)
	print('Read training data')
	val_data = pd.read_csv(args.val)
	print('Read validation data')
	test_data = pd.read_csv(args.test)
	print('Read test data')

	if args.init == 1:
		initializer = tf.contrib.layers.xavier_initializer()
	else:
		initializer = tf.initializers.he_normal()

	X_train = train_data.loc[:, 'feat0':'feat12287'].values.reshape((-1, 64, 64, 3)).astype(float)
	mn = X_train.mean(axis=0)
	stddev = X_train.std(axis=0)
	X_train = (X_train - mn)/stddev
	Y_train = np.eye(20)[train_data.loc[:,'label'].values]

	if args.dataAugment == 1:
		print('Performing data augmentation...')
		X_train_augment = X_train
		Y_train_augment = Y_train
		X_train_augment_1 = X_train
		for i in range(len(X_train_augment_1)):
			X_train_augment_1[i] = np.fliplr(X_train_augment_1[i])
		"""
		X_train_augment_1 = np.concatenate((X_train, X_train_augment))
		X_train_augment_2 = X_train_augment_1
		for i in range(len(X_train_augment_1)):
			#X_train_augment[i] = transform.warp(X_train_augment[i], transform.AffineTransform(translation = (np.random.randint(-10, 10), np.random.randint(-10, 10))), mode = 'edge')
			#noise = np.random.randn(64, 64, 3)*0.01
			#X_train_augment[i] += noise
			noise = np.random.randn(64, 64, 3)*0.01
			X_train_augment_2[i] += noise
			#X_train_augment_1[i] = transform.resize(X_train_augment_1[i][np.random.randint(5):64-np.random.randint(5), np.random.randint(5):64-np.random.randint(5), :], (64, 64, 3))
			#salt = [np.random.randint(0, i - 1, 50) for i in X_train_augment_2[i].shape]
			#pepper = [np.random.randint(0, i - 1, 50) for i in X_train_augment_2[i].shape]
			#X_train_augment_2[i][salt] = 1
			#X_train_augment_2[i][pepper] = 0
		X_train_augment = np.concatenate((X_train_augment, X_train_augment_2))
		Y_train_augment = np.concatenate((Y_train, Y_train, Y_train))
		#perm = np.random.permutation(len(X_train_augment))
		#X_train_augment = X_train_augment[perm]
		#Y_train_augment = Y_train_augment[perm]
		#X_train_augment = X_train_augment[:len(X_train)]
		#Y_train_augment = Y_train_augment[:len(Y_train)]
		"""
		X_train_augment = np.concatenate((X_train_augment, X_train_augment_1))
		Y_train_augment = np.concatenate((Y_train_augment, Y_train))
		"""
		X_train_augment_1 = X_train
		for i in range(len(X_train_augment_1)):
			X_train_augment_1[i] = np.flipud(X_train_augment_1[i])
		X_train_augment = np.concatenate((X_train_augment, X_train_augment_1))
		Y_train_augment = np.concatenate((Y_train_augment, Y_train))
		"""
		aug_img = tf.placeholder(tf.float32, shape = (1, 64, 64, 3))
		box = tf.placeholder(tf.float32, shape = (1, 4))
		scale_img = tf.image.crop_and_resize(aug_img, box, np.array([0]), np.array([64, 64]))

		X_train_augment_1 = X_train
		with tf.Session() as sess:
			for i in range(len(X_train_augment_1)):
				X_train_augment_1[i] = sess.run(scale_img, feed_dict={aug_img: X_train_augment_1[i][None, :, :, :], box: np.array([0.5-0.5*np.random.uniform(0.5, 1.0), 0.5-0.5*np.random.uniform(0.5, 1.0), 0.5+0.5*np.random.uniform(0.5, 1.0), 0.5+0.5*np.random.uniform(0.5, 1.0)])[None, :]})
		X_train_augment = np.concatenate((X_train_augment, X_train_augment_1))
		Y_train_augment = np.concatenate((Y_train_augment, Y_train))

		X_train_augment_1 = X_train
		for i in range(len(X_train_augment_1)):
			if np.random.rand()>0.5:
				X_train_augment_1[i] = transform.rotate(X_train_augment_1[i], np.random.uniform(10, 30), mode = 'symmetric')
			else:
				X_train_augment_1[i] = transform.rotate(X_train_augment_1[i], np.random.uniform(-30, -10), mode = 'symmetric')
		X_train_augment = np.concatenate((X_train_augment, X_train_augment_1))
		Y_train_augment = np.concatenate((Y_train_augment, Y_train))
		"""
		X_train_augment_1 = X_train
		for i in range(len(X_train_augment_1)):
			X_train_augment_1[i] = X_train_augment_1[i]*stddev+mn
			salt = [np.random.randint(0, i - 1, 50) for i in X_train_augment_1[i].shape]
			pepper = [np.random.randint(0, i - 1, 50) for i in X_train_augment_1[i].shape]
			X_train_augment_1[i][salt] = 255
			X_train_augment_1[i][pepper] = 0
			X_train_augment_1[i] = (X_train_augment_1[i]-mn)/stddev
		X_train_augment = np.concatenate((X_train_augment, X_train_augment_1))
		Y_train_augment = np.concatenate((Y_train_augment, Y_train))
		"""
		X_train = X_train_augment
		Y_train = Y_train_augment
		print('Data augmentation completed.')

	X_val = val_data.loc[:, 'feat0':'feat12287'].values.reshape((-1, 64, 64, 3)).astype(float)
	X_val = (X_val - mn)/stddev
	Y_val = np.eye(20)[val_data.loc[:,'label'].values]
	X_test = test_data.loc[:, 'feat0':'feat12287'].values.reshape((-1, 64, 64, 3)).astype(float)
	X_test = (X_test - mn)/stddev

	weights = {
		'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=initializer),
		'wc2': tf.get_variable('W1', shape=(3,3,32,32), initializer=initializer),
		'wc3': tf.get_variable('W2', shape=(3,3,32,64), initializer=initializer),
		'wc4': tf.get_variable('W3', shape=(3,3,64,64), initializer=initializer),
		'wc5': tf.get_variable('W4', shape=(3,3,64,64), initializer=initializer),
		'wc6': tf.get_variable('W5', shape=(3,3,64,128), initializer=initializer),
		'wd1': tf.get_variable('W6', shape=(6272,256), initializer=initializer),
		'out': tf.get_variable('W7', shape=(256,20), initializer=initializer),
	}

	biases = {
		'bc1': tf.get_variable('B0', shape=(32), initializer=initializer),
		'bc2': tf.get_variable('B1', shape=(32), initializer=initializer),
		'bc3': tf.get_variable('B2', shape=(64), initializer=initializer),
		'bc4': tf.get_variable('B3', shape=(64), initializer=initializer),
		'bc5': tf.get_variable('B4', shape=(64), initializer=initializer),
		'bc6': tf.get_variable('B5', shape=(128), initializer=initializer),
		'bd1': tf.get_variable('B6', shape=(256), initializer=initializer),
		'out': tf.get_variable('B7', shape=(20), initializer=initializer),
	}

	x = tf.placeholder("float", [None, 64, 64, 3])
	y = tf.placeholder("float", [None, 20])
	dropout_keep_prob = tf.placeholder("float", None)
	dropout_keep_prob_conv = tf.placeholder("float", None)
	training = tf.placeholder("bool", None)

	pred = conv_net(x, weights, biases)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(cost)

	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	saver = tf.train.Saver(max_to_keep = 50)

	with tf.Session() as sess:
		"""
		saver.restore(sess, "models/model71.ckpt")
		prediction = sess.run([pred], feed_dict={x: X_test, dropout_keep_prob: 1.0, dropout_keep_prob_conv: 1.0, training: False})
		print(prediction)
		output = np.argmax(prediction[0], 1)
		print(output)
		df = pd.DataFrame(output, columns = ['label'])
		print(df)
		df.to_csv('prediction.csv', index_label = 'id')
		"""
		sess.run(init)
		summary_writer = tf.summary.FileWriter('./Output', sess.graph)
		for i in range(args.epochs):
			perm = np.random.permutation(len(X_train))
			X_train = X_train[perm]
			Y_train = Y_train[perm]
			total_num_correct = 0.0
			total_loss = 0.0
			for batch in range(((len(X_train)-1)//args.batch_size)+1):
				if batch%10==0:
					print(batch)
				batch_x = X_train[batch*args.batch_size:min((batch+1)*args.batch_size,len(X_train))]
				batch_y = Y_train[batch*args.batch_size:min((batch+1)*args.batch_size,len(Y_train))]
				batch_x_augment = batch_x
				"""
				if args.dataAugment == 1:
					for j in range(batch_x_augment.shape[0]):
						if np.random.rand()>0.5:
							batch_x_augment[j] = np.fliplr(batch_x_augment[j])
						#if np.random.rand()>0.7:
						#	batch_x_augment[j] = transform.warp(batch_x_augment[j], transform.AffineTransform(translation = (np.random.randint(-5, 5), np.random.randint(-5, 5))), mode = 'edge')
						#	batch_x_augment[j] = transform.rotate(batch_x_augment[j]*255.0, np.random.uniform(-40, 40), mode = 'symmetric')/255.0
						#noise = np.random.randn(64, 64, 3)*abs(np.random.randn())*0.05
						noise = np.random.randn(64, 64, 3)*0.005
						batch_x_augment[j] += noise
						#if np.random.rand()>0.7:
						#	batch_x_augment[j] = np.minimum(batch_x_augment[j]*np.random.uniform(0.5, 1.5), 255)
				"""
				for j in range(batch_x_augment.shape[0]):
					noise = np.random.randn(64, 64, 3)*0.005
					batch_x_augment[j] += noise
				opt = sess.run(optimizer, feed_dict={x: batch_x_augment, y: batch_y, dropout_keep_prob: 0.5, dropout_keep_prob_conv: 1.0, training: True})
				loss, n_corr = sess.run([cost, num_correct], feed_dict={x: batch_x, y: batch_y, dropout_keep_prob: 1.0, dropout_keep_prob_conv: 1.0, training: False})
				total_loss += loss
				total_num_correct += n_corr
			print("Iter " + str(i) + ", Loss=", "{:.5f}".format(total_loss) , ", Training Accuracy=", "{:.5f}".format(total_num_correct/X_train.shape[0]))
			print("Optimization Finished!")

			test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: X_val, y : Y_val, dropout_keep_prob: 1.0, dropout_keep_prob_conv: 1.0, training: False})
			print("Validation Loss:","{:.6f}".format(valid_loss),", Validation Accuracy:","{:.5f}".format(test_acc))
			save_path = saver.save(sess, args.save_dir+"/model{}.ckpt".format(i))
		summary_writer.close()