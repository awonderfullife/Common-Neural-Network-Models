from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

class SimpleAutoEncoder(object):
	def __init__(self, sess, data,
		learning_rate=0.01,
		training_epochs=20,
		batch_size=256,
		display_step=1,
		examples_to_show=10,
		n_input=784,  # 28*28
		n_hidden_1=256,
		n_hidden_2=64,
		n_hidden_3=10,
		n_hidden_4=2
		):
		self.sess = sess
		self.data = data
		self.learning_rate = learning_rate
		self.training_epochs = training_epochs
		self.batch_size = batch_size
		self.display_step = display_step
		self.examples_to_show = examples_to_show
		self.n_input = n_input
		self.weights = {
		    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
		    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
		    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
		    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
		    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
		    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
		    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
		    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
		}
		self.biases = {
		    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
		    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
		    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
		    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
		    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
		    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
		    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
		    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
		}

		self.build_model()

	def build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.n_input])

		self.encoder_op = self.encoder()
		decoder_op = self.decoder()

		self.y_prediction = decoder_op
		y_true = self.x

		self.loss = tf.reduce_mean(tf.pow(y_true-self.y_prediction, 2))

	def encoder(self):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.weights['encoder_h1']),
										self.biases['encoder_b1']))
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
										self.biases['encoder_b2']))
		layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']),
										self.biases['encoder_b3']))
		layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights['encoder_h4']),
										self.biases['encoder_b4']))
		return layer_4

	def decoder(self):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_op, self.weights['decoder_h1']),
										self.biases['decoder_b1']))
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
										self.biases['decoder_b2']))
		layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['decoder_h3']),
										self.biases['decoder_b3']))
		layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights['decoder_h4']),
										self.biases['decoder_b4']))
		return layer_4

	def train(self):
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		try:
			init = tf.initialize_all_variables()
		except:
			init = tf.global_variables_initializer()
		self.sess.run(init)

		total_batch = int(self.data.train.num_examples/self.batch_size)
		for epoch in range(self.training_epochs):
			for i in range(total_batch):
				batch_xs, batch_ys = self.data.train.next_batch(self.batch_size)
				_, c = self.sess.run([optimizer, self.loss], feed_dict={self.x: batch_xs})
			if epoch % self.display_step == 0:
				print("Epoch: %04d" % (epoch+1), "cost={:.9f}".format(c))
		print ("Optimization Finished!")

		################ show the encoder-decoder result
		encoder_decoder = self.sess.run(self.y_prediction, feed_dict={self.x: self.data.test.images[:self.examples_to_show]})
		f, a = plt.subplots(2, 10, figsize=(10, 2))
		for i in range(self.examples_to_show):
			a[0][i].imshow(np.reshape(self.data.test.images[i], (28, 28)))
			a[1][i].imshow(np.reshape(encoder_decoder[i], (28, 28)))
		plt.show()

		################ show the laten space result
		color_list = []
		for list in self.data.test.labels:
			for i in range(10):
				if list[i] == 1:
					color_list.append(i)
					break

		encoder_result = self.sess.run(self.encoder_op, feed_dict={self.x: self.data.test.images})
		plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=color_list)
		plt.colorbar()
		plt.show()


def main():
	tf.set_random_seed(1)
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	sess = tf.Session()
	cnn = SimpleAutoEncoder(
		sess,
		data=mnist)

	cnn.train()



if __name__ == '__main__':
	main()