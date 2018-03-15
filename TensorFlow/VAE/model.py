import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

class VAE:
	def __init__(self, hidden_size, batch_size, learning_rate):
		self.input_tensor = tf.placeholder(tf.float32, [None, 28*28])
		with arg_scope([layers.conv2d, layers.conv2d_transpose], 
			activation_fn=tf.nn.relu,
			normalizer_fn=layers.batch_norm,
			normalizer_params={'scale':True}):
			with tf.variable_scope("vae_model") as scope:
				encoded = self.encoder(self.input_tensor, hidden_size*2)

				mean = encoded[:, hidden_size]
				stddev = tf.sqrt(tf.exp(encoded[:, hidden_size:]))

				epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])
				input_sample = mean + epsilon*stddev

				output_tensor = self.decoder(input_sample)

			with tf.variable_scope('vae_model', reuse=True) as scope:
				self.sampled_tensor = self.decoder(tf.random_normal([batch_size, hidden_size]))

		vae_loss = self.__get_vae_cost(mean, stddev)
		rec_loss = self.__get_reconstruction_cost(output_tensor, self.input_tensor)
		loss = vae_loss + rec_loss

		self.train = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(
		), learning_rate=learning_rate, optimizer='Adam', update_ops=[])

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def __get_vae_cost(self, mean, stddev, epsilon=1e-8):
		return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
		                2.0 * tf.log(stddev + epsilon) - 1.0))

	def __get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
		return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -
			(1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

	def update_params(self, input_tensor):
		return self.sess.run(self.train, {self.input_tensor: input_tensor})

	def encoder(self, input_tensor, output_size):
		net = tf.reshape(input_tensor, [-1, 28, 28, 1])
		net = layers.conv2d(net, 32, 5, stride=2)
		net = layers.conv2d(net, 64, 5, stride=2)
		net = layers.conv2d(net, 128, 5, stride=2)
		net = layers.dropout(net, keep_prob=0.9)
		net = layers.flatten(net)  # [-1, a, b, c] ==> [-1, a*b*c]
		return layers.fully_connected(net, output_size, activation_fn=None)

	def decoder(self, input_tensor):
		net = tf.expand_dims(input_tensor, 1) # [-1, output_size] ==> [-1, 1, output_size]
		net = tf.expand_dims(net, 1) # [-1, 1, output_size, 1] ==> [-1, 1, 1, output_size]
		net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
		net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
		net = layers.conv2d_transpose(net, 32, 5, stride=2)
		net = layers.conv2d_transpose(net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
		net = layers.flatten(net)
		return net

if __name__ == '__main__':
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	batch_size = 128
	updates_per_epoch = 1000
	epoches = 1

	model = VAE(hidden_size=128, batch_size=batch_size, learning_rate=1e-2)

	for epoch in range(epoches):
		training_loss = 0.0

		for i in range(updates_per_epoch):
			images, _ = mnist.train.next_batch(batch_size)
			loss_value = model.update_params(images)
			training_loss += loss_value
			print i, training_loss, loss_value

		training_loss = training_loss / \
			(updates_per_epoch * batch_size)

		print("Loss %f" % training_loss)

