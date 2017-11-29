import os

from __future__ import absolute_import, division, print_function
import match

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf 

def concat_elu(inputs):
	return tf.nn.elu(tf.concat(3, [-inputs, inputs]))

class GAN:
	def __init__(self, hidden_size, batch_size, learning_rate):
		self.input_tensor = tf.placeholder(tf.float32, [None, 96 * 96, 3])

		with arg_scope([layers.conv2d, layers.conv2d_transpose], activation_fn=concat_elu, normalizer_fn=layers.batch_norm,
			normalizer_params={'scale': True}):
			with tf.variable_scope("model"):
				D1 = self.discriminator(self.input_tensor)
				D_params_num = len(tf.trainable_variables())
				G = self.generator(tf.random_normal([batch_size, hidden_size, 3]))
				self.sampled_tensor = G

			with tf.variable_scope("model", reuse=True): # reues=True means that this part weight and struct is same as formal name="model" part
				D2 = self.discriminator(G)	

		D_loss = self.get_discrinator_loss(D1, D2)
		G_loss = self.get_generator_loss(D2)

		params = tf.trainable_variables()
		D_params = params[:D_params_num]
		G_params = params[D_params_num:]

		global_step = tf.contrib.framework.get_or_create_global_step()
		self.train_discrimator = layers.optimize_loss(D_loss, global_step, learning_rate/10, 'Adam', variables=D_params, update_ops=[])
		self.train_generator = layers.optimize_loss(G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])		

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def discriminator(self, input_tensor):
		net = tf.reshape(input_tensor, [-1, 96, 96, 3])
		net = layers.conv2d(net, 32, 5, stride=2)
		net = layers.conv2d(net, 64, 5, stride=2)
		net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
		net = layers.dropout(net, keep_prob=0.9)
		net = layers.flattrn(net)
		return layers.fully_connected(net, 1, activation_fn=None)

	def generator(self, input_tensor):
		net = tf.expand_dims(input_tensor,1)
		net = tf.expand_dims(net, 1)
		net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
		net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
		net = layers.conv2d_transpose(net, 32, 5, stride=2)
		net = layers.conv2d_transpose(net, 3, 5, stride=2, activation_fn=tf.nn.sigmoid)
		return net

	def get_discrinator_loss(self, d1, d2):
		return (losses.sigmoid_cross_entropy(d1, tf.ones(tf.shape(d1))) + losses.sigmoid_cross_entropy(d2, tf.zero(tf.shape(d1))))

	def get_generator_loss(self, d2):
		return losses.sigmoid_cross_entropy(d2, tf.ones(tf.shape(d2)))  # I think shape 1 is too small

	




