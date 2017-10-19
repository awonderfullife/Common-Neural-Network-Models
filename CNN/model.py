
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class CNN(object):
	def __init__(self, sess, data_set, input_width=28, input_height=28, patch_sizex=5, patch_sizey=5,
		batch_size=100):
		self.sess = sess
		self.data = data_set
		self.input_width = input_width
		self.input_height = input_height
		self.patch_sizex = patch_sizex
		self.patch_sizey = patch_sizey
		self.batch_size = batch_size
		self.build_model()

	def build_model(self):
		self.xs = tf.placeholder(tf.float32, [None, self.input_width*self.input_height])/255.   # 28x28
		self.ys = tf.placeholder(tf.float32, [None, 10])
		self.keep_prob = tf.placeholder(tf.float32)

		# gray picture with 28x28
		x_image = tf.reshape(self.xs, [-1,self.input_width,self.input_height,1]) 
		self.cnn_structure(x_image)

		# the error between prediction and real data
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction), reduction_indices=[1])) 
		self.cross_entropy = cross_entropy


	def cnn_structure(self, x_image):
		# conv1 layer
		w_conv1 = self.weight_variable([self.patch_sizex,self.patch_sizey,1,32]) #patch 5x5, insize height 1, outsize height 32 so-call channels
		b_conv1 = self.bias_variable([32])
		h_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)  # similar to y = ax + b; output size 28x28x32
		h_pool1 = self.max_pool_2x2(h_conv1)							  # output size 14x14x32

		# conv2 layer
		w_conv2 = self.weight_variable([self.patch_sizex,self.patch_sizex,32,64]) #patch 5x5, insize height 32, outsize height 64 so-call channels
		b_conv2 = self.bias_variable([64])
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)  # similar to y = ax + b; output size 14x14x64
		h_pool2 = self.max_pool_2x2(h_conv2)							  # output size 7x7x64

		# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
		conv_size = self.input_width/4 * self.input_height/4 * 64
		h_pool2_flat = tf.reshape(h_pool2,[-1,conv_size])

		#function layer
		w_fc1 = self.weight_variable([conv_size, 1024]) # normal network layer
		b_fc1 = self.bias_variable([1024])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
		h_fc1_dropout = tf.nn.dropout(h_fc1, self.keep_prob)

		#function layer
		w_fc2 = self.weight_variable([1024, 10]) # use for juder 0~9
		b_fc2 = self.bias_variable([10])
		self.prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)

	def train(self):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		try:
			init = tf.initialize_all_variables()
		except:
			init = tf.global_variables_initializer()
		self.sess.run(init)

		for i in range(1000):
			batch_xs, batch_ys = self.data.train.next_batch(self.batch_size)
			self.sess.run(train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.5})
			if i % 50 == 0:
				print(self.compute_accuracy(self.data.test.images[:1000], self.data.test.labels[:1000]))


	def compute_accuracy(self, v_xs, v_ys):
		y_pre = self.sess.run(self.prediction, feed_dict={self.xs:v_xs, self.keep_prob:1})
		correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		result = self.sess.run(accuracy, feed_dict={self.xs:v_xs, self.ys:v_ys, self.keep_prob: 1})
		return result

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	# x is the picture 
	def conv2d(self, x, weight):
		# stride [1, x_move, y_move, 1]   in tensorflow, the begin and end value should be 1
		# SAME the result have same x,y length with original picture;  VALID for smaller than original
		return tf.nn.conv2d(x, weight, strides=[1,1,1,1], padding="SAME")

	def max_pool_2x2(self, x):
		# you could use max poling or average polling
		# strides[0] = strides[1] = 1
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")



def main():
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	sess = tf.Session()
	cnn = CNN(
		sess,
		data_set=mnist,
		input_width=28,
		input_height=28,
		patch_sizex=5,
		patch_sizey=5,
		batch_size=100)

	cnn.train()



if __name__ == '__main__':
	main()