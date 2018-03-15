import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoder:
	def __init__(self, sess, data, batch_size=128, training_epochs=1, examples_to_show=128):
		self.sess = sess
		self.data = data
		self.batch_size = batch_size
		self.training_epochs = training_epochs
		self.examples_to_show = examples_to_show

		self.build_model()

	def build_model(self):
		self.x = tf.placeholder(tf.float32, [self.batch_size, 784], name="x")
		self.keep_prob = tf.placeholder(tf.float32)


		input_img = tf.reshape(self.x, [self.batch_size, 28, 28, 1])
		self.laten_vector = self.encoder(input_img)
		self.output_img = self.decoder(self.laten_vector)

		self.cost = tf.reduce_mean(tf.pow(self.output_img-input_img, 2))

	def train(self):
		optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

		try:
			init = tf.initialize_all_variables()
		except:
			init = tf.global_variables_initializer()
		self.sess.run(init)

		total_batch = int(self.data.train.num_examples/self.batch_size)
		for epoch in range(self.training_epochs):
			for i in range(total_batch):
				batch_x, batch_label = self.data.train.next_batch(self.batch_size)
				self.sess.run(optimizer, feed_dict={self.x:batch_x, self.keep_prob:0.75})
				print i, total_batch
			if epoch % 1 == 0:
				print "Epoch: %04d cost="%(epoch+1), self.compute_l2_loss(self.data.test.images[:self.batch_size])

		encoder_decoder = self.sess.run(self.output_img, feed_dict={self.x: self.data.test.images[:self.examples_to_show]})
		f, a = plt.subplots(2, 10, figsize=(10, 2))
		for i in range(10):
			a[0][i].imshow(np.reshape(self.data.test.images[i], (28, 28)))
			a[1][i].imshow(np.reshape(encoder_decoder[i], (28, 28)))
		plt.show()

		color_list = []
		for list in self.data.test.labels:
			for i in range(10):
				if list[i] == 1:
					color_list.append(i)
					break

		encoder_result = self.sess.run(self.laten_vector, feed_dict={self.x: self.data.test.images[:self.examples_to_show]})
		plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=color_list[:self.examples_to_show])
		plt.colorbar()
		plt.show()

	def encoder(self, input_img):
		conv1_out = tf.layers.conv2d(inputs=input_img, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
		pool1_out = tf.layers.max_pooling2d(inputs=conv1_out, pool_size=[2,2], strides=2)
		conv2_out = tf.layers.conv2d(inputs=pool1_out, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
		pool2_out = tf.layers.max_pooling2d(inputs=conv2_out, pool_size=[2,2], strides=2)

		dense_input = tf.reshape(pool2_out, [self.batch_size,7*7*64])
		dropout1 = tf.layers.dropout(inputs=dense_input, rate=self.keep_prob)
		dense1_out = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
		dropout2 = tf.layers.dropout(inputs=dense1_out, rate=self.keep_prob)

		return tf.layers.dense(inputs=dropout2, units=10, activation=tf.nn.relu)

	def decoder(self, latent_vector):

		dense1_out1 = tf.layers.dense(inputs=latent_vector, units=1024, activation=tf.nn.relu)
		dropout = tf.layers.dropout(inputs=dense1_out1, rate=self.keep_prob)
		dense1_out2 = tf.layers.dense(inputs=dropout, units=3136, activation=tf.nn.relu)
		decon1_input = tf.reshape(dense1_out2, [self.batch_size, 7, 7, 64])

		deconv_out1 = self.deconv2(decon1_input, 32, "filter1") 
		dense1_out2 = self.deconv2(deconv_out1, 1, "filter2")

		return dense1_out2

	def compute_l2_loss(self, images):
		input_img = tf.reshape(images, [self.batch_size, 28, 28, 1])
		l2_loss = tf.reduce_mean(tf.pow(input_img - self.output_img, 2))
		return self.sess.run(l2_loss, feed_dict={self.x:images, self.keep_prob:0.75})

	def deconv2(self, batch_input, out_channels, filter_name):
		batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
		filter = tf.get_variable(filter_name, [5, 5, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
		# [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
		#     => [batch, out_height, out_width, out_channels]
		conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
		return conv


def main():
	tf.set_random_seed(1)
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	sess = tf.Session()
	cnn = AutoEncoder(
		sess,
		data=mnist)

	cnn.train()



if __name__ == '__main__':
	main()

# TODO: need test number freeout from self.batch_size
