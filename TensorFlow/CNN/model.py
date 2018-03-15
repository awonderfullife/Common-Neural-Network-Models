import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

class CNN:
	def __init__(self, sess, data, batch_size=128, learning_rate=1e-4, dropout=0.75):
		self.sess = sess
		self.data = data
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dropout = dropout

		self.build_model()

	# tf.layers.*
	# conv2d(). Constructs a two-dimensional convolutional layer. Takes number of filters, filter kernel size, padding, and activation function as arguments.
	# max_pooling2d(). Constructs a two-dimensional pooling layer using the max-pooling algorithm. Takes pooling filter size and stride as arguments.
	# dense(). Constructs a dense layer. Takes number of neurons and activation function as arguments.
	# Each of these methods accepts a tensor as input and returns a transformed tensor as output
	def build_model(self):
		# input_layer: [None, 784] => [-1, 28, 28, 1]
		self.x = tf.placeholder(tf.float32, [None, 784], name="x")
		self.labels = tf.placeholder(tf.float32, [None, 10], name="labels")
		self.keep_prob = tf.placeholder(tf.float32)
		input_layer_input = tf.reshape(self.x, [-1,28, 28, 1])

		# convolution layer 1 with filters=32, kernel_size=[5,5], padding="same", activation=relu [-1, 28, 28, 1] ==> [-1, 28, 28, 32]
		conv1_out = tf.layers.conv2d(inputs=input_layer_input, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)

		# pooling layer 1 with pool_size=[2,2], strides=2 ...  [-1, 28, 28, 32] ==> [-1, 14, 14, 32]
		pool1_out = tf.layers.max_pooling2d(inputs=conv1_out, pool_size=[2,2], strides=2)

		# convolution layer 2 with filters=64, kernel_size=[5,5], ... [-1, 14, 14, 32] ==> [-1, 14, 14, 64]
		conv2_out = tf.layers.conv2d(inputs=pool1_out, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)

		# pooling layer 2 ...  [-1, 14, 14, 64] ==> [-1, 7, 7, 64]
		pool2_out = tf.layers.max_pooling2d(inputs=conv2_out, pool_size=[2,2], strides=2)

		# reshape [-1, 7, 7, 64] ==> [-1, 7*7*64]
		dense_layer_input = tf.reshape(pool2_out, [-1, 7*7*64])

		# dense layer with units=1024, activation=relu
		dense1_out = tf.layers.dense(inputs=dense_layer_input, units=1024, activation=tf.nn.relu)

		# dropout of dense layer with rate = 0.4
		dense1_drop_out = tf.layers.dropout(inputs=dense1_out, rate=self.keep_prob)

		# logits layer using dense layer with units=10
		self.y = tf.layers.dense(inputs=dense1_drop_out, units=10, name="softmax_tensor")

		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.labels)

	def train(self):
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
		try:
			init = tf.initialize_all_variables()
		except:
			init = tf.global_variables_initializer()
		self.sess.run(init)	

		for i in range(10000):
			batch_x, batch_labels = self.data.train.next_batch(100)
			self.sess.run(optimizer, feed_dict={self.x:batch_x, self.labels:batch_labels, self.keep_prob:self.dropout})
			if i%100 == 0:
				print self.compute_accuracy(self.data.test.images[:1000], self.data.test.labels[:1000])

	def compute_accuracy(self, images, labels):
		# tf.argmax:Returns the index with the largest value across axes of a tensor
		# tf.zeros:Returns the truth value of (x == y) element-wise
		# tf.cast: Casts a tensor to a new type.
		# tf.reduce_mean:Computes the mean of elements across dimensions of a tensor
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		print self.sess.run(accuracy, feed_dict={self.x:images, self.labels:labels, self.keep_prob:1.0})

def main():
	# 55000 for minist.train, 10000 for minist.test, 5000for minist.validation 
	# struct is : mnist.train.images [55000, 784] && mnist.train.labels [55000, 10]
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	sess = tf.Session()

	sr = CNN(
		sess=sess,
		data=mnist,
		)

	sr.train()

if __name__ == '__main__':
	main()






