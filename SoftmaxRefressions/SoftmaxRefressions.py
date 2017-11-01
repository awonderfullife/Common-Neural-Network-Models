import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class SoftmaxRefressions:
	def __init__(self, sess, data, batch_size):
		self.sess = sess
		self.batch_size = batch_size
		self.data = data
		self.build_model()

	def build_model(self):
		self.x = tf.placeholder(tf.float32, [None, 784], name="x")
		self.labels = tf.placeholder(tf.float32, [None, 10], name="labels")

		weight = tf.Variable(tf.zeros([784, 10]), name="weight")
		bias = tf.Variable(tf.zeros([10]), name="bias")

		self.y = tf.nn.softmax(tf.matmul(self.x, weight) + bias)

		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.labels)

	def train(self):
		optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		try:
			init = tf.initialize_all_variables()
		except:
			init = tf.global_variables_initializer()
		self.sess.run(init)

		for i in range(10000):
			batch_x, batch_labels = self.data.train.next_batch(self.batch_size)
			self.sess.run(optimizer, feed_dict={self.x:batch_x, self.labels:batch_labels})
			if i%50 == 0:
				print(self.computer_accuracy(self.data.test.images[:1000], self.data.test.labels[:1000]))

	def computer_accuracy(self, images, labels):
		# tf.argmax:Returns the index with the largest value across axes of a tensor
		# tf.zeros:Returns the truth value of (x == y) element-wise
		# tf.cast: Casts a tensor to a new type.
		# tf.reduce_mean:Computes the mean of elements across dimensions of a tensor
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		print self.sess.run(accuracy, feed_dict={self.x:images, self.labels:labels})



def main():
	# 55000 for minist.train, 10000 for minist.test, 5000for minist.validation 
	# struct is : mnist.train.images [55000, 784] && mnist.train.labels [55000, 10]
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	sess = tf.Session()

	sr = SoftmaxRefressions(
		sess=sess,
		data=mnist,
		batch_size=128)

	sr.train()

if __name__ == '__main__':
	main()