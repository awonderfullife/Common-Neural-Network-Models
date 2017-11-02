import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

class Perception:
	def __init__(self, sess, data, hiden_units, dropout=0.75, batch_size=128, learning_rate=1e-4):
		self.sess = sess
		self.data = data
		self.batch_size = batch_size
		self.hiden_units = hiden_units
		self.dropout = dropout
		self.learning_rate = learning_rate

		self.build_model()

	def build_model(self):
		self.x = tf.placeholder(tf.float32, [None, 784], name="x")
		self.labels = tf.placeholder(tf.float32, [None, 10], name="labels")

		# attention the init for hidden unit is diff from the init of output unit!
		weight1 = tf.Variable(tf.truncated_normal([784, self.hiden_units], stddev=0.1), name="weight1")
		bias1 = tf.Variable(tf.zeros([self.hiden_units]), name="bias1")
		weight2 = tf.Variable(tf.zeros([self.hiden_units, 10]), name="weight2")
		bias2 = tf.Variable(tf.zeros([10]), name="bias2")

		# attention dropout needs place holder
		self.keep_prob = tf.placeholder(tf.float32)

		# hidden layer 
		hidden_1 = tf.nn.relu(tf.matmul(self.x, weight1) + bias1)
		hidden_1_drop = tf.nn.dropout(hidden_1, self.keep_prob)

		self.y = tf.nn.softmax(tf.matmul(hidden_1_drop, weight2) + bias2)

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

	sr = Perception(
		sess=sess,
		data=mnist,
		hiden_units=300
		)

	sr.train()

if __name__ == '__main__':
	main()