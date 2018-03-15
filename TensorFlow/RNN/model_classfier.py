import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class RNN(object):
	def __init__(self, sess, data, 
		learning_rate=0.001, 
		train_iterators=100000,
		batch_size=128,
		n_inputs=28, # 28 pixels input as a sequence part
		n_steps=28, # 28 parts for sequence
		n_hidden_units=128, # neurons in hidden layer
		n_classes=10): # mnist classes 0~9
		self.sess = sess
		self.data = data
		self.learning_rate = learning_rate
		self.train_iterators = train_iterators
		self.batch_size = batch_size
		self.n_inputs = n_inputs
		self.n_steps = n_steps
		self.n_hidden_units = n_hidden_units
		self.n_classes = n_classes
		self.weight = {'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
					   'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
		self.biases = {'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
					   'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}
		self.build_model()

	def build_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])

		self.rnn_structure()
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))

		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def rnn_structure(self):
		# hiden layer for input to cell 
		#######################################
		# x = (128 batches, 28 steps, 28 inputs)
		# ==> [128 batches*28 steps, 28inputs]
		x = tf.reshape(self.x, [-1, self.n_inputs])

		x_in = tf.matmul(x, self.weight['in']) + self.biases['in']  # TODO what is hidden unit in RNN?
		# ==> [128 batches, 28 steps, 128 hidden]
		x_in = tf.reshape(x_in, [-1, self.n_steps, self.n_hidden_units])

		# cell
		########################################
		try:
			cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
		except:
			cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
		init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(cell, x_in, initial_state=init_state, time_major=False)

		try:
			outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
		except:
			outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
		self.prediction = tf.matmul(outputs[-1], self.weight['out']) + self.biases['out']

	def train(self):
		train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		try:
			init = tf.initialize_all_variables()
		except:
			init = tf.global_variables_initializer()
		self.sess.run(init)

		steps = 0
		while steps * self.batch_size < self.train_iterators:
			batch_x, batch_y = self.data.train.next_batch(self.batch_size)
			batch_x = batch_x.reshape([self.batch_size, self.n_steps, self.n_inputs])
			self.sess.run([train_step], feed_dict={self.x: batch_x, self.y:batch_y})
			if steps%20 == 0: print (self.sess.run(self.accuracy, feed_dict={self.x:batch_x, self.y:batch_y}))
			steps += 1


def main():
	tf.set_random_seed(1)
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	sess = tf.Session()
	cnn = RNN(
		sess,
		data=mnist)

	cnn.train()



if __name__ == '__main__':
	main()


