import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
"""
Some links you may interested in:
    https://www.cnblogs.com/smuxiaolei/p/8647207.html 
    https://blog.csdn.net/xiewenbo/article/details/79452843 
    https://blog.csdn.net/u014277388/article/details/78791188 
"""
class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]),name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """

        """
        hidden_dim so-called cell output size
        self.x: [batch_size, max_time, ... ]
        output : [batch_size, max_time, cell.output_size]
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            print(np.asarray(train_x).shape)
            print(np.asarray(train_y).shape)
            for i in range(5000):
                mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 1000 == 0:
                    print(i, mse)
                    save_path = self.saver.save(sess, './model/model.ckpt')
                    print('Model saved to {}'.format(save_path))

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model/model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            print(output)

if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=2, seq_size=4, hidden_dim=10)
    train_x = [[[1,2], [2,3], [5,3], [6,2]], [[5,1], [7,2], [7,3], [8,1]], [[3,1], [4,2], [5,3], [7,2]]]
    train_y = [[3, 8, 13, 16], [6, 15, 19, 19], [4, 10, 14, 17]]
    predictor.train(train_x, train_y)
    test_x = [[[1,1], [2,1], [3,3], [4,2]], [[4,1], [5,2], [6,3], [7,3]]]
    predictor.test(test_x)


