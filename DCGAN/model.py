import __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np 
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))

class DCGAN(oobject):
	def __init__(self, sess, input_height=108, input_width=108, crop=True,
		batch_size=64, sample_num=64, output_height=64, output_width=64,
		z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024,
		c_dim=3, dataset_name='default', input_fname_pattern='*.jpg',
		checkpoint_dir=None, sample_dir=None):
		"""
		Args:
		  sess: TensorFlow session
		  batch_size: The size of batch. Should be specified before training.
		  z_dim: (optional) Dimension of dim for Z. [100]
		  gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
		  df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
		  gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
		  dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
		  c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""
		self.sess = sess
		self.crop = crop

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		# batch_normalization: deal with poor initialization helps gradint flow
		self.d_bn1 = batch_norm(name='d_bn1')  #TODO 
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')
		self.g_bn3 = batch_norm(name='g_bn3')

		self.dataset_name = dataset_name
		self.input_fname_pattern = input_fname_pattern
		self.checkpoint_dir = checkpoint_dir

		self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
		imreadImg = imread(self.data[0])    #TODO
		if len(imreadImg.shape) >= 3:
			self.c_dim = imread(self.data[0]).shape[-1]
		else:
			self.c_dim = 1

		self.grayscale = (self.c_dim == 1)
		self.build_model()

	def build_model(self):
		self.y = None

		if self.crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_width, self.c_dim]

		self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

		inputs = self.inputs

		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = histogram_summary("z", self.z)  #TODO

		self.G = self.generator(self.z, self.y) 
		self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
		self.sampler = self.sampler(self.z, self.y)
		self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

		self.d_sum = histogram_summary("d", self.D)
		self.d__sum = histogram_summary("d_", self.D_)
		self.G_sum = image_summary("G", self.G) #TODO

		def sigmoid_cross_entropy_with_logits(x, y):
			try:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
			except:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, target=y)

		self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
		self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real) #TODO
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_fake_sum + self.d_loss_real_sum

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

	def train(self, config):
		d_optim = tf.train.AdamOptimizer(config.learning_rate, betal=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum]) # TODO
		self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = SummaryWriter("./logs", self.sess.graph) # TODO

		sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

		sample_files = self.data[0:self.sample_num]
		samples = [
			get_image(sample_file,   #TODO
					input_height=self.input_height,
					input_width=self.input_width,
					resize_height=self.output_height,
					resize_width=self.output_width,
					crop=self.crop,
					grayscale=self.grayscale) for sample_file in sample_files]
		if self.grayscale:
			sample_inputs = np.array(sample).astype(np.float32)[:,:,:,None]
		else:
			sample_inputs = np.array(sample).astype(np.float32)

		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print("Load success!")
		else:
			print("Load Failed")

		for epoch in xrange(config.epoch):
			self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
			batch_idxs = min(len(self.data), config.train_size) // config.batch_size

			for idx in xrange(0, batch_idxs):
				batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
				batch = [
					get_image(batch_file,
							input_height=self.input_height,
							input_width=self.input_width,
							resize_height=self.output_height,
							resize_width=self.output_width,
							crop=self.crop,
							grayscale=self.grayscale) for batch_file in batch_files]
				if self.grayscale:
					batch_images = np.array(batch).astype(np.float32)[:,:,:,None]
				else:
					batch_images = np.array(batch).astype(np.float32)

				batch_z = np.random.uniform(-1,1,[config.batch_size, self.z_dim]).astype(np.float32)

				# update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_images, self.z:batch_z})
				self.writer.add_summary(summary_str, counter)

				# update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z:batch_z})
				self.write.add_summary(summary_str, counter)

				# Run g_optim twice to make sure that d_loss not go to zero
				_, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z:batch_z})
				self.write.add_summary(summary_str, counter)			

				errD_fake = self.d_loss_fake.eval({self.z:batch_z})
				errD_real = self.d_loss_real.eval({self.inputs: batch_images})
				errG = self.g_loss.eval({self.z: batch_z})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
				  % (epoch, idx, batch_idxs,
					time.time() - start_time, errD_fake+errD_real, errG))

				if np.mod(counter, 100) == 1:
					try:
						samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
							feed_dict={self.z:sample_z, self.inputs:sample_inputs})
						save_images(samples, image_manifold_size(samples.shape[0]),
							'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
					except:
						print("one pic error!...")

				if np.mod(counter, 500) == 2:
					self.save(config.checkpoint_dir, counter)
		


