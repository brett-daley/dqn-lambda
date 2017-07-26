import tensorflow as tf
import numpy as np
from Network import Network
import tensorflow.contrib.layers as layers


class conv_3layer(Network):
	def __init__(self, cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_rnn=False, is_target_net=False, src_network=None):
		super(self.__class__, self).__init__(cfg_parser, sess, scope, trainable=(not is_target_net))
		self.is_rnn = cfg_parser.getboolean('root', 'make_recurrent')
		self.dim_state_input = dim_state_input
		self.n_actions = n_actions
		self.var_reuse = var_reuse
		self.h_size = 512

		# Init baseline and target nets
		self.create_Q_network()

		# Copy operation for target network
		if is_target_net:
			assert src_network is not None
			self.create_copy_op(src_network)
			self.initialized = False
		else:
			# Optimizer method only for prediction net
			self.create_training_method()

	def create_training_method(self):
		self.actionInput = tf.placeholder('float', shape=[None,self.n_actions])
		self.yInput = tf.placeholder('float', shape=[None]) # Q-learning target (defined in DQN paper)

		# Per DQN paper training algorithm, update SGD using ONLY the Q-Value for the specified replay memory action (basically zeros out all the other Q values due to 1-hot-coded action and element-wise tf.mul)
		self.Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis=1)

		# Mask out zero-paddings for varying length RNN sequences
		lower_triangular_ones = tf.constant(np.tril(np.ones([32, 32])), dtype=tf.float32) # 32 is the maximum tracelength. Temp hack since numpy needs an int, so self.tracelength doesn't work here since tf type.
		seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, self.truetracelengths - 1), [0, 0], [self.batch_size, self.tracelength])
		self.seqlen_mask = tf.reshape(seqlen_mask, [-1])

		# Hysteretic Q-learning (set alpha = 1 for decentralized Q-learning)
		self.td_err = self.yInput - self.Q_Action

		if self.hysteretic_q_learning:
			self.td_err = tf.maximum(self.hql_alpha * self.td_err, self.td_err)

		self.cost = tf.reduce_mean(tf.square(self.td_err) * self.seqlen_mask)

		self.trainStep = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

	def run_train_step(self, y_batch, a_batch, s_batch):
		self.trainStep.run(feed_dict={self.yInput: y_batch, self.actionInput: a_batch, self.stateInput: s_batch})

	def create_Q_network(self):
		with tf.variable_scope(self.scope, reuse=self.var_reuse):
			# Input shape is a [batchsize*tracelength, dim_state_input]
			self.stateInput = tf.placeholder(tf.float32, [None] + list(self.dim_state_input), name='stateInput')
			self.tracelength = tf.placeholder(dtype=tf.int32, name='tracelength')
			self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
			self.truetracelengths = tf.placeholder(tf.int32, [None], name='truetracelengths') # traces are varying length, this [batch_size] vector specifies the true length for each trace in the batch

			# These convolutional layers are described in the DQN paper
			net = layers.convolution2d(self.stateInput, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
			net = layers.convolution2d(net, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
			net = layers.convolution2d(net, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

			# Number of features in the output of the conv net
			n_features = np.prod(net.get_shape().as_list()[1:])

			if self.is_rnn:
				# Reshape to [batch_size, tracelength, n_features] for the recurrent layer
				net = tf.reshape(net, [self.batch_size, self.tracelength, n_features])

				# Create the RNN cell
				rnn_cell = tf.contrib.rnn.LSTMCell(num_units=self.h_size, state_is_tuple=True, activation=tf.nn.relu)
				self.rnn_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

				# RNN output shape is [batch_size, tracelength, h_size]
				self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(inputs=net, cell=rnn_cell, dtype=tf.float32, initial_state=self.rnn_state_in, sequence_length=self.truetracelengths)

				# Reshape RNN output to [batchsize*tracelength, h_size]
				# This keeps all the points in a single trace together; e.g., rnn_output[0:3,:] are all the outputs for the first 4-timestep long trace, rnn_output[3:7,:] for the second, and so on
				self.rnn_output = tf.reshape(self.rnn_output, shape=[-1, self.h_size])
				net = self.rnn_output
			else:
				# Reshape to [batch_size*tracelength, n_features] for the FC layer
				net = tf.reshape(net, [self.batch_size*self.tracelength, n_features])
				net = layers.fully_connected(net, num_outputs=self.h_size, activation_fn=tf.nn.relu)

			# Final linear FC layer to map everything to real value scores
			self.QValue = layers.fully_connected(net, num_outputs=self.n_actions, activation_fn=None)

			# This is a [batchsize*tracelength] long vector of best actions idx for every point in the trace
			self.Qmax = tf.reduce_max(self.QValue, 1)
			self.predict = tf.argmax(self.QValue, 1)
