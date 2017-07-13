import tensorflow as tf 
import numpy as np
from Network import Network
import tensorflow.contrib.slim as slim

class rnn_distillation_2layer(Network):
	def __init__(self, cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_target_net = False, src_network = None, is_distiller_companion = False):
		super(self.__class__, self).__init__(cfg_parser, sess, scope, trainable = not is_target_net)
		self.is_rnn = True
		self.dim_state_input = dim_state_input
		self.n_actions = n_actions
		self.h_size = 64
		self.var_reuse = var_reuse
		self.is_distiller_companion = is_distiller_companion
		
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

	def copy_target_Q_network(self, session):
		session.run(self.copy_target_Q_network_op)

	def create_training_method(self):
		# Distillation Q_value target vector. This is a [minibatch_size*tracelength, n_actions] matrix, with some of the rows all-zero due to variable tracelengths for RNN.
		self.Q_target = tf.placeholder("float", shape=[None,self.n_actions]) # distillation Q_value target vector

		# Mask out zero-paddings for varying length RNN sequences. 
		# 1. Make a triangular matrix (each row is a potential mask)
		lower_triangular_ones = tf.constant(np.tril(np.ones([32,32])),dtype=tf.float32) # 32 is the maximum tracelength. Temp hack since numpy needs an int, so self.tracelength doesn't work here since tf type.
		# 2. Sample the proper mask rows. But note that each row also has a huge number of columns. Thus, chop the extra columns off by slicing.
		seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, self.truetracelengths - 1),\
						 [0, 0], [self.batch_size, self.tracelength])
		# 3. Reshape the mask to be a column vector
		self.seqlen_mask_distill = tf.reshape(seqlen_mask, [-1,1])
		# 4. Want to mask out all the actions for that trace point, so tile it :)
		self.seqlen_mask_distill = tf.tile(self.seqlen_mask_distill, [1, self.n_actions])

		# MSE Q-loss
		if self.is_distiller_companion:
			self.cost = tf.reduce_mean(tf.square(self.Q_target - self.QValue)*self.seqlen_mask_distill)
			self.trainStepDistill = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

			# Also has Q-learning components
			self.actionInput = tf.placeholder("float",shape=[None,self.n_actions])
			self.yInput = tf.placeholder("float", shape=[None])
			self.Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis = 1)			
			self.seqlen_mask_qlearn = tf.reshape(seqlen_mask, [-1]) # Q learning mask (slightly diff than distillation mask)
			self.td_err = self.yInput - self.Q_Action
			
			# self.hql_alpha = 1.
			if self.hysteretic_q_learning:
				self.td_err = tf.maximum(self.hql_alpha*self.td_err,self.td_err)
			self.cost_qlearn = tf.reduce_mean(tf.square(self.td_err)*self.seqlen_mask_qlearn) # self.mask was here for 2016 stuff above
			self.trainStep = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost_qlearn)
		else:
			# KL divergence loss w/ softmax temperature. Note: the entropy term in KL div isn't a function of Qvalue so won't affect optimization, and has been removed.
			T = 0.01 # 0.01 is best softmax temperature
			self.cross_entropy = -tf.nn.softmax(self.Q_target/T)*tf.log(tf.nn.softmax(self.QValue))
			self.kl_divergence = tf.reduce_sum(self.cross_entropy*self.seqlen_mask_distill, axis = 1)
			self.cost = tf.reduce_mean(self.kl_divergence)
			self.trainStepDistill = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

	def create_Q_network(self):
		with tf.variable_scope(self.scope, reuse = self.var_reuse):
			# INPUT stateInput, I think each batch is technically batchsize*tracelength, so the input is a vstacked [batchsize*tracelength, input_dim] in dimension
			self.stateInput = tf.placeholder(tf.float32, [None, self.dim_state_input], name = "stateInput")
			self.tracelength = tf.placeholder(dtype=tf.int32, name = "tracelength")
			self.batch_size = tf.placeholder(dtype=tf.int32, name = "batch_size")
			self.truetracelengths = tf.placeholder(tf.int32, [None], name = "truetracelengths") # traces are varying length, this [batch_size] vector specifies the true length for each trace in the batch

			if self.is_distiller_companion:
				weights_regularizer=None#slim.l2_regularizer(0.05)
			else:
				weights_regularizer = None#slim.l2_regularizer(0.01)

			#tf.orthogonal_initializer()
			#tf.contrib.layers.xavier_initializer
			#tf.truncated_normal_initializer(stddev=0.01)
			weights_initializer = tf.contrib.layers.xavier_initializer()#xavier_initializer()#tf.truncated_normal_initializer(stddev=0.01)

			n_hidden = 64
			net = slim.fully_connected(inputs = self.stateInput, num_outputs = n_hidden, 
									   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
									   biases_initializer = tf.constant_initializer(value = 0.01),
									   trainable = self.trainable, weights_regularizer = weights_regularizer)
									   # normalizer_fn = slim.batch_norm,
			# net = slim.dropout(net, 0.5)

			n_hidden = self.h_size
			net = slim.fully_connected(inputs = net, num_outputs = n_hidden, 
									   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
									   biases_initializer = tf.constant_initializer(value = 0.01),
									   trainable = self.trainable, weights_regularizer = weights_regularizer)
			# net = slim.dropout(net, 0.5)

			# Output from lower MLP must be shaped as [batchsize*tracelength, self.h_size] for input into RNN

			# the way they've set up their convnet is that the output of each bathsize*tracelength conv4 is just a 512-vector (1 bit for each filter), so they can flatten it out to this shape using self.h_size = 512
			# we should probably make ours like a 16 or 32 long vector, and pass that into the LSTM
			# Reshape fc2 from [batch_size*tracelength, self.h_size] to [batch_size, tracelength, self.h_size]
			net = tf.reshape(net, [self.batch_size, self.tracelength, self.h_size])

			rnn_cell = tf.contrib.rnn.LSTMCell(num_units = self.h_size, state_is_tuple=True)
			self.rnn_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
			# Rnn output is [batch_size, tracelength, self.h_size]
			self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(inputs=net, cell=rnn_cell, dtype=tf.float32, initial_state=self.rnn_state_in, sequence_length = self.truetracelengths)
			
			# Reshape RNN output to [batchsize*tracelength, self.h_size]. 
			# This keeps all the points in a single trace together. E.g., rnn_output[0:3,:] are all the outputs for the first 4-timestep long trace, rnn_output[3:7,:] for the second, and so on.
			self.rnn_output = tf.reshape(self.rnn_output,shape=[-1,self.h_size])
			# print "rnn_output", self.rnn_output.get_shape()
			
			# Final MLP layer to map everything to real value scores
			# n_hidden = self.n_actions
			# self.QValue = slim.fully_connected(inputs = self.rnn_output, num_outputs = n_hidden, 
			# 								   activation_fn = None, weights_initializer = weights_initializer,
			# 								   biases_initializer = tf.constant_initializer(value = 0.01),
			# 								   trainable = self.trainable)

			n_hidden = 64
			net = slim.fully_connected(inputs = self.rnn_output, num_outputs = n_hidden, 
									   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
									   biases_initializer = tf.constant_initializer(value = 0.01),
									   trainable = self.trainable, weights_regularizer = weights_regularizer)

			# n_hidden = 32
			# net = slim.fully_connected(inputs = net, num_outputs = n_hidden, 
			# 						   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
			# 						   biases_initializer = tf.constant_initializer(value = 0.01),
			# 						   trainable = self.trainable, weights_regularizer = weights_regularizer)

			n_hidden = self.n_actions
			self.QValue = slim.fully_connected(inputs = net, num_outputs = n_hidden, 
											   activation_fn = None, weights_initializer = weights_initializer,
											   biases_initializer = tf.constant_initializer(value = 0.01),
											   trainable = self.trainable, weights_regularizer = weights_regularizer)

			# This is a [batchsize*tracelength] long vector of best actions idx for every point in the trace
			self.Qmax = tf.reduce_max(self.QValue, 1)
			self.predict = tf.argmax(self.QValue,1)