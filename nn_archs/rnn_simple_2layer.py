import tensorflow as tf 
import numpy as np
from Network import Network
import tensorflow.contrib.slim as slim

class rnn_simple_2layer(Network):
	def __init__(self, cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_target_net = False, src_network = None):
		super(self.__class__, self).__init__(cfg_parser, sess, scope, trainable = not is_target_net)
		self.is_rnn = True
		self.dim_state_input = dim_state_input
		self.n_actions = n_actions
		self.h_size = 64
		self.var_reuse = var_reuse

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
		self.actionInput = tf.placeholder("float",shape=[None,self.n_actions])
		self.yInput = tf.placeholder("float", shape=[None]) # Q-learning target (defined in DQN paper)

		# Per DQN paper training algorithm, update SGD using ONLY the Q-Value for the specified replay memory action (basically zeros out all the other Q values due to 1-hot-coded action and element-wise tf.mul)
		self.Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis = 1)
		
		# Mask out zero-paddings for varying length RNN sequences
		lower_triangular_ones = tf.constant(np.tril(np.ones([32,32])),dtype=tf.float32) # 32 is the maximum tracelength. Temp hack since numpy needs an int, so self.tracelength doesn't work here since tf type.
		seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, self.truetracelengths - 1),\
						 [0, 0], [self.batch_size, self.tracelength])
		self.seqlen_mask = tf.reshape(seqlen_mask, [-1])

		# TODO 2017 print out this seqlen_mask and operations tmrw to make sure it's doing right thing


		# Mask out first half of tracelength, since they have unstable hidden state values (per Lample and Catlot 2016 paper)
		# self.masked = tf.ones([self.batch_size,self.tracelength/2]) # TODO 2017 try enabling this zeros masking again
		# self.unmasked = tf.ones([self.batch_size,self.tracelength/2])
		# self.mask = tf.concat(1,[self.masked,self.unmasked])
		# self.mask = tf.reshape(self.mask,[-1])

		# Hysteretic Q-learning (set alpha = 1 for decentralized Q-learning)
		self.td_err = self.yInput - self.Q_Action
		
		if self.hysteretic_q_learning:
			self.td_err = tf.maximum(self.hql_alpha*self.td_err,self.td_err)
		self.cost = tf.reduce_mean(tf.square(self.td_err)*self.seqlen_mask) # self.mask was here for 2016 stuff above

		# self.term1 = self.yInput - self.Q_Action
		# self.term2 =  tf.square(self.yInput - self.Q_Action)* self.mask
		# self.term3 = self.cost

		# self.trainStep = tf.train.RMSPropOptimizer(learning_rate = 0.001, decay = 0.99, momentum = 0.0, epsilon = 1e-6).minimize(self.cost)
		self.trainStep = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
		# self.trainStepSlow = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		

	def run_train_step(self, y_batch, a_batch, s_batch):
		self.trainStep.run(feed_dict={	self.yInput : y_batch,
										self.actionInput : a_batch,
										self.stateInput : s_batch })

	def create_Q_network(self):
		with tf.variable_scope(self.scope, reuse = self.var_reuse):
			# INPUT stateInput, I think each batch is technically batchsize*tracelength, so the input is a vstacked [batchsize*tracelength, input_dim] in dimension
			self.stateInput = tf.placeholder(tf.float32, [None, self.dim_state_input], name = "stateInput")
			self.tracelength = tf.placeholder(dtype=tf.int32, name = "tracelength")
			self.batch_size = tf.placeholder(dtype=tf.int32, name = "batch_size")
			self.truetracelengths = tf.placeholder(tf.int32, [None], name = "truetracelengths") # traces are varying length, this [batch_size] vector specifies the true length for each trace in the batch

			#tf.orthogonal_initializer()
			#tf.contrib.layers.xavier_initializer
			#tf.truncated_normal_initializer(stddev=0.01)
			weights_initializer = tf.contrib.layers.xavier_initializer()#xavier_initializer()#tf.truncated_normal_initializer(stddev=0.01)

			n_hidden = 32
			net = slim.fully_connected(inputs = self.stateInput, num_outputs = n_hidden, 
									   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
									   biases_initializer = tf.constant_initializer(value = 0.01),
									   trainable = self.trainable)
									   # normalizer_fn = slim.batch_norm,
			# net = slim.dropout(net, 0.5)

			n_hidden = self.h_size
			net = slim.fully_connected(inputs = net, num_outputs = n_hidden, 
									   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
									   biases_initializer = tf.constant_initializer(value = 0.01),
									   trainable = self.trainable)
			# net = slim.dropout(net, 0.5)

			# Output from lower MLP must be shaped as [batchsize*tracelength, self.h_size] for input into RNN

			# the way they've set up their convnet is that the output of each bathsize*tracelength conv4 is just a 512-vector (1 bit for each filter), so they can flatten it out to this shape using self.h_size = 512
			# we should probably make ours like a 16 or 32 long vector, and pass that into the LSTM
			# Reshape fc2 from [batch_size*tracelength, self.h_size] to [batch_size, tracelength, self.h_size]
			net = tf.reshape(net, [self.batch_size, self.tracelength, self.h_size])

			rnn_cell = tf.contrib.rnn.LSTMCell(num_units = self.h_size, state_is_tuple=True)
			self.rnn_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
			# Rnn output is [batch_size, tracelength, self.h_size]
			self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(inputs=net, cell=rnn_cell, dtype=tf.float32, 
																initial_state=self.rnn_state_in, sequence_length = self.truetracelengths)
			
			# Reshape RNN output to [batchsize*tracelength, self.h_size]. 
			# This keeps all the points in a single trace together. E.g., rnn_output[0:3,:] are all the outputs for the first 4-timestep long trace, rnn_output[3:7,:] for the second, and so on.
			self.rnn_output = tf.reshape(self.rnn_output,shape=[-1,self.h_size])
			# print "rnn_output", self.rnn_output.get_shape()
			
			# Final MLP layer to map everything to real value scores
			n_hidden = 32
			net = slim.fully_connected(inputs = self.rnn_output, num_outputs = n_hidden, 
									   activation_fn = tf.nn.relu, weights_initializer = weights_initializer,
									   biases_initializer = tf.constant_initializer(value = 0.01),
									   trainable = self.trainable)

			n_hidden = self.n_actions
			# self.QValue = slim.fully_connected(inputs = self.rnn_output, num_outputs = n_hidden, 
			# 								   activation_fn = None, weights_initializer = weights_initializer,
			# 								   biases_initializer = tf.constant_initializer(value = 0.01),
			# 								   trainable = self.trainable)
			self.QValue = slim.fully_connected(inputs = net, num_outputs = n_hidden, 
											   activation_fn = None, weights_initializer = weights_initializer,
											   biases_initializer = tf.constant_initializer(value = 0.01),
											   trainable = self.trainable)

			# This is a [batchsize*tracelength] long vector of best actions idx for every point in the trace
			self.Qmax = tf.reduce_max(self.QValue, 1)
			self.predict = tf.argmax(self.QValue,1)
