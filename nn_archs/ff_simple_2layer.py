import tensorflow as tf 
import numpy as np
from Network import Network

class ff_simple_2layer(Network):
	def __init__(self, sess, scope, dim_state_input, n_actions, is_target_net = False, src_network = None):
		super(self.__class__, self).__init__(sess, scope, trainable = not is_target_net)
		# self.initialized = False
		self.is_rnn = False
		self.dim_state_input = dim_state_input
		self.n_actions = n_actions

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
		
		self.cost = tf.reduce_mean(tf.square(self.yInput - self.Q_Action))
		# self.trainStep = tf.train.RMSPropOptimizer(learning_rate = 0.00025, decay = 0.99, momentum = 0.0, epsilon = 1e-6).minimize(self.cost)
		self.trainStep = tf.train.RMSPropOptimizer(learning_rate = 0.001, decay = 0.99, momentum = 0.0, epsilon = 1e-6).minimize(self.cost)
		

	def run_train_step(self, y_batch, a_batch, s_batch):
		self.trainStep.run(feed_dict={	self.yInput : y_batch,
										self.actionInput : a_batch,
										self.stateInput : s_batch })

	def create_Q_network(self):
		with tf.variable_scope(self.scope):
			self.tracelength = tf.placeholder(dtype=tf.int32) # this is tracelength in each episode. Is always 1 for non-rnn networks.
			self.batch_size = tf.placeholder(dtype=tf.int32) #batch_size, not used in non-rnn networks (just here for consistency)

			# input layer
			self.stateInput = tf.placeholder(tf.float32, [None, self.dim_state_input])

			# hidden layer 1 - fc
			l1_n_hidden = 10
			self.var["W_l1"] = self.weight_variable(shape=[self.dim_state_input, l1_n_hidden])
			self.var["b_l1"] = self.bias_variable([l1_n_hidden])
			fc1 = tf.nn.relu(tf.matmul(self.stateInput, self.var["W_l1"]) + self.var["b_l1"])

			# hidden layer 2 - fc
			l2_n_hidden = self.n_actions
			self.var["W_l2"] = self.weight_variable(shape=[l1_n_hidden, l2_n_hidden])
			self.var["b_l2"] = self.bias_variable([l2_n_hidden])
			self.QValue = tf.matmul(fc1, self.var["W_l2"]) + self.var["b_l2"] # This is QValue for ALL the possible actions

			self.Qmax = tf.reduce_max(self.QValue, 1)
			self.predict = tf.argmax(self.QValue,1)
			# TODO can also add dropout etc., see mnist_softmax.py in 6.867 HW3