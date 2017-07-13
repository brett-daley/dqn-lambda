import tensorflow as tf 
import numpy as np

class ff_distillation_simple_3layer:
	def __init__(self, dim_state_input, n_actions):
		self.dim_state_input = dim_state_input
		self.n_actions = n_actions
		self.is_rnn = False
		
		# Init baseline and target nets
		self.stateInput, self.QValue, self.W_l1, self.b_l1, self.W_l2, self.b_l2, self.W_l3, self.b_l3 = self.create_Q_network()

		# Optimizer method
		self.create_training_method()

	def copy_target_Q_network(self, session):
		session.run(self.copy_target_Q_network_op)

	def create_training_method(self):
		# self.actionInput = tf.placeholder("float",shape=[None,self.n_actions])
		self.Q_target = tf.placeholder("float", shape=[None,self.n_actions]) # distillation Q_value target vector

		# Per DQN paper training algorithm, update SGD using ONLY the Q-Value for the specified replay memory action (basically zeros out all the other Q values due to 1-hot-coded action and element-wise tf.mul)
		# self.Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), axis = 1)
		
		# MSE Q-loss
		# self.cost = tf.reduce_mean(tf.square(self.Q_target - self.QValue))
		# self.trainStep = tf.train.RMSPropOptimizer(learning_rate = 0.00025, decay = 0.99, momentum = 0.0, epsilon = 1e-6).minimize(self.cost)

		# KL divergence loss w/ softmax temperature
		# Note: the entropy term in KL div isn't a function of Qvalue so won't affect optimization, and has been removed.
		T = 0.1 #0.1 is best # softmax temperature
		self.cross_entropy = -tf.nn.softmax(self.Q_target/T)*tf.log(tf.nn.softmax(self.QValue))
		# self.entropy = -tf.nn.softmax(self.Q_target/T)*tf.log(tf.nn.softmax(self.Q_target/T)+0.00001) 
		# self.kl_divergence = tf.reduce_sum(self.cross_entropy - self.entropy, axis = 1)
		self.kl_divergence = tf.reduce_sum(self.cross_entropy, axis = 1)
		self.cost = tf.reduce_mean(self.kl_divergence)
		self.trainStep = tf.train.RMSPropOptimizer(learning_rate = 1e-4, decay = 0.99, momentum = 0.0, epsilon = 1e-6).minimize(self.cost)

	def run_train_step(self, q_batch, s_batch):
		self.trainStep.run(feed_dict={	self.Q_target : q_batch,
									 	self.stateInput : s_batch })
										# self.actionInput : a_batch,

	def create_Q_network(self):
		# input layer
		stateInput = tf.placeholder(tf.float32, [None, self.dim_state_input])

		# hidden layer 1 - fc
		l1_n_hidden = 10
		W_l1 = self.weight_variable([self.dim_state_input, l1_n_hidden])
		b_l1 = self.bias_variable([l1_n_hidden])
		fc1 = tf.nn.relu(tf.matmul(stateInput, W_l1) + b_l1)

		# hidden layer 2 - fc
		l2_n_hidden = 10
		W_l2 = self.weight_variable([l1_n_hidden, l2_n_hidden])
		b_l2 = self.bias_variable([l2_n_hidden])
		fc2 = tf.nn.relu(tf.matmul(fc1, W_l2) + b_l2)

		# hidden layer 2 - fc
		l3_n_hidden = self.n_actions
		W_l3 = self.weight_variable([l2_n_hidden, l3_n_hidden])
		b_l3 = self.bias_variable([l3_n_hidden])
		QValue = tf.matmul(fc2, W_l3) + b_l3 # This is QValue for ALL the possible actions


		return stateInput, QValue, W_l1, b_l1, W_l2, b_l2, W_l3, b_l3

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.01)#stddev = np.sqrt(2./shape[0]))
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)