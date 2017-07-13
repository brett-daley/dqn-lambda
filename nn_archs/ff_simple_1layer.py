import tensorflow as tf 

class ff_simple_1layer:
	def __init__(self, dim_state_input, n_actions):
		self.dim_state_input = dim_state_input
		print n_actions
		self.n_actions = n_actions

		# Init baseline and target nets
		self.stateInput, self.QValue, self.W_1, self.b_1 = self.create_Q_network()
		self.stateInputT, self.QValueT, self.W_1T, self.b_1T = self.create_Q_network()
		self.copy_target_Q_network_op  = [self.W_1T.assign(self.W_1), self.b_1T.assign(self.b_1)]

		# Optimizer method
		self.create_training_method()

	def copy_target_Q_network(self, session):
		session.run(self.copy_target_Q_network_op)

	def create_training_method(self):
		self.actionInput = tf.placeholder("float",shape=[None,self.n_actions])
		self.yInput = tf.placeholder("float", shape=[None]) # target?
		self.Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - self.Q_Action))
		self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)
		# self.trainStep = tf.train.AdamOptimizer(1e-5).minimize(self.cost)

	def run_train_step(self, y_batch, a_batch, s_batch):
		self.trainStep.run(feed_dict={	self.yInput : y_batch,
										self.actionInput : a_batch,
										self.stateInput : s_batch })

	def create_Q_network(self):
		# input layer
		stateInput = tf.placeholder(tf.float32, [None, self.dim_state_input])

		# hidden layer 1 - fc
		l1_n_hidden = 5 # whatever the number of hidden layers might be, this FINAL output must be same size as action space
		W_1 = self.weight_variable([self.dim_state_input, l1_n_hidden])
		b_1 = self.bias_variable([l1_n_hidden])
		QValue = tf.matmul(stateInput, W_1) + b_1

		# TODO can also add dropout etc., see mnist_softmax.py in 6.867 HW3

		return stateInput, QValue, W_1, b_1

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)