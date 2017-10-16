import tensorflow as tf
import numpy as np
import os
import random
from rnn_simple_2layer import rnn_simple_2layer
from conv_3layer import conv_3layer


# Helper function for defining a unified NN architecture, to allow use in both AgentGround and externally
def create_specific_nn(cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_target_net=False, src_network=None):
	nn_arch = cfg_parser.get('nn', 'arch')

	# 2 ff + 1 rnn + 1 ff
	if nn_arch == 'rnn_simple_2layer':
		return rnn_simple_2layer(cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_target_net=is_target_net, src_network=src_network)

	# 3 conv + (1 rnn or 1 ff) + 1 ff
	elif nn_arch == 'conv_3layer':
		return conv_3layer(cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_target_net=is_target_net, src_network=src_network)

	else:
		raise ValueError('Unhandled neural net architecture:', nn_arch)


class Data2DTraj:
	def __init__(self):
		self.x_traj = np.array([])
		self.y_mean_traj = np.array([])
		self.y_stdev_traj = np.array([])
		self.y_lower_traj = np.array([])
		self.y_upper_traj = np.array([])

	def appendToTraj(self, x, y_mean, y_stdev):
		self.x_traj = np.append(self.x_traj, x)
		self.y_mean_traj = np.append(self.y_mean_traj, y_mean)
		self.y_stdev_traj = np.append(self.y_stdev_traj, y_stdev)
		self.y_lower_traj = np.append(self.y_lower_traj, y_mean - y_stdev)
		self.y_upper_traj = np.append(self.y_upper_traj, y_mean + y_stdev)

	def saveData(self,data_dir):
		# Save the data (replaces file if already exists)
		np.savetxt(data_dir, (self.x_traj, self.y_mean_traj, self.y_stdev_traj))
