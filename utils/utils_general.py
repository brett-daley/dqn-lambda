import tensorflow as tf
import numpy as np
import os
import copy
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

class TfSaver:
	def __init__(self, sess, data_dir, vars_to_restore, try_to_restore=True):
		# Note: this should only be used for LOADING during distillation phase, and SAVING during task specialization phase
		# Do NOT use this to load during task specialization phase (since each task has its own network, and each  meta files contain all the tasks' networks)
		self.data_dir = data_dir
		self.sess = sess

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

		self.saver = tf.train.Saver(max_to_keep=5, var_list=vars_to_restore)
		checkpoint = tf.train.get_checkpoint_state(self.data_dir)

		# Load checkpoint with latest timestamp (not latest training #! latest timestamp!) in folder if it exists. 
		# Otherwise, just create directory for future saving.
		if checkpoint and checkpoint.model_checkpoint_path and try_to_restore:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			self.pre_trained = True
			print '------- Successfully loaded:', checkpoint.model_checkpoint_path
		else:
			self.pre_trained = False
			print '------- Could not find old network weights for', data_dir, ' Training from scratch (could be distiller though!).' # First run

	def save_sess(self, timestep, save_freq):
		# save network every save_freq iterations
		if timestep % save_freq == 0:
			self.saver.save(self.sess, os.path.join(self.data_dir, 'networks-mtsa'), global_step=timestep)
			print 'Successfully saved tf session'

class ReplayMemory:
	def __init__(self, n_trajs_max, minibatch_size):
		self.minibatch_size = minibatch_size
		self.traj_mem = []
		self.n_trajs_max = n_trajs_max
		self.traj_pad_elem_is_calculated = False

		self.mem_tuple_size = 5 #(o,a,r,o',t)

	def reset(self):
		self.traj_mem = []

	def add(self, sarsa_traj):
		if len(self.traj_mem) >= self.n_trajs_max:
			# Deletes the first trajectory
			self.traj_mem[0:(1+len(self.traj_mem))-self.n_trajs_max] = []

		# Appends new trajectory, consisting of (o, a, r, o', terminal)
		self.traj_mem.append(sarsa_traj)

		# Initialize the padding element based on first added sarsa_traj, for future use
		if not self.traj_pad_elem_is_calculated:
			self.calc_traj_pad_elem()
			self.traj_pad_elem_is_calculated = True

	# Compute a blank/zero-filled multiagent trajectory point (o, a, r, o', terminal)
	# used for zero-padding data when training RNN traces of varying length
	def calc_traj_pad_elem(self):
		# Use first point in first trajectory as a reference
		self.padding_elem = copy.deepcopy(self.traj_mem[0][0])

		# Clear observation
		self.padding_elem[0] = np.zeros_like(self.padding_elem[0])

		# Clear action
		self.padding_elem[1] = 0.0

		# Clear reward
		self.padding_elem[2] = 0.0

		# Clear next observation
		self.padding_elem[3] = np.zeros_like(self.padding_elem[3])

		# Clear terminal signal
		self.padding_elem[4] = False

	# Samples an extended trace from a traj
	def sample_trace(self, tracelength):
		sampled_trajs = random.sample(self.traj_mem, self.minibatch_size)
		sampled_points = []
		truetracelengths = []

		for traj in sampled_trajs:
			i_start = np.random.randint(1-tracelength, len(traj))
			i_end = i_start + tracelength

			num_extra_pts = 0

			# Starting index is before first element in trajectory
			if i_start < 0:
				# Since tensorflow sequence_length RNN doesn't support front-padding (yet), keep the relevant elements at front, and set the suffix padding appropriately
				num_extra_pts = -i_start # points with negative indices are all extra, this just is a quick way to count them
				i_start = 0 # ignores points with negative index

			# Ending index is after final element in trajectory
			if i_end > len(traj):
				num_extra_pts += i_end - len(traj) 
				i_end = len(traj) # ignore points beyond trajectory length

			# Append points and apply zero padding to suffix 
			sampled_points.extend(traj[i_start:i_end])
			sampled_points.extend([self.padding_elem]*num_extra_pts)
			truetracelengths.extend([i_end-i_start])

		minibatch = np.reshape(sampled_points, [self.minibatch_size*tracelength, self.mem_tuple_size])

		return truetracelengths, minibatch

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
