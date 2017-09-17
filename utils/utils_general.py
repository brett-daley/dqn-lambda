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
	def __init__(self, capacity, minibatch_size):
		self.capacity = capacity              # Maximum number of experiences to store
		self.minibatch_size = minibatch_size  # Number of traces to sample at once

		self.reset()

	def reset(self):
		self.memory = None      # Holds the experiences
		self.size = 0           # Number of experiences currently in the memory
		self.insertion_ptr = 0  # Index of the next experience to be overwritten

	def add(self, obs, action, reward, next_obs, terminal):
		if self.memory is None:
			self._initialize_memory(obs.shape)

		self.memory[self.insertion_ptr] = np.array([obs, action, reward, next_obs, terminal])

		if self.size < self.capacity:
			self.size += 1

		# Increment the insertion pointer, wrapping around if the end is reached
		self.insertion_ptr = (self.insertion_ptr + 1) % self.capacity

	def _initialize_memory(self, obs_shape):
		# Compute a blank/zero-filled agent experience
		# for zero-padding data when training RNN traces of varying length
		self.null_experience = [
			np.zeros(obs_shape), # Observation
			0,                   # Action
			0.0,                 # Reward
			np.zeros(obs_shape), # Next observation
			False                # Terminal signal
		]

		# Pre-allocate the replay memory; this has better performance and we'll also know immediately if it's too big
		self.memory = np.array([self.null_experience] * self.capacity)

	def sample_traces(self, tracelength):
		sampled_traces = []
		truetracelengths = []

		for _ in xrange(self.minibatch_size):
			# Randomly sample a starting index for this trace
			start_idx = np.random.randint(1-tracelength, self.size)
			end_idx = start_idx + tracelength

			n_extra_experiences = 0

			if start_idx < 0:
				# Starting index is before first experience in memory
				# Since tensorflow sequence_length RNN doesn't support front-padding (yet),
				# keep the relevant experiences at front, and set the suffix padding appropriately
				n_extra_experiences = -start_idx  # Points with negative indices are all extra, this just is a quick way to count them
				start_idx = 0                     # Ignore points with negative index

			if end_idx > self.size:
				# Ending index is after final experience in memory
				n_extra_experiences += end_idx - self.size
				end_idx = self.size  # Ignore points beyond memory size

			# Add zero-padded trace to minibatch
			trace = self.memory[start_idx:end_idx]
			zero_padding = [self.null_experience]*n_extra_experiences
			sampled_traces.extend(trace)
			sampled_traces.extend(zero_padding)

			# Record the number of non-null experiences in this trace
			truetracelengths.append(end_idx-start_idx)

		minibatch = np.reshape(sampled_traces, [self.minibatch_size*tracelength, -1])

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
