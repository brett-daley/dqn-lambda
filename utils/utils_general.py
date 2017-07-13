import tensorflow as tf
import numpy as np
import os
import copy
import random
from ff_simple_1layer import ff_simple_1layer
from ff_simple_2layer import ff_simple_2layer
from ff_simple_3layer import ff_simple_3layer
from rnn_simple_2layer import rnn_simple_2layer
from rnn_distillation_2layer import rnn_distillation_2layer
from ff_distillation_2layer import ff_distillation_2layer

# Helper function for defining a unified NN architecture, to allow use in both AgentGround and externally
def create_specific_nn(cfg_parser, sess, scope, var_reuse, dim_state_input, n_actions, is_distillation_net = False, is_distiller_companion = False, is_target_net = False, src_network = None):
	if not is_distillation_net:
		nn_name = cfg_parser.get('root','teacher_nn_type')

		# 2 layer ff
		if nn_name == 'ff_simple_2layer':
			return ff_simple_2layer(sess = sess,  scope = scope, dim_state_input = dim_state_input, n_actions = n_actions, is_target_net = is_target_net, src_network = src_network)

		# 3 layer ff
		if nn_name == 'ff_simple_3layer':
			return ff_simple_3layer(sess = sess,  scope = scope, dim_state_input = dim_state_input, n_actions = n_actions, is_target_net = is_target_net, src_network = src_network)

		# 2 ff + 1 rnn + 1 ff
		if nn_name == 'rnn_simple_2layer':
			return rnn_simple_2layer(cfg_parser = cfg_parser, sess = sess, scope = scope, var_reuse = var_reuse, 
									dim_state_input = dim_state_input, n_actions = n_actions, is_target_net = is_target_net, src_network = src_network)
	else:
		nn_name = cfg_parser.get('root','distilled_nn_type')

		if nn_name == 'rnn_distillation_2layer':
			return rnn_distillation_2layer(cfg_parser = cfg_parser, sess = sess, scope = scope, var_reuse = var_reuse, 
										dim_state_input = dim_state_input, n_actions = n_actions, is_target_net = is_target_net, src_network = src_network,
										is_distiller_companion = is_distiller_companion)

class TfSaver:
	def __init__(self, sess, data_dir, vars_to_restore, try_to_restore = True):
		# Note: this should only be used for LOADING during distillation phase, and SAVING during task specialization phase
		# Do NOT use this to load during task specialization phase (since each task has its own network, and each  meta files contain all the tasks' networks)
		self.data_dir = data_dir
		self.sess = sess

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

		self.saver = tf.train.Saver(max_to_keep = 5, var_list = vars_to_restore) 
		checkpoint = tf.train.get_checkpoint_state(self.data_dir)

		# Load checkpoint with latest timestamp (not latest training #! latest timestamp!) in folder if it exists. 
		# Otherwise, just create directory for future saving.
		if checkpoint and checkpoint.model_checkpoint_path and try_to_restore:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			self.pre_trained = True
			print "------- Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			self.pre_trained = False
			print "------- Could not find old network weights for", data_dir, " Training from scratch (could be distiller though!)." # First run

	def save_sess(self, timestep, save_freq):
		# save network every save_freq iterations
		if timestep % save_freq == 0:
			self.saver.save(self.sess, os.path.join(self.data_dir, 'networks-mtsa'), global_step = timestep)
			print "Successfully saved tf session"

class ReplayMemory:
	def __init__(self, n_trajs_max, minibatch_size, is_distillation_mem = False):
		self.minibatch_size = minibatch_size
		self.traj_mem = []
		self.n_trajs_max = n_trajs_max
		self.is_distillation_mem = is_distillation_mem 
		self.is_traj_pad_elem_calculated = False

		if self.is_distillation_mem:
			self.mem_tuple_size = 6 #(o,a,r,o',t,q)
		else:
			self.mem_tuple_size = 5 #(o,a,r,o',t)

	def reset(self):
		self.traj_mem = []

	def add(self, sarsa_traj):
		if len(self.traj_mem) + 1 >= self.n_trajs_max:
			# Deletes the first trajectory
			self.traj_mem[0:(1+len(self.traj_mem))-self.n_trajs_max] = []
		# Appends new multiagent trajectory, consisting of (o_joint, a_joint, r, o_joint', terminal)
		self.traj_mem.append(sarsa_traj)

		# Initialize the padding element based on first added sarsa_traj, for future use
		if not self.is_traj_pad_elem_calculated:
			self.calc_traj_pad_elem()
			self.is_traj_pad_elem_calculated = True

	# Compute a blank/zero-filled multiagent trajectory point (o_joint, a_joint, r, o_joint', terminal)
	# used for zero-padding data when training RNN traces of varying length
	def calc_traj_pad_elem(self):
		# Use first point in first trajectory as a reference
		self.padding_elem = copy.deepcopy(self.traj_mem[0][0])

		# The [0] indexing below ensures that append works in self.sample_trace()

		# Set all cur_obs for all agents to 0 vector
		for i_agt in xrange(0,len(self.padding_elem[0][0])):
			self.padding_elem[0][0][i_agt] = self.padding_elem[0][0][i_agt]*0.

		# Set all cur_action for all agents to 0 vector
		for i_agt in xrange(0,len(self.padding_elem[0][1])):
			self.padding_elem[0][1][i_agt] = self.padding_elem[0][1][i_agt]*0.

		# Set reward to 0
		self.padding_elem[0][2] = 0

		# Set next_obs for all agents to 0 vector
		for i_agt in xrange(0,len(self.padding_elem[0][3])):
			self.padding_elem[0][3][i_agt] = self.padding_elem[0][3][i_agt]*0.

		# Set terminal signal to False
		self.padding_elem[0][4] = False

		if self.is_distillation_mem:
			# Set teacher q_values for all agents to 0 vector
			for i_agt in xrange(0,len(self.padding_elem[0][5])):
				self.padding_elem[0][5][i_agt] = self.padding_elem[0][5][i_agt]*0.

	# Samples an extended trace from a traj
	def sample_trace(self, tracelength):
		sampled_trajs = random.sample(self.traj_mem,self.minibatch_size)
		sampled_points = []
		truetracelengths = []

		for traj in sampled_trajs:
			i_start = np.random.randint(-1*tracelength+1,len(traj))
			i_end = i_start+tracelength
				
			num_extra_pts = 0

			# Starting index is before first element in trajectory
			if i_start < 0:
				# Since tensorflow sequence_length RNN doesn't support front-padding (yet), keep the relevant elements at front, and set the suffix padding appropriately
				num_extra_pts += -1*i_start # points with negative indices are all extra, this just is a quick way to count them
				i_start = 0 # ignores points with negative index

			# Ending index is after final element in trajectory
			if i_end > len(traj):
				num_extra_pts += i_end - len(traj) 
				i_end = len(traj) # ignore points beyond trajectory length
			
			# Append points and apply zero padding to suffix 
			sampled_points.extend(traj[i_start:i_end])
			sampled_points.extend([self.padding_elem]*num_extra_pts)
			truetracelengths.extend([i_end-i_start])

		sampled_points = np.array(sampled_points)
	
		return truetracelengths, np.reshape(sampled_points,[self.minibatch_size*tracelength,self.mem_tuple_size])

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