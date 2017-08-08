import sys
import numpy as np
import random
import os
from LineplotDynamic import LineplotDynamic
from utils_general import ReplayMemory


class DRQN:
	def __init__(self, cfg_parser, n_actions, sess, agt):
		self.cfg_parser = cfg_parser
		self.sess = sess

		# Initial phase where no training occurs. Allows population of replay memory before training.
		self.n_pretrain_steps = int(self.cfg_parser.get('env', 'n_pretrain_steps'))
		# Random policy exploration phase - frames over which to anneal epsilon
		self.n_explore_steps = int(self.cfg_parser.get('env', 'n_explore_steps'))
		self.double_q_learning = self.cfg_parser.getboolean('dqn', 'double_q_learning')
		# Size of replay memory transition minibatch in each training phase
		self.minibatch_size = int(self.cfg_parser.get('dqn', 'minibatch_size'))

		self.discount = float(self.cfg_parser.get('dqn', 'discount'))
		self.epsilon_init = float(self.cfg_parser.get('dqn', 'epsilon_init'))
		self.epsilon_final = float(self.cfg_parser.get('dqn', 'epsilon_final'))
		self.target_q_update_freq = int(self.cfg_parser.get('dqn', 'target_q_update_freq'))
		self.epsilon_test_time = float(self.cfg_parser.get('dqn', 'epsilon_test_time'))

		# init some parameters
		self.epsilon = self.epsilon_init
		self.n_actions = n_actions

		self.agt = agt
		self.tracelength = int(self.cfg_parser.get('nn', 'agent_history_length')) if self.agt.nn.is_rnn else 1

		# init replay memory (n_trajs_max is the number of trajectories! Each trajectory has many samples within it!)
		self.n_trajs_max = int(self.cfg_parser.get('dqn', 'replay_memory_size'))
		self.replay_memory = ReplayMemory(n_trajs_max=self.n_trajs_max, minibatch_size=self.minibatch_size)

		# Init plotting
		self.plot_qvalue_dqn = LineplotDynamic('Training Epoch', 'Q', '')
		self.plot_init_qvalue_dqn = LineplotDynamic('Training Epoch', 'Value anticipated', '')
		self.plot_value_dqn = LineplotDynamic('Training Epoch', 'Value actual', '')

	def reset_agts_rnn_states(self):
		self.agt.reset_rnn_state()

	def init_agts_nnTs(self):
		self.agt.init_nnT()

	def update_Q_plot(self, timestep, hl_name='q_plot', label=None, s_batch_prespecified=None):
		agt = self.agt

		if s_batch_prespecified is None:
			# Since tracelength of 1 used, this ensures that truetracelengths = [1,....,1] and no masking required below
			_, minibatch = self.replay_memory.sample_trace(tracelength=1)
			s_batch = np.array([sample[0] for sample in minibatch])
			minibatch_size_plot = self.minibatch_size
		else:
			s_batch = s_batch_prespecified
			minibatch_size_plot = len(s_batch_prespecified)

		x = timestep
		y = agt.nn.Qmax.eval(feed_dict={agt.nn.stateInput: s_batch,
										agt.nn.tracelength: 1,
										agt.nn.truetracelengths: [1]*minibatch_size_plot,
										agt.nn.batch_size: minibatch_size_plot})

		y_mean = np.mean(y)
		y_stdev = np.std(y)

		if s_batch_prespecified is None:
			self.plot_qvalue_dqn.update(hl_name=hl_name, label=label, x_new=x, y_new=y_mean, y_stdev_new=y_stdev)
		else:
			self.plot_init_qvalue_dqn.update(hl_name=hl_name, label=label, x_new=x, y_new=y_mean, y_stdev_new=y_stdev)

		return x, y_mean, y_stdev

	def get_processed_minibatch(self):
		truetracelengths, minibatch = self.replay_memory.sample_trace(self.tracelength)

		s_batch = np.array([sample[0] for sample in minibatch])
		a_batch = minibatch[:,1]
		r_batch = minibatch[:,2]
		s_next_batch = np.array([sample[3] for sample in minibatch])
		non_terminal_multiplier = 1 - minibatch[:,4]

		return s_batch, a_batch, r_batch, s_next_batch, non_terminal_multiplier, truetracelengths

	def train_Q_network(self, timestep):
		if (timestep > self.n_pretrain_steps):
			# Step 1: sample random minibatch of transitions from replay memory
			s_batch, a_batch, r_batch, s_next_batch, non_terminal_multiplier, truetracelengths = self.get_processed_minibatch()

			agt = self.agt

			# Calculate DRQN target
			feed_dict = {agt.nnT.stateInput: s_next_batch,
						 agt.nnT.tracelength: self.tracelength,
						 agt.nnT.truetracelengths: truetracelengths,
						 agt.nnT.batch_size: self.minibatch_size}

			if agt.nnT.is_rnn:
				rnn_state_train = (np.zeros([self.minibatch_size, agt.nn.h_size]), np.zeros([self.minibatch_size, agt.nn.h_size]))
				feed_dict[agt.nnT.rnn_state_in] = rnn_state_train

			if self.double_q_learning:
				# Below we perform the Double-DQN update to the target Q-values
				Q2 = self.sess.run(agt.nnT.QValue, feed_dict=feed_dict)

				# Also need to compose a feed_dict for agt.nn
				feed_dict = {agt.nn.stateInput: s_next_batch,
							 agt.nn.tracelength: self.tracelength,
							 agt.nn.truetracelengths: truetracelengths,
							 agt.nn.batch_size: self.minibatch_size}

				if agt.nn.is_rnn:
					feed_dict[agt.nn.rnn_state_in] = rnn_state_train

				predict_nn_actions = self.sess.run(agt.nn.predict, feed_dict=feed_dict)

				doubleQ = Q2[range(self.minibatch_size * self.tracelength), predict_nn_actions]
				y_batch = r_batch + (self.discount * doubleQ * non_terminal_multiplier)

			else:
				QmaxT = agt.nnT.Qmax.eval(feed_dict=feed_dict)
				y_batch = r_batch + (self.discount * QmaxT * non_terminal_multiplier)

			# Train
			feed_dict = {agt.nn.yInput: y_batch,
						 agt.nn.actionInput: a_batch,
						 agt.nn.stateInput: s_batch,
						 agt.nn.tracelength: self.tracelength,
						 agt.nn.truetracelengths: truetracelengths,
						 agt.nn.batch_size: self.minibatch_size}

			if agt.nn.is_rnn:
				feed_dict[agt.nn.rnn_state_in] = rnn_state_train

			agt.nn.trainStep.run(feed_dict=feed_dict)

			# Delay in a target network update - to improve learning stability
			if timestep % self.target_q_update_freq == 0:
				assert agt.nnT != None
				agt.nnT.run_copy()

		if timestep % 100 == 0:
			self.log_training_phase(timestep)

	def log_training_phase(self, timestep):
		if timestep <= self.n_pretrain_steps:
			phase = 'pre-train'
		elif timestep <= self.n_pretrain_steps + self.n_explore_steps:
			phase = 'train (e-greedy)'
		else:
			phase = 'train (e-greedy, min epsilon reached)'

		print 'ITER {} | PHASE {} | EPSILON {}'.format(timestep, phase, self.epsilon)

	def dec_epsilon(self, timestep):
		# Linearly decrease epsilon
		if self.epsilon > self.epsilon_final and timestep > self.n_pretrain_steps:
			self.epsilon -= (self.epsilon_init - self.epsilon_final)/self.n_explore_steps

	def get_qvalue(self, agt, input_obs):
		feed_dict= {agt.nn.stateInput:[input_obs],
					 agt.nn.tracelength: 1,
					 agt.nn.truetracelengths: [1],
					 agt.nn.batch_size: 1}

		if agt.nn.is_rnn:
			feed_dict[agt.nn.rnn_state_in] = agt.rnn_state
			QValue, agt.rnn_state = self.sess.run([agt.nn.QValue, agt.nn.rnn_state], feed_dict=feed_dict)
		else:
			QValue = self.sess.run(agt.nn.QValue, feed_dict=feed_dict)
			agt.rnn_state = None

		# stateInput is usually a batch input (due to mini-batch training), so [0] at end just grabs the QValue for this single input case
		return QValue[0]

	def get_action(self, agt, timestep, input_obs, test_mode=False, epsilon=None):
		epsilon_to_use = epsilon if epsilon else self.epsilon

		QValue = self.get_qvalue(agt, input_obs)

		# Select e-greedy action (also during pre-training phase)
		if (not test_mode and random.random() <= epsilon_to_use) or (not test_mode and timestep < self.n_pretrain_steps) or (test_mode and random.random() <= self.epsilon_test_time):
			action = random.randrange(agt.n_actions)
		# Select optimal action
		else:
			action = np.argmax(QValue)

		return action, QValue
