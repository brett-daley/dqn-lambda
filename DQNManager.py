from DRQN import DRQN
import tensorflow as tf
from utils_general import Data2DTraj
import numpy as np
import time


class DQNManager:
	def __init__(self, cfg_parser, n_actions, game, sess):
		self.cfg_parser = cfg_parser
		self.sess = sess

		self.dqn = DRQN(cfg_parser=cfg_parser, sess=self.sess, n_actions=n_actions, agt=game.agt)

		# Create a list of trainable teacher task variables, so TfSaver can later save and restore them
		self.teacher_vars = [v for v in tf.trainable_variables()]

		self.sess.run(tf.global_variables_initializer())

	def update_game(self, game, timestep=9999999, is_test_mode=False, epsilon_forced=None):
		epsilon = epsilon_forced if epsilon_forced else None

		obs = game.get_obs()
		action, qvalues = self.dqn.get_action(epsilon=epsilon, agt=game.agt, timestep=timestep, input_obs=obs, test_mode=is_test_mode)
		next_obs, reward, terminal, value_so_far = game.next(action=action)

		return obs, action, reward, next_obs, terminal, value_so_far

	def train_dqn(self, game):
		train_freq = 5
		i_train_steps = 0
		i_training_epoch = 0
		n_train_steps = int(self.cfg_parser.get('env', 'n_train_steps'))
		minibatch_size = int(self.cfg_parser.get('dqn', 'minibatch_size'))

		q_value_traj = Data2DTraj()
		init_q_value_traj = Data2DTraj()
		value_traj = Data2DTraj()

		game.init_agt_nnT()

		n_games_complete = 0

		while i_train_steps < n_train_steps:
			# Game episode completed, so reset
			terminal = False
			sarsa_traj = []

			while not terminal:
				# Execute game and collect a single-timestep experience
				obs, action, reward, next_obs, terminal, _ = self.update_game(game=game, timestep=i_train_steps)
				sarsa_traj.append(np.reshape(np.array([obs, action, reward, next_obs, terminal]), [1,5]))

				# Decrease e-greedy epsilon
				self.dqn.dec_epsilon(timestep=i_train_steps)

				# Train (at train_freq)
				if n_games_complete > minibatch_size and i_train_steps % train_freq == 0:
					self.dqn.train_Q_network(timestep=i_train_steps)
					i_training_epoch += 1

				i_train_steps += 1

				# Plot q value convergence
				if n_games_complete > minibatch_size and i_train_steps % 1000 == 0:
					x, y_mean, y_stdev = self.dqn.update_Q_plot(timestep=i_training_epoch)
					q_value_traj.appendToTraj(x, y_mean, y_stdev)

			# Once game completes, add entire trajectory to replay memory
			n_games_complete += 1
			self.dqn.replay_memory.add(sarsa_traj)

			if n_games_complete == 1 or n_games_complete % 50 == 0:
				value_mean, value_stdev, init_q_mean, init_q_stdev = self.benchmark_perf(game=game, i_train_steps=i_training_epoch)
				value_traj.appendToTraj(i_training_epoch, value_mean, value_stdev)
				init_q_value_traj.appendToTraj(i_training_epoch, init_q_mean, init_q_stdev)

		return q_value_traj, value_traj, init_q_value_traj

	def benchmark_perf(self, game, i_train_steps):
		print '----- Benchmarking performance -----'
		values_all = np.array([])

		# Initial states and actions taken -- for plotting predicted value against actual
		n_episodes = 50
		s_batch_initial = np.zeros([n_episodes] + list(self.dqn.agt.dim_obs))
		a_batch_initial = np.zeros((n_episodes, self.dqn.agt.n_actions))

		for i_episode in xrange(n_episodes):
			# Reset the game just in case anyone else was handling the game prior or forgot to reset RNN state. It should not hurt.
			game.reset_game()
			terminal = False
			collected_initial_conds = False

			while not terminal:
				obs, action, _, _, terminal, value_so_far = self.update_game(game=game, is_test_mode=True)
				if not collected_initial_conds:
					s_batch_initial[i_episode,:] = obs
					a_batch_initial[i_episode,:] = action
					collected_initial_conds = True

			# Append the value obtained. Re-initialization must occur on top of loop to ensure correct setting of dqn.
			values_all = np.append(values_all, value_so_far)

		values_mean = np.mean(values_all)
		values_stdev = np.std(values_all)

		x, init_q_mean, init_q_stdev = self.dqn.update_Q_plot(timestep=i_train_steps, s_batch_prespecified=s_batch_initial, a_batch_prespecified=a_batch_initial)
		self.dqn.plot_value_dqn.update(hl_name='Game', label='Task', x_new=i_train_steps, y_new=values_mean, y_stdev_new=values_stdev, init_at_origin=False)

		return values_mean, values_stdev, init_q_mean, init_q_stdev
