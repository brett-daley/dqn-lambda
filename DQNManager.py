from DRQN import DRQN
import tensorflow as tf
from utils_general import Data2DTraj
import numpy as np
import logging


class DQNManager:
	def __init__(self, cfg_parser, sess, game, n_actions):
		self.cfg_parser = cfg_parser
		self.sess = sess
		self.logger = logging.getLogger()

		self.benchmark_every_n_episodes = int(self.cfg_parser.get('env', 'benchmark_every_n_episodes'))
		self.benchmark_for_n_episodes = int(self.cfg_parser.get('env', 'benchmark_for_n_episodes'))

		self.dqn = DRQN(cfg_parser=cfg_parser, sess=self.sess, n_actions=n_actions, agt=game.agt)

		# Create a list of trainable teacher task variables, so TfSaver can later save and restore them
		self.teacher_vars = [v for v in tf.trainable_variables()]

		self.sess.run(tf.global_variables_initializer())

	def update_game(self, game, timestep=9999999, is_test_mode=False, epsilon_forced=None):
		obs = game.get_obs()
		action, qvalues = self.dqn.get_action(epsilon=epsilon_forced, agt=game.agt, timestep=timestep, input_obs=obs, test_mode=is_test_mode)
		next_obs, reward, terminal, disc_return, undisc_return, mov_avg_undisc_return = game.next(action)

		return obs, action, reward, next_obs, terminal, disc_return, undisc_return, mov_avg_undisc_return

	def train_dqn(self, game):
		train_freq = int(self.cfg_parser.get('nn', 'train_freq'))
		n_max_steps = int(self.cfg_parser.get('env', 'n_max_steps'))

		traj_predicted_disc_return = Data2DTraj()
		traj_actual_disc_return = Data2DTraj()
		traj_undisc_return = Data2DTraj()
		traj_mov_avg_undisc_return = Data2DTraj()
		self.dqn.init_plots()

		game.init_agt_nnT()

		i_training_epoch = 0
		n_games_complete = 0

		for i_train_step in xrange(n_max_steps):
			# Execute game and collect a single-timestep experience
			obs, action, reward, next_obs, terminal, disc_return, undisc_return, mov_avg_undisc_return = self.update_game(game, timestep=i_train_step)
			self.dqn.replay_memory.add(obs, action, reward, next_obs, terminal)

			# Decrease e-greedy epsilon
			self.dqn.dec_epsilon(timestep=i_train_step)

			# Train (at train_freq)
			if i_train_step % train_freq == 0:
				self.dqn.train_Q_network(timestep=i_train_step)
				i_training_epoch += 1

			if terminal:
				n_games_complete += 1

				self.dqn.plot_undisc_return.update(hl_name=None, label=None, x_new=i_train_step, y_new=undisc_return)
				self.dqn.plot_mov_avg_undisc_return.update(hl_name=None, label=None, x_new=i_train_step, y_new=mov_avg_undisc_return)

				traj_undisc_return.appendToTraj(i_train_step, undisc_return, y_stdev=0)
				traj_mov_avg_undisc_return.appendToTraj(i_train_step, mov_avg_undisc_return, y_stdev=0)

				# Benchmark performance
				if n_games_complete % self.benchmark_every_n_episodes == 0:
					mean_predicted_disc_return, stdev_predicted_disc_return, mean_actual_disc_return, stdev_actual_disc_return = self.benchmark_perf(game, timestep=i_train_step)
					traj_predicted_disc_return.appendToTraj(i_train_step, mean_predicted_disc_return, stdev_predicted_disc_return)
					traj_actual_disc_return.appendToTraj(i_train_step, mean_actual_disc_return, stdev_actual_disc_return)

		return traj_predicted_disc_return, traj_actual_disc_return, traj_undisc_return, traj_mov_avg_undisc_return

	def benchmark_perf(self, game, timestep):
		self.logger.info('----- Benchmarking performance -----')
		all_disc_returns = np.array([])

		# Initial states and actions taken -- for plotting predicted value against actual
		s_batch_initial = np.zeros([self.benchmark_for_n_episodes] + list(self.dqn.agt.dim_obs))

		for i_episode in xrange(self.benchmark_for_n_episodes):
			# Reset the game just in case anyone else was handling the game prior or forgot to reset RNN state. It should not hurt.
			game.reset_game()
			terminal = False
			collected_initial_conds = False

			while not terminal:
				obs, action, _, _, terminal, disc_return, _, _ = self.update_game(game, is_test_mode=True)
				if not collected_initial_conds:
					s_batch_initial[i_episode,:] = obs
					collected_initial_conds = True

			# Append the value obtained. Re-initialization must occur on top of loop to ensure correct setting of dqn.
			all_disc_returns = np.append(all_disc_returns, disc_return)

		mean_actual_disc_return = np.mean(all_disc_returns)
		stdev_actual_disc_return = np.std(all_disc_returns)

		mean_predicted_disc_return, stdev_predicted_disc_return = self.dqn.update_Q_plot(timestep, s_batch_prespecified=s_batch_initial)
		self.dqn.plot_actual_disc_return.update(hl_name=None, label=None, x_new=timestep, y_new=mean_actual_disc_return, y_stdev_new=stdev_actual_disc_return, init_at_origin=False)

		return mean_predicted_disc_return, stdev_predicted_disc_return, mean_actual_disc_return, stdev_actual_disc_return
