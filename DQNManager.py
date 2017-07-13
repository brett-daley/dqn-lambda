from DRQN import DRQN
import tensorflow as tf
from TargetPursuit import TargetPursuit
from PursuitEvader import PursuitEvader
from utils_general import Data2DTraj
import numpy as np
import time
from Distiller import Distiller


class DQNManager:
	def __init__(self, cfg_parser, n_teacher_dqns, n_actions, game_mgr, sess, enable_mdrqn = False, enable_distiller = True, enable_double_distiller = False):
		# Cannot have both vanilla MDRQN and MSE distillation companion
		if enable_mdrqn:
			assert enable_distiller is False
	

		self.cfg_parser = cfg_parser
		self.sess = sess
		
		self.dqns_all = list()
		# There should definitely be one DRQN handler per game, since each one has a distinct e-greedy epsilon etc.
		for i_game, game in enumerate(game_mgr.games):
			# The scope should be unique and identify the teacher network ID (done here) and agent ID (done in DRQN)
			self.dqns_all.append(DRQN(cfg_parser=cfg_parser, sess=self.sess, n_actions=n_actions, agts=game.agts))

		# Create a list of trainable teacher task variables, so TfSaver can later save and restore them
		if n_teacher_dqns > 0:
			self.teacher_vars = [v for v in tf.trainable_variables()]

		if enable_mdrqn:
			self.mdrqn = DRQN(cfg_parser=cfg_parser, sess=self.sess, n_actions=n_actions, is_mdrqn=True,  dim_obs_agts=[agt.dim_obs for agt in game_mgr.games[0].agts])

		# Note: distillation assumes the inputs/outputs are homogeneous across all games (but can still be heterogeneous across all agents)
		if enable_distiller:
			self.distiller = Distiller(cfg_parser = cfg_parser, sess = self.sess, n_actions = n_actions, 
										dim_obs_agts = [agt.dim_obs for agt in game_mgr.games[0].agts])
			self.all_vars = [v for v in tf.trainable_variables()]

			self.distiller_companion = None # Such that KL distillation training gets a None input for its companion, if no double distillation
			if enable_double_distiller:
				self.distiller_companion = DRQN(cfg_parser=cfg_parser, sess=self.sess, n_actions=n_actions, is_mdrqn=False, is_distiller_companion=True,  dim_obs_agts=[agt.dim_obs for agt in game_mgr.games[0].agts])

		self.sess.run(tf.global_variables_initializer())

	def update_game(self, game, timestep = 9999999, is_distillation_phase = False, is_test_mode = False, use_mdrqn = False, decision_maker_drqn = None, epsilon_forced = None):
		# DO NOT use this function for distillation case. This makes some assumptions about dqn/game setup that distillation manager should not use.

		# Only used by teacher DQNs, not by distillation, so is fine
		cur_joint_obs = game.get_joint_obs()

		# Joint action as list allows heterogeneous/different size action vectors
		joint_i_actions = list()
		if epsilon_forced is not None:
			epsilon = epsilon_forced
		else:
			epsilon = None

		if is_distillation_phase:
			joint_q_values = list()
			# test_mode = True # TODO 2017 this has to be changed if want to have exploration during distillation
			test_mode = False
		elif is_test_mode:
			test_mode = True
		else:
			test_mode = False

		if use_mdrqn:
			if decision_maker_drqn is None:
				# Use MDRQN for decision-making as well as MRDQN agent (since RNN is stored inside it)
				cur_dqn = self.mdrqn
				cur_agts = self.mdrqn.agts
			else:
				# Distiller-mimicer case, where usually the distillaton policy is passed in to take actions and make decisions
				cur_dqn = self.dqns_all[game.i_game]
				cur_agts = decision_maker_drqn.agts
				epsilon = epsilon_forced # decision_maker_drqn.epsilon # exploration for both distillation and mdrqn cases
			# epsilon = None
		else:
			# Use regular DQN and regular game agents
			cur_dqn = self.dqns_all[game.i_game]
			cur_agts = game.agts

		for agt in cur_agts:							
			cur_agt_action, cur_agt_qvalues = cur_dqn.get_action(epsilon = epsilon, agt = agt, timestep = timestep, 
																input_obs = cur_joint_obs[agt.i], test_mode = test_mode, i_game = game.i_game)

			joint_i_actions.append(cur_agt_action)
			if is_distillation_phase:
				joint_q_values.append(cur_agt_qvalues)

		next_joint_obs, r, terminal, value_so_far = game.next(i_actions = joint_i_actions)

		if not is_distillation_phase:
			return cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal, value_so_far
		else:
			# Send back everything for distillation phase, in case action exploration was enabled and data was collected for it
			return cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal, joint_q_values


	def train_mdrqn(self, game_mgr, mdrqn, decision_maker_drqn = None, x_timestep_plot_offset = 0):
		n_train_steps = 500000 # 20000 #100000
		train_freq = 5
		i_train_steps = 0
		i_train_steps_all_games = np.zeros((game_mgr.n_game_variants,),dtype=np.int32)
		
		minibatch_size = int(self.cfg_parser.get('root','minibatch_size'))
		# q_value_traj = Data2DTraj()
		mdrqn.init_agts_nnTs()
		mdrqn.replay_memory.reset()

		n_games_complete = 0
		# if decision_maker_drqn is not None:
		mdrqn.epsilon = 1.#0.3

		while i_train_steps < n_train_steps:
			for game in game_mgr.games:
				terminal = False
				game.reset_game() # This resets the game's internal agent RNNs, which aren't actually used in MDRQN case!
				mdrqn.reset_agts_rnn_states() # This resets the actual MDRQN RNN states, so is of utmost importance!
				if decision_maker_drqn is not None:
					decision_maker_drqn.reset_agts_rnn_states()
					mdrqn.n_iter_pretrain = 200 #don't do excessive pre-training data collection for mdrqn w/ distillation MSE decision maker


				joint_sarsa_traj = []
				have_trained_this_round = False

				while not terminal:
					# Execute game and collect a single-timestep experience
					cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal, _ = self.update_game(use_mdrqn = True, is_test_mode=False, game = game, 
																			timestep = i_train_steps_all_games[game.i_game], decision_maker_drqn = decision_maker_drqn,
																			epsilon_forced = mdrqn.epsilon)
					joint_sarsa_traj.append(np.reshape(np.array([cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal]),[1,5]))

					

					# Decrease e-greedy epsilon (if actually using mdrqn, instead of decision_maker_drqn). This is used for all agents' exploration policies.
					if decision_maker_drqn is not None:
						mdrqn.dec_epsilon(timestep = i_train_steps_all_games[game.i_game])

					# Train (at train_freq)
					if n_games_complete > minibatch_size and i_train_steps_all_games[game.i_game] % train_freq == 0:# and not have_trained_this_round:
						# print 'training at timestep', i_train_steps, 'current game', game.i_game, 'replay mem length', len(mdrqn.replay_memory.traj_mem)
						mdrqn.train_Q_network(timestep = i_train_steps_all_games[game.i_game], decision_maker_drqn = decision_maker_drqn)
						have_trained_this_round = True
					
					i_train_steps_all_games[game.i_game] += 1
					i_train_steps += 1

					# Plot q value convergence for first game
					if n_games_complete > minibatch_size and i_train_steps_all_games[game.i_game] % 1000 == 0:
						x, y_mean, y_stdev = mdrqn.update_Q_plot(hl_name = 'qplot_' + str(game.i_game), label = 'Task ' + str(game.i_game), timestep = x_timestep_plot_offset + i_train_steps)
				
					if i_train_steps % 1000 == 0:
						print '----- MDRQN iteration', i_train_steps, '-----'

				# Once game completes, add entire trajectory to replay memory
				n_games_complete += 1
				mdrqn.replay_memory.add(joint_sarsa_traj)

				if n_games_complete>32 and n_games_complete % 100 == 0:
					print '----- Benchmarking train_mdrqn policy -----'
					data_dict = self.benchmark_multitask_perf(game_mgr = game_mgr, multitask_dqn = mdrqn, x_timestep_plot = x_timestep_plot_offset + i_train_steps)

		# return x_data, y_mean_data, y_stdev_data
		return n_train_steps, data_dict

	def train_teacher_dqn(self, i_game, game):
		train_freq = 5
		i_train_steps = 0
		i_training_epoch = 0
		n_train_steps = int(self.cfg_parser.get('root','n_train_steps'))
		n_train_step_plot_game = int(self.cfg_parser.get('root','n_train_step_plot_game'))
		minibatch_size = int(self.cfg_parser.get('root','minibatch_size'))

		q_value_traj = Data2DTraj()
		# init_q_value_traj = Data2DTraj()
		init_q_value_traj = list()
		for agt in game.agts:
			init_q_value_traj.append(Data2DTraj())
		joint_value_traj = Data2DTraj()

		game.init_agts_nnTs()

		n_games_complete = 0

		while i_train_steps < n_train_steps:
			# Game episode completed, so reset
			terminal = False
			joint_sarsa_traj = []
			
			while not terminal:				
				# Execute game and collect a single-timestep experience
				cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal, _ = self.update_game(game = game, timestep = i_train_steps)
				joint_sarsa_traj.append(np.reshape(np.array([cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal]),[1,5]))

				# Decrease e-greedy epsilon (if needed). This is used for all agents' exploration policies.
				self.dqns_all[i_game].dec_epsilon(timestep = i_train_steps)

				# Train (at train_freq)
				if n_games_complete > minibatch_size and i_train_steps % train_freq == 0:
					self.dqns_all[i_game].train_Q_network(timestep = i_train_steps)
					i_training_epoch += 1

				i_train_steps += 1

				# Plot q value convergence
				if n_games_complete > minibatch_size and i_train_steps % 1000 == 0:
					x, y_mean, y_stdev = self.dqns_all[i_game].update_Q_plot(timestep = i_training_epoch)
					q_value_traj.appendToTraj(x, y_mean, y_stdev)

			# Once game completes, add entire trajectory to replay memory
			n_games_complete += 1
			self.dqns_all[i_game].replay_memory.add(joint_sarsa_traj)

			if i_train_steps>n_train_step_plot_game:
				self.show_game(game = game, i_game = i_game, dqn = self.dqns_all[i_game])

			if n_games_complete == 1 or n_games_complete % 50 == 0: 
				for agt in game.agts:
					joint_value_mean, joint_value_stdev, init_q_mean, init_q_stdev = self.benchmark_singletask_perf(game = game, i_train_steps = i_training_epoch, i_agt_to_plot = agt.i)
					joint_value_traj.appendToTraj(i_training_epoch, joint_value_mean, joint_value_stdev)
					init_q_value_traj[agt.i].appendToTraj(i_training_epoch, init_q_mean, init_q_stdev)

		return q_value_traj, joint_value_traj, init_q_value_traj
			
	def distill_dqns(self, game_mgr):
		# No Q-learning for distilled
		# self.distiller.epsilon = 0.05
		q_value_traj = Data2DTraj()

		# Reset teacher replay memories and set their e-greedy epsilon = 5%
		for dqn_teacher in self.dqns_all:
			dqn_teacher.epsilon = 1.#0.3 # This exploration is high just to provide coverage of the state space for regression

		# Reset the games
		for game in game_mgr.games:
			game.reset_game() 

		i_distilled_training = 0
		i_trained_distilled_companion = 0
		mdrqn_train_steps = 0
		self.distiller_companion = None

		# Main distillation loop
		while i_distilled_training < 100000:
			# Online data collection: each teacher refills a portion (10% in paper) of special distillation replay memory of {states, Q-for-all-actions} using its game
			print '----- Game sampling phase -----'
			for game in game_mgr.games:
				for n_games_complete in xrange(0,32):
					# Reset the game just in case anyone else was handling the game prior or forgot to reset RNN state. It should not hurt.
					game.reset_game()
					joint_distill_traj = []
					terminal = False

					while not terminal:
						cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal, joint_q_values = self.update_game(game = game, is_distillation_phase = True)
						joint_distill_traj.append(np.reshape(np.array([cur_joint_obs, joint_i_actions, r, next_joint_obs, terminal, joint_q_values]),[1,6]))
						# print joint_i_actions
				 	self.dqns_all[game.i_game].replay_memory_distillation.add(joint_distill_traj)


			# Reset teacher replay memories and set their e-greedy epsilon = 5%
			# if i_distilled_training>6000:
			# 	for dqn_teacher in self.dqns_all:
			# 		dqn_teacher.epsilon = 0.2

				 # Add some replay memory from explorer policy
				 # for n_games_complete in xrange(0,16):

			# Use specialized game DRQNs to train a specialized MDRQN with exploration
			# if i_distilled_training>=9000 and self.distiller_companion is not None:
			# print '----- Training exploration MDRQN -----'			
			# mdrqn_train_steps = self.train_mdrqn_using_game_dqns(game_mgr = game_mgr, mdrqn = self.distiller_companion, x_timestep_plot_offset = i_trained_distilled_companion)

			# print i_distilled_training
			if i_distilled_training>=9000 and self.distiller_companion is not None:
				print '----- Companion distiller exploratory training phase -----'
				mdrqn_train_steps, _= self.train_mdrqn(game_mgr = game_mgr, mdrqn = self.distiller_companion, decision_maker_drqn = self.distiller, x_timestep_plot_offset = i_trained_distilled_companion)
				i_trained_distilled_companion += 1

			print '----- Distillation phase iteration', i_distilled_training, '-----'
			# Train the student with minibatch updates (each minibatch is drawn from a randomly chosen single game memory)
			for i_minibatch in xrange(0,3001):
				# Randomly select a task
				i_teacher = np.random.randint(game_mgr.n_game_variants)

				# Train the distilled policy using the randomly selected task
				self.distiller.train_distillation_Q_network(teacher = self.dqns_all[i_teacher])#, distiller_companion = self.distiller_companion)
				i_distilled_training += 1

				if i_minibatch>100 and i_minibatch % 100 == 0:
					print 'Distillation minibatch', i_minibatch

					self.distiller.update_distillation_Q_plot(iter = i_minibatch, teacher = self.dqns_all[i_teacher])
					# if i_distilled_training>32:
					# 	self.distiller_companion.update_Q_plot(timestep = i_distilled_training)
			
					if i_minibatch % 1000 == 0:
						print '----- Benchmarking distilled policy (KL distiller) -----'
						data_dict = self.benchmark_multitask_perf(game_mgr, multitask_dqn = self.distiller, x_timestep_plot = i_distilled_training)
						# if self.distiller_companion is not None:
						# 	print '----- Benchmarking distilled policy (MSE companion distiller) -----'
						# 	x, y_mean, y_stdev = self.benchmark_multitask_perf(game_mgr, multitask_dqn = self.distiller_companion, x_timestep_plot = mdrqn_train_steps*i_trained_distilled_companion + i_distilled_training)
						# q_value_traj.appendToTraj(x,y_mean,y_stdev)
						# q_value_traj.saveData(data_dir = data_dir)

					# if i_minibatch % 1000 == 0 and i_distilled_training > 350000:
					# 	self.show_game(game_mgr, i_game = i_teacher, dqn = self.distiller)
		return data_dict


	def benchmark_singletask_perf(self, game, i_train_steps, i_agt_to_plot = 0):
		print '----- Benchmarking singletask teacher -----'
		values_all = np.array([])

		# Initial states and actions taken -- for plotting predicted value against actual
		n_episodes = 50
		i_agt_to_plot = i_agt_to_plot
		s_batch_initial = np.zeros((n_episodes, self.dqns_all[game.i_game].agts[0].dim_obs))
		a_batch_initial = np.zeros((n_episodes, self.dqns_all[game.i_game].agts[0].n_actions))

		for i_episode in xrange(0,n_episodes):
			# Reset the game just in case anyone else was handling the game prior or forgot to reset RNN state. It should not hurt.
			game.reset_game()
			terminal = False
			collected_initial_conds = False

			while not terminal:
				cur_joint_obs, joint_i_actions, _, _, terminal, value_so_far = self.update_game(game = game, is_test_mode = True)
				if not collected_initial_conds:
					s_batch_initial[i_episode,:] = cur_joint_obs[i_agt_to_plot]
					a_batch_initial[i_episode,:] = joint_i_actions[i_agt_to_plot]
					collected_initial_conds = True

			# Append the value obtained. Re-initialization must occur on top of loop to ensure correct setting of dqn.
			values_all = np.append(values_all, value_so_far)

		joint_values_mean = np.mean(values_all)
		joint_values_stdev = np.std(values_all)

		x, init_q_mean, init_q_stdev = self.dqns_all[game.i_game].update_Q_plot(timestep = i_train_steps, i_agt_to_plot = i_agt_to_plot, s_batch_prespecified = s_batch_initial, a_batch_prespecified = a_batch_initial)
		self.dqns_all[game.i_game].plot_value_dqn.update(hl_name = 'game_' + str(game.i_game), label = 'Task ' + str(game.i_game), x_new = i_train_steps, y_new = joint_values_mean, y_stdev_new = joint_values_stdev, init_at_origin = False)

		return joint_values_mean, joint_values_stdev, init_q_mean, init_q_stdev


	def benchmark_multitask_perf(self, game_mgr, multitask_dqn, x_timestep_plot = 0):
		for game in game_mgr.games:
			print '----- Benchmarking game', game.i_game, ' -----'
			values_all = np.array([])

			for n_episodes in xrange(0,50):
				# Reset the game just in case anyone else was handling the game prior or forgot to reset RNN state. It should not hurt.
				game.reset_game()
				terminal = False
				
				# Distillation case again
				multitask_dqn.reset_agts_rnn_states()

				while not terminal:
					cur_joint_obs = game.get_joint_obs()
					joint_i_actions = list()

					for agt_in_game in game.agts:
						# Distilled policy case, use distilled agent to choose actions, but the actual game agent to get observations and execute action
						cur_agt_action, cur_agt_qvalues = self.dqns_all[game.i_game].get_action(agt = multitask_dqn.agts[agt_in_game.i], timestep = 9999999, input_obs = cur_joint_obs[agt_in_game.i], test_mode = True)
						joint_i_actions.append(cur_agt_action)
					
					# Execute the joint action chosen by the distilled polic
					_, _, terminal, value_so_far = game.next(i_actions = joint_i_actions)

				# Append the value obtained. Re-initialization must occur on top of loop to ensure correct setting of dqn.
				values_all = np.append(values_all, value_so_far)

			values_mean = np.mean(values_all)
			values_stdev = np.std(values_all)

			multitask_dqn.plot_value_dqn.update(hl_name = 'game_' + str(game.i_game), label = 'Task ' + str(game.i_game), x_new = x_timestep_plot, y_new = values_mean, y_stdev_new = values_stdev, init_at_origin = True)

		# print multitask_dqn.plot_value_dqn.data_dict['game_0'].x_traj
		# print multitask_dqn.plot_value_dqn.data_dict['game_0'].y_mean_traj
		# print multitask_dqn.plot_value_dqn.data_dict['game_0'].y_stdev_traj
		# time.sleep(10)

		return multitask_dqn.plot_value_dqn.data_dict #x_timestep_plot, values_mean, values_stdev

	def show_game(self, game, i_game, dqn):
		terminal = False
		i = 0
		game.reset_game()

		while i<100:
			cur_joint_obs = game.get_joint_obs()
			joint_i_actions = list()
			for agt in game.agts:
				cur_agt_action, cur_agt_qvalues =  self.dqns_all[game.i_game].get_action(agt = agt, timestep = -1, input_obs = cur_joint_obs[agt.i], test_mode = True)
				joint_i_actions.append(cur_agt_action)
			game.next(i_actions = joint_i_actions)

			game.plot(0)

			if terminal:
				time.sleep(2)

			i +=1

class GameManager:
	def __init__(self, cfg_parser, sess):
		

		base_game_name = cfg_parser.get('root','base_game_name')
		self.games = list()
		
		# Variants of the base game
		variants_list = (variant for variant in cfg_parser.sections() if variant != 'root')
		self.n_game_variants =  0
		
		# Set up all game variants
		for variant in variants_list:
			self.n_game_variants += 1

			# Single agent target pursuit game
			if base_game_name == 'SingleAgentSingleTargetPursuit':
				self.games.append(PursuitEvader(cfg_parser = cfg_parser, game_variant = variant, sess = sess))
			# Multiagent target pursuit game
			elif base_game_name == 'MultiagentTargetPursuit':
				self.games.append(TargetPursuit(cfg_parser=cfg_parser, game_variant=variant, sess=sess))
				
			# Assuming actions and observation dimensions are consistent across games
			self.n_actions = self.games[0].agts[0].n_actions 
			# self.dim_agt_obs = self.games[0].dim_agt_obs

		print self.n_game_variants 
