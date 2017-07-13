from AgentGround import AgentGround
from utils_general import create_specific_nn
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

class PursuitEvader:
	def __init__(self, cfg_parser, game_variant, sess):
		self.cfg_parser = cfg_parser
		self.game_variant = game_variant

		# Domain settings
		self.n_agts = int(self.cfg_parser.get('root','n_agts'))
		self.is_toroidal = self.cfg_parser.getboolean('root','is_toroidal')
		self.max_time_game = int(self.cfg_parser.get('root','max_time_game'))
		self.parameter_sharing = self.cfg_parser.getboolean('root','parameter_sharing')
		self.end_game_on_simult_catch = self.cfg_parser.getboolean('root','end_game_on_simult_catch')

		# TODO 2017 make this a parameter/handled automatically in the game
		self.gamma = 0.95

		# Domain settings
		dim_grid_x = int(self.cfg_parser.get(game_variant,'dim_grid_x'))
		dim_grid_y = int(self.cfg_parser.get(game_variant,'dim_grid_y'))
		self.x_lim = [-dim_grid_x/2.,dim_grid_x/2.]
		self.y_lim = [-dim_grid_x/2.,dim_grid_x/2.]
		self.i_game = int(self.cfg_parser.get(game_variant,'i_game'))

		# Init agents/evader
		self.init_agts_tgts(sess = sess)
		# self.dim_single_agt_obs = len(self.agts[0].get_obs(self.get_env_state()))

		# Initialize agent NNs (need the game initialized at this point, since will need things like observation dimensions well-defined)
		self.create_agts_nns(sess = sess)

		# Reset game (and also RNN state when applicable)
		self.reset_game()

		# Init plotting
		should_plot = self.cfg_parser.getboolean('root','should_plot')
		if should_plot:
			self.init_plot()

	def get_init_player_state(self,player_type, i_agt = None):
		assert player_type == 'agent' or player_type == 'evader'
		if player_type == 'agent':
			# return np.asarray([self.x_lim[0], self.y_lim[0]])
			return np.asarray([np.random.randint(self.x_lim[0], self.x_lim[1]+1),
								   np.random.randint(self.y_lim[0], self.y_lim[1]+1)])
		elif player_type == 'evader':
			# return np.asarray([0,0])
			return np.asarray([np.random.randint(self.x_lim[0], self.x_lim[1]+1),
									 np.random.randint(self.y_lim[0], self.y_lim[1]+1)])

	def create_agts_nns(self, sess):
		scope_suffix = '_task' + str(self.i_game)

		# Initialize master NNs for parameters sharing case
		if self.parameter_sharing:
			scope_suffix = scope_suffix + "/shared"
			
			# Parameter sharing assumes homogeneous agents (same inputs/outputs)
			dim_obs = len(self.agts[0].get_obs(self.get_env_state()))
			n_actions = self.agts[0].n_actions
			
			# First initialization must have var_reuse disabled
			nn_predict_master = create_specific_nn(cfg_parser = self.cfg_parser, sess = sess, scope = "nn_predict" + scope_suffix, var_reuse = False, dim_state_input = dim_obs, n_actions = n_actions)
			nn_target_master  = create_specific_nn(cfg_parser = self.cfg_parser, sess = sess, scope = "nn_target"  + scope_suffix, var_reuse = False, dim_state_input = dim_obs, n_actions = n_actions, is_target_net = True, src_network = nn_predict_master)

		# Create actual player NNs
		for agt in self.agts:
			agt.create_nns(cfg_parser = self.cfg_parser, sess = sess, scope_suffix = scope_suffix, env_s_0 = self.get_env_state(), parameter_sharing = self.parameter_sharing)

	def init_agts_nnTs(self):
		# Placed here since must be run after tf.all_variables_initialized()
		for agt in self.agts:
			agt.init_nnT()

	def init_agts_tgts(self, sess):
		# Init agents
		self.agts = list()
		for i_agt in xrange(0,self.n_agts):
			xy_0 = self.get_init_player_state(player_type = 'agent', i_agt = i_agt)
			self.agts.append(AgentGround(player_type = 'agent', cfg_parser = self.cfg_parser, game_variant = self.game_variant, i_agt = i_agt, n_agts = self.n_agts, xy_0 = xy_0, 
							x_lim = self.x_lim, y_lim = self.y_lim, is_toroidal = self.is_toroidal, sess = sess))
			# print "Agent", self.agts[i_agt].i, "| Action:", self.agts[i_agt].action_map

		# Init evader (which is just a special agent with index -1)
		xy_0 = self.get_init_player_state(player_type = 'evader')
		self.evader = AgentGround(player_type = 'evader', cfg_parser = self.cfg_parser, game_variant = self.game_variant, i_agt = 0, xy_0 = xy_0, 
									x_lim = self.x_lim, y_lim = self.y_lim, is_toroidal = self.is_toroidal)

	def reset_game(self):
		# Game only runs for 40 timesteps, after which it is forcibly reset
		self.time_game = 0
		self.discount = 1.0
		self.value = 0.0
		self.next_joint_o = None # Store the observation to ensure consistency in noise/observation process when querying observations within same timestep

		# Re-initialize agents
		for agt in self.agts:
			xy_0 = self.get_init_player_state(player_type = 'agent', i_agt = agt.i)
			agt.reset_state(xy_0 = xy_0)
			# Reset RNN state (does nothing for non-RNN case)
			agt.reset_rnn_state()

		# Init evader (which is just a special agent with index -1)
		xy_0 = self.get_init_player_state(player_type = 'evader')
		self.evader.reset_state(xy_0 = xy_0)

	def init_plot(self):
		self.fig_game, self.ax = plt.subplots(1,1)
		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Y')
		self.ax.set_xlim(self.x_lim)
		self.ax.set_ylim(self.y_lim)
		plt.ion()
		
		x, y = np.random.random((2, 1))
		self.scat_evader = self.ax.scatter(x, y, c=[1, 0, 0], s=200)
		x, y, z = np.random.random((3, self.n_agts))
		self.scat_agts = self.ax.scatter(x, y, c=z, s=200)

	def plot(self, t):
		self.scat_evader.set_offsets(self.evader.s)
		s_list = [agt.s+(agt.i+1)/20. for agt in self.agts] # A bit of offset included in each agent so overlaps are easier to see
		self.scat_agts.set_offsets(s_list)
		plt.title('Time %d' % (t))
		plt.draw()
		plt.pause(0.1)
	
	def evader_captured(self, enforce_simult_capture):
		n_times_captured = 0

		# Count # of agents that simultaneously captured the evader in current timestep
		for agt in self.agts:
			if np.array_equal(agt.s, self.evader.s):
				n_times_captured += 1

		# if n_times_captured > 0:
		# 	print n_times_captured

		if (enforce_simult_capture and n_times_captured == self.n_agts) or (not enforce_simult_capture and n_times_captured >= 1):
			return True

		return False

	def get_env_state(self):
		# Mostly to make it easier to call agt.get_obs() if the environment state needed for the joint obs changes
		return self.evader.s

	def get_joint_obs(self):
		# Observations are pre-processed at the previous timestep (in self.next()) due to
		# POMDP-ness (to ensure multiple queried observations at same timestep don't change due to noise)
		if self.next_joint_o is None:
			# First game timestep, so no preprocessed next_joint_o yet, grab a fresh one
			return [agt.get_obs(self.get_env_state()) for agt in self.agts]
		else:
			# Get pre-processed observation
			return self.next_joint_o


	def next(self, i_actions):
		game_over = False
		r = 0

		if len(i_actions) < self.n_agts:
			raise ValueError('Not enough actions specified! %d agents need exactly %d actions!' %(self.n_agts,self.n_agts))

		# Game itself propagates (in this case, evader moves)
		self.evader.prop_action_evader()

		# Agents execute actions
		for i_agt, agt in enumerate(self.agts):
			agt.exec_action_agt(one_hot_action = i_actions[i_agt])

		# Check if evader was caught
		if self.evader_captured(enforce_simult_capture = True):
			print '-------------- Evader caught at', self.time_game, '!--------------'
			r = 1
			# Set end_game_on_simult_catch to false to increase success rate (more reward feedback)
			if self.end_game_on_simult_catch:
				game_over = True

		if self.time_game > self.max_time_game:
			game_over = True

		# Pack next joint state
		self.next_joint_o = [agt.get_obs(self.get_env_state()) for agt in self.agts]
		# print next_joint_o
		# print 'next_joint_o', next_joint_o
		
		# Accrue value
		self.value += self.discount*r
		self.discount *= self.gamma
		value_so_far = self.value # Must be here due to resetting logic below
		next_joint_o_latest = self.next_joint_o # Must be here due to resetting logic below

		if game_over:
			self.reset_game()
		else:
			self.time_game += 1

		return next_joint_o_latest, r, game_over, value_so_far