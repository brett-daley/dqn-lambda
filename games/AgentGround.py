import numpy as np
from GenericAgent import GenericAgent


class AgentGround(GenericAgent):
	def __init__(self, player_type, cfg_parser, game_variant, i_agt, xy_0, x_lim, y_lim, is_toroidal, n_agts=0, sess=None, scope_suffix=None, corresponding_agt=None):
		self.cfg_parser = cfg_parser

		if cfg_parser.has_option('root', 'include_color_obs_and_action'):
			self.include_color_obs_and_action = self.cfg_parser.getboolean('root', 'include_color_obs_and_action')
		else:
			self.include_color_obs_and_action = False

		if cfg_parser.has_option('root', 'p_obs_flicker'):
			self.p_obs_flicker = float(self.cfg_parser.get('root', 'p_obs_flicker'))
			print 'Observation flicker probability', self.p_obs_flicker
		else:
			self.p_obs_flicker = 0.

		if cfg_parser.has_option('root', 'obs_include_last_action'):
			self.obs_include_last_action = self.cfg_parser.getboolean('root', 'obs_include_last_action')
		else:
			self.obs_include_last_action = False

		if cfg_parser.has_option('root', 'obs_include_last_reward'):
			self.obs_include_last_reward = self.cfg_parser.getboolean('root', 'obs_include_last_reward')
		else:
			self.obs_include_last_reward = False

		if cfg_parser.has_option('root', 'obs_include_agt_id'):
			self.obs_include_agt_id = self.cfg_parser.getboolean('root', 'obs_include_agt_id')
		else:
			self.obs_include_agt_id = False

		if self.include_color_obs_and_action:
			self.actions = ['north','east','south','west','wait','observe_tgt_colors']
		else:
			self.actions = ['north','east','south','west','wait']

		# Keep this here. Order matters.
		super(self.__class__, self).__init__(i_agt=i_agt, n_actions=len(self.actions))

		self.n_agts = n_agts
		self.corresponding_agt = corresponding_agt
		self.player_type = player_type
		self.x_lim = x_lim
		self.y_lim = y_lim
		self.x_range = self.x_lim[1]-self.x_lim[0]
		self.y_range = self.y_lim[1]-self.y_lim[0]
		self.i_game = int(self.cfg_parser.get(game_variant, 'i_game'))

		self.is_toroidal = is_toroidal

		self.action_map = dict()
		for i in range(self.n_actions):
			self.action_map[self.actions[i]] = i

		self.s = xy_0
		self.append_color_state()
		self.evader_col_obs_mask = 0.
		self.last_action = [0]*self.n_actions
		self.last_reward = 0

	def get_state_xy(self):
		if self.player_type == 'agent':
			return self.s
		elif self.player_type == 'evader':
			if self.include_color_obs_and_action:
				return self.s[0:2] # exclude color state for evader
			else:
				return self.s

	def reset_state(self, xy_0):
		self.s = xy_0
		self.append_color_state() # Append color state (applies to tgts only)
		if self.player_type == 'agent' and self.include_color_obs_and_action:
			# Initially disable observation of target colors
			self.evader_col_obs_mask = 0.

		self.last_action = [0]*self.n_actions
		self.last_reward = 0

	def append_color_state(self):
		# Each target gets assigned a unique color to help indicate to its pursuing agent which target to try and catch
		if self.player_type == 'evader' and self.include_color_obs_and_action:
			self.s = np.append(self.s, self.corresponding_agt + 1) # Color state is target corresponding `pursuer' agent's ID + 1 (since 0 is a reserved colorless state)

	def post_process_next_state(self, s):
		if self.is_toroidal:
			# Wrap x
			if s[0] >= self.x_lim[1]:
				s[0] -= self.x_range
			elif s[0] < self.x_lim[0]:
				s[0] += self.x_range
			# Wrap y
			if s[1] >= self.y_lim[1]:
				s[1] -= self.y_range
			elif s[1] < self.y_lim[0]:
				s[1] += self.y_range
			return s 
		else:
			return np.clip(s, [self.x_lim[0], self.y_lim[0]], [self.x_lim[1], self.y_lim[1]])

	def exec_action_agt(self, one_hot_action):
		i_action = np.argmax(one_hot_action)
		action = self.actions[i_action]

		s_north = self.s + [0,1]
		s_west = self.s + [-1,0]
		s_south = self.s + [0,-1]
		s_east = self.s + [1,0]

		# Calculate possible noisy next states (left/fwd/right)
		if action == 'north': # North
			s_next = [s_west, s_north, s_east]
		elif action == 'east': # East
			s_next = [s_north, s_east, s_south]
		elif action == 'south': # South
			s_next = [s_east, s_south, s_west]
		elif action == 'west': # West
			s_next = [s_south, s_west, s_north]
		elif action == 'wait' or action == 'observe_tgt_colors':
			s_next = [self.s]*3
		else:
			raise ValueError('[exec_action_agt] Invalid action index!')

		if self.include_color_obs_and_action:
			if action == 'observe_tgt_colors':
				# Allow next obs of tgt colors
				self.evader_col_obs_mask = 1.
			else:
				# Mask next obs of tgt colors
				self.evader_col_obs_mask = 0.

		# Noisy transition
		noise = 0.1
		i_transition = np.random.choice(3, p=[noise/2.0, 1.0-noise, noise/2.0])
		s_next = s_next[i_transition]

		# Ensure state follows game dynamics (e.g., toroidal grid etc.)
		self.s = self.post_process_next_state(s_next)
		self.last_action = one_hot_action 

	def get_obs(self, s_evaders):
		if self.include_color_obs_and_action:
			assert np.shape(s_evaders)[1] % 3 == 0
			xy_evaders = s_evaders[:,:-1]
			col_evaders = s_evaders[:,-1]
		else:
			xy_evaders = s_evaders

		obs = np.array([])

		#Normalized own location
		obs = np.append(obs, 1./self.x_range*self.s)

		# Normalized targets' locations
		if np.random.random() < self.p_obs_flicker:
			obs = np.append(obs, 0.*xy_evaders)
		else:
			obs = np.append(obs, 1./self.x_range*xy_evaders)

		# Evader color
		if self.include_color_obs_and_action:
			col_evaders = np.reshape(np.eye(len(col_evaders))[col_evaders-1], -1) # Assumes one evader per agent!
			obs = np.append(obs, col_evaders*self.evader_col_obs_mask)
			self.evader_col_obs_mask = 0.
		
		# Own ID
		if self.obs_include_agt_id:
			obs = np.append(obs, np.eye(self.n_agts)[self.i])
		if self.obs_include_last_action:
			obs = np.append(obs, self.last_action)
		if self.obs_include_last_reward:
			obs = np.append(obs, self.last_reward)

		return obs

	def prop_action_evader(self):
		# Target is autonomous, so propagates itself using its own policy

		# Example policy: move horizontally. Can include more complex target policies here.
		if self.i_game == 0:
			self.s[0] += 1

		# Ensure state follows game dynamics (e.g., toroidal grid etc.)
		self.s = self.post_process_next_state(self.s)
