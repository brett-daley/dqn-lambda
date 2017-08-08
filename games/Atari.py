from Agent import Agent
import numpy as np
import gym


class Atari:
	def __init__(self, cfg_parser, sess):
		self.cfg_parser = cfg_parser

		self.env_name = self.cfg_parser.get('env', 'name')
		self.env = gym.make(self.env_name)
		self.n_actions = self.env.action_space.n
		self.discount_factor = float(self.cfg_parser.get('dqn', 'discount'))
		self.agt_nn_is_recurrent = self.cfg_parser.getboolean('nn', 'recurrent')
		self.last_obs = None

		if not self.agt_nn_is_recurrent:
			self.history = None
			self.history_length = int(self.cfg_parser.get('nn', 'agent_history_length'))

		# Init agent
		self.init_agt(sess=sess)

		# Reset game (and also RNN state when applicable)
		self.reset_game()

	def init_agt_nnT(self):
		# Placed here since must be run after tf.all_variables_initialized()
		self.agt.init_nnT()

	def init_agt(self, sess):
		self.reset_obs()
		self.agt = Agent(self.n_actions, dim_obs=self.get_obs().shape)
		self.agt.create_nns(cfg_parser=self.cfg_parser, sess=sess, scope_suffix='', parameter_sharing=False)

	def reset_game(self):
		self.discount = 1.0
		self.value = 0.0

		self.reset_obs()

		# Reset RNN state (does nothing for non-RNN case)
		self.agt.reset_rnn_state()

	def get_obs(self):
		return self.last_obs if self.agt_nn_is_recurrent else np.concatenate(self.history, axis=-1)

	def store_obs(self, obs):
		self.last_obs = obs

		if not self.agt_nn_is_recurrent:
			self.history = self.history[1:] + [self.last_obs]

	def reset_obs(self):
		self.last_obs = self.env.reset()

		if not self.agt_nn_is_recurrent:
			self.history = [self.last_obs]*self.history_length

	def next(self, action):
		# Agent executes actions
		next_obs, reward, terminal, info = self.env.step(action)
		self.env.render()

		# Accrue value
		self.value += self.discount*reward
		self.discount *= self.discount_factor
		value_so_far = self.value # Must be here due to resetting logic below

		self.store_obs(next_obs) # Must be here due to resetting logic below

		if terminal:
			print '-------------- Total episode reward', value_so_far, '!--------------'
			self.reset_game()

		return self.get_obs(), reward, terminal, value_so_far
