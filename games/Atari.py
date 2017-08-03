from Agent import Agent
import numpy as np
import gym


class Atari:
	def __init__(self, cfg_parser, sess):
		self.cfg_parser = cfg_parser

		self.env_name = self.cfg_parser.get('env', 'name')
		self.env = gym.make(self.env_name)
		self.n_actions = self.env.action_space.n
		self.obs = self.env.reset()

		self.discount_factor = float(self.cfg_parser.get('dqn', 'discount'))

		# Init agent
		self.init_agt(sess=sess)

		# Reset game (and also RNN state when applicable)
		self.reset_game()

	def init_agt_nnT(self):
		# Placed here since must be run after tf.all_variables_initialized()
		self.agt.init_nnT()

	def init_agt(self, sess):
		self.agt = Agent(self.n_actions, dim_obs=self.obs.shape)
		self.agt.create_nns(cfg_parser=self.cfg_parser, sess=sess, scope_suffix='', parameter_sharing=False)

	def reset_game(self):
		self.discount = 1.0
		self.value = 0.0
		self.obs = self.env.reset()

		# Reset RNN state (does nothing for non-RNN case)
		self.agt.reset_rnn_state()

	def get_obs(self):
		return self.obs

	def next(self, action):
		# Agent executes actions
		next_obs, reward, terminal, info = self.env.step(action)
		self.env.render()

		# Accrue value
		self.value += self.discount*reward
		self.discount *= self.discount_factor
		value_so_far = self.value # Must be here due to resetting logic below
		self.obs = next_obs # Must be here due to resetting logic below

		if terminal:
			print '-------------- Total episode reward', value_so_far, '!--------------'
			self.reset_game()

		return next_obs, reward, terminal, value_so_far
