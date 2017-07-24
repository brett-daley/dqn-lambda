import numpy as np
from abc import ABCMeta, abstractmethod
from utils_general import create_specific_nn


class GenericAgent(object):
	__metaclass__ = ABCMeta

	def __init__(self, i_agt, n_actions, dim_obs=None):
		self.i = i_agt
		self.dim_obs = dim_obs
		self.n_actions = n_actions

	def create_nns(self, cfg_parser, sess, scope_suffix, parameter_sharing, env_s_0=None):
		self.parameter_sharing = parameter_sharing

		if self.dim_obs is None:
			self.dim_obs = len(env_s_0)

		self.nn = create_specific_nn(cfg_parser=cfg_parser, sess=sess, scope='nn_predict' + scope_suffix, var_reuse=self.parameter_sharing, dim_state_input=self.dim_obs, n_actions=self.n_actions)
		self.nnT = create_specific_nn(cfg_parser=cfg_parser, sess=sess, scope='nn_target' + scope_suffix, var_reuse=self.parameter_sharing, dim_state_input=self.dim_obs, n_actions=self.n_actions, is_target_net=True, src_network=self.nn)

	def init_nnT(self):
		if not self.nnT.initialized:
			self.nnT.run_copy()
			self.nnT.initialized = True

	def reset_rnn_state(self):
		if self.nn.is_rnn:
			self.rnn_state = (np.zeros([1, self.nn.h_size]), np.zeros([1, self.nn.h_size]))
		else:
			self.rnn_state = None

	@abstractmethod
	def get_obs(self):
		raise NotImplementedError('Must implement GenericAgent.get_obs()!')
