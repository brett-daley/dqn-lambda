import numpy as np
from utils_general import create_specific_nn


class Agent:
	def __init__(self, n_actions, dim_obs):
		self.dim_obs = dim_obs
		self.n_actions = n_actions

	def create_nns(self, cfg_parser, sess, scope_suffix, parameter_sharing):
		self.parameter_sharing = parameter_sharing

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
