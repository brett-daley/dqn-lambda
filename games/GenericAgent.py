import tensorflow as tf 
import numpy as np
import time
from abc import ABCMeta, abstractmethod
from utils_general import create_specific_nn

class GenericAgent(object):
	__metaclass__ = ABCMeta

	def __init__(self, i_agt, n_actions, dim_obs = None):
		self.i = i_agt
		self.dim_obs = dim_obs
		self.n_actions = n_actions
	
	def create_nns(self, cfg_parser, sess, scope_suffix, parameter_sharing, is_distillation_net = False, is_distiller_companion = False, env_s_0 = None):
		self.parameter_sharing = parameter_sharing
		
		if self.dim_obs is None: # Only need to do this in teacher initialization case. In fact, TODO 2017 I think this can be unified with distilled case, where observation is passed in from DRQNMultiagentManager
			self.dim_obs = len(self.get_obs(env_s_0))

		if not self.parameter_sharing: 
			scope_suffix = scope_suffix + "/agt" + str(self.i)

		self.nn = create_specific_nn(cfg_parser = cfg_parser, sess = sess, scope = "nn_predict" + scope_suffix, 
									var_reuse = self.parameter_sharing, dim_state_input = self.dim_obs, 
									n_actions = self.n_actions, is_distillation_net = is_distillation_net,
									is_distiller_companion = is_distiller_companion) 

		self.nnT = create_specific_nn(cfg_parser = cfg_parser, sess = sess, scope = "nn_target" + scope_suffix, 
									var_reuse = self.parameter_sharing, dim_state_input = self.dim_obs, 
									n_actions = self.n_actions, is_target_net = True, src_network = self.nn,
									is_distillation_net = is_distillation_net,
									is_distiller_companion = is_distiller_companion) 

	def init_nnT(self):
		if not self.nnT.initialized:
			self.nnT.run_copy()
			self.nnT.initialized = True

	def reset_rnn_state(self):
		if self.nn.is_rnn:
			self.rnn_state = (np.zeros([1,self.nn.h_size]),np.zeros([1,self.nn.h_size]))
		else:
			self.rnn_state = None

	@abstractmethod
	def get_obs(self):
		raise NotImplementedError("Must implement GenericAgent.get_obs()!")

class AgentDistilled(GenericAgent):
	def __init__(self, i_agt, dim_obs, n_actions):
		super(self.__class__, self).__init__(i_agt = i_agt, dim_obs = dim_obs, n_actions = n_actions)

	def get_obs(self):
		pass