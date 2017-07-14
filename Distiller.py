import numpy as np
from LineplotDynamic import LineplotDynamic
from GenericAgent import AgentDistilled
import time
# import random

class Distiller:
	def __init__(self, cfg_parser, n_actions, dim_obs_agts, sess):
		self.cfg_parser = cfg_parser
		self.n_actions = n_actions
		self.sess = sess
		self.minibatch_size = int(self.cfg_parser.get('root','minibatch_size'))
		self.tracelength = int(self.cfg_parser.get('root','rnn_train_tracelength'))

		self.timeStep = 0
	
		# Extract the observation size for each agent (this allows agents to have heterogeneous/different length observation inputs)
		# TODO 2017 it is likely that elsewhere in the code ,mayeb during teacher training phase, that a single agent's observations are being used for dim obs. double check and make sure this same agent-specific observation dimension is being done (e.g., for multitask domain where one agent is blind and the other sees additional things)
		self.dim_obs_agts = dim_obs_agts
		self.init_distilled_agts()

		# Plotting
		self.plot_dist_cost = LineplotDynamic('iter','Distillation Cost','')
		self.plot_value_dqn = LineplotDynamic('iter','R','KL Distiller', adjust_right = 0.73)
		# self.plot_value_dqn.update(x_new = 0, y_new = 0, y_stdev_new = 0) # Initialize reward plot

		# Memory used either by MDRQN for learning a joint policy, or distillation for an exploration baseline
		# if self.is_double_distiller:
		# 	self.tracelength = int(self.cfg_parser.get('root','rnn_train_tracelength'))
		# 	# self.replay_memory = ReplayMemory(n_trajs_max = 200, minibatch_size = self.minibatch_size)
		# 	self.replay_memory_double_distiller = ReplayMemory(n_trajs_max = 200, minibatch_size = self.minibatch_size, is_distillation_mem = True)

	def init_distilled_agts(self):
		self.agt = AgentDistilled(i_agt=0, n_actions=self.n_actions, dim_obs=self.dim_obs_agts[0])

		# Initialize agent NNs (need the game initialized at this point, since will need things like observation dimensions well-defined)
		self.create_agts_nns()


	def reset_agts_rnn_states(self):
		self.agt.reset_rnn_state()

	def create_agts_nns(self):
		# if self.is_double_distiller:
		# 	scope_suffix = '_distiller_mse'
		# else:
		scope_suffix = '_distiller'

		# Create actual player NNs
		self.parameter_sharing = False # each agent has its own distilled net for now
		self.agt.create_nns(cfg_parser=self.cfg_parser, sess=self.sess, scope_suffix=scope_suffix, parameter_sharing=self.parameter_sharing, is_distillation_net=True)


	def update_distillation_Q_plot(self, iter, teacher):
		truetracelengths, minibatch = teacher.replay_memory_distillation.sample_trace(tracelength = 1)
		
		agt = self.agt
		s_batch = np.vstack([row[0] for row in minibatch[:,0]])
		q_teacher_batch = np.vstack([row[0] for row in minibatch[:,5]])

		# s_batch = [data[0] for data in minibatch]
		# q_teacher_batch = [data[1] for data in minibatch]
		self.timeStep += 1
		x = self.timeStep
		# y = np.mean(self.nn.QValue.eval(feed_dict={self.nn.stateInput : s_batch }))

		y = agt.nn.cost.eval(feed_dict={agt.nn.stateInput : s_batch,
										agt.nn.Q_target : q_teacher_batch,
										agt.nn.tracelength: 1,
										agt.nn.truetracelengths: [1]*self.minibatch_size,
										agt.nn.batch_size: self.minibatch_size })

		self.plot_dist_cost.update(hl_name = 'distill_cost', x_new = x, y_new = y)

	def run_train_step(self, agt, s_batch, q_batch, truetracelengths):
		# Train
		feed_dict={agt.nn.stateInput : s_batch,
					agt.nn.Q_target : q_batch,
					agt.nn.tracelength: self.tracelength,
					agt.nn.truetracelengths: truetracelengths,
					agt.nn.batch_size: self.minibatch_size}

		if agt.nn.is_rnn:
			rnn_state_train = (np.zeros([self.minibatch_size,agt.nn.h_size]),np.zeros([self.minibatch_size,agt.nn.h_size])) 
			feed_dict[agt.nn.rnn_state_in] = rnn_state_train

		agt.nn.trainStepDistill.run(feed_dict=feed_dict)

	def train_distillation_Q_network(self, teacher, distiller_companion = None):
		# TODO 2017 make sure truetracelengths are being used correctly here
		# self.tracelength = 2
		if self.agts[0].nn.is_rnn:
			truetracelengths, minibatch = teacher.replay_memory_distillation.sample_trace(tracelength = self.tracelength)
		else:
			truetracelengths, minibatch = teacher.replay_memory_distillation.sample_trace(tracelength = 1)

		for agt in self.agts:
			s_batch = np.vstack([row[agt.i] for row in minibatch[:,0]]) 
			q_batch = np.vstack([row[agt.i] for row in minibatch[:,5]])

			self.run_train_step(agt = agt, s_batch = s_batch, q_batch = q_batch, truetracelengths = truetracelengths)

			# Also train the companion network using exact same traces
			if distiller_companion is not None:
				self.run_train_step(agt = distiller_companion.agts[agt.i], s_batch = s_batch, q_batch = q_batch, truetracelengths = truetracelengths)
