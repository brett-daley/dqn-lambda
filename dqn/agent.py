import numpy as np
import random


class Agent:
    def __init__(self, config, session, obs_shape, n_actions):
        self.session = session
        self.n_actions = n_actions

        self.nn_arch = config.get('agent', 'nn_arch')
        self.discount = float(config.get('agent', 'discount'))  # TODO: this can be deleted once target calculation is moved into the neural net
        self.history_length = int(config.get('agent', 'history_length'))
        self.batch_size = int(config.get('replay_memory', 'batch_size'))

        DQN = getattr(__import__('neural_networks'), self.nn_arch)
        self.nn = DQN(config, session, 'nn_predict', obs_shape, n_actions)
        self.target_nn = DQN(config, session, 'nn_target', obs_shape, n_actions, src_network=self.nn)

    def random_policy(self):
        return self._get_random_action()

    def epsilon_greedy_policy(self, obs, epsilon):
        return self._get_random_action() if random.random() < epsilon else self._get_greedy_action(obs)

    def _get_random_action(self):
        return random.randrange(self.n_actions)

    def _get_greedy_action(self, obs):
        return np.argmax(self._get_qvalues(obs))

    def _get_qvalues(self, obs):
        feed_dict = {self.nn.obs_input: [obs]}

        if self.nn.is_recurrent():
            feed_dict[self.nn.init_state] = self.rnn_state
            qvalues, self.rnn_state = self.session.run([self.nn.qvalues, self.nn.state], feed_dict=feed_dict)
        else:
            qvalues = self.session.run(self.nn.qvalues, feed_dict=feed_dict)

        return qvalues

    def learn(self, replay_memory):
        obs_batch, action_batch, reward_batch, next_obs_batch, non_terminal_multiplier = replay_memory.sample_minibatch()

        # TODO: can this be combined into the trainstep below?
        feed_dict = {
            self.target_nn.obs_input: next_obs_batch,
            self.target_nn.tracelength: self.history_length,
            self.target_nn.batch_size: self.batch_size,
        }

        target_max_q = self.target_nn.max_q.eval(feed_dict=feed_dict)
        target_batch = reward_batch + (self.discount * target_max_q * non_terminal_multiplier)

        feed_dict = {
            self.nn.target_input: target_batch,
            self.nn.action_input: action_batch,
            self.nn.obs_input: obs_batch,
            self.nn.tracelength: self.history_length,
            self.nn.batch_size: self.batch_size,
        }

        self.nn.train_op.run(feed_dict=feed_dict)

    def update_target(self):
        self.target_nn.copy()

    def reset_rnn_state(self):
        if self.nn.is_recurrent():
            self.rnn_state = (self.nn.init_state[0].eval(), self.nn.init_state[1].eval())
