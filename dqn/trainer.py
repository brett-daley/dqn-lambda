import tensorflow as tf
import gym
import logging
from agent import Agent
from replay_memory import ReplayMemory
from plotting import Data2DTraj, LineplotDynamic
import numpy as np
import itertools
import cv2
import os


class Trainer:
    def __init__(self, config):
        self.logger = logging.getLogger()
        self.config = config
        self.session = tf.InteractiveSession()

        self.env_name = config.get('env', 'name')
        self.env = gym.make(self.env_name)
        self.render = config.getboolean('env', 'render')
        self.obs_shape = self._preprocess(self.env.reset()).shape

        self.n_pretrain_steps = int(config.get('env', 'n_pretrain_steps'))
        self.n_explore_steps = int(config.get('env', 'n_explore_steps'))
        self.n_max_steps = int(config.get('env', 'n_max_steps'))

        self.benchmark_every_n_episodes = int(config.get('env', 'benchmark_every_n_episodes'))
        self.benchmark_for_n_episodes = int(config.get('env', 'benchmark_for_n_episodes'))

        self.train_freq = int(config.get('agent', 'train_freq'))
        self.target_q_update_freq = int(config.get('agent', 'target_q_update_freq'))

        self.epsilon_init = float(config.get('agent', 'epsilon_init'))
        self.epsilon_final = float(config.get('agent', 'epsilon_final'))
        self.epsilon_test_time = float(config.get('agent', 'epsilon_test_time'))

        self.agent = Agent(config, self.session, self.obs_shape, n_actions=self.env.action_space.n)
        self.replay_memory = ReplayMemory(config, self.obs_shape, recurrent_mode=self.agent.nn.is_recurrent())

        self.session.run(tf.global_variables_initializer())

        self.traj_predicted_disc_return = Data2DTraj()
        self.traj_actual_disc_return = Data2DTraj()
        self.traj_undisc_return = Data2DTraj()
        self.traj_mov_avg_undisc_return = Data2DTraj()

        if self.render:
            self.plot_predicted_disc_return = LineplotDynamic(title='Predicted Discounted Episode Return', label_x='Timestep', label_y='Return')
            self.plot_actual_disc_return = LineplotDynamic(title='Actual Discounted Episode Return', label_x='Timestep', label_y='Return')
            self.plot_undisc_return = LineplotDynamic(title='Undiscounted Episode Return', label_x='Timestep', label_y='Return')
            self.plot_mov_avg_undisc_return = LineplotDynamic(title='Undiscounted Episode Return (Moving Avg)', label_x='Timestep', label_y='Return')

    def train(self):
        self._benchmark(timestep=0)

        obs = self.env.reset()
        self.agent.reset_rnn_state()

        n_episodes = 0
        undisc_return = 0.
        mov_avg_undisc_return = 0.
        disc_return = 0.
        discount = 1.

        for t in range(self.n_max_steps):
            epsilon = self._calculate_epsilon(t)

            if t % 100 == 0:
                self._log_training_phase(t, epsilon)

            if t % self.target_q_update_freq == 0:
                self.agent.update_target()

            obs = self._preprocess(obs)
            self.replay_memory.save_obs(obs)

            if t < self.n_pretrain_steps:
                action = self.agent.random_policy()
            else:
                if not self.agent.nn.is_recurrent():
                    obs = self.replay_memory.append_history_to_obs(obs)
                action = self.agent.epsilon_greedy_policy(obs, epsilon)

            obs, reward, terminal, _ = self.env.step(action)

            if self.render:
                self.env.render()

            self.replay_memory.save_outcome(action, reward, terminal)

            undisc_return += reward
            disc_return += discount * reward
            discount *= self.agent.discount

            if terminal:
                n_episodes += 1

                mov_avg_undisc_return = 0.05 * undisc_return + 0.95 * mov_avg_undisc_return
                self.logger.info('Episode completed. Return: {} (discounted), {} (undiscounted), {} (undiscounted, moving avg)'.format(disc_return, undisc_return, mov_avg_undisc_return))

                self.traj_undisc_return.append(t, undisc_return, y_stdev=0)
                self.traj_mov_avg_undisc_return.append(t, mov_avg_undisc_return, y_stdev=0)

                if self.render:
                    self.plot_undisc_return.update(hl_name=None, label=None, x_new=t, y_new=undisc_return)
                    self.plot_mov_avg_undisc_return.update(hl_name=None, label=None, x_new=t, y_new=mov_avg_undisc_return)

                undisc_return = 0.
                disc_return = 0.
                discount = 1.

                if n_episodes % self.benchmark_every_n_episodes == 0:
                    self._benchmark(t)

                obs = self.env.reset()
                self.agent.reset_rnn_state()

            if t > self.n_pretrain_steps and t % self.train_freq == 0:
                self.agent.learn(self.replay_memory)

    def _benchmark(self, timestep):
        self.logger.info('BENCHMARKING PERFORMANCE')

        predicted_disc_returns = []
        actual_disc_returns = []

        replay_memory = ReplayMemory(self.config, self.obs_shape, recurrent_mode=self.agent.nn.is_recurrent(), forced_capacity=self.agent.history_length)

        for i in range(self.benchmark_for_n_episodes):
            obs = self.env.reset()
            self.agent.reset_rnn_state()

            undisc_return = 0.
            disc_return = 0.
            discount = 1.
            collected_prediction = False

            for t in itertools.count():
                obs = self._preprocess(obs)
                replay_memory.save_obs(obs)

                if not self.agent.nn.is_recurrent():
                    obs = replay_memory.append_history_to_obs(obs)

                if not collected_prediction and (self.agent.nn.is_recurrent() or t == self.agent.history_length - 1):
                    prediction = self.agent.nn.max_q.eval(feed_dict={self.agent.nn.obs_input: [obs]})
                    predicted_disc_returns.append(prediction)
                    collected_prediction = True

                action = self.agent.epsilon_greedy_policy(obs, self.epsilon_test_time)

                obs, reward, terminal, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                replay_memory.save_outcome(action, reward, terminal)

                undisc_return += reward
                disc_return += discount * reward
                discount *= self.agent.discount

                if terminal:
                    break

            self.logger.info('Episode completed. Return: {} (discounted), {} (undiscounted)'.format(disc_return, undisc_return))
            actual_disc_returns.append(disc_return)

        mean_predicted_disc_return = np.mean(predicted_disc_returns)
        std_predicted_disc_return = np.std(predicted_disc_returns)
        self.traj_predicted_disc_return.append(timestep, mean_predicted_disc_return, std_predicted_disc_return)

        mean_actual_disc_return = np.mean(actual_disc_returns)
        std_actual_disc_return = np.std(actual_disc_returns)
        self.traj_actual_disc_return.append(timestep, mean_actual_disc_return, std_actual_disc_return)

        if self.render:
            self.plot_predicted_disc_return.update(hl_name=None, label=None, x_new=timestep, y_new=mean_predicted_disc_return, y_stdev_new=std_predicted_disc_return)
            self.plot_actual_disc_return.update(hl_name=None, label=None, x_new=timestep, y_new=mean_actual_disc_return, y_stdev_new=std_actual_disc_return)

        self.logger.info('DONE BENCHMARKING PERFORMANCE')

    def _log_training_phase(self, timestep, epsilon):
        if timestep < self.n_pretrain_steps:
            phase = 'pre-train'
        elif timestep < self.n_pretrain_steps + self.n_explore_steps:
            phase = 'train (e-greedy)'
        else:
            phase = 'train (e-greedy, min epsilon reached)'

        self.logger.info('TIMESTEP {} | PHASE {} | EPSILON {}'.format(timestep, phase, epsilon))

    def _calculate_epsilon(self, timestep):
        if timestep < self.n_pretrain_steps:
            epsilon = self.epsilon_init
        elif timestep < self.n_pretrain_steps + self.n_explore_steps:
            p = (timestep - self.n_pretrain_steps) / self.n_explore_steps
            epsilon = p * self.epsilon_final + (1-p) * self.epsilon_init
        else:
            epsilon = self.epsilon_final

        return epsilon

    def _preprocess(self, obs):
        # TODO: eventually, the desired preprocessor function should be specified
        # in the config and dynamically loaded from a separate module
        if len(obs.shape) > 1:
            obs = cv2.resize(obs, (84, 84))
        return obs

    def save_results(self, data_dir):
        self.traj_predicted_disc_return.save_data(data_dir=os.path.join(data_dir, 'traj_predicted_disc_return.txt'))
        self.traj_actual_disc_return.save_data(data_dir=os.path.join(data_dir, 'traj_actual_disc_return.txt'))
        self.traj_undisc_return.save_data(data_dir=os.path.join(data_dir, 'traj_undisc_return.txt'))
        self.traj_mov_avg_undisc_return.save_data(data_dir=os.path.join(data_dir, 'traj_mov_avg_undisc_return.txt'))

        if self.render:
            self.plot_predicted_disc_return.fig.savefig(os.path.join(data_dir, 'plot_predicted_disc_return.png'), bbox_inches='tight')
            self.plot_actual_disc_return.fig.savefig(os.path.join(data_dir, 'plot_actual_disc_return.png'), bbox_inches='tight')
            self.plot_undisc_return.fig.savefig(os.path.join(data_dir, 'plot_undisc_return.png'), bbox_inches='tight')
            self.plot_mov_avg_undisc_return.fig.savefig(os.path.join(data_dir, 'plot_mov_avg_undisc_return.png'), bbox_inches='tight')

        saver = tf.train.Saver()
        saver.save(self.session, save_path=os.path.join(data_dir, 'model'))
        logging.getLogger().info('Successfully saved Tensorflow model in {}'.format(data_dir))

        writer = tf.summary.FileWriter(data_dir, self.session.graph)
        writer.close()
