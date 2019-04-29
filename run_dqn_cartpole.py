import gym
import tensorflow as tf

import dqn
import utils
from wrappers import monitor
from q_functions import *
from replay_memory import make_replay_memory


def make_continuouscontrol_env(name, seed):
    env = gym.make(name)
    env = monitor(env, name)
    env.seed(seed)
    return env


def main():
    seed = 0
    utils.set_global_seeds(seed)

    name = 'CartPole-v0'
    env = make_continuouscontrol_env(name, seed)
    benchmark_env = make_continuouscontrol_env(name, seed+1)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    n_timesteps = 500000
    learning_starts = 50000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (learning_starts, 1.0), (learning_starts + 3e5, 0.1)],
                               outside_value=0.1,
                           )

    replay_memory = make_replay_memory(return_type='nstep-1', history_len=1, size=50000, discount=0.99)

    with utils.make_session() as session:
        dqn.learn(
            session,
            env,
            benchmark_env,
            CartPoleNet,
            replay_memory,
            optimizer=optimizer,
            exploration=exploration_schedule,
            max_timesteps=n_timesteps,
            batch_size=32,
            learning_starts=learning_starts,
            learning_freq=4,
            target_update_freq=10000,
            log_every_n_steps=10000,
        )
    env.close()


if __name__ == '__main__':
    main()
