import gym
import tensorflow as tf

import dqn
import utils
from wrappers import monitor
from q_functions import *
from replay_memory import make_replay_memory


def make_gym_env(name, seed):
    env = gym.make(name)
    env = monitor(env, name)
    env.seed(seed)
    return env


def main():
    seed = 0
    name = 'CartPole-v0'
    env = make_gym_env(name, seed)
    benchmark_env = make_gym_env(name, seed+1)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    prepopulate = 50000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (prepopulate, 1.0), (prepopulate + 3e5, 0.1)],
                               outside_value=0.1,
                           )

    replay_memory = make_replay_memory(return_est='nstep-5', capacity=500000, history_len=1, discount=0.99,
                                       cache_size=80000, block_size=100, priority=0.0)

    with utils.make_session(seed) as session:
        dqn.learn(
            session,
            env,
            benchmark_env,
            cartpole_mlp,
            replay_memory,
            optimizer=optimizer,
            exploration=exploration_schedule,
            max_timesteps=500000,
            batch_size=32,
            prepopulate=prepopulate,
            target_update_freq=10000,
            train_freq=4,
            log_every_n_steps=10000,
        )
    env.close()


if __name__ == '__main__':
    main()
