import gym
import tensorflow as tf

import dqn
import utils
from q_functions import *


def main():
    session = utils.get_session()

    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, 'videos/', force=True)

    seed = 0
    utils.set_global_seeds(seed)
    env.seed(seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    n_timesteps = 1000000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (2e5, 0.1)],
                               outside_value=0.1,
                           )

    dqn.learn(
        env,
        q_func=CartPoleNet(),
        optimizer=optimizer,
        session=session,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=10000,
        learning_freq=4,
        history_len=1,
        target_update_freq=250,
        use_float=True,
        log_every_n_steps=25000,
    )
    env.close()


if __name__ == '__main__':
    main()
