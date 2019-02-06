import gym
import tensorflow as tf

import dqn
import utils
from q_functions import *
from replay_memory import NStepReplayMemory


def main():
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, 'videos/', force=True, video_callable=lambda e: False)

    seed = 0
    utils.set_global_seeds(seed)
    env.seed(seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    n_timesteps = 1000000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (2e5, 0.1)],
                               outside_value=0.1,
                           )

    replay_memory = NStepReplayMemory(
                        size=1000000,
                        history_len=1,
                        discount=0.99,
                        nsteps=1,
                    )

    dqn.learn(
        env,
        CartPoleNet(),
        replay_memory,
        optimizer=optimizer,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        batch_size=32,
        learning_starts=10000,
        learning_freq=4,
        target_update_freq=250,
        log_every_n_steps=25000,
    )
    env.close()


if __name__ == '__main__':
    main()
