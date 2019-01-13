import gym
import tensorflow as tf

import dqn
import utils
import atari_wrappers
from q_functions import *


def make_atari_env(name):
    from gym.wrappers.monitor import Monitor
    from gym.envs.atari.atari_env import AtariEnv

    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = Monitor(env, 'videos/', force=True, video_callable=lambda e: False)
    env = atari_wrappers.wrap_deepmind(env)
    return env


def main():
    session = utils.get_session()
    env = make_atari_env('pong')

    seed = 0
    utils.set_global_seeds(seed)
    env.seed(seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)

    n_timesteps = 10000000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (1e6, 0.1), (n_timesteps, 0.05)],
                               outside_value=0.05,
                           )

    dqn.learn(
        env,
        q_func=AtariConvNet(),
        optimizer=optimizer,
        session=session,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        history_len=4,
        target_update_freq=10000,
        grad_clip=40.,
        log_every_n_steps=250000,
    )
    env.close()


if __name__ == '__main__':
    main()
