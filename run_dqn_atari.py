import gym
import tensorflow as tf
import argparse

import dqn
import utils
import atari_wrappers
from q_functions import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str,   default='PongNoFrameskip-v4')
    parser.add_argument('--Lambda',      type=float, default=0.0)
    parser.add_argument('--history_len', type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--recurrent',   action='store_true')
    parser.add_argument('--video',       action='store_true')
    args = parser.parse_args()

    session = utils.get_session()

    env = gym.make(args.env)
    env = gym.wrappers.Monitor(env, 'videos/', force=True)
    if not args.video:
        env.video_callable = lambda e: False
    env = atari_wrappers.wrap_deepmind(env)

    seed = args.seed
    utils.set_global_seeds(seed)
    env.seed(seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)

    n_timesteps = 10000000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (1e6, 0.1), (n_timesteps, 0.05)],
                               outside_value=0.05,
                           )

    q_func = AtariRecurrentConvNet() if args.recurrent else AtariConvNet()

    dqn.learn(
        env,
        q_func,
        optimizer=optimizer,
        session=session,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        Lambda=args.Lambda,
        learning_starts=50000,
        learning_freq=4,
        history_len=args.history_len,
        target_update_freq=10000,
        grad_clip=40.,
        log_every_n_steps=250000,
    )
    env.close()


if __name__ == '__main__':
    main()
