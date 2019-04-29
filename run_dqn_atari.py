import gym
import tensorflow as tf
import argparse

import dqn
import utils
from wrappers import monitor, wrap_deepmind
from q_functions import *
from replay_memory import make_replay_memory


def make_atari_env(name, seed):
    from gym.envs.atari.atari_env import AtariEnv
    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = monitor(env, name)
    env = wrap_deepmind(env)
    env.seed(seed)
    return env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str, default='pong')
    parser.add_argument('--return-type', type=str, default='nstep-1')
    parser.add_argument('--history-len', type=int, default=4)
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--recurrent',   action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    utils.set_global_seeds(args.seed)

    env = make_atari_env(args.env, args.seed)
    benchmark_env = make_atari_env(args.env, args.seed+1)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)

    n_timesteps = 10000000
    learning_starts = 50000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (learning_starts, 1.0), (learning_starts + 1e6, 0.1)],
                               outside_value=0.1,
                           )

    replay_memory = make_replay_memory(args.return_type, args.history_len, size=1000000, discount=0.99)

    with utils.make_session() as session:
        dqn.learn(
            session,
            env,
            benchmark_env,
            AtariRecurrentConvNet if args.recurrent else AtariConvNet,
            replay_memory,
            optimizer=optimizer,
            exploration=exploration_schedule,
            max_timesteps=n_timesteps,
            batch_size=32,
            learning_starts=learning_starts,
            learning_freq=4,
            target_update_freq=10000,
            grad_clip=40.,
            log_every_n_steps=1000,
        )
    env.close()


if __name__ == '__main__':
    main()
