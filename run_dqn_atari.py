import tensorflow as tf
import argparse

import dqn
import utils
from wrappers import monitor, wrap_deepmind
from q_functions import *
from replay_memory import make_replay_memory
from replay_memory_legacy import make_legacy_replay_memory


def make_atari_env(name, seed):
    from gym.envs.atari.atari_env import AtariEnv
    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = monitor(env, name)
    env = wrap_deepmind(env)
    env.seed(seed)
    return env


def get_args():
    parser = argparse.ArgumentParser(description='Trains DQN(lambda) on an Atari game. (https://arxiv.org/abs/1810.09967)')
    parser.add_argument('--env', type=str, default='pong',
                        help="(str) Name of Atari game to play. See README. Default: 'pong'")
    parser.add_argument('--timesteps', type=float, default=10e6,
                        help='(float) Training duration in timesteps. Default: 10e6')
    parser.add_argument('--return-est', type=str, default='nstep-1',
                        help="(str) Estimator used to compute returns. See README. Default: 'nstep-1'")
    parser.add_argument('--history-len', type=int, default=4,
                        help='(int) Number of recent observations fed to Q-network. Default: 4')
    parser.add_argument('--cache-size', type=float, default=80e3,
                        help='(float) Number of samples in the cache. Cannot use with --legacy. Default: 80e3')
    parser.add_argument('--block-size', type=int, default=100,
                        help='(int) Refresh the cache using sequences of this length. Cannot use with --legacy. Default: 100')
    parser.add_argument('--priority', type=float, default=0.0,
                        help='(float) Extent to which cache samples are prioritized by TD error. Must be in [0.0, 1.0]. '
                        'High value may degrade performance. Cannot use with --legacy. Default: 0.0')
    parser.add_argument('--seed', type=int, default=0,
                        help='(int) Seed for random number generation. Default: 0')
    parser.add_argument('--legacy', action='store_true',
                        help='If present, train DQN with target network instead of DQN(lambda) with cache.')
    return parser.parse_args()


def main():
    args = get_args()

    env = make_atari_env(args.env, args.seed)
    benchmark_env = make_atari_env(args.env, args.seed+1)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)

    learning_starts = 50000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (learning_starts, 1.0), (learning_starts + 1e6, 0.1)],
                               outside_value=0.1,
                           )
    discount = 0.99
    replay_mem_size = 1000000

    if not args.legacy:
        replay_memory = make_replay_memory(args.return_est, replay_mem_size, args.history_len, discount,
                                           args.cache_size, args.block_size, args.priority)
    else:
        assert args.cache_size == 80000      # Ensure cache-related args have not been set
        assert args.priority == 0.0
        assert args.block_size == 100
        replay_memory = make_legacy_replay_memory(args.return_est, replay_mem_size, args.history_len, discount)

    with utils.make_session(args.seed) as session:
        dqn.learn(
            session,
            env,
            benchmark_env,
            atari_cnn,
            replay_memory,
            optimizer=optimizer,
            exploration=exploration_schedule,
            max_timesteps=args.timesteps,
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
