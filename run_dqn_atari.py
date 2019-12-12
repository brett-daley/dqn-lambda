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


def intfloat(value):
    '''Allows an int argument to be formatted as a float.'''
    if float(value) != int(float(value)):
        raise argparse.ArgumentError
    return int(float(value))


def get_args():
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=32, width=132)
    parser = argparse.ArgumentParser(description='Trains DQN(lambda) on an Atari game. (https://arxiv.org/abs/1810.09967)',
                                     formatter_class=formatter)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='(int) Minibatch size for training. Default: 32')
    parser.add_argument('--block-size', type=intfloat, default=100,
                        help='(int) Refresh the cache using sequences of this length. Cannot use with --legacy. Default: 100')
    parser.add_argument('--cache-size', type=intfloat, default='80e3',
                        help='(int) Capacity of the cache. Cannot use with --legacy. Default: 80e3')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='(float) Discount factor for future rewards. Must be in [0, 1]. Default: 0.99')
    parser.add_argument('--env', type=str, default='pong',
                        help="(str) Name of Atari game to play. See README. Default: 'pong'")
    parser.add_argument('--explore-time', type=intfloat, default='1e6',
                        help='(int) Timeframe for annealing epsilon. Default: 1e6')
    parser.add_argument('--final-eps', type=float, default=0.1,
                        help='(float) Final epsilon value after annealing. Must be in [0, 1]. Default: 0.1')
    parser.add_argument('--grad-clip', type=float, default=40.0,
                        help='(float) Max magnitude for each gradient component. Default: 40.0')
    parser.add_argument('--history-len', type=int, default=4,
                        help='(int) Number of recent observations fed to Q-network. Default: 4')
    parser.add_argument('--legacy', action='store_true',
                        help='(flag) Train DQN with target network instead of DQN(lambda) with cache. Default: disabled')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='(float) Learning rate for Adam optimizer. Default: 1e-4')
    parser.add_argument('--mem-size', type=intfloat, default='1e6',
                        help='(int) Capacity of the replay memory. Default: 1e6')
    parser.add_argument('--prepopulate', type=intfloat, default='50e3',
                        help='(int) Initialize replay memory with random policy for this many timesteps. Default: 50e3')
    parser.add_argument('--priority', type=float, default=0.0,
                        help='(float) Extent to which cache samples are prioritized by TD error. Must be in [0, 1]. '
                        'High value may degrade performance. Cannot use with --legacy. Default: 0.0')
    parser.add_argument('--return-est', type=str, default='nstep-1',
                        help="(str) Estimator used to compute returns. See README. Default: 'nstep-1'")
    parser.add_argument('--seed', type=int, default=0,
                        help='(int) Seed for random number generation. Default: 0')
    parser.add_argument('--train-freq', type=int, default=4,
                        help='(int) Frequency of minibatch training. Default: 4')
    parser.add_argument('--timesteps', type=intfloat, default='10e6',
                        help='(int) Training duration in timesteps. Default: 10e6')
    parser.add_argument('--update-freq', type=intfloat, default='10e3',
                        help='(int) Frequency of cache update (or target network update, with --legacy). Default: 10e3')
    return parser.parse_args()


def main():
    args = get_args()

    env = make_atari_env(args.env, args.seed)
    benchmark_env = make_atari_env(args.env, args.seed+1)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4)

    exploration_schedule = utils.PiecewiseSchedule([
                                   (0, 1.0),
                                   (args.prepopulate, 1.0),
                                   (args.prepopulate + args.explore_time, args.final_eps),
                               ],
                               outside_value=args.final_eps,
                           )

    if not args.legacy:
        replay_memory = make_replay_memory(args.return_est, args.mem_size, args.history_len, args.discount,
                                           args.cache_size, args.block_size, args.priority)
    else:
        assert args.cache_size == 80000      # Ensure cache-related args have not been set
        assert args.priority == 0.0
        assert args.block_size == 100
        replay_memory = make_legacy_replay_memory(args.return_est, args.mem_size, args.history_len, args.discount)

    with utils.make_session(args.seed) as session:
        dqn.learn(
            session,
            env,
            benchmark_env,
            atari_cnn,
            replay_memory,
            optimizer,
            exploration_schedule,
            args.timesteps,
            args.batch_size,
            args.prepopulate,
            args.train_freq,
            args.update_freq,
            args.grad_clip,
            log_every_n_steps=1000,
        )
    env.close()


if __name__ == '__main__':
    main()
