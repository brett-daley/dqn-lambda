import tensorflow as tf
import argparse

import dqn
import utils
from wrappers import monitor, wrap_deepmind
from q_functions import *


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
    parser.add_argument('--oversample',  type=float, default=1.0)
    parser.add_argument('--priority',    type=float, default=0.0)
    parser.add_argument('--chunk-size',  type=int, default=100)
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--recurrent',   action='store_true')
    parser.add_argument('--legacy',      action='store_true')
    return parser.parse_args()


def main():
    args = get_args()

    env = make_atari_env(args.env, args.seed)
    benchmark_env = make_atari_env(args.env, args.seed+1)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)

    n_timesteps = 10000000
    learning_starts = 50000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (learning_starts, 1.0), (learning_starts + 1e6, 0.1)],
                               outside_value=0.1,
                           )

    if not args.legacy:
        from replay_memory import make_replay_memory
        replay_memory = make_replay_memory(args.return_type, args.history_len, size=1000000, discount=0.99)
        replay_memory.config_cache(args.oversample, args.priority, args.chunk_size)
    else:
        assert args.oversample == 1.0        # Ensure cache-related args have not been set
        assert args.priority == 0.0
        assert args.chunk_size == 100
        assert 'nstep-' in args.return_type  # Only n-step returns are supported

        from replay_memory_legacy import LegacyReplayMemory
        n = int( args.return_type.strip('nstep-') )
        replay_memory = LegacyReplayMemory(size=1000000, history_len=args.history_len, discount=0.99, n=n)

    with utils.make_session(args.seed) as session:
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
