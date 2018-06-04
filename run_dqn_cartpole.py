import gym
import tensorflow as tf

import dqn
import utils
from q_functions import *


def main():
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, 'videos/', force=True)

    seed = 0
    utils.set_global_seeds(seed)
    env.seed(seed)

    session = utils.get_session()

    n_timesteps = 1000000
    lr_schedule = utils.PiecewiseSchedule([
                                         (0,                   1e-4),
                                         (n_timesteps / 10, 1e-4),
                                         (n_timesteps / 2,  5e-5),
                                    ],
                                    outside_value=5e-5)

    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    exploration_schedule = utils.PiecewiseSchedule(
        [
            (0, 1.0),
            (n_timesteps / 5, 0.1),
            (n_timesteps / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=CartPoleNet(),
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        replay_buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        learning_starts=10000,
        learning_freq=4,
        frame_history_len=1,
        target_update_freq=2500,
        grad_norm_clipping=10,
        use_float=True,
        log_every_n_steps=10000,
    )
    env.close()


if __name__ == '__main__':
    main()
