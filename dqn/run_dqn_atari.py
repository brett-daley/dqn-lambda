import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *


class atari_recurrent:
    def is_recurrent(self):
        return True

    def __call__(self, img_in, num_actions, scope, reuse=False):
        out = tf.unstack(img_in, axis=1)
        out = tf.concat(out, axis=0)
        print('Recurrent', img_in.shape)

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("convnet"):
                # original architecture
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

            out = tf.reshape(out, [-1, tf.shape(img_in)[1], tf.size(out[0])])

            with tf.variable_scope("action_value"):
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
                init_state = cell.zero_state(tf.shape(img_in)[0], tf.float32)

                out, state = tf.nn.dynamic_rnn(cell, inputs=out, initial_state=init_state, dtype=tf.float32)

                out = out[:, -1]
                out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out

class atari_feedforward:
    def is_recurrent(self):
        return False

    def __call__(self, img_in, num_actions, scope, reuse=False):
        out = tf.unstack(img_in, axis=1)
        out = tf.concat(out, axis=-1)
        print('Feedforward', out.shape)

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("convnet"):
                # original architecture
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

            out = layers.flatten(out)

            with tf.variable_scope("action_value"):
                out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out

def atari_learn(env,
                session,
                n_timesteps):

    lr_schedule = PiecewiseSchedule([
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

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (n_timesteps / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_feedforward(),
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    env = wrappers.Monitor(env, 'videos/', force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari200M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, n_timesteps=5000000)

if __name__ == "__main__":
    main()
