import gym
import itertools
import numpy as np
import random
import tensorflow as tf
import time

from utils import *
from atari_wrappers import *


def learn(env,
          q_func,
          optimizer,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          max_timesteps=50000000,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          history_len=4,
          target_update_freq=10000,
          grad_clip=None,
          use_float=False,
          log_every_n_steps=100000,
    ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (history_len, img_h, img_w, img_c)

    n_actions = env.action_space.n

    # build model
    obs_dtype     = tf.float32 if use_float else tf.uint8

    obs_t_ph      = tf.placeholder(obs_dtype,  [None] + list(input_shape))
    act_t_ph      = tf.placeholder(tf.int32,   [None])
    rew_t_ph      = tf.placeholder(tf.float32, [None])
    obs_tp1_ph    = tf.placeholder(obs_dtype,  [None] + list(input_shape))
    done_mask_ph  = tf.placeholder(tf.float32, [None])

    qvalues, rnn_state_tf = q_func(obs_t_ph, n_actions, scope='q_func')
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

    target_qvalues, _ = q_func(obs_tp1_ph, n_actions, scope='target_q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    action_indices = tf.stack([tf.range(tf.size(act_t_ph)), act_t_ph], axis=-1)
    onpolicy_qvalues = tf.gather_nd(qvalues, action_indices)

    targets = tf.reduce_max(target_qvalues, axis=-1)

    done_td_error = rew_t_ph - onpolicy_qvalues
    not_done_td_error = done_td_error + (gamma * targets)

    td_error = tf.where(tf.cast(done_mask_ph, tf.bool), x=done_td_error, y=not_done_td_error)
    total_error = tf.reduce_mean(tf.square(td_error))

    # compute and clip gradients
    grads_and_vars = optimizer.compute_gradients(total_error, var_list=q_func_vars)
    if grad_clip is not None:
        grads_and_vars = [(tf.clip_by_value(g, -grad_clip, +grad_clip), v) for g, v in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, history_len, use_float)

    # initialize variables
    session.run(tf.global_variables_initializer())

    def epsilon_greedy(obs, rnn_state, epsilon):
        if q_func.is_recurrent():
            feed_dict = {obs_t_ph: obs[None]}

            if rnn_state is not None:
                feed_dict[q_func.rnn_state] = rnn_state

            q, rnn_state = session.run([qvalues, rnn_state_tf], feed_dict)

        else:
            q = session.run(qvalues, feed_dict={obs_t_ph: obs[None]})

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q)

        return action, rnn_state

    best_mean_reward = -float('inf')
    obs = env.reset()
    rnn_state = None
    n_epochs = 0
    epoch_begin = 0
    start_time = time.time()

    for t in itertools.count():
        if t % log_every_n_steps == 0:
            print('Epoch', n_epochs)
            print('Timestep', t)
            print('Realtime {:.3f}'.format(time.time() - start_time))

            if n_epochs == 0:
                rewards = random_baseline(env, n_episodes=100)
                start_episode = len(rewards)
                print('Episodes', 0)
            else:
                rewards = get_wrapper_by_name(env, 'Monitor').get_episode_rewards()[epoch_begin:]
                epoch_begin += len(rewards)
                print('Episodes', epoch_begin - start_episode)

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            best_mean_reward = max(mean_reward, best_mean_reward)

            print('Exploration', exploration.value(t))
            print('Mean reward', mean_reward)
            print('Best mean reward', best_mean_reward)
            print('Standard dev', std_reward)
            print(flush=True)

            n_epochs += 1

        if t >= max_timesteps:
            break

        if t % target_update_freq == 0:
            session.run(update_target_fn)

        idx = replay_buffer.store_frame(obs)
        obs = replay_buffer.encode_recent_observation()

        epsilon = exploration.value(t)
        action, rnn_state = epsilon_greedy(obs, rnn_state, epsilon)

        obs, reward, done, _ = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)

        if done:
            obs = env.reset()
            rnn_state = None

        if (t >= learning_starts and t % learning_freq == 0):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            session.run(train_op, feed_dict= {
                obs_t_ph: obs_batch,
                act_t_ph: act_batch,
                rew_t_ph: rew_batch,
                obs_tp1_ph: next_obs_batch,
                done_mask_ph: done_mask,
            })
