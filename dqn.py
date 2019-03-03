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
          replay_memory,
          optimizer,
          exploration=LinearSchedule(1000000, 0.1),
          max_timesteps=50000000,
          batch_size=32,
          learning_starts=50000,
          learning_freq=4,
          target_update_freq=10000,
          grad_clip=None,
          log_every_n_steps=100000,
    ):

    assert (learning_starts % target_update_freq) == 0
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    input_shape = (replay_memory.history_len, *env.observation_space.shape)
    n_actions = env.action_space.n

    # build model
    session = get_session()

    obs_t_ph  = tf.placeholder(env.observation_space.dtype, [None] + list(input_shape))
    act_t_ph  = tf.placeholder(tf.int32,   [None])
    return_ph = tf.placeholder(tf.float32, [None])

    qvalues, rnn_state_tf = q_func(obs_t_ph, n_actions, scope='q_func')
    greedy_action = tf.argmax(qvalues, axis=1)

    action_indices = tf.stack([tf.range(tf.size(act_t_ph)), act_t_ph], axis=-1)
    onpolicy_qvalues = tf.gather_nd(qvalues, action_indices)

    td_error = return_ph - onpolicy_qvalues
    total_error = tf.reduce_mean(tf.square(td_error))

    # compute and clip gradients
    grads_and_vars = optimizer.compute_gradients(total_error, var_list=tf.trainable_variables(scope='q_func'))
    if grad_clip is not None:
        grads_and_vars = [(tf.clip_by_value(g, -grad_clip, +grad_clip), v) for g, v in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)

    def refresh(states, actions):
        onpolicy_qvals, greedy = session.run([onpolicy_qvalues, greedy_action], feed_dict={
            obs_t_ph: states,
            act_t_ph: actions,
        })
        mask = (actions == greedy)
        return onpolicy_qvals, mask

    replay_memory.register_refresh_func(refresh)

    # initialize variables
    session.run(tf.global_variables_initializer())

    def epsilon_greedy(obs, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        return session.run(greedy_action, feed_dict={obs_t_ph: obs[None]})[0]

    def epsilon_greedy_rnn(obs, rnn_state, epsilon):
        feed_dict = {obs_t_ph: obs[None]}
        if rnn_state is not None:
            feed_dict[q_func.rnn_state] = rnn_state

        if random.random() < epsilon:
            action = env.action_space.sample()
            rnn_state = session.run(rnn_state_tf, feed_dict)
        else:
            action, rnn_state = session.run([greedy_action, rnn_state_tf], feed_dict)
            action = action[0]

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
                rewards = get_episode_rewards(env)[epoch_begin:]
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

        replay_memory.store_frame(obs)
        obs = replay_memory.encode_recent_observation()

        epsilon = exploration.value(t)
        if q_func.is_recurrent():
            action, rnn_state = epsilon_greedy_rnn(obs, rnn_state, epsilon)
        else:
            action = epsilon_greedy(obs, epsilon)

        obs, reward, done, _ = env.step(action)
        replay_memory.store_effect(action, reward, done)

        if done:
            obs = env.reset()
            rnn_state = None

        if t >= learning_starts:
            if t % target_update_freq == 0:
                replay_memory.refresh()

            if t % learning_freq == 0:
                obs_batch, act_batch, ret_batch = replay_memory.sample(batch_size)

                session.run(train_op, feed_dict={
                    obs_t_ph:  obs_batch,
                    act_t_ph:  act_batch,
                    return_ph: ret_batch,
                })
