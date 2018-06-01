import gym
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import namedtuple
import time
from dqn_utils import *
from atari_wrappers import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          max_timesteps=50000000,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
    ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (frame_history_len, img_h, img_w, img_c)

    num_actions = env.action_space.n

    # build model
    obs_t_ph      = tf.placeholder(tf.uint8,   [None] + list(input_shape))
    act_t_ph      = tf.placeholder(tf.int32,   [None])
    rew_t_ph      = tf.placeholder(tf.float32, [None])
    obs_tp1_ph    = tf.placeholder(tf.uint8,   [None] + list(input_shape))
    done_mask_ph  = tf.placeholder(tf.float32, [None])

    # TODO: move into policies
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
    
    qvalues, rnn_state_tf = q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

    target_qvalues, _ = q_func(obs_tp1_float, num_actions, scope='target_q_func', reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    indices = tf.stack([tf.range(tf.size(act_t_ph)), act_t_ph], axis=-1)
    q = tf.gather_nd(qvalues, indices)

    targets = tf.reduce_max(target_qvalues, axis=-1)

    done_td_error = rew_t_ph - q
    not_done_td_error = done_td_error + (gamma * targets)

    td_error = tf.where(tf.cast(done_mask_ph, tf.bool), x=done_td_error, y=not_done_td_error)
    total_error = tf.reduce_sum(tf.square(td_error))

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # for benchmarking
    bm_replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    bm_env = gym.make(env.spec.id)
    bm_env = gym.wrappers.Monitor(bm_env, 'videos/', force=True, video_callable=lambda e: False)
    bm_env = wrap_deepmind(bm_env)
    bm_env.seed(0)

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

    def benchmark(n_episodes):
        for i in range(n_episodes):
            obs = bm_env.reset()
            rnn_state = None
            done = False

            while not done:
                idx = bm_replay_buffer.store_frame(obs)
                obs = bm_replay_buffer.encode_recent_observation()

                action, rnn_state = epsilon_greedy(obs, rnn_state, epsilon=0.05)

                obs, reward, done, _ = bm_env.step(action)
                bm_replay_buffer.store_effect(idx, action, reward, done)

        rewards = get_wrapper_by_name(bm_env, 'Monitor').get_episode_rewards()[-n_episodes:]

        return np.mean(rewards), np.std(rewards)

    best_mean_reward = -float('inf')
    obs = env.reset()
    rnn_state = None
    n_epochs = 0
    LOG_EVERY_N_STEPS = 25000
    start_time = time.time()

    for t in itertools.count():
        if t % LOG_EVERY_N_STEPS == 0:
            print('Epoch', n_epochs)
            print('Timestep', t)
            print('Realtime {:.3f}'.format(time.time() - start_time))
            print('Episodes', len(get_wrapper_by_name(env, 'Monitor').get_episode_rewards()))
            print('Exploration', exploration.value(t))
            print('Learning rate', optimizer_spec.lr_schedule.value(t))

            mean_reward, std_reward = benchmark(n_episodes=30)
            best_mean_reward = max(mean_reward, best_mean_reward)

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

        if (t >= learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            feed_dict = {
                obs_t_ph: obs_batch,
                act_t_ph: act_batch,
                rew_t_ph: rew_batch,
                obs_tp1_ph: next_obs_batch,
                done_mask_ph: done_mask,
                learning_rate: optimizer_spec.lr_schedule.value(t),
            }

            session.run(train_fn, feed_dict=feed_dict)
