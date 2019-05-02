import gym
import itertools
import numpy as np
import tensorflow as tf
import time

from utils import *
from wrappers import HistoryWrapper
from replay_memory import DynamicLambdaReplayMemory
from replay_memory_legacy import LegacyReplayMemory


def learn(session,
          env,
          benchmark_env,
          QFunction,
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
          mov_avg_size=100,
    ):

    assert (learning_starts % target_update_freq) == 0
    assert (target_update_freq % learning_freq) == 0
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    input_shape = (replay_memory.history_len, *env.observation_space.shape)
    n_actions = env.action_space.n
    benchmark_env = HistoryWrapper(benchmark_env, replay_memory.history_len)

    legacy_mode = isinstance(replay_memory, LegacyReplayMemory)

    # build model
    obs_t_ph  = tf.placeholder(env.observation_space.dtype, [None] + list(input_shape))
    act_t_ph  = tf.placeholder(tf.int32,   [None])
    return_ph = tf.placeholder(tf.float32, [None])

    q_func = QFunction(obs_t_ph, n_actions, scope='main')
    qvalues = q_func.qvalues
    rnn_state_tf = q_func.rnn_state if q_func.is_recurrent() else None

    greedy_actions = tf.argmax(qvalues, axis=1)
    greedy_qvalues = tf.reduce_max(qvalues, axis=1)

    action_indices = tf.stack([tf.range(tf.size(act_t_ph)), act_t_ph], axis=-1)
    onpolicy_qvalues = tf.gather_nd(qvalues, action_indices)

    td_error = return_ph - onpolicy_qvalues
    loss = tf.reduce_mean(tf.square(td_error))

    if not legacy_mode:
        def refresh(states, actions):
            assert len(states) == len(actions) + 1  # We should have an extra bootstrap state
            greedy_qvals, greedy_acts = session.run([greedy_qvalues, greedy_actions], feed_dict={
                obs_t_ph: states,
                act_t_ph: actions,
            })
            mask = (actions == greedy_acts[:-1])
            return greedy_qvals, mask
    else:
        max_target_qvalues = tf.reduce_max(QFunction(obs_t_ph, n_actions, scope='target').qvalues, axis=1)
        target_update_op = create_copy_op(src_scope='main', dst_scope='target')

        def refresh(states):
            return session.run(max_target_qvalues, feed_dict={obs_t_ph: states})

    main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
    train_op = minimize_with_grad_clipping(optimizer, loss, main_vars, grad_clip)

    replay_memory.register_refresh_func(refresh)

    # initialize variables
    session.run(tf.global_variables_initializer())

    def epsilon_greedy(obs, rnn_state, epsilon):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = session.run(greedy_actions, feed_dict={obs_t_ph: obs[None]})[0]
        return action, None

    def epsilon_greedy_rnn(obs, rnn_state, epsilon):
        feed_dict = {obs_t_ph: obs[None]}
        if rnn_state is not None:
            feed_dict[q_func.rnn_state] = rnn_state

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
            rnn_state = session.run(rnn_state_tf, feed_dict)
        else:
            action, rnn_state = session.run([greedy_actions, rnn_state_tf], feed_dict)
            action = action[0]

        return action, rnn_state

    def train():
        obs_batch, act_batch, ret_batch = replay_memory.sample(batch_size)

        session.run(train_op, feed_dict={
            obs_t_ph:  obs_batch,
            act_t_ph:  act_batch,
            return_ph: ret_batch,
        })

    best_mean_reward = -float('inf')
    obs = env.reset()
    rnn_state = None
    n_epochs = 0

    policy = epsilon_greedy_rnn if q_func.is_recurrent() else epsilon_greedy
    benchmark_rewards = benchmark(benchmark_env, policy, epsilon=1.0, n_episodes=mov_avg_size)
    start_time = time.time()

    for t in itertools.count():
        train_frac = max(0.0, (t - learning_starts) / (max_timesteps - learning_starts))

        if t % log_every_n_steps == 0:
            print('Epoch', n_epochs)
            print('Timestep', t)
            print('Realtime {:.3f}'.format(time.time() - start_time))

            rewards = (benchmark_rewards + get_episode_rewards(env))[-mov_avg_size:]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            best_mean_reward = max(mean_reward, best_mean_reward)

            print('Episodes', len(get_episode_rewards(env)))
            print('Exploration', exploration.value(t))
            if not legacy_mode:
                print('Priority', replay_memory.priority_now(train_frac))
            if isinstance(replay_memory, DynamicLambdaReplayMemory):
                lambdas = replay_memory.lambdas_since_refresh
                if len(lambdas) > 0:
                    print('Mean lambda', np.mean(lambdas))
                    print('Std. lambda', np.std(lambdas))
            print('Mean reward', mean_reward)
            print('Best mean reward', best_mean_reward)
            print('Std. reward', std_reward)
            print(flush=True)

            n_epochs += 1

        if t >= max_timesteps:
            break

        replay_memory.store_obs(obs)
        obs = replay_memory.encode_recent_observation()

        action, rnn_state = policy(obs, rnn_state, epsilon=exploration.value(t))
        obs, reward, done, _ = env.step(action)

        replay_memory.store_effect(action, reward, done)

        if done:
            obs = env.reset()
            rnn_state = None

        if t >= learning_starts:
            if not legacy_mode:
                if t % target_update_freq == 0:
                    num_train_iterations = target_update_freq // learning_freq
                    cache_size = batch_size * num_train_iterations

                    replay_memory.refresh(cache_size, train_frac)

                    for _ in range(num_train_iterations):
                        train()
            else:
                if t % target_update_freq == 0:
                    session.run(target_update_op)

                if t % learning_freq == 0:
                    train()

    all_rewards = benchmark_rewards + get_episode_rewards(env)
    print('rewards=', all_rewards, sep='')
