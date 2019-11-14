import gym
import tensorflow as tf
import numpy as np
from subprocess import check_output


def benchmark(env, policy, epsilon, n_episodes):
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state, epsilon)
            state, _, done, _ = env.step(action)

    return list(get_episode_rewards(env)[-n_episodes:])


def get_episode_rewards(env):
    if isinstance(env, gym.wrappers.Monitor):
        return env.get_episode_rewards()
    elif hasattr(env, 'env'):
        return get_episode_rewards(env.env)
    raise ValueError('No Monitor wrapper around env')


def get_available_gpus():
    # Calling nvidia-smi does not allocate memory on the GPUs
    try:
        output = check_output(['nvidia-smi', '-L']).decode()
        return [gpu for gpu in output.split('\n') if 'UUID' in gpu]
    except FileNotFoundError:
        return []


def make_session(seed):
    print('AVAILABLE GPUS:', get_available_gpus(), flush=True)
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)
    session = tf.Session(config=config)

    np.random.seed(seed)
    tf.set_random_seed(seed)  # Must be called after graph creation or results will be non-deterministic

    return session


def minimize_with_grad_clipping(optimizer, loss, var_list, clip):
    grads_and_vars = optimizer.compute_gradients(loss, var_list)
    if clip is not None:
        grads_and_vars = [(tf.clip_by_value(g, -clip, +clip), v) for g, v in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op


def create_copy_op(src_scope, dst_scope):
    src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=src_scope)
    dst_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst_scope)
    assert len(src_vars) == len(dst_vars)

    src_vars = sorted(src_vars, key=lambda v: v.name)
    dst_vars = sorted(dst_vars, key=lambda v: v.name)
    return tf.group(*[dst.assign(src) for src, dst in zip(src_vars, dst_vars)])


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule:
    def __init__(self, endpoints, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = linear_interpolation
        self._outside_value = outside_value
        self._endpoints     = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value
