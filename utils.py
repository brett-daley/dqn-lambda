"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import tensorflow as tf
import numpy as np
from subprocess import check_output


def benchmark(env, policy, epsilon, n_episodes):
    for _ in range(n_episodes):
        state = env.reset()
        rnn_state = None
        done = False
        while not done:
            action, rnn_state = policy(state, rnn_state, epsilon)
            state, _, done, _ = env.step(action)

    return get_episode_rewards(env)[-n_episodes:]


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


def huber_loss(x, delta=1.0):
    # https://en.wikipedia.org/wiki/Huber_loss
    return tf.select(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op

def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                # If using an older version of TensorFlow, uncomment the line
                # below and comment out the line after it.
		#session.run(tf.initialize_variables([v]), feed_dict)
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happend if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left
