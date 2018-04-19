"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import tensorflow as tf
import numpy as np
import random


def random_baseline(env, n_episodes):
    for i in range(n_episodes):
        done = False

        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)

        env.reset()

    env = get_wrapper_by_name(env, 'Monitor')
    return env.get_episode_rewards()[-n_episodes:]


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
                    intra_op_parallelism_threads=1,
                )
    session = tf.Session(config=tf_config)
    print('AVAILABLE GPUS: ', get_available_gpus())
    return session


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


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)


class Episode(object):
    def __init__(self, history_len, discount, Lambda, refresh_func):
        self.finished = False
        self.history_len = history_len

        self.discount = discount
        self.Lambda = Lambda

        self.refresh_func = refresh_func

        self.obs    = []
        self.action = []
        self.reward = []
        self.lambda_return = None

        self.length = 0

    def store_frame(self, obs):
        self.obs.append(obs)
        self.length += 1

    def store_effect(self, action, reward):
        self.action.append(action)
        self.reward.append(reward)

    def finish(self):
        assert not self.finished
        self.finished = True

        self.obs = np.concatenate([obs[None] for obs in self.obs], 0)
        self.action = np.array(self.action)
        self.reward = np.array(self.reward)

        self.refresh()

    def _calc_lambda_return(self, qvalues):
        lambda_return = self.reward + (self.discount * qvalues)
        for i in reversed(range(self.length - 1)):
            lambda_return[i] += (self.discount * self.Lambda) * (lambda_return[i+1] - qvalues[i])
        return lambda_return

    def sample(self):
        i = random.randrange(self.length)
        return (self._encode_observation(i), self.action[i], self.lambda_return[i])

    def refresh(self):
        obs = np.array([self._encode_observation(i) for i in range(self.length)])

        qvalues = self.refresh_func(obs)
        qvalues = np.pad(qvalues[1:], pad_width=(0,1), mode='constant')

        self.lambda_return = self._calc_lambda_return(qvalues)

    def _encode_observation(self, idx):
        end = (idx % self.length) + 1 # make noninclusive
        start = end - self.history_len

        pad_len = max(0, -start)
        padding = [np.zeros_like(self.obs[0]) for _ in range(pad_len)]

        start = max(0, start)
        obs = [x for x in self.obs[start:end]]

        return np.array(padding + obs)


class ReplayBuffer(object):
    def __init__(self, size, history_len, discount, Lambda, refresh_func):
        self.size = size
        self.history_len = history_len

        self.discount = discount
        self.Lambda = Lambda

        self.refresh_func = refresh_func

        self.episodes = []
        self.current_episode = None

    def can_sample(self):
        return len(self.episodes) > 0

    def _encode_sample(self, idxes):
        samples = [self.episodes[i].sample() for i in idxes]
        obs_batch, act_batch, rew_batch = zip(*samples)

        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch)

    def sample(self, batch_size):
        assert self.can_sample()
        idxes = [random.randrange(len(self.episodes)) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        assert self.current_episode is not None
        if self.current_episode.length > 0:
            return self.current_episode._encode_observation(-1)
        else:
            return self.episodes[-1]._encode_observation(-1)

    def store_frame(self, frame):
        if self.current_episode is None:
            self.current_episode = self._new_episode()
        self.current_episode.store_frame(frame)

    def store_effect(self, action, reward, done):
        self.current_episode.store_effect(action, reward)

        if done:
            self._move_episode_to_buffer()

    def _move_episode_to_buffer(self):
        self.current_episode.finish()

        while sum([e.length for e in self.episodes]) > self.size:
            self.episodes.pop(0)

        self.episodes.append(self.current_episode)
        self.current_episode = self._new_episode()

    def refresh(self):
        for e in self.episodes:
            e.refresh()

    def _new_episode(self):
        return Episode(self.history_len, self.discount, self.Lambda, self.refresh_func)
