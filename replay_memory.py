import numpy as np
import re

from return_calculation import calculate_lambda_returns, calculate_nstep_returns


def make_replay_memory(return_type, capacity, history_len, discount, cache_size, chunk_size, priority):
    shared_args = (capacity, history_len, discount, cache_size, chunk_size, priority)
    int_capture = r'([0-9]+)'
    float_capture = r'([0-9]+\.[0-9]+)'

    match = re.match('nstep-' + int_capture, return_type)
    if match:
        n = int(match.group(1))
        return NStepReplayMemory(*shared_args, n)

    match = re.match('pengs-' + float_capture, return_type)
    if match:
        lambd = float(match.group(1))
        return LambdaReplayMemory(*shared_args, lambd, use_watkins=False)

    match = re.match('watkins-' + float_capture, return_type)
    if match:
        lambd = float(match.group(1))
        return LambdaReplayMemory(*shared_args, lambd, use_watkins=True)

    if return_type == 'pengs-median':
        return DynamicLambdaReplayMemory(*shared_args, use_watkins=False)

    if return_type == 'watkins-median':
        return DynamicLambdaReplayMemory(*shared_args, use_watkins=True)

    match = re.match('pengs-maxtd-' + float_capture, return_type)
    if match:
        raise NotImplementedError

    match = re.match('watkins-maxtd-' + float_capture, return_type)
    if match:
        raise NotImplementedError

    raise ValueError('Unrecognized return type {}'.format(return_type))


class ReplayMemory:
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority):
        assert (cache_size % chunk_size) == 0
        # Extra samples to fit exactly `capacity` (overlapping) chunks
        self.capacity = capacity + (history_len - 1) + chunk_size
        self.history_len = history_len
        self.discount = discount
        self.num_samples = 0

        self.cache_size = cache_size
        self.chunk_size = chunk_size
        self.priority = priority
        self.refresh_func = None

        # Main variables for memory
        self.obs = None  # Allocated dynamically once shape/dtype are known
        self.actions = np.empty([self.capacity], dtype=np.int32)
        self.rewards = np.empty([self.capacity], dtype=np.float32)
        self.dones = np.empty([self.capacity], dtype=np.bool)
        self.next = 0  # Points to next transition to be overwritten

        # Auxiliary buffers for the cache -- pre-allocated to smooth memory usage
        self.cached_obs = None  # Allocated dynamically once shape/dtype are known
        self.cached_actions = np.empty([self.cache_size], dtype=np.int32)
        self.cached_returns = np.empty([self.cache_size], dtype=np.float32)
        self.cached_errors  = np.empty([self.cache_size], dtype=np.float32)
        self.cached_indices = np.empty([self.cache_size], dtype=np.int32)

    def register_refresh_func(self, f):
        assert self.refresh_func is None
        self.refresh_func = f

    def sample(self, batch_size):
        start = self.batch_counter * batch_size
        end = start + batch_size
        indices = self.cached_indices[start:end]

        obs_batch = self.cached_obs[indices]
        act_batch = self.cached_actions[indices]
        ret_batch = self.cached_returns[indices]

        self.batch_counter += 1

        return np.array(obs_batch), np.array(act_batch), np.array(ret_batch)

    def encode_recent_observation(self):
        i = self.len()
        return self._encode_observation(i)

    def _encode_observation(self, i):
        i = self._align(i)

        # Start with blank observations except the last
        obs = np.zeros([self.history_len, *self.obs[0].shape], dtype=self.obs[0].dtype)
        obs[-1] = self.obs[i]

        # Fill-in backwards, break if we reach a terminal state
        for j in range(1, min(self.history_len, self.len())):
            if self.dones[i-j]:
                break
            obs[-1-j] = self.obs[i-j]

        return obs

    def _align(self, i):
        # Make relative to pointer when full
        if not self.full(): return i
        return (i + self.next) % self.capacity

    def store_obs(self, obs):
        if self.obs is None:
            self.obs = np.empty([self.capacity, *obs.shape], dtype=obs.dtype)
        if self.cached_obs is None:
            self.cached_obs = np.empty([self.cache_size, *obs.shape], dtype=obs.dtype)
        self.obs[self.next] = obs

    def store_effect(self, action, reward, done):
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.dones[self.next] = done

        self.next = (self.next + 1) % self.capacity
        self.num_samples = min(self.capacity, self.num_samples + 1)

    def len(self):
        return self.num_samples

    def full(self):
        return self.len() == self.capacity

    def refresh(self, train_frac):
        # Reset batch counter
        self.batch_counter = 0

        # Sample chunks until we have enough data
        num_chunks = self.cache_size // self.chunk_size
        chunk_ids = self._sample_chunk_ids(num_chunks)

        self._refresh(train_frac, chunk_ids)  # Separate function for unit testing

    def _refresh(self, train_frac, chunk_ids):
        # Refresh the chunks we sampled
        obs_chunks = [self._extract_chunk(None, i, obs=True) for i in chunk_ids]
        action_chunks = [self._extract_chunk(self.actions, i) for i in chunk_ids]
        reward_chunks = [self._extract_chunk(self.rewards, i) for i in chunk_ids]
        done_chunks = [self._extract_chunk(self.dones, i) for i in chunk_ids]

        return_chunks = []
        error_chunks = []
        for obs, actions, rewards, dones in zip(obs_chunks, action_chunks, reward_chunks, done_chunks):
            max_qvalues, mask, onpolicy_qvalues = self.refresh_func(obs, actions)

            returns = self._calculate_returns(rewards, max_qvalues, dones, mask)
            return_chunks.append(returns)

            errors = np.abs(returns - onpolicy_qvalues)
            error_chunks.append(errors)

        # Collect and store data
        self.cached_obs = np.concatenate([c[:-1] for c in obs_chunks])
        self.cached_actions = np.concatenate([c for c in action_chunks])
        self.cached_returns = np.concatenate([c for c in return_chunks])
        self.cached_errors = np.concatenate([c for c in error_chunks])

        # Prioritize samples
        distr = self._prioritized_distribution(self.cached_errors, train_frac)
        self.cached_indices = np.random.choice(self.cache_size, size=self.cache_size, replace=True, p=distr)

    def _sample_chunk_ids(self, n):
        return np.random.randint(self.history_len - 1, self.len() - self.chunk_size, size=n)

    def _extract_chunk(self, a, start, obs=False):
        end = start + self.chunk_size
        if obs:
            assert a is None
            return np.array([self._encode_observation(i) for i in range(start, end + 1)])
        return a[self._align(np.arange(start, end))]

    def _prioritized_distribution(self, errors, train_frac):
        # Start with the uniform distribution.
        distr = np.ones_like(errors) / self.cache_size
        # Adjust the probabilities based on whether their corresponding errors lie above/below the median.
        p = self.priority_now(train_frac)
        med = np.median(errors)
        distr[errors > med] *= (1.0 + p)
        distr[errors < med] *= (1.0 - p)
        # Note that if the error was identically equal to the median, its probability was not adjusted;
        # this is the correct behavior to guarantee the probabilities sum to 1.
        # However, due to floating point errors, we still need to re-normalize the distribution here:
        return distr / distr.sum()

    def priority_now(self, train_frac):
        return self.priority * (1.0 - train_frac)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        raise NotImplementedError


class LambdaReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority, lambd, use_watkins):
        self.lambd = lambd
        self.use_watkins = use_watkins
        super().__init__(capacity, history_len, discount, cache_size, chunk_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        return calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, self.lambd)


class DynamicLambdaReplayMemory(LambdaReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority, use_watkins):
        lambd = None
        super().__init__(capacity, history_len, discount, cache_size, chunk_size, priority, lambd, use_watkins)

    def _calculate_returns(self, rewards, qvalues, dones, mask, k=21):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        assert k > 1
        returns = np.empty(shape=[k, rewards.size], dtype=np.float32)
        for i in range(0, k):
            returns[i] = calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, lambd=i/(k-1))
        return np.median(returns, axis=0)


class NStepReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority, n):
        self.n = n
        super().__init__(capacity, history_len, discount, cache_size, chunk_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        return calculate_nstep_returns(rewards, qvalues, dones, self.discount, self.n)
