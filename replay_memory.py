import numpy as np


def remove(string, prefix):
    assert string.startswith(prefix)
    return string[len(prefix):]


def make_replay_memory(return_type, capacity, history_len, discount, cache_size, chunk_size, priority):
    _return_type = return_type  # In case we raise an exception
    shared_args = (capacity, history_len, discount, cache_size, chunk_size, priority)

    try:
        if return_type.startswith('nstep-'):
            return_type = remove(return_type, 'nstep-')
            n = int(return_type)
            return NStepReplayMemory(*shared_args, n)

        if return_type.startswith('pengs-'):
            return_type = remove(return_type, 'pengs-')
            use_watkins = False
        elif return_type.startswith('watkins-'):
            return_type = remove(return_type, 'watkins-')
            use_watkins = True
        else:
            raise ValueError

        if return_type == 'dynamic':
            return DynamicLambdaReplayMemory(*shared_args, use_watkins)

        lambd = float(return_type)
        return LambdaReplayMemory(*shared_args, lambd, use_watkins)

    except:
        raise ValueError('Unrecognized return type {}'.format(_return_type))


def pad_axis0(array, value):
    return np.pad(array, pad_width=(0,1), mode='constant', constant_values=value)


def shift(array):
        return pad_axis0(array, 0)[1:]


def calculate_lambda_returns(rewards, qvalues, dones, mask, discount, lambd):
    dones = dones.astype(np.float32)
    qvalues[-1] *= (1.0 - dones[-1])
    lambda_returns = rewards + (discount * qvalues[1:])
    for i in reversed(range(len(rewards) - 1)):
        a = lambda_returns[i] + (discount * lambd * mask[i]) * (lambda_returns[i+1] - qvalues[i+1])
        b = rewards[i]
        lambda_returns[i] = (1.0 - dones[i]) * a + dones[i] * b
    return lambda_returns


def calculate_nstep_returns(rewards, qvalues, dones, discount, n):
    # Counterintuitively, the bootstrap is treated is as a reward too
    rewards = pad_axis0(rewards, qvalues[-1])
    dones   = pad_axis0(dones, 1.0)

    mask    = np.ones_like(rewards)
    decay   = 1.0
    returns = np.copy(rewards)

    for i in range(n):
        decay *= discount
        mask *= (1.0 - dones)

        rewards = shift(rewards)
        qvalues = shift(qvalues)
        dones   = shift(dones)

        if i != (n-1):
            returns += (mask * decay * rewards)
        else:
            returns += (mask * decay * qvalues)

    return returns[:-1]  # Remove bootstrap placeholder


class ReplayMemory:
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority):
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
        self.cached_actions = np.empty([self.capacity], dtype=np.int32)
        self.cached_returns = np.empty([self.capacity], dtype=np.float32)
        self.cached_errors = np.empty([self.capacity], dtype=np.float32)

    def register_refresh_func(self, f):
        assert self.refresh_func is None
        self.refresh_func = f

    def sample(self, batch_size):
        start = self.batch_counter * batch_size
        end = start + batch_size
        indices = self.indices[start:end]

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
            self.obs = np.empty([self.capacity] + list(obs.shape), dtype=obs.dtype)
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
        p = self.priority_now(train_frac)
        threshold = np.quantile(self.cached_errors, p)
        distr = np.where(
            self.cached_errors >= threshold,
            np.ones_like(self.cached_errors),
            np.zeros_like(self.cached_errors),
        )
        distr /= distr.sum()  # Probabilities must sum to 1
        self.indices = np.random.choice(self.cache_size, size=self.cache_size, replace=True, p=distr)

    def _sample_chunk_ids(self, n):
        return np.random.randint(self.history_len - 1, self.len() - self.chunk_size, size=n)

    def _extract_chunk(self, a, start, obs=False):
        end = start + self.chunk_size
        if obs:
            assert a is None
            return np.array([self._encode_observation(i) for i in range(start, end + 1)])
        return a[self._align(np.arange(start, end))]

    def priority_now(self, train_frac):
        return self.priority * (1.0 - train_frac)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        raise NotImplementedError


class LambdaReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority, lambd, use_watkins):
        self.lambd = lambd
        self.use_watkins = use_watkins
        super().__init__(capacity, history_len, discount)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        return calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, self.lambd)


class DynamicLambdaReplayMemory(LambdaReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority, use_watkins):
        lambd = None
        super().__init__(capacity, history_len, discount, cache_size, chunk_size, priority, lambd, use_watkins)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        returns = np.empty(shape=[21, rewards.size], dtype=np.float32)
        for n in range(0, 21):
            lambd = n / 20.0
            returns[n] = calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, lambd)
        return np.median(returns, axis=0)


class NStepReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, chunk_size, priority, n):
        self.n = n
        super().__init__(capacity, history_len, discount, cache_size, chunk_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        return calculate_nstep_returns(rewards, qvalues, dones, self.discount, self.n)
