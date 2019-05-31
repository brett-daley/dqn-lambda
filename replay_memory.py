import numpy as np


def remove(string, prefix):
    assert string.startswith(prefix)
    return string[len(prefix):]


def make_replay_memory(return_type, history_len, size, discount):
    _return_type = return_type  # In case we raise an exception

    try:
        if return_type.startswith('nstep-'):
            return_type = remove(return_type, 'nstep-')
            n = int(return_type)
            return NStepReplayMemory(size, history_len, discount, n)

        renormalize = return_type.startswith('renorm-')
        if renormalize:
            return_type = remove(return_type, 'renorm-')

        if return_type.startswith('pengs-'):
            return_type = remove(return_type, 'pengs-')
            use_watkins = False
        elif return_type.startswith('watkins-'):
            return_type = remove(return_type, 'watkins-')
            use_watkins = True
        else:
            raise ValueError

        if return_type.startswith('dynamic-'):
            return_type = remove(return_type, 'dynamic-')
            max_td = float(return_type)
            return DynamicLambdaReplayMemory(size, history_len, discount, max_td, renormalize, use_watkins)

        lambd = float(return_type)
        return LambdaReplayMemory(size, history_len, discount, lambd, renormalize, use_watkins)

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


def calculate_renormalized_lambda_returns(rewards, qvalues, mask, discount, lambd):
    raise NotImplementedError

    def k(n):
        if n == 0:
            return 1.0
        return (1.0 - lambd**n) / (1.0 - lambd)

    next_qvalues = shift(qvalues)  # Final state in episode is terminal
    lambda_returns = rewards + (discount * next_qvalues)

    n = 1
    for i in reversed(range(len(rewards) - 1)):
        l = lambd * mask[i]
        n = (n * int(mask[i])) + 1
        lambda_returns[i] = (1. / k(n)) * (lambda_returns[i] + l * k(n-1) * (rewards[i] + discount * lambda_returns[i+1]))
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
    def __init__(self, size, history_len, discount):
        self.size = size
        self.history_len = history_len
        self.discount = discount
        self.num_samples = 0

        self.oversample = 1.0
        self.prioritize = 0.0
        self.chunk_size = 100
        self.size += (self.history_len - 1) + self.chunk_size  # Extra samples to fit exactly `size` chunks

        self.refresh_func = None

        self.obs = None  # Allocated dynamically once shape/dtype are known
        self.actions = np.empty([self.size], dtype=np.int32)
        self.rewards = np.empty([self.size], dtype=np.float64)
        self.dones = np.empty([self.size], dtype=np.bool)
        self.next = 0  # Points to next transition to be overwritten

    def register_refresh_func(self, f):
        assert self.refresh_func is None
        self.refresh_func = f

    def config_cache(self, oversample, priority, chunk_size):
        assert oversample >= 1.0
        assert 0.0 <= priority <= 1.0
        assert isinstance(chunk_size, int) and chunk_size >= 1
        if oversample == 1.0 and priority > 0.0:
            raise ValueError("Can't prioritize when oversampling ratio is 1.0")
        self.oversample = oversample
        self.priority = priority
        self.chunk_size = chunk_size

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
        return (i + self.next) % self.size

    def store_obs(self, obs):
        if self.obs is None:
            self.obs = np.empty([self.size] + list(obs.shape), dtype=obs.dtype)
        self.obs[self.next] = obs

    def store_effect(self, action, reward, done):
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.dones[self.next] = done

        self.next = (self.next + 1) % self.size
        self.num_samples = min(self.size, self.num_samples + 1)

    def len(self):
        return self.num_samples

    def full(self):
        return self.len() == self.size

    def refresh(self, cache_size, train_frac):
        # Reset batch counter
        self.batch_counter = 0

        # Sample chunks until we have enough data
        num_chunks = int(self.oversample * cache_size) // self.chunk_size
        chunk_ids = self._sample_chunk_ids(num_chunks)

        self._refresh(cache_size, train_frac, chunk_ids)  # Separate function for unit testing

    def _refresh(self, cache_size, train_frac, chunk_ids):
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

        self.indices = np.arange(len(self.cached_returns))
        np.random.shuffle(self.indices)

        if self.priority > 0.0:
            cached_errors = np.concatenate([c for c in error_chunks])
            sort = np.argsort(cached_errors)[::-1]
            prioritized = self.indices[sort]

            p = self.priority_now(train_frac)
            b = np.random.choice([0, 1], size=self.indices.shape, p=[1-p, p])
            self.indices[b == 1] = prioritized[b == 1]

        self.indices = self.indices[:cache_size]
        np.random.shuffle(self.indices)

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
    def __init__(self, size, history_len, discount, lambd, renormalize, use_watkins):
        self.lambd = lambd
        self.renormalize = renormalize
        self.use_watkins = use_watkins
        super().__init__(size, history_len, discount)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)

        f = (calculate_renormalized_lambda_returns if self.renormalize else calculate_lambda_returns)
        return f(rewards, qvalues, dones, mask, self.discount, self.lambd)


class DynamicLambdaReplayMemory(LambdaReplayMemory):
    def __init__(self, size, history_len, discount, max_td, renormalize, use_watkins):
        lambd = None
        self.lambdas_since_refresh = []
        self.max_td = max_td
        super().__init__(size, history_len, discount, lambd, renormalize, use_watkins)

    def refresh(self, cache_size, train_frac):
        self.lambdas_since_refresh.clear()
        super().refresh(cache_size, train_frac)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        f = super()._calculate_returns  # Use parent function to compute returns

        # Try the extremes first
        returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd=1.0)
        if ok:
            self.lambdas_since_refresh.append(1.0)
            return returns

        returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd=0.0)
        if not ok:
            self.lambdas_since_refresh.append(0.0)
            return returns

        # If we haven't returned by now, we need to search for a good lambda value
        best_returns, best_lambd = None, None
        lambd = 0.5
        num_iterations = 7

        for i in range(2, 2 + num_iterations):
            returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd)

            if ok:
                best_returns, best_lambd = returns, lambd
                lambd += 1.0 / (2.0 ** i)
            else:
                lambd -= 1.0 / (2.0 ** i)

        if best_returns is None:
            self.lambdas_since_refresh.append(lambd)
            return returns

        self.lambdas_since_refresh.append(best_lambd)
        return best_returns

    def _try_lambda(self, f, rewards, qvalues, dones, mask, lambd):
        self.lambd = lambd  # Pass implicitly to parent function
        returns = f(rewards, qvalues, dones, mask)
        td_error = np.square(returns - qvalues[:-1]).max()
        ok = (td_error <= self.max_td)
        return returns, ok


class NStepReplayMemory(ReplayMemory):
    def __init__(self, size, history_len, discount, n):
        self.n = n
        super().__init__(size, history_len, discount)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        return calculate_nstep_returns(rewards, qvalues, dones, self.discount, self.n)
