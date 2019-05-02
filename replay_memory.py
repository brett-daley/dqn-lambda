import numpy as np


def make_replay_memory(return_type, history_len, size, discount):
    if 'nstep-' in return_type:
        n = int( return_type.strip('nstep-') )
        replay_memory = NStepReplayMemory(size, history_len, discount, n)

    elif 'renorm-pengs-' in return_type:
        lambd = float( return_type.strip('renorm-pengs-') )
        replay_memory = LambdaReplayMemory(size, history_len, discount, lambd, renormalize=True, use_watkins=False)

    elif 'pengs-' in return_type:
        lambd = float( return_type.strip('pengs-') )
        replay_memory = LambdaReplayMemory(size, history_len, discount, lambd, renormalize=False, use_watkins=False)

    elif 'renorm-watkins-' in return_type:
        lambd = float( return_type.strip('renorm-watkins-') )
        replay_memory = LambdaReplayMemory(size, history_len, discount, lambd, renormalize=True, use_watkins=True)

    elif 'watkins-' in return_type:
        lambd = float( return_type.strip('watkins-') )
        replay_memory = LambdaReplayMemory(size, history_len, discount, lambd, renormalize=False, use_watkins=True)

    else:
        raise ValueError('Unrecognized return type')

    return replay_memory


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

        self.oversample = 1.0
        self.prioritize = 0.0
        self.chunk_size = 100

        self.refresh_func = None

        self.obs     = []
        self.actions = []
        self.rewards = []
        self.dones   = []

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
        i = self.len() - 1
        return self._encode_observation(i)

    def _encode_observation(self, i):
        end = i + 1  # Make non-inclusive
        start = end - self.history_len

        pad_len = max(0, -start)
        padding = [np.zeros_like(self.obs[0]) for _ in range(pad_len)]

        start = max(0, start)
        obs = self.obs[start:end]

        return np.array(padding + obs)

    def store_obs(self, obs):
        self.obs.append(obs)

    def store_effect(self, action, reward, done):
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        if self.len() > self.size:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

        assert len(self.obs) == len(self.actions) == len(self.rewards) == len(self.dones)

    def len(self):
        return len(self.obs)

    def refresh(self, cache_size, train_frac):
        # Reset batch counter
        self.batch_counter = 0

        # Sample chunks until we have enough data
        num_chunks = int(self.oversample * cache_size) // self.chunk_size
        chunk_ids = self._sample_chunk_ids(num_chunks)

        self._refresh(cache_size, train_frac, chunk_ids)  # Separate function for unit testing

    def _refresh(self, cache_size, train_frac, chunk_ids):
        # Refresh the chunks we sampled
        obs_chunks = [self._extract_chunk(self.obs, i, obs=True) for i in chunk_ids]
        action_chunks = [self._extract_chunk(self.actions, i) for i in chunk_ids]
        reward_chunks = [self._extract_chunk(self.rewards, i) for i in chunk_ids]
        done_chunks = [self._extract_chunk(self.dones, i) for i in chunk_ids]

        return_chunks = []
        error_chunks = []
        for obs, actions, rewards, dones in zip(obs_chunks, action_chunks, reward_chunks, done_chunks):
            qvalues, mask = self.refresh_func(obs, actions)

            returns = self._calculate_returns(rewards, qvalues, dones, mask)
            return_chunks.append(returns)

            one_step_returns = calculate_nstep_returns(rewards, qvalues, dones, self.discount, n=1)
            errors = np.abs(one_step_returns - qvalues[:-1])
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
        return np.random.randint(self.chunk_size, self.len() - 1, size=n)

    def _extract_chunk(self, list, end, obs=False):
        end += 1  # Make non-inclusive
        start = end - self.chunk_size
        if obs:
            return np.array([self._encode_observation(i) for i in range(start, end + 1)])
        return np.array(list[start:end])

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
        if self.renormalize:
            return calculate_renormalized_lambda_returns(rewards, qvalues, mask, self.discount, self.lambd)
        return calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, self.lambd)


class NStepReplayMemory(ReplayMemory):
    def __init__(self, size, history_len, discount, n):
        self.n = n
        super().__init__(size, history_len, discount)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        return calculate_nstep_returns(rewards, qvalues, dones, self.discount, self.n)
