import numpy as np
from replay_memory import NStepReplayMemory


class LegacyReplayMemory(NStepReplayMemory):
    def __init__(self, size, history_len, discount, n):
        super().__init__(size, history_len, discount, n)
        self.oversample = None
        self.prioritize = None
        self.chunk_size = n

    def config_cache(self, oversample, priority, chunk_size):
        raise NotImplementedError

    def sample(self, batch_size):
        indices = self._sample_chunk_ids(batch_size)
        return self._sample(indices)  # Separate function for unit testing

    def _sample(self, indices):
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []

        for i in indices:
            obs_batch.append( self._extract_chunk(None, i, obs=True) )
            act_batch.append( self._extract_chunk(self.actions, i) )
            rew_batch.append( self._extract_chunk(self.rewards, i) )
            done_batch.append( self._extract_chunk(self.dones, i).astype(np.float32) )

        obs_batch, act_batch, rew_batch, done_batch = map(np.array, [obs_batch, act_batch, rew_batch, done_batch])

        # Compute the n-step returns
        ret_batch = self.refresh_func(obs_batch[:, -1])  # Begin with bootstrap states
        for i in reversed(range(self.n)):
            ret_batch = rew_batch[:, i] + self.discount * ret_batch * (1.0 - done_batch[:, i])

        return obs_batch[:, 0], act_batch[:, 0], ret_batch

    def refresh(self, cache_size, train_frac):
        raise NotImplementedError

    def _refresh(self, cache_size, train_frac, chunk_ids):
        raise NotImplementedError

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        raise NotImplementedError
