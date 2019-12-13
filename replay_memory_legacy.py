import numpy as np
import re

from replay_memory import NStepReplayMemory


def make_legacy_replay_memory(return_est, capacity, history_len, discount):
    match = re.match('nstep-([0-9]+)', return_est)
    if not match:
        raise ValueError('Legacy mode only supports n-step returns but requested {}'.format(return_est))
    n = int(match.group(1))
    return LegacyReplayMemory(capacity, history_len, discount, n)


class LegacyReplayMemory(NStepReplayMemory):
    def __init__(self, capacity, history_len, discount, n):
        super().__init__(capacity, history_len, discount, cache_size=0, block_size=n, priority=0.0, n=n)

    def sample(self, batch_size):
        indices = self._sample_block_ids(batch_size)
        return self._sample(indices)  # Separate function for unit testing

    def _sample(self, indices):
        state_batch, action_batch, reward_batch, done_batch = [], [], [], []

        for i in indices:
            state_batch.append( self._extract_block(None, i, states=True) )
            action_batch.append( self._extract_block(self.actions, i) )
            reward_batch.append( self._extract_block(self.rewards, i) )
            done_batch.append( self._extract_block(self.dones, i).astype(np.float32) )

        state_batch, action_batch, reward_batch, done_batch = map(np.array, [state_batch, action_batch, reward_batch, done_batch])

        # Compute the n-step returns
        return_batch = self.refresh_func(state_batch[:, -1])  # Begin with bootstrap states
        for i in reversed(range(self.n)):
            return_batch = reward_batch[:, i] + self.discount * return_batch * (1.0 - done_batch[:, i])

        return state_batch[:, 0], action_batch[:, 0], return_batch

    def refresh(self, cache_size, train_frac):
        raise NotImplementedError

    def _refresh(self, cache_size, train_frac, block_ids):
        raise NotImplementedError

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        raise NotImplementedError
