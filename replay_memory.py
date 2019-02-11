import numpy as np
import random


class Episode:
    def __init__(self, history_len, discount, Lambda, refresh_func):
        self.finished = False
        self.history_len = history_len

        self.discount = discount
        self.Lambda = Lambda

        self.refresh_func = refresh_func

        self.obs     = []
        self.action  = []
        self.reward  = []
        self.returns = None

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

        self.obs = np.stack(self.obs)
        self.action = np.array(self.action)
        self.reward = np.array(self.reward)

    def refresh(self):
        obs = np.array([self._encode_observation(i) for i in range(self.length)])
        qvalues, mask = self.refresh_func(obs, self.action)
        self.returns = self._calc_returns(qvalues, mask)

    def _calc_returns(self, qvalues, mask):
        raise NotImplementedError

    def sample(self):
        i = random.randrange(self.length)
        return (self._encode_observation(i), self.action[i], self.returns[i])

    def _encode_observation(self, idx):
        end = (idx % self.length) + 1 # make noninclusive
        start = end - self.history_len

        pad_len = max(0, -start)
        padding = [np.zeros_like(self.obs[0]) for _ in range(pad_len)]

        start = max(0, start)
        obs = [x for x in self.obs[start:end]]

        return np.array(padding + obs)


def shifted(array, k):
    '''Shifts array left by k elements and pads with zeros'''
    return np.pad(array, (0, k), mode='constant')[k:]


class LambdaEpisode(Episode):
    def _calc_returns(self, qvalues, mask):
        return self._calc_lambda_returns(qvalues, mask)

    def _calc_lambda_returns(self, qvalues, mask):
        next_qvalues = shifted(qvalues, 1)  # Final state in episode is terminal
        lambda_returns = self.reward + (self.discount * next_qvalues)
        for i in reversed(range(self.length - 1)):
            lambda_returns[i] += (self.discount * self.Lambda * mask[i]) * (lambda_returns[i+1] - next_qvalues[i])
        return lambda_returns


class RenormalizedLambdaEpisode(LambdaEpisode):
    def _calc_lambda_returns(self, qvalues, mask):
        next_qvalues = shifted(qvalues, 1)  # Final state in episode is terminal
        lambda_returns = self.reward + (self.discount * next_qvalues)
        N = 1
        for i in reversed(range(self.length - 1)):
            def k(n):
                if n == 0:
                    return 1.0
                return sum([self.Lambda**i for i in range(n)])
            l = self.Lambda * mask[i]
            N = (N * int(mask[i])) + 1
            lambda_returns[i] = (1. / k(N)) * (lambda_returns[i] + l * k(N-1) * (self.reward[i] + self.discount * lambda_returns[i+1]))
        return lambda_returns


class NStepEpisode(LambdaEpisode):
    def __init__(self, history_len, discount, nsteps, refresh_func):
        self.nsteps = nsteps
        Lambda = 1.0
        super().__init__(history_len, discount, Lambda, refresh_func)

    def _calc_returns(self, qvalues, mask):
        return self._calc_nstep_returns(qvalues, mask)

    def _calc_nstep_returns(self, qvalues, mask):
        mask = np.ones_like(mask)
        mc_returns = self._calc_lambda_returns(qvalues, mask)
        n = self.nsteps
        nstep_returns = mc_returns - (self.discount ** n) * (shifted(mc_returns, n) - shifted(qvalues, n))
        return nstep_returns


class ReplayMemory:
    def __init__(self, size, history_len, discount):
        self.size = size
        self.history_len = history_len
        self.discount = discount

        self.refresh_func = None

        self.episodes = []
        self.waiting_episodes = []
        self.current_episode = None

    def register_refresh_func(self, f):
        self.refresh_func = f

    def can_sample(self):
        return len(self.episodes) > 0

    def _encode_sample(self, idxes):
        samples = [self.episodes[i].sample() for i in idxes]
        obs_batch, act_batch, rew_batch = zip(*samples)

        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch)

    def sample(self, batch_size):
        assert self.can_sample()

        lengths = np.array([e.length for e in self.episodes])
        bias_correction = lengths / np.sum(lengths)

        idxes = np.random.choice(
            a=np.arange(len(self.episodes)),
            size=batch_size,
            p=bias_correction,
        )
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        if self.current_episode.length > 0:
            return self.current_episode._encode_observation(-1)
        else:
            return self.waiting_episodes[-1]._encode_observation(-1)

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
        self.waiting_episodes.append(self.current_episode)
        self.current_episode = self._new_episode()

    def refresh(self):
        self.episodes.extend(self.waiting_episodes)
        self.waiting_episodes.clear()
        while sum([e.length for e in self.episodes]) > self.size:
            self.episodes.pop(0)
        for e in self.episodes:
            e.refresh()

    def _new_episode(self):
        raise NotImplementedError


class LambdaReplayMemory(ReplayMemory):
    def __init__(self, size, history_len, discount, Lambda):
        self.Lambda = Lambda
        super().__init__(size, history_len, discount)

    def _new_episode(self):
        return LambdaEpisode(self.history_len, self.discount, self.Lambda, self.refresh_func)


class RenormalizedLambdaReplayMemory(LambdaReplayMemory):
    def _new_episode(self):
        return RenormalizedLambdaEpisode(self.history_len, self.discount, self.Lambda, self.refresh_func)


class NStepReplayMemory(ReplayMemory):
    def __init__(self, size, history_len, discount, nsteps):
        self.nsteps = nsteps
        super().__init__(size, history_len, discount)

    def _new_episode(self):
        return NStepEpisode(self.history_len, self.discount, self.nsteps, self.refresh_func)
