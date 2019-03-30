import numpy as np
import random


def make_replay_memory(return_type, history_len, size, discount):
    if 'nstep-' in return_type:
        n = int( return_type.strip('nstep-') )
        replay_memory = NStepReplayMemory(size, history_len, discount, nsteps=n)

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


class Episode:
    def __init__(self, history_len, discount, refresh_func):
        self.finished = False
        self.history_len = history_len

        self.discount = discount
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
        self.returns = self._calculate_returns(self.reward, qvalues, mask, self.discount)

    def _calculate_returns(self, rewards, qvalues, mask, discount):
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


def calculate_lambda_returns(rewards, qvalues, mask, discount, lambd):
    next_qvalues = shifted(qvalues, 1)  # Final state in episode is terminal
    lambda_returns = rewards + (discount * next_qvalues)
    for i in reversed(range(len(rewards) - 1)):
        lambda_returns[i] += (discount * lambd * mask[i]) * (lambda_returns[i+1] - next_qvalues[i])
    return lambda_returns


def calculate_renormalized_lambda_returns(rewards, qvalues, mask, discount, lambd):
    next_qvalues = shifted(qvalues, 1)  # Final state in episode is terminal
    lambda_returns = rewards + (discount * next_qvalues)
    N = 1
    for i in reversed(range(len(rewards) - 1)):
        def k(n):
            if n == 0:
                return 1.0
            return sum([lambd**i for i in range(n)])
        l = lambd * mask[i]
        N = (N * int(mask[i])) + 1
        lambda_returns[i] = (1. / k(N)) * (lambda_returns[i] + l * k(N-1) * (rewards[i] + discount * lambda_returns[i+1]))
    return lambda_returns


def calculate_nstep_returns(rewards, qvalues, discount, nsteps):
    mask = np.ones_like(qvalues)
    mc_returns = calculate_lambda_returns(rewards, qvalues, mask, discount, lambd=1.0)
    nstep_returns = mc_returns - (discount ** nsteps) * (shifted(mc_returns, nsteps) - shifted(qvalues, nsteps))
    return nstep_returns


class LambdaEpisode(Episode):
    def __init__(self, history_len, discount, Lambda, renormalize, use_watkins, refresh_func):
        self.Lambda = Lambda
        self.renormalize = renormalize
        self.use_watkins = use_watkins
        super().__init__(history_len, discount, refresh_func)

    def _calculate_returns(self, rewards, qvalues, mask, discount):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        if self.renormalize:
            return calculate_renormalized_lambda_returns(rewards, qvalues, mask, discount, self.Lambda)
        return calculate_lambda_returns(rewards, qvalues, mask, discount, self.Lambda)


class NStepEpisode(Episode):
    def __init__(self, history_len, discount, nsteps, refresh_func):
        self.nsteps = nsteps
        super().__init__(history_len, discount, refresh_func)

    def _calculate_returns(self, rewards, qvalues, mask, discount):
        return calculate_nstep_returns(rewards, qvalues, discount, self.nsteps)


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
    def __init__(self, size, history_len, discount, Lambda, renormalize, use_watkins):
        self.Lambda = Lambda
        self.renormalize = renormalize
        self.use_watkins = use_watkins
        super().__init__(size, history_len, discount)

    def _new_episode(self):
        return LambdaEpisode(self.history_len, self.discount, self.Lambda, self.renormalize, self.use_watkins, self.refresh_func)


class NStepReplayMemory(ReplayMemory):
    def __init__(self, size, history_len, discount, nsteps):
        self.nsteps = nsteps
        super().__init__(size, history_len, discount)

    def _new_episode(self):
        return NStepEpisode(self.history_len, self.discount, self.nsteps, self.refresh_func)
