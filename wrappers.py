import cv2
import numpy as np
from collections import deque
import gym
from gym.wrappers import Monitor
from datetime import datetime
import os


class HistoryWrapper(gym.Wrapper):
    '''Automatically stacks the past `history_len - 1` observations
    onto the current observation. This should be used only for the
    benchmark env to emulate the effect of the replay memory.'''
    def __init__(self, env, history_len):
        super(HistoryWrapper, self).__init__(env)

        self.history_len = history_len
        self.deque = deque(maxlen=history_len)

        shape = list(self.observation_space.shape)
        shape[-1] *= history_len
        self.observation_space.shape = tuple(shape)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._contextualize(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._contextualize(obs, reset=True)

    def _contextualize(self, obs, reset=False):
        if reset:
            for i in range(self.history_len - 1):
                self.deque.append(np.zeros_like(obs))
        self.deque.append(obs)
        return np.stack(list(self.deque))


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.step(0)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.step(1)
        obs, _, _, _ = self.step(2)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_LINEAR)
        return np.reshape(observation, [84, 84, 1]).astype(np.uint8)

class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)

def monitor(env, name, video=False):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    monitor_dir = os.path.join('/tmp/', name + '_' + timestamp)
    print('Logging to', monitor_dir)
    env = Monitor(env, directory=monitor_dir)
    if not video:
        env.video_callable = lambda e: False
    return env

def wrap_deepmind(env):
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = NoopResetEnv(env, noop_max=20)
    env = EpisodicLifeEnv(env)
    env = ClippedRewardsWrapper(env)
    env = ProcessFrame84(env)
    return env
