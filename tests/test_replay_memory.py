import unittest
import numpy as np
from replay_memory import NStepReplayMemory, LambdaReplayMemory


class TestCaseCore(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.transitions = [
            # state, action, reward, done?, qvalue, greedy?
            (0, 7,  10.0, False, 100.0, True),
            (1, 0, -10.0, False, 200.0, False),
            (2, 1,  20.0, True,  300.0, True),
            (3, 6, -20.0, False, 400.0, True),
            (4, 4,  30.0, False, 300.0, True),
            (5, 2, -30.0, False, 200.0, False),
            (6, 5,  40.0, False, 100.0, True),
            (7, 9, -40.0, True,  200.0, True),
            (8, 7,  50.0, False, 300.0, True),
            (9, 8, -50.0, False, 400.0, False),
        ]
        # Mock refresh function to return our custom qvalues and greedy mask
        qvalues = np.array([t[4] for t in self.transitions])
        mask    = np.array([t[5] for t in self.transitions], dtype=np.float32)
        def refresh(states, actions):
            return (
                np.reshape([qvalues[s] for s in states], -1),
                np.reshape([mask[s] for s in states], -1),
            )
        self.refresh = refresh
        # Now remove Q-value information from transitions, because tests don't need it explicitly
        self.transitions = [(np.array(state), np.array(action), reward, done) for state, action, reward, done, _, _ in self.transitions]

    def fill(self, replay_memory):
        replay_memory.register_refresh_func(self.refresh)
        for state, action, reward, done in self.transitions:
            replay_memory.store_obs(state)
            state = replay_memory.encode_recent_observation()
            replay_memory.store_effect(action, reward, done)

    def assertNumpyEqual(self, x, y):
        f = lambda a: np.array(a).reshape(-1)
        x, y = map(f, [x, y])
        self.assertEqual(x.dtype, y.dtype)
        self.assertTrue(np.allclose(x - y, 0.0))


class TestCaseReplayMemory(TestCaseCore):
    def fill(self, replay_memory):
        super().fill(replay_memory)
        replay_memory._refresh(cache_size=9, train_frac=0.0, chunk_ids=[0, 3, 6])

    def test_1step(self):
        m = NStepReplayMemory(size=20, history_len=1, discount=0.9, n=1)
        m.config_cache(oversample=1.0, priority=0.0, chunk_size=3)
        self.fill(m)

        self.assertNumpyEqual(m.cached_obs,     [    0,     1,    2,     3,     4,    5,     6,     7,     8])
        self.assertNumpyEqual(m.cached_actions, [    7,     0,    1,     6,     4,    2,     5,     9,     7])
        self.assertNumpyEqual(m.cached_returns, [190.0, 260.0, 20.0, 250.0, 210.0, 60.0, 220.0, -40.0, 410.0])

    def test_nstep(self):
        m = NStepReplayMemory(size=20, history_len=1, discount=0.9, n=3)
        m.config_cache(oversample=1.0, priority=0.0, chunk_size=3)
        self.fill(m)

        self.assertNumpyEqual(m.cached_obs,     [   0,   1,    2,    3,    4,    5,   6,     7,     8])
        self.assertNumpyEqual(m.cached_actions, [   7,   0,    1,    6,    4,    2,   5,     9,     7])
        self.assertNumpyEqual(m.cached_returns, [17.2, 8.0, 20.0, 55.6, 84.0, 60.0, 4.0, -40.0, 410.0])

    def test_pengs_lambda(self):
        m = LambdaReplayMemory(size=20, history_len=1, discount=0.9, lambd=0.8,
                               renormalize=False, use_watkins=False)
        m.config_cache(oversample=1.0, priority=0.0, chunk_size=3)
        self.fill(m)

        self.assertNumpyEqual(m.cached_obs,     [     0,    1,    2,        3,     4,    5,    6,     7,     8])
        self.assertNumpyEqual(m.cached_actions, [     7,    0,    1,        6,     4,    2,    5,     9,     7])
        self.assertNumpyEqual(m.cached_returns, [88.048, 58.4, 20.0,  112.624, 109.2, 60.0, 47.2, -40.0, 410.0])

    def test_watkins_lambda(self):
        m = LambdaReplayMemory(size=20, history_len=1, discount=0.9, lambd=0.8,
                               renormalize=False, use_watkins=True)
        m.config_cache(oversample=1.0, priority=0.0, chunk_size=3)
        self.fill(m)

        self.assertNumpyEqual(m.cached_obs,     [    0,     1,    2,       3,     4,    5,    6,     7,     8])
        self.assertNumpyEqual(m.cached_actions, [    7,     0,    1,       6,     4,    2,    5,     9,     7])
        self.assertNumpyEqual(m.cached_returns, [233.2, 260.0, 20.0, 112.624, 109.2, 60.0, 47.2, -40.0, 410.0])

    def test_pengs_renormalized_lambda(self):
        pass

    def test_watkins_renormalized_lambda(self):
        pass
