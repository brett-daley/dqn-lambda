import unittest
import numpy as np
from replay_memory import NStepReplayMemory, LambdaReplayMemory


class TestCaseReturns(unittest.TestCase):
    def setUp(self):
        self.transitions = [
            # state, action, reward, done?, qvalue, greedy?
            (0, 7,  10.0, False, 100.0, True),
            (1, 0, -10.0, False, 200.0, False),
            (2, 1,  20.0, True,  300.0, True),
            (3, 6, -20.0, False, 400.0, True),
            (4, 4,  30.0, False, 300.0, True),
            (5, 2, -30.0, False, 200.0, False),
            (6, 5,  40.0, False, 100.0, True),
            (7, 3, -40.0, True,  200.0, True),
        ]
        # Mock refresh function to return our custom qvalues and greedy mask
        qvalues = np.array([t[4] for t in self.transitions])
        mask    = np.array([t[5] for t in self.transitions], dtype=np.float32)
        self.refresh = lambda s, a: (qvalues[s].reshape(-1), mask[s].reshape(-1))
        # Now remove Q-value information from transitions, because tests don't need it explicitly
        self.transitions = [(np.array(state), np.array(action), reward, done) for state, action, reward, done, _, _ in self.transitions]

    def fill(self, replay_memory):
        replay_memory.register_refresh_func(self.refresh)
        for state, action, reward, done in self.transitions:
            replay_memory.store_frame(state)
            state = replay_memory.encode_recent_observation()
            replay_memory.store_effect(action, reward, done)
        replay_memory.refresh()

    def assertNumpyEqual(self, x, y):
        x, y = map(np.array, [x, y])
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        self.assertTrue(np.allclose(x - y, 0.0))

    def test_1step(self):
        replay_memory = NStepReplayMemory(size=20, history_len=1, discount=0.9, nsteps=1)
        self.fill(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     [    0,     1,    2])
        self.assertNumpyEqual(e.action,  [    7,     0,    1])
        self.assertNumpyEqual(e.returns, [190.0, 260.0, 20.0])

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     [    3,     4,    5,     6,     7])
        self.assertNumpyEqual(e.action,  [    6,     4,    2,     5,     3])
        self.assertNumpyEqual(e.returns, [250.0, 210.0, 60.0, 220.0, -40.0])

    def test_nstep(self):
        replay_memory = NStepReplayMemory(size=20, history_len=1, discount=0.9, nsteps=3)
        self.fill(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     [   0,   1,    2])
        self.assertNumpyEqual(e.action,  [   7,   0,    1])
        self.assertNumpyEqual(e.returns, [17.2, 8.0, 20.0])

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     [   3,     4,     5,   6,     7])
        self.assertNumpyEqual(e.action,  [   6,     4,     2,   5,     3])
        self.assertNumpyEqual(e.returns, [55.6, 181.2, -26.4, 4.0, -40.0])

    def test_pengs_lambda(self):
        replay_memory = LambdaReplayMemory(size=20, history_len=1, discount=0.9, Lambda=0.8,
                                           renormalize=False, use_watkins=False)
        self.fill(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     [     0,    1,    2])
        self.assertNumpyEqual(e.action,  [     7,    0,    1])
        self.assertNumpyEqual(e.returns, [88.048, 58.4, 20.0])

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     [         3,        4,      5,    6,     7])
        self.assertNumpyEqual(e.action,  [         6,        4,      2,    5,     3])
        self.assertNumpyEqual(e.returns, [92.9165056, 81.82848, 21.984, 47.2, -40.0])

    def test_watkins_lambda(self):
        replay_memory = LambdaReplayMemory(size=20, history_len=1, discount=0.9, Lambda=0.8,
                                           renormalize=False, use_watkins=True)
        self.fill(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     [    0,     1,    2])
        self.assertNumpyEqual(e.action,  [    7,     0,    1])
        self.assertNumpyEqual(e.returns, [233.2, 260.0, 20.0])

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     [      3,     4,    5,    6,     7])
        self.assertNumpyEqual(e.action,  [      6,     4,    2,    5,     3])
        self.assertNumpyEqual(e.returns, [112.624, 109.2, 60.0, 47.2, -40.0])

    def test_pengs_renormalized_lambda(self):
        replay_memory = LambdaReplayMemory(size=20, history_len=1, discount=0.9, Lambda=0.5,
                                           renormalize=True, use_watkins=False)
        self.fill(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     [           0,     1,    2])
        self.assertNumpyEqual(e.action,  [           7,     0,    1])
        self.assertNumpyEqual(e.returns, [180.74285714, 176.0, 20.0])

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     [           3,       4,           5,     6,     7])
        self.assertNumpyEqual(e.action,  [           6,       4,           2,     5,     3])
        self.assertNumpyEqual(e.returns, [188.58632258, 158.976, 78.51428571, 148.0, -40.0])

    def test_watkins_renormalized_lambda(self):
        replay_memory = LambdaReplayMemory(size=20, history_len=1, discount=0.9, Lambda=0.5,
                                           renormalize=True, use_watkins=True)
        self.fill(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     [    0,     1,    2])
        self.assertNumpyEqual(e.action,  [    7,     0,    1])
        self.assertNumpyEqual(e.returns, [208.0, 260.0, 20.0])

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     [           3,     4,    5,     6,     7])
        self.assertNumpyEqual(e.action,  [           6,     4,    2,     5,     3])
        self.assertNumpyEqual(e.returns, [199.08571429, 168.0, 60.0, 148.0, -40.0])
