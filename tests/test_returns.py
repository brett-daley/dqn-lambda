import unittest
import numpy as np
from replay_memory import NStepReplayMemory, LambdaReplayMemory


class TestCaseReturns(unittest.TestCase):
    def setUp(self):
        self.transitions = [
            # state, action, reward, done?, qvalue
            (0, 7,  10.0, False, 100.0),
            (1, 0, -10.0, False, 200.0),
            (2, 1,  20.0, True,  300.0),
            (3, 6, -20.0, False, 400.0),
            (4, 4,  30.0, False, 300.0),
            (5, 2, -30.0, False, 200.0),
            (6, 5,  40.0, False, 100.0),
            (7, 3, -40.0, True,  200.0),
        ]
        # Mock refresh function to return our custom qvalues and exploratory mask
        qvalues = np.array([t[4] for t in self.transitions])
        self.exp_mask = np.array([1.0 for _ in range(len(self.transitions))])  # By default, assume every action is on-policy: i.e. Peng's Q(λ)
        self.refresh = lambda s, a: (qvalues[s].reshape(-1), self.exp_mask[s].reshape(-1))
        # Now remove qvalues from transitions, because tests don't need them explicitly
        self.transitions = [(np.array(state), np.array(action), reward, done) for state, action, reward, done, _ in self.transitions]

    def prepare(self, replay_memory):
        replay_memory.register_refresh_func(self.refresh)

        for state, action, reward, done in self.transitions:
            replay_memory.store_frame(state)
            state = replay_memory.encode_recent_observation()
            replay_memory.store_effect(action, reward, done)

        replay_memory.refresh()

    def assertNumpyEqual(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        self.assertTrue(np.allclose(x - y, 0.0))

    def test_1step(self):
        replay_memory = NStepReplayMemory(
            size=20,
            history_len=1,
            discount=0.9,
            nsteps=1,
        )
        self.prepare(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     np.array([    0,     1,    2]))
        self.assertNumpyEqual(e.action,  np.array([    7,     0,    1]))
        self.assertNumpyEqual(e.returns, np.array([190.0, 260.0, 20.0]))

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     np.array([    3,     4,    5,     6,     7]))
        self.assertNumpyEqual(e.action,  np.array([    6,     4,    2,     5,     3]))
        self.assertNumpyEqual(e.returns, np.array([250.0, 210.0, 60.0, 220.0, -40.0]))

    def test_nstep(self):
        replay_memory = NStepReplayMemory(
            size=20,
            history_len=1,
            discount=0.9,
            nsteps=3,
        )
        self.prepare(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     np.array([   0,   1,    2]))
        self.assertNumpyEqual(e.action,  np.array([   7,   0,    1]))
        self.assertNumpyEqual(e.returns, np.array([17.2, 8.0, 20.0]))

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     np.array([   3,     4,     5,   6,     7]))
        self.assertNumpyEqual(e.action,  np.array([   6,     4,     2,   5,     3]))
        self.assertNumpyEqual(e.returns, np.array([55.6, 181.2, -26.4, 4.0, -40.0]))

    def test_pengs_lambda(self):
        replay_memory = LambdaReplayMemory(
            size=20,
            history_len=1,
            discount=0.9,
            Lambda=0.8,
        )
        self.prepare(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     np.array([     0,    1,    2]))
        self.assertNumpyEqual(e.action,  np.array([     7,    0,    1]))
        self.assertNumpyEqual(e.returns, np.array([88.048, 58.4, 20.0]))

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     np.array([         3,        4,      5,    6,     7]))
        self.assertNumpyEqual(e.action,  np.array([         6,        4,      2,    5,     3]))
        self.assertNumpyEqual(e.returns, np.array([92.9165056, 81.82848, 21.984, 47.2, -40.0]))

    def test_watkins_lambda(self):
        self.exp_mask = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])  # Suppose a_1 and a_5 are exploratory

        replay_memory = LambdaReplayMemory(
            size=20,
            history_len=1,
            discount=0.9,
            Lambda=0.8,
        )
        self.prepare(replay_memory)

        # First episode
        e = replay_memory.episodes[0]
        self.assertNumpyEqual(e.obs,     np.array([    0,     1,    2]))
        self.assertNumpyEqual(e.action,  np.array([    7,     0,    1]))
        self.assertNumpyEqual(e.returns, np.array([233.2, 260.0, 20.0]))

        # Second episode
        e = replay_memory.episodes[1]
        self.assertNumpyEqual(e.obs,     np.array([      3,     4,    5,    6,     7]))
        self.assertNumpyEqual(e.action,  np.array([      6,     4,    2,    5,     3]))
        self.assertNumpyEqual(e.returns, np.array([112.624, 109.2, 60.0, 47.2, -40.0]))
