import numpy as np
from tests.test_replay_memory import TestCaseCore
from replay_memory_legacy import make_legacy_replay_memory


class TestCaseLegacyReplayMemory(TestCaseCore):
    def setUp(self):
        super().setUp()
        # Remove mask information from the refresh function
        refresh = self.refresh
        self.refresh = lambda s: refresh(s, None)[0]

    def test_1step(self):
        m = make_legacy_replay_memory('nstep-1', capacity=20, history_len=1, discount=0.9)
        self.fill(m)
        obs, actions, returns = m._sample(np.arange(0, 9))

        self.assertNumpyEqual(obs,     [    0,     1,    2,     3,     4,    5,     6,     7,     8])
        self.assertNumpyEqual(actions, [    7,     0,    1,     6,     4,    2,     5,     9,     7])
        self.assertNumpyEqual(returns, [190.0, 260.0, 20.0, 250.0, 210.0, 60.0, 220.0, -40.0, 410.0])

    def test_nstep(self):
        m = make_legacy_replay_memory('nstep-3', capacity=20, history_len=1, discount=0.9)
        self.fill(m)
        obs, actions, returns = m._sample(np.arange(0, 7))

        self.assertNumpyEqual(obs,     [   0,   1,    2,    3,     4,     5,   6])
        self.assertNumpyEqual(actions, [   7,   0,    1,    6,     4,     2,   5])
        self.assertNumpyEqual(returns, [17.2, 8.0, 20.0, 55.6, 181.2, -26.4, 4.0])
