from Atari import Atari


class GameManager:
    def __init__(self, cfg_parser, sess):
        self.game = Atari(cfg_parser=cfg_parser, sess=sess)
        self.n_actions = self.game.agt.n_actions
