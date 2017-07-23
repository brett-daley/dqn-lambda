from PursuitEvader import PursuitEvader
from TargetPursuit import TargetPursuit


class GameManager:
    def __init__(self, cfg_parser, sess):
        base_game_name = cfg_parser.get('root','base_game_name')

        # Single agent target pursuit game
        if base_game_name == 'PursuitEvader':
            self.game = PursuitEvader(cfg_parser=cfg_parser, game_variant='root', sess=sess)
        # Multiagent target pursuit game
        elif base_game_name == 'TargetPursuit':
            self.game = TargetPursuit(cfg_parser=cfg_parser, game_variant='root', sess=sess)

        self.n_actions = self.game.agt.n_actions
