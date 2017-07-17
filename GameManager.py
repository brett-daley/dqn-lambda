from PursuitEvader import PursuitEvader
from TargetPursuit import TargetPursuit


class GameManager:
    def __init__(self, cfg_parser, sess):
        base_game_name = cfg_parser.get('root','base_game_name')
        self.games = list()

        # Variants of the base game
        variants_list = (variant for variant in cfg_parser.sections() if variant != 'root')
        self.n_game_variants =  0

        # Set up all game variants
        for variant in variants_list:
            self.n_game_variants += 1

            # Single agent target pursuit game
            if base_game_name == 'PursuitEvader':
                self.games.append(PursuitEvader(cfg_parser = cfg_parser, game_variant = variant, sess = sess))
            # Multiagent target pursuit game
            elif base_game_name == 'TargetPursuit':
                self.games.append(TargetPursuit(cfg_parser=cfg_parser, game_variant=variant, sess=sess))

            # Assuming actions and observation dimensions are consistent across games
            self.n_actions = self.games[0].agt.n_actions
            # self.dim_agt_obs = self.games[0].dim_agt_obs

        print self.n_game_variants
