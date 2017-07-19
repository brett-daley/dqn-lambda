from GenericAgent import GenericAgent

class AgentDistilled(GenericAgent):
    def __init__(self, i_agt, dim_obs, n_actions):
        super(self.__class__, self).__init__(i_agt = i_agt, dim_obs = dim_obs, n_actions = n_actions)

    def get_obs(self):
        pass