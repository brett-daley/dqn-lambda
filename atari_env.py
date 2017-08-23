import gym
from scipy.misc import imresize
from rllab.envs.gym_env import *


class AtariEnv(GymEnv):
    def __init__(self, env_name, screen_dims=(84, 84), record_video=True, video_schedule=None, log_dir=None, record_log=True, force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log('Warning: skipping Gym environment monitoring since snapshot_dir not configured.')
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), 'gym_log')
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._screen_dims = screen_dims
        self._observation_space = self._create_observation_space(env)
        logger.log("observation space: {}".format(self._observation_space))

        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))

        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset

    def reset(self):
        obs = super().reset()
        return self._preprocess(obs)

    def step(self, action):
        step = super().step(action)
        return step._replace(observation=self._preprocess(step.observation))

    def _create_observation_space(self, env):
        if len(env.observation_space.shape) == 1:
            # The input is the RAM; return the original observation space
            return convert_gym_space(env.observation_space)
        else:
            # The input is the screen; create a new observation space according to the desired dimensions
            obs_shape = list(self._screen_dims) + list(env.observation_space.shape[2:])
            space = gym.spaces.Box(low=0, high=255, shape=obs_shape)
            return convert_gym_space(space)

    def _preprocess(self, obs):
        return imresize(obs, size=self._screen_dims)
