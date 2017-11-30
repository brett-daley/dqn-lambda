import numpy as np


class ReplayMemory:
    def __init__(self, config, obs_shape, recurrent_mode, forced_capacity=None):
        self.recurrent_mode = recurrent_mode
        self.agent_history_length = int(config.get('agent', 'history_length'))
        self.batch_size = int(config.get('replay_memory', 'batch_size'))
        self.capacity = forced_capacity if forced_capacity else int(config.get('replay_memory', 'capacity'))
        self.obs_dtype = config.get('replay_memory', 'obs_dtype')

        self.size = 0                   # Number of experiences currently in the memory
        self.insertion_ptr = 0          # Index of the next experience to be overwritten
        self.ready_for_outcome = False  # Ensures observation's outcome is saved before incrementing the insertion pointer

        # Pre-allocate the replay memory; this has better performance and we'll also know immediately if it's too big
        self.obs = np.zeros([self.capacity] + list(obs_shape), dtype=getattr(np, self.obs_dtype))
        self.action = np.zeros([self.capacity], dtype=np.int32)
        self.reward = np.zeros([self.capacity], dtype=np.float32)
        self.terminal = np.zeros([self.capacity], dtype=np.bool)

    def save_obs(self, obs):
        assert not self.ready_for_outcome
        self.ready_for_outcome = True

        self.obs[self.insertion_ptr] = obs

    def save_outcome(self, action, reward, terminal):
        assert self.ready_for_outcome
        self.ready_for_outcome = False

        self.action[self.insertion_ptr] = action
        self.reward[self.insertion_ptr] = reward
        self.terminal[self.insertion_ptr] = terminal

        self._update_insertion_ptr()

    def _update_insertion_ptr(self):
        # Increment size unless capacity is reached
        if self.size < self.capacity:
            self.size += 1

        # Increment the insertion pointer, wrapping around if the end is reached
        self.insertion_ptr = (self.insertion_ptr + 1) % self.capacity

    def append_history_to_obs(self, obs):
        starting_index = self.insertion_ptr - self.agent_history_length
        offsets = range(self.agent_history_length)

        obs = np.concatenate([self.obs[starting_index+j] for j in offsets], axis=-1)
        return obs

    def sample_minibatch(self):
        starting_indices = np.random.choice(self.size - self.agent_history_length, self.batch_size)
        offsets = range(self.agent_history_length)

        obs_batch = np.stack([np.stack([self.obs[i+j] for i in starting_indices]) for j in offsets])
        action_batch = np.stack([self.action[i+self.agent_history_length-1] for i in starting_indices])
        reward_batch = np.stack([self.reward[i+self.agent_history_length-1] for i in starting_indices])
        next_obs_batch = np.stack([np.stack([self.obs[i+j+1] for i in starting_indices]) for j in offsets])
        non_terminal_multiplier = 1 - np.stack([self.terminal[i+self.agent_history_length-1] for i in starting_indices])

        axis = 0 if self.recurrent_mode else -1
        obs_batch = np.concatenate(obs_batch, axis)
        next_obs_batch = np.concatenate(next_obs_batch, axis)

        return obs_batch, action_batch, reward_batch, next_obs_batch, non_terminal_multiplier
