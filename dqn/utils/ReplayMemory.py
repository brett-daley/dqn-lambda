import numpy as np


class ReplayMemory:
    def __init__(self, capacity, minibatch_size):
        self.capacity = capacity              # Maximum number of experiences to store
        self.minibatch_size = minibatch_size  # Number of traces to sample at once

        self.reset()

    def reset(self):
        self.memory = None      # Holds the experiences
        self.size = 0           # Number of experiences currently in the memory
        self.insertion_ptr = 0  # Index of the next experience to be overwritten

    def add(self, obs, action, reward, next_obs, terminal):
        if self.memory is None:
            self._initialize_memory(obs.shape)

        self.memory[self.insertion_ptr] = np.array([obs, action, reward, next_obs, terminal])

        if self.size < self.capacity:
            self.size += 1

        # Increment the insertion pointer, wrapping around if the end is reached
        self.insertion_ptr = (self.insertion_ptr + 1) % self.capacity

    def _initialize_memory(self, obs_shape):
        # Compute a blank/zero-filled agent experience
        # for zero-padding data when training RNN traces of varying length
        self.null_experience = [
            np.zeros(obs_shape), # Observation
            0,                   # Action
            0.0,                 # Reward
            np.zeros(obs_shape), # Next observation
            False                # Terminal signal
        ]

        # Pre-allocate the replay memory; this has better performance and we'll also know immediately if it's too big
        self.memory = np.array([self.null_experience] * self.capacity)

    def sample_traces(self, tracelength):
        sampled_traces = []
        truetracelengths = []

        for _ in range(self.minibatch_size):
            # Randomly sample a starting index for this trace
            start_idx = np.random.randint(1-tracelength, self.size)
            end_idx = start_idx + tracelength

            n_extra_experiences = 0

            if start_idx < 0:
                # Starting index is before first experience in memory
                # Since tensorflow sequence_length RNN doesn't support front-padding (yet),
                # keep the relevant experiences at front, and set the suffix padding appropriately
                n_extra_experiences = -start_idx  # Points with negative indices are all extra, this just is a quick way to count them
                start_idx = 0                     # Ignore points with negative index

            if end_idx > self.size:
                # Ending index is after final experience in memory
                n_extra_experiences += end_idx - self.size
                end_idx = self.size  # Ignore points beyond memory size

            # Add zero-padded trace to minibatch
            trace = self.memory[start_idx:end_idx]
            zero_padding = [self.null_experience]*n_extra_experiences
            sampled_traces.extend(trace)
            sampled_traces.extend(zero_padding)

            # Record the number of non-null experiences in this trace
            truetracelengths.append(end_idx-start_idx)

        minibatch = np.reshape(sampled_traces, [self.minibatch_size*tracelength, -1])

        return truetracelengths, minibatch
