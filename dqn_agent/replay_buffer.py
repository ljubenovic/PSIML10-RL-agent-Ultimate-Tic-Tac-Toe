import numpy as np
from collections import deque
from typing import Deque

class ReplayBuffer:
    """A numpy replay buffer for n-step learning."""

    def __init__(self, obs_dim, size, batch_size = 32, n_step = 3, gamma = 0.99):
        
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)

        self.actions_buf = np.zeros([size], dtype=np.float32)
        self.rewards_buf = np.zeros([size], dtype=np.float32)
        self.dones_buf = np.zeros(size, dtype=np.float32)

        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma


    def store_transition(self, obs, act, rew, next_obs, done):

        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            # single step transition is not ready
            return ()
        
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)  # make a n-step transition
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = act
        self.rewards_buf[self.ptr] = rew
        self.dones_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]


    def sample_batch(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.actions_buf[indices],
            rews=self.rewards_buf[indices],
            done=self.dones_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    

    def sample_batch_from_idxs(self, indices):
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.actions_buf[indices],
            rews=self.rewards_buf[indices],
            done=self.dones_buf[indices],
        )
    

    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float):
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        return rew, next_obs, done


    def __len__(self) -> int:
        return self.size

