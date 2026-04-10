"""Simple numpy replay buffer with random sampling."""

import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.skill = np.zeros(max_size, dtype=np.int32)
        self.done = np.zeros(max_size, dtype=np.float32)

    def add(self, obs, action, next_obs, skill, done): # circular buffer, ptr to head
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.skill[self.ptr] = skill
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, action, next_obs, skill, done):
        n = obs.shape[0]
        idx = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.obs[idx] = obs
        self.action[idx] = action
        self.next_obs[idx] = next_obs
        self.skill[idx] = skill
        self.done[idx] = done
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "next_obs": self.next_obs[idx],
            "skill": self.skill[idx],
            "done": self.done[idx],
        }

    def __len__(self):
        return self.size
