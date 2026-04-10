"""JAX on-device replay buffer with random sampling."""

import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.size = jnp.int32(0)
        self.ptr = jnp.int32(0)

        self.obs = jnp.zeros((max_size, obs_dim))
        self.action = jnp.zeros((max_size, action_dim))
        self.next_obs = jnp.zeros((max_size, obs_dim))
        self.skill = jnp.zeros(max_size, dtype=jnp.int32)
        self.done = jnp.zeros(max_size)

    def add_batch(self, obs, action, next_obs, skill, done):
        n = obs.shape[0]
        idx = (self.ptr + jnp.arange(n)) % self.max_size
        self.obs = self.obs.at[idx].set(obs)
        self.action = self.action.at[idx].set(action)
        self.next_obs = self.next_obs.at[idx].set(next_obs)
        self.skill = self.skill.at[idx].set(skill)
        self.done = self.done.at[idx].set(done)
        self.ptr = (self.ptr + n) % self.max_size
        self.size = jnp.minimum(self.size + n, self.max_size)

    def sample(self, batch_size: int, key: jax.Array):
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "next_obs": self.next_obs[idx],
            "skill": self.skill[idx],
            "done": self.done[idx],
        }

    def __len__(self):
        return int(self.size)
