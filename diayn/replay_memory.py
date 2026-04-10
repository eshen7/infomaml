"""Functional JAX replay buffer for use inside jax.lax.scan."""

import jax
import jax.numpy as jnp
import flax


@flax.struct.dataclass
class BufferState:
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    skill: jnp.ndarray
    done: jnp.ndarray
    size: jnp.ndarray
    ptr: jnp.ndarray


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, max_size: int, batch_size: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.batch_size = batch_size

    def init(self) -> BufferState:
        return BufferState(
            obs=jnp.zeros((self.max_size, self.obs_dim)),
            action=jnp.zeros((self.max_size, self.action_dim)),
            next_obs=jnp.zeros((self.max_size, self.obs_dim)),
            skill=jnp.zeros(self.max_size, dtype=jnp.int32),
            done=jnp.zeros(self.max_size),
            size=jnp.int32(0),
            ptr=jnp.int32(0),
        )

    def insert(self, state: BufferState, obs, action, next_obs, skill, done) -> BufferState:
        n = obs.shape[0]
        idx = (state.ptr + jnp.arange(n)) % self.max_size
        return state.replace(
            obs=state.obs.at[idx].set(obs),
            action=state.action.at[idx].set(action),
            next_obs=state.next_obs.at[idx].set(next_obs),
            skill=state.skill.at[idx].set(skill),
            done=state.done.at[idx].set(done),
            ptr=(state.ptr + n) % self.max_size,
            size=jnp.minimum(state.size + n, self.max_size),
        )

    def sample(self, state: BufferState, key):
        idx = jax.random.randint(key, (self.batch_size,), 0, state.size)
        return {
            "obs": state.obs[idx],
            "action": state.action[idx],
            "next_obs": state.next_obs[idx],
            "skill": state.skill[idx],
            "done": state.done[idx],
        }
