"""DIAYN network modules (Flax linen)."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple
from jax.nn import initializers


class PolicyNetwork(nn.Module):
    """Gaussian policy: obs -> (mu, log_std) -> tanh-squashed action."""

    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]: # 2 linear layers, inspired by DIAYN-PyTorch
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=initializers.he_normal())(x)
            x = nn.relu(x)
        mu = nn.Dense(self.action_dim, kernel_init=initializers.xavier_uniform())(x)
        log_std = nn.Dense(self.action_dim, kernel_init=initializers.xavier_uniform())(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mu, log_std

    def sample(self, params, obs: jnp.ndarray, key: jax.Array): # claude did this part tbh dont rly understand it
        """Sample action via reparameterization trick with tanh squashing.

        Returns (action, log_prob).
        """
        mu, log_std = self.apply(params, obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mu.shape)
        u = mu + std * noise
        action = jnp.tanh(u)

        log_prob = -0.5 * (((u - mu) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = log_prob - jnp.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True)
        return action, log_prob


class QNetwork(nn.Module):
    """Q-value network: (obs, action) -> scalar Q-value."""

    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=initializers.he_normal())(x)
            x = nn.relu(x)
        return nn.Dense(1, kernel_init=initializers.xavier_uniform())(x)


class Discriminator(nn.Module):
    """Skill discriminator: raw_obs -> logits over skills."""

    n_skills: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=initializers.he_normal())(x)
            x = nn.relu(x)
        return nn.Dense(self.n_skills, kernel_init=initializers.xavier_uniform())(x)
