"""DIAYN agent: SAC + skill discriminator with intrinsic reward."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from .model import PolicyNetwork, QNetwork, Discriminator
from .replay_memory import ReplayBuffer


class DIAYNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_skills: int = 10,
        hidden_dims=(256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_skills = n_skills
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.log_p_z = -jnp.log(jnp.float32(n_skills))

        self.rng = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)

        aug_obs_dim = obs_dim + n_skills

        self.policy_net = PolicyNetwork(action_dim=action_dim, hidden_dims=hidden_dims)
        self.q1_net = QNetwork(hidden_dims=hidden_dims)
        self.q2_net = QNetwork(hidden_dims=hidden_dims)
        self.discriminator = Discriminator(n_skills=n_skills, hidden_dims=hidden_dims)

        dummy_aug_obs = jnp.zeros((1, aug_obs_dim))
        dummy_raw_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
        self.policy_params = self.policy_net.init(k1, dummy_aug_obs)
        self.q1_params = self.q1_net.init(k2, dummy_aug_obs, dummy_action)
        self.q2_params = self.q2_net.init(k3, dummy_aug_obs, dummy_action)
        self.target_q1_params = self.q1_params
        self.target_q2_params = self.q2_params
        self.disc_params = self.discriminator.init(k4, dummy_raw_obs)

        self.policy_optimizer = optax.adam(lr)
        self.q1_optimizer = optax.adam(lr)
        self.q2_optimizer = optax.adam(lr)
        self.disc_optimizer = optax.adam(lr)

        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        self.q1_opt_state = self.q1_optimizer.init(self.q1_params)
        self.q2_opt_state = self.q2_optimizer.init(self.q2_params)
        self.disc_opt_state = self.disc_optimizer.init(self.disc_params)

        self.memory = ReplayBuffer(obs_dim, action_dim, buffer_size)

        self._train_step = jax.jit(self._train_step_impl)

    def _augment_obs(self, obs: np.ndarray, skill: int) -> np.ndarray:
        one_hot = np.zeros(self.n_skills, dtype=np.float32)
        one_hot[skill] = 1.0
        return np.concatenate([obs, one_hot])

    def _augment_obs_batch(self, obs: jnp.ndarray, skills: jnp.ndarray) -> jnp.ndarray:
        one_hot = jax.nn.one_hot(skills, self.n_skills)
        return jnp.concatenate([obs, one_hot], axis=-1)

    def choose_action(self, obs: np.ndarray, skill: int) -> np.ndarray:
        aug_obs = self._augment_obs(obs, skill)
        self.rng, key = jax.random.split(self.rng)
        action, _ = self.policy_net.sample(self.policy_params, aug_obs[None], key)
        return np.asarray(action[0])

    def choose_action_batch(self, obs: jnp.ndarray, skills: jnp.ndarray) -> jnp.ndarray:
        aug_obs = self._augment_obs_batch(obs, skills)
        self.rng, key = jax.random.split(self.rng)
        action, _ = self.policy_net.sample(self.policy_params, aug_obs, key)
        return action

    def store(self, obs, action, next_obs, skill, done):
        self.memory.add(obs, action, next_obs, skill, done)

    def store_batch(self, obs, action, next_obs, skill, done):
        self.memory.add_batch(
            np.asarray(obs), np.asarray(action), np.asarray(next_obs),
            np.asarray(skill), np.asarray(done),
        )

    def _train_step_impl(
        self,
        key,
        policy_params,
        q1_params,
        q2_params,
        target_q1_params,
        target_q2_params,
        disc_params,
        policy_opt_state,
        q1_opt_state,
        q2_opt_state,
        disc_opt_state,
        obs,
        actions,
        next_obs,
        skills,
        dones,
    ):
        alpha = self.alpha
        aug_obs = self._augment_obs_batch(obs, skills)
        aug_next_obs = self._augment_obs_batch(next_obs, skills)
        key_actor, key_critic = jax.random.split(key)

        def disc_loss_fn(dp):
            logits = self.discriminator.apply(dp, next_obs)
            return -jnp.mean(
                jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), skills]
            )

        disc_loss, disc_grads = jax.value_and_grad(disc_loss_fn)(disc_params)
        disc_updates, new_disc_opt_state = self.disc_optimizer.update(
            disc_grads, disc_opt_state, disc_params
        )
        new_disc_params = optax.apply_updates(disc_params, disc_updates)

        disc_logits = self.discriminator.apply(
            jax.lax.stop_gradient(new_disc_params), next_obs
        )
        log_q_z_s = jax.nn.log_softmax(disc_logits)[
            jnp.arange(disc_logits.shape[0]), skills
        ]
        reward = log_q_z_s - self.log_p_z

        def critic_loss_fn(q1p, q2p):
            next_action, next_log_prob = self.policy_net.sample(
                policy_params, aug_next_obs, key_critic
            )
            target_q1 = self.q1_net.apply(target_q1_params, aug_next_obs, next_action)
            target_q2 = self.q2_net.apply(target_q2_params, aug_next_obs, next_action)
            target_q = jnp.minimum(target_q1, target_q2) - alpha * next_log_prob
            target = reward[:, None] + self.gamma * (1.0 - dones[:, None]) * target_q

            q1 = self.q1_net.apply(q1p, aug_obs, actions)
            q2 = self.q2_net.apply(q2p, aug_obs, actions)
            q1_loss = jnp.mean((q1 - jax.lax.stop_gradient(target)) ** 2)
            q2_loss = jnp.mean((q2 - jax.lax.stop_gradient(target)) ** 2)
            return q1_loss + q2_loss

        critic_loss, (q1_grads, q2_grads) = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1)
        )(q1_params, q2_params)
        q1_updates, new_q1_opt_state = self.q1_optimizer.update(
            q1_grads, q1_opt_state, q1_params
        )
        q2_updates, new_q2_opt_state = self.q2_optimizer.update(
            q2_grads, q2_opt_state, q2_params
        )
        new_q1_params = optax.apply_updates(q1_params, q1_updates)
        new_q2_params = optax.apply_updates(q2_params, q2_updates)

        def actor_loss_fn(pp):
            new_action, log_prob = self.policy_net.sample(pp, aug_obs, key_actor)
            q1 = self.q1_net.apply(new_q1_params, aug_obs, new_action)
            q2 = self.q2_net.apply(new_q2_params, aug_obs, new_action)
            q = jnp.minimum(q1, q2)
            return jnp.mean(alpha * log_prob - q)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(policy_params)
        actor_updates, new_policy_opt_state = self.policy_optimizer.update(
            actor_grads, policy_opt_state, policy_params
        )
        new_policy_params = optax.apply_updates(policy_params, actor_updates)

        new_target_q1 = jax.tree.map(
            lambda t, s: t * (1 - self.tau) + s * self.tau,
            target_q1_params, new_q1_params,
        )
        new_target_q2 = jax.tree.map(
            lambda t, s: t * (1 - self.tau) + s * self.tau,
            target_q2_params, new_q2_params,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "disc_loss": disc_loss,
            "mean_reward": jnp.mean(reward),
        }

        return (
            new_policy_params,
            new_q1_params,
            new_q2_params,
            new_target_q1,
            new_target_q2,
            new_disc_params,
            new_policy_opt_state,
            new_q1_opt_state,
            new_q2_opt_state,
            new_disc_opt_state,
            metrics,
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size, self.np_rng)
        self.rng, key = jax.random.split(self.rng)

        (
            self.policy_params,
            self.q1_params,
            self.q2_params,
            self.target_q1_params,
            self.target_q2_params,
            self.disc_params,
            self.policy_opt_state,
            self.q1_opt_state,
            self.q2_opt_state,
            self.disc_opt_state,
            metrics,
        ) = self._train_step(
            key,
            self.policy_params,
            self.q1_params,
            self.q2_params,
            self.target_q1_params,
            self.target_q2_params,
            self.disc_params,
            self.policy_opt_state,
            self.q1_opt_state,
            self.q2_opt_state,
            self.disc_opt_state,
            jnp.array(batch["obs"]),
            jnp.array(batch["action"]),
            jnp.array(batch["next_obs"]),
            jnp.array(batch["skill"]),
            jnp.array(batch["done"]),
        )

        return {k: float(v) for k, v in metrics.items()}
