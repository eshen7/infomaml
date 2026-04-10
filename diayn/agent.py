"""DIAYN agent: SAC + skill discriminator with intrinsic reward."""

import jax
import jax.numpy as jnp
import optax
import flax
from .model import PolicyNetwork, QNetwork, Discriminator
from .replay_memory import ReplayBuffer, BufferState


@flax.struct.dataclass
class TrainingState:
    policy_params: any
    q1_params: any
    q2_params: any
    target_q1_params: any
    target_q2_params: any
    disc_params: any
    policy_opt_state: any
    q1_opt_state: any
    q2_opt_state: any
    disc_opt_state: any
    rng: jnp.ndarray


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
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_skills = n_skills
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.log_p_z = -jnp.log(jnp.float32(n_skills))

        self.policy_net = PolicyNetwork(action_dim=action_dim, hidden_dims=hidden_dims)
        self.q1_net = QNetwork(hidden_dims=hidden_dims)
        self.q2_net = QNetwork(hidden_dims=hidden_dims)
        self.discriminator = Discriminator(n_skills=n_skills, hidden_dims=hidden_dims)

        self.policy_optimizer = optax.adam(lr)
        self.q1_optimizer = optax.adam(lr)
        self.q2_optimizer = optax.adam(lr)
        self.disc_optimizer = optax.adam(lr)

        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_size, batch_size)

    def init_state(self, key) -> TrainingState:
        k1, k2, k3, k4, rng = jax.random.split(key, 5)

        aug_obs_dim = self.obs_dim + self.n_skills
        dummy_aug_obs = jnp.zeros((1, aug_obs_dim))
        dummy_raw_obs = jnp.zeros((1, self.obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        policy_params = self.policy_net.init(k1, dummy_aug_obs)
        q1_params = self.q1_net.init(k2, dummy_aug_obs, dummy_action)
        q2_params = self.q2_net.init(k3, dummy_aug_obs, dummy_action)
        disc_params = self.discriminator.init(k4, dummy_raw_obs)

        return TrainingState(
            policy_params=policy_params,
            q1_params=q1_params,
            q2_params=q2_params,
            target_q1_params=q1_params,
            target_q2_params=q2_params,
            disc_params=disc_params,
            policy_opt_state=self.policy_optimizer.init(policy_params),
            q1_opt_state=self.q1_optimizer.init(q1_params),
            q2_opt_state=self.q2_optimizer.init(q2_params),
            disc_opt_state=self.disc_optimizer.init(disc_params),
            rng=rng,
        )

    def _augment_obs(self, obs, skills):
        one_hot = jax.nn.one_hot(skills, self.n_skills)
        return jnp.concatenate([obs, one_hot], axis=-1)

    def choose_action(self, training_state, obs, skills, key):
        aug_obs = self._augment_obs(obs, skills)
        action, _ = self.policy_net.sample(training_state.policy_params, aug_obs, key)
        return action

    def train_step(self, training_state, obs, actions, next_obs, skills, dones):
        key, key_actor, key_critic = jax.random.split(training_state.rng, 3)

        alpha = self.alpha
        aug_obs = self._augment_obs(obs, skills)
        aug_next_obs = self._augment_obs(next_obs, skills)

        policy_params = training_state.policy_params
        q1_params = training_state.q1_params
        q2_params = training_state.q2_params
        target_q1_params = training_state.target_q1_params
        target_q2_params = training_state.target_q2_params
        disc_params = training_state.disc_params

        def disc_loss_fn(dp):
            logits = self.discriminator.apply(dp, obs)
            return -jnp.mean(
                jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), skills]
            )

        disc_loss, disc_grads = jax.value_and_grad(disc_loss_fn)(disc_params)
        disc_updates, new_disc_opt_state = self.disc_optimizer.update(
            disc_grads, training_state.disc_opt_state, disc_params
        )
        new_disc_params = optax.apply_updates(disc_params, disc_updates)

        disc_logits = self.discriminator.apply(
            jax.lax.stop_gradient(disc_params), obs
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
            q1_grads, training_state.q1_opt_state, q1_params
        )
        q2_updates, new_q2_opt_state = self.q2_optimizer.update(
            q2_grads, training_state.q2_opt_state, q2_params
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
            actor_grads, training_state.policy_opt_state, policy_params
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

        new_state = training_state.replace(
            policy_params=new_policy_params,
            q1_params=new_q1_params,
            q2_params=new_q2_params,
            target_q1_params=new_target_q1,
            target_q2_params=new_target_q2,
            disc_params=new_disc_params,
            policy_opt_state=new_policy_opt_state,
            q1_opt_state=new_q1_opt_state,
            q2_opt_state=new_q2_opt_state,
            disc_opt_state=new_disc_opt_state,
            rng=key,
        )

        return new_state, metrics
