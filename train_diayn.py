"""Train DIAYN on a brax environment."""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import brax.envs
from diayn import DIAYNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ant", help="brax env name")
    parser.add_argument("--n_skills", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    env = brax.envs.create(args.env)
    obs_dim = env.observation_size
    action_dim = env.action_size

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    agent = DIAYNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_skills=args.n_skills,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    key = jax.random.PRNGKey(args.seed)
    key, reset_key, skill_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    state = v_reset(reset_keys)
    skills = jax.random.randint(skill_key, (args.num_envs,), 0, args.n_skills)

    total_steps = 0
    while total_steps < args.max_steps:
        key, action_key, skill_key = jax.random.split(key, 3)

        obs = state.obs
        actions = agent.choose_action_batch(obs, skills)

        next_state = v_step(state, actions)
        next_obs = next_state.obs
        dones = next_state.done

        agent.store_batch(obs, actions, next_obs, skills, dones)

        new_skills = jax.random.randint(skill_key, (args.num_envs,), 0, args.n_skills)
        skills = jnp.where(dones, new_skills, skills)

        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, args.num_envs)
        reset_state = v_reset(reset_keys)
        state = jax.tree.map(
            lambda r, s: jnp.where(
                dones.reshape(-1, *([1] * (r.ndim - 1))), r, s
            ),
            reset_state, next_state,
        )

        metrics = agent.train()
        total_steps += args.num_envs

        if total_steps % (args.log_interval - args.log_interval % args.num_envs) < args.num_envs and metrics is not None:
            print(
                f"step={total_steps} "
                f"disc_loss={metrics['disc_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"mean_reward={metrics['mean_reward']:.4f}"
            )

    print(f"Training complete. {total_steps} steps.")


if __name__ == "__main__":
    main()
