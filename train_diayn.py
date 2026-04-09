"""Train DIAYN on a brax environment."""

import argparse
import jax
import numpy as np
import brax.envs
from diayn import DIAYNAgent


def make_env(env_name: str):
    return brax.envs.create(env_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ant", help="brax env name")
    parser.add_argument("--n_skills", type=int, default=10)
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

    env = make_env(args.env)
    obs_dim = env.observation_size
    action_dim = env.action_size

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

    rng = np.random.default_rng(args.seed)
    env_key = jax.random.PRNGKey(args.seed + 1)

    total_steps = 0
    episode_count = 0

    while total_steps < args.max_steps:
        skill = int(rng.integers(0, args.n_skills))

        env_key, reset_key = jax.random.split(env_key)
        state = env.reset(reset_key)
        obs = np.asarray(state.obs)

        episode_reward = 0.0
        for t in range(args.episode_length):
            action = agent.choose_action(obs, skill)
            state = env.step(state, jax.numpy.array(action))
            next_obs = np.asarray(state.obs)
            done = float(state.done)

            agent.store(obs, action, next_obs, skill, done)
            metrics = agent.train()

            obs = next_obs
            total_steps += 1

            if metrics is not None:
                episode_reward += metrics["mean_reward"]

            if total_steps % args.log_interval == 0 and metrics is not None:
                print(
                    f"step={total_steps} "
                    f"disc_loss={metrics['disc_loss']:.4f} "
                    f"critic_loss={metrics['critic_loss']:.4f} "
                    f"actor_loss={metrics['actor_loss']:.4f} "
                    f"mean_reward={metrics['mean_reward']:.4f}"
                )

            if done:
                break

        episode_count += 1

    print(f"Training complete. {total_steps} steps, {episode_count} episodes.")


if __name__ == "__main__":
    main()
