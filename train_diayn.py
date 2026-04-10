"""Train DIAYN on a brax environment."""

import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import brax.envs
from diayn import DIAYNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ant", help="brax env name")
    parser.add_argument("--n_skills", type=int, default=8)
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        print("=== Debug Info ===")
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}")
        try:
            import jaxlib
            print(f"jaxlib version: {jaxlib.__version__}")
        except Exception:
            pass
        print(f"Num envs: {args.num_envs}, Batch size: {args.batch_size}")
        print("==================")

    env = brax.envs.create(
        args.env,
        episode_length=args.episode_length,
        exclude_current_positions_from_observation=False,
    )
    obs_dim = env.observation_size
    action_dim = env.action_size

    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    agent = DIAYNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_skills=args.n_skills,
        hidden_dims=(300, 300),
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

    if args.debug:
        print(f"Env: {args.env}, obs_dim={obs_dim}, action_dim={action_dim}")
        print(f"State obs shape: {state.obs.shape}")
        print(f"State obs device: {state.obs.devices()}")
        print(f"Skills shape: {skills.shape}")

    total_steps = 0
    t_start = time.time()
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

        state = next_state

        metrics = agent.train()
        total_steps += args.num_envs

        if args.debug and total_steps == args.num_envs:
            jax.block_until_ready(actions)
            print(f"First step took {time.time() - t_start:.2f}s (includes JIT compilation)")
            print(f"Actions shape: {actions.shape}, device: {actions.devices()}")
            if metrics:
                for k, v in metrics.items():
                    print(f"  {k}: {v} (nan={np.isnan(v)})")

        if args.debug and total_steps == args.num_envs * 2 and metrics:
            t_second = time.time()
            print(f"Second step took {t_second - t_start:.2f}s total")
            for k, v in metrics.items():
                print(f"  {k}: {v} (nan={np.isnan(v)})")

        if total_steps % (args.log_interval - args.log_interval % args.num_envs) < args.num_envs and metrics is not None:
            elapsed = time.time() - t_start
            sps = total_steps / elapsed
            msg = (
                f"step={total_steps} "
                f"disc_loss={metrics['disc_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"mean_reward={metrics['mean_reward']:.4f}"
            )
            if args.debug:
                msg += f" sps={sps:.0f}"
                if any(np.isnan(v) for v in metrics.values()):
                    msg += " *** NaN DETECTED ***"
            print(msg)

    elapsed = time.time() - t_start
    print(f"Training complete. {total_steps} steps in {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s).")


if __name__ == "__main__":
    main()
