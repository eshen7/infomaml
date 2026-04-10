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
    parser.add_argument("--log_interval", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile", type=str, default=None, help="path to save JAX trace (e.g. /tmp/jax-trace)")
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

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

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
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key, reset_key, skill_key = jax.random.split(key, 4)

    training_state = agent.init_state(init_key)
    buffer_state = agent.buffer.init()

    reset_keys = jax.random.split(reset_key, args.num_envs)
    env_state = v_reset(reset_keys)
    skills = jax.random.randint(skill_key, (args.num_envs,), 0, args.n_skills)

    if args.debug:
        print(f"Env: {args.env}, obs_dim={obs_dim}, action_dim={action_dim}")
        print(f"State obs shape: {env_state.obs.shape}")

    prefill_steps = max(1, args.batch_size // args.num_envs) # prefill so dont have to do more conditionals
    for _ in range(prefill_steps):
        key, action_key, skill_key = jax.random.split(key, 3)
        actions = jax.random.uniform(action_key, (args.num_envs, action_dim), minval=-1.0, maxval=1.0)
        next_env_state = v_step(env_state, actions)
        buffer_state = agent.buffer.insert(
            buffer_state, env_state.obs, actions, next_env_state.obs, skills, next_env_state.done
        )
        new_skills = jax.random.randint(skill_key, (args.num_envs,), 0, args.n_skills)
        skills = jnp.where(next_env_state.done, new_skills, skills)
        env_state = next_env_state

    if args.debug:
        print(f"Prefilled buffer with {prefill_steps * args.num_envs} transitions")

    def scan_step(carry, _):
        ts, es, bs, sk = carry
        key, action_key, sample_key, skill_key = jax.random.split(ts.rng, 4)

        actions = agent.choose_action(ts, es.obs, sk, action_key)

        next_es = v_step(es, actions)

        bs = agent.buffer.insert(bs, es.obs, actions, next_es.obs, sk, next_es.done)

        batch = agent.buffer.sample(bs, sample_key)
        ts = ts.replace(rng=key)
        ts, metrics = agent.train_step(
            ts, batch["obs"], batch["action"], batch["next_obs"], batch["skill"], batch["done"]
        )

        new_skills = jax.random.randint(skill_key, (sk.shape[0],), 0, agent.n_skills)
        sk = jnp.where(next_es.done, new_skills, sk)

        return (ts, next_es, bs, sk), metrics

    steps_per_scan = args.log_interval - args.log_interval % args.num_envs
    steps_per_scan = max(steps_per_scan, args.num_envs)
    scan_length = steps_per_scan // args.num_envs

    total_steps = prefill_steps * args.num_envs
    t_start = time.time()

    if args.debug:
        print(f"Scan length: {scan_length} iters ({steps_per_scan} env steps per log)")
        print("Starting training...")

    def run_epoch(carry):
        return jax.lax.scan(scan_step, carry, None, length=scan_length)

    run_epoch_jit = jax.jit(run_epoch)

    if args.profile:
        print("Warming up JIT...")
        carry = (training_state, env_state, buffer_state, skills)
        (training_state, env_state, buffer_state, skills), _ = run_epoch_jit(carry)
        jax.block_until_ready(training_state.rng)
        total_steps += steps_per_scan
        print(f"JIT warm-up done. Starting profiled epochs -> {args.profile}")

    profile_epochs = 3
    epoch_num = 0

    while total_steps < args.max_steps:
        carry = (training_state, env_state, buffer_state, skills)

        profiling = args.profile and epoch_num < profile_epochs
        if profiling:
            jax.profiler.start_trace(args.profile)

        (training_state, env_state, buffer_state, skills), all_metrics = run_epoch_jit(carry)

        if profiling:
            jax.block_until_ready(training_state.rng)
            jax.profiler.stop_trace()

        total_steps += steps_per_scan
        epoch_num += 1

        avg_metrics = jax.tree.map(lambda x: float(jnp.mean(x)), all_metrics)

        elapsed = time.time() - t_start
        sps = total_steps / elapsed
        msg = (
            f"step={total_steps} "
            f"disc_loss={avg_metrics['disc_loss']:.4f} "
            f"critic_loss={avg_metrics['critic_loss']:.4f} "
            f"actor_loss={avg_metrics['actor_loss']:.4f} "
            f"mean_reward={avg_metrics['mean_reward']:.4f}"
        )
        if args.debug:
            msg += f" sps={sps:.0f}"
            if any(np.isnan(v) for v in avg_metrics.values()):
                msg += " *** NaN DETECTED ***"
        if profiling:
            msg += " [profiled]"
        print(msg)

    elapsed = time.time() - t_start
    print(f"Training complete. {total_steps} steps in {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s).")


if __name__ == "__main__":
    main()
