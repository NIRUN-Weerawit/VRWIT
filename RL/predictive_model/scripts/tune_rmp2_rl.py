#!/usr/bin/env python3
# tune_rmp2_rl.py
# =============================================================================
# RMP2-RL Hyperparameter Tuning with Optuna
# =============================================================================
# Each trial: short PPO training run (~100K steps) with sampled hyperparams
# Uses shared IsaacSim environment across trials to avoid re-initialization
#
# Usage:
#   ~/isaacsim/python.sh tune_rmp2_rl.py --n_trials 50 --study_name rmp2-v1
#   ~/isaacsim/python.sh tune_rmp2_rl.py --n_trials 30 --pruning
#
# Output: best_params.json + Optuna SQLite study file

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rmp2_env_isaacsim import RMP2TrainingEnv
from rmp2_rl_policy import PPOAgent, RolloutStorage

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: ~/isaacsim/python.sh -m pip install optuna")
    sys.exit(1)


# ==============================================================================
# Args
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RMP2-RL: Optuna hyperparameter tuning")
    p.add_argument("--n_trials", type=int, default=50,
                   help="Number of Optuna trials (default: 50)")
    p.add_argument("--study_name", type=str, default="rmp2-rl-tuning",
                   help="Optuna study name (default: rmp2-rl-tuning)")
    p.add_argument("--db_path", type=str, default="./optuna_study.db",
                   help="SQLite DB path for resumable study (default: ./optuna_study.db)")
    p.add_argument("--trial_steps", type=int, default=102400,
                   help="Training steps per trial (default: 102400 = 50 PPO epochs)")
    p.add_argument("--rollout_steps", type=int, default=2048,
                   help="Rollout steps per epoch (default: 2048)")
    p.add_argument("--n_epochs", type=int, default=8,
                   help="PPO epochs per update (default: 8)")
    p.add_argument("--pruning", action="store_true", default=True,
                   help="Enable MedianPruner early stopping (default: True)")
    p.add_argument("--no_pruning", action="store_true",
                   help="Disable pruning")
    p.add_argument("--stage_path", type=str, default=None,
                   help="Path to USD stage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5,
                   help="Pruning: consecutive epochs without improvement before kill (default: 5)")
    return p.parse_args()


# ==============================================================================
# Search Space
# ==============================================================================

def sample_params(trial: optuna.Trial) -> dict:
    """Sample hyperparameters from the Optuna trial object."""
    params = {}

    # PPO core
    params["lr"] = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    params["clip_epsilon"] = trial.suggest_float("clip_epsilon", 0.05, 0.4)
    params["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    params["vf_coef"] = trial.suggest_float("vf_coef", 0.1, 2.0)
    params["gamma"] = trial.suggest_float("gamma", 0.95, 0.999)
    params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)
    params["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 2.0)

    # Architecture
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    params["hidden_dims"] = [hidden_size, hidden_size // 2]

    # Reward weights
    params["reward_pos"] = trial.suggest_float("reward_pos", 0.3, 3.0)
    params["reward_obstacle"] = trial.suggest_float("reward_obstacle", 1.0, 30.0)
    params["reward_smooth"] = trial.suggest_float("reward_smooth", 0.001, 0.1, log=True)
    params["reward_success"] = trial.suggest_float("reward_success", 1.0, 30.0)

    return params


# ==============================================================================
# Reward Weights Override
# ==============================================================================

def apply_reward_weights(env, params):
    """Patch environment reward weights from trial params."""
    import rmp2_env_isaacsim as env_mod
    env_mod.REWARD_WEIGHTS["pos"] = params["reward_pos"]
    env_mod.REWARD_WEIGHTS["obstacle"] = params["reward_obstacle"]
    env_mod.REWARD_WEIGHTS["smooth"] = params["reward_smooth"]
    env_mod.REWARD_WEIGHTS["success"] = params["reward_success"]


# ==============================================================================
# Trial Objective
# ==============================================================================

def run_trial(trial: optuna.Trial, env: RMP2TrainingEnv, params: dict,
              rollout_steps: int, n_epochs: int, max_steps: int,
              pruning: bool, patience: int) -> float:
    """Run one tuning trial. Logs header, runs training, returns score."""
    trial_start = time.time()

    print(f"\n--- Trial {trial.number} ---")
    print(f"    lr={params['lr']:.2e} clip={params['clip_epsilon']:.3f} "
          f"ent={params['ent_coef']:.1e} vf={params['vf_coef']:.2f}")
    print(f"    gamma={params['gamma']:.3f} gae={params['gae_lambda']:.3f} "
          f"grad={params['max_grad_norm']:.2f}")
    print(f"    bs={params['batch_size']} hidden={params['hidden_dims'][0]} "
          f"rw_pos={params['reward_pos']:.1f} rw_obs={params['reward_obstacle']:.1f} "
          f"rw_smooth={params['reward_smooth']:.3f}")

    # Actually run the trial
    score = _run_trial_impl(
        trial, env, params, rollout_steps, n_epochs, max_steps, pruning, patience)
    elapsed = time.time() - trial_start
    print(f"    Trial {trial.number}: score={score:.1f} "
          f"({elapsed:.0f}s, {max_steps // 1000}k steps)")
    return score

def _run_trial_impl(trial: optuna.Trial, env: RMP2TrainingEnv, params: dict,
              rollout_steps: int, n_epochs: int, max_steps: int,
              pruning: bool, patience: int) -> float:
    """Inner implementation: runs training loop, returns best eval reward."""
    """Inner implementation: runs training loop, returns best eval reward."""

    # Apply reward weights
    apply_reward_weights(env, params)

    # Create agent
    agent = PPOAgent(
        state_dim=23,
        action_dim=11,
        hidden_dims=params["hidden_dims"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_epsilon=params["clip_epsilon"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        lr=params["lr"],
        batch_size=params["batch_size"],
        n_epochs=n_epochs,
    )

    storage = RolloutStorage(capacity=rollout_steps, state_dim=23, action_dim=11, device=agent.device)
    total_steps = 0
    best_eval_reward = -float("inf")
    no_improve_count = 0
    running_rewards = []  # track for pruning

    while total_steps < max_steps:
        # --- Data collection ---
        storage.reset()
        ep_rewards = []
        ep_reward = 0.0
        obs, _ = env.reset()

        for _ in range(rollout_steps):
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            storage.add(obs, action, reward, log_prob, value, float(done))
            ep_reward += reward
            obs = next_obs

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        total_steps += rollout_steps

        # --- PPO update ---
        rollout = storage.get_all()
        metrics = agent.update(rollout)

        # --- Evaluation (every epoch for Optuna intermediate reporting) ---
        eval_total = 0.0
        successes = 0
        n_eps = 3
        for _ in range(n_eps):
            obs, _ = env.reset()
            for s in range(500):
                action, _, _ = agent.get_action(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                eval_total += reward
                obs = next_obs
                if terminated or truncated:
                    if info.get("success"):
                        successes += 1
                    break

        avg_reward = eval_total / n_eps
        success_rate = successes / n_eps

        # Track for pruning
        running_rewards.append(avg_reward)

        # Report to Optuna (intermediate value for pruning)
        trial.report(avg_reward, step=total_steps)

        # Check pruning
        if pruning and trial.should_prune():
            raise optuna.TrialPruned()

        # Track best
        if avg_reward > best_eval_reward:
            best_eval_reward = avg_reward
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Log (compact, single-line per epoch)
        mean_ep = np.mean(ep_rewards) if ep_rewards else -999
        print(f"  [T{trial.number:02d}] E{total_steps//rollout_steps:02d} "
              f"steps={total_steps} ret={mean_ep:.0f} "
              f"eval={avg_reward:.1f} succ={success_rate:.0%} "
              f"vl={metrics['value_loss']:.0f} pl={metrics['policy_loss']:.4f}")

        # Prune if stuck (same logic as MedianPruner but explicit)
        if no_improve_count >= patience and len(running_rewards) >= patience + 3:
            print(f"  [T{trial.number:02d}] PRUNED: no improvement for {patience} epochs (best={best_eval_reward:.1f})")
            raise optuna.TrialPruned()

    return best_eval_reward


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Create Optuna study with SQLite storage ---
    storage_url = f"sqlite:///{os.path.abspath(args.db_path)}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1,
    ) if args.pruning and not args.no_pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
    )

    print(f"=== Optuna Tuning: {args.study_name} ===\n")
    print(f"DB:       {os.path.abspath(args.db_path)}")
    print(f"Trials:   {args.n_trials}")
    print(f"Steps:    {args.trial_steps} per trial ({args.trial_steps // args.rollout_steps} PPO epochs)")
    print(f"Pruning:  {'ON (MedianPruner)' if args.pruning and not args.no_pruning else 'OFF'}")
    if args.pruning and not args.no_pruning:
        print(f"Patience: {args.patience} epochs")
    print()

    # --- Init IsaacSim environment (once, reused across trials) ---
    print("[00] Initializing IsaacSim (headless)...")
    t0 = time.time()
    env = RMP2TrainingEnv(headless=True, stage_path=args.stage_path)
    print(f"     OK ({time.time() - t0:.1f}s)\n")

    # Use an attribute on the function itself to pass env into the callback
    def objective(trial):
        params = sample_params(trial)
        trial_start = time.time()

        print(f"\n--- Trial {trial.number} ---")
        print(f"    lr={params['lr']:.2e} clip={params['clip_epsilon']:.3f} "
              f"ent={params['ent_coef']:.1e} vf={params['vf_coef']:.2f}")
        print(f"    gamma={params['gamma']:.3f} gae={params['gae_lambda']:.3f} "
              f"grad={params['max_grad_norm']:.2f}")
        print(f"    bs={params['batch_size']} hidden={params['hidden_dims'][0]} "
              f"rw_pos={params['reward_pos']:.1f} rw_obs={params['reward_obstacle']:.1f} "
              f"rw_smooth={params['reward_smooth']:.3f}")

        try:
            score = run_trial(
                trial=trial,
                env=env,
                params=params,
                rollout_steps=args.rollout_steps,
                n_epochs=args.n_epochs,
                max_steps=args.trial_steps,
                pruning=args.pruning and not args.no_pruning,
                patience=args.patience,
            )
            elapsed = time.time() - trial_start
            print(f"    Trial {trial.number}: score={score:.1f} "
                  f"({elapsed:.0f}s, {args.trial_steps // 1000}k steps)")
        except optuna.TrialPruned:
            elapsed = time.time() - trial_start
            print(f"    Trial {trial.number}: PRUNED ({elapsed:.0f}s)")
            raise


    # --- Run optimization ---
    print(f"\n{'='*60}")
    print(f"Starting {args.n_trials} trials...\n")

    try:
        study.optimize(lambda trial: run_trial(trial, env, sample_params(trial),
                                                args.rollout_steps, args.n_epochs, args.trial_steps,
                                                args.pruning and not args.no_pruning, args.patience),
                       n_trials=args.n_trials, show_progress_bar=False)
    finally:
        # --- Cleanup ---
        print(f"\n{'='*60}")
        print("Shutting down IsaacSim...")
        env.close()
        print("OK")

    # --- Results ---
    print(f"\n=== Best Trial ===")
    best = study.best_trial
    print(f"Trial #{best.number}: value={best.value:.2f}")
    print(f"Params:")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.6f}")
        else:
            print(f"  {k:25s} = {v}")

    # Save to JSON
    result = {
        "study_name": args.study_name,
        "best_value": best.value,
        "best_trial": best.number,
        "params": best.params,
    }
    out_path = os.path.join(os.path.dirname(__file__), "best_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved best params to: {out_path}")

    # Show Optuna visualizations if available
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(os.path.dirname(__file__), "optuna_history.html"))
        print("Optimization history saved to: optuna_history.html")
    except Exception:
        pass

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(os.path.dirname(__file__), "optuna_importance.html"))
        print("Parameter importances saved to: optuna_importance.html")
    except Exception:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
