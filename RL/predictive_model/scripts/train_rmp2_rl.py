#!/usr/bin/env python3
# train_rmp2_rl.py
# =============================================================================
# PPO Training Script for RMP2-RL
# =============================================================================
# Trains the MLPPolicyNet to predict optimal RMP2 gain parameters
# using PPO algorithm in IsaacSim simulation
#
# Usage:
#   python train_rmp2_rl.py
#   python train_rmp2_rl.py --total_steps 500000
#   python train_rmp2_rl.py --eval_only --checkpoint ./trained_models/policy_epoch_100.pt

import os
import sys
import argparse
import time
import numpy as np
import torch
import wandb

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rmp2_env_isaacsim import RMP2TrainingEnv
from rmp2_rl_policy import PPOAgent, RolloutStorage


# ==============================================================================
# Training Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train RMP2-RL Policy with PPO')
    
    # Training steps
    parser.add_argument('--total_steps', type=int, default=1000000,
                        help='Total training steps (default: 1000000)')
    parser.add_argument('--rollout_steps', type=int, default=2048,
                        help='Steps collected per rollout (default: 2048)')
    parser.add_argument('--n_epochs', type=int, default=8,
                        help='PPO epochs per rollout (default: 8)')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size (default: 64)')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clip parameter (default: 0.2)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda (default: 0.95)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm (default: 0.5)')
    
    # Logging
    parser.add_argument('--wandb_project', type=str, default='rmp2-rl',
                        help='WandB project name (default: rmp2-rl)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WandB run name (default: auto-generated)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Logging interval in epochs (default: 1)')
    
    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Epochs between model saves (default: 50)')
    parser.add_argument('--save_dir', type=str, default='./trained_models',
                        help='Directory to save models (default: ./trained_models)')
    parser.add_argument('--save_start', type=int, default=10,
                        help='First epoch to save (default: 10)')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Epochs between evaluation runs (default: 10)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    
    # Environment
    parser.add_argument('--stage_path', type=str, default=None,
                        help='Path to USD stage')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run IsaacSim headless (default: True)')
    parser.add_argument('--no_headless', action='store_true',
                        help='Run IsaacSim with GUI')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    
    # Eval only mode
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    return parser.parse_args()


# ==============================================================================
# Training Functions
# ==============================================================================

def collect_rollout(env, agent, rollout_storage, max_steps):
    """Collect rollout data from environment.
    
    Args:
        env: RMP2TrainingEnv instance
        agent: PPOAgent instance
        rollout_storage: RolloutStorage instance
        max_steps: Maximum steps to collect
    
    Returns:
        episode_rewards: List of episode returns
        episode_lengths: List of episode lengths
        success_count: Number of successful episodes
        collision_count: Number of collision episodes
    """
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0.0
    episode_length = 0
    success_count = 0
    collision_count = 0
    
    obs, _ = env.reset(seed=np.random.randint(0, 10000))
    done = False
    
    for step in range(max_steps):
        # Get action from policy
        action, log_prob, value = agent.get_action(obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        rollout_storage.add(
            obs, action, reward, log_prob, value, float(done)
        )
        
        # Accumulate episode reward
        episode_reward += reward
        episode_length += 1
        
        # Move to next state
        obs = next_obs
        
        # Episode ended
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if terminated and 'success' in info and info['success']:
                success_count += 1
            if terminated and 'collision' in info and info['collision']:
                collision_count += 1
            
            # Reset for next episode
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
    
    return episode_rewards, episode_lengths, success_count, collision_count


def evaluate(env, agent, n_episodes=10):
    """Evaluate agent performance.
    
    Args:
        env: RMP2TrainingEnv instance
        agent: PPOAgent instance
        n_episodes: Number of evaluation episodes
    
    Returns:
        dict: Evaluation metrics
    """
    eval_rewards = []
    eval_lengths = []
    eval_successes = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Use deterministic policy for evaluation
            action, _, _ = agent.get_action(obs, deterministic=True)
            
            # Step (use clamped gains automatically in env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Safety: max 1000 steps
            if episode_length >= 1000:
                break
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        
        if terminated and 'success' in info and info['success']:
            eval_successes += 1
    
    return {
        'eval_return_mean': np.mean(eval_rewards),
        'eval_return_std': np.std(eval_rewards),
        'eval_length_mean': np.mean(eval_lengths),
        'eval_length_std': np.std(eval_lengths),
        'eval_success_rate': eval_successes / n_episodes,
    }


def train(args):
    """Main training loop."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize WandB
    run_name = args.wandb_run_name or f"rmp2-rl-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )
    
    # Initialize environment
    print("Initializing IsaacSim environment...")
    env = RMP2TrainingEnv(
        headless=not args.no_headless,
        stage_path=args.stage_path
    )
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Initialize agent
    print("Initializing PPO agent...")
    agent = PPOAgent(
        state_dim=23,
        action_dim=11,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)
    
    # Rollout storage
    rollout_storage = RolloutStorage(
        capacity=args.rollout_steps,
        state_dim=23,
        action_dim=11,
        device=agent.device
    )
    
    # Training loop
    print(f"\nStarting training for {args.total_steps} steps...")
    print(f"Rollout steps per epoch: {args.rollout_steps}")
    print(f"PPO epochs per update: {args.n_epochs}")
    
    total_steps = 0
    epoch = 0
    start_time = time.time()
    
    while total_steps < args.total_steps:
        epoch += 1
        
        # Collect rollout
        rollout_storage.reset()
        episode_rewards, episode_lengths, success_count, collision_count = collect_rollout(
            env, agent, rollout_storage, args.rollout_steps
        )
        
        total_steps += args.rollout_steps
        
        # Update policy
        metrics = agent.update(rollout_storage.get_all())
        
        # Compute additional metrics
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        total_episodes = len(episode_rewards)
        success_rate = success_count / max(total_episodes, 1)
        
        # Epoch time
        epoch_time = time.time() - start_time
        
        # Logging
        if epoch % args.log_interval == 0:
            log_dict = {
                'epoch': epoch,
                'total_steps': total_steps,
                'episode_return_mean': mean_reward,
                'episode_length_mean': mean_length,
                'success_rate': success_rate,
                'collision_count': collision_count,
                'policy_loss': metrics['policy_loss'],
                'value_loss': metrics['value_loss'],
                'entropy_loss': metrics['entropy_loss'],
                'total_loss': metrics['total_loss'],
                'learning_rate': metrics['learning_rate'],
                'fps': args.rollout_steps / (epoch_time / epoch),
                'epoch_time': epoch_time,
            }
            wandb.log(log_dict, step=total_steps)
            
            print(f"Epoch {epoch} | Steps {total_steps} | "
                  f"Return {mean_reward:.2f} | Length {mean_length:.0f} | "
                  f"Success {success_rate:.1%} | "
                  f"Policy Loss {metrics['policy_loss']:.4f} | "
                  f"Value Loss {metrics['value_loss']:.4f}")
        
        # Evaluation
        if epoch % args.eval_freq == 0 and epoch >= args.eval_freq:
            eval_metrics = evaluate(env, agent, args.eval_episodes)
            
            eval_log = {
                f'eval/{k}': v for k, v in eval_metrics.items()
            }
            wandb.log(eval_log, step=total_steps)
            
            print(f"  Eval: Return {eval_metrics['eval_return_mean']:.2f} +/- "
                  f"{eval_metrics['eval_return_std']:.2f} | "
                  f"Success {eval_metrics['eval_success_rate']:.1%}")
        
        # Save checkpoint
        if epoch % args.save_freq == 0 and epoch >= args.save_start:
            save_path = os.path.join(
                args.save_dir,
                f"policy_epoch_{epoch}.pt"
            )
            agent.save(save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    # Final save
    final_path = os.path.join(args.save_dir, "policy_final.pt")
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Cleanup
    env.close()
    wandb.finish()
    
    return agent


def evaluate_pretrained(args):
    """Run evaluation on a pretrained model."""
    
    if not args.checkpoint:
        print("Error: --checkpoint required for evaluation")
        sys.exit(1)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize WandB
    run_name = args.wandb_run_name or f"rmp2-rl-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )
    
    # Initialize environment
    print("Initializing IsaacSim environment...")
    env = RMP2TrainingEnv(
        headless=not args.no_headless,
        stage_path=args.stage_path
    )
    
    # Initialize agent
    print("Initializing PPO agent...")
    agent = PPOAgent(state_dim=23, action_dim=11)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)
    
    # Run evaluation
    print(f"\nRunning evaluation for {args.eval_episodes} episodes...")
    eval_metrics = evaluate(env, agent, args.eval_episodes)
    
    print("\n=== Evaluation Results ===")
    print(f"Return: {eval_metrics['eval_return_mean']:.2f} +/- {eval_metrics['eval_return_std']:.2f}")
    print(f"Length: {eval_metrics['eval_length_mean']:.0f} +/- {eval_metrics['eval_length_std']:.0f}")
    print(f"Success Rate: {eval_metrics['eval_success_rate']:.1%}")
    
    # Log to WandB
    eval_log = {f'eval/{k}': v for k, v in eval_metrics.items()}
    wandb.log(eval_log)
    
    # Cleanup
    env.close()
    wandb.finish()
    
    return eval_metrics


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    if args.eval_only:
        # Evaluation mode
        evaluate_pretrained(args)
    else:
        # Training mode
        train(args)