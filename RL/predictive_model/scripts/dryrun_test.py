#!/usr/bin/env python3
import sys, os, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/ucluser/VRWIT/RL/predictive_model/scripts')

# Redirect noisy IsaacSim stderr to /dev/null
import subprocess
devnull = open(os.devnull, 'w')
os.dup2(devnull.fileno(), 2)

# Write results to a file
OUT = open('/tmp/dryrun_results.txt', 'w')

def log(msg):
    OUT.write(msg + '\n')
    OUT.flush()
    # Also print to stdout (visible in terminal)
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

log("=== RMP2-RL Dry-Run ===\n")

log("[1] Creating env (headless)...")
t0 = time.time()
from rmp2_env_isaacsim import RMP2TrainingEnv
env = RMP2TrainingEnv(headless=True)
log(f"OK ({time.time()-t0:.1f}s)")

log("[2] Creating agent...")
from rmp2_rl_policy import PPOAgent, RolloutStorage
agent = PPOAgent(state_dim=23, action_dim=11)
log(f"Device: {agent.device}")

log("\n[3] Rollout (64 steps)...")
storage = RolloutStorage(capacity=64, state_dim=23, action_dim=11, device=agent.device)
obs, _ = env.reset()
episodes = 0
ep_reward = 0.0

t0 = time.time()
for step in range(64):
    try:
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        storage.add(obs, action, reward, log_prob, value, float(done))
        ep_reward += reward
        obs = next_obs
        if done:
            episodes += 1
            log(f"  Ep{episodes} step{step+1}: reward={ep_reward:.2f} "
                f"pos_err={info.get('pos_error',-1):.3f} success={info.get('success',False)}")
            obs, _ = env.reset()
            ep_reward = 0.0
    except Exception as e:
        log(f"  ERROR step {step}: {e}")
        import traceback
        traceback.print_exc(file=OUT)
        break

log(f"OK ({time.time()-t0:.1f}s)")

log("\n[4] PPO update...")
rollout = storage.get_all()
t0 = time.time()
metrics = agent.update(rollout)
log(f"Policy loss: {metrics['policy_loss']:.4f}")
log(f"Value loss:  {metrics['value_loss']:.4f}")
log(f"OK ({time.time()-t0:.1f}s)")

log("\n[5] Eval (3 eps)...")
successes = 0
import numpy as np
for ep in range(3):
    obs, _ = env.reset()
    ep_reward = 0.0
    steps = 0
    for s in range(500):
        action, _, _ = agent.get_action(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        obs = next_obs
        if terminated or truncated:
            if info.get('success'): successes += 1
            break
    log(f"  Ep{ep+1}: reward={ep_reward:.2f} steps={steps} "
        f"pos_err={info.get('pos_error',-1):.3f} success={info.get('success',False)}")
log(f"Success rate: {successes}/3")

log("\n[6] Shutting down...")
env.close()
log("OK")

log("\n=== DRY-RUN PASSED ===")
OUT.close()
