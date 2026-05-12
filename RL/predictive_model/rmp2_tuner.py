"""
RMP Gain Tuner - Grid search for optimal RMP2 solver gains.

Usage:
    python rmp2_tuner.py              # Full grid search (coarse + fine)
    python rmp2_tuner.py --quick      # Smaller grid (~1 min)
    python rmp2_tuner.py --step       # Manual step-by-step tuning
    python rmp2_tuner.py --test "40,20,10,12"  # Single config test
    python rmp2_tuner.py --plot       # Grid search + save convergence plots

Run in the project directory:
    cd /home/ucluser/VRWIT/RL/predictive_model && python rmp2_tuner.py
"""

import sys, os, time, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.rmp2_bridge import RMP2Solver, TargetAttractor, OrientationAttractor, JointDamping


# ==========================================================================
# Robot-specific parameters (PiPER arm, matches rdt_sim_vr.py)
# ==========================================================================
PIPER_JOINT_COUNT = 6
JOINT_LOW = np.array([-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439])
JOINT_HIGH = np.array([2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439])
DEFAULT_Q = np.array([0.0, 0.1, -0.2, 0.0, 0.0, 0.0])
DT = 0.02

# Goal pose (represents a typical reach in the workspace)
GOAL_POS = np.array([0.5, 0.3, 0.2], dtype=np.float64)
GOAL_QUAT = np.array([0, 0, 0.70710678, 0.70710678], dtype=np.float64)


# ==========================================================================
# Simple analytical FK for self-contained testing
# ==========================================================================
class SimpleFK:
    """Pure NumPy FK: q[0:3] drive EEF pos directly, q[3:6] drive orient."""

    def __init__(self, j_low=None, j_high=None):
        self.j_low = np.asarray(j_low if j_low is not None else JOINT_LOW,
                                dtype=np.float64)
        self.j_high = np.asarray(j_high if j_high is not None else JOINT_HIGH,
                                 dtype=np.float64)

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        pos = q[:3] * 1.0

        from numpy import sin as s, cos as c
        rx, ry, rz = q[3], q[4], q[5]
        csx, snx = c(rx / 2), s(rx / 2)
        cy, sy = c(ry / 2), s(ry / 2)
        cz, sz = c(rz / 2), s(rz / 2)
        quat = np.array([
            snx * cy * cz - csx * sy * sz,
            csx * sy * cz + snx * cy * sz,
            csx * cy * sz - snx * sy * cz,
            csx * cy * cz + snx * sy * sz,
        ], dtype=np.float64)

        J = np.eye(6, dtype=np.float64)
        return pos, quat, J


# ==========================================================================
# Convergence test with detailed metrics
# ==========================================================================
def test_gains(target_p, target_d, orient_p, orient_d,
               damping_gain=30.0, metric_base=1.5,
               steps=2000, verbose=False, save_trajectories=False):
    """Run solver with given gains and return detailed convergence metrics.

    Returns dict with:
        - pos/rot error stats (mean, std, final)
        - Timing: rise_time, settling_time, reach_time
        - Quality: overshoot, ITAE, ISE, oscillation_count
        - Dynamics: qddot_max, qd_max, energy
    """
    fk = SimpleFK()

    def coll_fn(q, eef, obs):
        return [float(np.linalg.norm(eef - o)) for o in obs] if obs else []

    solver = RMP2Solver(
        n_joints=PIPER_JOINT_COUNT, fk_fn=fk, collision_fn=coll_fn,
        joint_limits_low=JOINT_LOW, joint_limits_high=JOINT_HIGH,
        default_q=DEFAULT_Q, dt=DT,
    )

    solver._fixed_leaves[0] = TargetAttractor(
        accel_p_gain=target_p, accel_d_gain=target_d,
    )
    solver._fixed_leaves[1] = OrientationAttractor(
        accel_p_gain=orient_p, accel_d_gain=orient_d,
    )
    solver._fixed_leaves[5] = JointDamping(accel_d_gain=damping_gain)

    q = np.array(DEFAULT_Q, dtype=np.float64)
    qd = np.zeros(PIPER_JOINT_COUNT, dtype=np.float64)

    qddot_max = 0.0
    qd_max = 0.0
    pos_errors = []
    rot_errors = []
    qddot_norms = []
    qd_norms = []

    pos_init = float(np.linalg.norm(GOAL_POS - np.array([q[0], q[1], q[2]])))

    for i in range(steps):
        qddot = solver.solve(q, qd, goals=[GOAL_POS], goal_quat=list(GOAL_QUAT))
        q, qd = solver.integrate(q, qd, qddot)
        q = solver.apply_hard_limits(q)

        qddot_norm = float(np.linalg.norm(qddot))
        qd_norm = float(np.linalg.norm(qd))
        qddot_max = max(qddot_max, qddot_norm)
        qd_max = max(qd_max, qd_norm)

        pos_err = float(np.linalg.norm(GOAL_POS - np.array([q[0], q[1], q[2]])))
        current_q = fk(q)[1]
        quat_leaf = solver._fixed_leaves[1]
        rot_err = np.linalg.norm(quat_leaf.quat_error(current_q, GOAL_QUAT))

        pos_errors.append(pos_err)
        rot_errors.append(float(rot_err))
        qddot_norms.append(qddot_norm)
        qd_norms.append(qd_norm)

        if verbose and i % 200 == 0:
            print(f"  step {i:5d} | pos={pos_err:.4f} rot={rot_err:.4f} | "
                  f"|qddot|={qddot_norm:.1f} |qd|={qd_norm:.2f}")

    pe = np.array(pos_errors, dtype=np.float64)
    re = np.array(rot_errors, dtype=np.float64)
    tail = min(200, len(pe))

    # --- Position metrics ---
    pos_mean = float(np.mean(pe[-tail:]))
    pos_std = float(np.std(pe[-tail:]))
    pos_final = float(pe[-1])

    # Rise time: first step where error drops below 90% of initial
    pos_90 = pos_init * 0.1
    rise_idx = next((i for i, e in enumerate(pe) if e < pos_90), steps)

    # Settling time: first step after which error stays within 2% of initial
    settle_thresh = pos_init * 0.02
    settle_idx = steps
    for i in range(len(pe) - tail, len(pe)):
        if all(e < settle_thresh for e in pe[i:]):
            settle_idx = i
            break

    # Overshoot: max excursion past the settling band (negative = none)
    pos_overshoot = max(0.0, float(np.max(pe[:rise_idx]) - pos_mean) if rise_idx < steps else 0.0)

    # ITAE (Integral of Time-weighted Absolute Error) — penalizes late errors heavily
    itae_pos = float(np.sum(pe * np.arange(len(pe), dtype=np.float64)))
    itae_rot = float(np.sum(re * np.arange(len(re), dtype=np.float64)))

    # ISE (Integral of Squared Error)
    ise_pos = float(np.sum(pe ** 2))
    ise_rot = float(np.sum(re ** 2))

    # Oscillations in last 500 steps
    osc = 0
    for i in range(-500 + 1, 0):
        if (pe[i] - pe[i - 1]) * (pe[i - 1] - pe[i - 2]) < 0:
            osc += 1

    # Energy: integral of |qddot|^2 (proxy for actuator effort)
    energy = float(np.sum(np.array(qddot_norms) ** 2))

    # --- Rotation metrics ---
    rot_mean = float(np.mean(re[-tail:]))
    rot_std = float(np.std(re[-tail:]))
    rot_init = max(float(re[0]), 1e-8)
    rot_reach_step = next((i for i, e in enumerate(re) if e < rot_init * 0.1), steps)

    return {
        # Final error
        'pos_error': pos_mean,
        'rot_error': rot_mean,
        'pos_final': pos_final,
        'pos_std': pos_std,
        'rot_std': rot_std,

        # Boolean flags
        'pos_reached': pos_mean < 0.01,
        'rot_reached': rot_mean < 0.05,
        'pos_stable': pos_std < 0.005,
        'rot_stable': rot_std < 0.02,

        # Timing (in steps)
        'rise_time': rise_idx,
        'settle_time': settle_idx,
        'rot_time': rot_reach_step,

        # Quality
        'overshoot': pos_overshoot,
        'oscillation_count': osc,
        'itae_pos': itae_pos,
        'itae_rot': itae_rot,
        'ise_pos': ise_pos,
        'ise_rot': ise_rot,

        # Dynamics
        'qddot_max': qddot_max,
        'qd_max': qd_max,
        'energy': energy,

        # Trajectories (optional)
        'pos_errors': pe if save_trajectories else None,
        'rot_errors': re if save_trajectories else None,
    }


# ==========================================================================
# Scoring function — lower is better
# ==========================================================================
def score_result(r):
    """Compute composite score from metrics. Lower = better."""
    s = 0.0

    # Final error (heavily weighted)
    s += r['pos_error'] * 200
    s += r['rot_error'] * 30

    # Stability (std in tail)
    s += r['pos_std'] * 500
    s += r['rot_std'] * 100

    # Speed (rise time, settling time)
    s += r['rise_time'] * 0.3
    s += r['settle_time'] * 0.15

    # Quality (oscillations, overshoot)
    s += r['oscillation_count'] * 2.0
    s += r['overshoot'] * 100

    # Dynamics (smoothness)
    s += r['qddot_max'] * 0.05
    s += r['energy'] * 1e-6

    # ITAE (penalizes slow convergence)
    s += r['itae_pos'] * 1e-4
    s += r['itae_rot'] * 1e-3

    # Hard penalties for failure
    if not r['pos_reached']:
        s += 200
    if not r['rot_reached']:
        s += 100
    if not r['pos_stable']:
        s += 80
    if not r['rot_stable']:
        s += 50

    return s


# ==========================================================================
# Grid search with coarse → fine two-phase strategy
# ==========================================================================
def grid_search(quick=False, fine_search=True):
    """Run parameter sweep with optional fine-tuning phase."""

    if quick:
        target_ps = [20, 40, 80]
        target_ds = [10, 30, 60]
        orient_ps = [10, 25, 50]
        orient_ds = [5, 10, 15]
    else:
        target_ps = [10, 20, 30, 40, 60, 80]
        target_ds = [5, 10, 20, 30, 40, 50, 60]
        orient_ps = [5, 10, 15, 20, 25, 30, 50]
        orient_ds = [3, 5, 8, 10, 15, 20]

    # --- Phase 1: Coarse grid ---
    total = len(target_ps) * len(target_ds) * len(orient_ps) * len(orient_ds)
    print(f"\n{'='*70}")
    print(f"  Phase 1: Coarse grid search — {total} configs")
    print(f"{'='*70}")

    results = []
    t0 = time.time()
    count = 0

    for tp in target_ps:
        for td in target_ds:
            for op in orient_ps:
                for od in orient_ds:
                    count += 1
                    try:
                        r = test_gains(tp, td, op, od, steps=2000)
                        r['score'] = score_result(r)
                        r['tp'], r['td'], r['op'], r['od'] = tp, td, op, od
                        results.append(r)
                    except Exception:
                        continue

                    if count % 50 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / count * (total - count)
                        print(f"  {count:4d}/{total} | ETA: {eta:.0f}s",
                              end='\r', flush=True)

    results.sort(key=lambda x: x['score'])
    elapsed = time.time() - t0
    print(f"\n  Phase 1 done: {len(results)}/{total} in {elapsed:.1f}s\n")

    if not results:
        return results

    # --- Phase 2: Fine search around top-5 ---
    if fine_search and results:
        print(f"{'='*70}")
        print(f"  Phase 2: Fine search around top-5 coarse results")
        print(f"{'='*70}")

        fine_results = []
        top5 = results[:5]
        step_p, step_d = 2.0, 2.0  # finer granularity

        for best in top5:
            tp0, td0, op0, od0 = best['tp'], best['td'], best['op'], best['od']

            # Generate fine grid around each top result
            fine_tp = [max(2, tp0 - step_p * i) for i in range(4, -1, -1)] + \
                      [tp0 + step_p * i for i in range(1, 4)]
            fine_td = [max(1, td0 - step_d * i) for i in range(4, -1, -1)] + \
                      [td0 + step_d * i for i in range(1, 4)]
            fine_op = [max(1, op0 - step_d * i) for i in range(4, -1, -1)] + \
                      [op0 + step_d * i for i in range(1, 4)]
            fine_od = [max(0.5, od0 - step_d * i) for i in range(4, -1, -1)] + \
                      [od0 + step_d * i for i in range(1, 4)]

            fine_total = len(fine_tp) * len(fine_td) * len(fine_op) * len(fine_od)
            fcount = 0

            for tp in fine_tp:
                for td in fine_td:
                    for op in fine_op:
                        for od in fine_od:
                            fcount += 1
                            try:
                                r = test_gains(tp, td, op, od, steps=2000)
                                r['score'] = score_result(r)
                                r['tp'], r['td'], r['op'], r['od'] = tp, td, op, od
                                fine_results.append(r)
                            except Exception:
                                continue

                            if fcount % 100 == 0:
                                print(f"  around ({tp0},{td0},{op0},{od0}): "
                                      f"{fcount}/{fine_total}",
                                      end='\r', flush=True)

        print()
        fine_results.sort(key=lambda x: x['score'])

        # Merge: keep best from fine, discard coarse results that are worse
        coarse_scores = {id(r): r['score'] for r in results}
        merged = fine_results + [r for r in results if r['score'] > fine_results[0]['score'] * 1.5]
        merged.sort(key=lambda x: x['score'])
        # Deduplicate by rounding gains
        seen = set()
        deduped = []
        for r in merged:
            key = (round(r['tp'], 1), round(r['td'], 1),
                   round(r['op'], 1), round(r['od'], 1))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped

    return results


def print_results(results, top_n=15):
    """Print ranked results table with detailed metrics."""
    print(f"\n{'='*110}")
    print(f"  {'#':>3} | tp    td    op    od   | pos_err rot_err | "
          f"rise  settle | osc  overshoot | qddot_max | energy  | score")
    print(f"{'-'*110}")

    for i, r in enumerate(results[:top_n]):
        flag = ""
        if r['pos_reached'] and r['rot_reached']:
            flag = "R"
        elif r['pos_stable'] and r['rot_stable']:
            flag = "S"
        else:
            flag = "!"

        print(f"  {i+1:3d} [{flag}] {r['tp']:5.1f} {r['td']:5.1f} "
              f"{r['op']:5.1f} {r['od']:5.1f} | "
              f"{r['pos_error']:.4f}  {r['rot_error']:.4f} | "
              f"{r['rise_time']:4d}  {r['settle_time']:5d} | "
              f"{r['oscillation_count']:3d}  {r['overshoot']:.3f} | "
              f"{r['qddot_max']:6.1f} | {r['energy']:8.0f} | "
              f"{r['score']:7.1f}")

    if results:
        best = results[0]
        print(f"{'-'*110}")
        print(f"  BEST: tp={best['tp']:.1f} td={best['td']:.1f} "
              f"op={best['op']:.1f} od={best['od']:.1f} "
              f"(score={best['score']:.1f})")
        print(f"        pos_err={best['pos_error']:.4f} rot_err={best['rot_error']:.4f} "
              f"rise={best['rise_time']} settle={best['settle_time']} "
              f"osc={best['oscillation_count']}")
        print(f"{'='*110}")

    return results


# ==========================================================================
# ASCII convergence plot
# ==========================================================================
def plot_convergence_ascii(results, top_n=5):
    """Print ASCII convergence curves for top results."""
    for i, r in enumerate(results[:top_n]):
        if r.get('pos_errors') is None:
            continue
        pe = r['pos_errors']
        width, height = 60, 15
        max_e = max(float(pe[0]), 1e-8)

        print(f"\n  Top #{i+1}: tp={r['tp']:.1f} td={r['td']:.1f} "
              f"op={r['op']:.1f} od={r['od']:.1f} (score={r['score']:.1f})")
        print(f"  pos_err over time:")

        grid = [['.' for _ in range(width)] for _ in range(height)]
        for j, e in enumerate(pe):
            col = int(j / len(pe) * (width - 1))
            row = int((1 - e / max_e) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = '#'

        for row in grid:
            print(f"  |{''.join(row)}|")
        print(f"  {'0'+'-'*(width-1)+'max':^{width}}")
        print(f"  {max_e:.4f}{' '*(width-8)}0.0000")


# ==========================================================================
# Single test with full diagnostics
# ==========================================================================
def single_test(tp, td, op, od):
    """Run one config with full diagnostics."""
    print(f"\nTesting: tp={tp} td={td} op={op} od={od}")
    print("=" * 70)

    r = test_gains(tp, td, op, od, steps=3000, verbose=True,
                   save_trajectories=True)
    r['score'] = score_result(r)

    print(f"\n  === Summary ===")
    print(f"  Score:        {r['score']:.1f}")
    print(f"  Pos error:    {r['pos_error']:.4f} (final: {r['pos_final']:.4f})")
    print(f"  Rot error:    {r['rot_error']:.4f}")
    print(f"  Rise time:    {r['rise_time']} steps ({r['rise_time']*DT:.1f}s)")
    print(f"  Settle time:  {r['settle_time']} steps ({r['settle_time']*DT:.1f}s)")
    print(f"  Overshoot:    {r['overshoot']:.4f}")
    print(f"  Oscillations: {r['oscillation_count']}")
    print(f"  ITAE pos:     {r['itae_pos']:.1f}")
    print(f"  Max accel:    {r['qddot_max']:.1f}")
    print(f"  Max vel:      {r['qd_max']:.2f}")
    print(f"  Energy:       {r['energy']:.0f}")
    print(f"  Status:       pos_{'REACHED' if r['pos_reached'] else 'MISSED'} "
          f"{'STABLE' if r['pos_stable'] else 'UNSTABLE'} | "
          f"rot_{'REACHED' if r['rot_reached'] else 'MISSED'} "
          f"{'STABLE' if r['rot_stable'] else 'UNSTABLE'}")

    # ASCII plot
    r_copy = dict(r)
    plot_convergence_ascii([r_copy], top_n=1)

    return r


# ==========================================================================
# Save recommendations
# ==========================================================================
def save_best(results):
    """Save best config to JSON."""
    best = results[0]
    config = {
        "target_attractor": {
            "accel_p_gain": round(best['tp'], 2),
            "accel_d_gain": round(best['td'], 2),
        },
        "orientation_attractor": {
            "accel_p_gain": round(best['op'], 2),
            "accel_d_gain": round(best['od'], 2),
        },
        "metrics": {
            "score": round(best['score'], 2),
            "pos_error": round(best['pos_error'], 4),
            "rot_error": round(best['rot_error'], 4),
            "rise_time": best['rise_time'],
            "settle_time": best['settle_time'],
            "oscillations": best['oscillation_count'],
        }
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rmp2_best_gains.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved best config to {path}")
    return config


# ==========================================================================
# Manual tuning
# ==========================================================================
def manual_tuning():
    """Interactive tuner — adjust one parameter at a time."""
    print("\n=== Manual Gain Tuner ===")
    print("Adjust one gain at a time. Enter 'q' to quit.\n")

    tp, td, op, od = 30.0, 20.0, 10.0, 12.0
    print(f"Starting: tp={tp} td={td} op={op} od={od}\n")

    while True:
        r = test_gains(tp, td, op, od, steps=2000)
        r['score'] = score_result(r)
        print(f"> tp={tp:5.1f} td={td:5.1f} op={op:5.1f} od={od:5.1f} | "
              f"pos={r['pos_error']:.4f} rot={r['rot_error']:.4f} | "
              f"rise={r['rise_time']} settle={r['settle_time']} | "
              f"osc={r['oscillation_count']} | score={r['score']:.1f}")

        try:
            cmd = input("\n  [p+/p-] pos P  [d+/d-] pos D  "
                        "[o+/o-] orient P  [k+/k-] orient D  | "
                        "[P+/P-] pos P(x2) [D+/D-] pos D(x2)  "
                        "[O+/O-] orient P(x2) [K+/K-] orient D(x2)  | "
                        "[s] save  [t] verbose  [q] quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd == 'q':
            break
        elif cmd == 'p+':
            tp = min(tp + 5, 150)
        elif cmd == 'p-':
            tp = max(tp - 2, 1)
        elif cmd == 'd+':
            td = min(td + 3, 120)
        elif cmd == 'd-':
            td = max(td - 1, 0.5)
        elif cmd == 'o+':
            op = min(op + 3, 100)
        elif cmd == 'o-':
            op = max(op - 1, 1)
        elif cmd == 'k+':
            od = min(od + 1, 50)
        elif cmd == 'k-':
            od = max(od - 0.5, 0.5)
        elif cmd == 'P+':
            tp = min(tp + 15, 150)
        elif cmd == 'P-':
            tp = max(tp - 5, 1)
        elif cmd == 'D+':
            td = min(td + 10, 120)
        elif cmd == 'D-':
            td = max(td - 3, 0.5)
        elif cmd == 'O+':
            op = min(op + 8, 100)
        elif cmd == 'O-':
            op = max(op - 3, 1)
        elif cmd == 'K+':
            od = min(od + 3, 50)
        elif cmd == 'K-':
            od = max(od - 1.5, 0.5)
        elif cmd == 's':
            r['tp'], r['td'], r['op'], r['od'] = tp, td, op, od
            save_best([r])
        elif cmd == 't':
            single_test(tp, td, op, od)


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RMP2 Gain Tuner')
    parser.add_argument('--quick', action='store_true',
                        help='Smaller grid search (faster)')
    parser.add_argument('--step', action='store_true',
                        help='Manual step-by-step tuning')
    parser.add_argument('--plot', action='store_true',
                        help='Save convergence plot (needs matplotlib)')
    parser.add_argument('--no-fine', action='store_true',
                        help='Skip fine-tuning phase (coarse only)')
    parser.add_argument('--test', type=str, default=None,
                        help='Test single config: "tp,td,op,od" e.g. "40,20,10,12"')
    args = parser.parse_args()

    if args.step:
        manual_tuning()
    elif args.test:
        parts = args.test.split(',')
        if len(parts) != 4:
            print("Usage: --test tp,td,op,od (e.g. --test 40,20,10,12)")
            sys.exit(1)
        single_test(*[float(x) for x in parts])
    else:
        results = grid_search(quick=args.quick, fine_search=not args.no_fine)
        if results:
            print_results(results)
            save_best(results)
            # Show ASCII plots for top results
            plot_convergence_ascii(results, top_n=3)
        else:
            print("No valid configs found.")