"""
Main runner: Run all 5 problems with A题 and C题 parameters.
Uses faster settings for practical runtime.
Saves results to results_a.txt and results_c.txt.
"""
import sys
import os
import time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config_a
import config_c
import simulation
from solve_lbfgs import solve_problem2
from solve_p3 import solve_problem3
from solve_p4 import solve_problem4
from solve_p5 import solve_problem5


def _fmt_time(t):
    """Format time in mm:ss."""
    m = int(t // 60)
    s = int(t % 60)
    return f"{m}m{s:02d}s"


def main():
    t_start = time.time()

    # ============================================================
    # A题
    # ============================================================
    print("=" * 70)
    print("  A题 参数 (2025 CUMCM Problem A)")
    print("=" * 70)
    print(f"  SMOKE_SINK_SPEED = {config_a.SMOKE_SINK_SPEED} m/s")
    print(f"  DRONE_SPEED = [{config_a.DRONE_SPEED_MIN}, {config_a.DRONE_SPEED_MAX}] m/s")

    results_a = {}
    paper = config_a.PAPER_RESULTS

    # P1
    t0 = time.time()
    simulation.set_config(config_a)
    simulation.clear_keypoint_cache()
    kp = simulation.get_target_keypoints(36, 0)
    p1 = simulation.simulate_single_bomb(
        config_a.DRONES_INIT[0], config_a.P1_DRONE_THETA, config_a.P1_DRONE_SPEED,
        config_a.P1_RELEASE_TIME, config_a.P1_DETONATION_DELAY,
        missile_idx=0, target_keypoints=kp, dt=config_a.DT_FINE, t_total=30.0
    )
    t1 = time.time()
    diff1 = abs(p1 - paper['P1']) / paper['P1'] * 100
    results_a['P1'] = {'value': p1, 'time': t1 - t0, 'diff_pct': diff1}
    print(f"\nP1 A题: {p1:.4f}s (paper: {paper['P1']}s, diff: {diff1:.1f}%) [{_fmt_time(t1-t0)}]")

    # P2
    try:
        x_p2, f_p2 = solve_problem2(config_a, use_fine_dt=True, verbose=True)
        t2 = time.time()
        diff2 = abs(f_p2 - paper['P2']) / paper['P2'] * 100
        results_a['P2'] = {
            'value': f_p2, 'theta': x_p2[0], 'speed': x_p2[1],
            'release_time': x_p2[2], 'delay': x_p2[3],
            'time': t2 - t1, 'diff_pct': diff2
        }
        print(f"\n  Paper P2: {paper['P2']}s, diff: {diff2:.1f}%")
    except Exception as e:
        print(f"  P2 FAILED: {e}")
        results_a['P2'] = {'error': str(e)}
        t2 = time.time()

    # P3
    try:
        x_p3, f_p3 = solve_problem3(config_a, verbose=True)
        t3 = time.time()
        diff3 = abs(f_p3 - paper['P3']) / paper['P3'] * 100 if f_p3 > 0 else 100
        results_a['P3'] = {
            'value': f_p3, 'theta': x_p3[0], 'speed': x_p3[1],
            'time': t3 - t2, 'diff_pct': diff3
        }
        print(f"\n  Paper P3: {paper['P3']}s, diff: {diff3:.1f}%")
    except Exception as e:
        print(f"  P3 FAILED: {e}")
        results_a['P3'] = {'error': str(e)}
        t3 = time.time()

    # P4
    try:
        _, f_p4 = solve_problem4(config_a, verbose=True)
        t4 = time.time()
        diff4 = abs(f_p4 - paper['P4']) / paper['P4'] * 100 if f_p4 > 0 else 100
        results_a['P4'] = {'value': f_p4, 'time': t4 - t3, 'diff_pct': diff4}
        print(f"\n  Paper P4: {paper['P4']}s, diff: {diff4:.1f}%")
    except Exception as e:
        print(f"  P4 FAILED: {e}")
        results_a['P4'] = {'error': str(e)}
        t4 = time.time()

    # P5
    try:
        _, f_p5 = solve_problem5(config_a, verbose=True)
        t5 = time.time()
        diff5 = abs(f_p5 - paper['P5']) / paper['P5'] * 100 if f_p5 > 0 else 100
        results_a['P5'] = {'value': f_p5, 'time': t5 - t4, 'diff_pct': diff5}
        print(f"\n  Paper P5: {paper['P5']}s, diff: {diff5:.1f}%")
    except Exception as e:
        print(f"  P5 FAILED: {e}")
        results_a['P5'] = {'error': str(e)}
        t5 = time.time()

    t_a_total = t5 - t_start

    # A题 summary
    print("\n" + "=" * 70)
    print("  A题 汇总")
    print("=" * 70)
    for pk in ['P1', 'P2', 'P3', 'P4', 'P5']:
        r = results_a.get(pk, {})
        if 'error' in r:
            print(f"  {pk}: ERROR - {r['error']}")
        else:
            print(f"  {pk}: {r['value']:.4f}s (paper: {paper[pk]}s, diff: {r.get('diff_pct',0):.1f}%) [{_fmt_time(r.get('time',0))}]")
    print(f"  A题总耗时: {_fmt_time(t_a_total)}")

    # ============================================================
    # C题
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  C题 参数 (Modified)")
    print("=" * 70)
    print(f"  SMOKE_SINK_SPEED = {config_c.SMOKE_SINK_SPEED} m/s")
    print(f"  DRONE_SPEED = [{config_c.DRONE_SPEED_MIN}, {config_c.DRONE_SPEED_MAX}] m/s")

    results_c = {}
    t_c0 = time.time()

    # P1 C题
    simulation.set_config(config_c)
    simulation.clear_keypoint_cache()
    kp_c = simulation.get_target_keypoints(36, 0)
    p1c = simulation.simulate_single_bomb(
        config_c.DRONES_INIT[0], config_c.P1_DRONE_THETA, config_c.P1_DRONE_SPEED,
        config_c.P1_RELEASE_TIME, config_c.P1_DETONATION_DELAY,
        missile_idx=0, target_keypoints=kp_c, dt=config_c.DT_FINE, t_total=30.0
    )
    t_c1 = time.time()
    results_c['P1'] = {'value': p1c, 'time': t_c1 - t_c0}
    print(f"\nP1 C题: {p1c:.4f}s [{_fmt_time(t_c1-t_c0)}]")

    # P2 C题
    try:
        x_p2c, f_p2c = solve_problem2(config_c, use_fine_dt=True, verbose=True)
        t_c2 = time.time()
        results_c['P2'] = {
            'value': f_p2c, 'theta': x_p2c[0], 'speed': x_p2c[1],
            'release_time': x_p2c[2], 'delay': x_p2c[3], 'time': t_c2 - t_c1
        }
    except Exception as e:
        print(f"  P2 C题 FAILED: {e}")
        results_c['P2'] = {'error': str(e)}
        t_c2 = time.time()

    # P3 C题
    try:
        x_p3c, f_p3c = solve_problem3(config_c, verbose=True)
        t_c3 = time.time()
        results_c['P3'] = {'value': f_p3c, 'theta': x_p3c[0], 'speed': x_p3c[1], 'time': t_c3 - t_c2}
    except Exception as e:
        print(f"  P3 C题 FAILED: {e}")
        results_c['P3'] = {'error': str(e)}
        t_c3 = time.time()

    # P4 C题
    try:
        _, f_p4c = solve_problem4(config_c, verbose=True)
        t_c4 = time.time()
        results_c['P4'] = {'value': f_p4c, 'time': t_c4 - t_c3}
    except Exception as e:
        print(f"  P4 C题 FAILED: {e}")
        results_c['P4'] = {'error': str(e)}
        t_c4 = time.time()

    # P5 C题
    try:
        _, f_p5c = solve_problem5(config_c, verbose=True)
        t_c5 = time.time()
        results_c['P5'] = {'value': f_p5c, 'time': t_c5 - t_c4}
    except Exception as e:
        print(f"  P5 C题 FAILED: {e}")
        results_c['P5'] = {'error': str(e)}
        t_c5 = time.time()

    t_c_total = t_c5 - t_c0
    t_total_all = t_c5 - t_start

    # C题 summary
    print("\n" + "=" * 70)
    print("  C题 汇总")
    print("=" * 70)
    for pk in ['P1', 'P2', 'P3', 'P4', 'P5']:
        r = results_c.get(pk, {})
        if 'error' in r:
            print(f"  {pk}: ERROR - {r['error']}")
        elif 'theta' in r:
            print(f"  {pk}: {r['value']:.4f}s (theta={np.degrees(r['theta']):.1f}deg, v={r['speed']:.1f}m/s) [{_fmt_time(r.get('time',0))}]")
        else:
            print(f"  {pk}: {r['value']:.4f}s [{_fmt_time(r.get('time',0))}]")
    print(f"  C题总耗时: {_fmt_time(t_c_total)}")

    # Final comparison
    print("\n" + "=" * 70)
    print("                    最终汇总")
    print("=" * 70)
    print(f"{'Problem':<10} {'A题(s)':<12} {'C题(s)':<12} {'Paper A题(s)':<14}")
    print("-" * 50)
    for pk in ['P1', 'P2', 'P3', 'P4', 'P5']:
        av = f"{results_a.get(pk,{}).get('value','ERR'):.4f}" if 'value' in results_a.get(pk,{}) else 'ERR'
        cv = f"{results_c.get(pk,{}).get('value','ERR'):.4f}" if 'value' in results_c.get(pk,{}) else 'ERR'
        pv = f"{paper.get(pk,'N/A'):.4f}"
        print(f"  {pk:<8} {av:<12} {cv:<12} {pv:<14}")
    print("-" * 50)
    print(f"  A题总耗时: {_fmt_time(t_a_total)}")
    print(f"  C题总耗时: {_fmt_time(t_c_total)}")
    print(f"  全部总耗时: {_fmt_time(t_total_all)}")
    print("=" * 70)

    # Save results
    _save_results("results_a.txt", results_a, config_a, "A题")
    _save_results("results_c.txt", results_c, config_c, "C题")
    print(f"\nResults saved to results_a.txt, results_c.txt")


def _save_results(filename, results, cfg, label):
    """Save results to text file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"={label} Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"SMOKE_SINK_SPEED = {cfg.SMOKE_SINK_SPEED} m/s\n")
        f.write(f"DRONE_SPEED = [{cfg.DRONE_SPEED_MIN}, {cfg.DRONE_SPEED_MAX}] m/s\n")
        f.write(f"P1: release={cfg.P1_RELEASE_TIME}s, delay={cfg.P1_DETONATION_DELAY}s\n\n")
        for pk in ['P1', 'P2', 'P3', 'P4', 'P5']:
            r = results.get(pk, {})
            f.write(f"{pk}:\n")
            if 'error' in r:
                f.write(f"  ERROR: {r['error']}\n")
            elif 'theta' in r:
                f.write(f"  effective_time: {r['value']:.6f}s\n")
                f.write(f"  theta: {r['theta']:.6f}rad ({np.degrees(r['theta']):.4f}deg)\n")
                f.write(f"  speed: {r['speed']:.4f}m/s\n")
                if 'release_time' in r:
                    f.write(f"  release_time: {r['release_time']:.6f}s\n")
                    f.write(f"  delay: {r['delay']:.6f}s\n")
                f.write(f"  compute_time: {r.get('time',0):.2f}s\n")
            else:
                f.write(f"  effective_time: {r['value']:.6f}s\n")
                f.write(f"  compute_time: {r.get('time',0):.2f}s\n")
            f.write("\n")


if __name__ == "__main__":
    main()
