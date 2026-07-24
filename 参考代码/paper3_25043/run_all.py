"""
run_all.py - Main driver: runs A题 first (verification), then C题.
Paper results (A题):
  P1: 1.39s occlusion duration
  P2: coverage duration, theta=7.18deg, v=95.73m/s
  P3: theta=8.70deg, v=110.28m/s
  P4: 11.56s (union coverage)
  P5: 34.00s
"""
import sys
import time as time_module
import numpy as np

np.seterr(all='ignore')


def run_all(config_module, label):
    """Run all 5 problems with the given config."""
    config = config_module

    print(f"\n{'#'*60}")
    print(f"#  {label}")
    print(f"#  SINK={config.SMOKE_SINK_SPEED}, "
          f"V_DRONE=[{config.DRONE_SPEED_MIN},{config.DRONE_SPEED_MAX}]")
    print(f"#  P1: t_rel={config.P1_RELEASE_TIME}, t_e={config.P1_DETONATION_DELAY}")
    print(f"{'#'*60}")

    overall_start = time_module.time()
    results = {}

    # Problem 1
    print(f"\n>>> {label}-P1 <<<")
    try:
        import solve_p1
        dur, t1 = solve_p1.solve_p1(config, f"{label}-P1")
        results['P1'] = {'occlusion_duration': dur, 'time': t1}
    except Exception as e:
        print(f"  P1 ERROR: {e}")
        import traceback; traceback.print_exc()
        results['P1'] = {'error': str(e)}

    # Problem 2
    print(f"\n>>> {label}-P2 <<<")
    try:
        import solve_p2
        p2r = solve_p2.solve_p2(config, f"{label}-P2")
        results['P2'] = {k: v for k, v in p2r.items()}
    except Exception as e:
        print(f"  P2 ERROR: {e}")
        import traceback; traceback.print_exc()
        results['P2'] = {'error': str(e)}

    # Problem 3
    print(f"\n>>> {label}-P3 <<<")
    try:
        import solve_p3
        p3r = solve_p3.solve_p3(config, f"{label}-P3")
        results['P3'] = {k: v for k, v in p3r.items()}
    except Exception as e:
        print(f"  P3 ERROR: {e}")
        import traceback; traceback.print_exc()
        results['P3'] = {'error': str(e)}

    # Problem 4
    print(f"\n>>> {label}-P4 <<<")
    try:
        import solve_p4
        p4r = solve_p4.solve_p4(config, f"{label}-P4")
        results['P4'] = {k: v for k, v in p4r.items()}
    except Exception as e:
        print(f"  P4 ERROR: {e}")
        import traceback; traceback.print_exc()
        results['P4'] = {'error': str(e)}

    # Problem 5
    print(f"\n>>> {label}-P5 <<<")
    try:
        import solve_p5
        p5r = solve_p5.solve_p5(config, f"{label}-P5")
        results['P5'] = {k: v for k, v in p5r.items()}
    except Exception as e:
        print(f"  P5 ERROR: {e}")
        import traceback; traceback.print_exc()
        results['P5'] = {'error': str(e)}

    overall_time = time_module.time() - overall_start

    print(f"\n{'='*60}")
    print(f"  {label} SUMMARY (total time: {overall_time:.1f}s)")
    print(f"{'='*60}")
    for prob, res in results.items():
        if 'error' in res:
            print(f"  {prob}: ERROR - {res['error']}")
        elif prob == 'P1':
            print(f"  {prob}: dur={res['occlusion_duration']:.4f}s, time={res['time']:.4f}s")
        elif prob in ('P2', 'P3'):
            dur_key = res.get('duration_true', res.get('coverage_true', 'N/A'))
            print(f"  {prob}: theta={res.get('theta','?')}deg, v={res.get('v','?')}m/s, "
                  f"dur={dur_key}")
        elif prob == 'P4':
            print(f"  {prob}: union_cov={res.get('union_coverage_true','N/A')}, "
                  f"dur={res.get('duration_true','N/A')}")
        elif prob == 'P5':
            print(f"  {prob}: best_cov={res.get('coverage_true','N/A')}, "
                  f"dur={res.get('duration_true','N/A')}, bombs={res.get('n_bombs','?')}")

    return results, overall_time


if __name__ == "__main__":
    print("=" * 60)
    print("  CUMCM 2025 Problem A - Paper 25043 PSO Implementation")
    print("=" * 60)

    print("\n========== Phase 1: A题 Verification ==========")
    import config_a
    results_a, time_a = run_all(config_a, "A")

    print("\n\n========== Phase 2: C题 ==========")
    import config_c
    results_c, time_c = run_all(config_c, "C")

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    paper = {'P1': 1.39, 'P2': 4.59, 'P3': 6.41, 'P4': 11.56, 'P5': 34.00}
    for prob in ['P1', 'P2', 'P3', 'P4', 'P5']:
        ra = results_a.get(prob, {})
        rc = results_c.get(prob, {})

        if prob == 'P1':
            a_val = ra.get('occlusion_duration', 'ERR')
            c_val = rc.get('occlusion_duration', 'ERR')
        elif prob in ('P2', 'P3'):
            a_val = ra.get('duration_true', ra.get('coverage_true', 'ERR'))
            c_val = rc.get('duration_true', rc.get('coverage_true', 'ERR'))
        elif prob == 'P4':
            a_val = ra.get('duration_true', ra.get('union_coverage_true', 'ERR'))
            c_val = rc.get('duration_true', rc.get('union_coverage_true', 'ERR'))
        elif prob == 'P5':
            a_val = ra.get('duration_true', ra.get('coverage_true', 'ERR'))
            c_val = rc.get('duration_true', rc.get('coverage_true', 'ERR'))

        print(f"  {prob}: Paper={paper[prob]}, A={a_val}, C={c_val}")

    print(f"\nTotal time: A={time_a:.1f}s, C={time_c:.1f}s")
    print("Done.")
