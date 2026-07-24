"""
Main driver - Paper cumcm25086: Adaptive PSO & Multi-Island PSO

Usage:
    python run_all.py              # Run A题
    python run_all.py C            # Run C题
    python run_all.py --fast       # Fast mode (reduced iterations)
    python run_all.py --skip 4,5   # Skip P4, P5
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    args = {'problem': 'A', 'skip': set(), 'fast': False}
    for a in sys.argv[1:]:
        if a in ('A', 'C'):
            args['problem'] = a
        elif a == '--fast':
            args['fast'] = True
        elif a.startswith('--skip'):
            pass  # handled below
    # Parse --skip
    for i, a in enumerate(sys.argv):
        if a == '--skip' and i + 1 < len(sys.argv):
            args['skip'] = set(int(x) for x in sys.argv[i+1].split(','))
    return args


def run_problem(config_module, label):
    import simulation as sim
    sim.set_config(config_module)

    args = parse_args()
    results = {}
    total_start = time.time()

    print("\n" + "=" * 70)
    print(f"  {label} 参数测试 (Paper cumcm25086: APSO + Multi-Island PSO)")
    print(f"  Coverage: two-phase (optimize@0.01, verify@0.80)")
    print(f"  SMOKE_SINK={config_module.SMOKE_SINK_SPEED} m/s")
    print(f"  DRONE_SPEED=[{config_module.DRONE_SPEED_MIN}, {config_module.DRONE_SPEED_MAX}] m/s")
    print("=" * 70)

    # ---- P1 ----
    if 1 not in args['skip']:
        print("\n" + "=" * 70)
        print("  问题1: FY1投放1枚烟幕弹对M1 (固定参数仿真)")
        print("=" * 70)
        t0 = time.time()
        from solve_p1 import solve_p1
        p1_time = solve_p1(config_module, dt_fine=config_module.DT_FINE)
        elapsed = time.time() - t0
        results['P1'] = {'value': p1_time, 'runtime': elapsed}
        print(f"  P1 runtime: {elapsed:.1f}s")

    # ---- P2 ----
    if 2 not in args['skip']:
        print("\n" + "=" * 70)
        print("  问题2: FY1单机单弹最优策略 (APSO)")
        print("=" * 70)
        t0 = time.time()
        from solve_p2 import solve_p2

        orig_size, orig_iter = config_module.APSO_SWARM_SIZE, config_module.APSO_MAX_ITER
        if args['fast']:
            config_module.APSO_SWARM_SIZE = 30
            config_module.APSO_MAX_ITER = 40
            print(f"  [FAST: swarm=30, iter=40]")

        x_opt, f_opt = solve_p2(config_module)

        config_module.APSO_SWARM_SIZE = orig_size
        config_module.APSO_MAX_ITER = orig_iter

        elapsed = time.time() - t0
        results['P2'] = {'value': f_opt, 'params': x_opt, 'runtime': elapsed}
        print(f"  P2 runtime: {elapsed:.1f}s")

    # ---- P3 ----
    if 3 not in args['skip']:
        print("\n" + "=" * 70)
        print("  问题3: FY1投放3枚烟幕弹对M1 (APSO)")
        print("=" * 70)
        t0 = time.time()

        orig_size, orig_iter = config_module.APSO_SWARM_SIZE, config_module.APSO_MAX_ITER
        if args['fast']:
            config_module.APSO_SWARM_SIZE = 30
            config_module.APSO_MAX_ITER = 40
            print(f"  [FAST: swarm=30, iter=40]")

        from solve_p3 import solve_p3
        x_opt3, f_opt3 = solve_p3(config_module)

        config_module.APSO_SWARM_SIZE = orig_size
        config_module.APSO_MAX_ITER = orig_iter

        elapsed = time.time() - t0
        results['P3'] = {'value': f_opt3, 'params': x_opt3, 'runtime': elapsed}
        print(f"  P3 runtime: {elapsed:.1f}s")

    # ---- P4 ----
    if 4 not in args['skip']:
        print("\n" + "=" * 70)
        print("  问题4: FY1/FY2/FY3各1弹对M1 (Multi-Island PSO)")
        print("=" * 70)
        t0 = time.time()

        orig_iter, orig_swarm = config_module.MI_MAX_ITER, config_module.MI_SWARM_PER_ISLAND
        if args['fast']:
            config_module.MI_MAX_ITER = 30
            config_module.MI_SWARM_PER_ISLAND = 15
            print(f"  [FAST: islands={config_module.MI_N_ISLANDS}, swarm=15, iter=30]")

        from solve_p4 import solve_p4
        x_opt4, f_opt4 = solve_p4(config_module)

        config_module.MI_MAX_ITER = orig_iter
        config_module.MI_SWARM_PER_ISLAND = orig_swarm

        elapsed = time.time() - t0
        results['P4'] = {'value': f_opt4, 'params': x_opt4, 'runtime': elapsed}
        print(f"  P4 runtime: {elapsed:.1f}s")

    # ---- P5 ----
    if 5 not in args['skip']:
        print("\n" + "=" * 70)
        print("  问题5: 5机×3弹对M1/M2/M3 (Multi-Island PSO)")
        print("=" * 70)
        t0 = time.time()

        orig_iter, orig_swarm = config_module.MI_MAX_ITER, config_module.MI_SWARM_PER_ISLAND
        orig_size2, orig_iter2 = config_module.APSO_SWARM_SIZE, config_module.APSO_MAX_ITER
        if args['fast']:
            config_module.MI_MAX_ITER = 20
            config_module.MI_SWARM_PER_ISLAND = 10
            config_module.APSO_SWARM_SIZE = 20
            config_module.APSO_MAX_ITER = 30
            print(f"  [FAST]")

        from solve_p5 import solve_p5
        final_results, f_opt5 = solve_p5(config_module)

        config_module.MI_MAX_ITER = orig_iter
        config_module.MI_SWARM_PER_ISLAND = orig_swarm
        config_module.APSO_SWARM_SIZE = orig_size2
        config_module.APSO_MAX_ITER = orig_iter2

        elapsed = time.time() - t0
        results['P5'] = {'value': f_opt5, 'runtime': elapsed}
        print(f"  P5 runtime: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    results['_total_runtime'] = total_elapsed
    return results


def print_summary(results_a, results_c=None):
    print("\n\n" + "=" * 70)
    print("  结果汇总 (Paper cumcm25086: APSO + Multi-Island PSO)")
    print("=" * 70)
    header = f"{'Problem':<10} {'A题 value':<18} {'A题 runtime':<15}"
    if results_c:
        header += f" {'C题 value':<18} {'C题 runtime':<15}"
    print(header)
    print("-" * 70)

    for p in ['P1', 'P2', 'P3', 'P4', 'P5']:
        if p in results_a:
            ra = results_a[p]
            line = f"{p:<10} {ra['value']:<18.4f} {ra['runtime']:<15.1f}"
            if results_c and p in results_c:
                rc = results_c[p]
                line += f" {rc['value']:<18.4f} {rc['runtime']:<15.1f}"
            print(line)
    print("-" * 70)

    total_a = results_a.get('_total_runtime', 0)
    line = f"{'Total':<10} {'':<18} {total_a:<15.1f}"
    if results_c:
        total_c = results_c.get('_total_runtime', 0)
        line += f" {'':<18} {total_c:<15.1f}"
    print(line)

    if results_c:
        print("\n" + "=" * 70)
        print("  C题 vs A题 差异")
        print("=" * 70)
        print(f"{'Problem':<10} {'A题':<15} {'C题':<15} {'C/A':<12} {'Note'}")
        print("-" * 70)
        for p in ['P1', 'P2', 'P3', 'P4', 'P5']:
            if p in results_a and p in results_c:
                va, vc = results_a[p]['value'], results_c[p]['value']
                ratio = vc / va if va > 0 else 0
                note = "C更优(slower sink)" if ratio > 1 else "A更优(more speed range)" if ratio < 1 else "相同"
                print(f"{p:<10} {va:<15.4f} {vc:<15.4f} {ratio:<12.4f} {note}")
        print("-" * 70)


if __name__ == "__main__":
    args = parse_args()

    print("=" * 70)
    print("  Paper cumcm25086: Adaptive PSO & Multi-Island PSO")
    print("  2025 CUMCM Problem A / C题 烟幕干扰弹投放策略")
    print("=" * 70)
    print(f"  Algorithm: Clerc's Constriction APSO + Multi-Island PSO")
    print(f"  Two-phase: optimize@ratio=0.01, verify@ratio=0.80")
    print(f"  Options: problem={args['problem']}, fast={args['fast']}, skip={args['skip']}")

    if args['problem'] in ('A', 'both'):
        print("\n\n" + "#" * 70)
        print("#  A题 (原始参数)")
        print("#" * 70)
        import config_a
        results_a = run_problem(config_a, "A题")

        if args['problem'] == 'both':
            print("\n\n" + "#" * 70)
            print("#  C题 (修改参数)")
            print("#" * 70)
            import config_c
            results_c = run_problem(config_c, "C题")
            print_summary(results_a, results_c)
        else:
            print_summary(results_a)

    elif args['problem'] == 'C':
        print("\n\n" + "#" * 70)
        print("#  C题 (修改参数)")
        print("#" * 70)
        import config_c
        results_c = run_problem(config_c, "C题")
        print_summary({}, results_c)

    print("\nDone.")
