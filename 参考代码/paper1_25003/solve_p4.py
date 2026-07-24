"""
问题4: 三机各投放1枚烟幕弹，对M1实施干扰
Uses geometric initialization per drone + local search.
"""
import numpy as np
import simulation
import config_a
from solve_lbfgs import _geometric_feasible_points, evaluate_candidates


def solve_problem4(config_module=None, use_parallel=False, verbose=True):
    """
    Solve problem 4: FY1/FY2/FY3 each drop 1 bomb vs M1.

    Returns:
        results: list of per-drone results
        total_time: total effective shielding time
    """
    if config_module is None:
        config_module = config_a
    simulation.set_config(config_module)
    simulation.clear_keypoint_cache()
    cfg = config_module

    if verbose:
        print("=" * 60)
        print("问题4: FY1/FY2/FY3各投放1枚烟幕弹对M1")
        print("=" * 60)

    drone_names = ['FY1', 'FY2', 'FY3']
    n_drones = 3
    dt_opt = 0.02
    t_total = 30.0
    target_keypoints = simulation.get_target_keypoints(36, 3)

    per_drone_results = []

    for di in range(n_drones):
        drone_init = cfg.DRONES_INIT[di]

        if verbose:
            print(f"\n--- {drone_names[di]} 独立优化 ---")

        # Geometric initialization
        geom_points = _geometric_feasible_points(drone_init, 0, cfg, n_t_det=20, n_alpha=12)
        candidates = evaluate_candidates(geom_points, drone_init, target_keypoints, cfg,
                                         dt=dt_opt, t_total=t_total, top_k=15)

        if not candidates:
            if verbose:
                print(f"  {drone_names[di]}: 几何方法未找到可行解")
            per_drone_results.append({'x': None, 'f': 0.0})
            continue

        best_f = candidates[0][0]
        best_x = candidates[0][1].copy()

        if verbose:
            print(f"  初始: theta={np.degrees(best_x[0]):.1f}deg v={best_x[1]:.1f}m/s "
                  f"f={best_f:.4f}s")

        # Local perturbation search
        if verbose:
            print(f"  局部搜索...")

        n_local = 1000
        for j in range(n_local):
            ci = np.random.randint(0, min(len(candidates), 8))
            base = candidates[ci][1]
            scale = np.array([0.05, 15.0, 1.5, 1.5])
            delta = np.random.uniform(-1, 1, 4) * scale
            x_new = base + delta
            x_new[0] = x_new[0] % (2 * np.pi)
            x_new[1] = np.clip(x_new[1], cfg.DRONE_SPEED_MIN, cfg.DRONE_SPEED_MAX)
            x_new[2] = np.clip(x_new[2], 0.1, 15.0)
            x_new[3] = np.clip(x_new[3], 0.1, 10.0)

            val = simulation.simulate_single_bomb(
                drone_init, x_new[0], x_new[1], x_new[2], x_new[3],
                missile_idx=0, target_keypoints=target_keypoints,
                dt=dt_opt, t_total=t_total
            )
            if val > best_f:
                best_f = val
                best_x = x_new.copy()

        if verbose:
            print(f"  优化: f={best_f:.4f}s  theta={np.degrees(best_x[0]):.1f}deg  "
                  f"v={best_x[1]:.1f}m/s")

        per_drone_results.append({'x': best_x, 'f': best_f})

    # ============================================================
    # Combined evaluation
    # ============================================================
    if verbose:
        print(f"\n--- 协同评估 ---")

    drone_params_list = []
    for di in range(n_drones):
        if per_drone_results[di]['x'] is None:
            drone_params_list.append({
                'drone_init': cfg.DRONES_INIT[di],
                'theta': 0.0, 'speed': cfg.DRONE_SPEED_MIN,
                'release_times': np.array([1.0]),
                'detonation_delays': np.array([1.0]),
                'missile_indices': [0],
            })
        else:
            x_opt = per_drone_results[di]['x']
            drone_params_list.append({
                'drone_init': cfg.DRONES_INIT[di],
                'theta': x_opt[0],
                'speed': x_opt[1],
                'release_times': np.array([x_opt[2]]),
                'detonation_delays': np.array([x_opt[3]]),
                'missile_indices': [0],
            })

    total_time, per_missile = simulation.simulate_multi_drone_multi_bomb(
        drone_params_list, dt=cfg.DT_FINE, t_total=t_total
    )

    if verbose:
        print(f"\n协同结果:")
        for di in range(n_drones):
            if per_drone_results[di]['x'] is not None:
                x = per_drone_results[di]['x']
                print(f"  {drone_names[di]}: theta={np.degrees(x[0]):.2f}deg  "
                      f"v={x[1]:.1f}m/s  rt={x[2]:.2f}s  dd={x[3]:.2f}s  "
                      f"单机={per_drone_results[di]['f']:.4f}s")
            else:
                print(f"  {drone_names[di]}: NO SOLUTION")
        print(f"  总有效遮蔽时长: {total_time:.4f} s")

    return per_drone_results, total_time


if __name__ == "__main__":
    results, total = solve_problem4(config_a)
    print(f"\nPaper reference: 12.544s")
    print(f"Obtained: {total:.4f}s")
