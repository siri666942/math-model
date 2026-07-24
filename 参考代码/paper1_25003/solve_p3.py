"""
问题3: 单机三弹对M1 (1-3-1)
Strategy: Geometric initialization for first bomb + PSO for all 3 bombs
PSO params: 80 particles, 160 iterations, c1=c2=2.0, w=0.9->0.4
"""
import numpy as np
import simulation
import config_a
from solve_lbfgs import _geometric_feasible_points, evaluate_candidates


def solve_problem3(config_module=None, verbose=True):
    """
    Solve problem 3: FY1, 3 bombs vs M1.

    Returns:
        (best_solution, best_fitness)
    """
    if config_module is None:
        config_module = config_a
    simulation.set_config(config_module)
    simulation.clear_keypoint_cache()
    cfg = config_module

    if verbose:
        print("=" * 60)
        print("问题3: FY1投放3枚烟幕弹对M1 (PSO优化)")
        print("=" * 60)

    drone_init = cfg.DRONES_INIT[0]
    target_keypoints = simulation.get_target_keypoints(36, 3)
    dt_opt = 0.02
    t_total = 30.0

    # ============================================================
    # Stage 1: Geometric init for first bomb to find good theta/speed
    # ============================================================
    if verbose:
        print("\n阶段1: 几何初始化 (确定航向和速度)...")

    geom_points = _geometric_feasible_points(drone_init, 0, cfg, n_t_det=20, n_alpha=12)
    candidates = evaluate_candidates(geom_points, drone_init, target_keypoints, cfg,
                                     dt=dt_opt, t_total=t_total, top_k=10)

    if not candidates:
        if verbose:
            print("  几何方法未找到可行解")
        return None, 0.0

    best_init_theta = candidates[0][1][0]
    best_init_speed = candidates[0][1][1]
    best_init_f = candidates[0][0]

    if verbose:
        print(f"  最佳初始点: theta={np.degrees(best_init_theta):.2f}deg  "
              f"v={best_init_speed:.1f}m/s  f={best_init_f:.4f}s")

    # ============================================================
    # Stage 2: PSO for all 3 bombs with narrow theta/speed bounds
    # ============================================================
    if verbose:
        print(f"\n阶段2: PSO优化3枚弹参数...")

    # Narrow bounds around geometric solution
    d_theta = 0.2
    d_speed = 20.0

    bounds = np.array([
        [max(0.0, best_init_theta - d_theta), min(2*np.pi, best_init_theta + d_theta)],
        [max(cfg.DRONE_SPEED_MIN, best_init_speed - d_speed),
         min(cfg.DRONE_SPEED_MAX, best_init_speed + d_speed)],
        [0.1, 8.0],                            # release1
        [cfg.BOMB_INTERVAL_MIN, 6.0],          # interval 1->2
        [cfg.BOMB_INTERVAL_MIN, 6.0],          # interval 2->3
        [0.1, 6.0],                            # delay1
        [0.1, 6.0],                            # delay2
        [0.1, 6.0],                            # delay3
    ])
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    dim = 8

    def evaluate(x):
        theta, speed = x[0], x[1]
        release_times = np.array([x[2], x[2] + x[3], x[2] + x[3] + x[4]])
        delays = np.array([x[5], x[6], x[7]])
        return simulation.simulate_multi_bomb_single_drone(
            drone_init, theta, speed, release_times, delays,
            np.array([0, 0, 0]), target_keypoints, dt_opt, t_total
        )

    # PSO
    n_particles = 40
    max_iter = 80
    w_start, w_end = 0.9, 0.4
    c1, c2 = 2.0, 2.0

    positions = np.random.uniform(lb, ub, size=(n_particles, dim))
    # Seed with geometric init as first bomb
    positions[0, 0] = best_init_theta
    positions[0, 1] = best_init_speed
    # Set first bomb params from geometric solution
    positions[0, 2] = candidates[0][1][2]  # release1
    positions[0, 3] = cfg.BOMB_INTERVAL_MIN  # gap
    positions[0, 4] = cfg.BOMB_INTERVAL_MIN
    positions[0, 5] = candidates[0][1][3]  # delay1
    positions[0, 6] = candidates[0][1][3] + 1.0
    positions[0, 7] = candidates[0][1][3] + 2.0

    velocities = np.zeros((n_particles, dim))
    values = np.array([evaluate(p) for p in positions])
    pbest_pos = positions.copy()
    pbest_val = values.copy()
    gbest_idx = np.argmax(values)
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = values[gbest_idx]

    for it in range(max_iter):
        w = w_start - (w_start - w_end) * it / max_iter
        r1 = np.random.random((n_particles, dim))
        r2 = np.random.random((n_particles, dim))
        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + \
                     c2 * r2 * (gbest_pos - positions)
        positions = np.clip(positions + velocities, lb, ub)
        new_values = np.array([evaluate(p) for p in positions])
        improved = new_values > pbest_val
        pbest_pos[improved] = positions[improved].copy()
        pbest_val[improved] = new_values[improved]
        if new_values.max() > gbest_val:
            gbest_val = new_values.max()
            gbest_pos = positions[new_values.argmax()].copy()
        if verbose and (it + 1) % 40 == 0:
            print(f"  PSO iter {it+1}/{max_iter}: best = {gbest_val:.4f}s  w={w:.3f}")

    best_x = gbest_pos
    best_f = gbest_val

    if verbose:
        x = best_x
        theta = x[0]; speed = x[1]
        release_times = np.array([x[2], x[2] + x[3], x[2] + x[3] + x[4]])
        delays = np.array([x[5], x[6], x[7]])
        direction = np.array([np.cos(theta), np.sin(theta), 0.0])

        print(f"\n优化结果:")
        print(f"  航向角theta: {theta:.4f} rad ({np.degrees(theta):.2f} deg)")
        print(f"  飞行速度: {speed:.2f} m/s")
        for i in range(3):
            release_pos = drone_init + speed * direction * release_times[i]
            detonation_pos = release_pos + speed * direction * delays[i]
            detonation_pos[2] -= 0.5 * cfg.G * delays[i] ** 2
            print(f"\n  烟幕弹{i+1}:")
            print(f"    投放时间: {release_times[i]:.4f} s")
            print(f"    起爆延时: {delays[i]:.4f} s")
            print(f"    起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")
        print(f"\n  总有效遮蔽时长: {best_f:.4f} s")

    return best_x, best_f


if __name__ == "__main__":
    x, f = solve_problem3(config_a)
    if x is not None:
        print(f"\nPaper reference: 5.625s (theta=10.35deg, v=101.80m/s)")
        print(f"Obtained: {f:.4f}s (theta={np.degrees(x[0]):.2f}deg, v={x[1]:.2f}m/s)")
