"""
问题4: 利用FY1、FY2、FY3各投放1枚烟幕干扰弹，实施对M1的干扰

Paper cumcm25086 approach: Multi-Island PSO with "+"(union) strategy
Two-phase optimization: low coverage ratio for smooth landscape, then strict verification.
"""
import numpy as np
import simulation as sim
from multi_island_pso import MultiIslandPSO


def solve_p4(config_module, target_keypoints=None, dt=None, t_total=None, verbose=True):
    """
    求解问题4: 三机各一弹对M1 (Multi-Island PSO优化)

    Returns:
        x_opt: 12-D optimal parameters
        f_opt: maximum effective coverage time (verified)
    """
    sim.set_config(config_module)

    if dt is None:
        dt = 0.01  # Coarse for speed
    if t_total is None:
        t_total = config_module.T_TOTAL

    if target_keypoints is None:
        target_keypoints = sim.get_target_keypoints(n_circle=30, n_layers=3)

    n_drones = 3

    # 12 variables: 3 × [theta, speed, release_time, detonation_delay]
    bounds = []
    for i in range(n_drones):
        bounds.append((np.pi * 0.85, np.pi * 1.0))    # theta_i (toward origin)
        bounds.append((config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX))  # speed_i
        bounds.append((0.2, 20.0))                     # release_time_i
        bounds.append((0.5, 15.0))                     # detonation_delay_i

    def objective(x):
        drone_params = []
        for i in range(n_drones):
            drone_params.append({
                'drone_init': config_module.DRONES_INIT[i],
                'theta': x[i * 4],
                'speed': x[i * 4 + 1],
                'release_times': np.array([x[i * 4 + 2]]),
                'detonation_delays': np.array([x[i * 4 + 3]]),
                'missile_indices': [0],  # All targeting M1
            })
        total_time, _ = sim.simulate_multi_drone_multi_bomb(
            drone_params, dt=dt, t_total=t_total
        )
        return total_time

    # Phase 1: Optimize with permissive coverage ratio
    sim.set_coverage_ratio(0.01)

    if verbose:
        print("=" * 60)
        print("问题4: FY1/FY2/FY3各1弹对M1 (Multi-Island PSO)")
        print("=" * 60)
        print(f"\n12变量: 3 drones × [theta, speed, release_time, delay]")
        print(f"Optimization using permissive coverage (ratio=0.01)")
        print(f"Multi-Island: {config_module.MI_N_ISLANDS} islands × "
              f"{config_module.MI_SWARM_PER_ISLAND} particles, "
              f"max_iter={config_module.MI_MAX_ITER}")
        print()

    mi_pso = MultiIslandPSO(
        objective, bounds,
        n_islands=config_module.MI_N_ISLANDS,
        swarm_per_island=config_module.MI_SWARM_PER_ISLAND,
        max_iter=config_module.MI_MAX_ITER,
        migration_interval=config_module.MI_MIGRATION_INTERVAL,
        migration_rate=config_module.MI_MIGRATION_RATE,
        elite_size=config_module.MI_ELITE_SIZE,
        chi=config_module.APSO_CHI,
        c1=config_module.APSO_C1,
        c2=config_module.APSO_C2,
        w_start=config_module.APSO_W_START,
        w_end=config_module.APSO_W_END,
        maximize=True,
        verbose=verbose,
    )
    x_opt, f_opt_coarse = mi_pso.optimize()

    # Phase 2: Verify with strict coverage ratio
    sim.set_coverage_ratio(0.80)
    f_opt = objective(x_opt)

    # Fine verification
    kps_verify = sim.get_target_keypoints()
    dt_fine = config_module.DT_FINE

    def objective_fine(x):
        drone_params = []
        for i in range(n_drones):
            drone_params.append({
                'drone_init': config_module.DRONES_INIT[i],
                'theta': x[i * 4],
                'speed': x[i * 4 + 1],
                'release_times': np.array([x[i * 4 + 2]]),
                'detonation_delays': np.array([x[i * 4 + 3]]),
                'missile_indices': [0],
            })
        total_time, _ = sim.simulate_multi_drone_multi_bomb(
            drone_params, dt=dt_fine, t_total=t_total
        )
        return total_time

    f_opt_fine = objective_fine(x_opt)

    if verbose:
        drone_names = ['FY1', 'FY2', 'FY3']
        print(f"\n优化结果 (coarse_score={f_opt_coarse:.4f}):")
        print(f"  验证 (ratio=0.80, fine): {f_opt_fine:.4f} s")

        for i in range(n_drones):
            theta = x_opt[i * 4]
            speed = x_opt[i * 4 + 1]
            release_time = x_opt[i * 4 + 2]
            delay = x_opt[i * 4 + 3]
            direction = np.array([np.cos(theta), np.sin(theta), 0.0])
            release_pos = config_module.DRONES_INIT[i] + speed * direction * release_time
            detonation_pos = release_pos + speed * direction * delay
            detonation_pos[2] -= 0.5 * config_module.G * delay ** 2
            print(f"\n  {drone_names[i]}: theta={np.degrees(theta):.1f}deg, "
                  f"v={speed:.1f}m/s, t_rel={release_time:.4f}s, delay={delay:.4f}s")
            print(f"    起爆点: ({detonation_pos[0]:.0f}, {detonation_pos[1]:.0f}, {detonation_pos[2]:.0f})")

    return x_opt, f_opt_fine


if __name__ == "__main__":
    import config_a
    solve_p4(config_a)
