"""
问题3: 利用无人机FY1投放3枚烟幕干扰弹实施对M1的干扰

Paper cumcm25086 approach: APSO with 12 variables (per-bomb flexibility)
+ Lipschitz constant for adaptive step size

12 variables = 3 bombs × [theta, speed, release_time, detonation_delay]
Two-phase optimization: low coverage ratio for smooth landscape, then strict verification.
"""
import numpy as np
import simulation as sim
from apso import APSO


def solve_p3(config_module, target_keypoints=None, dt=None, t_total=None, verbose=True):
    """
    求解问题3: 单机三弹最优策略 (APSO with Lipschitz adaptation)

    Returns:
        x_opt: 12-D solution
        f_opt: maximum effective coverage time (verified)
    """
    sim.set_config(config_module)

    if dt is None:
        dt = 0.01  # Coarse for optimization speed
    if t_total is None:
        t_total = config_module.T_TOTAL

    drone_init = config_module.DRONES_INIT[0]
    bomb_interval_min = config_module.BOMB_INTERVAL_MIN

    if target_keypoints is None:
        target_keypoints = sim.get_target_keypoints(n_circle=30, n_layers=3)

    # 12 variables: per-bomb [theta, speed, release_time, detonation_delay] × 3
    bounds = []
    for i in range(3):
        bounds.append((np.pi * 0.97, np.pi * 1.0))          # theta_i (near pi)
        bounds.append((config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX))  # speed_i
        if i == 0:
            bounds.append((0.2, 6.0))                        # release_time_1
        else:
            bounds.append((bomb_interval_min, 6.0))          # interval from prev
        bounds.append((0.5, 6.0))                            # detonation_delay_i

    def objective(x):
        release_times = np.zeros(3)
        release_times[0] = x[2]
        release_times[1] = release_times[0] + max(bomb_interval_min, x[6])
        release_times[2] = release_times[1] + max(bomb_interval_min, x[10])
        detonation_delays = np.array([x[3], x[7], x[11]])

        masks = []
        for i in range(3):
            mask = sim._bomb_coverage_mask(
                drone_init, x[i*4], x[i*4+1], release_times[i],
                detonation_delays[i], 0, target_keypoints, dt, t_total
            )
            masks.append(mask)

        union_mask = sim.fit_union(masks)
        return union_mask.sum() * dt

    # Phase 1: Optimize with permissive coverage ratio
    sim.set_coverage_ratio(0.01)

    if verbose:
        print("=" * 60)
        print("问题3: FY1投放3枚烟幕弹对M1 (APSO优化)")
        print("=" * 60)
        print(f"\n12变量: 每枚弹独立 [theta, speed, release_time, delay] × 3")
        print(f"Optimization using permissive coverage (ratio=0.01)")
        print(f"APSO: swarm={config_module.APSO_SWARM_SIZE}, "
              f"iter={config_module.APSO_MAX_ITER}")
        print()

    init_center = np.array([
        np.pi, 100.0, 1.0, 3.8,     # bomb 1
        np.pi, 100.0, 1.5, 3.8,     # interval for bomb 2
        np.pi, 100.0, 1.5, 3.8,     # interval for bomb 3
    ])

    apso = APSO(
        objective, bounds,
        n_particles=config_module.APSO_SWARM_SIZE,
        max_iter=config_module.APSO_MAX_ITER,
        chi=config_module.APSO_CHI,
        c1=config_module.APSO_C1,
        c2=config_module.APSO_C2,
        w_start=config_module.APSO_W_START,
        w_end=config_module.APSO_W_END,
        maximize=True,
        verbose=verbose,
        init_center=init_center,
        init_spread=0.15,
    )
    x_opt, f_opt_coarse = apso.optimize()

    # Phase 2: Verify with strict coverage ratio
    sim.set_coverage_ratio(0.80)
    kps_verify = sim.get_target_keypoints()
    f_opt = objective(x_opt)  # Re-evaluate with strict ratio

    # Also verify with fine dt for accuracy
    dt_fine = config_module.DT if dt != config_module.DT else config_module.DT_FINE
    if dt != dt_fine:
        def objective_fine(x):
            release_times = np.zeros(3)
            release_times[0] = x[2]
            release_times[1] = release_times[0] + max(bomb_interval_min, x[6])
            release_times[2] = release_times[1] + max(bomb_interval_min, x[10])
            detonation_delays = np.array([x[3], x[7], x[11]])
            masks = []
            for i in range(3):
                mask = sim._bomb_coverage_mask(
                    drone_init, x[i*4], x[i*4+1], release_times[i],
                    detonation_delays[i], 0, kps_verify, dt_fine, t_total
                )
                masks.append(mask)
            return sim.fit_union(masks).sum() * dt_fine
        f_opt_fine = objective_fine(x_opt)
    else:
        f_opt_fine = f_opt

    if verbose:
        release_times = np.zeros(3)
        release_times[0] = x_opt[2]
        release_times[1] = release_times[0] + max(config_module.BOMB_INTERVAL_MIN, x_opt[6])
        release_times[2] = release_times[1] + max(config_module.BOMB_INTERVAL_MIN, x_opt[10])

        print(f"\n优化结果 (coarse_score={f_opt_coarse:.4f}):")
        print(f"  验证 (ratio=0.80, fine): {f_opt_fine:.4f} s")

        for i in range(3):
            theta_i = x_opt[i * 4]
            speed_i = x_opt[i * 4 + 1]
            delay_i = x_opt[i * 4 + 3]
            direction = np.array([np.cos(theta_i), np.sin(theta_i), 0.0])
            release_pos = drone_init + speed_i * direction * release_times[i]
            detonation_pos = release_pos + speed_i * direction * delay_i
            detonation_pos[2] -= 0.5 * config_module.G * delay_i ** 2
            print(f"\n  烟幕弹{i+1}: theta={np.degrees(theta_i):.2f}deg, "
                  f"v={speed_i:.1f}m/s, t_rel={release_times[i]:.4f}s, delay={delay_i:.4f}s")
            print(f"    起爆点: ({detonation_pos[0]:.0f}, {detonation_pos[1]:.0f}, {detonation_pos[2]:.0f})")

    return x_opt, f_opt_fine if dt != dt_fine else f_opt


if __name__ == "__main__":
    import config_a
    solve_p3(config_a)
