"""
问题2: 利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰
确定FY1的飞行方向、速度、投放点、起爆点，使遮蔽时间尽可能长

Paper cumcm25086 approach: APSO with 4 variables [theta, v, t_rel, t_e]
- Clerc's constriction factor for convergence guarantee
- Adaptive inertia weight
- Elite preservation
"""
import numpy as np
import simulation as sim
from apso import APSO


def solve_p2(config_module, target_keypoints=None, dt=None, t_total=None, verbose=True):
    """
    求解问题2: 单机单弹最优策略 (APSO优化)

    Parameters:
        config_module: config_a or config_c

    Returns:
        x_opt: [theta, speed, release_time, detonation_delay]
        f_opt: maximum effective coverage time
    """
    sim.set_config(config_module)

    if dt is None:
        dt = config_module.DT
    if t_total is None:
        t_total = config_module.T_TOTAL

    drone_init = config_module.DRONES_INIT[0]  # FY1

    if target_keypoints is None:
        # 使用较少的keypoints加速优化
        target_keypoints = sim.get_target_keypoints(n_circle=180, n_layers=5)

    # 决策变量: [theta, speed, release_time, detonation_delay]
    # Physics-informed search range (smooth objective with low coverage ratio)
    bounds = [
        (np.pi * 0.97, np.pi * 1.0),                    # theta: within ~5.4deg of pi
        (config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX),  # speed (m/s)
        (0.2, 15.0),                                      # release_time (s)
        (0.5, 6.0),                                       # detonation_delay (s)
    ]

    def objective(x):
        theta, speed, release_time, detonation_delay = x
        return sim.simulate_single_bomb(
            drone_init, theta, speed, release_time, detonation_delay,
            missile_idx=0, target_keypoints=target_keypoints,
            dt=dt, t_total=t_total
        )

    # Phase 1: Optimize with permissive coverage ratio for smooth objective
    sim.set_coverage_ratio(0.01)  # 1% threshold = smooth landscape

    if verbose:
        print("=" * 60)
        print("问题2: FY1单机单弹最优投放策略 (APSO优化)")
        print("=" * 60)
        print(f"\n变量范围:")
        print(f"  theta: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}] rad")
        print(f"  speed: [{bounds[1][0]}, {bounds[1][1]}] m/s")
        print(f"  release_time: [{bounds[2][0]}, {bounds[2][1]}] s")
        print(f"  detonation_delay: [{bounds[3][0]}, {bounds[3][1]}] s")
        print(f"\nAPSO参数: swarm={config_module.APSO_SWARM_SIZE}, "
              f"iter={config_module.APSO_MAX_ITER}, chi={config_module.APSO_CHI}")
        print(f"Optimization using permissive coverage (ratio=0.01)")
        print()

    # Seed near known-good region: theta=pi, speed=100, t_rel=1.5, delay=3.8
    init_center = np.array([np.pi, 100.0, 1.5, 3.8])

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
    f_opt = sim.simulate_single_bomb(
        drone_init, x_opt[0], x_opt[1], x_opt[2], x_opt[3],
        missile_idx=0, target_keypoints=target_keypoints,
        dt=config_module.DT_FINE, t_total=t_total
    )

    if verbose:
        print(f"\n优化结果 (优化阶段score={f_opt_coarse:.4f} @ ratio=0.01):")
        print(f"  最终验证 (ratio=0.80): {f_opt:.4f} s")
        print(f"  航向角theta: {x_opt[0]:.6f} rad ({np.degrees(x_opt[0]):.4f} deg)")
        print(f"  飞行速度: {x_opt[1]:.2f} m/s")
        print(f"  投放时间: {x_opt[2]:.4f} s")
        print(f"  起爆延时: {x_opt[3]:.4f} s")

        direction = np.array([np.cos(x_opt[0]), np.sin(x_opt[0]), 0.0])
        release_pos = drone_init + x_opt[1] * direction * x_opt[2]
        detonation_pos = release_pos + x_opt[1] * direction * x_opt[3]
        detonation_pos[2] -= 0.5 * config_module.G * x_opt[3] ** 2

        print(f"  投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
        print(f"  起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")

    return x_opt, f_opt


if __name__ == "__main__":
    import config_a
    solve_p2(config_a)
