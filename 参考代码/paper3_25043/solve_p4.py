"""
solve_p4.py - Problem 4: Three drones, one bomb each, one missile (3-1-1).
Two-stage independent PSO + union coverage.
Paper's result: 11.56s coverage.
"""
import numpy as np
import time as time_module
import simulation as sim
from pso_solver import PSOSolver


def solve_p4(config, label="P4"):
    """
    Problem 4 (3-1-1): Three drones, one bomb each, one missile.
    """
    print(f"\n{'='*60}")
    print(f"  {label} - 3 Independent Two-stage PSO + Union")
    print(f"{'='*60}")

    missile_start = config.MISSILES_INIT[0]
    target_center = config.TARGET_CENTER.copy()
    fake_target = config.FAKE_TARGET.copy()
    n_drones = 3
    all_results = []
    total_start = time_module.time()

    for drone_idx in range(n_drones):
        print(f"\n  --- Drone {drone_idx+1}/{n_drones} ---")
        drone_start = config.DRONES_INIT[drone_idx].copy()

        bounds = [
            (0.0, 2 * np.pi),
            (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX),
            (0.0, 10.0),
            (2.0, 8.0),
        ]

        def make_smooth(d_start):
            def obj(x):
                theta, v, t_rel, t_e = x
                dp, td = sim.compute_bomb_trajectory_general(
                    d_start, v, theta, t_rel, t_e, config
                )
                st = sim.compute_smoke_proximity_score(
                    dp, td, missile_start, config.MISSILE_SPEED,
                    target_center, config, R_multiplier=2.5
                )
                sf = sim.compute_smoke_proximity_score(
                    dp, td, missile_start, config.MISSILE_SPEED,
                    fake_target, config, R_multiplier=2.5
                )
                return st - 3.0 * sf
            return obj

        def make_exact(d_start):
            def obj(x):
                theta, v, t_rel, t_e = x
                dp, td = sim.compute_bomb_trajectory_general(
                    d_start, v, theta, t_rel, t_e, config
                )
                dur_t = sim.compute_total_occlusion_duration(
                    [dp], [td], missile_start, config.MISSILE_SPEED,
                    target_center, config
                )
                dur_f = sim.compute_total_occlusion_duration(
                    [dp], [td], missile_start, config.MISSILE_SPEED,
                    fake_target, config
                )
                mft = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
                mft_f = np.linalg.norm(fake_target - missile_start) / config.MISSILE_SPEED
                cov_t = min(dur_t/mft, 1.0) if mft > 0 else 0.0
                cov_f = min(dur_f/mft_f, 1.0) if mft_f > 0 else 0.0
                return cov_t - 5.0 * cov_f
            return obj

        # Stage 1: smooth
        theta_seed = sim.compute_optimal_theta_to_line(
            drone_start, 100.0, 5.0, 3.5, missile_start, target_center, config
        )
        init_pos = np.array([theta_seed, 100.0, 5.0, 3.5])

        s1 = PSOSolver(40, 50, config.W_START, config.W_END, config.C1, config.C2, verbose=False)
        bx1, _, _, t1 = s1.optimize(make_smooth(drone_start), bounds, maximize=True,
                                     init_positions=init_pos, init_radius=0.15)

        # Stage 2: exact
        s2 = PSOSolver(20, 30, 0.5, 0.3, config.C1, config.C2, verbose=False)
        bx, _, _, t2 = s2.optimize(make_exact(drone_start), bounds, maximize=True,
                                    init_positions=bx1, init_radius=0.05)

        theta_opt, v_opt, t_rel_opt, t_e_opt = bx
        dp, td = sim.compute_bomb_trajectory_general(
            drone_start, v_opt, theta_opt, t_rel_opt, t_e_opt, config
        )
        dur_t = sim.compute_total_occlusion_duration(
            [dp], [td], missile_start, config.MISSILE_SPEED, target_center, config
        )
        mft = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
        cov_t = min(dur_t/mft, 1.0) if mft > 0 else 0.0

        all_results.append({
            'drone_idx': drone_idx, 'theta': np.degrees(theta_opt), 'v': v_opt,
            't_rel': t_rel_opt, 't_e': t_e_opt,
            'det_point': dp, 't_det': td,
            'coverage_true': cov_t, 'duration': dur_t,
            'pso_time': t1 + t2,
        })
        print(f"    theta={np.degrees(theta_opt):.1f}deg, v={v_opt:.1f}m/s, "
              f"dur={dur_t:.2f}s, cov={cov_t:.3f}, time={t1+t2:.1f}s")

    # Union coverage
    all_det = [r['det_point'] for r in all_results]
    all_td = [r['t_det'] for r in all_results]
    dur_union = sim.compute_total_occlusion_duration(
        all_det, all_td, missile_start, config.MISSILE_SPEED, target_center, config
    )
    mft = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
    union_cov = min(dur_union / mft, 1.0)
    total_time = time_module.time() - total_start

    print(f"\n  P4 Final: union dur={dur_union:.4f}s, cov={union_cov:.4f}, total={total_time:.1f}s")
    for r in all_results:
        print(f"    Drone {r['drone_idx']+1}: theta={r['theta']:.1f}deg, v={r['v']:.0f}m/s, "
              f"t_rel={r['t_rel']:.2f}s, dur={r['duration']:.2f}s")

    return {
        'drones': all_results,
        'union_coverage_true': union_cov, 'duration_true': dur_union,
        'total_time': total_time,
    }


if __name__ == "__main__":
    import config_a
    solve_p4(config_a, "P4-A题")
