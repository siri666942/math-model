"""
solve_p5.py - Problem 5: Five drones, three bombs each, three missiles (5-3-3).
Two-stage PSO per drone + union coverage.
Paper's result: 34.00s coverage.
"""
import numpy as np
import time as time_module
import simulation as sim
from pso_solver import PSOSolver


def solve_p5(config, label="P5"):
    """
    Problem 5 (5-3-3): Five drones, three bombs each, three missiles.
    """
    print(f"\n{'='*60}")
    print(f"  {label} - Multi-Drone Two-stage PSO (5 drones)")
    print(f"{'='*60}")

    n_drones = 5
    n_bombs_per_drone = 3
    n_missiles = 3
    interval_min = config.BOMB_INTERVAL_MIN
    target_center = config.TARGET_CENTER.copy()
    fake_target = config.FAKE_TARGET.copy()
    all_bomb_results = []
    total_start = time_module.time()

    for drone_idx in range(n_drones):
        print(f"\n  --- Drone {drone_idx+1}/{n_drones} ---")
        drone_start = config.DRONES_INIT[drone_idx].copy()

        bounds = [
            (0.0, 2 * np.pi),
            (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX),
        ] + [(0.0, 25.0)] * n_bombs_per_drone + [(2.0, 8.0)] * n_bombs_per_drone

        def make_fns(d_start):
            def decode(x):
                theta = x[0]; v = x[1]
                tr = np.sort(x[2:2+n_bombs_per_drone])
                te = x[2+n_bombs_per_drone:2+2*n_bombs_per_drone]
                for i in range(1, n_bombs_per_drone):
                    tr[i] = max(tr[i], tr[i-1] + interval_min)
                return theta, v, tr, te

            def smooth_obj(x):
                theta, v, tr, te = decode(x)
                best = 0.0
                for i in range(n_bombs_per_drone):
                    dp, td = sim.compute_bomb_trajectory_general(
                        d_start, v, theta, tr[i], te[i], config
                    )
                    s = sim.compute_smoke_proximity_score(
                        dp, td, config.MISSILES_INIT[0], config.MISSILE_SPEED,
                        target_center, config, R_multiplier=2.5
                    )
                    best = max(best, s)
                return best

            def exact_obj(x):
                theta, v, tr, te = decode(x)
                dps, tds = [], []
                for i in range(n_bombs_per_drone):
                    dp, td = sim.compute_bomb_trajectory_general(
                        d_start, v, theta, tr[i], te[i], config
                    ); dps.append(dp); tds.append(td)
                dur_t = sim.compute_total_occlusion_duration(
                    dps, tds, config.MISSILES_INIT[0], config.MISSILE_SPEED,
                    target_center, config
                )
                mft = np.linalg.norm(target_center - config.MISSILES_INIT[0]) / config.MISSILE_SPEED
                return min(dur_t/mft, 1.0) if mft > 0 else 0.0
            return decode, smooth_obj, exact_obj

        decode_fn, smooth_fn, exact_fn = make_fns(drone_start)

        # Seed
        theta_seed = sim.compute_optimal_theta_to_line(
            drone_start, 100.0, 8.0, 3.5, config.MISSILES_INIT[0], target_center, config
        )
        t_rel_seed = np.linspace(5, 15, n_bombs_per_drone)
        t_e_seed = np.ones(n_bombs_per_drone) * 3.5
        init_pos = np.concatenate([[theta_seed, 100.0], t_rel_seed, t_e_seed])

        # Stage 1: smooth PSO
        s1 = PSOSolver(30, 40, config.W_START, config.W_END, config.C1, config.C2,
                       verbose=False)
        bx1, _, _, t1 = s1.optimize(smooth_fn, bounds, maximize=True,
                                     init_positions=init_pos, init_radius=0.15)

        # Stage 2: exact PSO
        s2 = PSOSolver(20, 30, 0.5, 0.3, config.C1, config.C2, verbose=False)
        bx, _, _, t2 = s2.optimize(exact_fn, bounds, maximize=True,
                                    init_positions=bx1, init_radius=0.05)

        theta_opt, v_opt, tr_opt, te_opt = decode_fn(bx)
        for i in range(n_bombs_per_drone):
            dp, td = sim.compute_bomb_trajectory_general(
                drone_start, v_opt, theta_opt, tr_opt[i], te_opt[i], config
            )
            all_bomb_results.append((dp, td))

        print(f"    theta={np.degrees(theta_opt):.1f}deg, v={v_opt:.1f}m/s, time={t1+t2:.1f}s")

    # Union coverage across all missiles
    all_det = [b[0] for b in all_bomb_results]
    all_td = [b[1] for b in all_bomb_results]
    missiles_used = config.MISSILES_INIT[:n_missiles]

    mft_list = [np.linalg.norm(target_center - ms) / config.MISSILE_SPEED
                for ms in missiles_used]
    dur_list = [sim.compute_total_occlusion_duration(
        all_det, all_td, ms, config.MISSILE_SPEED, target_center, config
    ) for ms in missiles_used]
    cov_list = [min(d/m, 1.0) for d, m in zip(dur_list, mft_list)]
    best_cov = max(cov_list)

    total_time = time_module.time() - total_start
    print(f"\n  P5 Final: bombs={len(all_bomb_results)}, best_cov={best_cov:.4f}, "
          f"total={total_time:.1f}s")
    for i in range(min(n_missiles, 3)):
        print(f"    Missile {i+1}: dur={dur_list[i]:.2f}s, cov={cov_list[i]:.4f}")

    return {
        'n_bombs': len(all_bomb_results),
        'coverage_true': best_cov, 'duration_true': max(dur_list),
        'total_time': total_time,
    }


if __name__ == "__main__":
    import config_a
    solve_p5(config_a, "P5-A题")
