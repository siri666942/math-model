"""
solve_p3.py - Problem 3: Single drone, three bombs, one missile (1-3-1).
Two-stage PSO+SA: smooth proximity + exact refinement.
Paper's result: 6.41s, theta=8.70deg, v=110.28m/s.
"""
import numpy as np
import time as time_module
import simulation as sim
from pso_solver import PSOSolver


def solve_p3(config, label="P3"):
    """
    Problem 3 (1-3-1): One drone, three bombs, one missile.
    Two-stage PSO with SA hybridization.
    """
    print(f"\n{'='*60}")
    print(f"  {label} - Two-stage PSO+SA (100+50 iter, 12 vars)")
    print(f"{'='*60}")

    drone_start = config.DRONES_INIT[0].copy()
    missile_start = config.MISSILES_INIT[0]
    target_center = config.TARGET_CENTER.copy()
    fake_target = config.FAKE_TARGET.copy()
    n_bombs = 3
    interval_min = config.BOMB_INTERVAL_MIN

    bounds = [
        (0.0, 2 * np.pi),
        (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX),
    ] + [(0.0, 15.0)] * n_bombs + [(2.0, 8.0)] * n_bombs

    def decode(x):
        theta = x[0]; v = x[1]
        t_rels = np.sort(x[2:2+n_bombs])
        t_es = x[2+n_bombs:2+2*n_bombs]
        for i in range(1, n_bombs):
            t_rels[i] = max(t_rels[i], t_rels[i-1] + interval_min)
        return theta, v, t_rels, t_es

    # Stage 1: Smooth proximity
    print(f"\n  Stage 1: Smooth proximity PSO+SA (R_multiplier=2.5)")
    def smooth_obj(x):
        theta, v, t_rels, t_es = decode(x)
        dps, tds = [], []
        for i in range(n_bombs):
            dp, td = sim.compute_bomb_trajectory_general(
                drone_start, v, theta, t_rels[i], t_es[i], config
            ); dps.append(dp); tds.append(td)
        # Use union of proximity scores
        best_true = 0.0; best_fake = 0.0
        for dp, td in zip(dps, tds):
            s = sim.compute_smoke_proximity_score(
                dp, td, missile_start, config.MISSILE_SPEED,
                target_center, config, R_multiplier=2.5
            )
            best_true = max(best_true, s)
            s_f = sim.compute_smoke_proximity_score(
                dp, td, missile_start, config.MISSILE_SPEED,
                fake_target, config, R_multiplier=2.5
            )
            best_fake = max(best_fake, s_f)
        return best_true - 3.0 * best_fake

    # Seed
    t_rel_init = np.array([4.0, 6.0, 8.0]) + np.random.randn(3) * 0.5
    t_e_init = np.ones(3) * 3.5 + np.random.randn(3) * 0.3
    v_seed = 100.0
    theta_seed = sim.compute_optimal_theta_to_line(
        drone_start, v_seed, 5.0, 3.5, missile_start, target_center, config
    )
    init_pos = np.concatenate([[theta_seed, v_seed], t_rel_init, t_e_init])

    solver1 = PSOSolver(
        n_particles=100, n_iterations=50,
        w_start=config.W_START, w_end=config.W_END,
        c1=config.C1, c2=config.C2,
        use_sa=True, sa_T0=1.0, sa_cooling=0.95, sa_interval=10, verbose=True
    )
    best_x1, best_fit1, h1, t1 = solver1.optimize(
        smooth_obj, bounds, maximize=True,
        init_positions=init_pos, init_radius=0.15
    )

    # Stage 2: Exact occlusion
    print(f"\n  Stage 2: Exact occlusion refinement")
    def exact_obj(x):
        theta, v, t_rels, t_es = decode(x)
        dps, tds = [], []
        for i in range(n_bombs):
            dp, td = sim.compute_bomb_trajectory_general(
                drone_start, v, theta, t_rels[i], t_es[i], config
            ); dps.append(dp); tds.append(td)
        dur_t = sim.compute_total_occlusion_duration(
            dps, tds, missile_start, config.MISSILE_SPEED, target_center, config
        )
        dur_f = sim.compute_total_occlusion_duration(
            dps, tds, missile_start, config.MISSILE_SPEED, fake_target, config
        )
        mft = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
        mft_f = np.linalg.norm(fake_target - missile_start) / config.MISSILE_SPEED
        cov_t = min(dur_t/mft, 1.0) if mft > 0 else 0.0
        cov_f = min(dur_f/mft_f, 1.0) if mft_f > 0 else 0.0
        return cov_t - 5.0 * cov_f

    solver2 = PSOSolver(
        n_particles=40, n_iterations=40,
        w_start=0.5, w_end=0.3,
        c1=config.C1, c2=config.C2, verbose=True
    )
    best_x, best_fit, h2, t2 = solver2.optimize(
        exact_obj, bounds, maximize=True,
        init_positions=best_x1, init_radius=0.05
    )

    theta_opt, v_opt, t_rels_opt, t_es_opt = decode(best_x)
    dps, tds = [], []
    for i in range(n_bombs):
        dp, td = sim.compute_bomb_trajectory_general(
            drone_start, v_opt, theta_opt, t_rels_opt[i], t_es_opt[i], config
        ); dps.append(dp); tds.append(td)

    dur_t = sim.compute_total_occlusion_duration(
        dps, tds, missile_start, config.MISSILE_SPEED, target_center, config
    )
    dur_f = sim.compute_total_occlusion_duration(
        dps, tds, missile_start, config.MISSILE_SPEED, fake_target, config
    )
    mft = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
    mft_f = np.linalg.norm(fake_target - missile_start) / config.MISSILE_SPEED
    cov_t = min(dur_t/mft, 1.0); cov_f = min(dur_f/mft_f, 1.0)

    total_t = t1 + t2
    print(f"\n  Optimal: theta={np.degrees(theta_opt):.2f}deg, v={v_opt:.2f}m/s")
    for i in range(n_bombs):
        print(f"    Bomb {i+1}: t_rel={t_rels_opt[i]:.3f}s, t_e={t_es_opt[i]:.3f}s")
    print(f"  True target cov: {cov_t:.4f}, dur: {dur_t:.4f}s")
    print(f"  False target cov: {cov_f:.4f}")
    print(f"  Total PSO time: {total_t:.2f}s (S1: {t1:.1f}s, S2: {t2:.1f}s)")

    return {
        'theta': np.degrees(theta_opt), 'v': v_opt,
        't_rels': t_rels_opt.tolist(), 't_es': t_es_opt.tolist(),
        'coverage_true': cov_t, 'duration_true': dur_t,
        'coverage_fake': cov_f, 'pso_time': total_t,
    }


if __name__ == "__main__":
    import config_a
    solve_p3(config_a, "P3-A题")
