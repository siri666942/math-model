"""
solve_p2.py - Problem 2: Single drone, single bomb (1-1-1).
Two-stage PSO: coarse proximity optimization + fine refinement.
Paper's result: 4.590s occlusion, theta=7.18deg, v=95.73m/s.
"""
import numpy as np
import time as time_module
import simulation as sim
from pso_solver import PSOSolver


def solve_p2(config, label="P2"):
    """
    Problem 2 (1-1-1): Two-stage optimization.
    Stage 1: PSO with smooth proximity objective (large radius).
    Stage 2: PSO refinement with exact occlusion duration.
    """
    print(f"\n{'='*60}")
    print(f"  {label} - Two-stage PSO (50 particles, 80+40 iterations)")
    print(f"{'='*60}")

    drone_start = config.DRONES_INIT[0].copy()
    missile_start = config.MISSILES_INIT[0]
    target_center = config.TARGET_CENTER.copy()
    fake_target = config.FAKE_TARGET.copy()

    bounds = [
        (0.0, 2 * np.pi),
        (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX),
        (0.0, 10.0),
        (2.0, 8.0),
    ]

    # ============ Stage 1: Smooth proximity PSO ============
    print(f"\n  Stage 1: Smooth proximity PSO (R_multiplier=3.0)")

    def smooth_objective(x):
        theta, v, t_rel, t_e = x
        det_pt, t_det = sim.compute_bomb_trajectory_general(
            drone_start, v, theta, t_rel, t_e, config
        )
        # True target: maximize proximity
        score_true = sim.compute_smoke_proximity_score(
            det_pt, t_det, missile_start, config.MISSILE_SPEED,
            target_center, config, R_multiplier=3.0
        )
        # Fake target: minimize proximity
        score_fake = sim.compute_smoke_proximity_score(
            det_pt, t_det, missile_start, config.MISSILE_SPEED,
            fake_target, config, R_multiplier=3.0
        )
        return score_true - 3.0 * score_fake

    # Test seed
    t_rel_seed = 5.0
    t_e_seed = 3.5
    v_seed = 100.0
    theta_seed = sim.compute_optimal_theta_to_line(
        drone_start, v_seed, t_rel_seed, t_e_seed,
        missile_start, target_center, config
    )
    init_pos = np.array([theta_seed, v_seed, t_rel_seed, t_e_seed])
    seed_fit = smooth_objective(init_pos)
    print(f"  Seed: theta={np.degrees(theta_seed):.1f}deg, v={v_seed:.0f}m/s, "
          f"t_rel={t_rel_seed:.1f}s, t_e={t_e_seed:.1f}s, fit={seed_fit:.4f}")

    solver1 = PSOSolver(
        n_particles=50, n_iterations=80,
        w_start=config.W_START, w_end=config.W_END,
        c1=config.C1, c2=config.C2, verbose=True
    )
    best_x1, best_fit1, hist1, t1 = solver1.optimize(
        smooth_objective, bounds, maximize=True,
        init_positions=init_pos, init_radius=0.2
    )

    theta1, v1, t_rel1, t_e1 = best_x1
    print(f"  Stage 1 best: theta={np.degrees(theta1):.1f}deg, v={v1:.1f}m/s, "
          f"t_rel={t_rel1:.2f}s, t_e={t_e1:.2f}s, fit={best_fit1:.4f}")

    # ============ Stage 2: Exact occlusion PSO ============
    print(f"\n  Stage 2: Exact occlusion refinement")

    def exact_objective(x):
        theta, v, t_rel, t_e = x
        det_pt, t_det = sim.compute_bomb_trajectory_general(
            drone_start, v, theta, t_rel, t_e, config
        )
        dur_true = sim.compute_total_occlusion_duration(
            [det_pt], [t_det], missile_start, config.MISSILE_SPEED,
            target_center, config
        )
        mft = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
        cov_true = min(dur_true / mft, 1.0) if mft > 0 else 0.0

        dur_fake = sim.compute_total_occlusion_duration(
            [det_pt], [t_det], missile_start, config.MISSILE_SPEED,
            fake_target, config
        )
        mft_f = np.linalg.norm(fake_target - missile_start) / config.MISSILE_SPEED
        cov_fake = min(dur_fake / mft_f, 1.0) if mft_f > 0 else 0.0

        return cov_true - 5.0 * cov_fake

    solver2 = PSOSolver(
        n_particles=30, n_iterations=40,
        w_start=0.6, w_end=0.3,
        c1=config.C1, c2=config.C2, verbose=True
    )
    best_x, best_fit, hist2, t2 = solver2.optimize(
        exact_objective, bounds, maximize=True,
        init_positions=best_x1, init_radius=0.05
    )

    theta_opt, v_opt, t_rel_opt, t_e_opt = best_x
    det_opt, t_det_opt = sim.compute_bomb_trajectory_general(
        drone_start, v_opt, theta_opt, t_rel_opt, t_e_opt, config
    )

    dur_true = sim.compute_total_occlusion_duration(
        [det_opt], [t_det_opt], missile_start, config.MISSILE_SPEED,
        target_center, config
    )
    dur_fake = sim.compute_total_occlusion_duration(
        [det_opt], [t_det_opt], missile_start, config.MISSILE_SPEED,
        fake_target, config
    )
    mft_true = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
    mft_fake = np.linalg.norm(fake_target - missile_start) / config.MISSILE_SPEED
    cov_true = min(dur_true / mft_true, 1.0)
    cov_fake = min(dur_fake / mft_fake, 1.0)

    total_pso_time = t1 + t2
    print(f"\n  Optimal: theta={np.degrees(theta_opt):.2f}deg, v={v_opt:.2f}m/s, "
          f"t_rel={t_rel_opt:.2f}s, t_e={t_e_opt:.2f}s")
    print(f"  True target cov: {cov_true:.4f} ({cov_true*100:.1f}%), dur: {dur_true:.4f}s")
    print(f"  False target cov: {cov_fake:.4f} ({cov_fake*100:.1f}%)")
    print(f"  Total PSO time: {total_pso_time:.2f}s (Stage1: {t1:.1f}s, Stage2: {t2:.1f}s)")

    return {
        'theta': np.degrees(theta_opt), 'v': v_opt,
        't_rel': t_rel_opt, 't_e': t_e_opt,
        'coverage_true': cov_true, 'duration_true': dur_true,
        'coverage_fake': cov_fake,
        'objective': best_fit, 'pso_time': total_pso_time,
    }


if __name__ == "__main__":
    import config_a
    solve_p2(config_a, "P2-A题")
