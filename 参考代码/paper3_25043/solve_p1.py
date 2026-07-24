"""
solve_p1.py - Problem 1: Single drone, single bomb, interval-based method.
Paper's result: 1.39s occlusion duration (computation time 0.027s).

The drone angle is positioned to place the smoke on the missile-target line.
This requires theta ~ 177.33 deg (paper states pi = 180 deg as approximation).
"""
import numpy as np
import time as time_module
import simulation as sim


def solve_p1(config, label="P1"):
    """
    Problem 1: Compute occlusion duration using interval-based method.
    """
    print(f"\n{'='*60}")
    print(f"  {label} - Interval-based Method (Paper's Innovation)")
    print(f"{'='*60}")

    drone_start = config.DRONES_INIT[0].copy()
    missile_start = config.MISSILES_INIT[0].copy()
    target_center = config.TARGET_CENTER.copy()
    drone_speed = config.P1_DRONE_SPEED
    release_time = config.P1_RELEASE_TIME
    detonation_delay = config.P1_DETONATION_DELAY

    # Compute optimal theta
    theta_opt = sim.compute_optimal_theta_to_line(
        drone_start, drone_speed, release_time, detonation_delay,
        missile_start, target_center, config
    )
    print(f"\n  Optimal theta: {np.degrees(theta_opt):.2f} deg "
          f"(paper: pi = 180 deg)")

    # Compute bomb trajectory
    det_point, t_det = sim.compute_bomb_trajectory_general(
        drone_start, drone_speed, theta_opt, release_time, detonation_delay, config
    )

    print(f"  Drone start: ({drone_start[0]:.0f}, {drone_start[1]:.0f}, {drone_start[2]:.0f})")
    print(f"  Drone speed: {drone_speed:.1f} m/s")
    print(f"  Release: {release_time:.1f}s, Delay: {detonation_delay:.1f}s")
    print(f"  Detonation: ({det_point[0]:.1f}, {det_point[1]:.1f}, {det_point[2]:.1f}) at t={t_det:.3f}s")

    # Start timing
    start_time = time_module.time()

    # Compute occlusion for TARGET CENTER (paper's simplified approach)
    interval_center = sim.find_occlusion_interval_single_bomb(
        det_point, t_det, missile_start, config.MISSILE_SPEED, target_center, config
    )
    merged_center = sim.merge_intervals(interval_center)
    total_duration = sum(b - a for a, b in merged_center)

    elapsed = time_module.time() - start_time

    missile_flight_time = np.linalg.norm(target_center - missile_start) / config.MISSILE_SPEED
    occlusion_ratio = total_duration / missile_flight_time if missile_flight_time > 0 else 0.0

    print(f"\n  P1 Results (target center):")
    print(f"    Occlusion duration: {total_duration:.4f}s")
    print(f"    Occlusion ratio: {occlusion_ratio:.4f} ({occlusion_ratio*100:.1f}%)")
    print(f"    Missile flight time: {missile_flight_time:.2f}s")
    print(f"    Computation time: {elapsed:.4f}s")

    if merged_center:
        print(f"    Intervals: ", end="")
        for a, b in merged_center:
            print(f"[{a:.4f},{b:.4f}] ", end="")
        print()

    return total_duration, elapsed


if __name__ == "__main__":
    import config_a
    solve_p1(config_a, "P1-A题")
