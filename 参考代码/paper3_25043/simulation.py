"""
simulation.py - Core simulation module with interval-based occlusion detection.
Paper's innovation: analytically find time windows where smoke occludes missile-target line.
Optimized with vectorized computation for performance.
"""
import numpy as np
from numpy.linalg import norm


# ============================================================
# Target keypoint generation (cylinder surface sampling)
# ============================================================
def generate_target_keypoints(center, radius, height,
                               n_circle=36, n_side_layers=5):
    """
    Generate keypoints on the surface of a cylinder target.
    Returns: (N, 3) array
    """
    points = []
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y, center[2] + height])  # top
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y, center[2]])  # bottom
    for layer in range(n_side_layers):
        z = center[2] + height * (layer + 1) / (n_side_layers + 1)
        for i in range(n_circle):
            angle = 2 * np.pi * i / n_circle
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y, z])
    return np.array(points)


# ============================================================
# Kinematics
# ============================================================
def missile_position_at_time(missile_start, missile_speed, target_center, t):
    """Missile flies from missile_start toward target_center at constant speed."""
    direction = target_center - missile_start
    dist = norm(direction)
    if dist < 1e-9:
        return missile_start.copy()
    direction = direction / dist
    d_traveled = missile_speed * t
    if d_traveled >= dist:
        return target_center.copy()
    return missile_start + direction * d_traveled


def drone_position_at_time(drone_start, speed, theta, t):
    """Drone flies at constant speed in direction theta (from X-axis)."""
    return drone_start + np.array([np.cos(theta), np.sin(theta), 0.0]) * speed * t


def smoke_center_at_time(detonation_point, t_det, t, config):
    """Smoke cloud center at time t (sinks at SMOKE_SINK_SPEED)."""
    dt = t - t_det
    center = detonation_point.copy()
    center[2] -= config.SMOKE_SINK_SPEED * dt
    return center


def compute_bomb_trajectory_general(drone_start, drone_speed, theta,
                                     release_time, detonation_delay, config):
    """
    Bomb trajectory: drone drops bomb at release_time, bomb detonates after detonation_delay.
    Bomb has drone's horizontal velocity, falls under gravity.
    Returns (detonation_point, detonation_time).
    """
    drop_pos = drone_position_at_time(drone_start, drone_speed, theta, release_time)
    vx = drone_speed * np.cos(theta)
    vy = drone_speed * np.sin(theta)
    det_point = np.array([
        drop_pos[0] + vx * detonation_delay,
        drop_pos[1] + vy * detonation_delay,
        drop_pos[2] - 0.5 * config.G * detonation_delay**2
    ])
    t_det = release_time + detonation_delay
    return det_point, t_det


# ============================================================
# Vectorized point-to-line distance for interval detection
# ============================================================
def _point_to_line_dist_vectorized(pt, line_start, line_end):
    """Vectorized distance from single point to line segment. Returns scalar."""
    line_vec = line_end - line_start
    pt_vec = pt - line_start
    line_len2 = np.dot(line_vec, line_vec)
    if line_len2 < 1e-12:
        return norm(pt - line_start)
    t_param = np.dot(pt_vec, line_vec) / line_len2
    t_param = np.clip(t_param, 0.0, 1.0)
    projection = line_start + t_param * line_vec
    return norm(pt - projection)


def find_occlusion_interval_single_bomb(
        detonation_point, t_det,
        missile_start, missile_speed, target_center,
        config, dt_sample=0.05):
    """
    Find time intervals where smoke sphere occludes the missile-target_center line.

    The missile flies toward target_center.
    Smoke is sphere of radius EFFECTIVE_RADIUS centered at detonation_point,
    sinking at SMOKE_SINK_SPEED.

    Returns: list of (t_start, t_end) intervals.
    """
    R = config.EFFECTIVE_RADIUS
    t_max = t_det + config.EFFECTIVE_DURATION

    n_samples = max(int((t_max - t_det) / dt_sample), 3)
    ts = np.linspace(t_det, t_max, n_samples)

    # Precompute missile positions
    direction = target_center - missile_start
    total_dist = norm(direction)
    direction_n = direction / total_dist

    distances = np.zeros(n_samples)
    missile_poses = np.zeros((n_samples, 3))
    smoke_poses = np.zeros((n_samples, 3))

    for i, t in enumerate(ts):
        smoke_poses[i] = detonation_point - np.array([0, 0, config.SMOKE_SINK_SPEED * (t - t_det)])
        traveled = missile_speed * t
        if traveled >= total_dist:
            missile_poses[i] = target_center.copy()
        else:
            missile_poses[i] = missile_start + direction_n * traveled

    # Vectorized distance: for each time, compute distance from smoke to line (missile->target)
    for i in range(n_samples):
        distances[i] = _point_to_line_dist_vectorized(
            smoke_poses[i], missile_poses[i], target_center
        )

    occluded = distances <= R

    # Find contiguous intervals
    intervals = []
    in_interval = False
    t_start = 0.0
    for i in range(n_samples):
        if occluded[i] and not in_interval:
            in_interval = True
            t_start = ts[max(i - 1, 0)]
        elif not occluded[i] and in_interval:
            in_interval = False
            intervals.append((t_start, ts[i]))
    if in_interval:
        intervals.append((t_start, ts[-1]))

    return intervals


def merge_intervals(intervals, tol=0.05):
    """Merge overlapping or near-adjacent intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for iv in sorted_iv[1:]:
        if iv[0] <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], iv[1])
        else:
            merged.append(list(iv))
    return [(a, b) for a, b in merged]


def compute_total_occlusion_duration(
        detonation_points, t_dets,
        missile_start, missile_speed, target_center,
        config):
    """
    Total occlusion duration for target center, given multiple bombs.
    """
    all_intervals = []
    for det_pt, t_det in zip(detonation_points, t_dets):
        intervals = find_occlusion_interval_single_bomb(
            det_pt, t_det, missile_start, missile_speed, target_center, config
        )
        all_intervals.extend(intervals)
    merged = merge_intervals(all_intervals)
    return sum(b - a for a, b in merged)


# ============================================================
# Objective function for optimization
# ============================================================
def objective_coverage_all_keypoints(
        detonation_points, t_dets,
        missiles_init, missile_speed,
        keypoints,
        config,
        use_min=True):
    """
    Compute coverage score across all missile-keypoint pairs.
    For each keypoint, use the best missile.
    use_min=True: worst-case keypoint coverage
    use_min=False: mean keypoint coverage
    """
    n_kp = len(keypoints)
    n_missiles = len(missiles_init)
    ratios = np.zeros(n_kp)

    for j in range(n_kp):
        best_ratio = 0.0
        for m_idx in range(n_missiles):
            # Missile flies toward TARGET_CENTER, check occlusion to keypoint j
            duration = compute_total_occlusion_duration(
                detonation_points, t_dets,
                missiles_init[m_idx], missile_speed,
                keypoints[j], config  # Check against this keypoint
            )
            missile_flight_time = norm(keypoints[j] - missiles_init[m_idx]) / missile_speed
            ratio = min(duration / missile_flight_time, 1.0) if missile_flight_time > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
        ratios[j] = best_ratio

    if use_min:
        return np.min(ratios)
    else:
        return np.mean(ratios)


# ============================================================
# Problem 1 specific: interval intersection across keypoints
# ============================================================
def p1_find_occlusion_intervals(
        detonation_point, t_det,
        missile_start, missile_speed,
        keypoints, config):
    """
    P1: Find the INTERSECTION of occlusion intervals across all keypoints.
    This is strict: a period is only counted if ALL keypoints are occluded.
    """
    all_keypoint_intervals = []
    for kp in keypoints:
        intervals = find_occlusion_interval_single_bomb(
            detonation_point, t_det,
            missile_start, missile_speed, kp, config
        )
        merged = merge_intervals(intervals)
        if not merged:
            return [], 0.0
        all_keypoint_intervals.append(merged)

    # Intersect all
    common = all_keypoint_intervals[0]
    for kp_intervals in all_keypoint_intervals[1:]:
        new_common = []
        for (a1, b1) in common:
            for (a2, b2) in kp_intervals:
                lo = max(a1, a2)
                hi = min(b1, b2)
                if lo < hi:
                    new_common.append((lo, hi))
        common = merge_intervals(new_common)
        if not common:
            break

    total_duration = sum(b - a for a, b in common)
    return common, total_duration


# ============================================================
# Utility: compute optimal theta for smoke positioning
# ============================================================
def compute_smoke_proximity_score(
        detonation_point, t_det,
        missile_start, missile_speed, target_center,
        config, R_multiplier=1.0):
    """
    Compute a smooth proximity score: for each time step, compute
    distance from smoke to missile-target line, and accumulate
    a sigmoid-smoothed occlusion contribution.

    Returns: score in [0, 1], higher is better (more occlusion).
    This is a continuous approximation useful as PSO objective.
    """
    R = config.EFFECTIVE_RADIUS * R_multiplier
    t_max = t_det + config.EFFECTIVE_DURATION
    dt = 0.1  # coarse sampling for speed
    n_steps = max(int((t_max - t_det) / dt), 2)

    direction = target_center - missile_start
    total_dist = np.linalg.norm(direction)
    direction_n = direction / total_dist
    mft = total_dist / missile_speed

    score = 0.0
    for step in range(n_steps):
        t = t_det + step * dt
        smoke = smoke_center_at_time(detonation_point, t_det, t, config)
        traveled = missile_speed * t
        if traveled >= total_dist:
            missile = target_center
        else:
            missile = missile_start + direction_n * traveled
        dist = _point_to_line_dist_vectorized(smoke, missile, target_center)
        # Sigmoid: 1.0 when dist << R, 0.0 when dist >> R
        occlusion = 1.0 / (1.0 + np.exp((dist - R) / (R * 0.2)))
        score += occlusion * dt

    if mft > 0:
        score = min(score / mft, 1.0)
    return score


def compute_optimal_theta_to_line(drone_start, drone_speed, t_rel, t_delay,
                                   missile_start, target_center, config):
    """
    Compute drone heading theta so the bomb detonation point is ON the
    missile-target line (x-y projection). Returns theta in radians.

    Uses quadratic solution to place (x_det, y_det) on 2D line missile->target.
    """
    x0, y0 = drone_start[0], drone_start[1]
    xm, ym = missile_start[0], missile_start[1]
    xt, yt = target_center[0], target_center[1]
    D = drone_speed * (t_rel + t_delay)

    dx_mt = xt - xm
    dy_mt = yt - ym

    # Coefficients: A*cos - B*sin = C
    A = D * dy_mt
    B = D * dx_mt
    C = (y0 - ym) * dx_mt - (x0 - xm) * dy_mt

    # Quadratic: (A^2 + B^2)*c^2 - 2*A*C*c + (C^2 - B^2) = 0
    a_coeff = A**2 + B**2
    b_coeff = -2 * A * C
    c_coeff = C**2 - B**2

    disc = b_coeff**2 - 4 * a_coeff * c_coeff
    if disc < 0:
        return np.arctan2(yt - y0, xt - x0)

    sqrt_disc = np.sqrt(disc)
    candidates = []
    for sign in [1, -1]:
        c = (-b_coeff + sign * sqrt_disc) / (2 * a_coeff)
        if abs(c) <= 1.0:
            s = (A * c - C) / B
            if abs(s) <= 1.0 and abs(c**2 + s**2 - 1) < 1e-6:
                theta = np.arctan2(s, c)
                candidates.append(theta)

    # Pick theta that puts detonation on the correct side:
    # prefer cos(theta) < 0 (westward) when target is west of drone
    # and detonation between missile and target (x-wise)
    xm_, xt_ = missile_start[0], target_center[0]

    # First: try theta where cos < 0 (flying toward decreasing x, i.e., toward target x=0)
    if x0 > xt_:  # drone is east of target, fly westward
        for theta in candidates:
            if np.cos(theta) < 0:
                x_det = x0 + np.cos(theta) * D
                if xm_ > xt_:
                    if xt_ < x_det < xm_:
                        return theta
                    elif x_det < xt_:  # detonation past the target
                        # Check if this is still acceptable (smoke between missile and target)
                        pass
                else:
                    if xm_ < x_det < xt_:
                        return theta
        # No westward candidate with det between missile and target
        # Pick the westward candidate closest to the right range
        for theta in candidates:
            if np.cos(theta) < 0:
                return theta

    # Then: try theta where cos > 0 (eastward)
    for theta in candidates:
        x_det = x0 + np.cos(theta) * D
        if xm_ > xt_:
            if xt_ < x_det < xm_:
                return theta
        else:
            if xm_ < x_det < xt_:
                return theta

    # Fallback
    return candidates[0] if candidates else np.arctan2(yt - y0, xt - x0)
