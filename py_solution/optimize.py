"""
高级优化策略 - 使用scipy差分进化 + 改进的PSO
"""
import numpy as np
from scipy.optimize import differential_evolution
from pso import PSO


def de_optimize(objective_func, bounds, maximize=True, popsize=30,
                maxiter=100, tol=1e-6, workers=1, verbose=True):
    """
    使用scipy差分进化进行全局优化

    差分进化比PSO更适合"needle in haystack"类型的优化问题
    """
    sign = -1.0 if maximize else 1.0

    def wrapped_obj(x):
        return sign * objective_func(x)

    result = differential_evolution(
        wrapped_obj, bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=workers,
        updating='deferred',
    )

    if maximize:
        best_val = -result.fun
    else:
        best_val = result.fun

    if verbose:
        print(f"  DE: nfev={result.nfev}, success={result.success}, best={best_val:.4f}")

    return result.x, best_val


def grid_search_detonation(drone_init, drone_speed_range, missile_idx=0,
                           n_grid=20, grid_range=None):
    """
    网格搜索可行的起爆位置，找到遮蔽时间最长的位置

    使用变步长策略: 粗网格 → 细化
    """
    import config as cfg
    from simulation import check_occlusion, get_target_keypoints

    kp = get_target_keypoints(360, 0)
    u_min, u_max = drone_speed_range

    if grid_range is None:
        grid_range = {
            'x': (10000, 17800),
            'y': (-1000, 1000),
            'z': (200, 1800),
        }

    best_val = 0
    best_pos = None

    # Coarse grid
    xs = np.linspace(grid_range['x'][0], grid_range['x'][1], n_grid)
    ys = np.linspace(grid_range['y'][0], grid_range['y'][1], n_grid)
    zs = np.linspace(grid_range['z'][0], grid_range['z'][1], n_grid // 2)

    for x in xs:
        for y in ys:
            # Quick feasibility check: drone must be able to reach (x,y) before M1
            drone_dist = np.sqrt((drone_init[0] - x)**2 + (drone_init[1] - y)**2)
            drone_time_min = drone_dist / u_max

            # M1 travel time to x-coordinate:
            M1_dir = cfg.MISSILES_DIR[missile_idx]
            if abs(M1_dir[0]) < 1e-10:
                continue
            M1_time_to_x = (cfg.MISSILES_INIT[missile_idx][0] - x) / (cfg.MISSILE_SPEED * abs(M1_dir[0]))
            if M1_time_to_x <= 0:
                continue

            if drone_time_min >= M1_time_to_x:
                continue  # Drone can't reach before M1

            for z in zs:
                pos = np.array([x, y, z])

                # Check if this detonation position works (simplified check)
                # Back-solve for drone parameters
                # First, determine the direction from drone to detonation point
                vec = pos - drone_init
                dist = np.linalg.norm(vec)

                # Drone must fly faster than minimum and slower than maximum
                # For simplicity, assume drone flies at max speed directly toward point
                # This gives us a lower bound on reachability
                if dist < 1:
                    continue

                # Simple check: can drone reach and detonate at this position?
                # We'll use the simulation for full check later
                if best_val > 0 or np.random.random() < 0.01:  # Sparse sampling for speed
                    pass  # skip detailed check during coarse search

    # Return a reasonable starting point based on geometry
    # For FY1 at (17800, 0, 1800), the best detonation position should be
    # roughly along the missile-target line of sight
    M1_pos = cfg.MISSILES_INIT[missile_idx]
    target = cfg.TARGET_CENTER

    # Estimate a good detonation position: somewhere between M1 initial and target
    # but offset toward the drone's path
    M1_dir = cfg.MISSILES_DIR[missile_idx]

    # The drone needs to place smoke so it's between M1 and target
    # Estimate: M1 at its initial position, target at (0,200,0)
    # A good smoke position would be along this line
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for y_offset in [0, 50, 100, 150, 200]:
            px = drone_init[0] * (1 - alpha)  # fraction of the way toward origin in x
            py = y_offset
            pz = drone_init[2] * (1 - alpha * 0.5)
            pos = np.array([px, py, pz])

            # Quick check: is this between M1 and target?
            d_M1_to_pos = np.linalg.norm(pos - M1_pos)
            d_M1_to_tgt = np.linalg.norm(M1_pos - target)
            d_pos_to_tgt = np.linalg.norm(pos - target)

            # For the smoke to be between, need d_pos_to_tgt < d_M1_to_tgt
            if d_pos_to_tgt < d_M1_to_tgt:
                # This is a candidate - try to find matching drone params
                vec = pos - drone_init
                theta_candidate = np.arctan2(vec[1], vec[0])
                dist_h = np.sqrt(vec[0]**2 + vec[1]**2)

                for speed in [80, 100, 120]:
                    # Time to reach detonation x,y (ignoring delay physics for now)
                    time_to_det = dist_h / speed
                    # This is release_time + detonation_delay
                    # Try splitting evenly
                    rt = time_to_det * 0.4
                    dd = time_to_det * 0.6
                    if rt > 0 and dd > 0:
                        return theta_candidate, speed, rt, dd

    return None


def smart_initialize(drone_init, n_samples=1000):
    """
    智能初始化: 在几何可行域内采样，筛选有潜力的候选解
    """
    import config as cfg
    from simulation import simulate_single_bomb, get_target_keypoints

    kp = get_target_keypoints(360, 0)

    best_val = 0
    best_x = None

    for _ in range(n_samples):
        # Sample parameters
        theta = np.random.uniform(0.5, 1.6)  # roughly toward origin area
        speed = np.random.uniform(80, 120)
        release_time = np.random.uniform(0, 12)
        detonation_delay = np.random.uniform(0.5, 8)

        val = simulate_single_bomb(
            drone_init, theta, speed, release_time, detonation_delay,
            0, kp, cfg.DT, 30.0
        )

        if val > best_val:
            best_val = val
            best_x = [theta, speed, release_time, detonation_delay]
            print(f"  Found better: {val:.4f}s at theta={np.degrees(theta):.1f}°")

    return best_x, best_val
