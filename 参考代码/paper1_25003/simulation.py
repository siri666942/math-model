"""
核心仿真引擎 - 运动学模型与遮蔽判定
Self-contained implementation for the paper's methodology.
Uses config module that is dynamically set to config_a or config_c.

Occlusion check method from the paper:
  For each target cylinder point PC, compute d0 = distance from PC to
  the line through PM (missile) and PO (smoke center).
  If max(d0) < effective_radius AND dot(PM-PC, PM-PO) >= 0,
  then all points are within cone -> occlusion is effective.
"""
import numpy as np

# Config is set dynamically by set_config()
_cfg = None


def set_config(cfg):
    """Set the active config module."""
    global _cfg
    _cfg = cfg


def _c():
    """Get current config."""
    return _cfg


# ============================================================
# Target keypoint generation
# ============================================================
def generate_target_keypoints(n_circle=720, n_layers=10):
    """
    生成目标圆柱体表面的关键点集

    包含:
    - 上底面圆周点
    - 下底面圆周点
    - 侧面多层圆周点

    Returns:
        keypoints: (N, 3) 关键点坐标数组
    """
    cfg = _c()
    keypoints = []

    # 上下底面圆周点
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = cfg.TARGET_RADIUS * np.cos(angle)
        y = cfg.TARGET_RADIUS * np.sin(angle)
        # 下底面
        keypoints.append([cfg.TARGET_CENTER[0] + x, cfg.TARGET_CENTER[1] + y, cfg.TARGET_CENTER[2]])
        # 上底面
        keypoints.append([cfg.TARGET_CENTER[0] + x, cfg.TARGET_CENTER[1] + y,
                          cfg.TARGET_CENTER[2] + cfg.TARGET_HEIGHT])

    # 侧面多层圆周点
    for layer in range(1, n_layers + 1):
        z_height = cfg.TARGET_CENTER[2] + cfg.TARGET_HEIGHT * layer / (n_layers + 1)
        for i in range(n_circle):
            angle = 2 * np.pi * i / n_circle
            x = cfg.TARGET_RADIUS * np.cos(angle)
            y = cfg.TARGET_RADIUS * np.sin(angle)
            keypoints.append([cfg.TARGET_CENTER[0] + x, cfg.TARGET_CENTER[1] + y, z_height])

    return np.array(keypoints)


# Cache for keypoints
_keypoint_cache = {}


def get_target_keypoints(n_circle=720, n_layers=10):
    """获取目标关键点集（缓存）"""
    key = (n_circle, n_layers)
    if key not in _keypoint_cache:
        _keypoint_cache[key] = generate_target_keypoints(n_circle, n_layers)
    return _keypoint_cache[key]


def clear_keypoint_cache():
    """Clear cache (needed when switching configs)."""
    global _keypoint_cache
    _keypoint_cache = {}


# ============================================================
# Kinematics
# ============================================================
def missile_position(t, missile_idx=0):
    """计算导弹在时刻t的位置"""
    cfg = _c()
    return cfg.MISSILES_INIT[missile_idx] + cfg.MISSILE_SPEED * cfg.MISSILES_DIR[missile_idx] * t


def missile_positions(t, n_missiles=3):
    """计算所有导弹在时刻t的位置 (t scalar or array)"""
    cfg = _c()
    t_arr = np.atleast_1d(t)
    M_pos = np.zeros((len(t_arr), n_missiles, 3))
    for k in range(n_missiles):
        M_pos[:, k, :] = cfg.MISSILES_INIT[k] + cfg.MISSILE_SPEED * cfg.MISSILES_DIR[k] * t_arr[:, None]
    if np.ndim(t) == 0:
        return M_pos[0]
    return M_pos


def drone_position(t, drone_init, theta, speed):
    """计算无人机在时刻t的位置"""
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    return drone_init + speed * direction * t


# ============================================================
# Occlusion check using paper's distance-to-line method
# ============================================================
def check_occlusion_paper_method(missile_pos, smoke_pos, target_keypoints):
    """
    Occlusion check using cone projection method.

    For each target point PC:
      - Check if all target points are within the cone subtended by the
        smoke cloud (center PO, radius EFFECTIVE_RADIUS) from the missile.
      - Cloud must be between missile and target.

    Cone half-angle alpha: sin(alpha) = EFFECTIVE_RADIUS / |missile-smoke|

    Returns True if target is effectively occluded.
    """
    cfg = _c()
    PM = missile_pos
    PO = smoke_pos

    vec_ms = PO - PM
    dist_ms = np.linalg.norm(vec_ms)

    if dist_ms < 1e-9:
        return False

    # Condition 1: missile inside cloud
    if dist_ms < cfg.EFFECTIVE_RADIUS:
        return True

    # Condition 2: cloud must be between missile and target
    dist_mt = np.linalg.norm(PM - cfg.TARGET_CENTER)
    dist_st = np.linalg.norm(PO - cfg.TARGET_CENTER)
    if dist_mt <= dist_st:
        return False

    # Cone check
    sin_alpha = cfg.EFFECTIVE_RADIUS / dist_ms
    if sin_alpha >= 1.0:
        return True
    cos_alpha = np.sqrt(1.0 - sin_alpha ** 2)

    for PC in target_keypoints:
        tar_axis = PC - PM
        norm_tar = np.linalg.norm(tar_axis)
        if norm_tar < 1e-12:
            continue
        cos_gamma = np.dot(vec_ms, tar_axis) / (dist_ms * norm_tar)
        if cos_alpha > cos_gamma:
            return False

    return True


def _check_occlusion_batch_vectorized(M_pos, smoke_pos, target_keypoints, chunk_size=None):
    """
    向量化遮蔽判据: 使用锥体投影方法 (cone method).

    遮蔽条件:
    1. 导弹在云团内部 (dist_ms < EFFECTIVE_RADIUS): 直接返回 True
    2. 否则: 检查云团在导弹和目标之间，且目标所有关键点在视锥内:
       - 视锥半顶角 alpha: sin(alpha) = EFFECTIVE_RADIUS / dist_ms
       - 对每个目标关键点, 检查 cos_gamma >= cos_alpha (即关键点在视锥内)

    This matches the existing py_solution implementation and is geometrically correct
    for line-of-sight occlusion by a spherical smoke cloud.

    Parameters:
        M_pos: (T, 3) 导弹位置序列
        smoke_pos: (T, 3) 云团位置序列
        target_keypoints: (K, 3) 目标关键点集

    Returns:
        (T,) 布尔数组
    """
    cfg = _c()
    T = M_pos.shape[0]
    K = target_keypoints.shape[0]

    if chunk_size is None:
        chunk_size = max(1, int(2e7 / max(K * 3, 1)))

    result = np.empty(T, dtype=bool)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        m = M_pos[start:end]          # (t_chunk, 3)
        s = smoke_pos[start:end]      # (t_chunk, 3)

        vec_ms = s - m                # (t_chunk, 3)
        dist_ms = np.linalg.norm(vec_ms, axis=1)  # (t_chunk,)

        # Condition 1: missile inside cloud
        inside_cloud = dist_ms < cfg.EFFECTIVE_RADIUS

        # Condition 2: cloud must be between missile and target
        dist_mt = np.linalg.norm(m - cfg.TARGET_CENTER, axis=1)  # (t_chunk,)
        dist_st = np.linalg.norm(s - cfg.TARGET_CENTER, axis=1)  # (t_chunk,)
        cloud_behind = dist_mt <= dist_st  # cloud behind or at target -> blocked

        # Cone check
        dist_ms_safe = np.maximum(dist_ms, 1e-9)
        sin_alpha = np.clip(cfg.EFFECTIVE_RADIUS / dist_ms_safe, 0.0, 1.0)
        full_cover = sin_alpha >= 1.0  # cloud envelops missile's field of view
        cos_alpha = np.sqrt(np.maximum(1.0 - sin_alpha ** 2, 0.0))

        # For each target point, check if within cone
        tar_axis = target_keypoints[None, :, :] - m[:, None, :]  # (t_chunk, K, 3)
        dot = np.einsum('ti,tki->tk', vec_ms, tar_axis)           # (t_chunk, K)
        norm_tar = np.linalg.norm(tar_axis, axis=2)               # (t_chunk, K)
        denom = dist_ms[:, None] * norm_tar
        denom_safe = np.where(denom < 1e-12, np.inf, denom)
        cos_gamma = dot / denom_safe                              # (t_chunk, K)
        all_in_cone = np.all(cos_alpha[:, None] <= cos_gamma, axis=1)  # (t_chunk,)

        cone_ok = full_cover | all_in_cone
        result[start:end] = inside_cloud | (~cloud_behind & cone_ok)

    return result


# ============================================================
# Single bomb coverage mask
# ============================================================
def _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                        missile_idx, target_keypoints, dt, t_total):
    """
    计算单枚烟幕弹在全局时间网格上、独自能否遮蔽指定导弹的布尔掩码。

    使用云团/导弹位置的解析式，在起爆~失效窗口内做向量化判据计算。
    """
    cfg = _c()
    n_steps = int(np.ceil(t_total / dt))
    mask = np.zeros(n_steps, dtype=bool)

    detonation_time = release_time + detonation_delay
    smoke_expire_time = detonation_time + cfg.EFFECTIVE_DURATION
    t_start = max(0.0, detonation_time)
    t_end = min(t_total, smoke_expire_time)
    if t_end <= t_start:
        return mask

    i_start = int(np.ceil(t_start / dt))
    i_end = min(n_steps, int(np.floor((t_end + 1e-9) / dt)) + 1)
    if i_end <= i_start:
        return mask

    ts = (i_start + np.arange(i_end - i_start)) * dt

    # Compute bomb detonation position
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    FY_at_release = drone_init + speed * direction * release_time
    bomb_x = FY_at_release[0] + speed * direction[0] * detonation_delay
    bomb_y = FY_at_release[1] + speed * direction[1] * detonation_delay
    bomb_h = FY_at_release[2] - 0.5 * cfg.G * detonation_delay ** 2

    # Smoke positions over time
    smoke_pos = np.empty((len(ts), 3))
    smoke_pos[:, 0] = bomb_x
    smoke_pos[:, 1] = bomb_y
    smoke_pos[:, 2] = bomb_h - cfg.SMOKE_SINK_SPEED * (ts - detonation_time)

    # Missile positions over time
    M_pos = cfg.MISSILES_INIT[missile_idx] + cfg.MISSILE_SPEED * \
        cfg.MISSILES_DIR[missile_idx] * ts[:, None]

    mask[i_start:i_end] = _check_occlusion_batch_vectorized(
        M_pos, smoke_pos, target_keypoints
    )
    return mask


# ============================================================
# Simulation functions
# ============================================================
def simulate_single_bomb(drone_init, theta, speed, release_time, detonation_delay,
                         missile_idx=0, target_keypoints=None, dt=None, t_total=None):
    """
    仿真单机单弹场景。

    Returns:
        total_effective_time: 总有效遮蔽时长 (s)
    """
    cfg = _c()
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    if dt is None:
        dt = cfg.DT
    if t_total is None:
        t_total = cfg.T_TOTAL

    mask = _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                               missile_idx, target_keypoints, dt, t_total)
    return mask.sum() * dt


def simulate_multi_bomb_single_drone(drone_init, theta, speed, release_times, detonation_delays,
                                     missile_indices, target_keypoints=None, dt=None, t_total=None):
    """
    仿真单机多弹场景（针对指定导弹）。

    Returns:
        total_effective_time: 总有效遮蔽时长（多枚弹的遮蔽区间取并集）
    """
    cfg = _c()
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    if dt is None:
        dt = cfg.DT
    if t_total is None:
        t_total = cfg.T_TOTAL

    n_steps = int(np.ceil(t_total / dt))
    any_covered = np.zeros(n_steps, dtype=bool)

    for i in range(len(release_times)):
        mask = _bomb_coverage_mask(drone_init, theta, speed, release_times[i],
                                   detonation_delays[i], missile_indices[i],
                                   target_keypoints, dt, t_total)
        any_covered |= mask

    return any_covered.sum() * dt


def simulate_multi_drone_multi_bomb(drone_params_list, dt=None, t_total=None):
    """
    仿真多机多弹多导弹场景。

    Parameters:
        drone_params_list: 每架无人机的参数字典列表
            [{
                'drone_init': np.array([x,y,z]),
                'theta': float,
                'speed': float,
                'release_times': np.array([t1, t2, t3]),
                'detonation_delays': np.array([d1, d2, d3]),
                'missile_indices': [k1, k2, k3],
            }, ...]

    Returns:
        total_effective_time: 总有效遮蔽时长
        per_missile_time: 每枚导弹的遮蔽时长
    """
    cfg = _c()
    if dt is None:
        dt = cfg.DT
    if t_total is None:
        t_total = cfg.T_TOTAL

    target_keypoints = get_target_keypoints(n_circle=360, n_layers=10)

    n_steps = int(np.ceil(t_total / dt))
    n_missiles = 3
    per_missile_mask = [np.zeros(n_steps, dtype=bool) for _ in range(n_missiles)]

    for p in drone_params_list:
        n_bombs = len(p['release_times'])
        for i in range(n_bombs):
            k = p['missile_indices'][i]
            mask = _bomb_coverage_mask(p['drone_init'], p['theta'], p['speed'],
                                       p['release_times'][i], p['detonation_delays'][i],
                                       k, target_keypoints, dt, t_total)
            per_missile_mask[k] |= mask

    per_missile_time = np.array([m.sum() * dt for m in per_missile_mask])
    total_effective_time = per_missile_time.sum()

    return total_effective_time, per_missile_time
