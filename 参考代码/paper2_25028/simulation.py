"""
核心仿真引擎 - 基于 cumcm25028 论文方法
锥体投影法遮蔽判定 + 550关键点目标离散化
"""
import numpy as np

# 运行时由 solve_p*.py 或 run_all.py 注入
G = None
TARGET_CENTER = None
TARGET_RADIUS = None
TARGET_HEIGHT = None
MISSILE_SPEED = None
SMOKE_SINK_SPEED = None
EFFECTIVE_RADIUS = None
EFFECTIVE_DURATION = None
DT = None
T_TOTAL = None
N_CIRCLE_POINTS = None
N_SIDE_LAYERS = None
MISSILES_INIT = None
MISSILES_DIR = None
DRONES_INIT = None


def set_config(config_module):
    """注入配置模块的所有参数到当前模块全局变量"""
    global G, TARGET_CENTER, TARGET_RADIUS, TARGET_HEIGHT
    global MISSILE_SPEED, SMOKE_SINK_SPEED, EFFECTIVE_RADIUS, EFFECTIVE_DURATION
    global DT, T_TOTAL, N_CIRCLE_POINTS, N_SIDE_LAYERS
    global MISSILES_INIT, MISSILES_DIR, DRONES_INIT

    cfg = config_module
    G = cfg.G
    TARGET_CENTER = cfg.TARGET_CENTER.copy()
    TARGET_RADIUS = cfg.TARGET_RADIUS
    TARGET_HEIGHT = cfg.TARGET_HEIGHT
    MISSILE_SPEED = cfg.MISSILE_SPEED
    SMOKE_SINK_SPEED = cfg.SMOKE_SINK_SPEED
    EFFECTIVE_RADIUS = cfg.EFFECTIVE_RADIUS
    EFFECTIVE_DURATION = cfg.EFFECTIVE_DURATION
    DT = cfg.DT
    T_TOTAL = cfg.T_TOTAL
    N_CIRCLE_POINTS = cfg.N_CIRCLE_POINTS
    N_SIDE_LAYERS = cfg.N_SIDE_LAYERS
    MISSILES_INIT = cfg.MISSILES_INIT.copy()
    MISSILES_DIR = cfg.MISSILES_DIR.copy()
    DRONES_INIT = cfg.DRONES_INIT.copy()
    _clear_keypoints_cache()


# ---- 目标关键点生成 ----

def generate_target_keypoints(n_circle, n_layers):
    """生成目标圆柱体表面关键点: n_circle * n_layers 点"""
    keypoints = np.empty((n_circle * n_layers, 3))
    idx = 0
    for layer in range(n_layers):
        if n_layers == 1:
            z = TARGET_CENTER[2] + TARGET_HEIGHT / 2
        elif layer == 0:
            z = TARGET_CENTER[2]
        elif layer == n_layers - 1:
            z = TARGET_CENTER[2] + TARGET_HEIGHT
        else:
            z = TARGET_CENTER[2] + TARGET_HEIGHT * layer / (n_layers - 1)
        for i in range(n_circle):
            angle = 2.0 * np.pi * i / n_circle
            x = TARGET_CENTER[0] + TARGET_RADIUS * np.cos(angle)
            y = TARGET_CENTER[1] + TARGET_RADIUS * np.sin(angle)
            keypoints[idx] = [x, y, z]
            idx += 1
    return keypoints


_keypoints_cache = None
_cache_n_circle = None
_cache_n_layers = None


def _clear_keypoints_cache():
    global _keypoints_cache, _cache_n_circle, _cache_n_layers
    _keypoints_cache = None
    _cache_n_circle = None
    _cache_n_layers = None


def get_target_keypoints(n_circle=None, n_layers=None):
    global _keypoints_cache, _cache_n_circle, _cache_n_layers
    if n_circle is None:
        n_circle = N_CIRCLE_POINTS
    if n_layers is None:
        n_layers = N_SIDE_LAYERS
    if (_keypoints_cache is not None and n_circle == _cache_n_circle and n_layers == _cache_n_layers):
        return _keypoints_cache
    _keypoints_cache = generate_target_keypoints(n_circle, n_layers)
    _cache_n_circle = n_circle
    _cache_n_layers = n_layers
    return _keypoints_cache


# ---- 运动学模型 ----

def missile_position(t, missile_idx=0):
    return MISSILES_INIT[missile_idx] + MISSILE_SPEED * MISSILES_DIR[missile_idx] * t


def drone_position(t, drone_init, theta, speed):
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    return drone_init + speed * direction * t


# ---- 遮蔽判定 (锥体投影法) ----

def _check_occlusion_cone(missile_pos, smoke_pos, target_keypoints):
    """单时刻锥体遮蔽判定"""
    vec_ms = smoke_pos - missile_pos
    dist_ms = np.linalg.norm(vec_ms)
    if dist_ms < EFFECTIVE_RADIUS:
        return True, 1.0

    sin_alpha = EFFECTIVE_RADIUS / dist_ms
    if sin_alpha >= 1.0:
        return True, 1.0
    cos_alpha = np.sqrt(1.0 - sin_alpha ** 2)

    vec_mk = target_keypoints - missile_pos
    dist_mk = np.linalg.norm(vec_mk, axis=1)
    dots = np.dot(vec_mk, vec_ms)
    cos_gamma = dots / (dist_ms * np.maximum(dist_mk, 1e-9))

    # 计算覆盖比例: cos_gamma >= cos_alpha 的点占比
    covered = cos_gamma >= cos_alpha
    coverage_ratio = covered.sum() / len(target_keypoints)

    all_in_cone = np.all(covered)
    if np.all(dots <= 0):
        return False, coverage_ratio

    return all_in_cone, coverage_ratio


def _check_occlusion_cone_vectorized(M_pos, smoke_pos, target_keypoints):
    """向量化批量锥体遮蔽判定"""
    T = M_pos.shape[0]
    K = target_keypoints.shape[0]

    vec_ms = smoke_pos - M_pos
    dist_ms = np.linalg.norm(vec_ms, axis=1)
    inside_cloud = dist_ms < EFFECTIVE_RADIUS

    dist_ms_safe = np.maximum(dist_ms, 1e-9)
    sin_alpha = np.clip(EFFECTIVE_RADIUS / dist_ms_safe, 0.0, 1.0)
    full_cover = sin_alpha >= 1.0
    cos_alpha = np.sqrt(np.maximum(1.0 - sin_alpha ** 2, 0.0))

    vec_mk = target_keypoints[None, :, :] - M_pos[:, None, :]
    dist_mk = np.linalg.norm(vec_mk, axis=2)
    dist_mk_safe = np.maximum(dist_mk, 1e-9)

    dots = np.sum(vec_mk * vec_ms[:, None, :], axis=2)
    cos_gamma = dots / (dist_ms_safe[:, None] * dist_mk_safe)

    covered = cos_gamma >= cos_alpha[:, None]
    all_in_cone = np.all(covered, axis=1)
    all_behind = np.all(dots <= 0, axis=1)

    result = np.zeros(T, dtype=bool)
    result[inside_cloud | full_cover] = True
    valid = ~inside_cloud & ~full_cover
    result[valid] = all_in_cone[valid] & ~all_behind[valid]

    return result


def _check_occlusion_soft_vectorized(M_pos, smoke_pos, target_keypoints):
    """
    向量化批量软遮蔽判定 - 返回每个时刻的覆盖比例 (用于GA引导)
    覆盖比例 = 在锥体内的关键点数 / 总关键点数
    """
    T = M_pos.shape[0]

    vec_ms = smoke_pos - M_pos
    dist_ms = np.linalg.norm(vec_ms, axis=1)
    inside_cloud = dist_ms < EFFECTIVE_RADIUS

    dist_ms_safe = np.maximum(dist_ms, 1e-9)
    sin_alpha = np.clip(EFFECTIVE_RADIUS / dist_ms_safe, 0.0, 1.0)
    full_cover = sin_alpha >= 1.0
    cos_alpha = np.sqrt(np.maximum(1.0 - sin_alpha ** 2, 0.0))

    vec_mk = target_keypoints[None, :, :] - M_pos[:, None, :]
    dist_mk = np.linalg.norm(vec_mk, axis=2)
    dist_mk_safe = np.maximum(dist_mk, 1e-9)

    dots = np.sum(vec_mk * vec_ms[:, None, :], axis=2)
    cos_gamma = dots / (dist_ms_safe[:, None] * dist_mk_safe)

    covered = cos_gamma >= cos_alpha[:, None]
    coverage_ratio = covered.mean(axis=1)  # (T,) 平均覆盖比例

    # 云团内部或全覆盖->覆盖率1.0
    coverage_ratio[inside_cloud | full_cover] = 1.0

    # 烟幕在导弹后方->覆盖率0
    all_behind = np.all(dots <= 0, axis=1)
    coverage_ratio[all_behind] = 0.0

    return coverage_ratio


# ---- 烟幕弹遮蔽掩码计算 ----

def _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                        missile_idx, target_keypoints, dt, t_total):
    """单枚烟幕弹的遮蔽掩码"""
    n_steps = int(np.ceil(t_total / dt))
    mask = np.zeros(n_steps, dtype=bool)

    detonation_time = release_time + detonation_delay
    smoke_expire_time = detonation_time + EFFECTIVE_DURATION
    t_start = max(0.0, detonation_time)
    t_end = min(t_total, smoke_expire_time)
    if t_end <= t_start + 1e-9:
        return mask

    i_start = int(np.ceil(t_start / dt))
    i_end = min(n_steps, int(np.floor(t_end / dt)) + 1)
    if i_end <= i_start:
        return mask

    ts = (i_start + np.arange(i_end - i_start)) * dt

    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    FY_at_release = drone_init + speed * direction * release_time

    bomb_x = FY_at_release[0] + speed * direction[0] * detonation_delay
    bomb_y = FY_at_release[1] + speed * direction[1] * detonation_delay
    bomb_h = FY_at_release[2] - 0.5 * G * detonation_delay ** 2

    smoke_pos = np.empty((len(ts), 3))
    smoke_pos[:, 0] = bomb_x
    smoke_pos[:, 1] = bomb_y
    smoke_pos[:, 2] = bomb_h - SMOKE_SINK_SPEED * (ts - detonation_time)

    M_pos = MISSILES_INIT[missile_idx] + MISSILE_SPEED * MISSILES_DIR[missile_idx] * ts[:, None]

    mask[i_start:i_end] = _check_occlusion_cone_vectorized(M_pos, smoke_pos, target_keypoints)
    return mask


def _bomb_soft_score(drone_init, theta, speed, release_time, detonation_delay,
                     missile_idx, target_keypoints, dt, t_total):
    """单枚烟幕弹的软遮蔽分数 (最大覆盖比例)"""
    detonation_time = release_time + detonation_delay
    smoke_expire_time = detonation_time + EFFECTIVE_DURATION
    t_start = max(0.0, detonation_time)
    t_end = min(t_total, smoke_expire_time)
    if t_end <= t_start + 1e-9:
        return 0.0

    i_start = int(np.ceil(t_start / dt))
    i_end = min(int(np.ceil(t_total / dt)), int(np.floor(t_end / dt)) + 1)
    if i_end <= i_start:
        return 0.0

    ts = (i_start + np.arange(i_end - i_start)) * dt
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    FY_at_release = drone_init + speed * direction * release_time

    bomb_x = FY_at_release[0] + speed * direction[0] * detonation_delay
    bomb_y = FY_at_release[1] + speed * direction[1] * detonation_delay
    bomb_h = FY_at_release[2] - 0.5 * G * detonation_delay ** 2

    smoke_pos = np.empty((len(ts), 3))
    smoke_pos[:, 0] = bomb_x
    smoke_pos[:, 1] = bomb_y
    smoke_pos[:, 2] = bomb_h - SMOKE_SINK_SPEED * (ts - detonation_time)

    M_pos = MISSILES_INIT[missile_idx] + MISSILE_SPEED * MISSILES_DIR[missile_idx] * ts[:, None]

    coverage = _check_occlusion_soft_vectorized(M_pos, smoke_pos, target_keypoints)
    return float(np.max(coverage))


# ---- 顶层仿真接口 ----

def simulate_single_bomb(drone_init, theta, speed, release_time, detonation_delay,
                         missile_idx=0, target_keypoints=None, dt=None, t_total=None):
    if dt is None:
        dt = DT
    if t_total is None:
        t_total = T_TOTAL
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    mask = _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                               missile_idx, target_keypoints, dt, t_total)
    return mask.sum() * dt


def soft_score_single_bomb(drone_init, theta, speed, release_time, detonation_delay,
                           missile_idx=0, target_keypoints=None, dt=None, t_total=None):
    """返回软遮蔽分数 (0-1) 表示最佳时刻的覆盖比例"""
    if dt is None:
        dt = DT
    if t_total is None:
        t_total = T_TOTAL
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    return _bomb_soft_score(drone_init, theta, speed, release_time, detonation_delay,
                            missile_idx, target_keypoints, dt, t_total)


def simulate_multi_bomb_single_drone(drone_init, theta, speed, release_times, detonation_delays,
                                     missile_indices, target_keypoints=None, dt=None, t_total=None):
    if dt is None:
        dt = DT
    if t_total is None:
        t_total = T_TOTAL
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    n_steps = int(np.ceil(t_total / dt))
    any_covered = np.zeros(n_steps, dtype=bool)
    for i in range(len(release_times)):
        mask = _bomb_coverage_mask(drone_init, theta, speed, release_times[i],
                                   detonation_delays[i], missile_indices[i],
                                   target_keypoints, dt, t_total)
        any_covered |= mask
    return any_covered.sum() * dt


def simulate_multi_drone_multi_bomb(drone_params_list, dt=None, t_total=None):
    if dt is None:
        dt = DT
    if t_total is None:
        t_total = T_TOTAL
    target_keypoints = get_target_keypoints()
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
    return per_missile_time.sum(), per_missile_time
