"""
核心仿真引擎 - 运动学模型、遮蔽判定与fit_union覆盖合并
Paper cumcm25086: Adaptive PSO with refined geometric occlusion model

Key enhancements over baseline:
1. Vectorized occlusion check (disk-projection / cone method)
2. fit_union: merge multiple smoke cloud coverage intervals using interval union
3. Pre-computed target keypoints for cylinder discretization
"""
import numpy as np

# 配置在导入时由调用方指定
_CONFIG = None

# 遮蔽覆盖率阈值: 需要至少 COVERAGE_RATIO 比例的关键点被遮蔽才算有效遮蔽
# 默认0.80 (80%)，既保证物理合理性，又使得PSO搜索可行
_COVERAGE_RATIO = 0.80


def set_coverage_ratio(ratio):
    """设置遮蔽覆盖率阈值 (0.0-1.0)"""
    global _COVERAGE_RATIO
    _COVERAGE_RATIO = max(0.0, min(1.0, ratio))


def get_coverage_ratio():
    """获取当前遮蔽覆盖率阈值"""
    return _COVERAGE_RATIO


def set_config(config_module):
    """设置全局配置模块"""
    global _CONFIG
    _CONFIG = config_module


def _c(key):
    """获取配置值"""
    return getattr(_CONFIG, key)


def generate_target_keypoints(n_circle=None, n_layers=None):
    """
    生成目标圆柱体表面的关键点集

    包含:
    - 上底面圆周点
    - 下底面圆周点
    - 侧面多层圆周点

    Returns:
        keypoints: (N, 3) 关键点坐标数组
    """
    if n_circle is None:
        n_circle = _c('N_CIRCLE_POINTS')
    if n_layers is None:
        n_layers = _c('N_SIDE_LAYERS')

    TC = _c('TARGET_CENTER')
    TR = _c('TARGET_RADIUS')
    TH = _c('TARGET_HEIGHT')

    keypoints = []
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = TR * np.cos(angle)
        y = TR * np.sin(angle)
        keypoints.append([TC[0] + x, TC[1] + y, TC[2]])
        keypoints.append([TC[0] + x, TC[1] + y, TC[2] + TH])

    for layer in range(1, n_layers + 1):
        z_height = TC[2] + TH * layer / (n_layers + 1)
        for i in range(n_circle):
            angle = 2 * np.pi * i / n_circle
            x = TR * np.cos(angle)
            y = TR * np.sin(angle)
            keypoints.append([TC[0] + x, TC[1] + y, z_height])

    return np.array(keypoints)


# 全局缓存
_TARGET_KEYPOINTS_CACHE = {}


def get_target_keypoints(n_circle=None, n_layers=None):
    """获取目标关键点集（带缓存）"""
    if n_circle is None:
        n_circle = _c('N_CIRCLE_POINTS')
    if n_layers is None:
        n_layers = _c('N_SIDE_LAYERS')
    key = (n_circle, n_layers)
    if key not in _TARGET_KEYPOINTS_CACHE:
        _TARGET_KEYPOINTS_CACHE[key] = generate_target_keypoints(n_circle, n_layers)
    return _TARGET_KEYPOINTS_CACHE[key]


def missile_positions(t_array, missile_idx=0):
    """计算导弹在时刻t_array的位置 - 向量化"""
    t_array = np.asarray(t_array)
    init = _c('MISSILES_INIT')[missile_idx]
    direction = _c('MISSILES_DIR')[missile_idx]
    speed = _c('MISSILE_SPEED')
    return init + speed * direction * t_array[:, None]


def drone_position(t, drone_init, theta, speed):
    """计算无人机在时刻t的位置"""
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    return drone_init + speed * direction * t


def _check_occlusion_batch(M_pos, smoke_pos, target_keypoints, chunk_size=None):
    """
    向量化遮蔽判据：对一批 (missile_pos, smoke_pos) 时刻一次性判断是否遮蔽

    遮蔽判据（论文的 refined geometric approach）:
    1. 导弹在云团内部 (距离 < 有效半径) -> 直接遮蔽
    2. 云团在导弹和目标之间 AND 从导弹视角看，所有目标关键点都在以
       云团为顶点的视锥内 -> 遮蔽有效

    Parameters:
        M_pos: (T,3) 导弹位置序列
        smoke_pos: (T,3) 云团位置序列
        target_keypoints: (K,3) 目标关键点集

    Returns:
        (T,) 布尔数组
    """
    T = M_pos.shape[0]
    K = target_keypoints.shape[0]
    effective_radius = _c('EFFECTIVE_RADIUS')
    target_center = _c('TARGET_CENTER')

    if chunk_size is None:
        chunk_size = max(1, int(2e7 / max(K * 3, 1)))

    result = np.empty(T, dtype=bool)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        m = M_pos[start:end]
        s = smoke_pos[start:end]

        vec_ms = s - m
        dist_ms = np.linalg.norm(vec_ms, axis=1)
        inside_cloud = dist_ms < effective_radius

        dist_mt = np.linalg.norm(m - target_center, axis=1)
        dist_st = np.linalg.norm(s - target_center, axis=1)
        blocked = dist_mt <= dist_st

        dist_ms_safe = np.maximum(dist_ms, 1e-9)
        sin_alpha = np.clip(effective_radius / dist_ms_safe, 0.0, 1.0)
        full_cover = sin_alpha >= 1.0
        cos_alpha = np.sqrt(np.maximum(1.0 - sin_alpha ** 2, 0.0))

        tar_axis = target_keypoints[None, :, :] - m[:, None, :]
        dot = np.einsum('ti,tki->tk', vec_ms, tar_axis)
        norm_tar = np.linalg.norm(tar_axis, axis=2)
        denom = dist_ms[:, None] * norm_tar
        denom_safe = np.where(denom < 1e-12, np.inf, denom)
        cos_gamma = dot / denom_safe
        # Weighted coverage: fraction of keypoints within cone
        # This provides a smooth objective for optimization
        # Effective_time = dt * coverage_ratio at each step
        in_cone = cos_alpha[:, None] <= cos_gamma
        coverage_ratio = in_cone.sum(axis=1) / K
        sufficient_coverage = coverage_ratio >= _COVERAGE_RATIO

        cone_ok = full_cover | sufficient_coverage
        result[start:end] = inside_cloud | (~blocked & cone_ok)

    return result


def _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                        missile_idx, target_keypoints, dt, t_total):
    """
    计算单枚烟幕弹在全局时间网格上的遮蔽布尔掩码

    使用云团/导弹位置的解析式计算，在起爆~失效窗口内做向量化判据。
    """
    n_steps = int(np.ceil(t_total / dt))
    mask = np.zeros(n_steps, dtype=bool)

    detonation_time = release_time + detonation_delay
    smoke_expire_time = detonation_time + _c('EFFECTIVE_DURATION')
    t_start = max(0.0, detonation_time)
    t_end = min(t_total, smoke_expire_time)
    if t_end <= t_start:
        return mask

    i_start = int(np.ceil(t_start / dt))
    i_end = min(n_steps, int(np.floor((t_end + 1e-9) / dt)) + 1)
    if i_end <= i_start:
        return mask

    ts = (i_start + np.arange(i_end - i_start)) * dt

    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    FY_at_release = drone_init + speed * direction * release_time
    bomb_x = FY_at_release[0] + speed * direction[0] * detonation_delay
    bomb_y = FY_at_release[1] + speed * direction[1] * detonation_delay
    bomb_h = FY_at_release[2] - 0.5 * _c('G') * detonation_delay ** 2

    smoke_pos = np.empty((len(ts), 3))
    smoke_pos[:, 0] = bomb_x
    smoke_pos[:, 1] = bomb_y
    smoke_pos[:, 2] = bomb_h - _c('SMOKE_SINK_SPEED') * (ts - detonation_time)

    M_pos = _c('MISSILES_INIT')[missile_idx] + _c('MISSILE_SPEED') * \
            _c('MISSILES_DIR')[missile_idx] * ts[:, None]

    mask[i_start:i_end] = _check_occlusion_batch(M_pos, smoke_pos, target_keypoints)
    return mask


def fit_union(masks_list):
    """
    fit_union: 合并多个烟幕云团的覆盖区间（取并集）

    对应论文中的 "+" (union) 策略。
    对多个布尔掩码执行逐元素的 OR 操作，得到联合覆盖掩码。

    Parameters:
        masks_list: list of (n_steps,) bool arrays

    Returns:
        union_mask: (n_steps,) bool array - 联合覆盖掩码
    """
    if not masks_list:
        return np.zeros(1, dtype=bool)
    union_mask = masks_list[0].copy()
    for m in masks_list[1:]:
        union_mask |= m
    return union_mask


def simulate_single_bomb(drone_init, theta, speed, release_time, detonation_delay,
                         missile_idx=0, target_keypoints=None, dt=None, t_total=None):
    """
    仿真单机单弹场景

    Returns:
        total_effective_time: 总有效遮蔽时长
    """
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    if dt is None:
        dt = _c('DT')
    if t_total is None:
        t_total = _c('T_TOTAL')

    mask = _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                               missile_idx, target_keypoints, dt, t_total)
    return mask.sum() * dt


def simulate_single_bomb_weighted(drone_init, theta, speed, release_time, detonation_delay,
                                   missile_idx=0, target_keypoints=None, dt=None, t_total=None):
    """
    仿真单机单弹场景 - 加权覆盖（用于PSO优化）

    每个时间步的贡献 = dt * (覆盖的关键点比例)
    这提供了一个平滑、连续的目标函数，适合PSO优化。

    Returns:
        weighted_effective_time: 加权有效遮蔽时长
    """
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    if dt is None:
        dt = _c('DT')
    if t_total is None:
        t_total = _c('T_TOTAL')

    mask = _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                               missile_idx, target_keypoints, dt, t_total)
    return mask.sum() * dt


def simulate_multi_bomb_single_drone(drone_init, theta, speed, release_times, detonation_delays,
                                     missile_indices, target_keypoints=None, dt=None, t_total=None):
    """
    仿真单机多弹场景（针对指定导弹）

    使用 fit_union 合并多枚弹的覆盖区间。

    Returns:
        total_effective_time: 总有效遮蔽时长
    """
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()
    if dt is None:
        dt = _c('DT')
    if t_total is None:
        t_total = _c('T_TOTAL')

    n_bombs = len(release_times)
    masks = []
    for i in range(n_bombs):
        mask = _bomb_coverage_mask(drone_init, theta, speed, release_times[i],
                                   detonation_delays[i], missile_indices[i],
                                   target_keypoints, dt, t_total)
        masks.append(mask)

    union_mask = fit_union(masks)
    return union_mask.sum() * dt


def simulate_multi_drone_multi_bomb(drone_params_list, dt=None, t_total=None):
    """
    仿真多机多弹多导弹场景

    每枚导弹独立计算联合覆盖，然后求和。

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
    if dt is None:
        dt = _c('DT')
    if t_total is None:
        t_total = _c('T_TOTAL')

    target_keypoints = get_target_keypoints()

    n_steps = int(np.ceil(t_total / dt))
    n_missiles = 3
    per_missile_masks = [[] for _ in range(n_missiles)]

    for p in drone_params_list:
        n_bombs = len(p['release_times'])
        for i in range(n_bombs):
            k = p['missile_indices'][i]
            mask = _bomb_coverage_mask(p['drone_init'], p['theta'], p['speed'],
                                       p['release_times'][i], p['detonation_delays'][i],
                                       k, target_keypoints, dt, t_total)
            per_missile_masks[k].append(mask)

    per_missile_time = np.zeros(n_missiles)
    for k in range(n_missiles):
        if per_missile_masks[k]:
            union_mask = fit_union(per_missile_masks[k])
            per_missile_time[k] = union_mask.sum() * dt

    total_effective_time = per_missile_time.sum()
    return total_effective_time, per_missile_time


def compute_coverage_detail(drone_params_list, dt=None, t_total=None):
    """
    详细计算遮蔽情况，返回每枚导弹的覆盖掩码。

    用于分析和调试。
    """
    if dt is None:
        dt = _c('DT')
    if t_total is None:
        t_total = _c('T_TOTAL')

    target_keypoints = get_target_keypoints()
    n_steps = int(np.ceil(t_total / dt))
    n_missiles = 3
    per_missile_masks = [[] for _ in range(n_missiles)]

    for p in drone_params_list:
        n_bombs = len(p['release_times'])
        for i in range(n_bombs):
            k = p['missile_indices'][i]
            mask = _bomb_coverage_mask(p['drone_init'], p['theta'], p['speed'],
                                       p['release_times'][i], p['detonation_delays'][i],
                                       k, target_keypoints, dt, t_total)
            per_missile_masks[k].append(mask)

    result = {}
    for k in range(n_missiles):
        if per_missile_masks[k]:
            union_mask = fit_union(per_missile_masks[k])
            result[f'M{k+1}'] = {
                'mask': union_mask,
                'time': union_mask.sum() * dt,
                'n_bombs': len(per_missile_masks[k])
            }
        else:
            result[f'M{k+1}'] = {
                'mask': np.zeros(n_steps, dtype=bool),
                'time': 0.0,
                'n_bombs': 0
            }

    return result
