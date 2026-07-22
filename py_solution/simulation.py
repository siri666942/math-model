"""
核心仿真引擎 - 运动学模型与遮蔽判定
"""
import numpy as np
from config import (
    G, TARGET_CENTER, TARGET_RADIUS, TARGET_HEIGHT,
    MISSILE_SPEED, SMOKE_SINK_SPEED, EFFECTIVE_RADIUS, EFFECTIVE_DURATION,
    DT, T_TOTAL, N_CIRCLE_POINTS, N_SIDE_LAYERS,
    MISSILES_INIT, MISSILES_DIR, DRONES_INIT,
)


def generate_target_keypoints(n_circle=N_CIRCLE_POINTS, n_layers=N_SIDE_LAYERS):
    """
    生成目标圆柱体表面的关键点集

    包含:
    - 上底面圆周点
    - 下底面圆周点
    - 侧面多层圆周点

    Returns:
        keypoints: (N, 3) 关键点坐标数组
    """
    keypoints = []

    # 上下底面圆周点
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = TARGET_RADIUS * np.cos(angle)
        y = TARGET_RADIUS * np.sin(angle)
        # 下底面
        keypoints.append([TARGET_CENTER[0] + x, TARGET_CENTER[1] + y, TARGET_CENTER[2]])
        # 上底面
        keypoints.append([TARGET_CENTER[0] + x, TARGET_CENTER[1] + y, TARGET_CENTER[2] + TARGET_HEIGHT])

    # 侧面多层圆周点
    for layer in range(1, n_layers + 1):
        z_height = TARGET_CENTER[2] + TARGET_HEIGHT * layer / (n_layers + 1)
        for i in range(n_circle):
            angle = 2 * np.pi * i / n_circle
            x = TARGET_RADIUS * np.cos(angle)
            y = TARGET_RADIUS * np.sin(angle)
            keypoints.append([TARGET_CENTER[0] + x, TARGET_CENTER[1] + y, z_height])

    return np.array(keypoints)


# 预生成关键点集（全局缓存）
_TARGET_KEYPOINTS = None
_TARGET_KEYPOINTS_SMALL = None


def get_target_keypoints(n_circle=N_CIRCLE_POINTS, n_layers=N_SIDE_LAYERS):
    """获取目标关键点集（缓存）"""
    global _TARGET_KEYPOINTS, _TARGET_KEYPOINTS_SMALL
    if n_circle == N_CIRCLE_POINTS and n_layers == N_SIDE_LAYERS:
        if _TARGET_KEYPOINTS is None:
            _TARGET_KEYPOINTS = generate_target_keypoints(n_circle, n_layers)
        return _TARGET_KEYPOINTS
    elif n_circle == 8 and n_layers == 0:
        if _TARGET_KEYPOINTS_SMALL is None:
            _TARGET_KEYPOINTS_SMALL = generate_target_keypoints(8, 0)
        return _TARGET_KEYPOINTS_SMALL
    else:
        return generate_target_keypoints(n_circle, n_layers)


def missile_position(t, missile_idx=0):
    """计算导弹在时刻t的位置"""
    return MISSILES_INIT[missile_idx] + MISSILE_SPEED * MISSILES_DIR[missile_idx] * t


def missile_positions(t, n_missiles=3):
    """计算所有导弹在时刻t的位置"""
    M_pos = np.zeros((n_missiles, 3))
    for k in range(n_missiles):
        M_pos[k] = MISSILES_INIT[k] + MISSILE_SPEED * MISSILES_DIR[k] * t
    return M_pos


def drone_position(t, drone_init, theta, speed):
    """计算无人机在时刻t的位置"""
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    return drone_init + speed * direction * t


def check_occlusion(missile_pos, smoke_pos, target_keypoints):
    """
    判断烟幕云团是否有效遮蔽真目标（从导弹视角）

    返回 True 如果目标被有效遮蔽

    遮蔽条件:
    1. 导弹在云团内部 (距离 < 有效半径)
    2. 云团在导弹和目标之间，且目标所有关键点都在视锥内
    """
    vec_missile_to_smoke = smoke_pos - missile_pos
    dist_ms = np.linalg.norm(vec_missile_to_smoke)

    # 条件1: 导弹进入云团内部
    if dist_ms < EFFECTIVE_RADIUS:
        return True

    # 条件2: 云团必须在导弹和目标之间
    dist_mt = np.linalg.norm(missile_pos - TARGET_CENTER)
    dist_st = np.linalg.norm(smoke_pos - TARGET_CENTER)

    if dist_mt <= dist_st:
        # 云团在目标后方或同位置，无法遮蔽
        return False

    # 计算视锥半顶角
    sin_alpha = EFFECTIVE_RADIUS / dist_ms
    if sin_alpha >= 1.0:
        return True  # 云团覆盖了导弹到目标的所有视线
    cos_alpha = np.sqrt(1.0 - sin_alpha ** 2)

    # 检查所有关键点是否都在视锥内
    for kp in target_keypoints:
        tar_axis = kp - missile_pos
        dot_product = np.dot(vec_missile_to_smoke, tar_axis)
        norm_product = dist_ms * np.linalg.norm(tar_axis)

        if norm_product < 1e-12:
            continue

        cos_gamma = dot_product / norm_product

        # 如果 cos_alpha > cos_gamma (即 alpha < gamma)
        # 表示该关键点在视锥外部，遮蔽失败
        if cos_alpha > cos_gamma:
            return False

    # 所有关键点都在视锥内，遮蔽有效
    return True


def _check_occlusion_batch(M_pos, smoke_pos, target_keypoints, chunk_size=None):
    """
    向量化遮蔽判据：对一批 (missile_pos, smoke_pos) 时刻一次性判断是否遮蔽

    Parameters:
        M_pos: (T,3) 导弹位置序列
        smoke_pos: (T,3) 云团位置序列（与M_pos一一对应同一时刻）
        target_keypoints: (K,3) 目标关键点集

    Returns:
        (T,) 布尔数组
    """
    T = M_pos.shape[0]
    K = target_keypoints.shape[0]
    if chunk_size is None:
        # 控制单个chunk的临时数组规模（T_chunk*K*3），避免内存爆炸
        chunk_size = max(1, int(2e7 / max(K * 3, 1)))

    result = np.empty(T, dtype=bool)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        m = M_pos[start:end]
        s = smoke_pos[start:end]

        vec_ms = s - m
        dist_ms = np.linalg.norm(vec_ms, axis=1)
        inside_cloud = dist_ms < EFFECTIVE_RADIUS

        dist_mt = np.linalg.norm(m - TARGET_CENTER, axis=1)
        dist_st = np.linalg.norm(s - TARGET_CENTER, axis=1)
        blocked = dist_mt <= dist_st

        dist_ms_safe = np.maximum(dist_ms, 1e-9)
        sin_alpha = np.clip(EFFECTIVE_RADIUS / dist_ms_safe, 0.0, 1.0)
        full_cover = sin_alpha >= 1.0
        cos_alpha = np.sqrt(np.maximum(1.0 - sin_alpha ** 2, 0.0))

        # 只有"不在云团内 & 导弹比云团更远离目标 & 视锥半顶角未满90度"这个子集，
        # 视锥采样(K个关键点的点积/夹角)才会影响最终结果，其余行算了也会被下面的
        # inside_cloud/~blocked/full_cover 掩掉，所以只对这个子集做最贵的那部分计算
        need = (~inside_cloud) & (~blocked) & (~full_cover)
        all_in_cone = np.zeros(end - start, dtype=bool)
        if np.any(need):
            m_n = m[need]
            vec_ms_n = vec_ms[need]
            dist_ms_n = dist_ms[need]
            cos_alpha_n = cos_alpha[need]

            tar_axis = target_keypoints[None, :, :] - m_n[:, None, :]     # (t_n,K,3)
            dot = np.einsum('ti,tki->tk', vec_ms_n, tar_axis)
            norm_tar = np.linalg.norm(tar_axis, axis=2)
            denom = dist_ms_n[:, None] * norm_tar
            denom_safe = np.where(denom < 1e-12, np.inf, denom)
            cos_gamma = dot / denom_safe
            all_in_cone[need] = np.all(cos_alpha_n[:, None] <= cos_gamma, axis=1)

        cone_ok = full_cover | all_in_cone
        result[start:end] = inside_cloud | (~blocked & cone_ok)

    return result


def _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                        missile_idx, target_keypoints, dt, t_total):
    """
    计算单枚烟幕弹在全局时间网格 arange(0, t_total, dt) 上、独自能否遮蔽指定导弹的布尔掩码

    使用云团/导弹位置的解析式（而非逐步累加），只在起爆~失效窗口内做向量化判据计算，
    是 0.1/0.2 模块的核心：把"全部关键点 × 该时刻"一次性算成矩阵运算，避免逐时刻/逐点的Python循环。
    """
    n_steps = int(np.ceil(t_total / dt))
    mask = np.zeros(n_steps, dtype=bool)

    detonation_time = release_time + detonation_delay
    smoke_expire_time = detonation_time + EFFECTIVE_DURATION
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
    bomb_h = FY_at_release[2] - 0.5 * G * detonation_delay ** 2

    smoke_pos = np.empty((len(ts), 3))
    smoke_pos[:, 0] = bomb_x
    smoke_pos[:, 1] = bomb_y
    smoke_pos[:, 2] = bomb_h - SMOKE_SINK_SPEED * (ts - detonation_time)

    M_pos = MISSILES_INIT[missile_idx] + MISSILE_SPEED * MISSILES_DIR[missile_idx] * ts[:, None]

    mask[i_start:i_end] = _check_occlusion_batch(M_pos, smoke_pos, target_keypoints)
    return mask


def simulate_single_bomb(drone_init, theta, speed, release_time, detonation_delay,
                         missile_idx=0, target_keypoints=None, dt=DT, t_total=T_TOTAL):
    """
    仿真单机单弹场景

    Parameters:
        drone_init: 无人机初始位置
        theta: 无人机航向角
        speed: 无人机飞行速度
        release_time: 投放时间
        detonation_delay: 起爆延时
        missile_idx: 目标导弹索引

    Returns:
        total_effective_time: 总有效遮蔽时长
    """
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()

    mask = _bomb_coverage_mask(drone_init, theta, speed, release_time, detonation_delay,
                               missile_idx, target_keypoints, dt, t_total)
    return mask.sum() * dt


def _multi_cloud_coverage_mask(bombs, missile_idx, target_keypoints, dt, t_total):
    """
    多枚烟幕弹的"互补遮蔽"判定：某一时刻只要目标每一个关键点都被"至少一朵"当前
    存活的云团挡住即可判定为有效遮蔽——不要求同一朵云单独挡住全部关键点。

    与逐弹独立判定再取并集（旧逻辑）的区别：旧逻辑要求某一时刻必须有一枚弹自己
    就覆盖了全部目标关键点；这里允许弹A挡住一部分关键点、弹B同时挡住另一部分，
    合起来覆盖全部关键点时也算有效遮蔽，对应"多弹互补遮蔽"的场景。

    bombs: [(drone_init, theta, speed, release_time, detonation_delay), ...]，
           全部针对同一枚导弹 missile_idx
    """
    n_steps = int(np.ceil(t_total / dt))
    K = target_keypoints.shape[0]
    result = np.zeros(n_steps, dtype=bool)

    windows = []
    for bomb in bombs:
        release_time, detonation_delay = bomb[3], bomb[4]
        detonation_time = release_time + detonation_delay
        smoke_expire_time = detonation_time + EFFECTIVE_DURATION
        t_start = max(0.0, detonation_time)
        t_end = min(t_total, smoke_expire_time)
        if t_end <= t_start:
            continue
        i_start = int(np.ceil(t_start / dt))
        i_end = min(n_steps, int(np.floor((t_end + 1e-9) / dt)) + 1)
        if i_end <= i_start:
            continue
        windows.append((i_start, i_end, bomb))

    if not windows:
        return result

    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    ts_all = (lo + np.arange(hi - lo)) * dt
    M_pos_all = MISSILES_INIT[missile_idx] + MISSILE_SPEED * MISSILES_DIR[missile_idx] * ts_all[:, None]

    inside_cloud_any = np.zeros(hi - lo, dtype=bool)
    keypoint_blocked = np.zeros((hi - lo, K), dtype=bool)
    any_active = np.zeros(hi - lo, dtype=bool)

    for i_start, i_end, (drone_init, theta, speed, release_time, detonation_delay) in windows:
        detonation_time = release_time + detonation_delay
        rel_start, rel_end = i_start - lo, i_end - lo
        ts = ts_all[rel_start:rel_end]

        direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        FY_at_release = drone_init + speed * direction * release_time
        bomb_x = FY_at_release[0] + speed * direction[0] * detonation_delay
        bomb_y = FY_at_release[1] + speed * direction[1] * detonation_delay
        bomb_h = FY_at_release[2] - 0.5 * G * detonation_delay ** 2

        smoke_pos = np.empty((len(ts), 3))
        smoke_pos[:, 0] = bomb_x
        smoke_pos[:, 1] = bomb_y
        smoke_pos[:, 2] = bomb_h - SMOKE_SINK_SPEED * (ts - detonation_time)

        m = M_pos_all[rel_start:rel_end]
        vec_ms = smoke_pos - m
        dist_ms = np.linalg.norm(vec_ms, axis=1)
        inside_cloud = dist_ms < EFFECTIVE_RADIUS
        inside_cloud_any[rel_start:rel_end] |= inside_cloud

        dist_mt = np.linalg.norm(m - TARGET_CENTER, axis=1)
        dist_st = np.linalg.norm(smoke_pos - TARGET_CENTER, axis=1)
        eligible = dist_mt > dist_st  # 这朵云得在导弹和目标之间，才谈得上挡住哪个关键点

        dist_ms_safe = np.maximum(dist_ms, 1e-9)
        sin_alpha = np.clip(EFFECTIVE_RADIUS / dist_ms_safe, 0.0, 1.0)
        full_cover = sin_alpha >= 1.0
        cos_alpha = np.sqrt(np.maximum(1.0 - sin_alpha ** 2, 0.0))

        need = eligible & (~full_cover)
        in_cone = np.zeros((len(ts), K), dtype=bool)
        if np.any(need):
            m_n = m[need]
            vec_ms_n = vec_ms[need]
            dist_ms_n = dist_ms[need]
            cos_alpha_n = cos_alpha[need]

            tar_axis = target_keypoints[None, :, :] - m_n[:, None, :]
            dot = np.einsum('ti,tki->tk', vec_ms_n, tar_axis)
            norm_tar = np.linalg.norm(tar_axis, axis=2)
            denom = dist_ms_n[:, None] * norm_tar
            denom_safe = np.where(denom < 1e-12, np.inf, denom)
            cos_gamma = dot / denom_safe
            in_cone[need] = cos_alpha_n[:, None] <= cos_gamma

        in_cone |= (eligible & full_cover)[:, None]

        keypoint_blocked[rel_start:rel_end] |= in_cone
        any_active[rel_start:rel_end] = True

    all_blocked = np.all(keypoint_blocked, axis=1) & any_active
    result[lo:hi] = inside_cloud_any | all_blocked
    return result


def simulate_multi_bomb_single_drone(drone_init, theta, speed, release_times, detonation_delays,
                                     missile_indices, target_keypoints=None, dt=DT, t_total=T_TOTAL):
    """
    仿真单机多弹场景（针对指定导弹）

    Parameters:
        drone_init: 无人机初始位置
        theta: 无人机航向角
        speed: 无人机飞行速度
        release_times: 投放时间列表 [n_bombs]
        detonation_delays: 起爆延时列表 [n_bombs]
        missile_indices: 每枚弹对应的目标导弹索引 [n_bombs]

    Returns:
        total_effective_time: 总有效遮蔽时长（同一导弹上的多枚弹按"互补遮蔽"判定后取并集）
    """
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()

    n_steps = int(np.ceil(t_total / dt))
    any_covered = np.zeros(n_steps, dtype=bool)

    missile_indices = np.asarray(missile_indices)
    for k in np.unique(missile_indices):
        bombs = [(drone_init, theta, speed, release_times[i], detonation_delays[i])
                 for i in range(len(release_times)) if missile_indices[i] == k]
        mask = _multi_cloud_coverage_mask(bombs, int(k), target_keypoints, dt, t_total)
        any_covered |= mask

    return any_covered.sum() * dt


def simulate_multi_drone_multi_bomb(drone_params_list, dt=DT, t_total=T_TOTAL):
    """
    仿真多机多弹多导弹场景

    Parameters:
        drone_params_list: 每架无人机的参数字典列表
            [{
                'drone_init': np.array([x,y,z]),
                'theta': float,
                'speed': float,
                'release_times': np.array([t1, t2, t3]),
                'detonation_delays': np.array([d1, d2, d3]),
                'missile_indices': [k1, k2, k3],  # 每枚弹对应的导弹索引
            }, ...]

    Returns:
        total_effective_time: 总有效遮蔽时长（同一导弹上的所有云团按"互补遮蔽"判定，再对三枚导弹求和）
        per_missile_time: 每枚导弹的遮蔽时长
    """
    target_keypoints = get_target_keypoints(n_circle=360, n_layers=10)

    n_steps = int(np.ceil(t_total / dt))
    n_missiles = 3
    per_missile_mask = [np.zeros(n_steps, dtype=bool) for _ in range(n_missiles)]

    bombs_by_missile = {0: [], 1: [], 2: []}
    for p in drone_params_list:
        n_bombs = len(p['release_times'])
        for i in range(n_bombs):
            k = int(p['missile_indices'][i])
            bombs_by_missile[k].append((p['drone_init'], p['theta'], p['speed'],
                                        p['release_times'][i], p['detonation_delays'][i]))

    for k, bombs in bombs_by_missile.items():
        if bombs:
            per_missile_mask[k] = _multi_cloud_coverage_mask(bombs, k, target_keypoints, dt, t_total)

    per_missile_time = np.array([m.sum() * dt for m in per_missile_mask])
    total_effective_time = per_missile_time.sum()

    return total_effective_time, per_missile_time
