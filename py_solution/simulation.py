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

    total_effective_time = 0.0
    smoke_pos = np.zeros(3)
    bomb_detonated = False
    bomb_h = 0.0  # 起爆时的高度（用于固定云团xy位置）
    detonation_time = release_time + detonation_delay
    smoke_expire_time = detonation_time + EFFECTIVE_DURATION

    for t in np.arange(0, t_total, dt):
        M_pos = missile_position(t, missile_idx)
        FY_pos = drone_position(t, drone_init, theta, speed)

        if t < release_time:
            # 尚未投放
            continue

        if t < detonation_time:
            # 平抛运动阶段
            delta_t = t - release_time
            vertical_dist = -0.5 * G * delta_t ** 2
            smoke_pos = np.array([FY_pos[0], FY_pos[1], FY_pos[2] + vertical_dist])
        else:
            # 爆炸后云团下沉
            if not bomb_detonated:
                bomb_detonated = True
                # 计算起爆时刻的位置
                FY_at_release = drone_position(release_time, drone_init, theta, speed)
                FY_at_detonation_x = FY_at_release[0] + speed * np.cos(theta) * detonation_delay
                FY_at_detonation_y = FY_at_release[1] + speed * np.sin(theta) * detonation_delay
                bomb_h = FY_at_release[2] - 0.5 * G * detonation_delay ** 2
                smoke_pos[0] = FY_at_detonation_x
                smoke_pos[1] = FY_at_detonation_y
                smoke_pos[2] = bomb_h

            # 云团下沉
            smoke_pos[2] -= SMOKE_SINK_SPEED * dt

            # 检查是否在有效期内
            if t <= smoke_expire_time:
                if check_occlusion(M_pos, smoke_pos, target_keypoints):
                    total_effective_time += dt

    return total_effective_time


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
        total_effective_time: 总有效遮蔽时长
    """
    if target_keypoints is None:
        target_keypoints = get_target_keypoints()

    n_bombs = len(release_times)
    total_effective_time = 0.0

    # 每枚弹的状态
    smoke_pos = np.zeros((n_bombs, 3))
    bomb_detonated = np.zeros(n_bombs, dtype=bool)
    bomb_h = np.zeros(n_bombs)
    detonation_times = release_times + detonation_delays
    smoke_expire_times = detonation_times + EFFECTIVE_DURATION

    for t in np.arange(0, t_total, dt):
        FY_pos = drone_position(t, drone_init, theta, speed)

        is_effective_any = False

        for i in range(n_bombs):
            if t < release_times[i]:
                continue

            if t < detonation_times[i]:
                # 平抛运动
                delta_t = t - release_times[i]
                vertical_dist = -0.5 * G * delta_t ** 2
                smoke_pos[i] = np.array([FY_pos[0], FY_pos[1], FY_pos[2] + vertical_dist])
            else:
                # 云团下沉
                if not bomb_detonated[i]:
                    bomb_detonated[i] = True
                    FY_at_release = drone_position(release_times[i], drone_init, theta, speed)
                    bomb_h[i] = FY_at_release[2] - 0.5 * G * detonation_delays[i] ** 2
                    smoke_pos[i, 0] = FY_at_release[0] + speed * np.cos(theta) * detonation_delays[i]
                    smoke_pos[i, 1] = FY_at_release[1] + speed * np.sin(theta) * detonation_delays[i]
                    smoke_pos[i, 2] = bomb_h[i]

                smoke_pos[i, 2] -= SMOKE_SINK_SPEED * dt

                if t <= smoke_expire_times[i]:
                    M_pos = missile_position(t, missile_indices[i])
                    if check_occlusion(M_pos, smoke_pos[i], target_keypoints):
                        is_effective_any = True

        if is_effective_any:
            total_effective_time += dt

    return total_effective_time


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
        total_effective_time: 总有效遮蔽时长（每个导弹独立计时，取并集）
        per_missile_time: 每枚导弹的遮蔽时长
    """
    target_keypoints = get_target_keypoints(n_circle=360, n_layers=10)

    n_drones = len(drone_params_list)
    n_missiles = 3

    # 初始化状态
    n_bombs_per_drone = [len(p['release_times']) for p in drone_params_list]
    smoke_pos = []  # 存储为 list of arrays
    bomb_detonated = []
    bomb_h = []
    detonation_times = []
    smoke_expire_times = []

    for drone_idx in range(n_drones):
        p = drone_params_list[drone_idx]
        nb = n_bombs_per_drone[drone_idx]
        smoke_pos.append(np.zeros((nb, 3)))
        bomb_detonated.append(np.zeros(nb, dtype=bool))
        bomb_h.append(np.zeros(nb))
        detonation_times.append(p['release_times'] + p['detonation_delays'])
        smoke_expire_times.append(detonation_times[-1] + EFFECTIVE_DURATION)

    total_effective_time = 0.0
    per_missile_time = np.zeros(n_missiles)

    for t in np.arange(0, t_total, dt):
        M_pos = missile_positions(t, n_missiles)

        # 对每枚导弹，检查是否有任何烟幕弹遮蔽它
        missile_covered = np.zeros(n_missiles, dtype=bool)

        for drone_idx in range(n_drones):
            p = drone_params_list[drone_idx]
            FY_pos = drone_position(t, p['drone_init'], p['theta'], p['speed'])
            nb = n_bombs_per_drone[drone_idx]

            for i in range(nb):
                if t < p['release_times'][i]:
                    continue

                if t < detonation_times[drone_idx][i]:
                    # 平抛运动
                    delta_t = t - p['release_times'][i]
                    vertical_dist = -0.5 * G * delta_t ** 2
                    smoke_pos[drone_idx][i] = np.array([FY_pos[0], FY_pos[1], FY_pos[2] + vertical_dist])
                else:
                    # 云团下沉
                    if not bomb_detonated[drone_idx][i]:
                        bomb_detonated[drone_idx][i] = True
                        FY_at_release = drone_position(p['release_times'][i], p['drone_init'], p['theta'], p['speed'])
                        bomb_h[drone_idx][i] = FY_at_release[2] - 0.5 * G * p['detonation_delays'][i] ** 2
                        smoke_pos[drone_idx][i, 0] = FY_at_release[0] + p['speed'] * np.cos(p['theta']) * p['detonation_delays'][i]
                        smoke_pos[drone_idx][i, 1] = FY_at_release[1] + p['speed'] * np.sin(p['theta']) * p['detonation_delays'][i]
                        smoke_pos[drone_idx][i, 2] = bomb_h[drone_idx][i]

                    smoke_pos[drone_idx][i, 2] -= SMOKE_SINK_SPEED * dt

                    if t <= smoke_expire_times[drone_idx][i]:
                        k = p['missile_indices'][i]
                        if not missile_covered[k]:
                            if check_occlusion(M_pos[k], smoke_pos[drone_idx][i], target_keypoints):
                                missile_covered[k] = True

        for k in range(n_missiles):
            if missile_covered[k]:
                total_effective_time += dt
                per_missile_time[k] += dt

    return total_effective_time, per_missile_time
