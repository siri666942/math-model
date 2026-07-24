"""
问题1: 固定参数仿真 - 单机单弹，给定投放时间和起爆延时
验证基本仿真模型的正确性
"""
import numpy as np
import time
import simulation as sim


def solve_p1(config_module, verbose=True):
    """
    问题1: 固定参数下的有效遮蔽时长

    参数:
        config_module: 配置模块 (config_a 或 config_c)

    返回:
        effective_time: 有效遮蔽时长 (s)
    """
    sim.set_config(config_module)

    target_keypoints = sim.get_target_keypoints()

    if verbose:
        print(f"  目标关键点数: {target_keypoints.shape[0]}")
        print(f"  投放时间: {config_module.P1_RELEASE_TIME}s, 起爆延时: {config_module.P1_DETONATION_DELAY}s")
        print(f"  无人机速度: {config_module.P1_DRONE_SPEED}m/s, 航向角: {config_module.P1_DRONE_THETA:.4f}rad")
        print(f"  烟幕下沉速度: {config_module.SMOKE_SINK_SPEED}m/s")

    t_start = time.time()
    effective_time = sim.simulate_single_bomb(
        config_module.DRONES_INIT[0],        # FY1
        config_module.P1_DRONE_THETA,        # 航向角
        config_module.P1_DRONE_SPEED,        # 速度
        config_module.P1_RELEASE_TIME,       # 投放时间
        config_module.P1_DETONATION_DELAY,   # 起爆延时
        missile_idx=0,                       # M1
        target_keypoints=target_keypoints,
    )
    elapsed = time.time() - t_start

    if verbose:
        print(f"  有效遮蔽时长: {effective_time:.4f}s")
        print(f"  仿真耗时: {elapsed:.3f}s")

    return effective_time
