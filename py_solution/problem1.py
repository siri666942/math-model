"""
问题1: 利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰
给定飞行参数，计算有效遮蔽时长

C题参数: FY1以120m/s朝向假目标(θ=π)飞行，受领任务1.2s后投放，3.2s后起爆
"""
import numpy as np
from config import (
    P1_RELEASE_TIME, P1_DETONATION_DELAY, P1_DRONE_SPEED, P1_DRONE_THETA,
    DT, T_TOTAL, DRONES_INIT,
)
from simulation import simulate_single_bomb


def solve_problem1():
    """求解问题1"""
    print("=" * 60)
    print("问题1: FY1投放1枚烟幕干扰弹对M1的有效遮蔽时长")
    print("=" * 60)

    drone_init = DRONES_INIT[0]  # FY1
    theta = P1_DRONE_THETA       # π (朝向假目标)
    speed = P1_DRONE_SPEED       # 120 m/s
    release_time = P1_RELEASE_TIME   # 1.2 s
    detonation_delay = P1_DETONATION_DELAY  # 3.2 s

    print(f"\n输入参数:")
    print(f"  FY1初始位置: ({drone_init[0]}, {drone_init[1]}, {drone_init[2]})")
    print(f"  航向角θ: {theta:.4f} rad ({np.degrees(theta):.2f}°)")
    print(f"  飞行速度: {speed} m/s")
    print(f"  投放时间(受领任务后): {release_time} s")
    print(f"  起爆延时(投放后): {detonation_delay} s")

    # 使用更精细的时间步长
    dt_fine = 0.0001  # 精细步长
    effective_time = simulate_single_bomb(
        drone_init, theta, speed, release_time, detonation_delay,
        missile_idx=0, dt=dt_fine, t_total=T_TOTAL
    )

    print(f"\n结果:")
    print(f"  烟幕干扰弹对M1的有效遮蔽时长: {effective_time:.4f} s")

    return effective_time


if __name__ == "__main__":
    solve_problem1()
