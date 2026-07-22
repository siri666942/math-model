"""
问题2: 利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰
确定FY1的飞行方向、速度、投放点、起爆点，使遮蔽时间尽可能长

优化变量: [theta, speed, release_time, detonation_delay]
"""
import numpy as np
from config import (
    DRONES_INIT, DRONE_SPEED_MIN, DRONE_SPEED_MAX, DT, T_TOTAL,
    PSO_SWARM_SIZE, PSO_MAX_ITER,
)
from simulation import simulate_single_bomb, get_target_keypoints
from pso import PSO


def solve_problem2():
    """求解问题2: 单机单弹最优策略"""
    print("=" * 60)
    print("问题2: FY1单机单弹最优投放策略 (PSO优化)")
    print("=" * 60)

    drone_init = DRONES_INIT[0]  # FY1
    target_keypoints = get_target_keypoints(360, 0)  # 上下底面720点

    # 决策变量: [theta, speed, release_time, detonation_delay]
    bounds = [
        (np.pi * 0.25, np.pi * 0.5),   # theta (rad) - 朝向原点方向
        (DRONE_SPEED_MIN, DRONE_SPEED_MAX),  # speed (m/s)
        (0.0, 15.0),                    # release_time (s)
        (0.0, 6.0),                     # detonation_delay (s)
    ]

    def objective(x):
        theta, speed, release_time, detonation_delay = x
        return simulate_single_bomb(
            drone_init, theta, speed, release_time, detonation_delay,
            missile_idx=0, target_keypoints=target_keypoints, dt=DT, t_total=T_TOTAL
        )

    print(f"\n变量范围:")
    print(f"  theta: [{bounds[0][0]:.2f}, {bounds[0][1]:.2f}] rad")
    print(f"  speed: [{bounds[1][0]}, {bounds[1][1]}] m/s")
    print(f"  release_time: [{bounds[2][0]}, {bounds[2][1]}] s")
    print(f"  detonation_delay: [{bounds[3][0]}, {bounds[3][1]}] s")
    print(f"\n粒子群规模: {PSO_SWARM_SIZE}, 迭代次数: {PSO_MAX_ITER}")

    # PSO 优化
    pso = PSO(objective, bounds, n_particles=200, max_iter=100, maximize=True, verbose=True)
    x_opt, f_opt = pso.optimize()

    print(f"\n优化结果:")
    print(f"  航向角θ: {x_opt[0]:.4f} rad ({np.degrees(x_opt[0]):.2f}°)")
    print(f"  飞行速度: {x_opt[1]:.2f} m/s")
    print(f"  投放时间: {x_opt[2]:.4f} s")
    print(f"  起爆延时: {x_opt[3]:.4f} s")

    # 计算投放点和起爆点坐标
    direction = np.array([np.cos(x_opt[0]), np.sin(x_opt[0]), 0.0])
    release_pos = drone_init + x_opt[1] * direction * x_opt[2]
    detonation_pos = release_pos + x_opt[1] * direction * x_opt[3]
    detonation_pos[2] -= 0.5 * 9.8 * x_opt[3] ** 2

    print(f"  投放点坐标: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
    print(f"  起爆点坐标: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")
    print(f"  最长有效遮蔽时长: {f_opt:.4f} s")

    return x_opt, f_opt


if __name__ == "__main__":
    solve_problem2()
