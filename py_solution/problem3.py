"""
问题3: 利用无人机FY1投放3枚烟幕干扰弹实施对M1的干扰
输出结果到 result1.xlsx

优化变量(8维): [theta, speed, release1, interval2, interval3, delay1, delay2, delay3]
实际投放时间: [release1, release1+interval2, release1+interval2+interval3]
"""
import numpy as np
import openpyxl
from config import (
    DRONES_INIT, DRONE_SPEED_MIN, DRONE_SPEED_MAX,
    BOMB_INTERVAL_MIN, DT, T_TOTAL,
)
from simulation import simulate_multi_bomb_single_drone, get_target_keypoints
from pso import PSO

# 为 problem3 使用独立的 PSO 参数
PSO_SWARM_SIZE_P3 = 300
PSO_MAX_ITER_P3 = 150


def solve_problem3():
    """求解问题3: 单机三弹最优策略"""
    print("=" * 60)
    print("问题3: FY1投放3枚烟幕干扰弹对M1的最优策略 (PSO优化)")
    print("=" * 60)

    drone_init = DRONES_INIT[0]  # FY1
    target_keypoints = get_target_keypoints(360, 10)

    # 决策变量(8维):
    # x[0] = theta, x[1] = speed
    # x[2] = release1, x[3] = interval_to_2nd, x[4] = interval_to_3rd
    # x[5] = delay1, x[6] = delay2, x[7] = delay3
    bounds = [
        (np.pi * 0.3, np.pi * 0.5),     # theta
        (DRONE_SPEED_MIN, DRONE_SPEED_MAX),  # speed
        (0.0, 5.0),                      # release1
        (BOMB_INTERVAL_MIN, 4.0),        # interval 1->2
        (BOMB_INTERVAL_MIN, 4.0),        # interval 2->3
        (0.0, 6.0),                      # delay1
        (0.0, 6.0),                      # delay2
        (0.0, 6.0),                      # delay3
    ]

    def objective(x):
        theta = x[0]
        speed = x[1]
        release_times = np.array([
            x[2],
            x[2] + x[3],
            x[2] + x[3] + x[4],
        ])
        detonation_delays = np.array([x[5], x[6], x[7]])
        missile_indices = np.array([0, 0, 0])  # 全部针对M1

        return simulate_multi_bomb_single_drone(
            drone_init, theta, speed, release_times, detonation_delays,
            missile_indices, target_keypoints=target_keypoints, dt=DT, t_total=T_TOTAL
        )

    print(f"\n变量维度: 8 (theta, speed, 3×release, 3×delay)")
    print(f"粒子群规模: {PSO_SWARM_SIZE_P3}, 迭代次数: {PSO_MAX_ITER_P3}")

    pso = PSO(objective, bounds, n_particles=PSO_SWARM_SIZE_P3,
              max_iter=PSO_MAX_ITER_P3, maximize=True, verbose=True)
    x_opt, f_opt = pso.optimize()

    # 解析结果
    theta = x_opt[0]
    speed = x_opt[1]
    release_times = np.array([x_opt[2], x_opt[2] + x_opt[3], x_opt[2] + x_opt[3] + x_opt[4]])
    detonation_delays = np.array([x_opt[5], x_opt[6], x_opt[7]])

    direction = np.array([np.cos(theta), np.sin(theta), 0.0])

    print(f"\n优化结果:")
    print(f"  航向角θ: {theta:.4f} rad ({np.degrees(theta):.2f}°)")
    print(f"  飞行速度: {speed:.2f} m/s")

    for i in range(3):
        release_pos = drone_init + speed * direction * release_times[i]
        detonation_pos = release_pos + speed * direction * detonation_delays[i]
        detonation_pos[2] -= 0.5 * 9.8 * detonation_delays[i] ** 2

        print(f"\n  烟幕弹{i+1}:")
        print(f"    投放时间: {release_times[i]:.4f} s")
        print(f"    起爆延时: {detonation_delays[i]:.4f} s")
        print(f"    投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
        print(f"    起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")

    print(f"\n  总有效遮蔽时长: {f_opt:.4f} s")

    # 保存到 result1.xlsx
    save_result1(theta, speed, release_times, detonation_delays, drone_init,
                 direction, f_opt)

    return x_opt, f_opt


def save_result1(theta, speed, release_times, detonation_delays,
                 drone_init, direction, total_time):
    """保存问题3结果到 result1.xlsx"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "问题3结果"

    # 表头
    headers = ["无人机", "航向角(rad)", "航向角(°)", "飞行速度(m/s)",
               "烟幕弹编号", "投放时间(s)", "起爆延时(s)",
               "投放点X", "投放点Y", "投放点Z",
               "起爆点X", "起爆点Y", "起爆点Z",
               "总有效遮蔽时长(s)"]
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)

    # 数据行
    for i in range(3):
        row = i + 2
        release_pos = drone_init + speed * direction * release_times[i]
        detonation_pos = release_pos + speed * direction * detonation_delays[i]
        detonation_pos[2] -= 0.5 * 9.8 * detonation_delays[i] ** 2

        ws.cell(row=row, column=1, value="FY1")
        ws.cell(row=row, column=2, value=round(theta, 6))
        ws.cell(row=row, column=3, value=round(np.degrees(theta), 4))
        ws.cell(row=row, column=4, value=round(speed, 2))
        ws.cell(row=row, column=5, value=i + 1)
        ws.cell(row=row, column=6, value=round(release_times[i], 4))
        ws.cell(row=row, column=7, value=round(detonation_delays[i], 4))
        ws.cell(row=row, column=8, value=round(release_pos[0], 2))
        ws.cell(row=row, column=9, value=round(release_pos[1], 2))
        ws.cell(row=row, column=10, value=round(release_pos[2], 2))
        ws.cell(row=row, column=11, value=round(detonation_pos[0], 2))
        ws.cell(row=row, column=12, value=round(detonation_pos[1], 2))
        ws.cell(row=row, column=13, value=round(detonation_pos[2], 2))
        if i == 0:
            ws.cell(row=row, column=14, value=round(total_time, 4))

    filepath = "result1.xlsx"
    wb.save(filepath)
    print(f"\n结果已保存至: {filepath}")


if __name__ == "__main__":
    solve_problem3()
