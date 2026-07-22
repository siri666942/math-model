"""
问题4: 利用FY1、FY2、FY3各投放1枚烟幕干扰弹，实施对M1的干扰
输出结果到 result2.xlsx

优化变量(12维):
[theta1, theta2, theta3, speed1, speed2, speed3,
 release1, release2, release3, delay1, delay2, delay3]
"""
import numpy as np
import openpyxl
from config import (
    DRONES_INIT, DRONE_SPEED_MIN, DRONE_SPEED_MAX, DT, T_TOTAL,
)
from simulation import simulate_multi_drone_multi_bomb, get_target_keypoints
from pso import PSO

PSO_SWARM_P4 = 500
PSO_ITER_P4 = 300


def solve_problem4():
    """求解问题4: 三机各一弹对M1"""
    print("=" * 60)
    print("问题4: FY1/FY2/FY3各投放1枚烟幕弹对M1 (PSO优化)")
    print("=" * 60)

    target_keypoints = get_target_keypoints(360, 10)
    n_drones = 3

    # 12个变量: 3×theta, 3×speed, 3×release_time, 3×delay
    bounds = []
    # theta: FY1朝向原点, FY2偏y负, FY3偏y正
    bounds += [(np.pi * 0.2, np.pi * 0.5),    # theta1 (FY1)
               (0.0, np.pi * 0.3),             # theta2 (FY2)
               (0.0, np.pi * 0.3)]             # theta3 (FY3)
    # speed
    bounds += [(DRONE_SPEED_MIN, DRONE_SPEED_MAX)] * n_drones
    # release_time
    bounds += [(0.0, 20.0)] * n_drones
    # detonation_delay
    bounds += [(0.0, 20.0)] * n_drones

    def objective(x):
        theta = x[0:3]
        speed = x[3:6]
        release_times = x[6:9]
        delays = x[9:12]

        drone_params = []
        for i in range(n_drones):
            drone_params.append({
                'drone_init': DRONES_INIT[i],
                'theta': theta[i],
                'speed': speed[i],
                'release_times': np.array([release_times[i]]),
                'detonation_delays': np.array([delays[i]]),
                'missile_indices': [0],  # 全部针对M1
            })

        total_time, per_missile = simulate_multi_drone_multi_bomb(
            drone_params, dt=DT, t_total=T_TOTAL
        )
        return total_time

    print(f"\n变量维度: 12 (3×theta, 3×speed, 3×release, 3×delay)")
    print(f"粒子群规模: {PSO_SWARM_P4}, 迭代次数: {PSO_ITER_P4}")

    pso = PSO(objective, bounds, n_particles=PSO_SWARM_P4,
              max_iter=PSO_ITER_P4, maximize=True, verbose=True)
    x_opt, f_opt = pso.optimize()

    # 解析结果
    theta = x_opt[0:3]
    speed = x_opt[3:6]
    release_times = x_opt[6:9]
    delays = x_opt[9:12]

    drone_names = ['FY1', 'FY2', 'FY3']

    print(f"\n优化结果:")
    for i in range(n_drones):
        direction = np.array([np.cos(theta[i]), np.sin(theta[i]), 0.0])
        release_pos = DRONES_INIT[i] + speed[i] * direction * release_times[i]
        detonation_pos = release_pos + speed[i] * direction * delays[i]
        detonation_pos[2] -= 0.5 * 9.8 * delays[i] ** 2

        print(f"\n  {drone_names[i]}:")
        print(f"    航向角θ: {theta[i]:.4f} rad ({np.degrees(theta[i]):.2f}°)")
        print(f"    飞行速度: {speed[i]:.2f} m/s")
        print(f"    投放时间: {release_times[i]:.4f} s")
        print(f"    起爆延时: {delays[i]:.4f} s")
        print(f"    投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
        print(f"    起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")

    print(f"\n  总有效遮蔽时长: {f_opt:.4f} s")

    # 保存到 result2.xlsx
    save_result2(theta, speed, release_times, delays, f_opt)

    return x_opt, f_opt


def save_result2(theta, speed, release_times, delays, total_time):
    """保存问题4结果到 result2.xlsx"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "问题4结果"

    headers = ["无人机", "航向角(rad)", "航向角(°)", "飞行速度(m/s)",
               "投放时间(s)", "起爆延时(s)",
               "投放点X", "投放点Y", "投放点Z",
               "起爆点X", "起爆点Y", "起爆点Z",
               "总有效遮蔽时长(s)"]
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)

    drone_names = ['FY1', 'FY2', 'FY3']
    for i in range(3):
        row = i + 2
        direction = np.array([np.cos(theta[i]), np.sin(theta[i]), 0.0])
        release_pos = DRONES_INIT[i] + speed[i] * direction * release_times[i]
        detonation_pos = release_pos + speed[i] * direction * delays[i]
        detonation_pos[2] -= 0.5 * 9.8 * delays[i] ** 2

        ws.cell(row=row, column=1, value=drone_names[i])
        ws.cell(row=row, column=2, value=round(theta[i], 6))
        ws.cell(row=row, column=3, value=round(np.degrees(theta[i]), 4))
        ws.cell(row=row, column=4, value=round(speed[i], 2))
        ws.cell(row=row, column=5, value=round(release_times[i], 4))
        ws.cell(row=row, column=6, value=round(delays[i], 4))
        ws.cell(row=row, column=7, value=round(release_pos[0], 2))
        ws.cell(row=row, column=8, value=round(release_pos[1], 2))
        ws.cell(row=row, column=9, value=round(release_pos[2], 2))
        ws.cell(row=row, column=10, value=round(detonation_pos[0], 2))
        ws.cell(row=row, column=11, value=round(detonation_pos[1], 2))
        ws.cell(row=row, column=12, value=round(detonation_pos[2], 2))
        if i == 0:
            ws.cell(row=row, column=13, value=round(total_time, 4))

    filepath = "result2.xlsx"
    wb.save(filepath)
    print(f"\n结果已保存至: {filepath}")


if __name__ == "__main__":
    solve_problem4()
