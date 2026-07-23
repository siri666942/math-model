"""
问题5: 5架无人机，每架至多投放3枚烟幕干扰弹，实施对M1、M2、M3的干扰
输出结果到 result3.xlsx

优化变量(40维):
- theta(5): 各无人机航向角
- speed(5): 各无人机飞行速度
- release_times(5×3=15): 各弹投放时间(编码为间隔)
- detonation_delays(5×3=15): 各弹起爆延时

采用分步优化: 先单机优化，再联合微调
"""
import numpy as np
import openpyxl
from config import (
    DRONES_INIT, DRONE_SPEED_MIN, DRONE_SPEED_MAX,
    BOMB_INTERVAL_MIN, INTERCEPT_ORDER, DT, T_TOTAL,
)
from simulation import simulate_multi_drone_multi_bomb, get_target_keypoints
from pso import PSO, local_polish

PSO_SWARM_P5 = 300
PSO_ITER_P5 = 200

N_DRONES = 5
N_BOMBS_PER_DRONE = 3
N_MISSILES = 3


class SingleDroneObjective:
    """阶段1: 单机独立优化用的目标函数对象，模块级可pickle，供PSO多进程worker调用"""

    def __init__(self, drone_idx, missile_order):
        self.drone_idx = drone_idx
        self.missile_order = missile_order

    def __call__(self, x):
        theta, speed = x[0], x[1]
        release_times = np.array([x[2], x[2] + x[3], x[2] + x[3] + x[4]])
        delays = np.array([x[5], x[6], x[7]])

        drone_params = [{
            'drone_init': DRONES_INIT[self.drone_idx],
            'theta': theta,
            'speed': speed,
            'release_times': release_times,
            'detonation_delays': delays,
            'missile_indices': self.missile_order,
        }]
        total_time, _ = simulate_multi_drone_multi_bomb(
            drone_params, dt=DT, t_total=T_TOTAL
        )
        return total_time


def _joint_objective(x):
    """阶段2: 40维联合微调用的目标函数，不捕获任何局部变量，模块级可pickle"""
    idx = 0
    drone_params = []
    for drone_idx in range(N_DRONES):
        theta = x[idx]; idx += 1
        speed = x[idx]; idx += 1

        release1 = x[idx]; idx += 1
        int2 = x[idx]; idx += 1
        int3 = x[idx]; idx += 1
        release_times = np.array([release1, release1 + int2, release1 + int2 + int3])

        delays = np.array([x[idx + j] for j in range(3)]); idx += 3
        order = INTERCEPT_ORDER[['FY1', 'FY2', 'FY3', 'FY4', 'FY5'][drone_idx]]

        drone_params.append({
            'drone_init': DRONES_INIT[drone_idx],
            'theta': theta,
            'speed': speed,
            'release_times': release_times,
            'detonation_delays': delays,
            'missile_indices': order,
        })

    total_time, _ = simulate_multi_drone_multi_bomb(
        drone_params, dt=DT, t_total=T_TOTAL
    )
    return total_time


def solve_problem5():
    """求解问题5: 五机多弹多导弹协同策略"""
    print("=" * 60)
    print("问题5: 5架无人机协同投放烟幕弹 (PSO分步优化)")
    print("=" * 60)

    # 预定义每架无人机的最佳搜索范围
    # 基于初始位置分析
    # theta_range 以各无人机指向真目标(0,200,0)的方位角为中心留出搜索余量
    # (原范围都取在0°附近的小角度，方向朝向战场正前方而非目标，PSO永远搜不到有效遮蔽)
    drone_configs = [
        {  # FY1: 目标方位约179.4°(3.13rad)
            'theta_range': (2.73, 3.53),
            'speed_range': (DRONE_SPEED_MIN, DRONE_SPEED_MAX),
            'release_range': (0.0, 5.0),
            'delay_range': (0.0, 8.0),
        },
        {  # FY2: 目标方位约-174.3°(-3.04rad)
            'theta_range': (-3.44, -2.64),
            'speed_range': (DRONE_SPEED_MIN, DRONE_SPEED_MAX),
            'release_range': (0.0, 8.0),
            'delay_range': (0.0, 10.0),
        },
        {  # FY3: 目标方位约151.9°(2.65rad)
            'theta_range': (2.25, 3.05),
            'speed_range': (DRONE_SPEED_MIN, DRONE_SPEED_MAX),
            'release_range': (0.0, 12.0),
            'delay_range': (0.0, 12.0),
        },
        {  # FY4: 目标方位约-170.7°(-2.98rad)
            'theta_range': (-3.38, -2.58),
            'speed_range': (DRONE_SPEED_MIN, DRONE_SPEED_MAX),
            'release_range': (0.0, 15.0),
            'delay_range': (0.0, 15.0),
        },
        {  # FY5: 目标方位约170.4°(2.97rad)
            'theta_range': (2.57, 3.37),
            'speed_range': (DRONE_SPEED_MIN, DRONE_SPEED_MAX),
            'release_range': (0.0, 15.0),
            'delay_range': (0.0, 15.0),
        },
    ]

    # ============================================================
    # 阶段1: 逐架无人机单独优化
    # ============================================================
    print("\n阶段1: 逐架无人机单独优化...")

    single_results = []
    for drone_idx in range(N_DRONES):
        cfg = drone_configs[drone_idx]
        print(f"\n--- 优化 {['FY1','FY2','FY3','FY4','FY5'][drone_idx]} ---")

        # 构建变量: [theta, speed, release1, int2, int3, delay1, delay2, delay3] = 8维
        bounds = [
            cfg['theta_range'],
            cfg['speed_range'],
            cfg['release_range'],
            (BOMB_INTERVAL_MIN, 5.0),   # interval 1->2
            (BOMB_INTERVAL_MIN, 5.0),   # interval 2->3
            cfg['delay_range'],
            cfg['delay_range'],
            cfg['delay_range'],
        ]

        intercept_order = INTERCEPT_ORDER[['FY1','FY2','FY3','FY4','FY5'][drone_idx]]

        obj_func = SingleDroneObjective(drone_idx, intercept_order)
        pso = PSO(obj_func, bounds, n_particles=150, max_iter=80,
                  maximize=True, verbose=False)
        x_opt, f_opt = pso.optimize()

        theta = x_opt[0]
        speed = x_opt[1]
        release_times = np.array([x_opt[2], x_opt[2]+x_opt[3], x_opt[2]+x_opt[3]+x_opt[4]])
        delays = np.array([x_opt[5], x_opt[6], x_opt[7]])

        single_results.append({
            'theta': theta, 'speed': speed,
            'release_times': release_times, 'delays': delays,
            'time': f_opt,
        })
        print(f"  单机最优遮蔽时长: {f_opt:.4f} s")

    total_single = sum(r['time'] for r in single_results)
    print(f"\n阶段1 单机优化总时长(简单求和): {total_single:.4f} s")

    # ============================================================
    # 阶段2: 联合局部微调 (缩小搜索范围)
    # ============================================================
    print("\n阶段2: 联合局部微调...")

    # 构建40维变量，搜索范围以阶段1结果为中心
    bounds_joint = []
    for drone_idx in range(N_DRONES):
        r = single_results[drone_idx]
        cfg = drone_configs[drone_idx]
        delta = 0.1

        # theta
        t_center = r['theta']
        t_range = max(0.1, abs(cfg['theta_range'][1] - cfg['theta_range'][0]) * delta)
        bounds_joint.append((
            max(cfg['theta_range'][0], t_center - t_range),
            min(cfg['theta_range'][1], t_center + t_range)
        ))
        # speed
        s_center = r['speed']
        s_range = (cfg['speed_range'][1] - cfg['speed_range'][0]) * delta
        bounds_joint.append((
            max(cfg['speed_range'][0], s_center - s_range),
            min(cfg['speed_range'][1], s_center + s_range)
        ))

    # release times (15个) 和 delays (15个)
    for drone_idx in range(N_DRONES):
        r = single_results[drone_idx]
        cfg = drone_configs[drone_idx]
        delta = 0.2
        # release1
        bounds_joint.append((
            max(0.0, r['release_times'][0] - delta),
            r['release_times'][0] + delta
        ))
        # interval 1->2
        int_center = r['release_times'][1] - r['release_times'][0]
        bounds_joint.append((
            max(BOMB_INTERVAL_MIN, int_center - delta),
            int_center + delta
        ))
        # interval 2->3
        int_center2 = r['release_times'][2] - r['release_times'][1]
        bounds_joint.append((
            max(BOMB_INTERVAL_MIN, int_center2 - delta),
            int_center2 + delta
        ))

    for drone_idx in range(N_DRONES):
        r = single_results[drone_idx]
        cfg = drone_configs[drone_idx]
        delta = 1.0
        for j in range(3):
            d_center = r['delays'][j]
            bounds_joint.append((
                max(cfg['delay_range'][0], d_center - delta),
                min(cfg['delay_range'][1], d_center + delta)
            ))

    print(f"联合优化变量维度: {len(bounds_joint)}")
    pso_joint = PSO(_joint_objective, bounds_joint, n_particles=PSO_SWARM_P5,
                    max_iter=PSO_ITER_P5, maximize=True, verbose=True)
    x_opt_joint, f_opt_joint = pso_joint.optimize()

    print("\n局部精修(Powell)...")
    x_polished, f_polished = local_polish(_joint_objective, x_opt_joint, bounds_joint)
    if f_polished > f_opt_joint:
        print(f"  精修有提升: {f_opt_joint:.4f}s -> {f_polished:.4f}s")
        x_opt_joint, f_opt_joint = x_polished, f_polished
    else:
        print(f"  精修没有提升({f_polished:.4f}s <= {f_opt_joint:.4f}s)，保留PSO原结果")

    print(f"\n联合优化总有效遮蔽时长: {f_opt_joint:.4f} s")

    # ============================================================
    # 解析并输出结果
    # ============================================================
    idx = 0
    final_results = []
    for drone_idx in range(N_DRONES):
        theta = x_opt_joint[idx]; idx += 1
        speed = x_opt_joint[idx]; idx += 1
        release1 = x_opt_joint[idx]; idx += 1
        int2 = x_opt_joint[idx]; idx += 1
        int3 = x_opt_joint[idx]; idx += 1
        release_times = np.array([release1, release1+int2, release1+int2+int3])
        delays = np.array([x_opt_joint[idx+j] for j in range(3)]); idx += 3
        order = INTERCEPT_ORDER[['FY1','FY2','FY3','FY4','FY5'][drone_idx]]
        final_results.append({
            'theta': theta, 'speed': speed,
            'release_times': release_times, 'delays': delays,
            'order': order,
        })

    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    print(f"\n{'='*60}")
    print("最终优化结果:")
    print(f"{'='*60}")

    for i, r in enumerate(final_results):
        direction = np.array([np.cos(r['theta']), np.sin(r['theta']), 0.0])
        print(f"\n{drone_names[i]}: θ={r['theta']:.4f}rad({np.degrees(r['theta']):.1f}°), "
              f"v={r['speed']:.2f}m/s")
        print(f"  拦截顺序: {r['order']}")
        for j in range(3):
            release_pos = DRONES_INIT[i] + r['speed'] * direction * r['release_times'][j]
            detonation_pos = release_pos + r['speed'] * direction * r['delays'][j]
            detonation_pos[2] -= 0.5 * 9.8 * r['delays'][j] ** 2
            print(f"  弹{j+1}: 投放t={r['release_times'][j]:.4f}s, 延时={r['delays'][j]:.4f}s, "
                  f"起爆点=({detonation_pos[0]:.1f},{detonation_pos[1]:.1f},{detonation_pos[2]:.1f})")

    print(f"\n总有效遮蔽时长: {f_opt_joint:.4f} s")

    # 保存
    save_result3(final_results, f_opt_joint)

    return final_results, f_opt_joint


def save_result3(final_results, total_time):
    """保存问题5结果到 result3.xlsx"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "问题5结果"

    headers = ["无人机", "航向角(rad)", "航向角(°)", "飞行速度(m/s)",
               "烟幕弹编号", "目标导弹", "投放时间(s)", "起爆延时(s)",
               "投放点X", "投放点Y", "投放点Z",
               "起爆点X", "起爆点Y", "起爆点Z",
               "总有效遮蔽时长(s)"]
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)

    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    row = 2
    for i, r in enumerate(final_results):
        direction = np.array([np.cos(r['theta']), np.sin(r['theta']), 0.0])
        for j in range(3):
            missile_label = f"M{r['order'][j]+1}"
            release_pos = DRONES_INIT[i] + r['speed'] * direction * r['release_times'][j]
            detonation_pos = release_pos + r['speed'] * direction * r['delays'][j]
            detonation_pos[2] -= 0.5 * 9.8 * r['delays'][j] ** 2

            ws.cell(row=row, column=1, value=drone_names[i])
            ws.cell(row=row, column=2, value=round(r['theta'], 6))
            ws.cell(row=row, column=3, value=round(np.degrees(r['theta']), 4))
            ws.cell(row=row, column=4, value=round(r['speed'], 2))
            ws.cell(row=row, column=5, value=j + 1)
            ws.cell(row=row, column=6, value=missile_label)
            ws.cell(row=row, column=7, value=round(r['release_times'][j], 4))
            ws.cell(row=row, column=8, value=round(r['delays'][j], 4))
            ws.cell(row=row, column=9, value=round(release_pos[0], 2))
            ws.cell(row=row, column=10, value=round(release_pos[1], 2))
            ws.cell(row=row, column=11, value=round(release_pos[2], 2))
            ws.cell(row=row, column=12, value=round(detonation_pos[0], 2))
            ws.cell(row=row, column=13, value=round(detonation_pos[1], 2))
            ws.cell(row=row, column=14, value=round(detonation_pos[2], 2))
            if i == 0 and j == 0:
                ws.cell(row=row, column=15, value=round(total_time, 4))
            row += 1

    filepath = "result3.xlsx"
    wb.save(filepath)
    print(f"\n结果已保存至: {filepath}")


if __name__ == "__main__":
    solve_problem5()
