"""
基于变步长网格搜索的优化策略（对应论文的变步长搜索方法）

核心思路：在可行的起爆位置3D区域内进行网格搜索，找到遮蔽时间最长的位置，
然后反推无人机飞行参数。
"""
import numpy as np
import config as cfg
from simulation import simulate_single_bomb, get_target_keypoints


def gs_optimize_problem2(drone_init, n_coarse=15, refine_steps=3):
    """
    对问题2使用变步长网格搜索

    策略:
    1. 在可行起爆位置3D空间内粗网格搜索
    2. 选取前K个最优位置
    3. 在K个最优位置邻域内细化搜索
    4. 迭代直至步长足够小
    """
    kp = get_target_keypoints(360, 0)

    # M1 x方向速度分量
    M1_vx = cfg.MISSILE_SPEED * abs(cfg.MISSILES_DIR[0][0])  # ≈ 298.5 m/s
    M1_start_x = cfg.MISSILES_INIT[0][0]  # 20000

    drone_x, drone_y, drone_z = drone_init
    v_min, v_max = cfg.DRONE_SPEED_MIN, cfg.DRONE_SPEED_MAX  # 80, 120

    # 可行起爆区域范围
    # x: 起爆位置在导弹路径上（从M1初始位置到原点之间）
    # y: 考虑到目标在y=200，起爆位置应有正y偏移
    # z: 起爆高度在0到无人机高度之间
    x_range = [12000, 17800]
    y_range = [-200, 600]
    z_range = [200, 1700]

    best_overall_val = 0
    best_overall_params = None

    step_sizes = [400, 100, 25, 5]  # 逐级细化的步长

    for step_idx, step in enumerate(step_sizes):
        print(f"\n--- 步长级别 {step_idx+1}: step={step}m ---")

        if step_idx == 0:
            # 首次: 在整个区域内搜索
            candidates = []
            xs = np.arange(x_range[0], x_range[1], step)
            ys = np.arange(y_range[0], y_range[1], step)
            zs = np.arange(z_range[0], z_range[1], step)
        else:
            # 后续: 在最优候选周围搜索
            xs_list, ys_list, zs_list = [], [], []
            for prev_cand in prev_best:
                cx, cy, cz = prev_cand['pos']
                half = step * 2.5  # 搜索范围
                for x in np.arange(max(x_range[0], cx-half), min(x_range[1], cx+half), step):
                    for y in np.arange(max(y_range[0], cy-half), min(y_range[1], cy+half), step):
                        for z in np.arange(max(z_range[0], cz-half), min(z_range[1], cz+half), step):
                            xs_list.append(x)
                            ys_list.append(y)
                            zs_list.append(z)
            xs = np.unique(xs_list)
            ys = np.unique(ys_list)
            zs = np.unique(zs_list)

        print(f"  搜索网格: {len(xs)}x{len(ys)}x{len(zs)} = {len(xs)*len(ys)*len(zs)} 点")

        candidates_this_level = []
        total = len(xs) * len(ys) * len(zs)
        count = 0

        for x in xs:
            for y in ys:
                for z in zs:
                    count += 1
                    if count % 5000 == 0:
                        print(f"    进度: {count}/{total}")

                    # === 可行性检查 ===
                    # 1. 无人机能否在导弹到达前将烟幕弹送到此位置？
                    drone_to_point_h = np.sqrt((x - drone_x)**2 + (y - drone_y)**2)
                    if drone_to_point_h < 1e-6:
                        continue

                    if z >= drone_z - 1:  # 无法起爆在无人机高度以上
                        continue

                    # 计算起爆延时 (从烟幕弹下落到指定z高度)
                    fall_dist = drone_z - z  # 需要下落的高度
                    if fall_dist <= 0:
                        continue
                    td = np.sqrt(2 * fall_dist / cfg.G)  # 自由落体时间

                    # 水平飞行距离 = drone_to_point_h
                    # 飞行时间 = tr + td
                    # v * (tr + td) = drone_to_point_h
                    # tr = drone_to_point_h / v - td

                    # 尝试不同速度
                    feasible = False
                    best_tr = None
                    best_v = None
                    best_theta = None

                    for v_test in [v_max, (v_min + v_max) / 2, v_min]:
                        tr = drone_to_point_h / v_test - td
                        if tr >= 0:
                            total_time = tr + td
                            # 导弹到达x的时间
                            M1_arrival = (M1_start_x - x) / M1_vx
                            if M1_arrival <= 0:
                                continue
                            if total_time <= M1_arrival:  # 无人机能在导弹到达前完成部署
                                feasible = True
                                best_tr = tr
                                best_v = v_test
                                break

                    if not feasible:
                        continue

                    # 计算无人机航向
                    theta = np.arctan2(y - drone_y, x - drone_x)

                    # === 遮蔽效果快速评估 ===
                    val = simulate_single_bomb(
                        drone_init, theta, best_v, best_tr, td,
                        0, kp, cfg.DT, 30.0
                    )

                    if val > 0.01:  # 有效遮蔽
                        candidates_this_level.append({
                            'pos': (x, y, z),
                            'val': val,
                            'theta': theta,
                            'speed': best_v,
                            'release_time': best_tr,
                            'delay': td,
                        })

                        if val > best_overall_val:
                            best_overall_val = val
                            best_overall_params = {
                                'pos': (x, y, z),
                                'theta': theta,
                                'speed': best_v,
                                'release_time': best_tr,
                                'delay': td,
                            }
                            print(f"  *** 新最优: {val:.4f}s at ({x:.0f},{y:.0f},{z:.0f}) "
                                  f"θ={np.degrees(theta):.1f}° v={best_v:.1f}m/s")

        if not candidates_this_level:
            print(f"  未找到有效候选，扩大搜索范围...")
            # 如果当前级别找不到，使用上一级别的最佳候选
            if step_idx > 0 and prev_best:
                candidates_this_level = prev_best
            else:
                # 扩大y范围
                y_range = [-500, 1000]
                continue

        # 保留前K个最优
        candidates_this_level.sort(key=lambda c: c['val'], reverse=True)
        prev_best = candidates_this_level[:min(9, len(candidates_this_level))]

        print(f"  找到 {len(candidates_this_level)} 个有效候选, "
              f"最优: {prev_best[0]['val']:.4f}s")

    if best_overall_params is None:
        print("WARNING: 未找到任何有效解!")
        return None, 0

    return best_overall_params, best_overall_val


def gs_optimize_problem4(drones_to_use=[0, 1, 2]):
    """
    对问题4: 逐架无人机分别优化，然后合并结果
    """
    results = []
    names = ['FY1', 'FY2', 'FY3']

    for idx in drones_to_use:
        print(f"\n{'='*50}")
        print(f"优化 {names[idx]} (初始位置: {cfg.DRONES_INIT[idx]})")
        print(f"{'='*50}")

        params, val = gs_optimize_problem2(cfg.DRONES_INIT[idx])
        if params:
            results.append({**params, 'drone_idx': idx, 'name': names[idx]})
            print(f"\n{names[idx]} 最优: {val:.4f}s")

    return results
