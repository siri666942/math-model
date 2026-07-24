"""
问题4: 三机协同单弹 (3-1-1)
策略: 每架无人机独立GA优化 (分解策略)
增强种子生成: 对每个导弹使用适当的搜索方向
"""
import numpy as np
import time
import simulation as sim
from ga_solver import GeneticAlgorithm


def _compute_missile_target_direction(missile_idx, t=5.0):
    """在时刻t，导弹到目标的方向向量"""
    M_pos = sim.MISSILES_INIT[missile_idx] + sim.MISSILE_SPEED * sim.MISSILES_DIR[missile_idx] * t
    target = sim.TARGET_CENTER
    vec = target - M_pos
    return vec / np.linalg.norm(vec)


def _generate_seeds_for_drone_missile(config_module, drone_idx, missile_idx, ga_kps, ga_dt,
                                       n_seeds=50):
    """为指定无人机-导弹对生成种子"""
    seeds = []
    np.random.seed(42 + drone_idx * 10 + missile_idx)

    drone_init = config_module.DRONES_INIT[drone_idx]

    # 策略1: 围绕P1参数采样 (对M1特别有效)
    for _ in range(n_seeds // 3):
        theta = config_module.P1_DRONE_THETA + np.random.uniform(-0.3, 0.3)
        speed = config_module.P1_DRONE_SPEED + np.random.uniform(-40, 30)
        t_rel = np.random.uniform(0, 12)
        t_lag = np.random.uniform(0.5, 8)
        theta = np.clip(theta, 0, 2*np.pi)
        speed = np.clip(speed, config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
        t_rel = np.clip(t_rel, 0, 15)
        t_lag = np.clip(t_lag, 0, 10)
        seeds.append([theta, speed, t_rel, t_lag])

    # 策略2: 计算导弹在t=5s和t=10s到目标的方向，围绕这些方向采样
    for t_ref in [5.0, 10.0, 15.0]:
        M_pos = sim.MISSILES_INIT[missile_idx] + sim.MISSILE_SPEED * sim.MISSILES_DIR[missile_idx] * t_ref
        target_dir = (sim.TARGET_CENTER - M_pos)
        target_dir = target_dir / np.linalg.norm(target_dir)

        # 无人机飞向目标方向的角度
        target_theta = np.arctan2(target_dir[1], target_dir[0])

        for _ in range(n_seeds // 6):
            theta = target_theta + np.random.uniform(-0.2, 0.2)
            speed = np.random.uniform(config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
            t_rel = np.random.uniform(0, 12)
            t_lag = np.random.uniform(0.5, 8)
            theta = np.clip(theta, 0, 2*np.pi)
            speed = np.clip(speed, config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
            t_rel = np.clip(t_rel, 0, 15)
            t_lag = np.clip(t_lag, 0, 10)
            seeds.append([theta, speed, t_rel, t_lag])

    # 策略3: 完全随机 (全覆盖)
    for _ in range(n_seeds // 3):
        theta = np.random.uniform(0, 2*np.pi)
        speed = np.random.uniform(config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
        t_rel = np.random.uniform(0, 12)
        t_lag = np.random.uniform(0.5, 8)
        seeds.append([theta, speed, t_rel, t_lag])

    # 评估筛选
    scored = []
    for s in seeds:
        hard = sim.simulate_single_bomb(
            drone_init, s[0], s[1], s[2], s[3],
            missile_idx=missile_idx, target_keypoints=ga_kps, dt=ga_dt,
        )
        if hard > 0:
            scored.append((hard, s))
        else:
            soft = sim.soft_score_single_bomb(
                drone_init, s[0], s[1], s[2], s[3],
                missile_idx=missile_idx, target_keypoints=ga_kps, dt=ga_dt,
            )
            if soft > 0:
                scored.append((soft * 0.01, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n_seeds]]


def _optimize_single_drone(config_module, drone_idx, missile_idx, pop_size, n_generations, verbose):
    """优化单架无人机"""
    ga_kps = sim.get_target_keypoints(config_module.GA_N_CIRCLE, config_module.GA_N_LAYERS)
    ga_dt = config_module.GA_DT

    seeds = _generate_seeds_for_drone_missile(
        config_module, drone_idx, missile_idx, ga_kps, ga_dt,
        n_seeds=max(pop_size//3, 30))

    if verbose:
        print(f"    FY{drone_idx+1}->M{missile_idx+1}: {len(seeds)} seeds, "
              f"pop={pop_size//2}, gen={n_generations//2}")

    bounds = [
        (0.0, 2.0 * np.pi),
        (config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX),
        (0.0, 15.0), (0.0, 10.0),
    ]

    def fitness_func(x):
        hard = sim.simulate_single_bomb(
            config_module.DRONES_INIT[drone_idx], x[0], x[1], x[2], x[3],
            missile_idx=missile_idx, target_keypoints=ga_kps, dt=ga_dt,
        )
        if hard > 0: return hard
        soft = sim.soft_score_single_bomb(
            config_module.DRONES_INIT[drone_idx], x[0], x[1], x[2], x[3],
            missile_idx=missile_idx, target_keypoints=ga_kps, dt=ga_dt,
        )
        return soft * 0.01

    ga = GeneticAlgorithm(
        fitness_func=fitness_func, bounds=bounds,
        pop_size=max(pop_size//2, 40), n_generations=max(n_generations//2, 20),
        crossover_rate=config_module.GA_CROSSOVER_RATE,
        mutation_rate=0.15,
        verbose=False,
    )
    best_solution, _ = ga.run(seeds=seeds if seeds else None)

    return {
        'drone': f'FY{drone_idx+1}', 'missile': f'M{missile_idx+1}',
        'theta': best_solution[0], 'speed': best_solution[1],
        't_rel': best_solution[2], 't_lag': best_solution[3],
    }


def solve_p4(config_module, pop_size=None, n_generations=None, verbose=True):
    sim.set_config(config_module)
    if pop_size is None: pop_size = config_module.GA_POP_P4
    if n_generations is None: n_generations = config_module.GA_GEN_P4

    if verbose:
        print(f"  问题4: 分解策略 - 每架无人机独立优化")

    results = []
    t_start = time.time()
    for i in range(3):
        r = _optimize_single_drone(config_module, i, i, pop_size, n_generations, verbose)
        results.append(r)

    fine_kps = sim.get_target_keypoints(config_module.FINE_N_CIRCLE, config_module.FINE_N_LAYERS)
    fine_dt = config_module.FINE_DT
    n_steps_fine = int(np.ceil(config_module.T_TOTAL / fine_dt))
    pm = [np.zeros(n_steps_fine, dtype=bool) for _ in range(3)]
    per_missile_times = []
    for i in range(3):
        mask = sim._bomb_coverage_mask(
            config_module.DRONES_INIT[i], results[i]['theta'], results[i]['speed'],
            results[i]['t_rel'], results[i]['t_lag'], i,
            fine_kps, fine_dt, config_module.T_TOTAL)
        pm[i] |= mask
        per_missile_times.append(mask.sum() * fine_dt)
    fine_time = sum(per_missile_times)
    ga_time = time.time() - t_start

    if verbose:
        print(f"\n  最优解 (精细验证):")
        for i, r in enumerate(results):
            print(f"    {r['drone']}->{r['missile']}: theta={r['theta']:.4f}rad "
                  f"({np.degrees(r['theta']):.1f}deg), v={r['speed']:.2f}m/s, "
                  f"t_rel={r['t_rel']:.4f}s, t_lag={r['t_lag']:.4f}s "
                  f"= {per_missile_times[i]:.4f}s")
        print(f"    总有效遮蔽时长 = {fine_time:.4f} s, 耗时: {ga_time:.1f}s")

    return results, fine_time
