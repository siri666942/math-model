"""
问题2: 单机单弹最优策略 (1-1-1)
GA优化4个变量: [theta, v_FY, t_rel, t_lag]
目标: 最大化对M1的有效遮蔽时长
"""
import numpy as np
import time
import simulation as sim
from ga_solver import GeneticAlgorithm


def _generate_seeds_p2(config_module, ga_kps, ga_dt, n_seeds=30):
    """生成P2的种子个体 - 围绕P1参数区域扰动"""
    seeds = []
    np.random.seed(42)

    # P1参数作为基准
    base = [config_module.P1_DRONE_THETA, config_module.P1_DRONE_SPEED,
            config_module.P1_RELEASE_TIME, config_module.P1_DETONATION_DELAY]

    # 在基准附近大量采样
    for _ in range(n_seeds * 3):
        theta = base[0] + np.random.uniform(-0.1, 0.1)
        speed = base[1] + np.random.uniform(-20, 20)
        t_rel = base[2] + np.random.uniform(-1.0, 2.0)
        t_lag = base[3] + np.random.uniform(-1.0, 2.0)
        # 裁剪到边界
        theta = np.clip(theta, 0, 2*np.pi)
        speed = np.clip(speed, config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
        t_rel = np.clip(t_rel, 0, 15)
        t_lag = np.clip(t_lag, 0, 10)
        seeds.append([theta, speed, t_rel, t_lag])

    # 评估并筛选有希望的种子
    scored = []
    for s in seeds:
        hard = sim.simulate_single_bomb(
            config_module.DRONES_INIT[0], s[0], s[1], s[2], s[3],
            missile_idx=0, target_keypoints=ga_kps, dt=ga_dt,
        )
        if hard > 0:
            scored.append((hard, s))
        else:
            soft = sim.soft_score_single_bomb(
                config_module.DRONES_INIT[0], s[0], s[1], s[2], s[3],
                missile_idx=0, target_keypoints=ga_kps, dt=ga_dt,
            )
            if soft > 0:
                scored.append((soft * 0.01, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n_seeds]]


def solve_p2(config_module, pop_size=None, n_generations=None, verbose=True):
    sim.set_config(config_module)

    if pop_size is None:
        pop_size = config_module.GA_POP_P2
    if n_generations is None:
        n_generations = config_module.GA_GEN_P2

    ga_kps = sim.get_target_keypoints(config_module.GA_N_CIRCLE, config_module.GA_N_LAYERS)
    ga_dt = config_module.GA_DT

    if verbose:
        print(f"  问题2: GA优化 (pop={pop_size}, gen={n_generations})")
        print(f"  GA阶段: dt={ga_dt}, kps={ga_kps.shape[0]}")

    bounds = [
        (0.0, 2.0 * np.pi),
        (config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX),
        (0.0, 15.0),
        (0.0, 10.0),
    ]

    def fitness_func(x):
        theta, speed, t_rel, t_lag = x
        hard_time = sim.simulate_single_bomb(
            config_module.DRONES_INIT[0], theta, speed, t_rel, t_lag,
            missile_idx=0, target_keypoints=ga_kps, dt=ga_dt,
        )
        if hard_time > 0:
            return hard_time
        soft = sim.soft_score_single_bomb(
            config_module.DRONES_INIT[0], theta, speed, t_rel, t_lag,
            missile_idx=0, target_keypoints=ga_kps, dt=ga_dt,
        )
        return soft * 0.01

    # 生成种子
    n_seeds = min(pop_size // 3, 30)
    if verbose:
        print(f"  生成 {n_seeds} 个种子个体...")
    seeds = _generate_seeds_p2(config_module, ga_kps, ga_dt, n_seeds)
    if verbose:
        print(f"  找到 {len(seeds)} 个有希望的种子")

    ga = GeneticAlgorithm(
        fitness_func=fitness_func, bounds=bounds,
        pop_size=pop_size, n_generations=n_generations,
        crossover_rate=config_module.GA_CROSSOVER_RATE,
        mutation_rate=config_module.GA_MUTATION_RATE,
        verbose=verbose,
    )

    t_start = time.time()
    best_solution, best_fitness = ga.run(seeds=seeds if seeds else None)
    ga_time = time.time() - t_start

    theta, speed, t_rel, t_lag = best_solution

    # 精细验证
    fine_kps = sim.get_target_keypoints(config_module.FINE_N_CIRCLE, config_module.FINE_N_LAYERS)
    fine_dt = config_module.FINE_DT
    fine_time = sim.simulate_single_bomb(
        config_module.DRONES_INIT[0], theta, speed, t_rel, t_lag,
        missile_idx=0, target_keypoints=fine_kps, dt=fine_dt,
    )

    if verbose:
        print(f"\n  最优解 (精细验证: dt={fine_dt}, kps={fine_kps.shape[0]}):")
        print(f"    theta = {theta:.4f} rad ({np.degrees(theta):.2f} deg)")
        print(f"    speed = {speed:.2f} m/s")
        print(f"    t_rel = {t_rel:.4f} s, t_lag = {t_lag:.4f} s")
        print(f"    起爆时间 = {t_rel + t_lag:.4f} s")
        print(f"    有效遮蔽时长(精细) = {fine_time:.4f} s")
        print(f"    GA耗时: {ga_time:.1f}s")

    return theta, speed, t_rel, t_lag, fine_time
