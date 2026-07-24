"""
问题3: 单机三弹最优策略 (1-3-1)
GA优化8个变量
约束: 相邻弹投放间隔 >= 1s
"""
import numpy as np
import time
import simulation as sim
from ga_solver import GeneticAlgorithm, make_sequential_interval_constraint


def _generate_seeds_p3(config_module, ga_kps, ga_dt, n_seeds=30):
    """生成P3种子 - 围绕P1参数，3弹均匀分布"""
    seeds = []
    np.random.seed(42)
    for _ in range(n_seeds * 5):
        theta = config_module.P1_DRONE_THETA + np.random.uniform(-0.15, 0.15)
        speed = config_module.P1_DRONE_SPEED + np.random.uniform(-30, 20)
        t_rel1 = config_module.P1_RELEASE_TIME + np.random.uniform(-1, 3)
        t_lag1 = config_module.P1_DETONATION_DELAY + np.random.uniform(-1, 3)
        t_rel2 = t_rel1 + 1.0 + np.random.uniform(0, 3)
        t_lag2 = config_module.P1_DETONATION_DELAY + np.random.uniform(-1, 3)
        t_rel3 = t_rel2 + 1.0 + np.random.uniform(0, 3)
        t_lag3 = config_module.P1_DETONATION_DELAY + np.random.uniform(-1, 3)

        theta = np.clip(theta, 0, 2*np.pi)
        speed = np.clip(speed, config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
        t_rel1 = np.clip(t_rel1, 0, 15); t_lag1 = np.clip(t_lag1, 0, 10)
        t_rel2 = np.clip(t_rel2, 1, 15); t_lag2 = np.clip(t_lag2, 0, 10)
        t_rel3 = np.clip(t_rel3, 2, 15); t_lag3 = np.clip(t_lag3, 0, 10)

        seeds.append([theta, speed, t_rel1, t_lag1, t_rel2, t_lag2, t_rel3, t_lag3])

    scored = []
    for s in seeds[:n_seeds * 3]:
        hard = sim.simulate_multi_bomb_single_drone(
            config_module.DRONES_INIT[0], s[0], s[1],
            [s[2], s[4], s[6]], [s[3], s[5], s[7]],
            [0, 0, 0], target_keypoints=ga_kps, dt=ga_dt,
        )
        scored.append((hard, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n_seeds]]


def solve_p3(config_module, pop_size=None, n_generations=None, verbose=True):
    sim.set_config(config_module)
    if pop_size is None: pop_size = config_module.GA_POP_P3
    if n_generations is None: n_generations = config_module.GA_GEN_P3

    ga_kps = sim.get_target_keypoints(config_module.GA_N_CIRCLE, config_module.GA_N_LAYERS)
    ga_dt = config_module.GA_DT

    if verbose:
        print(f"  问题3: GA优化 (pop={pop_size}, gen={n_generations})")
        print(f"  GA阶段: dt={ga_dt}, kps={ga_kps.shape[0]}")

    bounds = [
        (0.0, 2.0 * np.pi),
        (config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX),
        (0.0, 15.0), (0.0, 10.0),
        (1.0, 15.0), (0.0, 10.0),
        (2.0, 15.0), (0.0, 10.0),
    ]
    constraint = make_sequential_interval_constraint([2, 4, 6], min_interval=1.0)

    def fitness_func(x):
        return sim.simulate_multi_bomb_single_drone(
            config_module.DRONES_INIT[0], x[0], x[1],
            [x[2], x[4], x[6]], [x[3], x[5], x[7]],
            [0, 0, 0], target_keypoints=ga_kps, dt=ga_dt,
        )

    n_seeds = min(pop_size // 3, 30)
    seeds = _generate_seeds_p3(config_module, ga_kps, ga_dt, n_seeds)
    if verbose: print(f"  找到 {len(seeds)} 个种子")

    ga = GeneticAlgorithm(
        fitness_func=fitness_func, bounds=bounds,
        pop_size=pop_size, n_generations=n_generations,
        crossover_rate=config_module.GA_CROSSOVER_RATE,
        mutation_rate=config_module.GA_MUTATION_RATE,
        constraint_func=constraint, verbose=verbose,
    )

    t_start = time.time()
    best_solution, best_fitness = ga.run(seeds=seeds if seeds else None)
    ga_time = time.time() - t_start

    theta, speed = best_solution[0], best_solution[1]
    t_rels = [best_solution[2], best_solution[4], best_solution[6]]
    t_lags = [best_solution[3], best_solution[5], best_solution[7]]

    fine_kps = sim.get_target_keypoints(config_module.FINE_N_CIRCLE, config_module.FINE_N_LAYERS)
    fine_time = sim.simulate_multi_bomb_single_drone(
        config_module.DRONES_INIT[0], theta, speed, t_rels, t_lags,
        [0, 0, 0], target_keypoints=fine_kps, dt=config_module.FINE_DT,
    )

    if verbose:
        print(f"\n  最优解 (精细验证: dt={config_module.FINE_DT}, kps={fine_kps.shape[0]}):")
        print(f"    theta = {theta:.4f} rad ({np.degrees(theta):.2f} deg), speed = {speed:.2f} m/s")
        for i in range(3):
            print(f"    弹{i+1}: t_rel={t_rels[i]:.4f}s, t_lag={t_lags[i]:.4f}s, "
                  f"起爆={t_rels[i]+t_lags[i]:.4f}s")
        print(f"    有效遮蔽时长(精细) = {fine_time:.4f} s, GA耗时: {ga_time:.1f}s")

    return theta, speed, t_rels, t_lags, fine_time
