"""
问题5: 五机三弹协同 (5-3-3)
分解策略: 每架无人机独立优化3弹参数
"""
import numpy as np
import time
import simulation as sim
from ga_solver import GeneticAlgorithm, make_sequential_interval_constraint


MISSILE_ASSIGNMENT = [
    [0, 0, 0], [1, 0, 2], [2, 0, 1], [1, 0, 2], [2, 0, 1],
]


def _generate_seeds_p5_drone(config_module, drone_idx, missile_indices, ga_kps, ga_dt, n_seeds=40):
    """为单架无人机生成3弹种子"""
    seeds = []
    np.random.seed(42 + drone_idx * 100)

    # 多种策略生成候选
    for _ in range(n_seeds * 5):
        candidate = []
        for bomb_j in range(3):
            mi = missile_indices[bomb_j]
            # 基于目标导弹方向
            M_pos_5s = sim.MISSILES_INIT[mi] + sim.MISSILE_SPEED * sim.MISSILES_DIR[mi] * 5.0
            target_dir = (sim.TARGET_CENTER - M_pos_5s)
            target_dir = target_dir / np.linalg.norm(target_dir)
            theta_ref = np.arctan2(target_dir[1], target_dir[0])

            theta = theta_ref + np.random.uniform(-0.3, 0.3)
            speed = np.random.uniform(config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
            t_rel = np.random.uniform(0, 12)
            t_lag = np.random.uniform(0.5, 8)
            candidate += [theta, speed, t_rel, t_lag]

        # 确保约束 (t_rel 递增)
        for bj in range(1, 3):
            t_rel_idx = bj * 4 + 2  # 每个bomb block: theta(0), speed(1), t_rel(2), t_lag(3)
            prev_t_rel = candidate[(bj-1)*4 + 2]
            if candidate[t_rel_idx] < prev_t_rel + 1.0:
                candidate[t_rel_idx] = prev_t_rel + 1.0 + np.random.uniform(0, 2)

        # 裁剪
        for idx in range(0, 12, 4):
            candidate[idx] = np.clip(candidate[idx], 0, 2*np.pi)  # theta
            candidate[idx+1] = np.clip(candidate[idx+1], config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX)
            candidate[idx+2] = np.clip(candidate[idx+2], 0, 15)
            candidate[idx+3] = np.clip(candidate[idx+3], 0, 10)

        seeds.append(candidate)

    # 评估筛选
    scored = []
    for s in seeds[:n_seeds * 3]:
        total_time = 0.0
        for bj in range(3):
            mi = missile_indices[bj]
            bi = bj * 4
            hard = sim.simulate_single_bomb(
                config_module.DRONES_INIT[drone_idx], s[bi], s[bi+1], s[bi+2], s[bi+3],
                missile_idx=mi, target_keypoints=ga_kps, dt=ga_dt,
            )
            total_time += hard
        scored.append((total_time, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n_seeds]]


def solve_p5(config_module, pop_size=None, n_generations=None, verbose=True):
    sim.set_config(config_module)
    if pop_size is None: pop_size = config_module.GA_POP_P5
    if n_generations is None: n_generations = config_module.GA_GEN_P5

    ga_kps = sim.get_target_keypoints(config_module.GA_N_CIRCLE, config_module.GA_N_LAYERS)
    ga_dt = config_module.GA_DT

    if verbose:
        print(f"  问题5: 分解策略 - 每架无人机独立优化")

    results = []
    t_start = time.time()

    for drone_i in range(5):
        mi_list = MISSILE_ASSIGNMENT[drone_i]

        seeds = _generate_seeds_p5_drone(config_module, drone_i, mi_list, ga_kps, ga_dt, n_seeds=40)
        if verbose:
            print(f"    FY{drone_i+1}: {len(seeds)} seeds, pop=80, gen=25")

        bounds = []
        for bj in range(3):
            bounds += [
                (0.0, 2.0 * np.pi),
                (config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX),
                (0.0, 15.0), (0.0, 10.0),
            ]

        t_rel_indices = [2, 4+2, 8+2]
        constraint = make_sequential_interval_constraint(t_rel_indices, min_interval=1.0)

        def make_fitness(drone_init, mi_list_inner, kps, dt):
            def f(x):
                total = 0.0
                for bj in range(3):
                    bi = bj * 4
                    hard = sim.simulate_single_bomb(
                        drone_init, x[bi], x[bi+1], x[bi+2], x[bi+3],
                        missile_idx=mi_list_inner[bj], target_keypoints=kps, dt=dt,
                    )
                    total += hard
                return total
            return f

        fitness_func = make_fitness(
            config_module.DRONES_INIT[drone_i], mi_list, ga_kps, ga_dt)

        ga = GeneticAlgorithm(
            fitness_func=fitness_func, bounds=bounds,
            pop_size=80, n_generations=25,
            crossover_rate=config_module.GA_CROSSOVER_RATE,
            mutation_rate=0.15,
            constraint_func=constraint, verbose=False,
        )
        best_solution, _ = ga.run(seeds=seeds if seeds else None)

        r = {
            'drone': f'FY{drone_i+1}', 'missiles': mi_list,
            'theta': [], 'speed': [], 't_rels': [], 't_lags': [],
        }
        for bj in range(3):
            bi = bj * 4
            r['theta'].append(best_solution[bi])
            r['speed'].append(best_solution[bi+1])
            r['t_rels'].append(best_solution[bi+2])
            r['t_lags'].append(best_solution[bi+3])
        results.append(r)

    # 精细验证
    fine_kps = sim.get_target_keypoints(config_module.FINE_N_CIRCLE, config_module.FINE_N_LAYERS)
    fine_dt = config_module.FINE_DT
    n_steps_fine = int(np.ceil(config_module.T_TOTAL / fine_dt))
    pm = [np.zeros(n_steps_fine, dtype=bool) for _ in range(3)]
    per_missile_times = [0.0, 0.0, 0.0]
    for i in range(5):
        r = results[i]
        for bj in range(3):
            mi = MISSILE_ASSIGNMENT[i][bj]
            mask = sim._bomb_coverage_mask(
                config_module.DRONES_INIT[i], r['theta'][bj], r['speed'][bj],
                r['t_rels'][bj], r['t_lags'][bj], mi,
                fine_kps, fine_dt, config_module.T_TOTAL)
            pm[mi] |= mask
    for mi in range(3):
        per_missile_times[mi] = pm[mi].sum() * fine_dt
    fine_time = sum(per_missile_times)
    ga_time = time.time() - t_start

    if verbose:
        print(f"\n  最优解 (精细验证):")
        for r in results:
            print(f"    {r['drone']}: θ={[f'{t:.4f}' for t in r['theta']]}, "
                  f"v={[f'{s:.1f}' for s in r['speed']]}")
            for b in range(3):
                print(f"      弹{b+1}->M{r['missiles'][b]+1}: "
                      f"t_rel={r['t_rels'][b]:.4f}s, t_lag={r['t_lags'][b]:.4f}s")
        print(f"    各导弹遮蔽: M1={per_missile_times[0]:.4f}s, "
              f"M2={per_missile_times[1]:.4f}s, M3={per_missile_times[2]:.4f}s")
        print(f"    总有效遮蔽时长 = {fine_time:.4f} s, 耗时: {ga_time:.1f}s")

    return results, fine_time
