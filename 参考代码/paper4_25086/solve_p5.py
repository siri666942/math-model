"""
问题5: 5架无人机，每架至多投放3枚烟幕干扰弹，实施对M1、M2、M3的干扰

Paper cumcm25086 approach: Multi-Island PSO with greedy assignment
Two-phase: Phase1 per-drone APSO → Phase2 Multi-Island PSO joint
"""
import numpy as np
import simulation as sim
from multi_island_pso import MultiIslandPSO
from apso import APSO


def solve_p5(config_module, target_keypoints=None, dt=None, t_total=None, verbose=True):
    """
    求解问题5: 五机多弹多导弹协同策略

    Returns:
        final_results, f_opt
    """
    sim.set_config(config_module)

    if dt is None:
        dt = 0.01  # Coarse for speed
    if t_total is None:
        t_total = config_module.T_TOTAL

    if target_keypoints is None:
        target_keypoints = sim.get_target_keypoints(n_circle=30, n_layers=3)

    N_DRONES = 5
    bomb_interval_min = config_module.BOMB_INTERVAL_MIN
    intercept_order = config_module.INTERCEPT_ORDER
    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']

    # Permissive coverage for optimization
    sim.set_coverage_ratio(0.01)

    # Drone-specific search ranges
    drone_configs = [
        {'theta_range': (np.pi * 0.90, np.pi * 1.0), 'release_range': (0.2, 6.0), 'delay_range': (0.5, 8.0)},
        {'theta_range': (np.pi * 0.80, np.pi * 1.0), 'release_range': (0.2, 10.0), 'delay_range': (0.5, 12.0)},
        {'theta_range': (np.pi * 0.75, np.pi * 0.95), 'release_range': (0.2, 14.0), 'delay_range': (0.5, 14.0)},
        {'theta_range': (np.pi * 0.70, np.pi * 0.95), 'release_range': (0.2, 18.0), 'delay_range': (0.5, 16.0)},
        {'theta_range': (np.pi * 0.75, np.pi * 0.95), 'release_range': (0.2, 18.0), 'delay_range': (0.5, 16.0)},
    ]

    # ============================================================
    # Phase 1: Per-drone APSO
    # ============================================================
    if verbose:
        print("=" * 60)
        print("问题5: 5机×3弹对M1/M2/M3 (Multi-Island PSO)")
        print("=" * 60)
        print("\nPhase 1: 逐架无人机APSO优化...")

    single_results = []
    for drone_idx in range(N_DRONES):
        cfg = drone_configs[drone_idx]
        orders = intercept_order[drone_names[drone_idx]]

        bounds = [
            cfg['theta_range'],
            (config_module.DRONE_SPEED_MIN, config_module.DRONE_SPEED_MAX),
            cfg['release_range'],
            (bomb_interval_min, 5.0),
            (bomb_interval_min, 5.0),
            cfg['delay_range'], cfg['delay_range'], cfg['delay_range'],
        ]

        def make_obj(d_idx, order):
            def obj(x):
                theta, speed = x[0], x[1]
                rel1 = x[2]
                int2 = max(bomb_interval_min, x[3])
                int3 = max(bomb_interval_min, x[4])
                release_times = np.array([rel1, rel1+int2, rel1+int2+int3])
                delays = np.array([x[5], x[6], x[7]])
                drone_params = [{
                    'drone_init': config_module.DRONES_INIT[d_idx],
                    'theta': theta, 'speed': speed,
                    'release_times': release_times,
                    'detonation_delays': delays,
                    'missile_indices': order,
                }]
                total_time, _ = sim.simulate_multi_drone_multi_bomb(
                    drone_params, dt=dt, t_total=t_total
                )
                return total_time
            return obj

        apso = APSO(make_obj(drone_idx, orders), bounds,
                    n_particles=max(20, config_module.APSO_SWARM_SIZE // 2),
                    max_iter=config_module.APSO_MAX_ITER // 2,
                    chi=config_module.APSO_CHI, c1=config_module.APSO_C1,
                    c2=config_module.APSO_C2,
                    w_start=config_module.APSO_W_START,
                    w_end=config_module.APSO_W_END,
                    maximize=True, verbose=False)
        x_opt_s, f_opt_s = apso.optimize()

        theta = x_opt_s[0]; speed = x_opt_s[1]
        rel1 = x_opt_s[2]
        int2 = max(bomb_interval_min, x_opt_s[3])
        int3 = max(bomb_interval_min, x_opt_s[4])
        release_times = np.array([rel1, rel1+int2, rel1+int2+int3])
        delays = np.array([x_opt_s[5], x_opt_s[6], x_opt_s[7]])

        single_results.append({
            'theta': theta, 'speed': speed,
            'release_times': release_times, 'delays': delays, 'time': f_opt_s,
        })
        if verbose:
            print(f"  {drone_names[drone_idx]}: {f_opt_s:.4f}s, "
                  f"theta={np.degrees(theta):.0f}deg, v={speed:.0f}m/s")

    if verbose:
        print(f"\nPhase 1 naive sum: {sum(r['time'] for r in single_results):.4f}s")

    # ============================================================
    # Phase 2: Multi-Island PSO joint
    # ============================================================
    if verbose:
        print("\nPhase 2: Multi-Island PSO联合优化...")

    bounds_joint = []
    for drone_idx in range(N_DRONES):
        r = single_results[drone_idx]; cfg = drone_configs[drone_idx]
        td, sd, dd2 = 0.15, 15.0, 3.0

        bounds_joint.append((
            max(cfg['theta_range'][0], min(cfg['theta_range'][1], r['theta']-td)),
            min(cfg['theta_range'][1], max(cfg['theta_range'][0], r['theta']+td))))
        bounds_joint.append((
            max(config_module.DRONE_SPEED_MIN, r['speed']-sd),
            min(config_module.DRONE_SPEED_MAX, r['speed']+sd)))
        bounds_joint.append((max(0.1, r['release_times'][0]-dd2), r['release_times'][0]+dd2))
        int_c = r['release_times'][1]-r['release_times'][0]
        bounds_joint.append((max(bomb_interval_min, int_c-dd2), int_c+dd2))
        int_c2 = r['release_times'][2]-r['release_times'][1]
        bounds_joint.append((max(bomb_interval_min, int_c2-dd2), int_c2+dd2))
        for j in range(3):
            bounds_joint.append((max(0.1, r['delays'][j]-dd2), r['delays'][j]+dd2))

    def objective_joint(x):
        idx = 0; drone_params = []
        for di in range(N_DRONES):
            theta = x[idx]; idx+=1; speed=x[idx]; idx+=1
            rel1=x[idx]; idx+=1
            int2=max(bomb_interval_min,x[idx]); idx+=1
            int3=max(bomb_interval_min,x[idx]); idx+=1
            rts=np.array([rel1, rel1+int2, rel1+int2+int3])
            dls=np.array([x[idx+j] for j in range(3)]); idx+=3
            orders=intercept_order[drone_names[di]]
            drone_params.append({
                'drone_init': config_module.DRONES_INIT[di],
                'theta':theta,'speed':speed,
                'release_times':rts,'detonation_delays':dls,
                'missile_indices':orders,
            })
        total_time,_=sim.simulate_multi_drone_multi_bomb(drone_params,dt=dt,t_total=t_total)
        return total_time

    if verbose:
        print(f"  Joint: {len(bounds_joint)}D, {config_module.MI_N_ISLANDS} islands × "
              f"{config_module.MI_SWARM_PER_ISLAND} particles")

    mi_pso = MultiIslandPSO(objective_joint, bounds_joint,
        n_islands=config_module.MI_N_ISLANDS,
        swarm_per_island=config_module.MI_SWARM_PER_ISLAND,
        max_iter=config_module.MI_MAX_ITER,
        migration_interval=config_module.MI_MIGRATION_INTERVAL,
        migration_rate=config_module.MI_MIGRATION_RATE,
        elite_size=config_module.MI_ELITE_SIZE,
        chi=config_module.APSO_CHI, c1=config_module.APSO_C1, c2=config_module.APSO_C2,
        w_start=config_module.APSO_W_START, w_end=config_module.APSO_W_END,
        maximize=True, verbose=verbose)
    x_opt_joint, f_opt_coarse = mi_pso.optimize()

    # Phase 3: Verify with strict coverage
    sim.set_coverage_ratio(0.80)
    f_opt_fine = objective_joint(x_opt_joint)

    # Parse results
    idx = 0; final_results = []
    for di in range(N_DRONES):
        theta=x_opt_joint[idx]; idx+=1; speed=x_opt_joint[idx]; idx+=1
        rel1=x_opt_joint[idx]; idx+=1
        int2=max(bomb_interval_min,x_opt_joint[idx]); idx+=1
        int3=max(bomb_interval_min,x_opt_joint[idx]); idx+=1
        rts=np.array([rel1,rel1+int2,rel1+int2+int3])
        dls=np.array([x_opt_joint[idx+j] for j in range(3)]); idx+=3
        orders=intercept_order[drone_names[di]]
        final_results.append({
            'name':drone_names[di],'theta':theta,'speed':speed,
            'release_times':rts,'delays':dls,'order':orders,
        })

    if verbose:
        print(f"\n最终结果 (coarse={f_opt_coarse:.4f}, verify={f_opt_fine:.4f}s):")
        for r in final_results:
            d=np.array([np.cos(r['theta']),np.sin(r['theta']),0.0])
            print(f"\n{r['name']}: theta={np.degrees(r['theta']):.0f}deg, v={r['speed']:.0f}m/s")
            print(f"  拦截: M{r['order'][0]+1},M{r['order'][1]+1},M{r['order'][2]+1}")
            for j in range(3):
                dp=config_module.DRONES_INIT[drone_names.index(r['name'])]+r['speed']*d*r['release_times'][j]
                bp=dp+r['speed']*d*r['delays'][j]
                bp[2]-=0.5*config_module.G*r['delays'][j]**2
                print(f"  弹{j+1}: t={r['release_times'][j]:.4f}s, delay={r['delays'][j]:.4f}s, "
                      f"det=({bp[0]:.0f},{bp[1]:.0f},{bp[2]:.0f})")
        print(f"\n总有效遮蔽时长: {f_opt_fine:.4f} s")

    return final_results, f_opt_fine


if __name__ == "__main__":
    import config_a
    solve_p5(config_a)
