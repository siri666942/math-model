"""
问题5: 五机多弹多导弹协同优化
Two-stage: per-drone geometric init + PSO, then joint PSO refinement.
"""
import numpy as np
import simulation
import config_a
from solve_lbfgs import _geometric_feasible_points, evaluate_candidates


def solve_problem5(config_module=None, verbose=True):
    """
    Solve problem 5: 5 drones, up to 3 bombs each, vs 3 missiles.
    """
    if config_module is None:
        config_module = config_a
    simulation.set_config(config_module)
    simulation.clear_keypoint_cache()
    cfg = config_module

    if verbose:
        print("=" * 60)
        print("问题5: 5架无人机协同投放烟幕弹")
        print("=" * 60)

    n_drones = 5
    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    intercept_order = cfg.INTERCEPT_ORDER
    dt_opt = 0.02
    t_total = 30.0
    target_keypoints = simulation.get_target_keypoints(36, 3)

    # ============================================================
    # Stage 1: Per-drone geometric + PSO
    # ============================================================
    if verbose:
        print("\n阶段1: 各无人机独立优化...")

    single_results = []

    for di in range(n_drones):
        drone_init = cfg.DRONES_INIT[di]
        od = intercept_order[drone_names[di]]

        if verbose:
            print(f"\n  --- {drone_names[di]} (拦截: {[f'M{k+1}' for k in od]}) ---")

        # For each missile in this drone's intercept order, find best single-bomb params
        best_overall_theta = None
        best_overall_speed = None

        for missile_k in set(od):
            geom_points = _geometric_feasible_points(
                drone_init, missile_k, cfg, n_t_det=15, n_alpha=10)
            candidates = evaluate_candidates(
                geom_points, drone_init, target_keypoints, cfg,
                dt=dt_opt, t_total=t_total, top_k=5)

            if candidates:
                if best_overall_theta is None:
                    best_overall_theta = candidates[0][1][0]
                    best_overall_speed = candidates[0][1][1]
                break  # Use first valid missile's solution

        if best_overall_theta is None:
            # Fallback random
            best_overall_theta = np.random.uniform(0, 2 * np.pi)
            best_overall_speed = np.random.uniform(cfg.DRONE_SPEED_MIN, cfg.DRONE_SPEED_MAX)

        if verbose:
            print(f"    几何初始化: theta={np.degrees(best_overall_theta):.1f}deg  "
                  f"v={best_overall_speed:.1f}m/s")

        # PSO with 8D per drone
        d_theta = 0.3
        d_speed = 25.0
        bounds = np.array([
            [max(0.0, best_overall_theta - d_theta),
             min(2*np.pi, best_overall_theta + d_theta)],
            [max(cfg.DRONE_SPEED_MIN, best_overall_speed - d_speed),
             min(cfg.DRONE_SPEED_MAX, best_overall_speed + d_speed)],
            [0.1, 8.0], [cfg.BOMB_INTERVAL_MIN, 6.0], [cfg.BOMB_INTERVAL_MIN, 6.0],
            [0.1, 8.0], [0.1, 8.0], [0.1, 8.0],
        ])
        lb = bounds[:, 0]; ub = bounds[:, 1]; dim = 8

        def make_eval(di, od):
            def eval_fn(x):
                theta, speed = x[0], x[1]
                rt = np.array([x[2], x[2]+x[3], x[2]+x[3]+x[4]])
                dl = np.array([x[5], x[6], x[7]])
                return simulation.simulate_multi_bomb_single_drone(
                    cfg.DRONES_INIT[di], theta, speed, rt, dl,
                    np.array(od), target_keypoints, dt_opt, t_total)
            return eval_fn

        evaluate = make_eval(di, od)
        n_p = 50; max_it = 60
        ws, we = 0.9, 0.4

        pos = np.random.uniform(lb, ub, size=(n_p, dim))
        pos[0, 0] = best_overall_theta
        pos[0, 1] = best_overall_speed
        vel = np.zeros((n_p, dim))
        vals = np.array([evaluate(p) for p in pos])
        pb_pos = pos.copy(); pb_val = vals.copy()
        gbest_idx = np.argmax(vals)
        gb_pos = pos[gbest_idx].copy(); gb_val = vals[gbest_idx]

        for it in range(max_it):
            w = ws - (ws - we) * it / max_it
            r1 = np.random.random((n_p, dim)); r2 = np.random.random((n_p, dim))
            vel = w*vel + 2.0*r1*(pb_pos-pos) + 2.0*r2*(gb_pos-pos)
            pos = np.clip(pos + vel, lb, ub)
            nv = np.array([evaluate(p) for p in pos])
            imp = nv > pb_val
            pb_pos[imp] = pos[imp].copy(); pb_val[imp] = nv[imp]
            if nv.max() > gb_val:
                gb_val = nv.max(); gb_pos = pos[nv.argmax()].copy()

        xo = gb_pos
        rt = np.array([xo[2], xo[2]+xo[3], xo[2]+xo[3]+xo[4]])
        dl = np.array([xo[5], xo[6], xo[7]])
        single_results.append({'th': xo[0], 'sp': xo[1], 'rt': rt, 'dl': dl, 't': gb_val})
        if verbose:
            print(f"    PSO最佳: {gb_val:.4f}s  theta={np.degrees(xo[0]):.1f}deg  v={xo[1]:.1f}m/s")

    total_single = sum(r['t'] for r in single_results)
    if verbose:
        print(f"\n  阶段1求和: {total_single:.4f}s")

    # ============================================================
    # Stage 2: Joint refinement (40D PSO)
    # ============================================================
    if verbose:
        print(f"\n阶段2: 联合微调 (40D PSO)...")

    bj = []
    for di in range(n_drones):
        r = single_results[di]
        bj.append([max(0.0, r['th']-0.1), min(2*np.pi, r['th']+0.1)])
        bj.append([max(cfg.DRONE_SPEED_MIN, r['sp']-10), min(cfg.DRONE_SPEED_MAX, r['sp']+10)])
    for di in range(n_drones):
        r = single_results[di]
        bj.append([max(0.1, r['rt'][0]-0.3), r['rt'][0]+0.3])
        bj.append([max(cfg.BOMB_INTERVAL_MIN, r['rt'][1]-r['rt'][0]-0.3), r['rt'][1]-r['rt'][0]+0.3])
        bj.append([max(cfg.BOMB_INTERVAL_MIN, r['rt'][2]-r['rt'][1]-0.3), r['rt'][2]-r['rt'][1]+0.3])
    for di in range(n_drones):
        r = single_results[di]
        for j in range(3):
            bj.append([max(0.1, r['dl'][j]-1.0), min(8.0, r['dl'][j]+1.0)])

    lb_j = np.array([b[0] for b in bj])
    ub_j = np.array([b[1] for b in bj])
    dim_j = len(bj)

    def evaluate_joint(x):
        idx = 0; dps = []
        for di in range(n_drones):
            th = x[idx]; idx+=1; sp = x[idx]; idx+=1
            r1 = x[idx]; idx+=1; i2 = x[idx]; idx+=1; i3 = x[idx]; idx+=1
            rt = np.array([r1, r1+i2, r1+i2+i3])
            dl = np.array([x[idx+j] for j in range(3)]); idx+=3
            od = intercept_order[drone_names[di]]
            dps.append({'drone_init': cfg.DRONES_INIT[di], 'theta': th, 'speed': sp,
                        'release_times': rt, 'detonation_delays': dl, 'missile_indices': od})
        tt, _ = simulation.simulate_multi_drone_multi_bomb(dps, dt_opt, 35.0)
        return tt

    # Build x0 from stage 1
    x0_j = np.zeros(dim_j)
    idx = 0
    for di in range(n_drones):
        r = single_results[di]
        x0_j[idx] = r['th']; idx+=1
        x0_j[idx] = r['sp']; idx+=1
    for di in range(n_drones):
        r = single_results[di]
        x0_j[idx] = r['rt'][0]; idx+=1
        x0_j[idx] = r['rt'][1]-r['rt'][0]; idx+=1
        x0_j[idx] = r['rt'][2]-r['rt'][1]; idx+=1
    for di in range(n_drones):
        r = single_results[di]
        for j in range(3):
            x0_j[idx] = r['dl'][j]; idx+=1

    n_pj = 60; max_it_j = 80
    pos_j = np.random.uniform(lb_j, ub_j, size=(n_pj, dim_j))
    pos_j[0] = np.clip(x0_j, lb_j, ub_j)
    vel_j = np.zeros((n_pj, dim_j))
    vals_j = np.array([evaluate_joint(p) for p in pos_j])
    pb_pos_j = pos_j.copy(); pb_val_j = vals_j.copy()
    gb_idx_j = np.argmax(vals_j)
    gb_pos_j = pos_j[gb_idx_j].copy(); gb_val_j = vals_j[gb_idx_j]

    for it in range(max_it_j):
        w = 0.9 - 0.5 * it / max_it_j
        r1 = np.random.random((n_pj, dim_j)); r2 = np.random.random((n_pj, dim_j))
        vel_j = w*vel_j + 2.0*r1*(pb_pos_j-pos_j) + 2.0*r2*(gb_pos_j-pos_j)
        pos_j = np.clip(pos_j + vel_j, lb_j, ub_j)
        nv_j = np.array([evaluate_joint(p) for p in pos_j])
        imp_j = nv_j > pb_val_j
        pb_pos_j[imp_j] = pos_j[imp_j].copy(); pb_val_j[imp_j] = nv_j[imp_j]
        if nv_j.max() > gb_val_j:
            gb_val_j = nv_j.max(); gb_pos_j = pos_j[nv_j.argmax()].copy()
        if verbose and (it+1) % 20 == 0:
            print(f"  联合PSO iter {it+1}/{max_it_j}: best = {gb_val_j:.4f}s")

    # Parse final results
    idx = 0
    final = []
    for di in range(n_drones):
        th = gb_pos_j[idx]; idx+=1; sp = gb_pos_j[idx]; idx+=1
        r1 = gb_pos_j[idx]; idx+=1; i2 = gb_pos_j[idx]; idx+=1; i3 = gb_pos_j[idx]; idx+=1
        rt = np.array([r1, r1+i2, r1+i2+i3])
        dl = np.array([gb_pos_j[idx+j] for j in range(3)]); idx+=3
        od = intercept_order[drone_names[di]]
        final.append({'th': th, 'sp': sp, 'rt': rt, 'dl': dl, 'od': od})

    if verbose:
        print(f"\n联合优化总遮蔽时长: {gb_val_j:.4f}s")
        for di in range(n_drones):
            r = final[di]
            d = np.array([np.cos(r['th']), np.sin(r['th']), 0.])
            print(f"  {drone_names[di]}: theta={np.degrees(r['th']):.1f}deg  v={r['sp']:.1f}m/s  "
                  f"拦截:{[f'M{k+1}' for k in r['od']]}")
            for j in range(3):
                dp = cfg.DRONES_INIT[di]+r['sp']*d*r['rt'][j]+r['sp']*d*r['dl'][j]
                dp[2] -= 0.5*cfg.G*r['dl'][j]**2
                print(f"    弹{j+1}: rt={r['rt'][j]:.3f}s  dd={r['dl'][j]:.3f}s  "
                      f"起爆({dp[0]:.0f},{dp[1]:.0f},{dp[2]:.0f})")

    return final, gb_val_j


if __name__ == "__main__":
    results, total = solve_problem5(config_a)
    print(f"\nPaper reference: 17.413s")
    print(f"Obtained: {total:.4f}s")
