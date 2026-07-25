"""
GA / SA 简易实现，用于在论文里做同条件 PSO 对比。
仅供敏感性/对比章节使用，不参与正式求解路径。
"""
import numpy as np


def run_ga(objective, bounds, n_eval_budget=6000, seed=42):
    """实数编码 GA：SBX 交叉 + 多项式变异 + 锦标赛选择。
    返回: 历次评估后的最优值曲线(每代记录一次)，长度等于代数。
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    n_dim = len(bounds)
    pop_size = 60
    n_gen = max(1, n_eval_budget // pop_size)
    eta_c, eta_m = 20, 20  # SBX/多项式分布指数
    # 初始化种群
    pop = lb + rng.random((pop_size, n_dim)) * (ub - lb)
    fit = np.array([objective(ind) for ind in pop])
    n_eval = pop_size
    curve = [float(fit.max())]

    def sbx_crossover(p1, p2, eta=eta_c):
        if rng.random() > 0.9:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2*u) ** (1/(eta+1))
                    else:
                        beta = (1/(2*(1-u))) ** (1/(eta+1))
                    c1[i] = 0.5*((p1[i]+p2[i]) - beta*abs(p1[i]-p2[i]))
                    c2[i] = 0.5*((p1[i]+p2[i]) + beta*abs(p1[i]-p2[i]))
        return c1, c2

    def poly_mutation(x, eta=eta_m):
        y = x.copy()
        for i in range(len(x)):
            if rng.random() < 1.0/len(x):
                u = rng.random()
                if u < 0.5:
                    delta = (2*u)**(1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta+1))
                y[i] = x[i] + delta * (ub[i] - lb[i])
        return y

    def tournament(pop, fit, k=3):
        idx = rng.integers(0, len(pop), k)
        winner = idx[fit[idx].argmax()]
        return pop[winner].copy()

    while n_eval < n_eval_budget:
        new_pop = []
        for _ in range(pop_size // 2):
            p1 = tournament(pop, fit)
            p2 = tournament(pop, fit)
            c1, c2 = sbx_crossover(p1, p2)
            c1 = poly_mutation(np.clip(c1, lb, ub))
            c2 = poly_mutation(np.clip(c2, lb, ub))
            new_pop.extend([c1, c2])
        new_pop = np.array(new_pop)
        new_fit = np.array([objective(ind) for ind in new_pop])
        n_eval += pop_size
        # 精英保留
        merged_pop = np.vstack([pop, new_pop])
        merged_fit = np.concatenate([fit, new_fit])
        top_idx = np.argsort(merged_fit)[-pop_size:]
        pop, fit = merged_pop[top_idx], merged_fit[top_idx]
        curve.append(float(fit.max()))
    return curve


def run_sa(objective, bounds, n_eval_budget=6000, seed=42, T0=1.0, Tend=1e-3):
    """模拟退火：指数降温，高斯扰动。
    返回: 历次评估后的最优值曲线（每 N 次评估记录一次，约 n_eval_budget//50 个点）。
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    n_dim = len(bounds)
    x = lb + rng.random(n_dim) * (ub - lb)
    fx = objective(x)
    best_x, best_f = x.copy(), fx
    alpha = (Tend / T0) ** (1.0 / max(1, n_eval_budget - 1))
    T = T0
    record_every = max(1, n_eval_budget // 50)
    curve = [float(best_f)]
    for it in range(1, n_eval_budget):
        # 邻域扰动：与边界宽度成比例
        step = (ub - lb) * 0.10
        x_new = x + rng.normal(0, 1, n_dim) * step
        x_new = np.clip(x_new, lb, ub)
        fx_new = objective(x_new)
        df = fx_new - fx
        if df >= 0 or rng.random() < np.exp(df / max(T, 1e-12)):
            x, fx = x_new, fx_new
            if fx > best_f:
                best_x, best_f = x.copy(), fx
        T *= alpha
        if it % record_every == 0:
            curve.append(float(best_f))
    return curve