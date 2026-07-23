"""
粒子群优化算法 (PSO) 实现，支持多进程并行评估种群
"""
import os
import numpy as np
import multiprocessing as mp
from config import PSO_SWARM_SIZE, PSO_MAX_ITER, PSO_W, PSO_C1, PSO_C2


# 子进程worker用的模块级状态：Pool的initializer只在每个worker进程启动时跑一次，
# 把目标函数存到worker自己的全局变量里，避免每次pool.map都重新pickle一遍
# objective_func(包括它内部可能带的关键点数组等数据)。
_worker_objective = None


def _pool_init(objective_func):
    global _worker_objective
    _worker_objective = objective_func


def _pool_eval(x):
    return _worker_objective(x)


class PSO:
    """粒子群优化器"""

    def __init__(self, objective_func, bounds, n_particles=None, max_iter=None,
                 w=None, c1=None, c2=None, maximize=True, verbose=True, n_workers='auto',
                 seed_positions=None):
        """
        Parameters:
            objective_func: 目标函数 f(x) -> float
                如果要用多进程(n_workers>1)，这个必须是可以被pickle的对象——
                模块级的普通函数、或者一个__init__/__call__都只存简单数据(numpy数组/数字)
                的类实例都可以；写在另一个函数内部的闭包(def objective(x): ...)不行，
                pickle序列化不了闭包，Windows/macOS的多进程默认用spawn方式启动子进程，
                启动时必须能把目标函数序列化过去。
            bounds: [(low1, high1), (low2, high2), ...] 每个维度的上下界
            n_particles: 粒子数量
            max_iter: 最大迭代次数
            maximize: True 表示最大化目标函数
            n_workers: 并行进程数。'auto'=自动按CPU核心数检测(os.cpu_count())；
                       传1或者检测失败时自动退回到单进程串行，不会报错崩溃。
            seed_positions: 可选，[(x1,...,xn), ...] 一组"热启动"初始解(比如用更便宜的
                       方法先粗搜出来的一个不错的候选)，会替换掉初始种群里的前几个
                       粒子(不是追加)，让PSO从这些已知不错的点附近开始搜，而不是纯
                       随机初始化。对应国奖论文里"贪心算法找可接受解，PSO在其附近精修"
                       的思路。
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_particles = n_particles or PSO_SWARM_SIZE
        self.max_iter = max_iter or PSO_MAX_ITER
        self.w = w or PSO_W
        self.c1 = c1 or PSO_C1
        self.c2 = c2 or PSO_C2
        self.maximize = maximize
        self.verbose = verbose
        self.seed_positions = seed_positions

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        if n_workers == 'auto':
            n_workers = os.cpu_count() or 1
        self.n_workers = max(1, int(n_workers))

        # 状态
        self.best_position = None
        self.best_value = None
        self.history = []

    def _evaluate_batch(self, positions, pool):
        """对整批粒子求值，有pool就并行算，没有就退回串行列表推导"""
        if pool is not None:
            raw = np.array(pool.map(_pool_eval, list(positions)))
        else:
            raw = np.array([self.objective_func(p) for p in positions])
        # PSO内部统一按"求最小值"处理，maximize=True时目标值取反
        return -raw if self.maximize else raw

    def optimize(self):
        """运行优化，返回 (best_position, best_value)"""
        pool = None
        if self.n_workers > 1:
            try:
                pool = mp.Pool(self.n_workers, initializer=_pool_init,
                                initargs=(self.objective_func,))
                if self.verbose:
                    detected = os.cpu_count()
                    print(f"  PSO 并行评估: 使用 {self.n_workers} 个进程 "
                          f"(检测到 {detected} 个CPU核心)")
            except Exception as e:
                if self.verbose:
                    print(f"  多进程启动失败({e})，回退到单进程串行")
                pool = None

        try:
            # 初始化粒子位置和速度
            positions = np.random.uniform(
                self.lb, self.ub, size=(self.n_particles, self.n_dims)
            )
            if self.seed_positions:
                n_seed = min(len(self.seed_positions), self.n_particles)
                for i in range(n_seed):
                    positions[i] = np.clip(self.seed_positions[i], self.lb, self.ub)
                if self.verbose:
                    print(f"  PSO 热启动: 用 {n_seed} 个预设起点替换初始种群中的对应粒子")
            velocities = np.zeros((self.n_particles, self.n_dims))

            values = self._evaluate_batch(positions, pool)

            # 个体最优
            pbest_positions = positions.copy()
            pbest_values = values.copy()

            # 全局最优（内部统一按"求最小值"处理）
            gbest_idx = np.argmin(values)
            self.best_position = positions[gbest_idx].copy()
            self.best_value = values[gbest_idx]
            self.history = [self.best_value]

            for iteration in range(self.max_iter):
                # 更新速度和位置
                r1 = np.random.random((self.n_particles, self.n_dims))
                r2 = np.random.random((self.n_particles, self.n_dims))

                velocities = (self.w * velocities +
                              self.c1 * r1 * (pbest_positions - positions) +
                              self.c2 * r2 * (self.best_position - positions))

                positions = positions + velocities

                # 边界处理 - clamp到边界内
                positions = np.clip(positions, self.lb, self.ub)

                # 评估
                new_values = self._evaluate_batch(positions, pool)

                # 更新个体最优
                improved = new_values < pbest_values
                pbest_positions[improved] = positions[improved].copy()
                pbest_values[improved] = new_values[improved]

                # 更新全局最优
                current_best_idx = np.argmin(new_values)
                current_best_value = new_values[current_best_idx]
                if current_best_value < self.best_value:
                    self.best_value = current_best_value
                    self.best_position = positions[current_best_idx].copy()

                self.history.append(self.best_value)

                if self.verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                    display_val = -self.best_value if self.maximize else self.best_value
                    print(f"  PSO iter {iteration+1}/{self.max_iter}: best = {display_val:.4f}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # 最终输出
        if self.maximize:
            return self.best_position, -self.best_value
        else:
            return self.best_position, self.best_value


def local_polish(objective_func, x0, bounds, method='Powell', maxiter=200, radius_frac=0.05):
    """
    PSO收敛完之后，在最优解附近做一次无梯度局部精修——对应国奖论文/MATLAB代码里
    "PSO+禁忌搜索"或者particleswarm的HybridFcn=@fmincon那一层收尾。

    我们的目标函数是"遮蔽时长对参数的积分"，不是纯离散的0/1判定，随参数小幅变化时
    通常是连续变化的(遮蔽区间的边界会平滑地移动)，但仿真本身有dt离散粒度带来的
    噪声，用有限差分梯度法(BFGS一类)容易被这点噪声带偏，所以选无梯度的Powell/
    Nelder-Mead，比强求梯度稳一些。

    实测过一个教训：如果直接把PSO用的那套完整bounds原样传给Powell，它会在大范围里
    大步搜索，从一个4.9s的解直接走丢到0s(这类占空比很低的目标函数——大部分参数
    组合都是0——对无梯度法很不友好，走出好解所在的那个窄"盆地"就再也回不来)。所以
    精修阶段强制把搜索范围收紧到x0周围 ±radius_frac*(hi-lo) 的一个小邻域内，真正
    做"局部"精修，而不是让它在全局范围里重新摸索。

    objective_func: 要最大化的目标函数 f(x)（不是PSO内部那种取反后的版本，直接传
                     原始的、值越大越好的目标函数）
    x0: PSO给出的最优解，作为精修起点
    bounds: [(low,high), ...]，跟PSO用的应该是同一套完整边界(函数内部会自动收紧)
    radius_frac: 精修邻域半径，占每一维完整区间长度的比例，默认5%

    Returns:
        (x_polished, f_polished): 精修后的解和对应目标值。如果精修没有找到更好的解
        (可能落回一个稍差的点，无梯度法不保证单调)，调用方应该自己跟PSO原结果比较，
        取较大的那个——这个函数只负责精修，不负责判断要不要采纳。
    """
    from scipy.optimize import minimize

    x0 = np.asarray(x0, dtype=float)
    bounds_arr = np.asarray(bounds, dtype=float)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    margin = radius_frac * (hi - lo)
    local_bounds = list(zip(np.maximum(lo, x0 - margin), np.minimum(hi, x0 + margin)))

    def neg_obj(x):
        return -objective_func(np.asarray(x))

    result = minimize(
        neg_obj, x0, method=method, bounds=local_bounds,
        options={'maxiter': maxiter, 'xtol': 1e-4, 'ftol': 1e-5}
    )
    return np.asarray(result.x), -result.fun
