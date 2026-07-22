"""
粒子群优化算法 (PSO) 实现
"""
import numpy as np
from config import PSO_SWARM_SIZE, PSO_MAX_ITER, PSO_W, PSO_C1, PSO_C2


class PSO:
    """粒子群优化器"""

    def __init__(self, objective_func, bounds, n_particles=None, max_iter=None,
                 w=None, c1=None, c2=None, maximize=True, verbose=True):
        """
        Parameters:
            objective_func: 目标函数 f(x) -> float
            bounds: [(low1, high1), (low2, high2), ...] 每个维度的上下界
            n_particles: 粒子数量
            max_iter: 最大迭代次数
            maximize: True 表示最大化目标函数
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

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        # 状态
        self.best_position = None
        self.best_value = None
        self.history = []

    def optimize(self):
        """运行优化，返回 (best_position, best_value)"""
        # 初始化粒子位置和速度
        positions = np.random.uniform(
            self.lb, self.ub, size=(self.n_particles, self.n_dims)
        )
        velocities = np.zeros((self.n_particles, self.n_dims))

        # 评估初始位置
        values = np.array([self._evaluate(p) for p in positions])

        # 个体最优
        pbest_positions = positions.copy()
        pbest_values = values.copy()

        # 全局最优
        if self.maximize:
            gbest_idx = np.argmax(values)
        else:
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
            new_values = np.array([self._evaluate(p) for p in positions])

            # 更新个体最优
            if self.maximize:
                improved = new_values > pbest_values
            else:
                improved = new_values < pbest_values

            pbest_positions[improved] = positions[improved].copy()
            pbest_values[improved] = new_values[improved]

            # 更新全局最优
            if self.maximize:
                current_best_idx = np.argmax(new_values)
                current_best_value = new_values[current_best_idx]
                if current_best_value > self.best_value:
                    self.best_value = current_best_value
                    self.best_position = positions[current_best_idx].copy()
            else:
                current_best_idx = np.argmin(new_values)
                current_best_value = new_values[current_best_idx]
                if current_best_value < self.best_value:
                    self.best_value = current_best_value
                    self.best_position = positions[current_best_idx].copy()

            self.history.append(self.best_value)

            if self.verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                sign = '-' if self.maximize else ''
                best_display = sign + f'{abs(self.best_value):.6f}'
                # 如果是最大化，显示正的最优值
                display_val = -self.best_value if self.maximize else self.best_value
                print(f"  PSO iter {iteration+1}/{self.max_iter}: best = {display_val:.4f}")

        # 最终输出
        if self.maximize:
            return self.best_position, -self.best_value
        else:
            return self.best_position, self.best_value

    def _evaluate(self, x):
        """评估单个粒子，处理越界和异常"""
        val = self.objective_func(x)
        # PSO内部用负值表示最大化问题
        if self.maximize:
            return -val  # 最小化负值 = 最大化原值
        return val
