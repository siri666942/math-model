"""
Adaptive PSO (APSO) with Clerc's Constriction Factor
Paper cumcm25086: Adaptive Particle Swarm Optimization

Key features:
1. Clerc's constriction factor for guaranteed convergence
2. Adaptive inertia weight (linear decreasing)
3. Velocity clamping to prevent explosion
4. Boundary handling with reflection
5. Elite preservation (keep best particle)
"""
import numpy as np


class APSO:
    """
    Adaptive PSO with Clerc's constriction factor.

    Velocity update:
        v = chi * [w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)]

    where:
        chi = constriction factor (default 0.729)
        w = adaptive inertia weight (linearly decreasing)
        c1, c2 = acceleration coefficients (default 2.05 each)
    """

    def __init__(self, objective_func, bounds, n_particles=60, max_iter=120,
                 chi=0.729, c1=2.05, c2=2.05, w_start=0.9, w_end=0.4,
                 maximize=True, verbose=True, vmax_ratio=0.2,
                 early_stop_rounds=None, seed=None,
                 init_center=None, init_spread=0.3):
        """
        Parameters:
            objective_func: callable f(x) -> float
            bounds: list of (low, high) tuples, one per dimension
            n_particles: swarm size
            max_iter: maximum iterations
            chi: constriction factor (Clerc's: 0.729)
            c1: cognitive acceleration coefficient
            c2: social acceleration coefficient
            w_start: initial inertia weight
            w_end: final inertia weight
            maximize: True for maximization, False for minimization
            verbose: print progress
            vmax_ratio: max velocity as ratio of search range
            early_stop_rounds: stop if no improvement for this many iterations
            seed: random seed for reproducibility
            init_center: array-like, center point for initial population seeding.
                         Half of the particles will be initialized near this point.
            init_spread: spread ratio for seeded particles (fraction of search range)
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds, dtype=float)
        self.n_dims = len(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.chi = chi
        self.c1 = c1
        self.c2 = c2
        self.w_start = w_start
        self.w_end = w_end
        self.maximize = maximize
        self.verbose = verbose
        self.vmax_ratio = vmax_ratio
        self.early_stop_rounds = early_stop_rounds
        self.init_center = np.asarray(init_center) if init_center is not None else None
        self.init_spread = init_spread

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.search_range = self.ub - self.lb
        self.vmax = self.vmax_ratio * self.search_range

        # Internal state
        self.best_position = None
        self.best_value = None
        self.history = []
        self.position_history = []  # best position at each iteration
        self._fitness_calls = 0

        if seed is not None:
            np.random.seed(seed)

    def _inertia_weight(self, iteration):
        """Adaptive inertia weight (linear decreasing)"""
        return self.w_start - (self.w_start - self.w_end) * iteration / self.max_iter

    def _clamp_position(self, pos):
        """边界处理：反射法 (reflection)"""
        for d in range(self.n_dims):
            if pos[d] < self.lb[d]:
                pos[d] = 2 * self.lb[d] - pos[d]  # reflect
                if pos[d] > self.ub[d]:
                    pos[d] = self.ub[d]
            elif pos[d] > self.ub[d]:
                pos[d] = 2 * self.ub[d] - pos[d]  # reflect
                if pos[d] < self.lb[d]:
                    pos[d] = self.lb[d]
        return np.clip(pos, self.lb, self.ub)

    def _clamp_velocity(self, vel):
        """速度钳制"""
        return np.clip(vel, -self.vmax, self.vmax)

    def _evaluate(self, x):
        """评估单个粒子"""
        self._fitness_calls += 1
        val = self.objective_func(x)
        # PSO内部用负值表示最大化
        if self.maximize:
            return -val if np.isfinite(val) else 1e10
        return val if np.isfinite(val) else 1e10

    def optimize(self):
        """运行APSO优化，返回 (best_position, best_value)"""
        # 初始化
        positions = np.random.uniform(self.lb, self.ub,
                                      size=(self.n_particles, self.n_dims))

        # Seed half the particles near the init_center if provided
        if self.init_center is not None:
            n_seed = self.n_particles // 2
            center = self.init_center
            for i in range(n_seed):
                # Gaussian perturbation around center, clamped to bounds
                noise = np.random.randn(self.n_dims) * self.init_spread * self.search_range
                seeded = center + noise
                positions[i] = np.clip(seeded, self.lb, self.ub)

        velocities = np.zeros((self.n_particles, self.n_dims))

        # 评估初始种群
        values = np.array([self._evaluate(p) for p in positions])

        # 个体历史最优
        pbest_positions = positions.copy()
        pbest_values = values.copy()

        # 全局最优
        if self.maximize:
            gbest_idx = np.argmin(values)  # 最小化内部负值 = 最大化实际值
        else:
            gbest_idx = np.argmin(values)

        self.best_position = positions[gbest_idx].copy()
        self.best_value = values[gbest_idx]
        self.history = [self._actual_value(self.best_value)]
        self.position_history = [self.best_position.copy()]

        no_improve_count = 0

        for iteration in range(self.max_iter):
            w = self._inertia_weight(iteration)

            # 随机因子
            r1 = np.random.random((self.n_particles, self.n_dims))
            r2 = np.random.random((self.n_particles, self.n_dims))

            # APSO速度更新: v = chi * [w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)]
            cognitive = self.c1 * r1 * (pbest_positions - positions)
            social = self.c2 * r2 * (self.best_position - positions)
            velocities = self.chi * (w * velocities + cognitive + social)

            # 速度钳制
            velocities = self._clamp_velocity(velocities)

            # 位置更新
            positions = positions + velocities
            for i in range(self.n_particles):
                positions[i] = self._clamp_position(positions[i])

            # 评估
            new_values = np.array([self._evaluate(p) for p in positions])

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
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.history.append(self._actual_value(self.best_value))
            self.position_history.append(self.best_position.copy())

            # 精英保留: 每次迭代将全局最优粒子放回种群
            # (ensures best solution is not lost)
            worst_idx = np.argmax(pbest_values)
            pbest_positions[worst_idx] = self.best_position.copy()
            pbest_values[worst_idx] = self.best_value

            if self.verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                actual = self._actual_value(self.best_value)
                print(f"  APSO iter {iteration+1}/{self.max_iter}: "
                      f"best={actual:.4f}, w={w:.4f}, calls={self._fitness_calls}")

            # 早停
            if self.early_stop_rounds and no_improve_count >= self.early_stop_rounds:
                if self.verbose:
                    print(f"  APSO early stop at iter {iteration+1} "
                          f"(no improvement for {self.early_stop_rounds} rounds)")
                break

        if self.verbose:
            actual = self._actual_value(self.best_value)
            print(f"  APSO final: best={actual:.6f}, "
                  f"total fitness calls={self._fitness_calls}")

        return self.best_position, self._actual_value(self.best_value)

    def _actual_value(self, internal_val):
        """将内部值转换为实际目标值"""
        return -internal_val if self.maximize else internal_val

    def get_convergence_data(self):
        """返回收敛数据用于绘图"""
        return {
            'history': self.history,
            'position_history': self.position_history,
            'n_calls': self._fitness_calls,
        }
