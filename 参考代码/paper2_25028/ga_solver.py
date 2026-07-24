"""
遗传算法求解器 (自实现, 不依赖外部GA库)
支持约束处理 (投放间隔约束等)
"""
import numpy as np
import time


class GeneticAlgorithm:
    """
    遗传算法求解器

    特点:
    - 锦标赛选择
    - 混合交叉 (BLX-alpha)
    - 高斯变异
    - 精英保留策略
    - 约束处理（投放间隔等）
    """

    def __init__(self, fitness_func, bounds, n_vars=None,
                 pop_size=100, n_generations=30,
                 crossover_rate=0.8, mutation_rate=0.1,
                 elite_size=2, tournament_size=3,
                 constraint_func=None,
                 discrete_indices=None,
                 verbose=True):
        """
        参数:
            fitness_func: 适应度函数 f(x) -> float (最大化问题)
            bounds: 变量边界 list of (min, max) 或 (n_vars, 2) ndarray
            n_vars: 变量数量（从bounds推断）
            pop_size: 种群大小
            n_generations: 进化代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率（每个基因独立）
            elite_size: 精英保留数量
            tournament_size: 锦标赛选择大小
            constraint_func: 约束函数 f(x) -> bool, True表示可行
            discrete_indices: 离散变量的索引列表
            verbose: 是否打印进度
        """
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.n_vars = self.bounds.shape[0]
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.constraint_func = constraint_func
        self.discrete_indices = discrete_indices or []
        self.verbose = verbose

        # 历史记录
        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
        }

    def _initialize_population(self, seeds=None):
        """初始化种群，可注入种子个体"""
        population = np.zeros((self.pop_size, self.n_vars))
        if seeds is not None and len(seeds) > 0:
            n_seeds = min(len(seeds), self.pop_size)
            population[:n_seeds] = np.array(seeds[:n_seeds])
            start_idx = n_seeds
        else:
            start_idx = 0

        for i in range(self.n_vars):
            population[start_idx:, i] = np.random.uniform(
                self.bounds[i, 0], self.bounds[i, 1], self.pop_size - start_idx
            )
        return population

    def _evaluate_fitness(self, population):
        """评估种群适应度"""
        fitness = np.full(self.pop_size, -np.inf)
        for i in range(self.pop_size):
            x = population[i]
            # 约束检查
            if self.constraint_func is not None and not self.constraint_func(x):
                fitness[i] = -1e10  # 不可行解的惩罚
                continue
            try:
                val = self.fitness_func(x)
                fitness[i] = val if not np.isnan(val) and not np.isinf(val) else -1e10
            except Exception:
                fitness[i] = -1e10
        return fitness

    def _tournament_select(self, population, fitness):
        """锦标赛选择（最大化适应度）"""
        indices = np.random.randint(0, self.pop_size, self.tournament_size)
        best_idx = indices[np.argmax(fitness[indices])]
        return population[best_idx].copy()

    def _blend_crossover(self, parent1, parent2, alpha=0.5):
        """BLX-alpha 混合交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(self.n_vars):
            if np.random.random() < self.crossover_rate:
                x1, x2 = parent1[i], parent2[i]
                if x1 > x2:
                    x1, x2 = x2, x1
                delta = (x2 - x1) * alpha
                low = max(self.bounds[i, 0], x1 - delta)
                high = min(self.bounds[i, 1], x2 + delta)
                child1[i] = np.random.uniform(low, high)
                child2[i] = np.random.uniform(low, high)

        return child1, child2

    def _gaussian_mutation(self, individual):
        """高斯变异"""
        for i in range(self.n_vars):
            if np.random.random() < self.mutation_rate:
                sigma = (self.bounds[i, 1] - self.bounds[i, 0]) * 0.05
                individual[i] += np.random.normal(0, sigma)
                individual[i] = np.clip(individual[i], self.bounds[i, 0], self.bounds[i, 1])
        return individual

    def run(self, seeds=None):
        """运行遗传算法，可注入种子个体"""
        t_start = time.time()

        # 初始化种群
        population = self._initialize_population(seeds=seeds)
        fitness = self._evaluate_fitness(population)

        # 找出初始最优
        best_idx = np.argmax(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]

        for gen in range(self.n_generations):
            # 记录统计
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(fitness[fitness > -1e9]))
            self.history['worst_fitness'].append(np.min(fitness[fitness > -1e9])
                                                  if np.any(fitness > -1e9) else -1e10)

            if self.verbose and (gen % max(1, self.n_generations // 10) == 0 or gen == self.n_generations - 1):
                elapsed = time.time() - t_start
                print(f"  Gen {gen:4d}/{self.n_generations} | "
                      f"Best: {self.best_fitness:.6f} | "
                      f"Avg: {self.history['avg_fitness'][-1]:.6f} | "
                      f"Time: {elapsed:.1f}s")

            # 生成新一代
            new_population = np.zeros_like(population)

            # 精英保留
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            for j, idx in enumerate(elite_indices):
                new_population[j] = population[idx].copy()

            # 生成子代
            for j in range(self.elite_size, self.pop_size, 2):
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)
                child1, child2 = self._blend_crossover(parent1, parent2)
                child1 = self._gaussian_mutation(child1)
                child2 = self._gaussian_mutation(child2)
                new_population[j] = child1
                if j + 1 < self.pop_size:
                    new_population[j + 1] = child2

            # 替换种群
            population = new_population
            fitness = self._evaluate_fitness(population)

            # 更新最优
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitness[gen_best_idx]
                self.best_solution = population[gen_best_idx].copy()

        if self.verbose:
            elapsed = time.time() - t_start
            print(f"  GA完成 | Best fitness: {self.best_fitness:.6f} | Total time: {elapsed:.1f}s")

        return self.best_solution, self.best_fitness


# ---- 约束函数构建 ----

def make_interval_constraint(bomb_indices, min_interval=1.0):
    """
    构建投放间隔约束函数

    参数:
        bomb_indices: list of tuples, e.g. [(2,3), (4,5), (6,7)]
                      表示 t_rel[2] 和 t_rel[3] 之间必须间隔 >= min_interval
        min_interval: 最小间隔

    返回:
        constraint_func: f(x) -> bool
    """
    def constraint(x):
        for i, j in bomb_indices:
            if abs(x[i] - x[j]) < min_interval - 1e-9:
                return False
        return True
    return constraint


def make_sequential_interval_constraint(t_rel_indices, min_interval=1.0):
    """
    构建顺序投放间隔约束：t_rel[1] >= t_rel[0] + min_interval

    参数:
        t_rel_indices: 投放时间变量索引列表（按顺序）
        min_interval: 最小间隔

    返回:
        constraint_func: f(x) -> bool
    """
    def constraint(x):
        for k in range(1, len(t_rel_indices)):
            if x[t_rel_indices[k]] < x[t_rel_indices[k - 1]] + min_interval - 1e-9:
                return False
        return True
    return constraint


def combine_constraints(*constraints):
    """组合多个约束函数"""
    def combined(x):
        for c in constraints:
            if not c(x):
                return False
        return True
    return combined


# ---- 适应度函数构建 ----

def make_fitness_from_simulate(simulate_func, penalty_factor=1e10):
    """
    将仿真函数包装为适应度函数

    参数:
        simulate_func: f(x) -> float (遮蔽时长, >=0)
        penalty_factor: 约束违反惩罚因子

    返回:
        fitness_func: f(x) -> float
    """
    def fitness(x):
        result = simulate_func(x)
        if result is None or np.isnan(result) or np.isinf(result):
            return -penalty_factor
        return result
    return fitness
