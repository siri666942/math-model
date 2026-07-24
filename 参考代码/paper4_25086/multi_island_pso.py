"""
Multi-Island Particle Swarm Optimization
Paper cumcm25086: Multi-Island PSO for multi-drone coordination

Key features:
1. Multiple independent PSO swarms (islands) running in parallel
2. Periodic migration of best solutions between islands
3. Elite preservation within each island
4. Ring topology for island communication
5. Prevents premature convergence through population diversity

Architecture:
    N islands, each with M particles
    Each island runs APSO independently for migration_interval iterations
    Then best particles migrate between islands (ring topology)
    Final solution = best across all islands
"""
import numpy as np
from apso import APSO


class MultiIslandPSO:
    """
    Multi-Island PSO: multiple independent swarms with periodic migration.

    Island topology: ring (each island migrates to next neighbor)
    Migration strategy: replace worst particles with best from neighbor
    """

    def __init__(self, objective_func, bounds, n_islands=5,
                 swarm_per_island=30, max_iter=80, migration_interval=10,
                 migration_rate=2, elite_size=3,
                 chi=0.729, c1=2.05, c2=2.05,
                 w_start=0.9, w_end=0.4,
                 maximize=True, verbose=True, seed=None):
        """
        Parameters:
            objective_func: callable f(x) -> float
            bounds: list of (low, high) tuples
            n_islands: number of parallel islands
            swarm_per_island: particles per island
            max_iter: total iterations per island
            migration_interval: iterations between migrations
            migration_rate: number of particles to migrate per exchange
            elite_size: number of elite particles preserved per island
            chi, c1, c2: APSO constriction parameters
            w_start, w_end: inertia weight range
            maximize: True for maximization
            verbose: print progress
            seed: random seed
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_islands = n_islands
        self.swarm_per_island = swarm_per_island
        self.max_iter = max_iter
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.elite_size = elite_size
        self.chi = chi
        self.c1 = c1
        self.c2 = c2
        self.w_start = w_start
        self.w_end = w_end
        self.maximize = maximize
        self.verbose = verbose
        self.seed = seed

        self.best_position = None
        self.best_value = None
        self.island_bests = []  # best from each island
        self.history = []
        self._total_calls = 0

        if seed is not None:
            np.random.seed(seed)

    def _create_island(self, island_id, init_positions=None, init_velocities=None):
        """创建一个独立的APSO岛屿"""
        seed = None if self.seed is None else self.seed + island_id * 1000

        # We subclass APSO functionality directly for each island
        island = {
            'id': island_id,
            'positions': None,
            'velocities': None,
            'pbest_positions': None,
            'pbest_values': None,
            'best_position': None,
            'best_value': None,
            'history': [],
        }

        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        search_range = ub - lb
        vmax = 0.2 * search_range

        rng = np.random.RandomState(self.seed + island_id * 1000 if self.seed else None)

        if init_positions is not None:
            island['positions'] = init_positions.copy()
        else:
            island['positions'] = rng.uniform(lb, ub,
                                              size=(self.swarm_per_island, len(self.bounds)))

        if init_velocities is not None:
            island['velocities'] = init_velocities.copy()
        else:
            island['velocities'] = np.zeros((self.swarm_per_island, len(self.bounds)))

        island['vmax'] = vmax
        island['lb'] = lb
        island['ub'] = ub
        island['search_range'] = search_range
        island['rng'] = rng

        return island

    def _evaluate_island(self, island):
        """评估岛上所有粒子的适应度"""
        positions = island['positions']
        n = len(positions)
        values = np.zeros(n)
        for i in range(n):
            val = self.objective_func(positions[i])
            self._total_calls += 1
            if self.maximize:
                values[i] = -val if np.isfinite(val) else 1e10
            else:
                values[i] = val if np.isfinite(val) else 1e10
        return values

    def _clamp_position(self, pos, island):
        """边界处理：反射法"""
        for d in range(len(pos)):
            if pos[d] < island['lb'][d]:
                pos[d] = 2 * island['lb'][d] - pos[d]
                if pos[d] > island['ub'][d]:
                    pos[d] = island['ub'][d]
            elif pos[d] > island['ub'][d]:
                pos[d] = 2 * island['ub'][d] - pos[d]
                if pos[d] < island['lb'][d]:
                    pos[d] = island['lb'][d]
        return np.clip(pos, island['lb'], island['ub'])

    def _initialize_islands(self):
        """初始化所有岛屿"""
        islands = []
        for i in range(self.n_islands):
            island = self._create_island(i)
            values = self._evaluate_island(island)

            island['pbest_positions'] = island['positions'].copy()
            island['pbest_values'] = values.copy()

            best_idx = np.argmin(values)
            island['best_position'] = island['positions'][best_idx].copy()
            island['best_value'] = values[best_idx]
            island['history'] = [self._actual_value(island['best_value'])]

            islands.append(island)

        return islands

    def _run_island_iteration(self, island, iteration, total_iter):
        """运行单个岛屿的一次迭代"""
        w = self.w_start - (self.w_start - self.w_end) * iteration / max(total_iter, 1)

        n = self.swarm_per_island
        rng = island['rng']
        r1 = rng.random((n, len(self.bounds)))
        r2 = rng.random((n, len(self.bounds)))

        cognitive = self.c1 * r1 * (island['pbest_positions'] - island['positions'])
        social = self.c2 * r2 * (island['best_position'] - island['positions'])
        island['velocities'] = self.chi * (w * island['velocities'] + cognitive + social)

        # Velocity clamping
        island['velocities'] = np.clip(island['velocities'], -island['vmax'], island['vmax'])

        # Position update
        island['positions'] = island['positions'] + island['velocities']
        for i in range(n):
            island['positions'][i] = self._clamp_position(island['positions'][i], island)

        # Evaluate
        new_values = self._evaluate_island(island)

        # Update pbest
        improved = new_values < island['pbest_values']
        island['pbest_positions'][improved] = island['positions'][improved].copy()
        island['pbest_values'][improved] = new_values[improved]

        # Update gbest
        best_idx = np.argmin(new_values)
        if new_values[best_idx] < island['best_value']:
            island['best_value'] = new_values[best_idx]
            island['best_position'] = island['positions'][best_idx].copy()

        island['history'].append(self._actual_value(island['best_value']))

        # Elite preservation
        worst_idx = np.argmax(island['pbest_values'])
        island['pbest_positions'][worst_idx] = island['best_position'].copy()
        island['pbest_values'][worst_idx] = island['best_value']

        return island

    def _migrate(self, islands):
        """
        Migration between islands (ring topology).

        Each island sends its best migration_rate particles to the next island,
        which replaces its worst particles.
        """
        n_islands = len(islands)
        for i in range(n_islands):
            src = islands[i]
            dst = islands[(i + 1) % n_islands]  # ring: next neighbor

            # Source: find best particles (lowest internal values = best)
            sorted_idx = np.argsort(src['pbest_values'])
            n_migrate = min(self.migration_rate, len(sorted_idx))

            # Avoid migrating the same position as destination's best
            migrant_positions = src['pbest_positions'][sorted_idx[:n_migrate]].copy()
            migrant_values = src['pbest_values'][sorted_idx[:n_migrate]].copy()

            # Destination: replace worst particles
            dst_sorted = np.argsort(dst['pbest_values'])[::-1]  # worst first
            n_replace = min(n_migrate, len(dst_sorted))

            for j in range(n_replace):
                dst['pbest_positions'][dst_sorted[j]] = migrant_positions[j].copy()
                dst['pbest_values'][dst_sorted[j]] = migrant_values[j]
                dst['positions'][dst_sorted[j]] = migrant_positions[j].copy()

            # Re-evaluate destination's best
            vals = dst['pbest_values']
            best_idx = np.argmin(vals)
            dst['best_value'] = vals[best_idx]
            dst['best_position'] = dst['pbest_positions'][best_idx].copy()

    def _actual_value(self, internal_val):
        """Convert internal value to actual objective value"""
        return -internal_val if self.maximize else internal_val

    def optimize(self):
        """Run Multi-Island PSO optimization"""
        if self.verbose:
            print(f"Multi-Island PSO: {self.n_islands} islands × "
                  f"{self.swarm_per_island} particles, "
                  f"max_iter={self.max_iter}, migration every {self.migration_interval} iters")

        # Initialize islands
        islands = self._initialize_islands()

        # Global best
        self._update_global_best(islands)

        n_migrations = self.max_iter // self.migration_interval
        if self.max_iter % self.migration_interval != 0:
            n_migrations += 1

        global_iter = 0
        for mig_round in range(n_migrations):
            iters_this_round = min(self.migration_interval,
                                   self.max_iter - global_iter)

            # Each island runs independently
            for island in islands:
                for _ in range(iters_this_round):
                    self._run_island_iteration(island, global_iter, self.max_iter)
                    global_iter += 1
                    if global_iter >= self.max_iter:
                        break
                if global_iter >= self.max_iter:
                    break

            # Migration
            if global_iter < self.max_iter:
                self._migrate(islands)

            self._update_global_best(islands)

            if self.verbose:
                actual = self._actual_value(self.best_value)
                print(f"  MI-PSO round {mig_round+1}/{n_migrations}: "
                      f"global_best={actual:.4f}, calls={self._total_calls}")

        if self.verbose:
            actual = self._actual_value(self.best_value)
            print(f"  MI-PSO final: best={actual:.6f}, "
                  f"total fitness calls={self._total_calls}")

        return self.best_position, self._actual_value(self.best_value)

    def _update_global_best(self, islands):
        """Update global best across all islands"""
        for island in islands:
            val = island['best_value']
            if self.best_value is None or val < self.best_value:
                self.best_value = val
                self.best_position = island['best_position'].copy()

        self.history.append(self._actual_value(self.best_value))
        self.island_bests = [(i['id'], self._actual_value(i['best_value']))
                             for i in islands]

    def get_island_stats(self, islands_data):
        """Return statistics about each island"""
        stats = []
        for i, island in enumerate(islands_data):
            stats.append({
                'id': i,
                'best': self._actual_value(island['best_value']),
                'mean': self._actual_value(np.mean(island['pbest_values'])),
                'std': np.std([self._actual_value(v) for v in island['pbest_values']]),
            })
        return stats
