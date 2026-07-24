"""
pso_solver.py - Particle Swarm Optimization with SA hybridization.
Based on paper: w=0.9->0.4 linear decay, c1=c2=1.49.
"""
import numpy as np
import time as time_module


class PSOSolver:
    """
    Particle Swarm Optimization solver with:
    - Inertia weight linear decay (w: 0.9 -> 0.4)
    - c1=c2=1.49 acceleration coefficients
    - Optional SA hybridization
    - Boundary constraint handling
    """

    def __init__(self, n_particles=50, n_iterations=80,
                 w_start=0.9, w_end=0.4, c1=1.49, c2=1.49,
                 use_sa=False, sa_T0=1.0, sa_cooling=0.95, sa_interval=10,
                 verbose=True):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.use_sa = use_sa
        self.sa_T0 = sa_T0
        self.sa_cooling = sa_cooling
        self.sa_interval = sa_interval
        self.verbose = verbose

        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_position': None,
        }

    def optimize(self, objective_fn, bounds, n_dims=None, maximize=True,
                 init_positions=None, init_radius=0.1):
        """
        Run PSO optimization.

        objective_fn: function(particle_position) -> scalar fitness
        bounds: list of (low, high) for each dimension, or single (low, high) for all
        n_dims: number of dimensions (if bounds is single tuple)
        maximize: True for maximization, False for minimization
        init_positions: optional numpy array of shape (n_dims,) -- if given,
                        particles are initialized around this position with
                        radius init_radius relative to each bound range.
        init_radius: relative radius for scattering around init_positions

        Returns: (best_position, best_fitness, history_dict, elapsed_time)
        """
        # Parse bounds
        if isinstance(bounds, tuple):
            lb, ub = bounds
            bounds_list = [(lb, ub)] * n_dims
        else:
            bounds_list = list(bounds)
            n_dims = len(bounds_list)

        # Initialize particles
        positions = np.zeros((self.n_particles, n_dims))
        velocities = np.zeros((self.n_particles, n_dims))
        for d in range(n_dims):
            lo, hi = bounds_list[d]
            if init_positions is not None:
                # Initialize around the given seed position
                center = init_positions[d]
                scatter = init_radius * (hi - lo)
                positions[:, d] = np.clip(
                    center + np.random.randn(self.n_particles) * scatter, lo, hi
                )
            else:
                positions[:, d] = lo + (hi - lo) * np.random.rand(self.n_particles)

        # Evaluate initial fitness
        fitness = np.array([objective_fn(positions[i]) for i in range(self.n_particles)])

        # Initialize personal best
        p_best_pos = positions.copy()
        p_best_fit = fitness.copy()

        # Initialize global best
        if maximize:
            g_best_idx = np.argmax(fitness)
        else:
            g_best_idx = np.argmin(fitness)
        g_best_pos = positions[g_best_idx].copy()
        g_best_fit = fitness[g_best_idx]

        # SA temperature
        T = self.sa_T0
        start_time = time_module.time()

        for iteration in range(self.n_iterations):
            # Inertia weight linear decay
            w = self.w_start - (self.w_start - self.w_end) * iteration / max(self.n_iterations - 1, 1)

            # Update each particle
            r1 = np.random.rand(self.n_particles, n_dims)
            r2 = np.random.rand(self.n_particles, n_dims)

            velocities = (w * velocities +
                          self.c1 * r1 * (p_best_pos - positions) +
                          self.c2 * r2 * (g_best_pos - positions))

            # Clamp velocities (10% of bound range)
            for d in range(n_dims):
                lo, hi = bounds_list[d]
                v_max = 0.1 * (hi - lo)
                velocities[:, d] = np.clip(velocities[:, d], -v_max, v_max)

            # Update positions
            positions = positions + velocities

            # Enforce bounds (reflect)
            for d in range(n_dims):
                lo, hi = bounds_list[d]
                mask_low = positions[:, d] < lo
                mask_high = positions[:, d] > hi
                positions[mask_low, d] = lo + (lo - positions[mask_low, d])
                positions[mask_high, d] = hi - (positions[mask_high, d] - hi)
                positions[:, d] = np.clip(positions[:, d], lo, hi)

            # Evaluate fitness
            fitness = np.array([objective_fn(positions[i]) for i in range(self.n_particles)])

            # Update personal best
            if maximize:
                improved = fitness > p_best_fit
            else:
                improved = fitness < p_best_fit
            p_best_pos[improved] = positions[improved].copy()
            p_best_fit[improved] = fitness[improved]

            # Update global best
            if maximize:
                best_idx = np.argmax(fitness)
            else:
                best_idx = np.argmin(fitness)
            if (maximize and fitness[best_idx] > g_best_fit) or \
               (not maximize and fitness[best_idx] < g_best_fit):
                g_best_pos = positions[best_idx].copy()
                g_best_fit = fitness[best_idx]

            # SA hybridization: probabilistic acceptance of worse solutions
            if self.use_sa and iteration % self.sa_interval == 0 and iteration > 0:
                # Random perturbation of g_best
                perturbed = g_best_pos.copy()
                for d in range(n_dims):
                    lo, hi = bounds_list[d]
                    perturbed[d] += np.random.normal(0, 0.05 * (hi - lo))
                    perturbed[d] = np.clip(perturbed[d], lo, hi)
                perturbed_fit = objective_fn(perturbed)

                if maximize:
                    delta = perturbed_fit - g_best_fit
                else:
                    delta = -(perturbed_fit - g_best_fit)  # want positive delta for improvement

                if delta > 0 or np.random.rand() < np.exp(delta / max(T, 1e-10)):
                    g_best_pos = perturbed.copy()
                    g_best_fit = perturbed_fit

                T *= self.sa_cooling

            # Record history
            self.history['best_fitness'].append(g_best_fit)
            self.history['mean_fitness'].append(float(np.mean(fitness)))

            if self.verbose and (iteration % max(self.n_iterations // 10, 1) == 0 or
                                 iteration == self.n_iterations - 1):
                print(f"  PSO iter {iteration+1:4d}/{self.n_iterations} | "
                      f"best={g_best_fit:.6f} | mean={np.mean(fitness):.6f} | w={w:.4f}")

        elapsed = time_module.time() - start_time
        if self.verbose:
            print(f"  PSO complete: best={g_best_fit:.6f}, time={elapsed:.3f}s")

        self.history['best_position'] = g_best_pos
        return g_best_pos, g_best_fit, self.history, elapsed
