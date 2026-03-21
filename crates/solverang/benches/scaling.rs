//! Benchmarks for solver scaling behavior.
//!
//! These benchmarks measure how solver performance scales with problem size
//! and compare different solver strategies.
//!
//! Run with: cargo bench -p solverang --features parallel,sparse

use std::hint::black_box;
use std::time::{Duration, Instant};

use solverang::{
    decompose, DecomposableProblem, LMConfig, LMSolver, ParallelSolver, ParallelSolverConfig,
    Problem, SolverConfig, SparseSolver, SparseSolverConfig,
};

// ============================================================================
// Test Problems
// ============================================================================

/// Diagonal problem - maximally parallel, each constraint independent.
struct DiagonalProblem {
    size: usize,
}

impl Problem for DiagonalProblem {
    fn name(&self) -> &str {
        "diagonal"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| xi * xi - (i as f64 + 1.0))
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| (i, i, 2.0 * xi))
            .collect()
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        (1..=self.size).map(|i| (i as f64).sqrt() * 0.5).collect()
    }
}

impl DecomposableProblem for DiagonalProblem {}

/// Tridiagonal problem - sparse but single component.
struct TridiagonalProblem {
    size: usize,
}

impl Problem for TridiagonalProblem {
    fn name(&self) -> &str {
        "tridiagonal"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = self.size;
        let mut r = Vec::with_capacity(n);

        for i in 0..n {
            let xi = x[i];
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i < n - 1 { x[i + 1] } else { 0.0 };
            r.push(-left + 2.0 * xi - right - 1.0);
        }

        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.size;
        let mut entries = Vec::with_capacity(3 * n - 2);

        for i in 0..n {
            if i > 0 {
                entries.push((i, i - 1, -1.0));
            }
            entries.push((i, i, 2.0));
            if i < n - 1 {
                entries.push((i, i + 1, -1.0));
            }
        }

        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.5; self.size]
    }
}

impl DecomposableProblem for TridiagonalProblem {}

/// Multi-component problem with k independent sub-problems.
struct MultiComponentProblem {
    components: usize,
    vars_per_component: usize,
}

impl Problem for MultiComponentProblem {
    fn name(&self) -> &str {
        "multi-component"
    }

    fn residual_count(&self) -> usize {
        self.components * self.vars_per_component
    }

    fn variable_count(&self) -> usize {
        self.components * self.vars_per_component
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Each component is a simple circle constraint in 2D
        let mut r = Vec::with_capacity(self.residual_count());

        for comp in 0..self.components {
            let offset = comp * self.vars_per_component;
            for i in 0..self.vars_per_component {
                let idx = offset + i;
                r.push(x[idx] - (idx as f64 + 1.0));
            }
        }

        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Each component is independent - diagonal Jacobian within component
        let mut entries = Vec::new();

        for comp in 0..self.components {
            let offset = comp * self.vars_per_component;
            for i in 0..self.vars_per_component {
                let idx = offset + i;
                entries.push((idx, idx, 1.0));
            }
        }

        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0; self.variable_count()]
    }
}

impl DecomposableProblem for MultiComponentProblem {
    fn constraint_graph(&self) -> Vec<(usize, usize)> {
        let mut graph = Vec::new();

        for comp in 0..self.components {
            let offset = comp * self.vars_per_component;
            for i in 0..self.vars_per_component {
                let idx = offset + i;
                graph.push((idx, idx));
            }
        }

        graph
    }
}

// ============================================================================
// Benchmarking Infrastructure
// ============================================================================

/// Run a benchmark function multiple times and report statistics.
fn benchmark<F>(name: &str, iterations: usize, mut f: F)
where
    F: FnMut(),
{
    // Warm up
    for _ in 0..3 {
        f();
    }

    // Timed runs
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    // Compute statistics
    let total: Duration = times.iter().sum();
    let mean = total / iterations as u32;

    times.sort();
    let median = times[iterations / 2];
    let min = times[0];
    let max = times[iterations - 1];

    println!(
        "{}: mean={:?}, median={:?}, min={:?}, max={:?}",
        name, mean, median, min, max
    );
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_problem_sizes() {
    println!("\n=== Problem Size Scaling ===\n");

    let sizes = [10, 50, 100, 500];
    let iterations = 10;

    for &size in &sizes {
        let problem = DiagonalProblem { size };
        let x0 = problem.initial_point(1.0);

        // Newton-Raphson
        benchmark(&format!("NR diagonal n={}", size), iterations, || {
            let solver = solverang::Solver::new(SolverConfig::default());
            let result = solver.solve(&problem, &x0);
            black_box(result);
        });

        // Levenberg-Marquardt
        benchmark(&format!("LM diagonal n={}", size), iterations, || {
            let solver = LMSolver::new(LMConfig::default());
            let result = solver.solve(&problem, &x0);
            black_box(result);
        });

        // Sparse solver
        benchmark(&format!("Sparse diagonal n={}", size), iterations, || {
            let mut solver = SparseSolver::new(SparseSolverConfig::default());
            let result = solver.solve(&problem, &x0);
            black_box(result);
        });

        println!();
    }
}

fn bench_parallel_vs_sequential() {
    println!("\n=== Parallel vs Sequential ===\n");

    let component_counts = [2, 4, 8, 16];
    let vars_per_component = 10;
    let iterations = 10;

    for &components in &component_counts {
        let problem = MultiComponentProblem {
            components,
            vars_per_component,
        };
        let x0 = problem.initial_point(1.0);

        // Verify decomposition
        let decomposed = decompose(&problem);
        println!(
            "Components={}, decomposed into {} parts",
            components,
            decomposed.len()
        );

        // Sequential (using regular NR)
        benchmark(
            &format!("Sequential comp={}", components),
            iterations,
            || {
                let solver = solverang::Solver::new(SolverConfig::default());
                let result = solver.solve(&problem, &x0);
                black_box(result);
            },
        );

        // Parallel solver
        benchmark(&format!("Parallel comp={}", components), iterations, || {
            let solver = ParallelSolver::new(ParallelSolverConfig {
                min_parallel_size: 1,
                min_parallel_components: 2,
                ..Default::default()
            });
            let result = solver.solve(&problem, &x0);
            black_box(result);
        });

        println!();
    }
}

fn bench_sparse_vs_dense() {
    println!("\n=== Sparse vs Dense ===\n");

    let sizes = [20, 50, 100, 200];
    let iterations = 10;

    for &size in &sizes {
        let problem = TridiagonalProblem { size };
        let x0 = problem.initial_point(1.0);

        // Calculate sparsity
        let jacobian = problem.jacobian(&x0);
        let nnz = jacobian.len();
        let total = size * size;
        let density = nnz as f64 / total as f64 * 100.0;
        println!("n={}, nnz={}, density={:.1}%", size, nnz, density);

        // Dense (NR with nalgebra)
        benchmark(&format!("Dense NR n={}", size), iterations, || {
            let solver = solverang::Solver::new(SolverConfig::default());
            let result = solver.solve(&problem, &x0);
            black_box(result);
        });

        // Sparse solver
        benchmark(&format!("Sparse n={}", size), iterations, || {
            let mut solver = SparseSolver::new(SparseSolverConfig::default());
            let result = solver.solve(&problem, &x0);
            black_box(result);
        });

        println!();
    }
}

fn bench_pattern_caching() {
    println!("\n=== Pattern Caching ===\n");

    let size = 100;
    let problem = TridiagonalProblem { size };
    let x0 = problem.initial_point(1.0);
    let iterations = 20;

    // Without caching (new solver each time)
    benchmark("No caching", iterations, || {
        let mut solver = SparseSolver::new(SparseSolverConfig {
            use_pattern_cache: false,
            ..Default::default()
        });
        let result = solver.solve(&problem, &x0);
        black_box(result);
    });

    // With caching (reuse solver)
    let mut cached_solver = SparseSolver::new(SparseSolverConfig {
        use_pattern_cache: true,
        ..Default::default()
    });

    benchmark("With caching", iterations, || {
        let result = cached_solver.solve(&problem, &x0);
        black_box(result);
    });
}

fn bench_decomposition() {
    println!("\n=== Decomposition Overhead ===\n");

    let sizes = [100, 500, 1000];
    let iterations = 100;

    for &size in &sizes {
        let problem = DiagonalProblem { size };

        benchmark(&format!("Decompose n={}", size), iterations, || {
            let components = decompose(&problem);
            black_box(components);
        });
    }
}

fn main() {
    println!("=== solverang Scaling Benchmarks ===");
    println!("Run with: cargo bench -p solverang --features parallel,sparse");
    println!();

    bench_problem_sizes();
    bench_parallel_vs_sequential();
    bench_sparse_vs_dense();
    bench_pattern_caching();
    bench_decomposition();

    println!("\n=== Benchmarks Complete ===");
}
