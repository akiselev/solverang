//! Comprehensive benchmarks for solverang.
//!
//! This benchmark suite compares:
//! - Newton-Raphson vs Levenberg-Marquardt vs AutoSolver across problem sizes
//! - Sparse vs dense matrix operations to find crossover points
//! - Parallel vs sequential solving for decomposable problems
//! - Pattern caching effectiveness
//!
//! Run with:
//! ```bash
//! cargo bench -p solverang --features geometry,parallel,sparse
//! ```
//!
//! For HTML reports:
//! ```bash
//! cargo bench -p solverang --features geometry,parallel,sparse -- --save-baseline main
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use solverang::{
    decompose, AutoSolver, DecomposableProblem, LMConfig, LMSolver, Problem, SolverConfig,
    SparseSolver, SparseSolverConfig,
};

#[cfg(feature = "parallel")]
use solverang::{ParallelSolver, ParallelSolverConfig};

#[cfg(feature = "geometry")]
use solverang::geometry::{
    constraints::{DistanceConstraint, HorizontalConstraint},
    ConstraintSystem, ConstraintSystemBuilder, Point2D,
};

// =============================================================================
// Test Problem Definitions
// =============================================================================

/// Diagonal problem - maximally parallel, each equation independent.
/// F_i(x) = x_i^2 - (i+1) = 0, solution: x_i = sqrt(i+1)
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

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        (1..=self.size)
            .map(|i| (i as f64).sqrt() * 0.5 * factor)
            .collect()
    }
}

impl DecomposableProblem for DiagonalProblem {}

/// Tridiagonal problem - sparse, single component.
/// -x_{i-1} + 2*x_i - x_{i+1} = 1
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

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.5 * factor; self.size]
    }
}

impl DecomposableProblem for TridiagonalProblem {}

/// Dense nonlinear problem - all variables coupled.
/// F_i(x) = sum_j(x_j * (i+j+2)/(j+1)) - (i+1) = 0
struct DenseNonlinearProblem {
    size: usize,
}

impl Problem for DenseNonlinearProblem {
    fn name(&self) -> &str {
        "dense-nonlinear"
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
            let mut sum = 0.0;
            for j in 0..n {
                sum += x[j] * ((i + j + 2) as f64) / ((j + 1) as f64);
            }
            r.push(sum - (i as f64 + 1.0));
        }

        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.size;
        let mut entries = Vec::with_capacity(n * n);

        for i in 0..n {
            for j in 0..n {
                let val = ((i + j + 2) as f64) / ((j + 1) as f64);
                entries.push((i, j, val));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.1 * factor; self.size]
    }
}

/// Multi-component problem - k independent sub-problems.
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
        let mut r = Vec::with_capacity(self.residual_count());

        for comp in 0..self.components {
            let offset = comp * self.vars_per_component;
            for i in 0..self.vars_per_component {
                let idx = offset + i;
                // Simple linear residual within component
                r.push(x[idx] - (idx as f64 + 1.0));
            }
        }

        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
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

/// Rosenbrock function (classic optimization test).
/// F(x) = [10*(x_2 - x_1^2), 1 - x_1]
struct RosenbrockProblem;

impl Problem for RosenbrockProblem {
    fn name(&self) -> &str {
        "rosenbrock"
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, 0, -20.0 * x[0]),
            (0, 1, 10.0),
            (1, 0, -1.0),
            (1, 1, 0.0),
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-1.0 * factor, 1.0 * factor]
    }
}

// =============================================================================
// Benchmark Functions
// =============================================================================

/// Benchmark NR vs LM vs Auto across problem sizes.
fn bench_solver_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_comparison");

    let sizes = [5, 10, 25, 50, 100];

    for &size in &sizes {
        let problem = DiagonalProblem { size };
        let x0 = problem.initial_point(1.0);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("NR", size), &size, |b, _| {
            let solver = solverang::Solver::new(SolverConfig::default());
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("LM", size), &size, |b, _| {
            let solver = LMSolver::new(LMConfig::default());
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Auto", size), &size, |b, _| {
            let solver = AutoSolver::new();
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark sparse vs dense crossover point.
fn bench_sparse_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense");

    // Tridiagonal problems of increasing size
    let sizes = [10, 25, 50, 100, 200, 500];

    for &size in &sizes {
        let problem = TridiagonalProblem { size };
        let x0 = problem.initial_point(1.0);

        // Calculate sparsity for reference
        let nnz = 3 * size - 2; // tridiagonal has 3*n - 2 non-zeros
        let total = size * size;
        let density = (nnz as f64 / total as f64) * 100.0;

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("Dense_d={:.1}%", density), size),
            &size,
            |b, _| {
                let solver = solverang::Solver::new(SolverConfig::default());
                b.iter(|| {
                    let result = solver.solve(&problem, &x0);
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Sparse_d={:.1}%", density), size),
            &size,
            |b, _| {
                let mut solver = SparseSolver::new(SparseSolverConfig::default());
                b.iter(|| {
                    let result = solver.solve(&problem, &x0);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dense problems to show where dense solver excels.
fn bench_dense_problems(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_problems");

    let sizes = [5, 10, 20, 30, 50];

    for &size in &sizes {
        let problem = DenseNonlinearProblem { size };
        let x0 = problem.initial_point(1.0);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("LM", size), &size, |b, _| {
            let solver = LMSolver::new(LMConfig::default());
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Sparse", size), &size, |b, _| {
            let mut solver = SparseSolver::new(SparseSolverConfig::default());
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark parallel vs sequential for decomposable problems.
#[cfg(feature = "parallel")]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    let component_configs = [(2, 20), (4, 20), (8, 20), (16, 10), (32, 5)];

    for &(components, vars_per) in &component_configs {
        let problem = MultiComponentProblem {
            components,
            vars_per_component: vars_per,
        };
        let x0 = problem.initial_point(1.0);
        let total_vars = components * vars_per;

        group.throughput(Throughput::Elements(total_vars as u64));

        let label = format!("{}x{}", components, vars_per);

        group.bench_with_input(BenchmarkId::new("Sequential", &label), &label, |b, _| {
            let solver = solverang::Solver::new(SolverConfig::default());
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Parallel", &label), &label, |b, _| {
            let solver = ParallelSolver::new(ParallelSolverConfig {
                min_parallel_size: 1,
                min_parallel_components: 2,
                ..Default::default()
            });
            b.iter(|| {
                let result = solver.solve(&problem, &x0);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark decomposition overhead.
fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");

    let sizes = [50, 100, 500, 1000];

    for &size in &sizes {
        let problem = DiagonalProblem { size };

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("decompose", size), &size, |b, _| {
            b.iter(|| {
                let components = decompose(&problem);
                black_box(components)
            });
        });
    }

    group.finish();
}

/// Benchmark pattern caching effectiveness.
fn bench_pattern_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_caching");

    let size = 100;
    let problem = TridiagonalProblem { size };
    let x0 = problem.initial_point(1.0);

    // Without caching - new solver each time
    group.bench_function("no_cache", |b| {
        b.iter(|| {
            let mut solver = SparseSolver::new(SparseSolverConfig {
                use_pattern_cache: false,
                ..Default::default()
            });
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    // With caching - reuse solver
    let mut cached_solver = SparseSolver::new(SparseSolverConfig {
        use_pattern_cache: true,
        ..Default::default()
    });
    // Warm up to populate cache
    let _ = cached_solver.solve(&problem, &x0);

    group.bench_function("with_cache", |b| {
        b.iter(|| {
            let result = cached_solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark classic optimization problems.
fn bench_classic_problems(c: &mut Criterion) {
    let mut group = c.benchmark_group("classic_problems");

    // Rosenbrock
    let rosenbrock = RosenbrockProblem;
    let x0_rb = rosenbrock.initial_point(1.0);

    group.bench_function("Rosenbrock_NR", |b| {
        let solver = solverang::Solver::new(SolverConfig::default());
        b.iter(|| {
            let result = solver.solve(&rosenbrock, &x0_rb);
            black_box(result)
        });
    });

    group.bench_function("Rosenbrock_LM", |b| {
        let solver = LMSolver::new(LMConfig::default());
        b.iter(|| {
            let result = solver.solve(&rosenbrock, &x0_rb);
            black_box(result)
        });
    });

    group.bench_function("Rosenbrock_Robust", |b| {
        let solver = solverang::RobustSolver::new();
        b.iter(|| {
            let result = solver.solve(&rosenbrock, &x0_rb);
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark geometric constraint systems.
#[cfg(feature = "geometry")]
fn bench_geometric_systems(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_systems");

    // Triangle benchmark
    {
        let system = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(10.0, 0.0))
            .point(Point2D::new(5.0, 1.0))
            .fix(0)
            .horizontal(0, 1)
            .distance(0, 1, 10.0)
            .distance(1, 2, 8.0)
            .distance(2, 0, 6.0)
            .build();

        let x0 = system.current_values();

        group.bench_function("Triangle_LM", |b| {
            let solver = LMSolver::new(LMConfig::default());
            b.iter(|| {
                let result = solver.solve(&system, &x0);
                black_box(result)
            });
        });
    }

    // Rectangle benchmark
    {
        let system = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(10.0, 0.5))
            .point(Point2D::new(10.5, 5.0))
            .point(Point2D::new(0.5, 5.5))
            .fix(0)
            .horizontal(0, 1)
            .horizontal(3, 2)
            .vertical(0, 3)
            .vertical(1, 2)
            .distance(0, 1, 10.0)
            .distance(0, 3, 5.0)
            .build();

        let x0 = system.current_values();

        group.bench_function("Rectangle_LM", |b| {
            let solver = LMSolver::new(LMConfig::default());
            b.iter(|| {
                let result = solver.solve(&system, &x0);
                black_box(result)
            });
        });
    }

    // Chain of points (scaling test)
    for &num_points in &[5, 10, 20, 50] {
        let mut system = ConstraintSystem::<2>::new();

        // Create chain of points
        for i in 0..num_points {
            system.add_point(Point2D::new(i as f64 * 2.0 + 0.1 * i as f64, 0.1 * i as f64));
        }
        system.fix_point(0);

        // Distance constraints between adjacent points
        for i in 0..num_points - 1 {
            system.add_constraint(Box::new(DistanceConstraint::<2>::new(i, i + 1, 2.0)));
        }

        // Horizontal constraint for first segment
        system.add_constraint(Box::new(HorizontalConstraint::new(0, 1)));

        let x0 = system.current_values();

        group.bench_with_input(
            BenchmarkId::new("Chain_LM", num_points),
            &num_points,
            |b, _| {
                let solver = LMSolver::new(LMConfig::default());
                b.iter(|| {
                    let result = solver.solve(&system, &x0);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark solver configuration presets.
fn bench_config_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_presets");

    let problem = DiagonalProblem { size: 50 };
    let x0 = problem.initial_point(1.0);

    group.bench_function("LM_default", |b| {
        let solver = LMSolver::new(LMConfig::default());
        b.iter(|| {
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.bench_function("LM_fast", |b| {
        let solver = LMSolver::new(LMConfig::fast());
        b.iter(|| {
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.bench_function("LM_robust", |b| {
        let solver = LMSolver::new(LMConfig::robust());
        b.iter(|| {
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.bench_function("LM_precise", |b| {
        let solver = LMSolver::new(LMConfig::precise());
        b.iter(|| {
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.bench_function("NR_default", |b| {
        let solver = solverang::Solver::new(SolverConfig::default());
        b.iter(|| {
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.bench_function("NR_fast", |b| {
        let solver = solverang::Solver::new(SolverConfig::fast());
        b.iter(|| {
            let result = solver.solve(&problem, &x0);
            black_box(result)
        });
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_solver_comparison,
    bench_sparse_vs_dense,
    bench_dense_problems,
    bench_parallel_vs_sequential,
    bench_decomposition,
    bench_pattern_caching,
    bench_classic_problems,
    bench_config_presets,
);

#[cfg(all(feature = "geometry", feature = "parallel"))]
criterion_group!(
    geometry_benches,
    bench_geometric_systems,
);

#[cfg(all(feature = "geometry", feature = "parallel"))]
criterion_main!(benches, geometry_benches);

#[cfg(all(feature = "parallel", not(feature = "geometry")))]
criterion_main!(benches);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches_no_parallel,
    bench_solver_comparison,
    bench_sparse_vs_dense,
    bench_dense_problems,
    bench_decomposition,
    bench_pattern_caching,
    bench_classic_problems,
    bench_config_presets,
);

#[cfg(all(not(feature = "parallel"), feature = "geometry"))]
criterion_group!(
    geometry_benches_no_parallel,
    bench_geometric_systems,
);

#[cfg(all(not(feature = "parallel"), feature = "geometry"))]
criterion_main!(benches_no_parallel, geometry_benches_no_parallel);

#[cfg(all(not(feature = "parallel"), not(feature = "geometry")))]
criterion_main!(benches_no_parallel);
