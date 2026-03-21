//! NIST StRD Nonlinear Regression Benchmark Suite.
//!
//! This benchmark suite tests the solver against the NIST Statistical Reference
//! Datasets for nonlinear regression. All 27 problems are included with their
//! certified values.
//!
//! Run with:
//! ```bash
//! cargo bench -p solverang --features nist -- nist
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use solverang::test_problems::nist::all_problems_by_difficulty;
use solverang::{LMConfig, LMSolver};

/// Benchmark NIST problems grouped by difficulty.
fn bench_nist_by_difficulty(c: &mut Criterion) {
    let problems_by_difficulty = all_problems_by_difficulty();

    for (difficulty, problems) in problems_by_difficulty {
        let group_name = format!("NIST_{}", difficulty);
        let mut group = c.benchmark_group(&group_name);

        // Use more iterations for faster problems
        group.sample_size(50);

        for problem in problems {
            let name = problem.name();
            let x0_close = problem.starting_values_2();

            // Benchmark with starting values closer to solution
            group.bench_with_input(BenchmarkId::new("sv2", name), &x0_close, |b, x0| {
                let solver = LMSolver::new(LMConfig::default());
                b.iter(|| {
                    let result = solver.solve(problem.as_ref(), x0);
                    black_box(result)
                });
            });
        }

        group.finish();
    }
}

/// Benchmark all NIST problems with robust solver configuration.
fn bench_nist_robust(c: &mut Criterion) {
    let mut group = c.benchmark_group("NIST_Robust");
    group.sample_size(30);

    let problems_by_difficulty = all_problems_by_difficulty();

    for (difficulty, problems) in problems_by_difficulty {
        for problem in problems {
            let name = format!("{}_{}", difficulty, problem.name());

            // Use starting values set 1 (farther from solution) with robust config
            let x0 = problem.starting_values_1();

            group.bench_with_input(BenchmarkId::new("robust", &name), &x0, |b, x0| {
                let solver = LMSolver::new(LMConfig::robust());
                b.iter(|| {
                    let result = solver.solve(problem.as_ref(), x0);
                    black_box(result)
                });
            });
        }
    }

    group.finish();
}

/// Verify all NIST problems converge to certified values.
fn bench_nist_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("NIST_Verification");
    group.sample_size(20);

    let tolerance = 1e-4; // Relative tolerance for parameter comparison

    let problems_by_difficulty = all_problems_by_difficulty();
    let solver = LMSolver::new(LMConfig::robust());

    for (difficulty, problems) in &problems_by_difficulty {
        for problem in problems {
            let name = format!("{}_{}", difficulty, problem.name());
            let x0 = problem.starting_values_2();

            group.bench_function(&name, |b| {
                b.iter(|| {
                    let result = solver.solve(problem.as_ref(), &x0);

                    // Verify solution matches certified values
                    if let Some(solution) = result.solution() {
                        let verify_result = problem.verify_solution(solution, tolerance);
                        black_box(verify_result)
                    } else {
                        black_box(Err("No solution".to_string()))
                    }
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    nist_benches,
    bench_nist_by_difficulty,
    bench_nist_robust,
    bench_nist_verification,
);

criterion_main!(nist_benches);
