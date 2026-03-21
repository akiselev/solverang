//! JIT vs Interpreted benchmark suite.
//!
//! Compares JIT-compiled evaluation against interpreted evaluation for:
//! - PCB physics equations (microstrip, differential impedance, skin depth)
//! - Scalable MINPACK problems (BroydenTridiagonal, VariablyDimensioned)
//!
//! Run with: cargo bench -p solverang --bench jit_benchmark --features jit,macros

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use solverang::{auto_jacobian, JITCompiler, Problem};

// ============================================================================
// PCB Physics Problems (macro-annotated → auto JIT)
// ============================================================================

struct MicrostripImpedance {
    height: f64,
    thickness: f64,
    er: f64,
    target_zo: f64,
}

#[auto_jacobian(array_param = "x")]
impl MicrostripImpedance {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        let w = x[0];
        let denom = (self.er + 1.41).sqrt();
        let inner = 5.98 * self.height / (0.8 * w + self.thickness);
        87.0 / denom * inner.ln() - self.target_zo
    }
}

impl Problem for MicrostripImpedance {
    fn name(&self) -> &str { "MicrostripImpedance" }
    fn residual_count(&self) -> usize { 1 }
    fn variable_count(&self) -> usize { 1 }
    fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
    fn initial_point(&self, _: f64) -> Vec<f64> { vec![8.0] }
}

struct DiffImpedance {
    height: f64,
    thickness: f64,
    er: f64,
    target_zdiff: f64,
}

#[auto_jacobian(array_param = "x")]
impl DiffImpedance {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        let w = x[0];
        let s = x[1];
        let z0 = 87.0 / (self.er + 1.41).sqrt()
            * (5.98 * self.height / (0.8 * w + self.thickness)).ln();
        let zdiff = 2.0 * z0 * (1.0 - 0.48 * (-0.96 * s / self.height).exp());
        zdiff - self.target_zdiff
    }
}

impl Problem for DiffImpedance {
    fn name(&self) -> &str { "DiffImpedance" }
    fn residual_count(&self) -> usize { 1 }
    fn variable_count(&self) -> usize { 2 }
    fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
    fn initial_point(&self, _: f64) -> Vec<f64> { vec![5.0, 5.0] }
}

struct SkinDepth {
    resistivity: f64,
    target_depth: f64,
}

#[auto_jacobian(array_param = "x")]
impl SkinDepth {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        let mu_0 = 4.0e-7 * std::f64::consts::PI;
        let delta = (self.resistivity / (std::f64::consts::PI * x[0] * mu_0)).sqrt();
        delta - self.target_depth
    }
}

impl Problem for SkinDepth {
    fn name(&self) -> &str { "SkinDepth" }
    fn residual_count(&self) -> usize { 1 }
    fn variable_count(&self) -> usize { 1 }
    fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
    fn initial_point(&self, _: f64) -> Vec<f64> { vec![1e9] }
}

// ============================================================================
// Benchmark: PCB Equations — JIT vs Interpreted Residual Evaluation
// ============================================================================

fn bench_pcb_residuals(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcb_residual");

    // Microstrip impedance
    let microstrip = MicrostripImpedance {
        height: 4.0,
        thickness: 1.4,
        er: 4.2,
        target_zo: 50.0,
    };
    let microstrip_cc = microstrip.lower_to_compiled_constraints();
    let mut compiler = JITCompiler::new().unwrap();
    let microstrip_jit = compiler.compile(&microstrip_cc).unwrap();

    group.bench_function("microstrip/interpreted", |b| {
        let x = [8.0];
        b.iter(|| black_box(microstrip.residuals(black_box(&x))));
    });

    group.bench_function("microstrip/jit", |b| {
        let x = [8.0];
        let mut r = [0.0];
        b.iter(|| {
            microstrip_jit.evaluate_residuals(black_box(&x), &mut r);
            black_box(r[0]);
        });
    });

    // Differential impedance (2 variables, uses ln + exp)
    let diff = DiffImpedance {
        height: 4.0,
        thickness: 1.4,
        er: 4.2,
        target_zdiff: 100.0,
    };
    let diff_cc = diff.lower_to_compiled_constraints();
    let mut compiler2 = JITCompiler::new().unwrap();
    let diff_jit = compiler2.compile(&diff_cc).unwrap();

    group.bench_function("diff_impedance/interpreted", |b| {
        let x = [5.0, 5.0];
        b.iter(|| black_box(diff.residuals(black_box(&x))));
    });

    group.bench_function("diff_impedance/jit", |b| {
        let x = [5.0, 5.0];
        let mut r = [0.0];
        b.iter(|| {
            diff_jit.evaluate_residuals(black_box(&x), &mut r);
            black_box(r[0]);
        });
    });

    // Skin depth
    let skin = SkinDepth {
        resistivity: 1.68e-8,
        target_depth: 2.0e-6,
    };
    let skin_cc = skin.lower_to_compiled_constraints();
    let mut compiler3 = JITCompiler::new().unwrap();
    let skin_jit = compiler3.compile(&skin_cc).unwrap();

    group.bench_function("skin_depth/interpreted", |b| {
        let x = [1e9];
        b.iter(|| black_box(skin.residuals(black_box(&x))));
    });

    group.bench_function("skin_depth/jit", |b| {
        let x = [1e9];
        let mut r = [0.0];
        b.iter(|| {
            skin_jit.evaluate_residuals(black_box(&x), &mut r);
            black_box(r[0]);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: PCB Equations — JIT vs Interpreted Jacobian Evaluation
// ============================================================================

fn bench_pcb_jacobians(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcb_jacobian");

    let microstrip = MicrostripImpedance {
        height: 4.0,
        thickness: 1.4,
        er: 4.2,
        target_zo: 50.0,
    };
    let cc = microstrip.lower_to_compiled_constraints();
    let mut compiler = JITCompiler::new().unwrap();
    let jit_fn = compiler.compile(&cc).unwrap();

    group.bench_function("microstrip/interpreted", |b| {
        let x = [8.0];
        b.iter(|| black_box(microstrip.jacobian(black_box(&x))));
    });

    group.bench_function("microstrip/jit", |b| {
        let x = [8.0];
        let mut v = vec![0.0; jit_fn.jacobian_nnz()];
        b.iter(|| {
            jit_fn.evaluate_jacobian(black_box(&x), &mut v);
            black_box(v[0]);
        });
    });

    let diff = DiffImpedance {
        height: 4.0,
        thickness: 1.4,
        er: 4.2,
        target_zdiff: 100.0,
    };
    let cc2 = diff.lower_to_compiled_constraints();
    let mut compiler2 = JITCompiler::new().unwrap();
    let jit_fn2 = compiler2.compile(&cc2).unwrap();

    group.bench_function("diff_impedance/interpreted", |b| {
        let x = [5.0, 5.0];
        b.iter(|| black_box(diff.jacobian(black_box(&x))));
    });

    group.bench_function("diff_impedance/jit", |b| {
        let x = [5.0, 5.0];
        let mut v = vec![0.0; jit_fn2.jacobian_nnz()];
        b.iter(|| {
            jit_fn2.evaluate_jacobian(black_box(&x), &mut v);
            black_box(v[0]);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Scalable MINPACK — Interpreted residual/Jacobian throughput
// ============================================================================

fn bench_scalable_minpack(c: &mut Criterion) {
    use solverang::test_problems::{BroydenTridiagonal, Trigonometric, VariablyDimensioned};

    let mut group = c.benchmark_group("scalable_minpack");

    for &n in &[10, 50, 100, 500] {
        let broyden = BroydenTridiagonal::new(n);
        let x0 = broyden.initial_point(1.0);

        group.bench_with_input(
            BenchmarkId::new("broyden_tridiag/residuals", n),
            &n,
            |b, _| {
                b.iter(|| black_box(broyden.residuals(black_box(&x0))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("broyden_tridiag/jacobian", n),
            &n,
            |b, _| {
                b.iter(|| black_box(broyden.jacobian(black_box(&x0))));
            },
        );

        let variably = VariablyDimensioned::new(n);
        let x0v = variably.initial_point(1.0);

        group.bench_with_input(
            BenchmarkId::new("variably_dim/residuals", n),
            &n,
            |b, _| {
                b.iter(|| black_box(variably.residuals(black_box(&x0v))));
            },
        );

        let trig = Trigonometric::new(n);
        let x0t = trig.initial_point(1.0);

        group.bench_with_input(
            BenchmarkId::new("trigonometric/residuals", n),
            &n,
            |b, _| {
                b.iter(|| black_box(trig.residuals(black_box(&x0t))));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark: JIT Compilation Time (compile overhead)
// ============================================================================

fn bench_jit_compile_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile");

    let microstrip = MicrostripImpedance {
        height: 4.0,
        thickness: 1.4,
        er: 4.2,
        target_zo: 50.0,
    };

    group.bench_function("microstrip/compile", |b| {
        let cc = microstrip.lower_to_compiled_constraints();
        b.iter(|| {
            let mut compiler = JITCompiler::new().unwrap();
            black_box(compiler.compile(black_box(&cc)).unwrap());
        });
    });

    let diff = DiffImpedance {
        height: 4.0,
        thickness: 1.4,
        er: 4.2,
        target_zdiff: 100.0,
    };

    group.bench_function("diff_impedance/compile", |b| {
        let cc = diff.lower_to_compiled_constraints();
        b.iter(|| {
            let mut compiler = JITCompiler::new().unwrap();
            black_box(compiler.compile(black_box(&cc)).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pcb_residuals,
    bench_pcb_jacobians,
    bench_scalable_minpack,
    bench_jit_compile_time,
);
criterion_main!(benches);
