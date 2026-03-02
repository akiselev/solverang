# solverang

A silly solver for serious systems.

Solverang is a vibe-coded, domain-agnostic numerical solver for nonlinear equations and least-squares problems, written in Rust. It will find the zeros of your functions or die trying (gracefully, with diagnostics). The core library knows nothing about geometry, CAD, or any other domain -- it operates purely on parameter IDs and trait objects, which means you can bolt it onto whatever problem you have and it will dutifully attempt to make your residuals vanish.

It ships with batteries-included 2D sketch, 3D sketch, and rigid-body assembly constraint modules, because what is a solver without something to solve.

## Installation

```toml
[dependencies]
solverang = "0.1"
```

All the interesting features are on by default. If you want to be selective:

```toml
[dependencies]
solverang = { version = "0.1", default-features = false, features = ["std"] }
```

## A Tour of the Silly Solver

### The Low Road: Problem Trait

If you have a system of equations and know what a Jacobian is, implement the `Problem` trait and hand it to a solver. Here we find the square root of two, which humanity has needed since at least 1800 BC:

```rust
use solverang::{Problem, Solver, SolverConfig, SolveResult};

struct SqrtTwo;

impl Problem for SqrtTwo {
    fn name(&self) -> &str { "sqrt(2)" }
    fn residual_count(&self) -> usize { 1 }
    fn variable_count(&self) -> usize { 1 }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] - 2.0]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 2.0 * x[0])]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![1.0 * factor]
    }
}

let solver = Solver::new(SolverConfig::default());
let result = solver.solve(&SqrtTwo, &[1.5]);

if let SolveResult::Converged { solution, .. } = result {
    assert!((solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6);
}
```

### The High Road: Sketch2DBuilder

If you would rather describe constraints between geometric entities and let the solver figure out the rest, use the builder API. Here we define a triangle by its three side lengths and ask the solver where the vertices go:

```rust
use solverang::sketch2d::Sketch2DBuilder;
use solverang::system::SystemStatus;

let mut b = Sketch2DBuilder::new();
let p0 = b.add_fixed_point(0.0, 0.0);  // nail this one down
let p1 = b.add_fixed_point(10.0, 0.0); // and this one
let p2 = b.add_point(5.0, 1.0);        // this one has to figure itself out

b.constrain_distance(p0, p1, 10.0);
b.constrain_distance(p1, p2, 8.0);
b.constrain_distance(p2, p0, 6.0);

let mut system = b.build();
let result = system.solve();
assert!(matches!(result.status, SystemStatus::Solved));
```

### The Middle Road: ConstraintSystem

For 3D sketches and assemblies, or when you want finer control, construct a `ConstraintSystem` directly with entities and constraints from the `sketch3d` and `assembly` modules.

## Solvers

Six solvers walk into a function space:

| Solver | Best for | Personality |
|--------|----------|-------------|
| `Solver` (Newton-Raphson) | Square systems, good initial guesses | Fast and fragile. Quadratic convergence when it works, spectacular failure when it doesn't. |
| `LMSolver` (Levenberg-Marquardt) | Over-determined systems, poor starting points | The reliable one. Interpolates between gradient descent and Gauss-Newton so you don't have to. |
| `AutoSolver` | When you don't want to think about it | Inspects your problem and picks a solver. Usually right. |
| `RobustSolver` | Unknown territory | Tries Newton-Raphson first, then falls back to LM. Belt and suspenders. |
| `ParallelSolver` | Independent sub-problems | Decomposes your system and solves the pieces in parallel. Requires the `parallel` feature. |
| `SparseSolver` | Large systems (1000+ variables) | Uses sparse factorization. The difference between "runs" and "runs before the heat death of the universe." |

## Constraint Modules

### sketch2d

2D parametric sketch constraints, the kind you find in any CAD sketcher. Points, lines, circles, arcs, and 15 constraint types including distance, horizontal, vertical, parallel, perpendicular, tangent, point-on-circle, equal length, angle, midpoint, and symmetric. The `Sketch2DBuilder` provides an ergonomic API for constructing these systems.

### sketch3d

3D sketch primitives: points, line segments, planes, and axes. Constraints include 3D distance, coincident, fixed position, point-on-plane, coplanar, parallel, perpendicular, and coaxial.

### assembly

Rigid-body assembly constraints using quaternion orientation. Entities are rigid bodies with 7 parameters (translation + unit quaternion). Constraints include mate (point coincidence), coaxial alignment, insert (coaxial + flush), and gear ratio.

## Automatic Jacobians

Writing Jacobians by hand is tedious and error-prone. The `#[auto_jacobian]` procedural macro generates them via symbolic differentiation:

```rust
use solverang::{auto_jacobian, residual, Problem};

#[auto_jacobian(array_param = "x")]
struct CircleLine {
    radius: f64,
    slope: f64,
    intercept: f64,
}

#[residual]
fn circle(x: &[f64]) -> f64 {
    x[0] * x[0] + x[1] * x[1] - self.radius * self.radius
}

#[residual]
fn line(x: &[f64]) -> f64 {
    x[1] - self.slope * x[0] - self.intercept
}
```

If you don't trust it (reasonable), verify against finite differences:

```rust
use solverang::verify_jacobian;

let result = verify_jacobian(&my_problem, &x, 1e-7, 1e-5);
assert!(result.passed);
```

## Feature Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[auto_jacobian]` procedural macro via `solverang_macros` |
| `sparse` | yes | Sparse matrix operations via `faer` |
| `parallel` | yes | Parallel solving via `rayon` |
| `jit` | yes | Cranelift-based JIT compilation for constraint evaluation |
| `nist` | yes | NIST StRD nonlinear regression test problems |

## Architecture

The design follows a principle that could be called "solver-first": the core library is a general-purpose nonlinear solver that knows nothing about the domain it serves. The constraint modules (sketch2d, sketch3d, assembly) implement extension traits, making them plugins rather than core dependencies.

The solve pipeline is pluggable, with five phases: Decompose, Analyze, Reduce, Solve, PostProcess. The reduce phase performs symbolic elimination (substituting fixed parameters, merging coincident parameters, eliminating trivial constraints) before handing the reduced system to the numerical solver. An incremental dataflow tracker and solution cache enable warm-starting when constraints change.

For the gory details, see `docs/plans/solver-first-v3.md`.

## Testing

```bash
cargo test                    # everything (default features include all modules)
cargo test -p solverang       # just the solver
cargo test --features nist    # include NIST StRD validation
```

The test suite includes unit tests across all modules, a solver megatest, property-based tests via proptest for sketch2d/sketch3d/assembly, contract tests validating trait compliance, MINPACK reference validation, and solver comparison tests.

## Benchmarks

```bash
cargo bench -p solverang
```

Three benchmark suites: scaling behavior across problem sizes, solver algorithm comparison (NR vs LM vs AutoSolver), and NIST problem performance.

## Project Status

See [STATUS.md](STATUS.md) for detailed implementation status, known issues, and architecture notes.

## License

Apache-2.0

## Contributing

Contributions are welcome. Please ensure all tests pass and add tests for new functionality. If you are adding a new constraint type, include finite-difference Jacobian verification in your tests -- the solver is only as good as its derivatives.
