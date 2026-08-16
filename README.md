# solverang

> **Experimental.** Solver APIs, constraint modules, and optional execution backends may change while the library is still being validated.

Solverang is a domain-agnostic Rust solver for nonlinear equations and least-squares problems. The core library operates on parameter IDs and problem traits rather than geometry-specific types, and the workspace includes 2D sketch, 3D sketch, and rigid-body assembly constraint modules.

## Installation

```toml
[dependencies]
solverang = "0.1"
```

The default build includes `std` and the `macros` feature for `#[auto_jacobian]`.
Optional subsystems include Cranelift JIT compilation, sparse algebra, parallel solving, and the NIST test problems:

```toml
[dependencies]
solverang = { version = "0.1", features = ["jit", "sparse", "parallel"] }
```

## Examples

### Problem trait

Implement `Problem` directly when you have a system of equations and can provide its Jacobian. This example solves for the square root of two:

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

### Sketch2DBuilder

The builder API describes constraints between geometric entities and constructs the corresponding system. This example defines a triangle by its side lengths:

```rust
use solverang::sketch2d::Sketch2DBuilder;
use solverang::system::SystemStatus;

let mut b = Sketch2DBuilder::new();
let p0 = b.add_fixed_point(0.0, 0.0);
let p1 = b.add_fixed_point(10.0, 0.0);
let p2 = b.add_point(5.0, 1.0);

b.constrain_distance(p0, p1, 10.0).unwrap();
b.constrain_distance(p1, p2, 8.0).unwrap();
b.constrain_distance(p2, p0, 6.0).unwrap();

let mut system = b.build();
let result = system.solve();
assert!(matches!(result.status, SystemStatus::Solved));
```

### ConstraintSystem

For 3D sketches and assemblies, or when finer control is required, construct a `ConstraintSystem` directly with entities and constraints from the `sketch3d` and `assembly` modules.

## Solvers

| Solver | Best for | Behavior |
|--------|----------|----------|
| `Solver` (Newton-Raphson) | Square systems, good initial guesses | Newton-Raphson with quadratic local convergence when the Jacobian and initial point are suitable. |
| `LMSolver` (Levenberg-Marquardt) | Over-determined systems, poor starting points | Interpolates between gradient descent and Gauss-Newton. |
| `AutoSolver` | Automatic solver selection | Inspects problem structure and selects a solver. |
| `RobustSolver` | Problems that may need fallback behavior | Tries Newton-Raphson first and falls back to Levenberg-Marquardt. |
| `ParallelSolver` | Independent sub-problems | Decomposes the system and solves independent components in parallel. Requires the `parallel` feature. |
| `SparseSolver` | Large systems | Uses sparse factorization. |

## Constraint modules

### sketch2d

2D parametric sketch constraints for points, lines, circles, and arcs. The module includes distance, horizontal, vertical, parallel, perpendicular, tangent, point-on-circle, equal-length, angle, midpoint, symmetric, and related constraints. `Sketch2DBuilder` provides the high-level construction API.

### sketch3d

3D sketch primitives include points, line segments, planes, and axes. Constraints include 3D distance, coincident, fixed position, point-on-plane, coplanar, parallel, perpendicular, and coaxial.

### assembly

Rigid-body assembly constraints use quaternion orientation. Entities are rigid bodies with seven parameters (translation plus unit quaternion). Constraints include mate, coaxial alignment, insert, and gear ratio.

## Automatic Jacobians

The `#[auto_jacobian]` procedural macro generates Jacobians through symbolic differentiation:

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

Generated Jacobians can be checked against finite differences:

```rust
use solverang::verify_jacobian;

let result = verify_jacobian(&my_problem, &x, 1e-7, 1e-5);
assert!(result.passed);
```

## Feature flags

| Flag | Default | What it does |
|------|---------|--------------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[auto_jacobian]` procedural macro via `solverang_macros` |
| `sparse` | no | Sparse matrix operations via `faer` |
| `parallel` | no | Parallel solving via `rayon` |
| `jit` | no | Cranelift-based JIT compilation for constraint evaluation |
| `nist` | no | NIST StRD nonlinear regression test problems |

## Architecture

The core library is a general-purpose nonlinear solver that does not depend on a particular application domain. The constraint modules (`sketch2d`, `sketch3d`, and `assembly`) implement extension traits rather than introducing domain types into the solver core.

The solve pipeline has five phases: Decompose, Analyze, Reduce, Solve, and PostProcess. The reduce phase performs symbolic elimination, including substituting fixed parameters, merging coincident parameters, and eliminating trivial constraints, before passing the reduced system to the numerical solver. An incremental dataflow tracker and solution cache support warm starts when constraints change.

See `docs/plans/solver-first-v3.md` for the detailed design.

## Testing

```bash
cargo test --all-features     # everything, including JIT/sparse/parallel/NIST
cargo test                    # default features (core solver + macros)
cargo test --features nist    # include NIST StRD validation
```

The test suite includes unit tests across all modules, a solver megatest, property-based tests via proptest for sketch2d/sketch3d/assembly, contract tests validating trait compliance, MINPACK reference validation, and solver comparison tests.

## Benchmarks

```bash
cargo bench -p solverang
```

Three benchmark suites cover scaling across problem sizes, solver algorithm comparison (NR vs LM vs AutoSolver), and NIST problem performance.

## Project status

See [STATUS.md](STATUS.md) for detailed implementation status, known issues, and architecture notes.

## License

Apache-2.0

## Contributing

Contributions are welcome. Please ensure all tests pass and add tests for new functionality. New constraint types should include finite-difference Jacobian verification.
