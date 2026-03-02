# solverang

A silly solver for serious systems.

Domain-agnostic numerical solver for nonlinear equations (F(x) = 0) and least-squares problems (min ||F(x)||^2) in Rust. The core operates on parameter IDs and trait objects, so it works for any domain -- CAD, robotics, circuit simulation, whatever you need to make converge.

Ships with batteries-included 2D sketch, 3D sketch, rigid-body assembly, and legacy geometry constraint modules.

## Quick Start

### Low-level: Problem Trait

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
assert!(result.is_converged());
```

### High-level: Sketch2DBuilder

```rust
use solverang::sketch2d::Sketch2DBuilder;
use solverang::system::SystemStatus;

let mut b = Sketch2DBuilder::new();
let p0 = b.add_fixed_point(0.0, 0.0);
let p1 = b.add_fixed_point(10.0, 0.0);
let p2 = b.add_point(5.0, 1.0);

b.constrain_distance(p0, p1, 10.0);
b.constrain_distance(p1, p2, 8.0);
b.constrain_distance(p2, p0, 6.0);

let mut system = b.build();
let result = system.solve();
assert!(matches!(result.status, SystemStatus::Solved));
```

## Solvers

| Solver | Best for | Notes |
|--------|----------|-------|
| `Solver` (Newton-Raphson) | Square systems (m = n) | Fast quadratic convergence near solution |
| `LMSolver` (Levenberg-Marquardt) | Over-determined (m > n) | Robust, handles poor starting points |
| `AutoSolver` | General use | Auto-selects based on problem structure |
| `RobustSolver` | Unknown problems | Tries NR, falls back to LM |
| `ParallelSolver` | Independent sub-problems | Solves components in parallel |
| `SparseSolver` | Large, sparse systems | Efficient for 1000+ variables |

## Constraint Modules

- **sketch2d** -- 2D points, lines, circles, arcs with 15 constraint types. Ergonomic `Sketch2DBuilder` API.
- **sketch3d** -- 3D points, line segments, planes, axes with 8 constraint types.
- **assembly** -- Rigid bodies with quaternion orientation. Mate, coaxial, insert, and gear ratio constraints.
- **geometry** (legacy) -- Dimension-generic 2D/3D constraint system with 16 constraint types and builder API.

## Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[auto_jacobian]` procedural macro |
| `geometry` | yes | Legacy geometric constraint library |
| `sparse` | yes | Sparse matrix operations (faer) |
| `parallel` | yes | Parallel solving (rayon) |
| `jit` | yes | Cranelift JIT compilation |
| `nist` | yes | NIST StRD test problems |

## License

Apache-2.0
