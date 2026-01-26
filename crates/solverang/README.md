# solverang

A domain-agnostic numerical solver for nonlinear systems and least-squares problems in Rust.

## Overview

`solverang` provides a generic framework for solving nonlinear equation systems F(x) = 0 and nonlinear least-squares problems min ||F(x)||^2. It is designed to be completely independent of any specific domain (e.g., electronic CAD, mechanical CAD) and can serve as a foundation for constraint solvers in various applications.

## Features

- **Multiple Solver Algorithms**: Newton-Raphson, Levenberg-Marquardt, and automatic selection
- **Sparse Matrix Support**: Efficient handling of large, sparse systems via the `sparse` feature
- **Parallel Solving**: Decompose problems into independent components for parallel execution via the `parallel` feature
- **Geometric Constraints**: 2D/3D constraint library for CAD applications via the `geometry` feature
- **MINPACK Test Suite**: Includes all 18 MINPACK least-squares test problems for validation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
solverang = "0.1"

# Optional features
solverang = { version = "0.1", features = ["geometry", "parallel", "sparse"] }
```

## Quick Start

### Basic Problem Solving

Define a problem by implementing the `Problem` trait:

```rust
use solverang::{Problem, Solver, SolverConfig, SolveResult};

// Find x such that x^2 - 2 = 0 (i.e., x = sqrt(2))
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

fn main() {
    let problem = SqrtTwo;
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &[1.5]);

    if let SolveResult::Converged { solution, iterations, residual_norm } = result {
        println!("Solution: x = {} (sqrt(2) = {})", solution[0], std::f64::consts::SQRT_2);
        println!("Converged in {} iterations", iterations);
        println!("Final residual: {}", residual_norm);
    }
}
```

### Over-determined Systems (Least Squares)

Use Levenberg-Marquardt for systems with more equations than unknowns:

```rust
use solverang::{Problem, LMSolver, LMConfig};

// 3 equations, 2 unknowns - find least-squares solution
struct Overdetermined;

impl Problem for Overdetermined {
    fn name(&self) -> &str { "overdetermined" }
    fn residual_count(&self) -> usize { 3 }
    fn variable_count(&self) -> usize { 2 }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![
            x[0] - 1.0,        // x = 1
            x[1] - 2.0,        // y = 2
            x[0] + x[1] - 3.0, // x + y = 3
        ]
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, 0, 1.0), (0, 1, 0.0),
            (1, 0, 0.0), (1, 1, 1.0),
            (2, 0, 1.0), (2, 1, 1.0),
        ]
    }

    fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0] }
}

fn main() {
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&Overdetermined, &[0.0, 0.0]);

    assert!(result.is_converged());
    println!("Solution: {:?}", result.solution());
}
```

### Automatic Solver Selection

Let the library choose the best algorithm:

```rust
use solverang::{Problem, AutoSolver, SolverChoice};

fn solve_any_problem<P: Problem>(problem: &P, x0: &[f64]) {
    let solver = AutoSolver::new();

    // See which solver will be used
    println!("Using solver: {:?}", solver.which_solver(problem));

    let result = solver.solve(problem, x0);
    println!("Result: {:?}", result);
}
```

### Geometric Constraint Solving

Build 2D/3D constraint systems using the fluent builder API:

```rust
use solverang::geometry::{ConstraintSystemBuilder, Point2D};
use solverang::{LMSolver, LMConfig, SolveResult};

fn main() {
    // Create an equilateral triangle with fixed base
    let side = 10.0;
    let height = side * (3.0_f64).sqrt() / 2.0;

    let mut system = ConstraintSystemBuilder::<2>::new()
        .name("EquilateralTriangle")
        .point(Point2D::new(0.0, 0.0))       // p0 - base left
        .point(Point2D::new(side, 0.0))      // p1 - base right
        .point(Point2D::new(side / 2.0, 1.0)) // p2 - apex (initial guess)
        .fix(0)                              // Fix p0 at origin
        .fix(1)                              // Fix p1 on x-axis
        .distance(0, 1, side)                // |p0-p1| = side
        .distance(1, 2, side)                // |p1-p2| = side
        .distance(2, 0, side)                // |p2-p0| = side
        .build();

    // Check degrees of freedom
    println!("DOF: {} (should be 0 for well-constrained)", system.degrees_of_freedom());

    // Solve
    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        let apex = system.get_point(2).unwrap();
        println!("Apex at ({:.3}, {:.3})", apex.x(), apex.y());
        println!("Expected: ({:.3}, {:.3})", side / 2.0, height);
    }
}
```

### Rectangle with Constraints

```rust
use solverang::geometry::{ConstraintSystemBuilder, Point2D};
use solverang::{LMSolver, LMConfig, SolveResult};

fn main() {
    let mut system = ConstraintSystemBuilder::<2>::new()
        .name("Rectangle")
        .point(Point2D::new(0.0, 0.0))     // p0 - bottom-left (fixed)
        .point(Point2D::new(8.0, 0.5))     // p1 - bottom-right (perturbed)
        .point(Point2D::new(7.5, 5.0))     // p2 - top-right (perturbed)
        .point(Point2D::new(0.5, 4.5))     // p3 - top-left (perturbed)
        .fix(0)
        .horizontal(0, 1)                  // Bottom edge horizontal
        .horizontal(3, 2)                  // Top edge horizontal
        .vertical(0, 3)                    // Left edge vertical
        .vertical(1, 2)                    // Right edge vertical
        .distance(0, 1, 10.0)              // Width = 10
        .distance(0, 3, 5.0)               // Height = 5
        .build();

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    if result.is_converged() {
        if let Some(solution) = result.solution() {
            system.set_values(solution);

            for i in 0..4 {
                let p = system.get_point(i).unwrap();
                println!("p{}: ({:.3}, {:.3})", i, p.x(), p.y());
            }
        }
    }
}
```

## Available Solvers

| Solver                           | Best For                 | Notes                                    |
| -------------------------------- | ------------------------ | ---------------------------------------- |
| `Solver` (Newton-Raphson)        | Square systems (m = n)   | Fast quadratic convergence near solution |
| `LMSolver` (Levenberg-Marquardt) | Over-constrained (m > n) | Robust, handles poor starting points     |
| `AutoSolver`                     | General use              | Auto-selects based on problem structure  |
| `RobustSolver`                   | Unknown problems         | Tries NR, falls back to LM               |
| `ParallelSolver`                 | Independent sub-problems | Solves components in parallel            |
| `SparseSolver`                   | Large, sparse systems    | Efficient for 1000+ variables            |

## Solver Configuration

### LM Configuration Presets

```rust
use solverang::LMConfig;

// Fast convergence, fewer iterations
let fast = LMConfig::fast();

// More iterations, handles difficult problems
let robust = LMConfig::robust();

// High precision, tight tolerances
let precise = LMConfig::precise();

// Custom configuration
let custom = LMConfig {
    ftol: 1e-10,       // Function tolerance
    xtol: 1e-10,       // Parameter tolerance
    gtol: 1e-10,       // Gradient tolerance
    stepbound: 100.0,  // Initial step bound
    patience: 200,     // Max iterations
    scale_diag: true,  // Scale by diagonal
};
```

### NR Configuration Presets

```rust
use solverang::SolverConfig;

let fast = SolverConfig::fast();
let robust = SolverConfig::robust();
let precise = SolverConfig::precise();
```

## Available Geometric Constraints

| Constraint                | 2D  | 3D  | Description                |
| ------------------------- | --- | --- | -------------------------- |
| `DistanceConstraint`      | Yes | Yes | Point-to-point distance    |
| `CoincidentConstraint`    | Yes | Yes | Points at same location    |
| `FixedConstraint`         | Yes | Yes | Point at fixed position    |
| `HorizontalConstraint`    | Yes | -   | Same y-coordinate          |
| `VerticalConstraint`      | Yes | -   | Same x-coordinate          |
| `AngleConstraint`         | Yes | -   | Line angle from horizontal |
| `ParallelConstraint`      | Yes | Yes | Parallel lines             |
| `PerpendicularConstraint` | Yes | Yes | Perpendicular lines        |
| `MidpointConstraint`      | Yes | Yes | Point at line midpoint     |
| `PointOnLineConstraint`   | Yes | Yes | Point lies on line         |
| `PointOnCircleConstraint` | Yes | Yes | Point on circle/sphere     |
| `LineTangentConstraint`   | Yes | -   | Line tangent to circle     |
| `CircleTangentConstraint` | Yes | Yes | Circle/sphere tangency     |
| `SymmetricConstraint`     | Yes | Yes | Point symmetry             |
| `CollinearConstraint`     | Yes | Yes | Collinear segments         |
| `EqualLengthConstraint`   | Yes | Yes | Equal line lengths         |

## Jacobian Verification

Verify your analytical Jacobians against finite differences:

```rust
use solverang::{verify_jacobian, Problem};

fn verify_my_problem<P: Problem>(problem: &P, x: &[f64]) {
    let result = verify_jacobian(problem, x, 1e-7, 1e-5);

    if result.passed {
        println!("Jacobian OK (max error: {})", result.max_absolute_error);
    } else {
        println!("Jacobian ERROR at {:?}: {}",
            result.max_error_location,
            result.max_absolute_error);
    }
}
```

## Feature Flags

- `std` (default): Standard library support
- `parallel`: Enable parallel component solving with rayon
- `sparse`: Enable sparse matrix operations with faer
- `geometry`: Enable geometric constraint library for 2D/3D CAD

## Performance Tips

1. **Use Sparse Solver for Large Systems**: For systems with 100+ variables and sparse Jacobians, `SparseSolver` can be 10x faster.

2. **Leverage Decomposition**: If your problem decomposes into independent components, `ParallelSolver` automatically parallelizes.

3. **Good Initial Guesses**: Both NR and LM converge faster with good starting points.

4. **Use AutoSolver**: When unsure, `AutoSolver` makes reasonable choices based on problem structure.

5. **Enable Pattern Caching**: For repeated solves with same sparsity pattern, reuse the `SparseSolver` instance.

## Running Tests

```bash
# All tests
cargo test -p solverang --features geometry,parallel,sparse

# Property-based tests
cargo test -p solverang --features geometry prop_

# MINPACK validation
cargo test -p solverang minpack
```

## Running Benchmarks

```bash
# All benchmarks
cargo bench -p solverang --features geometry,parallel,sparse

# With HTML reports
cargo bench -p solverang --features geometry,parallel,sparse -- --save-baseline main
```

## License

Apache-2.0

## Contributing

Contributions are welcome. Please ensure all tests pass and add appropriate tests for new functionality.
