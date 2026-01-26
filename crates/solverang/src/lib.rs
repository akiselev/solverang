//! Domain-agnostic numerical solver for nonlinear systems and least-squares problems.
//!
//! This crate provides a generic framework for solving nonlinear equation systems
//! and nonlinear least-squares problems. It is designed to be completely independent
//! of any specific domain (e.g., electronic CAD, mechanical CAD) and can be used
//! as a foundation for constraint solvers in various applications.
//!
//! # Overview
//!
//! `solverang` supports finding solutions to problems of the form:
//!
//! - **Nonlinear equations**: Find x such that F(x) = 0
//! - **Least-squares**: Minimize ||F(x)||^2
//!
//! The library provides multiple solver algorithms, each optimized for different
//! problem characteristics, along with utilities for problem decomposition, sparse
//! matrix handling, and Jacobian verification.
//!
//! # Core Concepts
//!
//! ## The Problem Trait
//!
//! All problems implement the [`Problem`] trait, which defines:
//!
//! - `residuals(x)` - Evaluate F(x), returning a vector of residual values
//! - `jacobian(x)` - Compute the Jacobian matrix J where J\[i,j\] = dF\[i\]/dx\[j\]
//! - `residual_count()` / `variable_count()` - Problem dimensions (m equations, n variables)
//!
//! ## Available Solvers
//!
//! | Solver | Best For | Notes |
//! |--------|----------|-------|
//! | [`Solver`] | Square systems (m = n) | Fast quadratic convergence near solution |
//! | [`LMSolver`] | Over-constrained (m > n) | Robust, handles poor starting points |
//! | [`AutoSolver`] | General use | Auto-selects based on problem structure |
//! | [`RobustSolver`] | Unknown problems | Tries NR, falls back to LM |
//! | [`ParallelSolver`] | Independent sub-problems | Solves components in parallel |
//! | [`SparseSolver`] | Large, sparse systems | Efficient for 1000+ variables |
//!
//! ## Solver Selection Guidelines
//!
//! - **Square systems (m == n) with good initial guess**: Use [`Solver`] (Newton-Raphson)
//! - **Over-determined systems (m > n)**: Use [`LMSolver`] (Levenberg-Marquardt)
//! - **Under-determined systems (m < n)**: Use [`LMSolver`]
//! - **Unknown problem characteristics**: Use [`AutoSolver`] or [`RobustSolver`]
//! - **Large sparse systems (n > 100)**: Use [`SparseSolver`]
//! - **Problems that decompose into independent parts**: Use [`ParallelSolver`]
//!
//! # Quick Start
//!
//! ## Basic Problem Solving
//!
//! ```rust
//! use solverang::{Problem, Solver, SolverConfig, SolveResult};
//!
//! // Define a simple problem: find x such that x^2 - 2 = 0
//! struct SqrtTwo;
//!
//! impl Problem for SqrtTwo {
//!     fn name(&self) -> &str { "sqrt(2)" }
//!     fn residual_count(&self) -> usize { 1 }
//!     fn variable_count(&self) -> usize { 1 }
//!
//!     fn residuals(&self, x: &[f64]) -> Vec<f64> {
//!         vec![x[0] * x[0] - 2.0]
//!     }
//!
//!     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
//!         vec![(0, 0, 2.0 * x[0])]
//!     }
//!
//!     fn initial_point(&self, factor: f64) -> Vec<f64> {
//!         vec![1.0 * factor]
//!     }
//! }
//!
//! let problem = SqrtTwo;
//! let solver = Solver::new(SolverConfig::default());
//! let result = solver.solve(&problem, &[1.5]);
//!
//! if let SolveResult::Converged { solution, .. } = result {
//!     assert!((solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6);
//! }
//! ```
//!
//! ## Levenberg-Marquardt for Over-constrained Systems
//!
//! Use LM when you have more equations than unknowns:
//!
//! ```rust
//! use solverang::{Problem, LMSolver, LMConfig};
//!
//! // Over-determined system: 3 equations, 2 unknowns
//! struct Overdetermined;
//!
//! impl Problem for Overdetermined {
//!     fn name(&self) -> &str { "overdetermined" }
//!     fn residual_count(&self) -> usize { 3 }
//!     fn variable_count(&self) -> usize { 2 }
//!
//!     fn residuals(&self, x: &[f64]) -> Vec<f64> {
//!         vec![x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 3.0]
//!     }
//!
//!     fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
//!         vec![
//!             (0, 0, 1.0), (0, 1, 0.0),
//!             (1, 0, 0.0), (1, 1, 1.0),
//!             (2, 0, 1.0), (2, 1, 1.0),
//!         ]
//!     }
//!
//!     fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0] }
//! }
//!
//! let solver = LMSolver::new(LMConfig::default());
//! let result = solver.solve(&Overdetermined, &[0.0, 0.0]);
//! assert!(result.is_converged());
//! ```
//!
//! ## Automatic Solver Selection
//!
//! Let the library choose the best algorithm based on problem characteristics:
//!
//! ```rust
//! use solverang::{Problem, AutoSolver, SolverChoice};
//!
//! # struct MyProblem;
//! # impl Problem for MyProblem {
//! #     fn name(&self) -> &str { "my" }
//! #     fn residual_count(&self) -> usize { 2 }
//! #     fn variable_count(&self) -> usize { 2 }
//! #     fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0] - 1.0, x[1] - 2.0] }
//! #     fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
//! #         vec![(0, 0, 1.0), (1, 1, 1.0)]
//! #     }
//! #     fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0] }
//! # }
//! let problem = MyProblem;
//! let solver = AutoSolver::new();
//!
//! // See which solver will be used
//! println!("Selected solver: {:?}", solver.which_solver(&problem));
//!
//! let result = solver.solve(&problem, &[0.0, 0.0]);
//! ```
//!
//! ## Jacobian Verification
//!
//! Verify your analytical Jacobians against finite differences:
//!
//! ```rust
//! use solverang::{verify_jacobian, Problem};
//!
//! # struct MyProblem;
//! # impl Problem for MyProblem {
//! #     fn name(&self) -> &str { "my" }
//! #     fn residual_count(&self) -> usize { 1 }
//! #     fn variable_count(&self) -> usize { 1 }
//! #     fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0] * x[0]] }
//! #     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
//! #         vec![(0, 0, 2.0 * x[0])]
//! #     }
//! #     fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0] }
//! # }
//! let problem = MyProblem;
//! let x = vec![2.0];
//!
//! let result = verify_jacobian(&problem, &x, 1e-7, 1e-5);
//!
//! if result.passed {
//!     println!("Jacobian OK (max error: {})", result.max_absolute_error);
//! } else {
//!     println!("Jacobian ERROR at {:?}: {}",
//!         result.max_error_location,
//!         result.max_absolute_error);
//! }
//! ```
//!
//! # Geometric Constraints (Feature: `geometry`)
//!
//! With the `geometry` feature, you can build 2D and 3D constraint systems:
//!
//! ```rust,ignore
//! use solverang::geometry::{ConstraintSystemBuilder, Point2D};
//! use solverang::{LMSolver, LMConfig, SolveResult};
//!
//! // Create a triangle with fixed side lengths
//! let mut system = ConstraintSystemBuilder::<2>::new()
//!     .name("Triangle")
//!     .point(Point2D::new(0.0, 0.0))    // p0 - fixed at origin
//!     .point(Point2D::new(10.0, 0.0))   // p1 - on x-axis
//!     .point(Point2D::new(5.0, 1.0))    // p2 - apex (initial guess)
//!     .fix(0)                           // Fix p0
//!     .fix(1)                           // Fix p1
//!     .distance(0, 1, 10.0)             // |p0-p1| = 10
//!     .distance(1, 2, 8.0)              // |p1-p2| = 8
//!     .distance(2, 0, 6.0)              // |p2-p0| = 6
//!     .build();
//!
//! let solver = LMSolver::new(LMConfig::default());
//! let initial = system.current_values();
//! let result = solver.solve(&system, &initial);
//!
//! if let SolveResult::Converged { solution, .. } = result {
//!     system.set_values(&solution);
//!     // Triangle is now properly constrained
//! }
//! ```
//!
//! ## Available Geometric Constraints
//!
//! | Constraint | 2D | 3D | Description |
//! |------------|----|----|-------------|
//! | `DistanceConstraint` | Yes | Yes | Point-to-point distance |
//! | `CoincidentConstraint` | Yes | Yes | Points at same location |
//! | `FixedConstraint` | Yes | Yes | Point at fixed position |
//! | `HorizontalConstraint` | Yes | - | Same y-coordinate |
//! | `VerticalConstraint` | Yes | - | Same x-coordinate |
//! | `AngleConstraint` | Yes | - | Line angle from horizontal |
//! | `ParallelConstraint` | Yes | Yes | Parallel lines |
//! | `PerpendicularConstraint` | Yes | Yes | Perpendicular lines |
//! | `MidpointConstraint` | Yes | Yes | Point at line midpoint |
//! | `PointOnLineConstraint` | Yes | Yes | Point lies on line |
//! | `PointOnCircleConstraint` | Yes | Yes | Point on circle/sphere |
//! | `SymmetricConstraint` | Yes | Yes | Point symmetry |
//! | `CollinearConstraint` | Yes | Yes | Collinear segments |
//! | `EqualLengthConstraint` | Yes | Yes | Equal line lengths |
//!
//! # MINPACK Test Suite
//!
//! This crate includes implementations of all 18 MINPACK least-squares test problems
//! and 14 nonlinear equation test problems for validation and benchmarking. These
//! problems are used extensively in numerical optimization literature and provide
//! a standardized way to test solver correctness and performance.
//!
//! The test problems are available in the [`test_problems`] module.
//!
//! # Feature Flags
//!
//! - `std` (default) - Standard library support
//! - `parallel` - Enable parallel component solving with rayon
//! - `sparse` - Enable sparse matrix operations with faer
//! - `geometry` - Enable geometric constraint library for 2D/3D CAD applications
//!
//! # Performance Considerations
//!
//! ## When to Use Each Solver
//!
//! - **Newton-Raphson (`Solver`)**: Best for square systems (m == n) with good
//!   initial guesses. Provides quadratic convergence near the solution but may
//!   fail to converge from poor starting points.
//!
//! - **Levenberg-Marquardt (`LMSolver`)**: More robust than NR, especially for
//!   over-determined systems. Handles poor initial guesses better by interpolating
//!   between gradient descent and Gauss-Newton.
//!
//! - **Sparse Solver (`SparseSolver`)**: For large systems (n > 100) where the
//!   Jacobian is sparse. Uses efficient sparse factorization that scales better
//!   than dense methods.
//!
//! - **Parallel Solver (`ParallelSolver`)**: When the problem naturally decomposes
//!   into independent components. Automatically detects components and solves them
//!   in parallel.
//!
//! ## Tips for Better Performance
//!
//! 1. **Good Initial Guesses**: Both NR and LM converge faster with reasonable
//!    starting points. If possible, use domain knowledge to initialize variables.
//!
//! 2. **Correct Jacobians**: Incorrect Jacobians cause convergence failures. Use
//!    [`verify_jacobian`] during development to check your implementations.
//!
//! 3. **Scaling**: Problems with widely varying scales can cause numerical issues.
//!    Consider normalizing your problem formulation.
//!
//! 4. **Sparsity**: For sparse problems, ensure your `jacobian()` implementation
//!    only returns non-zero entries. This enables efficient sparse operations.

pub mod constraints;
pub mod decomposition;
pub mod jacobian;
pub mod problem;
pub mod solver;
pub mod test_problems;

#[cfg(feature = "geometry")]
pub mod geometry;

#[cfg(feature = "jit")]
pub mod jit;

// Re-export main types at crate root
pub use problem::{ConfigurableProblem, Problem};
pub use solver::{
    AutoSolver, LMConfig, LMSolver, ParallelSolver, ParallelSolverConfig, RobustSolver, SolveError,
    SolveResult, Solver, SolverChoice, SolverConfig, SparseSolver, SparseSolverConfig,
};

// Re-export decomposition types
pub use decomposition::{
    decompose, decompose_from_edges, Component, ComponentId, DecomposableProblem, SubProblem,
};

// Re-export constraint types
pub use constraints::{
    BoundsConstraint, ClearanceConstraint, InequalityConstraint, InequalityProblem,
    SlackVariableTransform,
};

// Re-export jacobian utilities
pub use jacobian::{
    finite_difference_jacobian, verify_jacobian, CsrMatrix, JacobianVerification, SparseJacobian,
    SparsityPattern,
};

// Re-export macros for automatic Jacobian generation
#[cfg(feature = "macros")]
pub use solverang_macros::{auto_jacobian, residual};

// Re-export JIT types
#[cfg(feature = "jit")]
pub use jit::{
    jit_available, lower_constraints, CompiledConstraints, ConstraintOp, JITCompiler, JITConfig,
    JITError, JITFunction, Lowerable, LoweringContext, OpcodeEmitter, Reg,
};

// Re-export JIT solver
#[cfg(feature = "jit")]
pub use solver::JITSolver;
