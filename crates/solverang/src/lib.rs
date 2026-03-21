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
//! # V3 Sketch2D (Builder API)
//!
//! Use [`sketch2d::Sketch2DBuilder`] to construct 2D constraint systems:
//!
//! ```rust
//! use solverang::sketch2d::Sketch2DBuilder;
//! use solverang::system::SystemStatus;
//!
//! let mut b = Sketch2DBuilder::new();
//! let p0 = b.add_fixed_point(0.0, 0.0);
//! let p1 = b.add_fixed_point(10.0, 0.0);
//! let p2 = b.add_point(5.0, 1.0);
//! b.constrain_distance(p0, p1, 10.0);
//! b.constrain_distance(p1, p2, 8.0);
//! b.constrain_distance(p2, p0, 6.0);
//! let mut system = b.build();
//!
//! let result = system.solve();
//! assert!(matches!(result.status, SystemStatus::Solved));
//! ```
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
//! # Two Paths to Solving
//!
//! `solverang` exposes two complementary APIs depending on how much control
//! you need:
//!
//! ## `Problem` trait — low-level, full control
//!
//! Implement [`Problem`] directly when you want to supply residuals and the
//! Jacobian yourself. You own every detail: variable layout, sparsity pattern,
//! and initial point.  The quick-start example above uses this path.
//!
//! ## `ConstraintSystem` — high-level, entity + constraint model
//!
//! [`ConstraintSystem`] (and the [`sketch2d::Sketch2DBuilder`] /
//! [`sketch3d::Sketch3DBuilder`] convenience wrappers) let you think in terms
//! of geometric *entities* (points, lines, circles, rigid bodies) and
//! *constraints* between them (distance, angle, coincident, mate …).
//!
//! The system automatically:
//!
//! 1. Builds a bipartite entity-constraint graph.
//! 2. Decomposes it into independent clusters.
//! 3. Applies symbolic reductions (fixed-param elimination, coincident merging).
//! 4. Selects the right solver for each cluster.
//! 5. Writes solutions back to the parameter store.
//!
//! See the `V3 Sketch2D` section below for a worked example.
//!
//! # Automatic Jacobian with `#[auto_jacobian]`
//!
//! Enable the `macros` feature and annotate your `impl` block to have the
//! Jacobian computed via symbolic differentiation at compile time.
//!
//! ## Single-residual (distance constraint)
//!
//! ```rust,ignore
//! use solverang::{auto_jacobian, Problem};
//!
//! struct DistanceConstraint { target: f64 }
//!
//! #[auto_jacobian(array_param = "x")]
//! impl DistanceConstraint {
//!     #[residual]
//!     fn residual(&self, x: &[f64]) -> f64 {
//!         let dx = x[2] - x[0];
//!         let dy = x[3] - x[1];
//!         (dx * dx + dy * dy).sqrt() - self.target
//!     }
//! }
//!
//! impl Problem for DistanceConstraint {
//!     fn name(&self) -> &str { "Distance" }
//!     fn residual_count(&self) -> usize { 1 }
//!     fn variable_count(&self) -> usize { 4 }
//!     fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
//!     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
//!         self.jacobian_entries(x)
//!     }
//!     fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0, 1.0, 0.0] }
//! }
//! ```
//!
//! ## Multi-residual (Rosenbrock)
//!
//! Multiple `#[residual]` methods produce a combined `jacobian_entries` that
//! assembles all rows in order.
//!
//! ```rust,ignore
//! use solverang::{auto_jacobian, Problem};
//!
//! struct Rosenbrock;
//!
//! #[auto_jacobian(array_param = "x")]
//! impl Rosenbrock {
//!     #[residual]
//!     fn r1(&self, x: &[f64]) -> f64 {
//!         10.0 * (x[1] - x[0] * x[0])
//!     }
//!
//!     #[residual]
//!     fn r2(&self, x: &[f64]) -> f64 {
//!         1.0 - x[0]
//!     }
//! }
//!
//! impl Problem for Rosenbrock {
//!     fn name(&self) -> &str { "Rosenbrock" }
//!     fn residual_count(&self) -> usize { 2 }
//!     fn variable_count(&self) -> usize { 2 }
//!
//!     fn residuals(&self, x: &[f64]) -> Vec<f64> {
//!         vec![self.r1(x), self.r2(x)]
//!     }
//!
//!     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
//!         self.jacobian_entries(x)  // rows 0 and 1 filled automatically
//!     }
//!
//!     fn initial_point(&self, factor: f64) -> Vec<f64> {
//!         vec![-1.2 * factor, factor]
//!     }
//! }
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | yes | Standard library support |
//! | `macros` | yes | Enables `#[auto_jacobian]` proc macro for automatic symbolic differentiation |
//! | `jit` | yes | Enables JIT compilation of constraint evaluators via Cranelift |
//! | `parallel` | yes | Parallel component solving with rayon |
//! | `sparse` | yes | Sparse matrix operations with faer |
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

// --- V3 Solver-First Architecture modules ---
pub mod assembly;
pub mod constraint;
pub(crate) mod dataflow;
pub mod entity;
pub(crate) mod graph;
pub mod id;
pub mod param;
pub mod pipeline;
pub(crate) mod reduce;
pub mod sketch2d;
pub mod sketch3d;
pub mod solve;
pub mod system;

// --- Existing modules (kept as-is) ---
pub mod constraints;
pub mod decomposition;
pub mod jacobian;
pub mod problem;
pub mod solver;
pub mod test_problems;

#[cfg(feature = "jit")]
pub mod jit;

// --- Re-export existing types at crate root ---
pub use problem::{ClosureProblem, ConfigurableProblem, Problem, ProblemBuilder};
pub use solver::{
    AutoSolver, LMConfig, LMSolver, ParallelSolver, ParallelSolverConfig, RobustSolver, SolveError,
    SolveResult, Solver, SolverChoice, SolverConfig, SparseSolver, SparseSolverConfig,
};

pub use decomposition::{
    decompose, decompose_from_edges, Component, ComponentId, DecomposableProblem, SubProblem,
};

pub use constraints::{
    BoundsConstraint, ClearanceConstraint, InequalityConstraint, InequalityProblem,
    SlackVariableTransform,
};

pub use jacobian::{
    finite_difference_jacobian, verify_jacobian, CsrMatrix, JacobianVerification, SparseJacobian,
    SparsityPattern,
};

#[cfg(feature = "macros")]
pub use solverang_macros::{auto_jacobian, residual};

#[cfg(feature = "jit")]
pub use jit::{
    jit_available, CompiledConstraints, ConstraintOp, JITCompiler, JITConfig, JITError,
    JITFunction, OpcodeEmitter, Reg,
};

#[cfg(feature = "jit")]
pub use solver::JITSolver;

/// Hidden re-exports for the `#[auto_jacobian]` macro's JIT code generation.
///
/// The proc macro cannot use `$crate`, so generated code references types
/// via `::solverang::__jit_reexports::OpcodeEmitter` etc. This follows the
/// same pattern used by serde, thiserror, and other proc-macro crates.
#[cfg(feature = "jit")]
#[doc(hidden)]
pub mod __jit_reexports {
    pub use crate::jit::{CompiledConstraints, OpcodeEmitter};
}

// --- Re-export V3 types ---
pub use id::{ClusterId, ConstraintId, EntityId, ParamId};
pub use param::ParamStore;
pub use system::ConstraintSystem;
