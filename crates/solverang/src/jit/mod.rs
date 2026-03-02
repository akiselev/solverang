//! JIT compilation for constraint evaluation.
//!
//! This module provides Copy-and-Patch JIT compilation that transforms constraint
//! systems into optimized native code, eliminating virtual function call overhead
//! during the iterative solve loop.
//!
//! # Architecture
//!
//! The JIT system works in three phases:
//!
//! 1. **Lowering**: Constraints are lowered from high-level representations to
//!    a stream of simple opcodes using the [`Lowerable`] trait.
//!
//! 2. **Compilation**: The opcode stream is compiled to native code using
//!    Cranelift, producing executable functions for residual and Jacobian evaluation.
//!
//! 3. **Execution**: The compiled functions are called during the solver iteration,
//!    providing significant speedup for large constraint systems.
//!
//! # Feature Flag
//!
//! This module requires the `jit` feature flag:
//!
//! ```toml
//! [dependencies]
//! solverang = { version = "0.1", features = ["jit"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use solverang::jit::{JITSolver, JITConfig};
//! use solverang::Problem;
//!
//! let solver = JITSolver::new(JITConfig::default());
//! let result = solver.solve(&my_problem, &initial_point);
//! ```
//!
//! # Performance
//!
//! JIT compilation provides 2-5x speedup on large constraint systems (1000+ variables)
//! compared to interpreted evaluation. The compilation overhead is amortized over
//! the many iterations of the solve loop.
//!
//! # Platform Support
//!
//! JIT compilation is supported on:
//! - x86_64 (Linux, macOS, Windows)
//! - aarch64 (Linux, macOS)
//!
//! On unsupported platforms, the solver automatically falls back to interpreted
//! evaluation.

mod cranelift;
mod lower;
mod opcodes;

pub use cranelift::{JITCompiler, JITError, JITFunction};
pub use lower::{
    CompiledProblem, Lowerable, LoweringContext, OpcodeEmitter, lower_problem, lower_constraints,
};
pub use opcodes::{CompiledConstraints, ConstraintOp, JacobianEntry, Reg, ValidationError};

/// Configuration for JIT-enabled solving.
#[derive(Clone, Debug)]
pub struct JITConfig {
    /// Threshold for JIT compilation (constraints * estimated_iterations).
    ///
    /// Problems with estimated work below this threshold use interpreted evaluation.
    /// Default: 1000
    pub jit_threshold: usize,

    /// Estimated number of iterations for threshold calculation.
    ///
    /// Default: 50
    pub estimated_iterations: usize,

    /// Maximum number of solver iterations.
    ///
    /// Default: 200
    pub max_iterations: usize,

    /// Convergence tolerance for residual norm.
    ///
    /// Default: 1e-8
    pub tolerance: f64,

    /// Whether to force JIT compilation regardless of problem size.
    ///
    /// Useful for benchmarking.
    /// Default: false
    pub force_jit: bool,

    /// Whether to force interpreted evaluation regardless of problem size.
    ///
    /// Useful for debugging.
    /// Default: false
    pub force_interpreted: bool,
}

impl Default for JITConfig {
    fn default() -> Self {
        Self {
            jit_threshold: 1000,
            estimated_iterations: 50,
            max_iterations: 200,
            tolerance: 1e-8,
            force_jit: false,
            force_interpreted: false,
        }
    }
}

impl JITConfig {
    /// Create a configuration that always uses JIT compilation.
    pub fn always_jit() -> Self {
        Self {
            force_jit: true,
            ..Default::default()
        }
    }

    /// Create a configuration that always uses interpreted evaluation.
    pub fn always_interpreted() -> Self {
        Self {
            force_interpreted: true,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for large problems.
    pub fn for_large_problems() -> Self {
        Self {
            jit_threshold: 500,
            max_iterations: 500,
            tolerance: 1e-10,
            ..Default::default()
        }
    }
}

/// Check if JIT compilation is available on this platform.
pub fn jit_available() -> bool {
    // Cranelift supports x86_64 and aarch64
    cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
}
