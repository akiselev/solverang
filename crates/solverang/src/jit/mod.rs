//! JIT compilation for Problem evaluation.
//!
//! This module compiles opcode streams to native machine code via Cranelift,
//! producing callable function pointers for residual and Jacobian evaluation.
//!
//! # Architecture
//!
//! The JIT system works in two phases:
//!
//! 1. **Opcode emission**: Build an opcode stream using [`OpcodeEmitter`].
//!    The `#[auto_jacobian]` macro does this automatically — it generates
//!    `lower_to_compiled_constraints()` which produces a [`CompiledConstraints`]
//!    ready for compilation.
//!
//! 2. **Compilation + Execution**: [`JITCompiler`] compiles the opcode stream
//!    to native x86_64/aarch64 code via Cranelift, returning a [`JITFunction`]
//!    with `evaluate_residuals()` and `evaluate_jacobian()` methods.
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

mod compiled_newton;
mod cranelift;
mod lower;
mod opcodes;

pub use compiled_newton::CompiledNewtonStep;
pub use cranelift::{JITCompiler, JITError, JITFunction};
pub use lower::OpcodeEmitter;
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
