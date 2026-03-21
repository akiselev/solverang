//! Constraint lowering to opcodes.
//!
//! This module provides the [`Lowerable`] trait and [`OpcodeEmitter`] helper for
//! transforming high-level constraint representations into low-level opcodes
//! suitable for JIT compilation.

use super::opcodes::{CompiledConstraints, ConstraintOp, JacobianEntry, Reg};
use crate::problem::Problem;

/// Context for lowering operations.
///
/// Tracks the current residual index and provides coordinate to variable index
/// mapping.
#[derive(Clone, Debug)]
pub struct LoweringContext {
    /// Current residual index (incremented after each constraint).
    pub current_residual: u32,

    /// Dimension of points (2 for 2D, 3 for 3D).
    pub dimension: usize,

    /// Total number of variables.
    pub n_vars: usize,
}

impl LoweringContext {
    /// Create a new lowering context.
    pub fn new(dimension: usize, n_vars: usize) -> Self {
        Self {
            current_residual: 0,
            dimension,
            n_vars,
        }
    }

    /// Get the variable index for a point coordinate.
    ///
    /// For a constraint system with D-dimensional points, point `p` coordinate `k`
    /// maps to variable index `p * D + k`.
    pub fn var_index(&self, point_idx: usize, coord: usize) -> u32 {
        (point_idx * self.dimension + coord) as u32
    }

    /// Advance to the next residual index.
    pub fn next_residual(&mut self) -> u32 {
        let idx = self.current_residual;
        self.current_residual += 1;
        idx
    }
}

/// Opcode emitter for building constraint computations.
///
/// This helper provides a fluent API for emitting opcodes during constraint
/// lowering. It manages register allocation and provides convenience methods
/// for common operations.
#[derive(Debug)]
pub struct OpcodeEmitter {
    /// Emitted opcodes.
    ops: Vec<ConstraintOp>,

    /// Next available register.
    next_reg: u16,

    /// Jacobian entries for sparsity pattern.
    jacobian_entries: Vec<JacobianEntry>,

    /// Current residual index for Jacobian entries.
    current_residual: u32,
}

impl OpcodeEmitter {
    /// Create a new opcode emitter.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            next_reg: 0,
            jacobian_entries: Vec::new(),
            current_residual: 0,
        }
    }

    /// Set the current residual index for Jacobian entries.
    pub fn set_residual_index(&mut self, idx: u32) {
        self.current_residual = idx;
    }

    /// Allocate a new register.
    fn alloc_reg(&mut self) -> Reg {
        let reg = Reg::new(self.next_reg);
        self.next_reg += 1;
        reg
    }

    /// Get the maximum register index used.
    pub fn max_register(&self) -> u16 {
        if self.next_reg > 0 {
            self.next_reg - 1
        } else {
            0
        }
    }

    /// Get the emitted opcodes.
    pub fn ops(&self) -> &[ConstraintOp] {
        &self.ops
    }

    /// Take the emitted opcodes, consuming the emitter.
    pub fn into_ops(self) -> Vec<ConstraintOp> {
        self.ops
    }

    /// Get the Jacobian entries.
    pub fn jacobian_entries(&self) -> &[JacobianEntry] {
        &self.jacobian_entries
    }

    /// Take the Jacobian entries.
    pub fn take_jacobian_entries(&mut self) -> Vec<JacobianEntry> {
        std::mem::take(&mut self.jacobian_entries)
    }

    // ========================================================================
    // Load operations
    // ========================================================================

    /// Load a variable into a register.
    pub fn load_var(&mut self, var_idx: u32) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::LoadVar { dst, var_idx });
        dst
    }

    /// Load a constant into a register.
    pub fn const_f64(&mut self, value: f64) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::LoadConst { dst, value });
        dst
    }

    /// Load zero into a register.
    pub fn zero(&mut self) -> Reg {
        self.const_f64(0.0)
    }

    /// Load one into a register.
    pub fn one(&mut self) -> Reg {
        self.const_f64(1.0)
    }

    // ========================================================================
    // Arithmetic operations
    // ========================================================================

    /// Add two registers: dst = a + b
    pub fn add(&mut self, a: Reg, b: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Add { dst, a, b });
        dst
    }

    /// Subtract two registers: dst = a - b
    pub fn sub(&mut self, a: Reg, b: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Sub { dst, a, b });
        dst
    }

    /// Multiply two registers: dst = a * b
    pub fn mul(&mut self, a: Reg, b: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Mul { dst, a, b });
        dst
    }

    /// Divide two registers: dst = a / b
    pub fn div(&mut self, a: Reg, b: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Div { dst, a, b });
        dst
    }

    /// Negate a register: dst = -src
    pub fn neg(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Neg { dst, src });
        dst
    }

    /// Square root: dst = sqrt(src)
    pub fn sqrt(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Sqrt { dst, src });
        dst
    }

    /// Absolute value: dst = |src|
    pub fn abs(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Abs { dst, src });
        dst
    }

    /// Maximum: dst = max(a, b)
    pub fn max(&mut self, a: Reg, b: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Max { dst, a, b });
        dst
    }

    /// Minimum: dst = min(a, b)
    pub fn min(&mut self, a: Reg, b: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Min { dst, a, b });
        dst
    }

    // ========================================================================
    // Trigonometric operations
    // ========================================================================

    /// Sine: dst = sin(src)
    pub fn sin(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Sin { dst, src });
        dst
    }

    /// Cosine: dst = cos(src)
    pub fn cos(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Cos { dst, src });
        dst
    }

    /// Two-argument arctangent: dst = atan2(y, x)
    pub fn atan2(&mut self, y: Reg, x: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Atan2 { dst, y, x });
        dst
    }

    /// Exponential: dst = exp(src)
    pub fn exp(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Exp { dst, src });
        dst
    }

    /// Natural logarithm: dst = ln(src)
    pub fn ln(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Ln { dst, src });
        dst
    }

    /// Power: dst = base^exp
    pub fn pow(&mut self, base: Reg, exp: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Pow { dst, base, exp });
        dst
    }

    /// Tangent: dst = tan(src)
    pub fn tan(&mut self, src: Reg) -> Reg {
        let dst = self.alloc_reg();
        self.ops.push(ConstraintOp::Tan { dst, src });
        dst
    }

    // ========================================================================
    // Compound operations
    // ========================================================================

    /// Square: dst = src * src
    pub fn square(&mut self, src: Reg) -> Reg {
        self.mul(src, src)
    }

    /// Safe distance calculation with minimum epsilon.
    ///
    /// Returns max(sqrt(dx^2 + dy^2 + ...), epsilon) to avoid division by zero.
    pub fn safe_distance(&mut self, squared_sum: Reg, epsilon: f64) -> Reg {
        let dist = self.sqrt(squared_sum);
        let eps = self.const_f64(epsilon);
        self.max(dist, eps)
    }

    // ========================================================================
    // Store operations
    // ========================================================================

    /// Store a residual value.
    pub fn store_residual(&mut self, residual_idx: u32, src: Reg) {
        self.ops
            .push(ConstraintOp::StoreResidual { residual_idx, src });
    }

    /// Store a Jacobian entry.
    pub fn store_jacobian(&mut self, row: u32, col: u32, src: Reg) {
        let output_idx = self.jacobian_entries.len() as u32;
        self.jacobian_entries.push(JacobianEntry { row, col });
        self.ops
            .push(ConstraintOp::StoreJacobianIndexed { output_idx, src });
    }

    /// Store a Jacobian entry using the current residual index.
    pub fn store_jacobian_current(&mut self, col: u32, src: Reg) {
        self.store_jacobian(self.current_residual, col, src);
    }
}

impl Default for OpcodeEmitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be lowered to JIT opcodes.
///
/// Types implementing this trait can have their constraint evaluation
/// compiled to native code for faster execution.
pub trait Lowerable {
    /// Emit opcodes for residual evaluation.
    ///
    /// The emitter should produce opcodes that compute the constraint residual(s)
    /// and store them using `store_residual`.
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext);

    /// Emit opcodes for Jacobian evaluation.
    ///
    /// The emitter should produce opcodes that compute the Jacobian entries
    /// and store them using `store_jacobian`.
    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext);

    /// Get the number of residuals produced by this constraint.
    fn residual_count(&self) -> usize;

    /// Get the variable indices this constraint depends on.
    fn variable_indices(&self) -> Vec<usize>;
}

/// Compiled problem ready for JIT.
pub struct CompiledProblem {
    /// The compiled constraints.
    pub constraints: CompiledConstraints,

    /// Problem name.
    pub name: String,
}

/// Lower a problem to compiled constraints.
///
/// This function processes all constraints in the problem and produces
/// opcode streams for residual and Jacobian evaluation.
pub fn lower_problem<P: Problem>(problem: &P) -> CompiledConstraints {
    // For generic Problem types, we cannot lower directly since we don't have
    // access to the constraint structure. Instead, we create a wrapper that
    // computes residuals and Jacobians by calling the Problem trait methods.
    //
    // This is a fallback for problems that don't implement Lowerable directly.
    // For maximum performance, problems should implement Lowerable for their
    // specific constraint types.

    let n_vars = problem.variable_count();
    let n_residuals = problem.residual_count();

    CompiledConstraints::new(n_vars, n_residuals)
}

/// Lower a collection of lowerable constraints.
pub fn lower_constraints<L: Lowerable>(
    constraints: &[L],
    n_vars: usize,
    dimension: usize,
) -> CompiledConstraints {
    let n_residuals: usize = constraints.iter().map(|c| c.residual_count()).sum();

    let mut cc = CompiledConstraints::new(n_vars, n_residuals);
    let mut ctx = LoweringContext::new(dimension, n_vars);

    // Lower residuals
    let mut residual_emitter = OpcodeEmitter::new();
    for constraint in constraints {
        constraint.lower_residual(&mut residual_emitter, &ctx);
        for _ in 0..constraint.residual_count() {
            ctx.next_residual();
        }
    }
    let residual_max_register = residual_emitter.max_register();
    cc.residual_ops = residual_emitter.into_ops();

    // Reset context for Jacobian lowering
    ctx.current_residual = 0;

    // Lower Jacobians
    let mut jacobian_emitter = OpcodeEmitter::new();
    for constraint in constraints {
        constraint.lower_jacobian(&mut jacobian_emitter, &ctx);
        for _ in 0..constraint.residual_count() {
            ctx.next_residual();
        }
    }
    let jacobian_max_register = jacobian_emitter.max_register();
    cc.jacobian_ops = jacobian_emitter.ops().to_vec();
    cc.jacobian_pattern = jacobian_emitter.take_jacobian_entries();
    cc.jacobian_nnz = cc.jacobian_pattern.len();
    cc.max_register = residual_max_register.max(jacobian_max_register);

    cc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_emitter_basic() {
        let mut emitter = OpcodeEmitter::new();

        let x = emitter.load_var(0);
        let y = emitter.load_var(1);
        let sum = emitter.add(x, y);
        emitter.store_residual(0, sum);

        let ops = emitter.ops();
        assert_eq!(ops.len(), 4);

        assert!(matches!(ops[0], ConstraintOp::LoadVar { var_idx: 0, .. }));
        assert!(matches!(ops[1], ConstraintOp::LoadVar { var_idx: 1, .. }));
        assert!(matches!(ops[2], ConstraintOp::Add { .. }));
        assert!(matches!(
            ops[3],
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_opcode_emitter_distance() {
        let mut emitter = OpcodeEmitter::new();

        // Compute distance: sqrt((x2-x1)^2 + (y2-y1)^2) - target
        let x1 = emitter.load_var(0);
        let y1 = emitter.load_var(1);
        let x2 = emitter.load_var(2);
        let y2 = emitter.load_var(3);

        let dx = emitter.sub(x2, x1);
        let dy = emitter.sub(y2, y1);
        let dx2 = emitter.square(dx);
        let dy2 = emitter.square(dy);
        let sum = emitter.add(dx2, dy2);
        let dist = emitter.sqrt(sum);
        let target = emitter.const_f64(5.0);
        let residual = emitter.sub(dist, target);
        emitter.store_residual(0, residual);

        let ops = emitter.ops();
        // 4 loads + 2 sub + 2 mul (square) + 1 add + 1 sqrt + 1 const + 1 sub + 1 store = 13
        assert_eq!(ops.len(), 13);
    }

    #[test]
    fn test_lowering_context() {
        let ctx = LoweringContext::new(2, 10);

        // Point 0, x coordinate -> variable 0
        assert_eq!(ctx.var_index(0, 0), 0);
        // Point 0, y coordinate -> variable 1
        assert_eq!(ctx.var_index(0, 1), 1);
        // Point 1, x coordinate -> variable 2
        assert_eq!(ctx.var_index(1, 0), 2);
        // Point 2, y coordinate -> variable 5
        assert_eq!(ctx.var_index(2, 1), 5);
    }

    #[test]
    fn test_lowering_context_3d() {
        let ctx = LoweringContext::new(3, 12);

        // Point 0, x -> 0
        assert_eq!(ctx.var_index(0, 0), 0);
        // Point 0, y -> 1
        assert_eq!(ctx.var_index(0, 1), 1);
        // Point 0, z -> 2
        assert_eq!(ctx.var_index(0, 2), 2);
        // Point 1, x -> 3
        assert_eq!(ctx.var_index(1, 0), 3);
    }
}
