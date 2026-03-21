//! Opcode emission for JIT compilation.
//!
//! This module provides the [`OpcodeEmitter`] — a fluent API for building
//! opcode streams that can be compiled to native code via Cranelift.

use super::opcodes::{ConstraintOp, JacobianEntry, Reg};

/// Opcode emitter for building opcode streams.
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
        assert_eq!(ops.len(), 13);
    }
}
