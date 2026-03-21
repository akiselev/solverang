//! Opcode representation for JIT compilation.
//!
//! This module defines the low-level operations that constraints are lowered to
//! before JIT compilation. The opcodes form a simple register-based intermediate
//! representation that can be efficiently compiled to native code.

use std::fmt;

/// Virtual register identifier.
///
/// Registers are allocated during lowering and represent intermediate values
/// in the computation. The JIT compiler maps these to actual machine registers
/// or stack slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u16);

impl Reg {
    /// Create a new register with the given index.
    pub fn new(index: u16) -> Self {
        Self(index)
    }

    /// Get the register index.
    pub fn index(self) -> u16 {
        self.0
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Low-level constraint operations for JIT compilation.
///
/// These opcodes represent the fundamental operations needed to evaluate
/// constraint residuals and Jacobians. They form a simple register-based
/// intermediate representation.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintOp {
    /// Load a variable into a register.
    ///
    /// Loads the value at `var_idx` from the variable array into register `dst`.
    LoadVar {
        /// Destination register.
        dst: Reg,
        /// Variable index in the input array.
        var_idx: u32,
    },

    /// Load a constant into a register.
    LoadConst {
        /// Destination register.
        dst: Reg,
        /// Constant value to load.
        value: f64,
    },

    /// Addition: dst = a + b
    Add {
        /// Destination register.
        dst: Reg,
        /// First operand register.
        a: Reg,
        /// Second operand register.
        b: Reg,
    },

    /// Subtraction: dst = a - b
    Sub {
        /// Destination register.
        dst: Reg,
        /// First operand register.
        a: Reg,
        /// Second operand register.
        b: Reg,
    },

    /// Multiplication: dst = a * b
    Mul {
        /// Destination register.
        dst: Reg,
        /// First operand register.
        a: Reg,
        /// Second operand register.
        b: Reg,
    },

    /// Division: dst = a / b
    ///
    /// Division by zero produces infinity or NaN following IEEE 754 rules.
    Div {
        /// Destination register.
        dst: Reg,
        /// Dividend register.
        a: Reg,
        /// Divisor register.
        b: Reg,
    },

    /// Negation: dst = -src
    Neg {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
    },

    /// Square root: dst = sqrt(src)
    Sqrt {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
    },

    /// Sine: dst = sin(src)
    Sin {
        /// Destination register.
        dst: Reg,
        /// Source register (angle in radians).
        src: Reg,
    },

    /// Cosine: dst = cos(src)
    Cos {
        /// Destination register.
        dst: Reg,
        /// Source register (angle in radians).
        src: Reg,
    },

    /// Two-argument arctangent: dst = atan2(y, x)
    Atan2 {
        /// Destination register.
        dst: Reg,
        /// Y coordinate register.
        y: Reg,
        /// X coordinate register.
        x: Reg,
    },

    /// Absolute value: dst = |src|
    Abs {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
    },

    /// Maximum: dst = max(a, b)
    Max {
        /// Destination register.
        dst: Reg,
        /// First operand register.
        a: Reg,
        /// Second operand register.
        b: Reg,
    },

    /// Minimum: dst = min(a, b)
    Min {
        /// Destination register.
        dst: Reg,
        /// First operand register.
        a: Reg,
        /// Second operand register.
        b: Reg,
    },

    /// Exponential: dst = exp(src)
    Exp {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
    },

    /// Natural logarithm: dst = ln(src)
    Ln {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
    },

    /// Power: dst = base^exp
    Pow {
        /// Destination register.
        dst: Reg,
        /// Base register.
        base: Reg,
        /// Exponent register.
        exp: Reg,
    },

    /// Tangent: dst = tan(src)
    Tan {
        /// Destination register.
        dst: Reg,
        /// Source register (angle in radians).
        src: Reg,
    },

    /// Store a residual value.
    ///
    /// Stores the value in register `src` to the residual array at index `residual_idx`.
    StoreResidual {
        /// Index in the residual output array.
        residual_idx: u32,
        /// Source register containing the residual value.
        src: Reg,
    },

    /// Store a Jacobian entry (sparse COO format).
    ///
    /// Stores the value in register `src` along with its row and column indices
    /// in the sparse Jacobian output arrays.
    StoreJacobian {
        /// Row index (residual index).
        row: u32,
        /// Column index (variable index).
        col: u32,
        /// Source register containing the Jacobian value.
        src: Reg,
    },

    /// Store a Jacobian entry at a specific output index.
    ///
    /// This is used when the sparsity pattern is known at compile time,
    /// allowing direct indexing into the value array.
    StoreJacobianIndexed {
        /// Index in the Jacobian values array.
        output_idx: u32,
        /// Source register containing the Jacobian value.
        src: Reg,
    },
}

impl ConstraintOp {
    /// Check if this operation uses the given register as input.
    pub fn uses_register(&self, reg: Reg) -> bool {
        match self {
            ConstraintOp::LoadVar { .. } | ConstraintOp::LoadConst { .. } => false,
            ConstraintOp::Add { a, b, .. }
            | ConstraintOp::Sub { a, b, .. }
            | ConstraintOp::Mul { a, b, .. }
            | ConstraintOp::Div { a, b, .. }
            | ConstraintOp::Max { a, b, .. }
            | ConstraintOp::Min { a, b, .. } => *a == reg || *b == reg,
            ConstraintOp::Atan2 { y, x, .. } => *y == reg || *x == reg,
            ConstraintOp::Pow { base, exp, .. } => *base == reg || *exp == reg,
            ConstraintOp::Neg { src, .. }
            | ConstraintOp::Sqrt { src, .. }
            | ConstraintOp::Sin { src, .. }
            | ConstraintOp::Cos { src, .. }
            | ConstraintOp::Abs { src, .. }
            | ConstraintOp::Exp { src, .. }
            | ConstraintOp::Ln { src, .. }
            | ConstraintOp::Tan { src, .. } => *src == reg,
            ConstraintOp::StoreResidual { src, .. }
            | ConstraintOp::StoreJacobian { src, .. }
            | ConstraintOp::StoreJacobianIndexed { src, .. } => *src == reg,
        }
    }

    /// Check if this operation defines (writes to) the given register.
    pub fn defines_register(&self, reg: Reg) -> bool {
        match self {
            ConstraintOp::LoadVar { dst, .. }
            | ConstraintOp::LoadConst { dst, .. }
            | ConstraintOp::Add { dst, .. }
            | ConstraintOp::Sub { dst, .. }
            | ConstraintOp::Mul { dst, .. }
            | ConstraintOp::Div { dst, .. }
            | ConstraintOp::Neg { dst, .. }
            | ConstraintOp::Sqrt { dst, .. }
            | ConstraintOp::Sin { dst, .. }
            | ConstraintOp::Cos { dst, .. }
            | ConstraintOp::Atan2 { dst, .. }
            | ConstraintOp::Abs { dst, .. }
            | ConstraintOp::Max { dst, .. }
            | ConstraintOp::Min { dst, .. }
            | ConstraintOp::Exp { dst, .. }
            | ConstraintOp::Ln { dst, .. }
            | ConstraintOp::Pow { dst, .. }
            | ConstraintOp::Tan { dst, .. } => *dst == reg,
            ConstraintOp::StoreResidual { .. }
            | ConstraintOp::StoreJacobian { .. }
            | ConstraintOp::StoreJacobianIndexed { .. } => false,
        }
    }
}

/// Jacobian entry in COO (coordinate) format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JacobianEntry {
    /// Row index (residual index).
    pub row: u32,
    /// Column index (variable index).
    pub col: u32,
}

/// Compiled constraint system ready for JIT.
///
/// This structure contains the lowered opcodes and metadata needed for
/// JIT compilation. It is produced by the lowering phase and consumed
/// by the Cranelift code generator.
#[derive(Clone, Debug)]
pub struct CompiledConstraints {
    /// Opcode stream for residual evaluation.
    pub residual_ops: Vec<ConstraintOp>,

    /// Opcode stream for Jacobian evaluation.
    pub jacobian_ops: Vec<ConstraintOp>,

    /// Number of residuals produced.
    pub n_residuals: usize,

    /// Number of input variables.
    pub n_vars: usize,

    /// Number of non-zero entries in the Jacobian.
    pub jacobian_nnz: usize,

    /// Sparsity pattern for the Jacobian (row, col) pairs.
    ///
    /// This defines the structure of the sparse Jacobian matrix.
    /// The i-th entry corresponds to the i-th StoreJacobianIndexed operation.
    pub jacobian_pattern: Vec<JacobianEntry>,

    /// Maximum register index used.
    ///
    /// This is used to allocate the register file during compilation.
    pub max_register: u16,
}

impl CompiledConstraints {
    /// Create an empty compiled constraint system.
    pub fn new(n_vars: usize, n_residuals: usize) -> Self {
        Self {
            residual_ops: Vec::new(),
            jacobian_ops: Vec::new(),
            n_residuals,
            n_vars,
            jacobian_nnz: 0,
            jacobian_pattern: Vec::new(),
            max_register: 0,
        }
    }

    /// Check if the compiled constraints are valid.
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Check that all residual store indices are in bounds
        for op in &self.residual_ops {
            if let ConstraintOp::StoreResidual { residual_idx, .. } = op {
                if *residual_idx as usize >= self.n_residuals {
                    return Err(ValidationError::ResidualIndexOutOfBounds {
                        index: *residual_idx as usize,
                        count: self.n_residuals,
                    });
                }
            }
        }

        // Check that all variable load indices are in bounds
        for op in self.residual_ops.iter().chain(self.jacobian_ops.iter()) {
            if let ConstraintOp::LoadVar { var_idx, .. } = op {
                if *var_idx as usize >= self.n_vars {
                    return Err(ValidationError::VariableIndexOutOfBounds {
                        index: *var_idx as usize,
                        count: self.n_vars,
                    });
                }
            }
        }

        // Check Jacobian pattern
        if self.jacobian_nnz != self.jacobian_pattern.len() {
            return Err(ValidationError::JacobianPatternMismatch {
                nnz: self.jacobian_nnz,
                pattern_len: self.jacobian_pattern.len(),
            });
        }

        Ok(())
    }

    /// Get the total number of opcodes.
    pub fn total_ops(&self) -> usize {
        self.residual_ops.len() + self.jacobian_ops.len()
    }
}

/// Validation error for compiled constraints.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationError {
    /// Residual index is out of bounds.
    ResidualIndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// The number of residuals.
        count: usize,
    },

    /// Variable index is out of bounds.
    VariableIndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// The number of variables.
        count: usize,
    },

    /// Jacobian non-zero count doesn't match pattern length.
    JacobianPatternMismatch {
        /// Declared number of non-zeros.
        nnz: usize,
        /// Actual pattern length.
        pattern_len: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::ResidualIndexOutOfBounds { index, count } => {
                write!(
                    f,
                    "residual index {} out of bounds (count: {})",
                    index, count
                )
            }
            ValidationError::VariableIndexOutOfBounds { index, count } => {
                write!(
                    f,
                    "variable index {} out of bounds (count: {})",
                    index, count
                )
            }
            ValidationError::JacobianPatternMismatch { nnz, pattern_len } => {
                write!(
                    f,
                    "Jacobian nnz ({}) doesn't match pattern length ({})",
                    nnz, pattern_len
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg_display() {
        let r = Reg::new(5);
        assert_eq!(format!("{}", r), "r5");
    }

    #[test]
    fn test_op_uses_register() {
        let r0 = Reg::new(0);
        let r1 = Reg::new(1);
        let r2 = Reg::new(2);

        let add = ConstraintOp::Add {
            dst: r2,
            a: r0,
            b: r1,
        };

        assert!(add.uses_register(r0));
        assert!(add.uses_register(r1));
        assert!(!add.uses_register(r2));
    }

    #[test]
    fn test_op_defines_register() {
        let r0 = Reg::new(0);
        let r1 = Reg::new(1);
        let r2 = Reg::new(2);

        let add = ConstraintOp::Add {
            dst: r2,
            a: r0,
            b: r1,
        };

        assert!(!add.defines_register(r0));
        assert!(!add.defines_register(r1));
        assert!(add.defines_register(r2));
    }

    #[test]
    fn test_compiled_constraints_validate() {
        let mut cc = CompiledConstraints::new(4, 2);
        cc.residual_ops.push(ConstraintOp::LoadVar {
            dst: Reg::new(0),
            var_idx: 0,
        });
        cc.residual_ops.push(ConstraintOp::StoreResidual {
            residual_idx: 0,
            src: Reg::new(0),
        });
        cc.max_register = 0;

        assert!(cc.validate().is_ok());
    }

    #[test]
    fn test_compiled_constraints_validate_out_of_bounds_residual() {
        let mut cc = CompiledConstraints::new(4, 2);
        cc.residual_ops.push(ConstraintOp::StoreResidual {
            residual_idx: 5, // Out of bounds!
            src: Reg::new(0),
        });

        let err = cc.validate().unwrap_err();
        assert!(matches!(
            err,
            ValidationError::ResidualIndexOutOfBounds { .. }
        ));
    }

    #[test]
    fn test_compiled_constraints_validate_out_of_bounds_var() {
        let mut cc = CompiledConstraints::new(4, 2);
        cc.residual_ops.push(ConstraintOp::LoadVar {
            dst: Reg::new(0),
            var_idx: 10, // Out of bounds!
        });

        let err = cc.validate().unwrap_err();
        assert!(matches!(
            err,
            ValidationError::VariableIndexOutOfBounds { .. }
        ));
    }
}
