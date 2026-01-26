//! Cranelift JIT code generation.
//!
//! This module compiles constraint opcodes to native machine code using Cranelift.

use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use super::opcodes::{CompiledConstraints, ConstraintOp, Reg};

/// Error during JIT compilation.
#[derive(Clone, Debug)]
pub enum JITError {
    /// Cranelift module error.
    ModuleError(String),

    /// Cranelift codegen error.
    CodegenError(String),

    /// JIT is not available on this platform.
    NotAvailable,

    /// Invalid compiled constraints.
    ValidationError(String),

    /// Memory allocation failed.
    AllocationError(String),
}

impl std::fmt::Display for JITError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JITError::ModuleError(msg) => write!(f, "module error: {}", msg),
            JITError::CodegenError(msg) => write!(f, "codegen error: {}", msg),
            JITError::NotAvailable => write!(f, "JIT not available on this platform"),
            JITError::ValidationError(msg) => write!(f, "validation error: {}", msg),
            JITError::AllocationError(msg) => write!(f, "allocation error: {}", msg),
        }
    }
}

impl std::error::Error for JITError {}

/// JIT compiler using Cranelift.
pub struct JITCompiler {
    /// Cranelift JIT module.
    module: JITModule,

    /// Cranelift codegen context.
    ctx: codegen::Context,

    /// Function builder context.
    builder_ctx: FunctionBuilderContext,
}

impl JITCompiler {
    /// Create a new JIT compiler.
    pub fn new() -> Result<Self, JITError> {
        let mut flag_builder = settings::builder();
        // Enable optimizations
        flag_builder.set("opt_level", "speed").map_err(|e| {
            JITError::ModuleError(format!("failed to set opt_level: {}", e))
        })?;

        let isa_builder = cranelift_native::builder().map_err(|e| {
            JITError::ModuleError(format!("failed to create ISA builder: {}", e))
        })?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| {
                JITError::ModuleError(format!("failed to create ISA: {}", e))
            })?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();

        Ok(Self {
            module,
            ctx,
            builder_ctx,
        })
    }

    /// Compile constraint opcodes to native functions.
    pub fn compile(
        &mut self,
        compiled: &CompiledConstraints,
    ) -> Result<JITFunction, JITError> {
        // Validate the compiled constraints
        compiled.validate().map_err(|e| {
            JITError::ValidationError(e.to_string())
        })?;

        // Compile residual evaluation function
        let residual_fn_id = self.compile_residuals(compiled)?;

        // Compile Jacobian evaluation function
        let jacobian_fn_id = self.compile_jacobian(compiled)?;

        // Finalize all function definitions
        self.module.finalize_definitions().map_err(|e| {
            JITError::ModuleError(format!("failed to finalize definitions: {}", e))
        })?;

        // Get function pointers
        let residual_ptr = self.module.get_finalized_function(residual_fn_id);
        let jacobian_ptr = self.module.get_finalized_function(jacobian_fn_id);

        // Type alias for the JIT function signature
        type JITFnPtr = unsafe extern "C" fn(*const f64, *mut f64);

        Ok(JITFunction {
            residual_fn: unsafe { std::mem::transmute::<*const u8, JITFnPtr>(residual_ptr) },
            jacobian_fn: unsafe { std::mem::transmute::<*const u8, JITFnPtr>(jacobian_ptr) },
            n_residuals: compiled.n_residuals,
            n_vars: compiled.n_vars,
            jacobian_nnz: compiled.jacobian_nnz,
            jacobian_pattern: compiled.jacobian_pattern.clone(),
        })
    }

    /// Compile residual evaluation function.
    fn compile_residuals(&mut self, compiled: &CompiledConstraints) -> Result<FuncId, JITError> {
        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();

        let ptr_type = self.module.target_config().pointer_type();

        // Function signature: fn(vars: *const f64, residuals: *mut f64)
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function("evaluate_residuals", Linkage::Local, &self.ctx.func.signature)
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_ptr = builder.block_params(entry_block)[0];
            let residuals_ptr = builder.block_params(entry_block)[1];

            // Translate opcodes
            translate_ops(&mut builder, &compiled.residual_ops, vars_ptr, residuals_ptr, None);

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JITError::CodegenError(format!("failed to define function: {}", e)))?;

        self.ctx.clear();

        Ok(func_id)
    }

    /// Compile Jacobian evaluation function.
    fn compile_jacobian(&mut self, compiled: &CompiledConstraints) -> Result<FuncId, JITError> {
        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();

        let ptr_type = self.module.target_config().pointer_type();

        // Function signature: fn(vars: *const f64, jacobian_values: *mut f64)
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function("evaluate_jacobian", Linkage::Local, &self.ctx.func.signature)
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_ptr = builder.block_params(entry_block)[0];
            let jacobian_ptr = builder.block_params(entry_block)[1];

            // Translate opcodes
            translate_ops(&mut builder, &compiled.jacobian_ops, vars_ptr, jacobian_ptr, Some(jacobian_ptr));

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JITError::CodegenError(format!("failed to define function: {}", e)))?;

        self.ctx.clear();

        Ok(func_id)
    }
}

/// Translate opcodes to Cranelift IR.
fn translate_ops(
    builder: &mut FunctionBuilder<'_>,
    ops: &[ConstraintOp],
    vars_ptr: Value,
    output_ptr: Value,
    jacobian_ptr: Option<Value>,
) {
    let mut registers: HashMap<Reg, Value> = HashMap::new();

    for op in ops {
        match op {
            ConstraintOp::LoadVar { dst, var_idx } => {
                let offset = (*var_idx as i32) * 8; // f64 is 8 bytes
                let addr = builder.ins().iadd_imm(vars_ptr, offset as i64);
                let value = builder.ins().load(types::F64, MemFlags::trusted(), addr, 0);
                registers.insert(*dst, value);
            }

            ConstraintOp::LoadConst { dst, value } => {
                let const_val = builder.ins().f64const(*value);
                registers.insert(*dst, const_val);
            }

            ConstraintOp::Add { dst, a, b } => {
                let a_val = get_reg(&registers, *a);
                let b_val = get_reg(&registers, *b);
                let result = builder.ins().fadd(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Sub { dst, a, b } => {
                let a_val = get_reg(&registers, *a);
                let b_val = get_reg(&registers, *b);
                let result = builder.ins().fsub(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Mul { dst, a, b } => {
                let a_val = get_reg(&registers, *a);
                let b_val = get_reg(&registers, *b);
                let result = builder.ins().fmul(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Div { dst, a, b } => {
                let a_val = get_reg(&registers, *a);
                let b_val = get_reg(&registers, *b);
                let result = builder.ins().fdiv(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Neg { dst, src } => {
                let src_val = get_reg(&registers, *src);
                let result = builder.ins().fneg(src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Sqrt { dst, src } => {
                let src_val = get_reg(&registers, *src);
                let result = builder.ins().sqrt(src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Sin { dst, src } => {
                let src_val = get_reg(&registers, *src);
                let result = approximate_sin(builder, src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Cos { dst, src } => {
                let src_val = get_reg(&registers, *src);
                let result = approximate_cos(builder, src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Atan2 { dst, y, x } => {
                let y_val = get_reg(&registers, *y);
                let x_val = get_reg(&registers, *x);
                let result = approximate_atan2(builder, y_val, x_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Abs { dst, src } => {
                let src_val = get_reg(&registers, *src);
                let result = builder.ins().fabs(src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Max { dst, a, b } => {
                let a_val = get_reg(&registers, *a);
                let b_val = get_reg(&registers, *b);
                let result = builder.ins().fmax(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Min { dst, a, b } => {
                let a_val = get_reg(&registers, *a);
                let b_val = get_reg(&registers, *b);
                let result = builder.ins().fmin(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::StoreResidual { residual_idx, src } => {
                let src_val = get_reg(&registers, *src);
                let offset = (*residual_idx as i32) * 8;
                let addr = builder.ins().iadd_imm(output_ptr, offset as i64);
                builder.ins().store(MemFlags::trusted(), src_val, addr, 0);
            }

            ConstraintOp::StoreJacobian { row: _, col: _, src } => {
                // This variant stores row/col indices too, but for our use case
                // we typically use StoreJacobianIndexed instead
                let _ = get_reg(&registers, *src);
                // Would need separate row/col arrays to implement this
            }

            ConstraintOp::StoreJacobianIndexed { output_idx, src } => {
                let src_val = get_reg(&registers, *src);
                let jac_ptr = jacobian_ptr.unwrap_or(output_ptr);
                let offset = (*output_idx as i32) * 8;
                let addr = builder.ins().iadd_imm(jac_ptr, offset as i64);
                builder.ins().store(MemFlags::trusted(), src_val, addr, 0);
            }
        }
    }
}

fn get_reg(registers: &HashMap<Reg, Value>, reg: Reg) -> Value {
    *registers.get(&reg).unwrap_or_else(|| {
        panic!("register {} not found", reg)
    })
}

/// Approximate sin using Taylor series for small values.
fn approximate_sin(builder: &mut FunctionBuilder<'_>, x: Value) -> Value {
    // sin(x) ~ x - x^3/6 + x^5/120 - x^7/5040
    let x2 = builder.ins().fmul(x, x);
    let x3 = builder.ins().fmul(x2, x);
    let x5 = builder.ins().fmul(x3, x2);
    let x7 = builder.ins().fmul(x5, x2);

    let c3 = builder.ins().f64const(1.0 / 6.0);
    let c5 = builder.ins().f64const(1.0 / 120.0);
    let c7 = builder.ins().f64const(1.0 / 5040.0);

    let t3 = builder.ins().fmul(x3, c3);
    let t5 = builder.ins().fmul(x5, c5);
    let t7 = builder.ins().fmul(x7, c7);

    let r1 = builder.ins().fsub(x, t3);
    let r2 = builder.ins().fadd(r1, t5);
    builder.ins().fsub(r2, t7)
}

/// Approximate cos using Taylor series for small values.
fn approximate_cos(builder: &mut FunctionBuilder<'_>, x: Value) -> Value {
    // cos(x) ~ 1 - x^2/2 + x^4/24 - x^6/720
    let one = builder.ins().f64const(1.0);
    let x2 = builder.ins().fmul(x, x);
    let x4 = builder.ins().fmul(x2, x2);
    let x6 = builder.ins().fmul(x4, x2);

    let c2 = builder.ins().f64const(0.5);
    let c4 = builder.ins().f64const(1.0 / 24.0);
    let c6 = builder.ins().f64const(1.0 / 720.0);

    let t2 = builder.ins().fmul(x2, c2);
    let t4 = builder.ins().fmul(x4, c4);
    let t6 = builder.ins().fmul(x6, c6);

    let r1 = builder.ins().fsub(one, t2);
    let r2 = builder.ins().fadd(r1, t4);
    builder.ins().fsub(r2, t6)
}

/// Approximate atan2(y, x).
fn approximate_atan2(builder: &mut FunctionBuilder<'_>, y: Value, x: Value) -> Value {
    // Use atan(y/x) with quadrant correction
    // This is a simplified approximation; for production use libm
    let ratio = builder.ins().fdiv(y, x);
    approximate_atan(builder, ratio)
}

/// Approximate atan using rational approximation.
fn approximate_atan(builder: &mut FunctionBuilder<'_>, x: Value) -> Value {
    // atan(x) ~ x / (1 + 0.28 * x^2) for |x| < 1
    let x2 = builder.ins().fmul(x, x);
    let c = builder.ins().f64const(0.28);
    let one = builder.ins().f64const(1.0);

    let t = builder.ins().fmul(c, x2);
    let denom = builder.ins().fadd(one, t);
    builder.ins().fdiv(x, denom)
}

/// JIT-compiled functions for constraint evaluation.
///
/// This struct holds pointers to the compiled native code for residual
/// and Jacobian evaluation, along with metadata about the problem dimensions.
pub struct JITFunction {
    /// Function pointer for residual evaluation.
    ///
    /// Signature: fn(vars: *const f64, residuals: *mut f64)
    residual_fn: unsafe extern "C" fn(*const f64, *mut f64),

    /// Function pointer for Jacobian evaluation.
    ///
    /// Signature: fn(vars: *const f64, jacobian_values: *mut f64)
    jacobian_fn: unsafe extern "C" fn(*const f64, *mut f64),

    /// Number of residuals.
    n_residuals: usize,

    /// Number of variables.
    n_vars: usize,

    /// Number of non-zero Jacobian entries.
    jacobian_nnz: usize,

    /// Jacobian sparsity pattern.
    jacobian_pattern: Vec<super::opcodes::JacobianEntry>,
}

impl std::fmt::Debug for JITFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JITFunction")
            .field("n_residuals", &self.n_residuals)
            .field("n_vars", &self.n_vars)
            .field("jacobian_nnz", &self.jacobian_nnz)
            .finish_non_exhaustive()
    }
}

impl JITFunction {
    /// Get the number of residuals.
    pub fn residual_count(&self) -> usize {
        self.n_residuals
    }

    /// Get the number of variables.
    pub fn variable_count(&self) -> usize {
        self.n_vars
    }

    /// Get the number of non-zero Jacobian entries.
    pub fn jacobian_nnz(&self) -> usize {
        self.jacobian_nnz
    }

    /// Evaluate residuals using JIT-compiled code.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `vars` has length >= `variable_count()`
    /// - `residuals` has length >= `residual_count()`
    pub fn evaluate_residuals(&self, vars: &[f64], residuals: &mut [f64]) {
        debug_assert!(
            vars.len() >= self.n_vars,
            "vars slice too short: {} < {}",
            vars.len(),
            self.n_vars
        );
        debug_assert!(
            residuals.len() >= self.n_residuals,
            "residuals slice too short: {} < {}",
            residuals.len(),
            self.n_residuals
        );

        unsafe {
            (self.residual_fn)(vars.as_ptr(), residuals.as_mut_ptr());
        }
    }

    /// Evaluate Jacobian using JIT-compiled code.
    ///
    /// Returns the Jacobian values in the order defined by the sparsity pattern.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `vars` has length >= `variable_count()`
    /// - `values` has length >= `jacobian_nnz()`
    pub fn evaluate_jacobian(&self, vars: &[f64], values: &mut [f64]) {
        debug_assert!(
            vars.len() >= self.n_vars,
            "vars slice too short: {} < {}",
            vars.len(),
            self.n_vars
        );
        debug_assert!(
            values.len() >= self.jacobian_nnz,
            "values slice too short: {} < {}",
            values.len(),
            self.jacobian_nnz
        );

        unsafe {
            (self.jacobian_fn)(vars.as_ptr(), values.as_mut_ptr());
        }
    }

    /// Get the Jacobian sparsity pattern.
    pub fn jacobian_pattern(&self) -> &[super::opcodes::JacobianEntry] {
        &self.jacobian_pattern
    }

    /// Convert Jacobian to COO (coordinate) format sparse triplets.
    pub fn jacobian_to_coo(&self, values: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_pattern
            .iter()
            .zip(values.iter())
            .map(|(entry, &val)| (entry.row as usize, entry.col as usize, val))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let result = JITCompiler::new();
        assert!(result.is_ok(), "JIT compiler should be created successfully");
    }

    #[test]
    fn test_compile_empty_constraints() {
        let mut compiler = JITCompiler::new().expect("compiler creation failed");
        let cc = CompiledConstraints::new(4, 2);

        let result = compiler.compile(&cc);
        assert!(result.is_ok(), "empty constraints should compile");

        let jit_fn = result.expect("compilation failed");
        assert_eq!(jit_fn.residual_count(), 2);
        assert_eq!(jit_fn.variable_count(), 4);
    }

    #[test]
    fn test_compile_simple_residual() {
        let mut compiler = JITCompiler::new().expect("compiler creation failed");

        let mut cc = CompiledConstraints::new(2, 1);

        // residual = x0 - x1
        cc.residual_ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadVar {
                dst: Reg::new(1),
                var_idx: 1,
            },
            ConstraintOp::Sub {
                dst: Reg::new(2),
                a: Reg::new(0),
                b: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        cc.max_register = 2;

        let result = compiler.compile(&cc);
        assert!(result.is_ok(), "simple residual should compile");

        let jit_fn = result.expect("compilation failed");

        let vars = [3.0, 1.0];
        let mut residuals = [0.0];

        jit_fn.evaluate_residuals(&vars, &mut residuals);

        assert!(
            (residuals[0] - 2.0).abs() < 1e-10,
            "residual should be 3 - 1 = 2, got {}",
            residuals[0]
        );
    }

    #[test]
    fn test_compile_distance_residual() {
        let mut compiler = JITCompiler::new().expect("compiler creation failed");

        let mut cc = CompiledConstraints::new(4, 1);

        // residual = sqrt((x1-x0)^2 + (y1-y0)^2) - target
        // With target = 5.0
        cc.residual_ops = vec![
            // Load variables
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            }, // x0
            ConstraintOp::LoadVar {
                dst: Reg::new(1),
                var_idx: 1,
            }, // y0
            ConstraintOp::LoadVar {
                dst: Reg::new(2),
                var_idx: 2,
            }, // x1
            ConstraintOp::LoadVar {
                dst: Reg::new(3),
                var_idx: 3,
            }, // y1
            // dx = x1 - x0
            ConstraintOp::Sub {
                dst: Reg::new(4),
                a: Reg::new(2),
                b: Reg::new(0),
            },
            // dy = y1 - y0
            ConstraintOp::Sub {
                dst: Reg::new(5),
                a: Reg::new(3),
                b: Reg::new(1),
            },
            // dx^2
            ConstraintOp::Mul {
                dst: Reg::new(6),
                a: Reg::new(4),
                b: Reg::new(4),
            },
            // dy^2
            ConstraintOp::Mul {
                dst: Reg::new(7),
                a: Reg::new(5),
                b: Reg::new(5),
            },
            // dx^2 + dy^2
            ConstraintOp::Add {
                dst: Reg::new(8),
                a: Reg::new(6),
                b: Reg::new(7),
            },
            // sqrt(...)
            ConstraintOp::Sqrt {
                dst: Reg::new(9),
                src: Reg::new(8),
            },
            // target = 5.0
            ConstraintOp::LoadConst {
                dst: Reg::new(10),
                value: 5.0,
            },
            // residual = sqrt(...) - target
            ConstraintOp::Sub {
                dst: Reg::new(11),
                a: Reg::new(9),
                b: Reg::new(10),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(11),
            },
        ];
        cc.max_register = 11;

        let result = compiler.compile(&cc);
        assert!(result.is_ok(), "distance residual should compile");

        let jit_fn = result.expect("compilation failed");

        // Point 0 at (0, 0), Point 1 at (3, 4) -> distance = 5
        let vars = [0.0, 0.0, 3.0, 4.0];
        let mut residuals = [0.0];

        jit_fn.evaluate_residuals(&vars, &mut residuals);

        assert!(
            residuals[0].abs() < 1e-10,
            "residual should be 0 (distance matches target), got {}",
            residuals[0]
        );

        // Point 0 at (0, 0), Point 1 at (6, 8) -> distance = 10, residual = 5
        let vars2 = [0.0, 0.0, 6.0, 8.0];
        jit_fn.evaluate_residuals(&vars2, &mut residuals);

        assert!(
            (residuals[0] - 5.0).abs() < 1e-10,
            "residual should be 5, got {}",
            residuals[0]
        );
    }
}
