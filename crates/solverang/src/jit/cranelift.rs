//! Cranelift JIT code generation.
//!
//! This module compiles constraint opcodes to native machine code using Cranelift.

use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use codegen::ir::FuncRef;

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

// ============================================================================
// extern "C" wrappers for math functions.
//
// Cranelift emits calls using the platform C calling convention. Rust's
// f64::sin etc. use Rust ABI which may differ. These wrappers guarantee
// ABI compatibility.
// ============================================================================

extern "C" fn jit_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn jit_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn jit_tan(x: f64) -> f64 {
    x.tan()
}
extern "C" fn jit_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn jit_ln(x: f64) -> f64 {
    x.ln()
}
extern "C" fn jit_pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}
extern "C" fn jit_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}
extern "C" fn jit_asin(x: f64) -> f64 {
    x.asin()
}
extern "C" fn jit_acos(x: f64) -> f64 {
    x.acos()
}
extern "C" fn jit_sinh(x: f64) -> f64 {
    x.sinh()
}
extern "C" fn jit_cosh(x: f64) -> f64 {
    x.cosh()
}
extern "C" fn jit_tanh(x: f64) -> f64 {
    x.tanh()
}

/// Cached FuncIds for math functions declared in the JIT module.
struct MathFunctions {
    sin: FuncId,
    cos: FuncId,
    tan: FuncId,
    exp: FuncId,
    ln: FuncId,
    pow: FuncId,
    atan2: FuncId,
    asin: FuncId,
    acos: FuncId,
    sinh: FuncId,
    cosh: FuncId,
    tanh: FuncId,
}

/// JIT compiler using Cranelift.
pub struct JITCompiler {
    /// Cranelift JIT module.
    module: JITModule,

    /// Cranelift codegen context.
    ctx: codegen::Context,

    /// Function builder context.
    builder_ctx: FunctionBuilderContext,

    /// Cached math function IDs for calling libm.
    math: MathFunctions,
}

impl JITCompiler {
    /// Create a new JIT compiler.
    pub fn new() -> Result<Self, JITError> {
        let mut flag_builder = settings::builder();
        // Enable optimizations
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| JITError::ModuleError(format!("failed to set opt_level: {}", e)))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|e| JITError::ModuleError(format!("failed to create ISA builder: {}", e)))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JITError::ModuleError(format!("failed to create ISA: {}", e)))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register math functions as callable symbols via extern "C" wrappers.
        builder.symbol("jit_sin", jit_sin as *const u8);
        builder.symbol("jit_cos", jit_cos as *const u8);
        builder.symbol("jit_tan", jit_tan as *const u8);
        builder.symbol("jit_exp", jit_exp as *const u8);
        builder.symbol("jit_ln", jit_ln as *const u8);
        builder.symbol("jit_pow", jit_pow as *const u8);
        builder.symbol("jit_atan2", jit_atan2 as *const u8);
        builder.symbol("jit_asin", jit_asin as *const u8);
        builder.symbol("jit_acos", jit_acos as *const u8);
        builder.symbol("jit_sinh", jit_sinh as *const u8);
        builder.symbol("jit_cosh", jit_cosh as *const u8);
        builder.symbol("jit_tanh", jit_tanh as *const u8);

        let mut module = JITModule::new(builder);

        // Declare math functions in the module with their signatures.
        let math = {
            // Unary: (f64) -> f64
            let mut sig1 = module.make_signature();
            sig1.params.push(AbiParam::new(types::F64));
            sig1.returns.push(AbiParam::new(types::F64));

            // Binary: (f64, f64) -> f64
            let mut sig2 = module.make_signature();
            sig2.params.push(AbiParam::new(types::F64));
            sig2.params.push(AbiParam::new(types::F64));
            sig2.returns.push(AbiParam::new(types::F64));

            let sin = module
                .declare_function("jit_sin", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_sin: {}", e)))?;
            let cos = module
                .declare_function("jit_cos", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_cos: {}", e)))?;
            let tan = module
                .declare_function("jit_tan", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_tan: {}", e)))?;
            let exp = module
                .declare_function("jit_exp", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_exp: {}", e)))?;
            let ln = module
                .declare_function("jit_ln", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_ln: {}", e)))?;
            let pow = module
                .declare_function("jit_pow", Linkage::Import, &sig2)
                .map_err(|e| JITError::ModuleError(format!("declare jit_pow: {}", e)))?;
            let atan2 = module
                .declare_function("jit_atan2", Linkage::Import, &sig2)
                .map_err(|e| JITError::ModuleError(format!("declare jit_atan2: {}", e)))?;
            let asin = module
                .declare_function("jit_asin", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_asin: {}", e)))?;
            let acos = module
                .declare_function("jit_acos", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_acos: {}", e)))?;
            let sinh = module
                .declare_function("jit_sinh", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_sinh: {}", e)))?;
            let cosh = module
                .declare_function("jit_cosh", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_cosh: {}", e)))?;
            let tanh = module
                .declare_function("jit_tanh", Linkage::Import, &sig1)
                .map_err(|e| JITError::ModuleError(format!("declare jit_tanh: {}", e)))?;

            MathFunctions {
                sin,
                cos,
                tan,
                exp,
                ln,
                pow,
                atan2,
                asin,
                acos,
                sinh,
                cosh,
                tanh,
            }
        };

        let ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();

        Ok(Self {
            module,
            ctx,
            builder_ctx,
            math,
        })
    }

    /// Compile a single Newton step into native code.
    ///
    /// For small square systems (N < 30), this compiles:
    /// residual evaluation → dense Jacobian assembly → LU decompose → back-substitute → update x
    ///
    /// The compiled function signature:
    /// `fn(vars_in: *const f64, vars_out: *mut f64, scratch: *mut f64) -> f64`
    ///
    /// scratch layout: `[residuals (m), dense_jacobian (m*n), delta (n)]`
    pub fn compile_newton_step(
        &mut self,
        compiled: &CompiledConstraints,
    ) -> Result<super::compiled_newton::CompiledNewtonStep, JITError> {
        compiled
            .validate()
            .map_err(|e| JITError::ValidationError(e.to_string()))?;

        let m = compiled.n_residuals;
        let n = compiled.n_vars;

        if m != n {
            return Err(JITError::ValidationError(
                "Compiled Newton step requires square systems (m == n)".to_string(),
            ));
        }

        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();

        let ptr_type = self.module.target_config().pointer_type();

        // fn(vars_in: *const f64, vars_out: *mut f64, scratch: *mut f64) -> f64
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.returns.push(AbiParam::new(types::F64));

        let func_id = self
            .module
            .declare_function(
                "newton_step",
                Linkage::Local,
                &self.ctx.func.signature,
            )
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_in = builder.block_params(entry_block)[0];
            let vars_out = builder.block_params(entry_block)[1];
            let scratch_ptr = builder.block_params(entry_block)[2];

            // scratch layout: [residuals (m f64), jacobian (m*n f64), delta (n f64)]
            let residuals_ptr = scratch_ptr;
            let jac_offset = (m * 8) as i64;
            let jac_ptr = builder.ins().iadd_imm(scratch_ptr, jac_offset);
            let delta_offset = ((m + m * n) * 8) as i64;
            let delta_ptr = builder.ins().iadd_imm(scratch_ptr, delta_offset);

            let math_refs = MathFuncRefs {
                sin: self.module.declare_func_in_func(self.math.sin, builder.func),
                cos: self.module.declare_func_in_func(self.math.cos, builder.func),
                tan: self.module.declare_func_in_func(self.math.tan, builder.func),
                exp: self.module.declare_func_in_func(self.math.exp, builder.func),
                ln: self.module.declare_func_in_func(self.math.ln, builder.func),
                pow: self.module.declare_func_in_func(self.math.pow, builder.func),
                atan2: self.module.declare_func_in_func(self.math.atan2, builder.func),
                asin: self.module.declare_func_in_func(self.math.asin, builder.func),
                acos: self.module.declare_func_in_func(self.math.acos, builder.func),
                sinh: self.module.declare_func_in_func(self.math.sinh, builder.func),
                cosh: self.module.declare_func_in_func(self.math.cosh, builder.func),
                tanh: self.module.declare_func_in_func(self.math.tanh, builder.func),
            };

            // Step 1: Zero the dense Jacobian buffer
            let zero = builder.ins().f64const(0.0);
            for i in 0..(m * n) {
                let offset = (i * 8) as i32;
                let addr = builder.ins().iadd_imm(jac_ptr, offset as i64);
                builder.ins().store(MemFlags::trusted(), zero, addr, 0);
            }

            // Step 2: Evaluate residuals and dense Jacobian via fused ops
            let (dense_fused_ops, _) = compiled.fuse_ops_dense(m);
            translate_ops(
                &mut builder,
                &math_refs,
                &dense_fused_ops,
                vars_in,
                residuals_ptr,
                Some(jac_ptr),
            )?;

            // Step 3: Compute residual norm ||F(x)||
            // sum = sum(residuals[i]^2)
            let mut sum_val = builder.ins().f64const(0.0);
            for i in 0..m {
                let addr = builder.ins().iadd_imm(residuals_ptr, (i * 8) as i64);
                let ri = builder.ins().load(types::F64, MemFlags::trusted(), addr, 0);
                let ri_sq = builder.ins().fmul(ri, ri);
                sum_val = builder.ins().fadd(sum_val, ri_sq);
            }
            let norm = builder.ins().sqrt(sum_val);

            // Step 4: Unrolled LU decomposition (no pivoting) on dense column-major Jacobian
            // J is m×n column-major at jac_ptr. Entry (r,c) at offset (c*m + r)*8.
            // After LU: L stored below diagonal, U on/above diagonal (in-place).
            for k in 0..n {
                // Load pivot J[k,k]
                let pivot_offset = ((k * m + k) * 8) as i64;
                let pivot_addr = builder.ins().iadd_imm(jac_ptr, pivot_offset);
                let pivot = builder.ins().load(types::F64, MemFlags::trusted(), pivot_addr, 0);

                for i in (k + 1)..m {
                    // L[i,k] = J[i,k] / J[k,k]
                    let ik_offset = ((k * m + i) * 8) as i64;
                    let ik_addr = builder.ins().iadd_imm(jac_ptr, ik_offset);
                    let jik = builder.ins().load(types::F64, MemFlags::trusted(), ik_addr, 0);
                    let lik = builder.ins().fdiv(jik, pivot);
                    builder.ins().store(MemFlags::trusted(), lik, ik_addr, 0);

                    for j in (k + 1)..n {
                        // J[i,j] -= L[i,k] * J[k,j]
                        let ij_offset = ((j * m + i) * 8) as i64;
                        let ij_addr = builder.ins().iadd_imm(jac_ptr, ij_offset);
                        let jij = builder.ins().load(types::F64, MemFlags::trusted(), ij_addr, 0);

                        let kj_offset = ((j * m + k) * 8) as i64;
                        let kj_addr = builder.ins().iadd_imm(jac_ptr, kj_offset);
                        let jkj = builder.ins().load(types::F64, MemFlags::trusted(), kj_addr, 0);

                        let prod = builder.ins().fmul(lik, jkj);
                        let new_jij = builder.ins().fsub(jij, prod);
                        builder.ins().store(MemFlags::trusted(), new_jij, ij_addr, 0);
                    }
                }
            }

            // Step 5: Forward substitution — solve L*y = -r
            // Store y in delta buffer. L is lower triangular (below diagonal of LU),
            // with implicit 1s on diagonal.
            for i in 0..n {
                // y[i] = -r[i]
                let ri_addr = builder.ins().iadd_imm(residuals_ptr, (i * 8) as i64);
                let ri = builder.ins().load(types::F64, MemFlags::trusted(), ri_addr, 0);
                let mut yi = builder.ins().fneg(ri);

                for j in 0..i {
                    // y[i] -= L[i,j] * y[j]
                    let lij_offset = ((j * m + i) * 8) as i64;
                    let lij_addr = builder.ins().iadd_imm(jac_ptr, lij_offset);
                    let lij = builder.ins().load(types::F64, MemFlags::trusted(), lij_addr, 0);

                    let yj_addr = builder.ins().iadd_imm(delta_ptr, (j * 8) as i64);
                    let yj = builder.ins().load(types::F64, MemFlags::trusted(), yj_addr, 0);

                    let prod = builder.ins().fmul(lij, yj);
                    yi = builder.ins().fsub(yi, prod);
                }

                let yi_addr = builder.ins().iadd_imm(delta_ptr, (i * 8) as i64);
                builder.ins().store(MemFlags::trusted(), yi, yi_addr, 0);
            }

            // Step 6: Back substitution — solve U*delta = y
            // U is upper triangular (on/above diagonal of LU). delta overwrites y.
            for i in (0..n).rev() {
                let yi_addr = builder.ins().iadd_imm(delta_ptr, (i * 8) as i64);
                let mut di = builder.ins().load(types::F64, MemFlags::trusted(), yi_addr, 0);

                for j in (i + 1)..n {
                    // di -= U[i,j] * delta[j]
                    let uij_offset = ((j * m + i) * 8) as i64;
                    let uij_addr = builder.ins().iadd_imm(jac_ptr, uij_offset);
                    let uij = builder.ins().load(types::F64, MemFlags::trusted(), uij_addr, 0);

                    let dj_addr = builder.ins().iadd_imm(delta_ptr, (j * 8) as i64);
                    let dj = builder.ins().load(types::F64, MemFlags::trusted(), dj_addr, 0);

                    let prod = builder.ins().fmul(uij, dj);
                    di = builder.ins().fsub(di, prod);
                }

                // di /= U[i,i]
                let uii_offset = ((i * m + i) * 8) as i64;
                let uii_addr = builder.ins().iadd_imm(jac_ptr, uii_offset);
                let uii = builder.ins().load(types::F64, MemFlags::trusted(), uii_addr, 0);
                di = builder.ins().fdiv(di, uii);

                builder.ins().store(MemFlags::trusted(), di, yi_addr, 0);
            }

            // Step 7: Update vars_out = vars_in + delta
            for i in 0..n {
                let xi_addr = builder.ins().iadd_imm(vars_in, (i * 8) as i64);
                let xi = builder.ins().load(types::F64, MemFlags::trusted(), xi_addr, 0);

                let di_addr = builder.ins().iadd_imm(delta_ptr, (i * 8) as i64);
                let di = builder.ins().load(types::F64, MemFlags::trusted(), di_addr, 0);

                let xi_new = builder.ins().fadd(xi, di);
                let xo_addr = builder.ins().iadd_imm(vars_out, (i * 8) as i64);
                builder.ins().store(MemFlags::trusted(), xi_new, xo_addr, 0);
            }

            // Return residual norm
            builder.ins().return_(&[norm]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JITError::CodegenError(format!("failed to define function: {}", e)))?;

        self.module
            .finalize_definitions()
            .map_err(|e| JITError::ModuleError(format!("failed to finalize definitions: {}", e)))?;

        let ptr = self.module.get_finalized_function(func_id);

        type StepFnPtr = unsafe extern "C" fn(*const f64, *mut f64, *mut f64) -> f64;
        let step_fn = unsafe { std::mem::transmute::<*const u8, StepFnPtr>(ptr) };

        self.ctx.clear();

        Ok(super::compiled_newton::CompiledNewtonStep::new(step_fn, n))
    }

    /// Compile constraint opcodes to native functions.
    pub fn compile(&mut self, compiled: &CompiledConstraints) -> Result<JITFunction, JITError> {
        // Validate the compiled constraints
        compiled
            .validate()
            .map_err(|e| JITError::ValidationError(e.to_string()))?;

        // Compile residual evaluation function
        let residual_fn_id = self.compile_residuals(compiled)?;

        // Compile Jacobian evaluation function
        let jacobian_fn_id = self.compile_jacobian(compiled)?;

        // Compile fused residual+Jacobian evaluation function
        let fused_fn_id = self.compile_fused(compiled)?;

        // Compile fused dense residual+Jacobian evaluation function
        let dense_fused_fn_id = self.compile_fused_dense(compiled)?;

        // Finalize all function definitions
        self.module
            .finalize_definitions()
            .map_err(|e| JITError::ModuleError(format!("failed to finalize definitions: {}", e)))?;

        // Get function pointers
        let residual_ptr = self.module.get_finalized_function(residual_fn_id);
        let jacobian_ptr = self.module.get_finalized_function(jacobian_fn_id);
        let fused_ptr = self.module.get_finalized_function(fused_fn_id);
        let dense_fused_ptr = self.module.get_finalized_function(dense_fused_fn_id);

        // Type aliases for JIT function signatures
        type JITFnPtr = unsafe extern "C" fn(*const f64, *mut f64);
        type FusedFnPtr = unsafe extern "C" fn(*const f64, *mut f64, *mut f64);

        Ok(JITFunction {
            residual_fn: unsafe { std::mem::transmute::<*const u8, JITFnPtr>(residual_ptr) },
            jacobian_fn: unsafe { std::mem::transmute::<*const u8, JITFnPtr>(jacobian_ptr) },
            fused_fn: unsafe { std::mem::transmute::<*const u8, FusedFnPtr>(fused_ptr) },
            dense_fused_fn: unsafe { std::mem::transmute::<*const u8, FusedFnPtr>(dense_fused_ptr) },
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
            .declare_function(
                "evaluate_residuals",
                Linkage::Local,
                &self.ctx.func.signature,
            )
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_ptr = builder.block_params(entry_block)[0];
            let residuals_ptr = builder.block_params(entry_block)[1];

            // Import math functions into this Cranelift function.
            let math_refs = MathFuncRefs {
                sin: self.module.declare_func_in_func(self.math.sin, builder.func),
                cos: self.module.declare_func_in_func(self.math.cos, builder.func),
                tan: self.module.declare_func_in_func(self.math.tan, builder.func),
                exp: self.module.declare_func_in_func(self.math.exp, builder.func),
                ln: self.module.declare_func_in_func(self.math.ln, builder.func),
                pow: self.module.declare_func_in_func(self.math.pow, builder.func),
                atan2: self.module.declare_func_in_func(self.math.atan2, builder.func),
                asin: self.module.declare_func_in_func(self.math.asin, builder.func),
                acos: self.module.declare_func_in_func(self.math.acos, builder.func),
                sinh: self.module.declare_func_in_func(self.math.sinh, builder.func),
                cosh: self.module.declare_func_in_func(self.math.cosh, builder.func),
                tanh: self.module.declare_func_in_func(self.math.tanh, builder.func),
            };

            // Translate opcodes
            translate_ops(
                &mut builder,
                &math_refs,
                &compiled.residual_ops,
                vars_ptr,
                residuals_ptr,
                None,
            )?;

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JITError::CodegenError(format!("failed to define function: {}", e)))?;

        self.ctx.clear();

        Ok(func_id)
    }

    /// Compile fused residual+Jacobian evaluation function.
    ///
    /// The fused function computes both residuals and Jacobian in a single pass,
    /// sharing variable loads between the two computations.
    fn compile_fused(&mut self, compiled: &CompiledConstraints) -> Result<FuncId, JITError> {
        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();

        let ptr_type = self.module.target_config().pointer_type();

        // Function signature: fn(vars: *const f64, residuals: *mut f64, jacobian_values: *mut f64)
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(
                "evaluate_fused",
                Linkage::Local,
                &self.ctx.func.signature,
            )
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_ptr = builder.block_params(entry_block)[0];
            let residuals_ptr = builder.block_params(entry_block)[1];
            let jacobian_ptr = builder.block_params(entry_block)[2];

            let math_refs = MathFuncRefs {
                sin: self.module.declare_func_in_func(self.math.sin, builder.func),
                cos: self.module.declare_func_in_func(self.math.cos, builder.func),
                tan: self.module.declare_func_in_func(self.math.tan, builder.func),
                exp: self.module.declare_func_in_func(self.math.exp, builder.func),
                ln: self.module.declare_func_in_func(self.math.ln, builder.func),
                pow: self.module.declare_func_in_func(self.math.pow, builder.func),
                atan2: self.module.declare_func_in_func(self.math.atan2, builder.func),
                asin: self.module.declare_func_in_func(self.math.asin, builder.func),
                acos: self.module.declare_func_in_func(self.math.acos, builder.func),
                sinh: self.module.declare_func_in_func(self.math.sinh, builder.func),
                cosh: self.module.declare_func_in_func(self.math.cosh, builder.func),
                tanh: self.module.declare_func_in_func(self.math.tanh, builder.func),
            };

            // Use fused opcode stream with deduplicated variable loads
            let (fused_ops, _) = compiled.fuse_ops();

            translate_ops(
                &mut builder,
                &math_refs,
                &fused_ops,
                vars_ptr,
                residuals_ptr,
                Some(jacobian_ptr),
            )?;

            builder.ins().return_(&[]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JITError::CodegenError(format!("failed to define function: {}", e)))?;

        self.ctx.clear();

        Ok(func_id)
    }

    /// Compile fused residual+Jacobian function that writes Jacobian directly
    /// into dense column-major storage.
    ///
    /// Signature: fn(vars: *const f64, residuals: *mut f64, dense_jacobian: *mut f64)
    fn compile_fused_dense(
        &mut self,
        compiled: &CompiledConstraints,
    ) -> Result<FuncId, JITError> {
        self.ctx.func.signature.params.clear();
        self.ctx.func.signature.returns.clear();

        let ptr_type = self.module.target_config().pointer_type();

        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(
                "evaluate_fused_dense",
                Linkage::Local,
                &self.ctx.func.signature,
            )
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_ptr = builder.block_params(entry_block)[0];
            let residuals_ptr = builder.block_params(entry_block)[1];
            let dense_jac_ptr = builder.block_params(entry_block)[2];

            let math_refs = MathFuncRefs {
                sin: self.module.declare_func_in_func(self.math.sin, builder.func),
                cos: self.module.declare_func_in_func(self.math.cos, builder.func),
                tan: self.module.declare_func_in_func(self.math.tan, builder.func),
                exp: self.module.declare_func_in_func(self.math.exp, builder.func),
                ln: self.module.declare_func_in_func(self.math.ln, builder.func),
                pow: self.module.declare_func_in_func(self.math.pow, builder.func),
                atan2: self.module.declare_func_in_func(self.math.atan2, builder.func),
                asin: self.module.declare_func_in_func(self.math.asin, builder.func),
                acos: self.module.declare_func_in_func(self.math.acos, builder.func),
                sinh: self.module.declare_func_in_func(self.math.sinh, builder.func),
                cosh: self.module.declare_func_in_func(self.math.cosh, builder.func),
                tanh: self.module.declare_func_in_func(self.math.tanh, builder.func),
            };

            // Use fused+densified opcode stream
            let (dense_fused_ops, _) = compiled.fuse_ops_dense(compiled.n_residuals);

            translate_ops(
                &mut builder,
                &math_refs,
                &dense_fused_ops,
                vars_ptr,
                residuals_ptr,
                Some(dense_jac_ptr),
            )?;

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
            .declare_function(
                "evaluate_jacobian",
                Linkage::Local,
                &self.ctx.func.signature,
            )
            .map_err(|e| JITError::ModuleError(format!("failed to declare function: {}", e)))?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let vars_ptr = builder.block_params(entry_block)[0];
            let jacobian_ptr = builder.block_params(entry_block)[1];

            let math_refs = MathFuncRefs {
                sin: self.module.declare_func_in_func(self.math.sin, builder.func),
                cos: self.module.declare_func_in_func(self.math.cos, builder.func),
                tan: self.module.declare_func_in_func(self.math.tan, builder.func),
                exp: self.module.declare_func_in_func(self.math.exp, builder.func),
                ln: self.module.declare_func_in_func(self.math.ln, builder.func),
                pow: self.module.declare_func_in_func(self.math.pow, builder.func),
                atan2: self.module.declare_func_in_func(self.math.atan2, builder.func),
                asin: self.module.declare_func_in_func(self.math.asin, builder.func),
                acos: self.module.declare_func_in_func(self.math.acos, builder.func),
                sinh: self.module.declare_func_in_func(self.math.sinh, builder.func),
                cosh: self.module.declare_func_in_func(self.math.cosh, builder.func),
                tanh: self.module.declare_func_in_func(self.math.tanh, builder.func),
            };

            // Translate opcodes
            translate_ops(
                &mut builder,
                &math_refs,
                &compiled.jacobian_ops,
                vars_ptr,
                jacobian_ptr,
                Some(jacobian_ptr),
            )?;

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

/// Function references for math calls within a single Cranelift function.
struct MathFuncRefs {
    sin: FuncRef,
    cos: FuncRef,
    tan: FuncRef,
    exp: FuncRef,
    ln: FuncRef,
    pow: FuncRef,
    atan2: FuncRef,
    asin: FuncRef,
    acos: FuncRef,
    sinh: FuncRef,
    cosh: FuncRef,
    tanh: FuncRef,
}

/// Translate opcodes to Cranelift IR.
///
/// Math functions (sin, cos, tan, exp, ln, pow, atan2) are called via
/// registered extern "C" wrappers — no Taylor approximations.
fn translate_ops(
    builder: &mut FunctionBuilder<'_>,
    math_refs: &MathFuncRefs,
    ops: &[ConstraintOp],
    vars_ptr: Value,
    output_ptr: Value,
    jacobian_ptr: Option<Value>,
) -> Result<(), JITError> {
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
                let a_val = get_reg(&registers, *a)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", a)))?;
                let b_val = get_reg(&registers, *b)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", b)))?;
                let result = builder.ins().fadd(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Sub { dst, a, b } => {
                let a_val = get_reg(&registers, *a)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", a)))?;
                let b_val = get_reg(&registers, *b)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", b)))?;
                let result = builder.ins().fsub(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Mul { dst, a, b } => {
                let a_val = get_reg(&registers, *a)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", a)))?;
                let b_val = get_reg(&registers, *b)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", b)))?;
                let result = builder.ins().fmul(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Div { dst, a, b } => {
                let a_val = get_reg(&registers, *a)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", a)))?;
                let b_val = get_reg(&registers, *b)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", b)))?;
                let result = builder.ins().fdiv(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Neg { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let result = builder.ins().fneg(src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Sqrt { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let result = builder.ins().sqrt(src_val);
                registers.insert(*dst, result);
            }

            // --- Math functions via libm calls ---

            ConstraintOp::Sin { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.sin, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Cos { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.cos, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Tan { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.tan, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Exp { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.exp, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Ln { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.ln, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Pow { dst, base, exp } => {
                let base_val = get_reg(&registers, *base)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", base)))?;
                let exp_val = get_reg(&registers, *exp)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", exp)))?;
                let call = builder.ins().call(math_refs.pow, &[base_val, exp_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Atan2 { dst, y, x } => {
                let y_val = get_reg(&registers, *y)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", y)))?;
                let x_val = get_reg(&registers, *x)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", x)))?;
                let call = builder.ins().call(math_refs.atan2, &[y_val, x_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Asin { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.asin, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Acos { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.acos, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Sinh { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.sinh, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Cosh { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.cosh, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            ConstraintOp::Tanh { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let call = builder.ins().call(math_refs.tanh, &[src_val]);
                let result = builder.inst_results(call)[0];
                registers.insert(*dst, result);
            }

            // --- Non-math ops ---

            ConstraintOp::Abs { dst, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let result = builder.ins().fabs(src_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Max { dst, a, b } => {
                let a_val = get_reg(&registers, *a)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", a)))?;
                let b_val = get_reg(&registers, *b)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", b)))?;
                let result = builder.ins().fmax(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::Min { dst, a, b } => {
                let a_val = get_reg(&registers, *a)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", a)))?;
                let b_val = get_reg(&registers, *b)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", b)))?;
                let result = builder.ins().fmin(a_val, b_val);
                registers.insert(*dst, result);
            }

            ConstraintOp::StoreResidual { residual_idx, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let offset = (*residual_idx as i32) * 8;
                let addr = builder.ins().iadd_imm(output_ptr, offset as i64);
                builder.ins().store(MemFlags::trusted(), src_val, addr, 0);
            }

            ConstraintOp::StoreJacobianIndexed { output_idx, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let jac_ptr = jacobian_ptr.unwrap_or(output_ptr);
                let offset = (*output_idx as i32) * 8;
                let addr = builder.ins().iadd_imm(jac_ptr, offset as i64);
                builder.ins().store(MemFlags::trusted(), src_val, addr, 0);
            }

            ConstraintOp::StoreHessianIndexed { output_idx, src } => {
                let src_val = get_reg(&registers, *src)
                    .ok_or_else(|| JITError::CodegenError(format!("undefined register {}", src)))?;
                let hess_ptr = jacobian_ptr.unwrap_or(output_ptr);
                let offset = (*output_idx as i32) * 8;
                let addr = builder.ins().iadd_imm(hess_ptr, offset as i64);
                builder.ins().store(MemFlags::trusted(), src_val, addr, 0);
            }
        }
    }

    Ok(())
}

fn get_reg(registers: &HashMap<Reg, Value>, reg: Reg) -> Option<Value> {
    registers.get(&reg).copied()
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

    /// Function pointer for fused residual+Jacobian evaluation (sparse COO output).
    ///
    /// Signature: fn(vars: *const f64, residuals: *mut f64, jacobian_values: *mut f64)
    fused_fn: unsafe extern "C" fn(*const f64, *mut f64, *mut f64),

    /// Function pointer for fused evaluation writing Jacobian directly into
    /// dense column-major storage.
    ///
    /// Signature: fn(vars: *const f64, residuals: *mut f64, dense_jacobian: *mut f64)
    dense_fused_fn: unsafe extern "C" fn(*const f64, *mut f64, *mut f64),

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

    /// Evaluate both residuals and Jacobian using a single fused JIT function.
    ///
    /// This is faster than calling `evaluate_residuals` and `evaluate_jacobian`
    /// separately because variable loads are shared between the two computations.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `vars` has length >= `variable_count()`
    /// - `residuals` has length >= `residual_count()`
    /// - `jacobian_values` has length >= `jacobian_nnz()`
    pub fn evaluate_both(
        &self,
        vars: &[f64],
        residuals: &mut [f64],
        jacobian_values: &mut [f64],
    ) {
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
        debug_assert!(
            jacobian_values.len() >= self.jacobian_nnz,
            "jacobian values slice too short: {} < {}",
            jacobian_values.len(),
            self.jacobian_nnz
        );

        unsafe {
            (self.fused_fn)(
                vars.as_ptr(),
                residuals.as_mut_ptr(),
                jacobian_values.as_mut_ptr(),
            );
        }
    }

    /// Evaluate both residuals and Jacobian, writing the Jacobian directly
    /// into dense column-major storage.
    ///
    /// The `dense_jacobian` buffer must have length >= `residual_count() * variable_count()`
    /// and is zeroed before the JIT call (JIT only writes non-zero entries).
    pub fn evaluate_both_dense(
        &self,
        vars: &[f64],
        residuals: &mut [f64],
        dense_jacobian: &mut [f64],
    ) {
        let m = self.n_residuals;
        let n = self.n_vars;
        debug_assert!(
            vars.len() >= n,
            "vars slice too short: {} < {}",
            vars.len(),
            n
        );
        debug_assert!(
            residuals.len() >= m,
            "residuals slice too short: {} < {}",
            residuals.len(),
            m
        );
        debug_assert!(
            dense_jacobian.len() >= m * n,
            "dense_jacobian slice too short: {} < {}",
            dense_jacobian.len(),
            m * n
        );

        // Zero the dense matrix (JIT only writes non-zero entries)
        dense_jacobian[..m * n].iter_mut().for_each(|v| *v = 0.0);

        unsafe {
            (self.dense_fused_fn)(
                vars.as_ptr(),
                residuals.as_mut_ptr(),
                dense_jacobian.as_mut_ptr(),
            );
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
        assert!(
            result.is_ok(),
            "JIT compiler should be created successfully"
        );
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
