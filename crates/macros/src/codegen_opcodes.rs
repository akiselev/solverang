//! Opcode code generation from the expression AST.
//!
//! This module generates Rust code that calls `OpcodeEmitter` methods to emit
//! JIT opcodes for a given expression tree. The generated code, when compiled,
//! builds an opcode stream at runtime that can be JIT-compiled to native code.
//!
//! The key insight: `Expr::RuntimeConst("self.target")` generates
//! `emitter.const_f64(self.target as f64)`, so struct field values are baked
//! into the opcode stream as `LoadConst` instructions when `lower_to_compiled_constraints`
//! is called on a concrete instance.

use crate::expr::{Expr, VarRef};
use proc_macro2::TokenStream;
use quote::quote;

impl Expr {
    /// Generate Rust code that calls `OpcodeEmitter` methods to emit opcodes
    /// for this expression.
    ///
    /// The returned `TokenStream` evaluates to a `Reg` (the register holding
    /// the result of this expression).
    ///
    /// `emitter` is the identifier of the `&mut OpcodeEmitter` variable in scope.
    pub fn to_opcode_tokens(&self, emitter: &proc_macro2::Ident) -> TokenStream {
        match self {
            Expr::Var(vref) => {
                let idx: TokenStream = vref
                    .index_tokens
                    .parse()
                    .expect("valid variable index tokens");
                quote! { #emitter.load_var(#idx as u32) }
            }

            Expr::Const(v) => {
                if v.is_nan() {
                    quote! { #emitter.const_f64(f64::NAN) }
                } else if v.is_infinite() {
                    if *v > 0.0 {
                        quote! { #emitter.const_f64(f64::INFINITY) }
                    } else {
                        quote! { #emitter.const_f64(f64::NEG_INFINITY) }
                    }
                } else {
                    quote! { #emitter.const_f64(#v) }
                }
            }

            Expr::RuntimeConst(tokens) => {
                // self.target → emitter.const_f64(self.target as f64)
                // At runtime, self.target evaluates to the instance's field value,
                // which becomes a LoadConst opcode with that numeric value baked in.
                let rt_tokens: TokenStream = tokens.parse().expect("valid runtime const tokens");
                quote! { #emitter.const_f64(#rt_tokens as f64) }
            }

            Expr::Neg(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.neg(__e)
                } }
            }

            Expr::Add(a, b) => {
                let a_code = a.to_opcode_tokens(emitter);
                let b_code = b.to_opcode_tokens(emitter);
                quote! { {
                    let __a = #a_code;
                    let __b = #b_code;
                    #emitter.add(__a, __b)
                } }
            }

            Expr::Sub(a, b) => {
                let a_code = a.to_opcode_tokens(emitter);
                let b_code = b.to_opcode_tokens(emitter);
                quote! { {
                    let __a = #a_code;
                    let __b = #b_code;
                    #emitter.sub(__a, __b)
                } }
            }

            Expr::Mul(a, b) => {
                let a_code = a.to_opcode_tokens(emitter);
                let b_code = b.to_opcode_tokens(emitter);
                quote! { {
                    let __a = #a_code;
                    let __b = #b_code;
                    #emitter.mul(__a, __b)
                } }
            }

            Expr::Div(a, b) => {
                let a_code = a.to_opcode_tokens(emitter);
                let b_code = b.to_opcode_tokens(emitter);
                quote! { {
                    let __a = #a_code;
                    let __b = #b_code;
                    #emitter.div(__a, __b)
                } }
            }

            Expr::Sqrt(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.sqrt(__e)
                } }
            }

            Expr::Sin(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.sin(__e)
                } }
            }

            Expr::Cos(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.cos(__e)
                } }
            }

            Expr::Tan(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.tan(__e)
                } }
            }

            Expr::Ln(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.ln(__e)
                } }
            }

            Expr::Exp(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.exp(__e)
                } }
            }

            Expr::Atan2(y, x) => {
                let y_code = y.to_opcode_tokens(emitter);
                let x_code = x.to_opcode_tokens(emitter);
                quote! { {
                    let __y = #y_code;
                    let __x = #x_code;
                    #emitter.atan2(__y, __x)
                } }
            }

            Expr::Pow(base, const_exp) => {
                // Expr::Pow has a constant exponent. Load it as a const register,
                // then call emitter.pow(base_reg, exp_reg).
                let base_code = base.to_opcode_tokens(emitter);
                let exp_val = *const_exp;
                quote! { {
                    let __base = #base_code;
                    let __exp = #emitter.const_f64(#exp_val);
                    #emitter.pow(__base, __exp)
                } }
            }

            Expr::Abs(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.abs(__e)
                } }
            }
            Expr::Asin(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.asin(__e)
                } }
            }
            Expr::Acos(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.acos(__e)
                } }
            }
            Expr::Sinh(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.sinh(__e)
                } }
            }
            Expr::Cosh(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.cosh(__e)
                } }
            }
            Expr::Tanh(e) => {
                let e_code = e.to_opcode_tokens(emitter);
                quote! { {
                    let __e = #e_code;
                    #emitter.tanh(__e)
                } }
            }
        }
    }
}

/// Generate the `lower_residual_ops` method body.
///
/// Emits opcodes for the residual expression and stores the result.
pub fn generate_residual_opcode_method(
    residual_expr: &Expr,
    emitter_ident: &proc_macro2::Ident,
    residual_idx_ident: &proc_macro2::Ident,
) -> TokenStream {
    let opcode_tokens = residual_expr.to_opcode_tokens(emitter_ident);
    quote! {
        let __result = #opcode_tokens;
        #emitter_ident.store_residual(#residual_idx_ident, __result);
    }
}

/// Generate the `lower_jacobian_ops` method body.
///
/// For each variable with a non-zero derivative, emits opcodes for the
/// derivative expression and stores the Jacobian entry.
pub fn generate_jacobian_opcode_method(
    residual_expr: &Expr,
    variables: &[VarRef],
    emitter_ident: &proc_macro2::Ident,
    residual_idx_ident: &proc_macro2::Ident,
) -> TokenStream {
    let mut stmts = Vec::new();

    for var in variables {
        let derivative = residual_expr.differentiate(var.id).simplify();

        // Skip zero derivatives — no Jacobian entry needed.
        if derivative.is_zero() {
            continue;
        }

        let deriv_tokens = derivative.to_opcode_tokens(emitter_ident);
        let col: TokenStream = var
            .index_tokens
            .parse()
            .expect("valid variable index tokens");

        stmts.push(quote! {
            {
                let __deriv = #deriv_tokens;
                #emitter_ident.store_jacobian(#residual_idx_ident, #col as u32, __deriv);
            }
        });
    }

    quote! {
        #(#stmts)*
    }
}

/// Generate the `lower_hessian_ops` method body.
///
/// For each (i,j) pair where i >= j (lower triangle) with a non-zero second
/// derivative, emits opcodes for the second derivative expression and stores
/// the Hessian entry.
pub fn generate_hessian_opcode_method(
    objective_expr: &Expr,
    variables: &[VarRef],
    emitter_ident: &proc_macro2::Ident,
) -> TokenStream {
    let mut stmts = Vec::new();

    for (j_idx, var_j) in variables.iter().enumerate() {
        let first_deriv = objective_expr.differentiate(var_j.id).simplify();
        if first_deriv.is_zero() {
            continue;
        }

        for (i_idx, var_i) in variables.iter().enumerate() {
            if i_idx < j_idx {
                continue; // lower triangle only: i >= j
            }

            let second_deriv = first_deriv.differentiate(var_i.id).simplify();
            if second_deriv.is_zero() {
                continue;
            }

            let deriv_tokens = second_deriv.to_opcode_tokens(emitter_ident);
            let row: TokenStream = var_i
                .index_tokens
                .parse()
                .expect("valid variable index tokens");
            let col: TokenStream = var_j
                .index_tokens
                .parse()
                .expect("valid variable index tokens");

            stmts.push(quote! {
                {
                    let __deriv = #deriv_tokens;
                    #emitter_ident.store_hessian(#row as u32, #col as u32, __deriv);
                }
            });
        }
    }

    quote! {
        #(#stmts)*
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_opcode_tokens() {
        let expr = Expr::var("0".to_string(), 0);
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("load_var"),
            "Var should generate load_var, got: {}",
            code
        );
    }

    #[test]
    fn test_const_opcode_tokens() {
        let expr = Expr::Const(3.14);
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("const_f64"),
            "Const should generate const_f64, got: {}",
            code
        );
    }

    #[test]
    fn test_runtime_const_opcode_tokens() {
        let expr = Expr::RuntimeConst("self.target".to_string());
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("const_f64") && code.contains("self"),
            "RuntimeConst should generate const_f64(self.target), got: {}",
            code
        );
    }

    #[test]
    fn test_add_opcode_tokens() {
        let a = Expr::var("0".to_string(), 0);
        let b = Expr::var("1".to_string(), 1);
        let expr = Expr::Add(Box::new(a), Box::new(b));
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("add"),
            "Add should generate .add(), got: {}",
            code
        );
    }

    #[test]
    fn test_ln_opcode_tokens() {
        let x = Expr::var("0".to_string(), 0);
        let expr = Expr::Ln(Box::new(x));
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("ln"),
            "Ln should generate .ln(), got: {}",
            code
        );
    }

    #[test]
    fn test_pow_opcode_tokens() {
        let base = Expr::var("0".to_string(), 0);
        let expr = Expr::Pow(Box::new(base), 3.0);
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("pow") && code.contains("const_f64"),
            "Pow should generate const_f64 for exponent then .pow(), got: {}",
            code
        );
    }

    #[test]
    fn test_asin_opcode_tokens() {
        let x = Expr::var("0".to_string(), 0);
        let expr = Expr::Asin(Box::new(x));
        let emitter = quote::format_ident!("e");
        let tokens = expr.to_opcode_tokens(&emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("asin"),
            "Asin should generate .asin(), got: {}",
            code
        );
    }

    #[test]
    fn test_hessian_opcode_method_quadratic() {
        // f(x0, x1) = x0^2 + 3*x1^2
        // H = [[2, 0], [0, 6]]
        // Lower triangle: (0,0)=2, (1,1)=6
        let x0 = Expr::var("0".to_string(), 0);
        let x1 = Expr::var("1".to_string(), 1);
        let expr = Expr::Add(
            Box::new(Expr::Pow(Box::new(x0), 2.0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Const(3.0)),
                Box::new(Expr::Pow(Box::new(x1), 2.0)),
            )),
        );
        let vars = vec![
            VarRef { id: 0, index_tokens: "0".to_string() },
            VarRef { id: 1, index_tokens: "1".to_string() },
        ];
        let emitter = quote::format_ident!("e");
        let tokens = generate_hessian_opcode_method(&expr, &vars, &emitter);
        let code = tokens.to_string();
        assert!(
            code.contains("store_hessian"),
            "Hessian method should contain store_hessian calls, got: {}",
            code
        );
    }
}
