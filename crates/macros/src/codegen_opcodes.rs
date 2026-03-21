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
}
