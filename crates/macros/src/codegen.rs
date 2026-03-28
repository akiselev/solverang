//! Code generation for Jacobian implementations.
//!
//! This module generates the `jacobian()` method implementation from
//! symbolically differentiated expressions.

use crate::expr::{Expr, VarRef};
use proc_macro2::TokenStream;
use quote::quote;

/// Generate Jacobian entries for a single residual.
///
/// Returns a list of (variable_index_tokens, derivative_tokens) pairs.
pub fn generate_jacobian_entries(
    residual_expr: &Expr,
    variables: &[VarRef],
) -> Vec<(String, TokenStream)> {
    let mut entries = Vec::new();

    for var in variables {
        let derivative = residual_expr.differentiate(var.id).simplify();

        // Skip zero derivatives
        if derivative.is_zero() {
            continue;
        }

        let derivative_tokens = derivative.to_tokens();
        entries.push((var.index_tokens.clone(), derivative_tokens));
    }

    entries
}

/// Information needed to generate a complete jacobian implementation.
pub struct JacobianInfo {
    /// The row index for this residual in the Jacobian matrix.
    pub residual_row: usize,
    /// Entries: (column_index_expr, derivative_expr).
    pub entries: Vec<(String, TokenStream)>,
}

/// Generate the complete jacobian method body.
pub fn generate_jacobian_method(jacobians: &[JacobianInfo]) -> TokenStream {
    let mut all_entries = Vec::new();

    for jac_info in jacobians {
        let row = jac_info.residual_row;

        for (col_expr, deriv_tokens) in &jac_info.entries {
            let col: TokenStream = col_expr.parse().expect("valid column expression");
            all_entries.push(quote! {
                (#row, #col, #deriv_tokens)
            });
        }
    }

    let capacity = all_entries.len();

    quote! {
        let mut entries = Vec::with_capacity(#capacity);
        #(
            entries.push(#all_entries);
        )*
        entries
    }
}

/// Generate code that adds Jacobian entries to a mutable Vec.
///
/// This variant is useful when the caller wants to control the container.
#[allow(dead_code)]
pub fn generate_jacobian_append(
    jacobians: &[JacobianInfo],
    entries_var: &syn::Ident,
) -> TokenStream {
    let mut pushes = Vec::new();

    for jac_info in jacobians {
        let row = jac_info.residual_row;

        for (col_expr, deriv_tokens) in &jac_info.entries {
            let col: TokenStream = col_expr.parse().expect("valid column expression");
            pushes.push(quote! {
                #entries_var.push((#row, #col, #deriv_tokens));
            });
        }
    }

    quote! {
        #(#pushes)*
    }
}

/// Generate Hessian entries for a scalar expression.
/// Returns (var_i_index, var_j_index, second_derivative_tokens) for lower triangle (i >= j).
pub fn generate_hessian_entries(
    expr: &Expr,
    variables: &[VarRef],
) -> Vec<(String, String, TokenStream)> {
    let mut entries = Vec::new();

    for (j_idx, var_j) in variables.iter().enumerate() {
        let first_deriv = expr.differentiate(var_j.id).simplify();

        if first_deriv.is_zero() {
            continue;
        }

        for (i_idx, var_i) in variables.iter().enumerate() {
            if i_idx < j_idx {
                continue;
            }

            let second_deriv = first_deriv.differentiate(var_i.id).simplify();

            if second_deriv.is_zero() {
                continue;
            }

            let deriv_tokens = second_deriv.to_tokens();
            entries.push((
                var_i.index_tokens.clone(),
                var_j.index_tokens.clone(),
                deriv_tokens,
            ));
        }
    }

    entries
}

/// Generate the hessian_entries method body.
pub fn generate_hessian_method(entries: &[(String, String, TokenStream)]) -> TokenStream {
    let capacity = entries.len();
    let mut entry_stmts = Vec::new();

    for (row_expr, col_expr, deriv_tokens) in entries {
        let row: TokenStream = row_expr.parse().expect("valid row expression");
        let col: TokenStream = col_expr.parse().expect("valid col expression");
        entry_stmts.push(quote! {
            (#row, #col, #deriv_tokens)
        });
    }

    quote! {
        let mut entries = Vec::with_capacity(#capacity);
        #(
            {
                let val = #entry_stmts;
                if val.2.abs() > 1e-30 {
                    entries.push(val);
                }
            }
        )*
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_simple_derivative() {
        // f(x) = x^2, df/dx = 2x
        let x = Expr::var("0".to_string(), 0);
        let expr = Expr::Mul(Box::new(x.clone()), Box::new(x.clone()));
        let variables = vec![VarRef::new("0".to_string(), 0)];

        let entries = generate_jacobian_entries(&expr, &variables);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "0");
        // The derivative tokens should represent 2x (or x + x)
    }

    #[test]
    fn test_skip_zero_derivative() {
        // f(x, y) = x + 3, df/dy = 0
        let x = Expr::var("0".to_string(), 0);
        let expr = Expr::Add(Box::new(x), Box::new(Expr::Const(3.0)));
        let variables = vec![
            VarRef::new("0".to_string(), 0),
            VarRef::new("1".to_string(), 1),
        ];

        let entries = generate_jacobian_entries(&expr, &variables);
        assert_eq!(entries.len(), 1); // Only x, not y
        assert_eq!(entries[0].0, "0");
    }
}
