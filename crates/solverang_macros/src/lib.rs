//! Procedural macros for automatic Jacobian generation via symbolic differentiation.
//!
//! This crate provides the `#[auto_jacobian]` attribute macro which automatically
//! generates Jacobian matrix entries from residual expressions. This eliminates
//! the need to manually implement derivatives, reducing errors and maintenance burden.
//!
//! # Usage via solverang
//!
//! Enable the `macros` feature in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! solverang = { version = "0.1", features = ["macros"] }
//! ```
//!
//! Then use the macros via the re-export:
//!
//! ```ignore
//! use solverang::{auto_jacobian, Problem};
//!
//! struct DistanceConstraint {
//!     target: f64,
//! }
//!
//! #[auto_jacobian(array_param = "x")]
//! impl DistanceConstraint {
//!     #[residual]
//!     fn residual(&self, x: &[f64]) -> f64 {
//!         let dx = x[2] - x[0];
//!         let dy = x[3] - x[1];
//!         (dx * dx + dy * dy).sqrt() - self.target
//!     }
//! }
//!
//! // Now implement Problem using the generated jacobian_entries method:
//! impl Problem for DistanceConstraint {
//!     fn name(&self) -> &str { "Distance" }
//!     fn residual_count(&self) -> usize { 1 }
//!     fn variable_count(&self) -> usize { 4 }
//!
//!     fn residuals(&self, x: &[f64]) -> Vec<f64> {
//!         vec![self.residual(x)]
//!     }
//!
//!     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
//!         self.jacobian_entries(x)  // Use the generated method
//!     }
//!
//!     fn initial_point(&self, _: f64) -> Vec<f64> {
//!         vec![0.0, 0.0, 1.0, 0.0]
//!     }
//! }
//! ```
//!
//! # Features
//!
//! - **Symbolic Differentiation**: Automatically computes partial derivatives
//! - **Algebraic Simplification**: Reduces generated expressions where possible
//! - **Chain Rule Support**: Handles compound expressions correctly
//! - **Common Functions**: Supports sqrt, sin, cos, tan, atan2, abs, pow
//! - **Runtime Constants**: Struct fields like `self.target` are treated as constants
//!
//! # Supported Operations
//!
//! - **Arithmetic**: `+`, `-`, `*`, `/`
//! - **Unary**: `-` (negation)
//! - **Methods**: `.sqrt()`, `.sin()`, `.cos()`, `.tan()`, `.abs()`, `.powi()`, `.powf()`, `.atan2()`
//! - **Functions**: `f64::sqrt()`, `f64::sin()`, `f64::cos()`, `f64::tan()`, `f64::atan2()`, `f64::abs()`
//! - **Struct Fields**: `self.field` is treated as a runtime constant (derivative = 0)
//!
//! # Limitations
//!
//! - Control flow (if/else, loops) is not supported
//! - Method calls other than the listed math functions are not supported
//! - The residual function must end with a single expression (no `return` statement)

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::{
    parse::Parse, parse::ParseStream, parse_macro_input, spanned::Spanned, Block, Expr, FnArg,
    Ident, ImplItem, ItemImpl, Lit, Pat, ReturnType, Signature, Token, Type,
};

mod codegen;
mod expr;
mod parse;

/// Attribute macro that generates a `jacobian` method from a `residual` method.
///
/// Apply this to an impl block containing a method marked with `#[residual]`.
/// The macro will parse the residual expression and generate the corresponding
/// Jacobian computation using symbolic differentiation.
///
/// # Attributes
///
/// - `#[jacobian(array_param = "x")]` - Name of the state vector parameter (default: "x")
///
/// # Example
///
/// ```ignore
/// #[auto_jacobian(array_param = "x")]
/// impl MyConstraint {
///     #[residual]
///     fn residual(&self, x: &[f64]) -> f64 {
///         x[0] * x[0] + x[1] * x[1] - 1.0  // Unit circle constraint
///     }
/// }
///
/// // Generates:
/// // fn jacobian_entries(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
/// //     vec![
/// //         (0, 0, 2.0 * x[0]),  // d/dx[0]
/// //         (0, 1, 2.0 * x[1]),  // d/dx[1]
/// //     ]
/// // }
/// ```
#[proc_macro_attribute]
pub fn auto_jacobian(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as JacobianArgs);
    let impl_block = parse_macro_input!(item as ItemImpl);

    match generate_jacobian_impl(args, impl_block) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Arguments for the `#[auto_jacobian]` attribute.
struct JacobianArgs {
    /// Name of the array parameter in the residual function.
    array_param: String,
}

impl Default for JacobianArgs {
    fn default() -> Self {
        Self {
            array_param: "x".to_string(),
        }
    }
}

impl Parse for JacobianArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = JacobianArgs::default();

        if input.is_empty() {
            return Ok(args);
        }

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match ident.to_string().as_str() {
                "array_param" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Str(s) = lit {
                        args.array_param = s.value();
                    } else {
                        return Err(syn::Error::new(lit.span(), "Expected string literal"));
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("Unknown argument: {}", other),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(args)
    }
}

/// Find the residual method in an impl block.
fn find_residual_method(impl_block: &ItemImpl) -> Option<(&ImplItem, &Signature, &Block)> {
    for item in &impl_block.items {
        if let ImplItem::Fn(method) = item {
            for attr in &method.attrs {
                if attr.path().is_ident("residual") {
                    return Some((item, &method.sig, &method.block));
                }
            }
        }
    }
    None
}

/// Extract the return expression from a function block.
///
/// Handles simple cases:
/// - `{ expr }` - single expression
/// - `{ let x = ...; expr }` - let bindings followed by expression
fn extract_return_expr(block: &Block) -> syn::Result<&Expr> {
    // Check for expression without semicolon at the end
    if let Some(syn::Stmt::Expr(expr, None)) = block.stmts.last() {
        return Ok(expr);
    }

    Err(syn::Error::new(
        block.span(),
        "Residual function must end with an expression (no semicolon). \
         Complex control flow is not supported.",
    ))
}

/// Expand let bindings in an expression.
///
/// This is a simplified version that handles basic let bindings by
/// extracting variable definitions from the block.
fn collect_let_bindings(block: &Block) -> Vec<(&Ident, &Expr)> {
    let mut bindings = Vec::new();

    for stmt in &block.stmts {
        if let syn::Stmt::Local(local) = stmt {
            if let Pat::Ident(pat_ident) = &local.pat {
                if let Some(init) = &local.init {
                    bindings.push((&pat_ident.ident, init.expr.as_ref()));
                }
            }
        }
    }

    bindings
}

/// Check if an expression references a let-bound variable.
fn references_binding(expr: &Expr, ident: &Ident) -> bool {
    match expr {
        Expr::Path(path) => {
            if let Some(id) = path.path.get_ident() {
                return id == ident;
            }
            false
        }
        Expr::Binary(bin) => {
            references_binding(&bin.left, ident) || references_binding(&bin.right, ident)
        }
        Expr::Unary(un) => references_binding(&un.expr, ident),
        Expr::Paren(paren) => references_binding(&paren.expr, ident),
        Expr::MethodCall(call) => {
            references_binding(&call.receiver, ident)
                || call.args.iter().any(|a| references_binding(a, ident))
        }
        Expr::Call(call) => call.args.iter().any(|a| references_binding(a, ident)),
        Expr::Index(idx) => {
            references_binding(&idx.expr, ident) || references_binding(&idx.index, ident)
        }
        _ => false,
    }
}

/// Substitute a variable reference with its definition.
fn substitute_binding(expr: &Expr, ident: &Ident, replacement: &Expr) -> Expr {
    match expr {
        Expr::Path(path) => {
            if let Some(id) = path.path.get_ident() {
                if id == ident {
                    return replacement.clone();
                }
            }
            expr.clone()
        }
        Expr::Binary(bin) => {
            let left = substitute_binding(&bin.left, ident, replacement);
            let right = substitute_binding(&bin.right, ident, replacement);
            Expr::Binary(syn::ExprBinary {
                attrs: bin.attrs.clone(),
                left: Box::new(left),
                op: bin.op.clone(),
                right: Box::new(right),
            })
        }
        Expr::Unary(un) => {
            let inner = substitute_binding(&un.expr, ident, replacement);
            Expr::Unary(syn::ExprUnary {
                attrs: un.attrs.clone(),
                op: un.op.clone(),
                expr: Box::new(inner),
            })
        }
        Expr::Paren(paren) => {
            let inner = substitute_binding(&paren.expr, ident, replacement);
            Expr::Paren(syn::ExprParen {
                attrs: paren.attrs.clone(),
                paren_token: paren.paren_token,
                expr: Box::new(inner),
            })
        }
        Expr::MethodCall(call) => {
            let receiver = substitute_binding(&call.receiver, ident, replacement);
            let args = call
                .args
                .iter()
                .map(|a| substitute_binding(a, ident, replacement))
                .collect();
            Expr::MethodCall(syn::ExprMethodCall {
                attrs: call.attrs.clone(),
                receiver: Box::new(receiver),
                dot_token: call.dot_token,
                method: call.method.clone(),
                turbofish: call.turbofish.clone(),
                paren_token: call.paren_token,
                args,
            })
        }
        Expr::Call(call) => {
            let args = call
                .args
                .iter()
                .map(|a| substitute_binding(a, ident, replacement))
                .collect();
            Expr::Call(syn::ExprCall {
                attrs: call.attrs.clone(),
                func: call.func.clone(),
                paren_token: call.paren_token,
                args,
            })
        }
        Expr::Index(idx) => {
            let base = substitute_binding(&idx.expr, ident, replacement);
            let index = substitute_binding(&idx.index, ident, replacement);
            Expr::Index(syn::ExprIndex {
                attrs: idx.attrs.clone(),
                expr: Box::new(base),
                bracket_token: idx.bracket_token,
                index: Box::new(index),
            })
        }
        _ => expr.clone(),
    }
}

/// Expand all let bindings in the return expression.
fn expand_bindings(return_expr: &Expr, bindings: &[(&Ident, &Expr)]) -> Expr {
    let mut result = return_expr.clone();

    // Process bindings in reverse order to handle dependencies
    for (ident, definition) in bindings.iter().rev() {
        if references_binding(&result, ident) {
            // First expand any bindings in the definition itself
            let expanded_def = expand_bindings(definition, bindings);
            result = substitute_binding(&result, ident, &expanded_def);
        }
    }

    result
}

/// Generate the jacobian implementation.
fn generate_jacobian_impl(
    args: JacobianArgs,
    mut impl_block: ItemImpl,
) -> syn::Result<TokenStream2> {
    // Find the residual method
    let (_, sig, block) = find_residual_method(&impl_block).ok_or_else(|| {
        syn::Error::new(
            impl_block.self_ty.span(),
            "No method marked with #[residual] found in impl block",
        )
    })?;

    // Validate the signature
    validate_residual_signature(sig, &args.array_param)?;

    // Get let bindings and return expression
    let bindings = collect_let_bindings(block);
    let return_expr = extract_return_expr(block)?;

    // Expand all let bindings into the return expression
    let expanded_expr = expand_bindings(return_expr, &bindings);

    // Parse the expression into our AST
    let parsed = parse::parse_residual(&expanded_expr, &args.array_param)?;

    // Generate Jacobian entries
    let entries = codegen::generate_jacobian_entries(&parsed.expr, &parsed.variables);

    // Build the jacobian info
    let jac_info = codegen::JacobianInfo {
        residual_row: 0,
        entries,
    };

    // Generate the jacobian method body
    let jacobian_body = codegen::generate_jacobian_method(&[jac_info]);

    // Get the array parameter name as an identifier
    let array_param_ident: Ident = syn::parse_str(&args.array_param)
        .map_err(|e| syn::Error::new(sig.span(), format!("Invalid array_param: {}", e)))?;

    // Create the jacobian method
    // Named `jacobian_entries` to avoid conflicts with trait methods
    let jacobian_method: ImplItem = syn::parse_quote! {
        /// Compute Jacobian entries as sparse triplets (row, column, value).
        ///
        /// This method was automatically generated by symbolic differentiation
        /// of the residual expression. Use this in your `Problem::jacobian` implementation.
        fn jacobian_entries(&self, #array_param_ident: &[f64]) -> Vec<(usize, usize, f64)> {
            #jacobian_body
        }
    };

    // Remove the #[residual] attribute from the residual method
    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item {
            method
                .attrs
                .retain(|attr| !attr.path().is_ident("residual"));
        }
    }

    // Add the jacobian method to the impl block
    impl_block.items.push(jacobian_method);

    Ok(impl_block.into_token_stream())
}

/// Validate that the residual method has the expected signature.
fn validate_residual_signature(sig: &Signature, array_param: &str) -> syn::Result<()> {
    // Check that it takes &self and &[f64]
    let mut has_self = false;
    let mut has_array = false;

    for arg in &sig.inputs {
        match arg {
            FnArg::Receiver(_) => has_self = true,
            FnArg::Typed(pat_type) => {
                if let Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                    if pat_ident.ident == array_param {
                        has_array = true;
                        // Verify it's &[f64]
                        if let Type::Reference(type_ref) = pat_type.ty.as_ref() {
                            if let Type::Slice(slice) = type_ref.elem.as_ref() {
                                let elem_str = slice.elem.to_token_stream().to_string();
                                if elem_str != "f64" {
                                    return Err(syn::Error::new(
                                        slice.elem.span(),
                                        format!(
                                            "Array parameter must be &[f64], found &[{}]",
                                            elem_str
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if !has_self {
        return Err(syn::Error::new(
            sig.span(),
            "Residual method must take &self",
        ));
    }

    if !has_array {
        return Err(syn::Error::new(
            sig.span(),
            format!(
                "Residual method must have parameter '{}' of type &[f64]",
                array_param
            ),
        ));
    }

    // Check return type
    match &sig.output {
        ReturnType::Type(_, ty) => {
            let ty_str = ty.to_token_stream().to_string();
            if ty_str != "f64" {
                return Err(syn::Error::new(ty.span(), "Residual must return f64"));
            }
        }
        ReturnType::Default => {
            return Err(syn::Error::new(
                sig.span(),
                "Residual method must return f64",
            ));
        }
    }

    Ok(())
}

/// Marker attribute for residual methods.
///
/// Place this on a method within an `#[auto_jacobian]` impl block to mark it
/// as the residual function to differentiate.
#[proc_macro_attribute]
pub fn residual(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This attribute is a marker - the actual processing happens in auto_jacobian
    item
}
