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
mod codegen_opcodes;
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

/// Find all residual methods in an impl block.
fn find_residual_methods(impl_block: &ItemImpl) -> Vec<(&ImplItem, &Signature, &Block)> {
    let mut methods = Vec::new();
    for item in &impl_block.items {
        if let ImplItem::Fn(method) = item {
            for attr in &method.attrs {
                if attr.path().is_ident("residual") {
                    methods.push((item as &ImplItem, &method.sig, &method.block));
                    break;
                }
            }
        }
    }
    methods
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
    // Find all residual methods — extract data and drop borrows before mutating impl_block
    struct ResidualInfo {
        parsed: parse::ParsedResidual,
        jac_info: codegen::JacobianInfo,
        method_name: Ident,
    }

    let mut residual_infos = Vec::new();
    let mut first_sig_span = None;

    {
        let residual_methods = find_residual_methods(&impl_block);
        if residual_methods.is_empty() {
            return Err(syn::Error::new(
                impl_block.self_ty.span(),
                "No method marked with #[residual] found in impl block",
            ));
        }

        for (row_index, (_, sig, block)) in residual_methods.iter().enumerate() {
            // Validate the signature (all must share the same array_param)
            validate_residual_signature(sig, &args.array_param)?;
            if first_sig_span.is_none() {
                first_sig_span = Some(sig.span());
            }

            let method_name = sig.ident.clone();

            // Get let bindings and return expression
            let bindings = collect_let_bindings(block);
            let return_expr = extract_return_expr(block)?;

            // Expand all let bindings into the return expression
            let expanded_expr = expand_bindings(return_expr, &bindings);

            // Parse the expression into our AST
            let parsed = parse::parse_residual(&expanded_expr, &args.array_param)?;

            // Generate Jacobian entries for this residual
            let entries = codegen::generate_jacobian_entries(&parsed.expr, &parsed.variables);

            let jac_info = codegen::JacobianInfo {
                residual_row: row_index,
                entries,
            };

            residual_infos.push(ResidualInfo { parsed, jac_info, method_name });
        }
    } // residual_methods borrows dropped here

    let num_residuals = residual_infos.len();

    // Generate the jacobian method body from ALL residuals.
    // We need to move the JacobianInfos out since generate_jacobian_method takes &[JacobianInfo].
    let all_jac_infos: Vec<codegen::JacobianInfo> =
        residual_infos.iter().map(|ri| {
            codegen::JacobianInfo {
                residual_row: ri.jac_info.residual_row,
                entries: ri.jac_info.entries.clone(),
            }
        }).collect();
    let jacobian_body = codegen::generate_jacobian_method(&all_jac_infos);

    // Get the array parameter name as an identifier
    let sig_span = first_sig_span.unwrap();
    let array_param_ident: Ident = syn::parse_str(&args.array_param)
        .map_err(|e| syn::Error::new(sig_span, format!("Invalid array_param: {}", e)))?;

    // Create the jacobian method
    let jacobian_method: ImplItem = syn::parse_quote! {
        /// Compute Jacobian entries as sparse triplets (row, column, value).
        ///
        /// This method was automatically generated by symbolic differentiation
        /// of the residual expressions. Use this in your `Problem::jacobian` implementation.
        fn jacobian_entries(&self, #array_param_ident: &[f64]) -> Vec<(usize, usize, f64)> {
            #jacobian_body
        }
    };

    // Remove the #[residual] attribute from all residual methods
    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item {
            method
                .attrs
                .retain(|attr| !attr.path().is_ident("residual"));
        }
    }

    // Add the jacobian method to the impl block
    impl_block.items.push(jacobian_method);

    // ====================================================================
    // Generate JIT lowering methods (behind #[cfg(feature = "jit")])
    // ====================================================================

    // Determine the number of variables (max index + 1) across ALL residuals.
    let n_vars: usize = residual_infos
        .iter()
        .flat_map(|ri| ri.parsed.variables.iter())
        .filter_map(|v| v.index_tokens.parse::<usize>().ok())
        .max()
        .map(|max_idx| max_idx + 1)
        .unwrap_or(0);

    // Generate opcode-emitting code for ALL residuals.
    let emitter_ident: Ident = syn::parse_str("__emitter").unwrap();

    // Build residual opcode body: emit opcodes for each residual with its row index
    let mut residual_opcode_stmts = Vec::new();
    let mut jacobian_opcode_stmts = Vec::new();

    for (row_index, ri) in residual_infos.iter().enumerate() {
        let row_idx = row_index as u32;
        let row_idx_ident: Ident =
            syn::parse_str(&format!("__residual_idx_{}", row_index)).unwrap();

        let res_body = codegen_opcodes::generate_residual_opcode_method(
            &ri.parsed.expr,
            &emitter_ident,
            &row_idx_ident,
        );
        residual_opcode_stmts.push(quote::quote! {
            let #row_idx_ident: u32 = #row_idx;
            #res_body
        });

        let jac_body = codegen_opcodes::generate_jacobian_opcode_method(
            &ri.parsed.expr,
            &ri.parsed.variables,
            &emitter_ident,
            &row_idx_ident,
        );
        jacobian_opcode_stmts.push(quote::quote! {
            let #row_idx_ident: u32 = #row_idx;
            #jac_body
        });
    }

    let n_vars_lit = n_vars;
    let num_residuals_lit = num_residuals;

    // Method 1: lower_residual_ops — emits opcodes for ALL residuals
    let lower_residual_method: ImplItem = syn::parse_quote! {
        /// Emit JIT opcodes for residual computation.
        ///
        /// Auto-generated by `#[auto_jacobian]`. Calls `OpcodeEmitter` methods
        /// to build an opcode stream that computes all residuals.
        #[cfg(feature = "jit")]
        #[allow(unused_variables)]
        fn lower_residual_ops(
            &self,
            __emitter: &mut ::solverang::__jit_reexports::OpcodeEmitter,
            __residual_idx: u32,
        ) {
            #(#residual_opcode_stmts)*
        }
    };

    // Method 2: lower_jacobian_ops — emits Jacobian opcodes for ALL residuals
    let lower_jacobian_method: ImplItem = syn::parse_quote! {
        /// Emit JIT opcodes for Jacobian computation.
        ///
        /// Auto-generated by `#[auto_jacobian]`. Emits opcodes for each non-zero
        /// partial derivative of all residuals.
        #[cfg(feature = "jit")]
        #[allow(unused_variables)]
        fn lower_jacobian_ops(
            &self,
            __emitter: &mut ::solverang::__jit_reexports::OpcodeEmitter,
            __residual_idx: u32,
        ) {
            #(#jacobian_opcode_stmts)*
        }
    };

    // Method 3: lower_to_compiled_constraints
    let lower_compiled_method: ImplItem = syn::parse_quote! {
        /// Build a `CompiledConstraints` from this problem instance, ready for JIT compilation.
        ///
        /// Auto-generated by `#[auto_jacobian]`. Struct field values (e.g. `self.target`)
        /// are baked into the opcode stream as `LoadConst` instructions.
        #[cfg(feature = "jit")]
        fn lower_to_compiled_constraints(&self) -> ::solverang::__jit_reexports::CompiledConstraints {
            let mut __res_emitter = ::solverang::__jit_reexports::OpcodeEmitter::new();
            self.lower_residual_ops(&mut __res_emitter, 0);
            let __res_max = __res_emitter.max_register();

            let mut __jac_emitter = ::solverang::__jit_reexports::OpcodeEmitter::new();
            self.lower_jacobian_ops(&mut __jac_emitter, 0);
            let __jac_max = __jac_emitter.max_register();

            let mut __cc = ::solverang::__jit_reexports::CompiledConstraints::new(#n_vars_lit, #num_residuals_lit);
            __cc.residual_ops = __res_emitter.into_ops();
            let __jac_ops = __jac_emitter.ops().to_vec();
            __cc.jacobian_ops = __jac_ops;
            __cc.jacobian_pattern = __jac_emitter.take_jacobian_entries();
            __cc.jacobian_nnz = __cc.jacobian_pattern.len();
            __cc.max_register = if __res_max > __jac_max { __res_max } else { __jac_max };
            __cc
        }
    };

    impl_block.items.push(lower_residual_method);
    impl_block.items.push(lower_jacobian_method);
    impl_block.items.push(lower_compiled_method);

    // Generate residuals_all helper if there are multiple residuals
    if num_residuals > 1 {
        let residual_method_names: Vec<&Ident> = residual_infos
            .iter()
            .map(|ri| &ri.method_name)
            .collect();

        let residuals_all_method: ImplItem = syn::parse_quote! {
            /// Evaluate all residuals and return them as a Vec.
            ///
            /// Auto-generated by `#[auto_jacobian]`. Calls each `#[residual]` method
            /// in order and collects the results.
            fn residuals_all(&self, #array_param_ident: &[f64]) -> Vec<f64> {
                vec![#(self.#residual_method_names(#array_param_ident)),*]
            }
        };
        impl_block.items.push(residuals_all_method);
    }

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

/// Marker attribute for objective methods.
///
/// Place this on a method within an `#[auto_diff]` impl block to mark it
/// as the objective function to differentiate. Generates a `gradient_entries`
/// method that returns `Vec<(usize, f64)>` (variable_index, partial_derivative).
///
/// # Example
///
/// ```ignore
/// #[auto_diff(array_param = "x")]
/// impl MyObjective {
///     #[objective]
///     fn value(&self, x: &[f64]) -> f64 {
///         (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.0) * (x[1] - 2.0)
///     }
/// }
/// // Generates: fn gradient_entries(&self, x: &[f64]) -> Vec<(usize, f64)>
/// ```
#[proc_macro_attribute]
pub fn objective(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This attribute is a marker - the actual processing happens in auto_diff
    item
}

/// Marker attribute for opt-in Hessian generation.
///
/// Place this alongside `#[objective]` on a method within an `#[auto_diff]` impl block
/// to also generate a `hessian_entries()` method via symbolic second differentiation.
///
/// # Example
///
/// ```ignore
/// #[auto_diff(array_param = "x")]
/// impl MyObjective {
///     #[objective]
///     #[hessian]
///     fn value(&self, x: &[f64]) -> f64 {
///         x[0] * x[0] + x[1] * x[1]
///     }
/// }
/// // Generates gradient_entries() AND hessian_entries() returning lower-triangle triplets.
/// ```
#[proc_macro_attribute]
pub fn hessian(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Superset of `#[auto_jacobian]` that also recognizes `#[objective]` and
/// `#[inequality]` attributes for optimization support.
///
/// - `#[residual]` methods → generates `jacobian_entries()` (same as `#[auto_jacobian]`)
/// - `#[objective]` methods → generates `gradient_entries()` (sparse gradient)
///
/// # Example
///
/// ```ignore
/// #[auto_diff(array_param = "x")]
/// impl MyObjective {
///     #[objective]
///     fn value(&self, x: &[f64]) -> f64 {
///         100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1.0 - x[0]) * (1.0 - x[0])
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn auto_diff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as JacobianArgs);
    let impl_block = parse_macro_input!(item as ItemImpl);

    match generate_auto_diff_impl(args, impl_block) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Find all objective methods in an impl block.
fn find_objective_methods(impl_block: &ItemImpl) -> Vec<(&ImplItem, &Signature, &Block)> {
    let mut methods = Vec::new();
    for item in &impl_block.items {
        if let ImplItem::Fn(method) = item {
            for attr in &method.attrs {
                if attr.path().is_ident("objective") {
                    methods.push((item as &ImplItem, &method.sig, &method.block));
                    break;
                }
            }
        }
    }
    methods
}

/// Check if any method in the impl block has the `#[hessian]` attribute.
fn has_hessian_marker(impl_block: &ItemImpl) -> bool {
    impl_block.items.iter().any(|item| {
        if let ImplItem::Fn(method) = item {
            method.attrs.iter().any(|attr| attr.path().is_ident("hessian"))
        } else {
            false
        }
    })
}

/// Generate the auto_diff implementation (handles both #[residual] and #[objective]).
fn generate_auto_diff_impl(
    args: JacobianArgs,
    mut impl_block: ItemImpl,
) -> syn::Result<TokenStream2> {
    let has_residuals = !find_residual_methods(&impl_block).is_empty();
    let has_objectives = !find_objective_methods(&impl_block).is_empty();

    if !has_residuals && !has_objectives {
        return Err(syn::Error::new(
            impl_block.self_ty.span(),
            "No method marked with #[residual] or #[objective] found in impl block",
        ));
    }

    // If there are residual methods, delegate to the existing jacobian generator
    // (which handles residuals, JIT codegen, etc.)
    if has_residuals {
        if has_hessian_marker(&impl_block) {
            return Err(syn::Error::new(
                impl_block.self_ty.span(),
                "#[hessian] is only valid on #[objective] methods, not #[residual] methods",
            ));
        }
        // For now, delegate to the existing implementation
        return generate_jacobian_impl(args, impl_block);
    }

    // Handle #[objective] methods — generate gradient
    let objective_methods = find_objective_methods(&impl_block);
    if objective_methods.len() > 1 {
        return Err(syn::Error::new(
            impl_block.self_ty.span(),
            "Only one #[objective] method is allowed per impl block",
        ));
    }

    let (_, sig, block) = objective_methods[0];
    validate_residual_signature(sig, &args.array_param)?;

    let bindings = collect_let_bindings(block);
    let return_expr = extract_return_expr(block)?;
    let expanded_expr = expand_bindings(return_expr, &bindings);
    let parsed = parse::parse_residual(&expanded_expr, &args.array_param)?;

    // Generate gradient entries (reuse jacobian entry generation — gradient is row-0 Jacobian)
    let gradient_entries = codegen::generate_jacobian_entries(&parsed.expr, &parsed.variables);

    // Build gradient method body
    let array_param_ident: Ident = syn::parse_str(&args.array_param)
        .map_err(|e| syn::Error::new(sig.span(), format!("Invalid array_param: {}", e)))?;

    let mut grad_stmts = Vec::new();
    for (col_expr, deriv_tokens) in &gradient_entries {
        let col: TokenStream2 = col_expr.parse().expect("valid column expression");
        grad_stmts.push(quote::quote! {
            (#col, #deriv_tokens)
        });
    }
    let capacity = grad_stmts.len();

    let gradient_method: ImplItem = syn::parse_quote! {
        /// Compute gradient entries as sparse pairs (variable_index, partial_derivative).
        ///
        /// Auto-generated by symbolic differentiation of the `#[objective]` expression.
        fn gradient_entries(&self, #array_param_ident: &[f64]) -> Vec<(usize, f64)> {
            let mut entries = Vec::with_capacity(#capacity);
            #(
                {
                    let val = #grad_stmts;
                    if val.1.abs() > 1e-30 {
                        entries.push(val);
                    }
                }
            )*
            entries
        }
    };

    // Check for #[hessian] marker before stripping attributes
    let want_hessian = has_hessian_marker(&impl_block);

    // Remove #[objective] and #[hessian] attributes from all methods
    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item {
            method.attrs.retain(|attr| !attr.path().is_ident("objective"));
            method.attrs.retain(|attr| !attr.path().is_ident("hessian"));
        }
    }

    impl_block.items.push(gradient_method);

    if want_hessian {
        let hessian_entries = codegen::generate_hessian_entries(&parsed.expr, &parsed.variables);
        let hessian_body = codegen::generate_hessian_method(&hessian_entries);

        let hessian_method: ImplItem = syn::parse_quote! {
            /// Compute Hessian entries as sparse triplets (row_var_index, col_var_index, value).
            /// Lower triangle only (row >= col).
            ///
            /// Auto-generated by symbolic second differentiation of the `#[objective]` expression.
            fn hessian_entries(&self, #array_param_ident: &[f64]) -> Vec<(usize, usize, f64)> {
                #hessian_body
            }
        };

        impl_block.items.push(hessian_method);
    }

    Ok(impl_block.into_token_stream())
}

/// Check if an expression contains Abs (non-differentiable at zero).
#[allow(dead_code)]
fn contains_abs(expr: &expr::Expr) -> bool {
    match expr {
        expr::Expr::Abs(_) => true,
        expr::Expr::Neg(e) | expr::Expr::Sqrt(e) | expr::Expr::Sin(e)
        | expr::Expr::Cos(e) | expr::Expr::Tan(e) | expr::Expr::Ln(e)
        | expr::Expr::Exp(e) | expr::Expr::Asin(e) | expr::Expr::Acos(e)
        | expr::Expr::Sinh(e) | expr::Expr::Cosh(e) | expr::Expr::Tanh(e) => contains_abs(e),
        expr::Expr::Add(a, b) | expr::Expr::Sub(a, b) | expr::Expr::Mul(a, b)
        | expr::Expr::Div(a, b) | expr::Expr::Atan2(a, b) => contains_abs(a) || contains_abs(b),
        expr::Expr::Pow(base, _) => contains_abs(base),
        expr::Expr::Var(_) | expr::Expr::Const(_) | expr::Expr::RuntimeConst(_) => false,
    }
}
