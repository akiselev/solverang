//! Expression parsing from Rust syntax to our AST.
//!
//! This module parses Rust expressions into our symbolic expression tree
//! for differentiation.

use crate::expr::{Expr, VarRef};
use quote::ToTokens;
use syn::{
    punctuated::Punctuated, spanned::Spanned, BinOp, Expr as SynExpr, ExprBinary, ExprCall,
    ExprIndex, ExprLit, ExprMethodCall, ExprParen, ExprPath, ExprUnary, Lit, Token, UnOp,
};

/// State for parsing expressions, tracks variable assignments.
#[derive(Default)]
pub struct ParseContext {
    /// Counter for unique variable IDs.
    next_var_id: usize,
    /// Map from variable token strings to their IDs.
    var_map: Vec<(String, usize)>,
    /// The name of the array parameter (e.g., "x" in fn residual(&self, x: &[f64])).
    pub array_param: String,
}

impl ParseContext {
    /// Create a new parse context.
    pub fn new(array_param: &str) -> Self {
        Self {
            next_var_id: 0,
            var_map: Vec::new(),
            array_param: array_param.to_string(),
        }
    }

    /// Get or create a variable ID for the given index expression.
    pub fn get_or_create_var_id(&mut self, index_tokens: &str) -> usize {
        for (tokens, id) in &self.var_map {
            if tokens == index_tokens {
                return *id;
            }
        }
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.var_map.push((index_tokens.to_string(), id));
        id
    }

    /// Get all collected variables.
    pub fn variables(&self) -> Vec<VarRef> {
        self.var_map
            .iter()
            .map(|(tokens, id)| VarRef::new(tokens.clone(), *id))
            .collect()
    }
}

/// Result of parsing an expression.
pub type ParseResult = Result<Expr, syn::Error>;

/// Parse a Rust expression into our symbolic expression tree.
pub fn parse_expr(expr: &SynExpr, ctx: &mut ParseContext) -> ParseResult {
    match expr {
        SynExpr::Paren(ExprParen { expr, .. }) => parse_expr(expr, ctx),

        SynExpr::Lit(ExprLit { lit, .. }) => parse_literal(lit),

        SynExpr::Path(ExprPath { path, .. }) => {
            // Could be a constant or a reference to self.field
            if let Some(ident) = path.get_ident() {
                let name = ident.to_string();
                // Check for known constants
                match name.as_str() {
                    "PI" => Ok(Expr::Const(std::f64::consts::PI)),
                    "E" => Ok(Expr::Const(std::f64::consts::E)),
                    "SQRT_2" => Ok(Expr::Const(std::f64::consts::SQRT_2)),
                    _ => Err(syn::Error::new(
                        expr.span(),
                        format!("Unsupported identifier: {}. Only x[index] array accesses and numeric constants are supported.", name),
                    )),
                }
            } else {
                // Could be std::f64::consts::PI etc.
                let path_str = quote::quote!(#path).to_string();
                if path_str.contains("PI") {
                    Ok(Expr::Const(std::f64::consts::PI))
                } else if path_str.contains("E") && !path_str.contains("EPSILON") {
                    Ok(Expr::Const(std::f64::consts::E))
                } else {
                    Err(syn::Error::new(
                        expr.span(),
                        format!("Unsupported path expression: {}", path_str),
                    ))
                }
            }
        }

        SynExpr::Index(ExprIndex {
            expr: base, index, ..
        }) => {
            // Check if this is accessing the array parameter (e.g., x[i])
            if let SynExpr::Path(ExprPath { path, .. }) = base.as_ref() {
                if let Some(ident) = path.get_ident() {
                    if ident == &ctx.array_param {
                        let index_tokens = index.to_token_stream().to_string();
                        let var_id = ctx.get_or_create_var_id(&index_tokens);
                        return Ok(Expr::var(index_tokens, var_id));
                    }
                }
            }
            Err(syn::Error::new(
                expr.span(),
                "Only array indexing on the state vector parameter is supported",
            ))
        }

        SynExpr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let left_expr = parse_expr(left, ctx)?;
            let right_expr = parse_expr(right, ctx)?;
            parse_binary_op(*op, left_expr, right_expr, expr.span())
        }

        SynExpr::Unary(ExprUnary {
            op, expr: inner, ..
        }) => {
            let inner_expr = parse_expr(inner, ctx)?;
            match op {
                UnOp::Neg(_) => Ok(Expr::Neg(Box::new(inner_expr))),
                UnOp::Not(_) => Err(syn::Error::new(expr.span(), "Logical not is not supported")),
                UnOp::Deref(_) => Err(syn::Error::new(expr.span(), "Dereference is not supported")),
                _ => Err(syn::Error::new(expr.span(), "Unsupported unary operator")),
            }
        }

        SynExpr::MethodCall(ExprMethodCall {
            receiver,
            method,
            args,
            ..
        }) => parse_method_call(receiver, method, args, ctx),

        SynExpr::Call(ExprCall { func, args, .. }) => parse_function_call(func, args, ctx),

        SynExpr::Field(field_expr) => {
            // Handle self.field access - treat as a runtime constant from the constraint.
            // These are constant with respect to the variables in x[], so derivatives
            // will be zero. The generated code will evaluate them at runtime.
            let field_tokens = field_expr.to_token_stream().to_string();
            Ok(Expr::RuntimeConst(field_tokens))
        }

        _ => Err(syn::Error::new(
            expr.span(),
            format!("Unsupported expression type: {:?}", expr),
        )),
    }
}

fn parse_literal(lit: &Lit) -> ParseResult {
    match lit {
        Lit::Float(f) => {
            let value: f64 = f.base10_parse().map_err(|e| syn::Error::new(f.span(), e))?;
            Ok(Expr::Const(value))
        }
        Lit::Int(i) => {
            let value: i64 = i.base10_parse().map_err(|e| syn::Error::new(i.span(), e))?;
            Ok(Expr::Const(value as f64))
        }
        _ => Err(syn::Error::new(
            lit.span(),
            "Only numeric literals are supported",
        )),
    }
}

fn parse_binary_op(op: BinOp, left: Expr, right: Expr, span: proc_macro2::Span) -> ParseResult {
    match op {
        BinOp::Add(_) => Ok(Expr::Add(Box::new(left), Box::new(right))),
        BinOp::Sub(_) => Ok(Expr::Sub(Box::new(left), Box::new(right))),
        BinOp::Mul(_) => Ok(Expr::Mul(Box::new(left), Box::new(right))),
        BinOp::Div(_) => Ok(Expr::Div(Box::new(left), Box::new(right))),
        _ => Err(syn::Error::new(
            span,
            format!("Unsupported binary operator: {:?}", op),
        )),
    }
}

fn parse_method_call(
    receiver: &SynExpr,
    method: &syn::Ident,
    args: &Punctuated<SynExpr, Token![,]>,
    ctx: &mut ParseContext,
) -> ParseResult {
    let method_name = method.to_string();
    let receiver_expr = parse_expr(receiver, ctx)?;

    match method_name.as_str() {
        "sqrt" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "sqrt() takes no arguments"));
            }
            Ok(Expr::Sqrt(Box::new(receiver_expr)))
        }
        "sin" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "sin() takes no arguments"));
            }
            Ok(Expr::Sin(Box::new(receiver_expr)))
        }
        "cos" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "cos() takes no arguments"));
            }
            Ok(Expr::Cos(Box::new(receiver_expr)))
        }
        "tan" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "tan() takes no arguments"));
            }
            Ok(Expr::Tan(Box::new(receiver_expr)))
        }
        "abs" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "abs() takes no arguments"));
            }
            Ok(Expr::Abs(Box::new(receiver_expr)))
        }
        "ln" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "ln() takes no arguments"));
            }
            Ok(Expr::Ln(Box::new(receiver_expr)))
        }
        "exp" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "exp() takes no arguments"));
            }
            Ok(Expr::Exp(Box::new(receiver_expr)))
        }
        "asin" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "asin() takes no arguments"));
            }
            Ok(Expr::Asin(Box::new(receiver_expr)))
        }
        "acos" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "acos() takes no arguments"));
            }
            Ok(Expr::Acos(Box::new(receiver_expr)))
        }
        "sinh" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "sinh() takes no arguments"));
            }
            Ok(Expr::Sinh(Box::new(receiver_expr)))
        }
        "cosh" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "cosh() takes no arguments"));
            }
            Ok(Expr::Cosh(Box::new(receiver_expr)))
        }
        "tanh" => {
            if !args.is_empty() {
                return Err(syn::Error::new(method.span(), "tanh() takes no arguments"));
            }
            Ok(Expr::Tanh(Box::new(receiver_expr)))
        }
        "powi" => {
            if args.len() != 1 {
                return Err(syn::Error::new(
                    method.span(),
                    "powi() takes exactly one argument",
                ));
            }
            let exp = parse_constant_expr(&args[0])?;
            Ok(Expr::Pow(Box::new(receiver_expr), exp))
        }
        "powf" => {
            if args.len() != 1 {
                return Err(syn::Error::new(
                    method.span(),
                    "powf() takes exactly one argument",
                ));
            }
            let exp = parse_constant_expr(&args[0])?;
            Ok(Expr::Pow(Box::new(receiver_expr), exp))
        }
        "atan2" => {
            if args.len() != 1 {
                return Err(syn::Error::new(
                    method.span(),
                    "atan2() takes exactly one argument",
                ));
            }
            let x = parse_expr(&args[0], ctx)?;
            Ok(Expr::Atan2(Box::new(receiver_expr), Box::new(x)))
        }
        _ => Err(syn::Error::new(
            method.span(),
            format!("Unsupported method: {}", method_name),
        )),
    }
}

fn parse_function_call(
    func: &SynExpr,
    args: &Punctuated<SynExpr, Token![,]>,
    ctx: &mut ParseContext,
) -> ParseResult {
    // Handle f64::sqrt(x), f64::sin(x), etc.
    if let SynExpr::Path(ExprPath { path, .. }) = func {
        let path_str = path.to_token_stream().to_string();

        // Check for f64:: prefix functions
        if path_str.starts_with("f64 ::") || path_str.starts_with("f64::") {
            let func_name = path.segments.last().map(|s| s.ident.to_string());
            match func_name.as_deref() {
                Some("sqrt") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Sqrt(Box::new(arg)));
                }
                Some("sin") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Sin(Box::new(arg)));
                }
                Some("cos") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Cos(Box::new(arg)));
                }
                Some("tan") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Tan(Box::new(arg)));
                }
                Some("atan2") if args.len() == 2 => {
                    let y = parse_expr(&args[0], ctx)?;
                    let x = parse_expr(&args[1], ctx)?;
                    return Ok(Expr::Atan2(Box::new(y), Box::new(x)));
                }
                Some("abs") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Abs(Box::new(arg)));
                }
                Some("ln") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Ln(Box::new(arg)));
                }
                Some("exp") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Exp(Box::new(arg)));
                }
                Some("asin") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Asin(Box::new(arg)));
                }
                Some("acos") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Acos(Box::new(arg)));
                }
                Some("sinh") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Sinh(Box::new(arg)));
                }
                Some("cosh") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Cosh(Box::new(arg)));
                }
                Some("tanh") if args.len() == 1 => {
                    let arg = parse_expr(&args[0], ctx)?;
                    return Ok(Expr::Tanh(Box::new(arg)));
                }
                _ => {}
            }
        }
    }

    Err(syn::Error::new(
        func.span(),
        "Unsupported function call. Use method syntax (e.g., x.sqrt()) or f64::sqrt(x)",
    ))
}

/// Parse a constant expression (must evaluate to a compile-time constant).
fn parse_constant_expr(expr: &SynExpr) -> Result<f64, syn::Error> {
    match expr {
        SynExpr::Lit(ExprLit {
            lit: Lit::Float(f), ..
        }) => f.base10_parse().map_err(|e| syn::Error::new(f.span(), e)),
        SynExpr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => {
            let value: i64 = i.base10_parse().map_err(|e| syn::Error::new(i.span(), e))?;
            Ok(value as f64)
        }
        SynExpr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let value = parse_constant_expr(expr)?;
            Ok(-value)
        }
        _ => Err(syn::Error::new(
            expr.span(),
            "Power exponent must be a numeric constant",
        )),
    }
}

/// Information about a parsed residual expression.
pub struct ParsedResidual {
    /// The parsed expression.
    pub expr: Expr,
    /// All variables referenced in the expression.
    pub variables: Vec<VarRef>,
}

/// Parse a residual expression and collect variable information.
pub fn parse_residual(expr: &SynExpr, array_param: &str) -> Result<ParsedResidual, syn::Error> {
    let mut ctx = ParseContext::new(array_param);
    let parsed_expr = parse_expr(expr, &mut ctx)?;
    let variables = ctx.variables();

    Ok(ParsedResidual {
        expr: parsed_expr,
        variables,
    })
}
