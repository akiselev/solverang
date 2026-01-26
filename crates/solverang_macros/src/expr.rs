//! Expression AST for symbolic differentiation.
//!
//! This module defines a mathematical expression tree that supports automatic
//! differentiation. The expression tree can represent arithmetic operations,
//! trigonometric functions, and variable references.

use proc_macro2::TokenStream;
use quote::quote;

/// A mathematical expression that can be symbolically differentiated.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A variable reference by index (e.g., x[0], x[1]).
    Var(VarRef),
    /// A constant numeric value.
    Const(f64),
    /// A runtime constant (e.g., self.target).
    /// This is treated as a constant for differentiation but generates code that
    /// evaluates the expression at runtime.
    RuntimeConst(String),
    /// Negation: -e.
    Neg(Box<Expr>),
    /// Addition: a + b.
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: a - b.
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: a * b.
    Mul(Box<Expr>, Box<Expr>),
    /// Division: a / b.
    Div(Box<Expr>, Box<Expr>),
    /// Square root: sqrt(e).
    Sqrt(Box<Expr>),
    /// Sine: sin(e).
    Sin(Box<Expr>),
    /// Cosine: cos(e).
    Cos(Box<Expr>),
    /// Tangent: tan(e).
    Tan(Box<Expr>),
    /// Arc tangent of y/x: atan2(y, x).
    Atan2(Box<Expr>, Box<Expr>),
    /// Power: base^exponent (where exponent is a constant).
    Pow(Box<Expr>, f64),
    /// Absolute value: |e|.
    Abs(Box<Expr>),
}

/// A reference to a variable in the state vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VarRef {
    /// The variable index expression as tokens.
    pub index_tokens: String,
    /// Unique identifier for this variable reference within the expression.
    pub id: usize,
}

impl VarRef {
    /// Create a new variable reference.
    pub fn new(index_tokens: String, id: usize) -> Self {
        Self { index_tokens, id }
    }
}

impl Expr {
    /// Create a variable reference.
    pub fn var(index_tokens: String, id: usize) -> Self {
        Expr::Var(VarRef::new(index_tokens, id))
    }

    /// Check if this expression is a constant zero.
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Const(v) if *v == 0.0)
    }

    /// Symbolically differentiate this expression with respect to the variable at the given id.
    pub fn differentiate(&self, var_id: usize) -> Expr {
        match self {
            Expr::Var(vref) => {
                if vref.id == var_id {
                    Expr::Const(1.0)
                } else {
                    Expr::Const(0.0)
                }
            }
            Expr::Const(_) => Expr::Const(0.0),
            Expr::RuntimeConst(_) => Expr::Const(0.0), // Runtime constants are constant wrt variables
            Expr::Neg(e) => Expr::Neg(Box::new(e.differentiate(var_id))),
            Expr::Add(a, b) => {
                // d(a + b) = da + db
                let da = a.differentiate(var_id);
                let db = b.differentiate(var_id);
                Expr::Add(Box::new(da), Box::new(db))
            }
            Expr::Sub(a, b) => {
                // d(a - b) = da - db
                let da = a.differentiate(var_id);
                let db = b.differentiate(var_id);
                Expr::Sub(Box::new(da), Box::new(db))
            }
            Expr::Mul(a, b) => {
                // d(a * b) = a * db + da * b (product rule)
                let da = a.differentiate(var_id);
                let db = b.differentiate(var_id);
                Expr::Add(
                    Box::new(Expr::Mul(a.clone(), Box::new(db))),
                    Box::new(Expr::Mul(Box::new(da), b.clone())),
                )
            }
            Expr::Div(a, b) => {
                // d(a / b) = (da * b - a * db) / b^2 (quotient rule)
                let da = a.differentiate(var_id);
                let db = b.differentiate(var_id);
                Expr::Div(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Mul(Box::new(da), b.clone())),
                        Box::new(Expr::Mul(a.clone(), Box::new(db))),
                    )),
                    Box::new(Expr::Pow(b.clone(), 2.0)),
                )
            }
            Expr::Sqrt(e) => {
                // d(sqrt(e)) = de / (2 * sqrt(e)) (chain rule)
                let de = e.differentiate(var_id);
                Expr::Div(
                    Box::new(de),
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(2.0)),
                        Box::new(Expr::Sqrt(e.clone())),
                    )),
                )
            }
            Expr::Sin(e) => {
                // d(sin(e)) = cos(e) * de (chain rule)
                let de = e.differentiate(var_id);
                Expr::Mul(Box::new(Expr::Cos(e.clone())), Box::new(de))
            }
            Expr::Cos(e) => {
                // d(cos(e)) = -sin(e) * de (chain rule)
                let de = e.differentiate(var_id);
                Expr::Neg(Box::new(Expr::Mul(Box::new(Expr::Sin(e.clone())), Box::new(de))))
            }
            Expr::Tan(e) => {
                // d(tan(e)) = de / cos^2(e) = de * (1 + tan^2(e)) (chain rule)
                let de = e.differentiate(var_id);
                Expr::Div(
                    Box::new(de),
                    Box::new(Expr::Pow(Box::new(Expr::Cos(e.clone())), 2.0)),
                )
            }
            Expr::Atan2(y, x) => {
                // d(atan2(y, x)) = (x * dy - y * dx) / (x^2 + y^2)
                let dy = y.differentiate(var_id);
                let dx = x.differentiate(var_id);
                Expr::Div(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Mul(x.clone(), Box::new(dy))),
                        Box::new(Expr::Mul(y.clone(), Box::new(dx))),
                    )),
                    Box::new(Expr::Add(
                        Box::new(Expr::Pow(x.clone(), 2.0)),
                        Box::new(Expr::Pow(y.clone(), 2.0)),
                    )),
                )
            }
            Expr::Pow(base, exp) => {
                // d(base^exp) = exp * base^(exp-1) * d(base)
                let dbase = base.differentiate(var_id);
                Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(*exp)),
                        Box::new(Expr::Pow(base.clone(), exp - 1.0)),
                    )),
                    Box::new(dbase),
                )
            }
            Expr::Abs(e) => {
                // d(|e|) = sign(e) * de
                // sign(e) = e / |e|
                let de = e.differentiate(var_id);
                Expr::Mul(
                    Box::new(Expr::Div(e.clone(), Box::new(Expr::Abs(e.clone())))),
                    Box::new(de),
                )
            }
        }
    }

    /// Simplify the expression by applying algebraic identities.
    pub fn simplify(self) -> Expr {
        match self {
            Expr::Neg(e) => {
                let e = e.simplify();
                match e {
                    Expr::Const(v) => Expr::Const(-v),
                    Expr::Neg(inner) => *inner,
                    other => Expr::Neg(Box::new(other)),
                }
            }
            Expr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(0.0), _) => b,
                    (_, Expr::Const(0.0)) => a,
                    (Expr::Const(va), Expr::Const(vb)) => Expr::Const(va + vb),
                    _ => Expr::Add(Box::new(a), Box::new(b)),
                }
            }
            Expr::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (_, Expr::Const(0.0)) => a,
                    (Expr::Const(0.0), _) => Expr::Neg(Box::new(b)).simplify(),
                    (Expr::Const(va), Expr::Const(vb)) => Expr::Const(va - vb),
                    _ => Expr::Sub(Box::new(a), Box::new(b)),
                }
            }
            Expr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                    (Expr::Const(1.0), _) => b,
                    (_, Expr::Const(1.0)) => a,
                    (Expr::Const(-1.0), _) => Expr::Neg(Box::new(b)).simplify(),
                    (_, Expr::Const(-1.0)) => Expr::Neg(Box::new(a)).simplify(),
                    (Expr::Const(va), Expr::Const(vb)) => Expr::Const(va * vb),
                    _ => Expr::Mul(Box::new(a), Box::new(b)),
                }
            }
            Expr::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(0.0), _) => Expr::Const(0.0),
                    (_, Expr::Const(1.0)) => a,
                    (Expr::Const(va), Expr::Const(vb)) if *vb != 0.0 => Expr::Const(va / vb),
                    _ => Expr::Div(Box::new(a), Box::new(b)),
                }
            }
            Expr::Pow(base, exp) => {
                let base = base.simplify();
                match (&base, exp) {
                    (_, 0.0) => Expr::Const(1.0),
                    (_, 1.0) => base,
                    (Expr::Const(v), _) => Expr::Const(v.powf(exp)),
                    _ => Expr::Pow(Box::new(base), exp),
                }
            }
            Expr::Sqrt(e) => {
                let e = e.simplify();
                match &e {
                    Expr::Const(v) if *v >= 0.0 => Expr::Const(v.sqrt()),
                    _ => Expr::Sqrt(Box::new(e)),
                }
            }
            Expr::Sin(e) => Expr::Sin(Box::new(e.simplify())),
            Expr::Cos(e) => Expr::Cos(Box::new(e.simplify())),
            Expr::Tan(e) => Expr::Tan(Box::new(e.simplify())),
            Expr::Atan2(y, x) => Expr::Atan2(Box::new(y.simplify()), Box::new(x.simplify())),
            Expr::Abs(e) => Expr::Abs(Box::new(e.simplify())),
            Expr::RuntimeConst(_) => self,
            other => other,
        }
    }

    /// Collect all variable references in this expression.
    #[allow(dead_code)]
    pub fn collect_variables(&self) -> Vec<VarRef> {
        let mut vars = Vec::new();
        self.collect_variables_into(&mut vars);
        vars
    }

    fn collect_variables_into(&self, vars: &mut Vec<VarRef>) {
        match self {
            Expr::Var(vref) => {
                if !vars.iter().any(|v| v.id == vref.id) {
                    vars.push(vref.clone());
                }
            }
            Expr::Const(_) | Expr::RuntimeConst(_) => {}
            Expr::Neg(e) | Expr::Sqrt(e) | Expr::Sin(e) | Expr::Cos(e) | Expr::Tan(e) | Expr::Abs(e) => {
                e.collect_variables_into(vars);
            }
            Expr::Pow(base, _) => {
                base.collect_variables_into(vars);
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Atan2(a, b) => {
                a.collect_variables_into(vars);
                b.collect_variables_into(vars);
            }
        }
    }

    /// Generate Rust code for this expression.
    #[allow(dead_code)]
    pub fn to_tokens(&self) -> TokenStream {
        match self {
            Expr::Var(vref) => {
                let index: TokenStream = vref.index_tokens.parse().expect("valid tokens");
                quote! { x[#index] }
            }
            Expr::Const(v) => {
                if v.is_nan() {
                    quote! { f64::NAN }
                } else if v.is_infinite() {
                    if *v > 0.0 {
                        quote! { f64::INFINITY }
                    } else {
                        quote! { f64::NEG_INFINITY }
                    }
                } else {
                    quote! { #v }
                }
            }
            Expr::RuntimeConst(tokens) => {
                let tokens: TokenStream = tokens.parse().expect("valid runtime const tokens");
                quote! { (#tokens) }
            }
            Expr::Neg(e) => {
                let e = e.to_tokens();
                quote! { (-(#e)) }
            }
            Expr::Add(a, b) => {
                let a = a.to_tokens();
                let b = b.to_tokens();
                quote! { ((#a) + (#b)) }
            }
            Expr::Sub(a, b) => {
                let a = a.to_tokens();
                let b = b.to_tokens();
                quote! { ((#a) - (#b)) }
            }
            Expr::Mul(a, b) => {
                let a = a.to_tokens();
                let b = b.to_tokens();
                quote! { ((#a) * (#b)) }
            }
            Expr::Div(a, b) => {
                let a = a.to_tokens();
                let b = b.to_tokens();
                quote! { ((#a) / (#b)) }
            }
            Expr::Sqrt(e) => {
                let e = e.to_tokens();
                quote! { (#e).sqrt() }
            }
            Expr::Sin(e) => {
                let e = e.to_tokens();
                quote! { (#e).sin() }
            }
            Expr::Cos(e) => {
                let e = e.to_tokens();
                quote! { (#e).cos() }
            }
            Expr::Tan(e) => {
                let e = e.to_tokens();
                quote! { (#e).tan() }
            }
            Expr::Atan2(y, x) => {
                let y = y.to_tokens();
                let x = x.to_tokens();
                quote! { (#y).atan2(#x) }
            }
            Expr::Pow(base, exp) => {
                let base = base.to_tokens();
                let exp_lit = *exp;
                // Use special cases for common exponents
                if *exp == 2.0 {
                    quote! { { let _b = #base; _b * _b } }
                } else if *exp == 0.5 {
                    quote! { (#base).sqrt() }
                } else if *exp == -1.0 {
                    quote! { (1.0 / (#base)) }
                } else {
                    quote! { (#base).powf(#exp_lit) }
                }
            }
            Expr::Abs(e) => {
                let e = e.to_tokens();
                quote! { (#e).abs() }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiate_constant() {
        let expr = Expr::Const(5.0);
        let deriv = expr.differentiate(0);
        assert!(matches!(deriv, Expr::Const(0.0)));
    }

    #[test]
    fn test_differentiate_variable() {
        let expr = Expr::var("0".to_string(), 0);
        let deriv = expr.differentiate(0);
        assert!(matches!(deriv.simplify(), Expr::Const(1.0)));
    }

    #[test]
    fn test_differentiate_different_variable() {
        let expr = Expr::var("0".to_string(), 0);
        let deriv = expr.differentiate(1);
        assert!(matches!(deriv.simplify(), Expr::Const(0.0)));
    }

    #[test]
    fn test_differentiate_add() {
        // d(x + 3) / dx = 1
        let expr = Expr::Add(
            Box::new(Expr::var("0".to_string(), 0)),
            Box::new(Expr::Const(3.0)),
        );
        let deriv = expr.differentiate(0).simplify();
        assert!(matches!(deriv, Expr::Const(1.0)));
    }

    #[test]
    fn test_differentiate_mul() {
        // d(x * x) / dx = 2x
        let x = Expr::var("0".to_string(), 0);
        let expr = Expr::Mul(Box::new(x.clone()), Box::new(x));
        let deriv = expr.differentiate(0).simplify();
        // The result should simplify to something equivalent to 2*x
        // After simplification: x*1 + 1*x = x + x
        match deriv {
            Expr::Add(a, b) => {
                assert!(matches!(*a, Expr::Var(_)));
                assert!(matches!(*b, Expr::Var(_)));
            }
            _ => panic!("Expected Add expression, got {:?}", deriv),
        }
    }

    #[test]
    fn test_simplify_zero_add() {
        let expr = Expr::Add(
            Box::new(Expr::Const(0.0)),
            Box::new(Expr::var("0".to_string(), 0)),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, Expr::Var(_)));
    }

    #[test]
    fn test_simplify_zero_mul() {
        let expr = Expr::Mul(
            Box::new(Expr::Const(0.0)),
            Box::new(Expr::var("0".to_string(), 0)),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, Expr::Const(0.0)));
    }

    #[test]
    fn test_simplify_one_mul() {
        let expr = Expr::Mul(
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::var("0".to_string(), 0)),
        );
        let simplified = expr.simplify();
        assert!(matches!(simplified, Expr::Var(_)));
    }
}
