//! Optional symbolic export seam for constraint systems.
//!
//! Solverang remains a batteries-included geometric/general constraint solver. This module
//! does not make Resolvent a required dependency and does not replace the numeric
//! [`super::Constraint`] contract. Instead, algebraic constraints may additionally expose
//! their residuals through a generic expression builder. A Resolvent adapter can implement
//! the builder without Solverang owning or cloning Resolvent's expression IR.

use crate::id::ParamId;

use super::Constraint;

/// Minimal construction vocabulary needed by Solverang's algebraic constraint residuals.
///
/// The trait is intentionally consumer-owned at the expression level: `Expr` may be a
/// Resolvent handle, a test AST, or another symbolic backend. `constant_f64_exact` must
/// preserve the exact IEEE-754 value (normally as a dyadic rational) or return `None`; it
/// must never guess a "nice" rational.
pub trait SymbolicExpressionBuilder {
    type Expr: Clone;

    fn parameter(&mut self, id: ParamId) -> Self::Expr;
    fn constant_f64_exact(&mut self, value: f64) -> Option<Self::Expr>;

    fn add(&mut self, lhs: Self::Expr, rhs: Self::Expr) -> Self::Expr;
    fn sub(&mut self, lhs: Self::Expr, rhs: Self::Expr) -> Self::Expr;
    fn mul(&mut self, lhs: Self::Expr, rhs: Self::Expr) -> Self::Expr;
    fn neg(&mut self, value: Self::Expr) -> Self::Expr;

    fn square(&mut self, value: Self::Expr) -> Self::Expr {
        self.mul(value.clone(), value)
    }
}

/// Optional capability implemented only by constraints whose residual semantics can be
/// exported without approximation or probing.
///
/// The generic builder makes this trait intentionally non-object-safe. Runtime collections
/// continue to use `dyn Constraint`; symbolic analysis is an explicit side path over known
/// concrete constraint types/adapters rather than a tax on every interactive solve.
pub trait SymbolicConstraint: Constraint {
    fn symbolic_residuals<B: SymbolicExpressionBuilder>(
        &self,
        builder: &mut B,
    ) -> Option<Vec<B::Expr>>;
}

/// Why a concrete constraint does not expose exact symbolic residuals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicUnavailable {
    /// No symbolic implementation has been written yet.
    NotImplemented,
    /// The constraint contains a genuinely non-algebraic or piecewise operation that the
    /// current exact export policy deliberately excludes.
    UnsupportedOperation,
    /// A stored floating-point constant was non-finite and therefore cannot be lifted to an
    /// exact dyadic rational.
    NonFiniteConstant,
}

/// Helper for adapters that want to advertise capability without downcasting by name.
pub trait SymbolicConstraintInfo: Constraint {
    fn symbolic_availability(&self) -> Result<(), SymbolicUnavailable> {
        Err(SymbolicUnavailable::NotImplemented)
    }
}
