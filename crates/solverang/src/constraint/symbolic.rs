//! Optional symbolic export for high-level constraints.
//!
//! Solverang remains a batteries-included numerical/geometric solver and does not depend on
//! a particular CAS. Instead, constraints may describe their residuals through the small
//! object-safe [`SymbolicSink`] protocol. A consumer such as Resolvent implements the sink
//! and receives its own expression handles back. Constraints that have no useful symbolic
//! representation simply return `None` from
//! [`Constraint::symbolic_residuals`](crate::constraint::Constraint::symbolic_residuals).

use crate::id::ParamId;

/// Opaque handle allocated by a [`SymbolicSink`]. The number has meaning only to the sink
/// instance that returned it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolicNode(pub u32);

/// Minimal mathematical vocabulary used by Solverang constraints.
///
/// It is deliberately smaller than a CAS API: the sink owns canonicalization,
/// differentiation, exact arithmetic and any richer analysis. `constant_f64_exact` means
/// the exact dyadic rational denoted by finite IEEE-754 bits, never a guessed "nice"
/// rational.
pub trait SymbolicSink {
    /// A symbolic parameter corresponding to a Solverang parameter id.
    fn parameter(&mut self, parameter: ParamId) -> SymbolicNode;

    /// A finite `f64` interpreted exactly by its IEEE-754 value. Returns `None` for NaN or
    /// infinity when the sink cannot represent them as mathematical constants.
    fn constant_f64_exact(&mut self, value: f64) -> Option<SymbolicNode>;

    /// Add two nodes.
    fn add(&mut self, left: SymbolicNode, right: SymbolicNode) -> SymbolicNode;

    /// Subtract two nodes.
    fn sub(&mut self, left: SymbolicNode, right: SymbolicNode) -> SymbolicNode;

    /// Multiply two nodes.
    fn mul(&mut self, left: SymbolicNode, right: SymbolicNode) -> SymbolicNode;

    /// Divide two nodes.
    fn div(&mut self, numerator: SymbolicNode, denominator: SymbolicNode) -> SymbolicNode;

    /// Integer power.
    fn pow_i(&mut self, base: SymbolicNode, exponent: i32) -> SymbolicNode;

    /// Apply a named mathematical function. This keeps Solverang's seam open to the few
    /// non-polynomial constraints without requiring Resolvent-specific function ids.
    fn apply(&mut self, function: &str, args: &[SymbolicNode]) -> SymbolicNode;

    /// Convenience: negate a node using multiplication by exact -1. Sinks may override it
    /// if they have a native negation node.
    fn neg(&mut self, value: SymbolicNode) -> Option<SymbolicNode> {
        let minus_one = self.constant_f64_exact(-1.0)?;
        Some(self.mul(minus_one, value))
    }
}

/// Exact/structural diagnostics supplied by an optional symbolic backend.
///
/// This is deliberately additive to Solverang's native diagnostics. A backend can fill
/// `implied_by` after generic finite-field rank or attach a checkable conflict witness,
/// while the default numerical diagnosis remains available to every user.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagnosticSupplement {
    /// `(redundant_constraint, constraints that imply it)`.
    pub redundant_implied_by: Vec<(crate::id::ConstraintId, Vec<crate::id::ConstraintId>)>,
    /// Human/checker-readable certificates indexed by a stable local label.
    pub certificates: Vec<(String, String)>,
}

impl DiagnosticSupplement {
    /// Return the exact implication support for a redundant constraint, if supplied.
    pub fn implied_by(&self, id: crate::id::ConstraintId) -> Option<&[crate::id::ConstraintId]> {
        self.redundant_implied_by
            .iter()
            .find_map(|(candidate, support)| (*candidate == id).then_some(support.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecordingSink {
        next: u32,
        params: Vec<ParamId>,
    }

    impl RecordingSink {
        fn node(&mut self) -> SymbolicNode {
            let n = SymbolicNode(self.next);
            self.next += 1;
            n
        }
    }

    impl SymbolicSink for RecordingSink {
        fn parameter(&mut self, parameter: ParamId) -> SymbolicNode {
            self.params.push(parameter);
            self.node()
        }
        fn constant_f64_exact(&mut self, value: f64) -> Option<SymbolicNode> {
            value.is_finite().then(|| self.node())
        }
        fn add(&mut self, _: SymbolicNode, _: SymbolicNode) -> SymbolicNode {
            self.node()
        }
        fn sub(&mut self, _: SymbolicNode, _: SymbolicNode) -> SymbolicNode {
            self.node()
        }
        fn mul(&mut self, _: SymbolicNode, _: SymbolicNode) -> SymbolicNode {
            self.node()
        }
        fn div(&mut self, _: SymbolicNode, _: SymbolicNode) -> SymbolicNode {
            self.node()
        }
        fn pow_i(&mut self, _: SymbolicNode, _: i32) -> SymbolicNode {
            self.node()
        }
        fn apply(&mut self, _: &str, _: &[SymbolicNode]) -> SymbolicNode {
            self.node()
        }
    }

    #[test]
    fn sink_is_cas_neutral_and_exact_f64_is_explicit() {
        let mut sink = RecordingSink::default();
        let p = ParamId::new(2, 7);
        let x = sink.parameter(p);
        let two = sink.constant_f64_exact(2.0).unwrap();
        let square = sink.pow_i(x, 2);
        let _square_minus_two = sink.sub(square, two);
        assert_eq!(sink.params, vec![p]);
        assert!(sink.constant_f64_exact(f64::NAN).is_none());
    }
}
