//! Optional symbolic export for high-level constraints.
//!
//! Solverang remains a batteries-included numerical/geometric solver and does not depend on
//! a particular CAS. Constraints may describe their residuals through the small object-safe
//! [`SymbolicSink`] protocol. A consumer such as Resolvent implements the sink and receives
//! opaque expression handles back. Constraints with no useful symbolic representation simply
//! use the default `None` implementation on [`super::Constraint::symbolic_residuals`].

use crate::id::{ConstraintId, ParamId};

/// Opaque handle allocated by a [`SymbolicSink`]. It has meaning only to the sink instance
/// that returned it.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SymbolicNode(pub u32);

/// Minimal, dependency-neutral mathematical construction vocabulary used by Solverang
/// constraints.
///
/// `constant_f64_exact` means the exact finite IEEE-754 value (normally lifted as a dyadic
/// rational), never a guessed "nice" rational. The sink owns canonicalization,
/// differentiation, exact coefficient domains and all richer algebra.
pub trait SymbolicSink {
    fn parameter(&mut self, parameter: ParamId) -> SymbolicNode;
    fn constant_f64_exact(&mut self, value: f64) -> Option<SymbolicNode>;

    fn add(&mut self, left: SymbolicNode, right: SymbolicNode) -> SymbolicNode;
    fn sub(&mut self, left: SymbolicNode, right: SymbolicNode) -> SymbolicNode;
    fn mul(&mut self, left: SymbolicNode, right: SymbolicNode) -> SymbolicNode;
    fn div(&mut self, numerator: SymbolicNode, denominator: SymbolicNode) -> SymbolicNode;
    fn pow_i(&mut self, base: SymbolicNode, exponent: i32) -> SymbolicNode;
    fn apply(&mut self, function: &str, args: &[SymbolicNode]) -> SymbolicNode;

    fn neg(&mut self, value: SymbolicNode) -> Option<SymbolicNode> {
        let minus_one = self.constant_f64_exact(-1.0)?;
        Some(self.mul(minus_one, value))
    }

    fn square(&mut self, value: SymbolicNode) -> SymbolicNode {
        self.pow_i(value, 2)
    }
}

/// Why a concrete constraint cannot provide an exact symbolic representation through the
/// current protocol. This is diagnostic metadata only; unsupported constraints continue down
/// the existing numeric path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicUnavailable {
    NotImplemented,
    UnsupportedOperation,
    NonFiniteConstant,
}

/// Exact/structural diagnostics supplied by an optional symbolic backend.
///
/// This augments rather than replaces Solverang's native diagnostics. A Resolvent-backed
/// analyzer can fill implication support from generic finite-field rank or attach checkable
/// algebraic conflict certificates while every user retains the standard numeric analysis.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiagnosticSupplement {
    pub redundant_implied_by: Vec<(ConstraintId, Vec<ConstraintId>)>,
    pub certificates: Vec<(String, String)>,
}

impl DiagnosticSupplement {
    pub fn implied_by(&self, id: ConstraintId) -> Option<&[ConstraintId]> {
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
            let node = SymbolicNode(self.next);
            self.next += 1;
            node
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
    fn sink_is_object_safe_and_exact_f64_policy_is_explicit() {
        fn consume(sink: &mut dyn SymbolicSink, p: ParamId) -> Option<SymbolicNode> {
            let x = sink.parameter(p);
            let two = sink.constant_f64_exact(2.0)?;
            let x2 = sink.square(x);
            Some(sink.sub(x2, two))
        }

        let mut sink = RecordingSink::default();
        let p = ParamId::new(2, 7);
        assert!(consume(&mut sink, p).is_some());
        assert_eq!(sink.params, vec![p]);
        assert!(sink.constant_f64_exact(f64::NAN).is_none());
    }
}
