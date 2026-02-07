//! Fixed-parameter substitution.
//!
//! When parameters are marked as fixed in the [`ParamStore`], their values are
//! known constants. This module identifies constraints that become trivially
//! satisfied once all their parameters are fixed, and reports which parameters
//! were eliminated so that higher-level code can skip them during solving.

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::ParamStore;

/// Result of fixed-parameter substitution analysis.
#[derive(Clone, Debug)]
pub struct SubstitutionResult {
    /// Parameters that were identified as fixed and can be substituted out.
    pub eliminated_params: Vec<ParamId>,
    /// Total number of parameters removed from the solve.
    pub params_removed: usize,
    /// Indices of constraints that are trivially satisfied (all params fixed,
    /// residual near zero).
    pub trivially_satisfied: Vec<usize>,
    /// Indices of constraints that are trivially violated (all params fixed,
    /// residual NOT near zero).
    pub trivially_violated: Vec<usize>,
}

/// Default tolerance used when checking if a residual is "near zero".
const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Analyze which parameters are fixed and can be substituted out.
///
/// For each constraint, checks whether ALL of its parameters are fixed.
/// If so, the constraint is either trivially satisfied (residual near zero)
/// or trivially violated (residual far from zero). The caller can use this
/// information to skip satisfied constraints and report violated ones as
/// errors.
///
/// Returns a [`SubstitutionResult`] describing what was found.
pub fn analyze_substitutions(
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> SubstitutionResult {
    let mut eliminated_params: Vec<ParamId> = Vec::new();
    let mut trivially_satisfied: Vec<usize> = Vec::new();
    let mut trivially_violated: Vec<usize> = Vec::new();

    // Collect all fixed params that appear in any constraint.
    let mut seen_fixed = std::collections::HashSet::new();

    for (idx, c) in constraints.iter().enumerate() {
        let params = c.param_ids();
        let all_fixed = params.iter().all(|&p| store.is_fixed(p));

        // Track which fixed params we encounter.
        for &p in params {
            if store.is_fixed(p) && seen_fixed.insert(p) {
                eliminated_params.push(p);
            }
        }

        if all_fixed {
            // All params are fixed -- evaluate the residual to see if the
            // constraint is satisfied.
            let residuals = c.residuals(store);
            let satisfied = residuals.iter().all(|r| r.abs() < DEFAULT_TOLERANCE);
            if satisfied {
                trivially_satisfied.push(idx);
            } else {
                trivially_violated.push(idx);
            }
        }
    }

    let params_removed = eliminated_params.len();

    SubstitutionResult {
        eliminated_params,
        params_removed,
        trivially_satisfied,
        trivially_violated,
    }
}

/// Check if a constraint is trivially satisfied given the current fixed params.
///
/// Returns `true` if **all** parameters the constraint depends on are fixed
/// and every residual component is within `tolerance` of zero.
pub fn is_trivially_satisfied(
    constraint: &dyn Constraint,
    store: &ParamStore,
    tolerance: f64,
) -> bool {
    let all_fixed = constraint.param_ids().iter().all(|&p| store.is_fixed(p));
    if !all_fixed {
        return false;
    }
    let residuals = constraint.residuals(store);
    residuals.iter().all(|r| r.abs() < tolerance)
}

/// Check if a constraint is trivially violated given the current fixed params.
///
/// Returns `true` if **all** parameters the constraint depends on are fixed
/// and at least one residual component exceeds `tolerance`.
pub fn is_trivially_violated(
    constraint: &dyn Constraint,
    store: &ParamStore,
    tolerance: f64,
) -> bool {
    let all_fixed = constraint.param_ids().iter().all(|&p| store.is_fixed(p));
    if !all_fixed {
        return false;
    }
    let residuals = constraint.residuals(store);
    residuals.iter().any(|r| r.abs() >= tolerance)
}

/// Count how many free (non-fixed) parameters a constraint depends on.
pub fn free_param_count(constraint: &dyn Constraint, store: &ParamStore) -> usize {
    constraint
        .param_ids()
        .iter()
        .filter(|&&p| !store.is_fixed(p))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    /// A simple equality constraint: param_a - target = 0.
    struct FixValueConstraint {
        id: ConstraintId,
        param: ParamId,
        target: f64,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "fix_value"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 1.0)]
        }
    }

    fn dummy_owner() -> EntityId {
        EntityId::new(0, 0)
    }

    #[test]
    fn test_trivially_satisfied_when_all_fixed_and_residual_zero() {
        let mut store = ParamStore::new();
        let p = store.alloc(5.0, dummy_owner());
        store.fix(p);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 5.0, // residual = 5.0 - 5.0 = 0
        };

        assert!(is_trivially_satisfied(&c, &store, 1e-10));
        assert!(!is_trivially_violated(&c, &store, 1e-10));
    }

    #[test]
    fn test_trivially_violated_when_all_fixed_and_residual_nonzero() {
        let mut store = ParamStore::new();
        let p = store.alloc(5.0, dummy_owner());
        store.fix(p);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 10.0, // residual = 5.0 - 10.0 = -5.0
        };

        assert!(!is_trivially_satisfied(&c, &store, 1e-10));
        assert!(is_trivially_violated(&c, &store, 1e-10));
    }

    #[test]
    fn test_not_trivially_satisfied_when_free_params() {
        let mut store = ParamStore::new();
        let p = store.alloc(5.0, dummy_owner());
        // p is free, not fixed

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 5.0,
        };

        assert!(!is_trivially_satisfied(&c, &store, 1e-10));
    }

    #[test]
    fn test_analyze_substitutions() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(3.0, owner);
        let p2 = store.alloc(7.0, owner);

        store.fix(p1);
        store.fix(p2);

        let c1 = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 3.0, // satisfied
        };
        let c2 = FixValueConstraint {
            id: ConstraintId::new(1, 0),
            param: p2,
            target: 99.0, // violated
        };

        let constraints: Vec<&dyn Constraint> = vec![&c1, &c2];
        let result = analyze_substitutions(&constraints, &store);

        assert_eq!(result.eliminated_params.len(), 2);
        assert_eq!(result.params_removed, 2);
        assert_eq!(result.trivially_satisfied, vec![0]);
        assert_eq!(result.trivially_violated, vec![1]);
    }

    #[test]
    fn test_analyze_substitutions_mixed_fixed_free() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p_fixed = store.alloc(3.0, owner);
        let p_free = store.alloc(7.0, owner);

        store.fix(p_fixed);

        let c1 = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p_fixed,
            target: 3.0,
        };
        let c2 = FixValueConstraint {
            id: ConstraintId::new(1, 0),
            param: p_free,
            target: 7.0,
        };

        let constraints: Vec<&dyn Constraint> = vec![&c1, &c2];
        let result = analyze_substitutions(&constraints, &store);

        // Only p_fixed is eliminated.
        assert_eq!(result.eliminated_params.len(), 1);
        assert_eq!(result.eliminated_params[0], p_fixed);
        // Only c1 is trivially satisfied; c2 has a free param.
        assert_eq!(result.trivially_satisfied, vec![0]);
        assert!(result.trivially_violated.is_empty());
    }

    #[test]
    fn test_free_param_count() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p = store.alloc(1.0, owner);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 1.0,
        };

        assert_eq!(free_param_count(&c, &store), 1);

        store.fix(p);
        assert_eq!(free_param_count(&c, &store), 0);
    }
}
