//! Trivial constraint elimination.
//!
//! When a constraint has exactly one equation and exactly one free (non-fixed)
//! parameter, the parameter value can be determined analytically from the
//! linearized equation:
//!
//! ```text
//! residual(current) + J * (x - current) = 0
//! x = current - residual / J
//! ```
//!
//! Once determined, the parameter is set to its computed value and marked as
//! fixed, and the constraint can be removed from the solve. This reduces both
//! the variable count and the equation count by one per elimination.

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::ParamStore;

/// A trivial elimination: a constraint directly determines a parameter value.
#[derive(Clone, Debug)]
pub struct TrivialElimination {
    /// The parameter whose value was determined.
    pub param: ParamId,
    /// The analytically determined value.
    pub determined_value: f64,
    /// Index of the constraint that was used for elimination (in the input
    /// slice provided to [`detect_trivial_eliminations`]).
    pub constraint_index: usize,
}

/// Minimum absolute Jacobian magnitude below which we refuse to divide
/// (avoids division by near-zero).
const MIN_JACOBIAN_MAGNITUDE: f64 = 1e-12;

/// Detect constraints that trivially determine a single parameter.
///
/// A constraint trivially determines a parameter if:
/// - It has exactly 1 equation.
/// - It depends on exactly 1 **free** (non-fixed) parameter.
/// - The Jacobian entry for that parameter is non-zero.
///
/// The determined value is computed from the linearization:
///
/// ```text
/// x_new = x_current - residual / jacobian_entry
/// ```
///
/// The caller provides `(original_index, constraint_ref)` pairs so that
/// `constraint_index` in the returned [`TrivialElimination`] refers to the
/// original numbering.
pub fn detect_trivial_eliminations(
    constraints: &[(usize, &dyn Constraint)],
    store: &ParamStore,
) -> Vec<TrivialElimination> {
    let mut eliminations = Vec::new();

    for &(idx, c) in constraints {
        // Must be a single-equation constraint.
        if c.equation_count() != 1 {
            continue;
        }

        // Find free parameters among the constraint's dependencies.
        let params = c.param_ids();
        let free_params: Vec<ParamId> = params
            .iter()
            .copied()
            .filter(|&p| !store.is_fixed(p))
            .collect();

        if free_params.len() != 1 {
            continue;
        }

        let free_param = free_params[0];

        // Get the residual and Jacobian.
        let residuals = c.residuals(store);
        if residuals.is_empty() {
            continue;
        }
        let residual = residuals[0];

        let jac = c.jacobian(store);

        // Find the Jacobian entry for our free param in row 0.
        let jac_entry = jac
            .iter()
            .find(|&&(row, pid, _)| row == 0 && pid == free_param)
            .map(|&(_, _, val)| val);

        let jac_value = match jac_entry {
            Some(v) if v.abs() > MIN_JACOBIAN_MAGNITUDE => v,
            _ => continue, // Zero or missing Jacobian -- cannot solve.
        };

        // Linearized solve: x_new = x_current - residual / J.
        let current = store.get(free_param);
        let determined_value = current - residual / jac_value;

        // Sanity check: the determined value should be finite.
        if !determined_value.is_finite() {
            continue;
        }

        eliminations.push(TrivialElimination {
            param: free_param,
            determined_value,
            constraint_index: idx,
        });
    }

    eliminations
}

/// Apply trivial eliminations to the parameter store.
///
/// For each elimination, sets the parameter to its determined value and marks
/// it as fixed so it is excluded from future solves.
pub fn apply_eliminations(eliminations: &[TrivialElimination], store: &mut ParamStore) {
    for elim in eliminations {
        store.set(elim.param, elim.determined_value);
        store.fix(elim.param);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    /// Constraint: param - target = 0.
    /// Residual: param_value - target.
    /// Jacobian: [(0, param, 1.0)].
    struct FixValueConstraint {
        id: ConstraintId,
        param: ParamId,
        target: f64,
    }

    impl crate::constraint::Constraint for FixValueConstraint {
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

    /// Constraint: 2*param - target = 0  (scaled Jacobian).
    /// Residual: 2*param_value - target.
    /// Jacobian: [(0, param, 2.0)].
    struct ScaledFixConstraint {
        id: ConstraintId,
        param: ParamId,
        target: f64,
    }

    impl crate::constraint::Constraint for ScaledFixConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "scaled_fix"
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
            vec![2.0 * store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 2.0)]
        }
    }

    /// A two-param constraint that should NOT be trivially eliminable
    /// when both params are free.
    struct TwoParamConstraint {
        id: ConstraintId,
        params: [ParamId; 2],
    }

    impl crate::constraint::Constraint for TwoParamConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "two_param"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.params[0]) - store.get(self.params[1])]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.params[0], 1.0), (0, self.params[1], -1.0)]
        }
    }

    fn dummy_owner() -> EntityId {
        EntityId::new(0, 0)
    }

    #[test]
    fn test_detect_trivial_single_free_param() {
        let mut store = ParamStore::new();
        let p = store.alloc(0.0, dummy_owner());

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 7.0,
        };

        let constraints: Vec<(usize, &dyn crate::constraint::Constraint)> = vec![(0, &c)];
        let elims = detect_trivial_eliminations(&constraints, &store);

        assert_eq!(elims.len(), 1);
        assert_eq!(elims[0].param, p);
        // x_new = 0.0 - (0.0 - 7.0) / 1.0 = 7.0
        assert!((elims[0].determined_value - 7.0).abs() < 1e-12);
        assert_eq!(elims[0].constraint_index, 0);
    }

    #[test]
    fn test_detect_trivial_with_scaled_jacobian() {
        let mut store = ParamStore::new();
        let p = store.alloc(0.0, dummy_owner());

        let c = ScaledFixConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 10.0, // 2*x - 10 = 0 => x = 5
        };

        let constraints: Vec<(usize, &dyn crate::constraint::Constraint)> = vec![(0, &c)];
        let elims = detect_trivial_eliminations(&constraints, &store);

        assert_eq!(elims.len(), 1);
        // x_new = 0.0 - (2*0.0 - 10.0) / 2.0 = 0.0 - (-10.0)/2.0 = 5.0
        assert!((elims[0].determined_value - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_no_elimination_with_two_free_params() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c = TwoParamConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        };

        let constraints: Vec<(usize, &dyn crate::constraint::Constraint)> = vec![(0, &c)];
        let elims = detect_trivial_eliminations(&constraints, &store);

        assert_eq!(elims.len(), 0);
    }

    #[test]
    fn test_elimination_with_one_fixed_one_free() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(5.0, owner);
        let b = store.alloc(0.0, owner);
        store.fix(a); // a is fixed at 5.0

        // Constraint: a - b = 0, so b should be determined as 5.0.
        let c = TwoParamConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        };

        let constraints: Vec<(usize, &dyn crate::constraint::Constraint)> = vec![(0, &c)];
        let elims = detect_trivial_eliminations(&constraints, &store);

        assert_eq!(elims.len(), 1);
        assert_eq!(elims[0].param, b);
        // residual = a - b = 5.0 - 0.0 = 5.0
        // J for b = -1.0
        // x_new = 0.0 - 5.0 / (-1.0) = 5.0
        assert!((elims[0].determined_value - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_eliminations() {
        let mut store = ParamStore::new();
        let p = store.alloc(0.0, dummy_owner());

        let elim = TrivialElimination {
            param: p,
            determined_value: 42.0,
            constraint_index: 0,
        };

        assert!(!store.is_fixed(p));
        apply_eliminations(&[elim], &mut store);

        assert!((store.get(p) - 42.0).abs() < 1e-15);
        assert!(store.is_fixed(p));
    }

    #[test]
    fn test_preserves_constraint_index() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);

        let c1 = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 3.0,
        };
        let c2 = FixValueConstraint {
            id: ConstraintId::new(1, 0),
            param: p2,
            target: 9.0,
        };

        // Use original indices 5 and 12 to verify they are preserved.
        let constraints: Vec<(usize, &dyn crate::constraint::Constraint)> =
            vec![(5, &c1), (12, &c2)];
        let elims = detect_trivial_eliminations(&constraints, &store);

        assert_eq!(elims.len(), 2);
        assert_eq!(elims[0].constraint_index, 5);
        assert_eq!(elims[1].constraint_index, 12);
    }
}
