//! Contract / Design-by-Contract tests for the V3 Constraint and Entity traits.
//!
//! Every `Constraint` implementor has implicit contracts:
//!
//! 1. `residuals()` returns exactly `equation_count()` elements
//! 2. All residual values are finite (no NaN or Inf)
//! 3. Jacobian row indices are all `< equation_count()`
//! 4. Jacobian ParamIds are a subset of `param_ids()`
//! 5. All Jacobian values are finite
//! 6. Analytical Jacobian matches finite-difference Jacobian
//! 7. `param_ids()` is non-empty
//! 8. `entity_ids()` is non-empty
//! 9. `name()` is non-empty
//! 10. `equation_count() > 0`
//! 11. `weight() > 0`
//! 12. `residuals()` and `jacobian()` are deterministic (same input → same output)
//!
//! Every `Entity` implementor has implicit contracts:
//!
//! 1. `params()` is non-empty
//! 2. `name()` is non-empty
//! 3. All param IDs are valid in the ParamStore
//!
//! These tests catch bugs instantly when adding new constraint types.

use std::collections::HashSet;

use solverang::constraint::Constraint;
use solverang::entity::Entity;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;

// =========================================================================
// Contract Validator
// =========================================================================

/// Validation result for a single contract check.
#[derive(Debug)]
struct ContractViolation {
    constraint_name: String,
    contract: &'static str,
    detail: String,
}

impl std::fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] Contract '{}' violated: {}",
            self.constraint_name, self.contract, self.detail
        )
    }
}

/// Validates all implicit contracts of a Constraint implementation.
///
/// This is the core contract validator. Given a constraint and a ParamStore
/// with valid parameters, it checks every contract and returns a list of
/// violations. An empty list means the constraint passes all contracts.
fn validate_constraint_contracts(
    constraint: &dyn Constraint,
    store: &ParamStore,
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    let name = constraint.name().to_string();

    // --- Contract 1: name is non-empty ---
    if constraint.name().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "name_non_empty",
            detail: "name() returned empty string".into(),
        });
    }

    // --- Contract 2: entity_ids is non-empty ---
    if constraint.entity_ids().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "entity_ids_non_empty",
            detail: "entity_ids() returned empty slice".into(),
        });
    }

    // --- Contract 3: param_ids is non-empty ---
    if constraint.param_ids().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "param_ids_non_empty",
            detail: "param_ids() returned empty slice".into(),
        });
    }

    // --- Contract 4: equation_count > 0 ---
    let eq_count = constraint.equation_count();
    if eq_count == 0 {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "equation_count_positive",
            detail: "equation_count() returned 0".into(),
        });
    }

    // --- Contract 5: weight > 0 ---
    let weight = constraint.weight();
    if weight <= 0.0 || !weight.is_finite() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "weight_positive_finite",
            detail: format!("weight() = {}", weight),
        });
    }

    // --- Contract 6: residuals count matches equation_count ---
    let residuals = constraint.residuals(store);
    if residuals.len() != eq_count {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "residual_count_matches",
            detail: format!(
                "equation_count()={} but residuals() returned {} elements",
                eq_count,
                residuals.len()
            ),
        });
    }

    // --- Contract 7: all residuals are finite ---
    for (i, &r) in residuals.iter().enumerate() {
        if !r.is_finite() {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "residuals_finite",
                detail: format!("residuals()[{}] = {} (not finite)", i, r),
            });
        }
    }

    // --- Contract 8: Jacobian row indices valid ---
    let jacobian = constraint.jacobian(store);
    for (idx, &(row, _pid, _val)) in jacobian.iter().enumerate() {
        if row >= eq_count {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "jacobian_row_valid",
                detail: format!(
                    "jacobian entry {} has row={} but equation_count()={}",
                    idx, row, eq_count
                ),
            });
        }
    }

    // --- Contract 9: Jacobian ParamIds are subset of param_ids ---
    let declared_params: HashSet<ParamId> = constraint.param_ids().iter().copied().collect();
    for (idx, &(_row, pid, _val)) in jacobian.iter().enumerate() {
        if !declared_params.contains(&pid) {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "jacobian_param_ids_valid",
                detail: format!(
                    "jacobian entry {} references {:?} which is not in param_ids()",
                    idx, pid
                ),
            });
        }
    }

    // --- Contract 10: all Jacobian values are finite ---
    for (idx, &(_row, _pid, val)) in jacobian.iter().enumerate() {
        if !val.is_finite() {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "jacobian_values_finite",
                detail: format!("jacobian entry {} has value={} (not finite)", idx, val),
            });
        }
    }

    // --- Contract 11: deterministic residuals ---
    let residuals2 = constraint.residuals(store);
    if residuals != residuals2 {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "residuals_deterministic",
            detail: "Two calls to residuals() with same store returned different results".into(),
        });
    }

    // --- Contract 12: deterministic Jacobian ---
    let jacobian2 = constraint.jacobian(store);
    if jacobian.len() != jacobian2.len() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "jacobian_deterministic",
            detail: format!(
                "Two calls to jacobian() returned different lengths: {} vs {}",
                jacobian.len(),
                jacobian2.len()
            ),
        });
    } else {
        for (i, (&(r1, p1, v1), &(r2, p2, v2))) in jacobian.iter().zip(jacobian2.iter()).enumerate()
        {
            if r1 != r2 || p1 != p2 || (v1 - v2).abs() > 1e-15 {
                violations.push(ContractViolation {
                    constraint_name: name.clone(),
                    contract: "jacobian_deterministic",
                    detail: format!(
                        "Entry {} differs: ({},{:?},{}) vs ({},{:?},{})",
                        i, r1, p1, v1, r2, p2, v2
                    ),
                });
                break;
            }
        }
    }

    violations
}

/// Validates the Jacobian accuracy against finite differences.
///
/// This is separated from `validate_constraint_contracts` because it requires
/// a mutable store for perturbation and has configurable tolerance.
fn validate_jacobian_accuracy(
    constraint: &dyn Constraint,
    store: &ParamStore,
    eps: f64,
    tol: f64,
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    let name = constraint.name().to_string();
    let params = constraint.param_ids().to_vec();
    let eq_count = constraint.equation_count();
    let jacobian = constraint.jacobian(store);

    // Build a map from (row, param_id) -> sum of analytical values
    // (handles cases where a param appears multiple times for same row)
    let mut analytical_map: std::collections::HashMap<(usize, ParamId), f64> =
        std::collections::HashMap::new();
    for &(row, pid, val) in &jacobian {
        *analytical_map.entry((row, pid)).or_insert(0.0) += val;
    }

    for eq in 0..eq_count {
        for &pid in &params {
            // Central finite difference
            let mut plus_store = store.snapshot();
            let orig = plus_store.get(pid);
            plus_store.set(pid, orig + eps);
            let r_plus = constraint.residuals(&plus_store);

            let mut minus_store = store.snapshot();
            minus_store.set(pid, orig - eps);
            let r_minus = constraint.residuals(&minus_store);

            if eq >= r_plus.len() || eq >= r_minus.len() {
                continue; // Already caught by residual_count check
            }

            let fd = (r_plus[eq] - r_minus[eq]) / (2.0 * eps);
            let analytical = analytical_map.get(&(eq, pid)).copied().unwrap_or(0.0);

            let error = (fd - analytical).abs();
            // Use relative tolerance for large values
            let scale = fd.abs().max(analytical.abs()).max(1.0);
            if error > tol * scale {
                violations.push(ContractViolation {
                    constraint_name: name.clone(),
                    contract: "jacobian_accuracy",
                    detail: format!(
                        "eq={}, param={:?}: analytical={:.10}, fd={:.10}, error={:.2e}",
                        eq, pid, analytical, fd, error
                    ),
                });
            }
        }
    }

    violations
}

/// Validates all Entity trait contracts.
fn validate_entity_contracts(
    entity: &dyn Entity,
    store: &ParamStore,
) -> Vec<String> {
    let mut violations = Vec::new();
    let name = entity.name().to_string();

    // Contract: name is non-empty
    if entity.name().is_empty() {
        violations.push(format!("[{}] name() returned empty string", name));
    }

    // Contract: params is non-empty
    if entity.params().is_empty() {
        violations.push(format!("[{}] params() returned empty slice", name));
    }

    // Contract: all param IDs are valid in the store (readable without panic)
    for &pid in entity.params() {
        let val = store.get(pid);
        if !val.is_finite() {
            violations.push(format!(
                "[{}] param {:?} has non-finite value {}",
                name, pid, val
            ));
        }
    }

    violations
}

/// Assert that a constraint passes all contracts, panicking with a detailed
/// message if any violation is found.
fn assert_contracts(constraint: &dyn Constraint, store: &ParamStore) {
    let violations = validate_constraint_contracts(constraint, store);
    if !violations.is_empty() {
        let msgs: Vec<String> = violations.iter().map(|v| v.to_string()).collect();
        panic!(
            "Contract violations for '{}':\n  {}",
            constraint.name(),
            msgs.join("\n  ")
        );
    }
}

/// Assert that a constraint's Jacobian is accurate, with configurable tolerance.
fn assert_jacobian_accurate(constraint: &dyn Constraint, store: &ParamStore, eps: f64, tol: f64) {
    let violations = validate_jacobian_accuracy(constraint, store, eps, tol);
    if !violations.is_empty() {
        let msgs: Vec<String> = violations.iter().map(|v| v.to_string()).collect();
        panic!(
            "Jacobian accuracy violations for '{}':\n  {}",
            constraint.name(),
            msgs.join("\n  ")
        );
    }
}

/// Assert that an entity passes all contracts.
fn assert_entity_contracts(entity: &dyn Entity, store: &ParamStore) {
    let violations = validate_entity_contracts(entity, store);
    if !violations.is_empty() {
        panic!(
            "Entity contract violations for '{}':\n  {}",
            entity.name(),
            violations.join("\n  ")
        );
    }
}

// =========================================================================
// Test helpers
// =========================================================================

fn eid(i: u32) -> EntityId {
    EntityId::new(i, 0)
}

fn cid(i: u32) -> ConstraintId {
    ConstraintId::new(i, 0)
}

// =========================================================================
// 2D Sketch Constraint Contract Tests
// =========================================================================

mod sketch2d_constraints {
    use super::*;
    use solverang::sketch2d::*;

    // --- DistancePtPt ---

    #[test]
    fn distance_pt_pt_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e1);
        let y2 = store.alloc(6.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    #[test]
    fn distance_pt_pt_contracts_at_zero_displacement() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(3.0, e0);
        let y1 = store.alloc(4.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 0.0);
        assert_contracts(&c, &store);
        // At zero displacement, Jacobian is zero everywhere - still valid
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- DistancePtLine ---

    #[test]
    fn distance_pt_line_contracts() {
        let ep = eid(0);
        let el = eid(1);
        let mut store = ParamStore::new();
        let px = store.alloc(3.0, ep);
        let py = store.alloc(2.0, ep);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(0.5, el);
        let x2 = store.alloc(7.0, el);
        let y2 = store.alloc(3.0, el);

        let c = DistancePtLine::new(cid(0), ep, el, px, py, x1, y1, x2, y2, 1.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Coincident ---

    #[test]
    fn coincident_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(5.0, e1);

        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    #[test]
    fn coincident_contracts_when_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(5.0, e0);
        let y1 = store.alloc(7.0, e0);
        let x2 = store.alloc(5.0, e1);
        let y2 = store.alloc(7.0, e1);

        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        assert_contracts(&c, &store);

        // Verify residuals are actually zero when satisfied
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-14));
    }

    // --- TangentLineCircle ---

    #[test]
    fn tangent_line_circle_contracts() {
        let el = eid(0);
        let ec = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(2.0, el);
        let x2 = store.alloc(5.0, el);
        let y2 = store.alloc(4.0, el);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(7.0, ec);
        let r = store.alloc(2.0, ec);

        let c = TangentLineCircle::new(cid(0), el, ec, x1, y1, x2, y2, cx, cy, r);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- TangentCircleCircle ---

    #[test]
    fn tangent_circle_circle_external_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let cx1 = store.alloc(1.0, e0);
        let cy1 = store.alloc(2.0, e0);
        let r1 = store.alloc(3.0, e0);
        let cx2 = store.alloc(6.0, e1);
        let cy2 = store.alloc(4.0, e1);
        let r2 = store.alloc(2.0, e1);

        let c = TangentCircleCircle::external(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    #[test]
    fn tangent_circle_circle_internal_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let cx1 = store.alloc(0.0, e0);
        let cy1 = store.alloc(0.0, e0);
        let r1 = store.alloc(5.0, e0);
        let cx2 = store.alloc(2.0, e1);
        let cy2 = store.alloc(0.0, e1);
        let r2 = store.alloc(3.0, e1);

        let c = TangentCircleCircle::internal(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Parallel ---

    #[test]
    fn parallel_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(6.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = Parallel::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Perpendicular ---

    #[test]
    fn perpendicular_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(3.0, e0);
        let x3 = store.alloc(2.0, e1);
        let y3 = store.alloc(0.0, e1);
        let x4 = store.alloc(5.0, e1);
        let y4 = store.alloc(7.0, e1);

        let c = Perpendicular::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Angle ---

    #[test]
    fn angle_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e);
        let y1 = store.alloc(2.0, e);
        let x2 = store.alloc(4.0, e);
        let y2 = store.alloc(6.0, e);

        let c = Angle::new(cid(0), e, x1, y1, x2, y2, 0.7);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Horizontal ---

    #[test]
    fn horizontal_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let y1 = store.alloc(1.0, e0);
        let y2 = store.alloc(5.0, e1);

        let c = Horizontal::new(cid(0), e0, e1, y1, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Vertical ---

    #[test]
    fn vertical_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(2.0, e0);
        let x2 = store.alloc(8.0, e1);

        let c = Vertical::new(cid(0), e0, e1, x1, x2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Fixed ---

    #[test]
    fn fixed_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);

        let c = Fixed::new(cid(0), e, x, y, 5.0, 7.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Midpoint ---

    #[test]
    fn midpoint_contracts() {
        let ep = eid(0);
        let el = eid(1);
        let mut store = ParamStore::new();
        let mx = store.alloc(3.0, ep);
        let my = store.alloc(4.0, ep);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(2.0, el);
        let x2 = store.alloc(7.0, el);
        let y2 = store.alloc(9.0, el);

        let c = Midpoint::new(cid(0), ep, el, mx, my, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Symmetric ---

    #[test]
    fn symmetric_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(6.0, e1);
        let y2 = store.alloc(9.0, e1);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(5.0, ec);

        let c = Symmetric::new(cid(0), e0, e1, ec, x1, y1, x2, y2, cx, cy);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- EqualLength ---

    #[test]
    fn equal_length_contracts() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(6.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(3.0, e1);

        let c = EqualLength::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- PointOnCircle ---

    #[test]
    fn point_on_circle_contracts() {
        let ep = eid(0);
        let ec = eid(1);
        let mut store = ParamStore::new();
        let px = store.alloc(2.0, ep);
        let py = store.alloc(3.0, ep);
        let cx = store.alloc(1.0, ec);
        let cy = store.alloc(1.0, ec);
        let r = store.alloc(4.0, ec);

        let c = PointOnCircle::new(cid(0), ep, ec, px, py, cx, cy, r);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Multi-point validation: all 2D constraints at multiple parameter configurations ---

    #[test]
    fn all_2d_constraints_contracts_at_large_values() {
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let mut store = ParamStore::new();

        // Use large coordinate values to test numerical stability
        let x1 = store.alloc(1000.0, e0);
        let y1 = store.alloc(2000.0, e0);
        let x2 = store.alloc(1003.0, e1);
        let y2 = store.alloc(2004.0, e1);
        let x3 = store.alloc(1010.0, ec);
        let y3 = store.alloc(2010.0, ec);
        let x4 = store.alloc(1013.0, ec);
        let y4 = store.alloc(2014.0, ec);
        let cx = store.alloc(1005.0, ec);
        let cy = store.alloc(2005.0, ec);
        let r = store.alloc(50.0, ec);

        // DistancePtPt
        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);

        // Coincident
        let c = Coincident::new(cid(1), e0, e1, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);

        // Parallel
        let c = Parallel::new(cid(2), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);

        // Perpendicular
        let c = Perpendicular::new(cid(3), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);

        // EqualLength
        let c = EqualLength::new(cid(4), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);

        // PointOnCircle
        let c = PointOnCircle::new(cid(5), e0, ec, x1, y1, cx, cy, r);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);
    }

    #[test]
    fn all_2d_constraints_contracts_at_small_values() {
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let mut store = ParamStore::new();

        // Use very small coordinate values
        let x1 = store.alloc(0.001, e0);
        let y1 = store.alloc(0.002, e0);
        let x2 = store.alloc(0.004, e1);
        let y2 = store.alloc(0.006, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 0.005);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-9, 1e-4);

        let c = Coincident::new(cid(1), e0, e1, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-9, 1e-5);

        let cx = store.alloc(0.0, ec);
        let cy = store.alloc(0.0, ec);
        let r = store.alloc(0.01, ec);
        let c = PointOnCircle::new(cid(2), e0, ec, x1, y1, cx, cy, r);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-9, 1e-4);
    }
}

// =========================================================================
// 3D Sketch Constraint Contract Tests
// =========================================================================

mod sketch3d_constraints {
    use super::*;
    use solverang::sketch3d::*;

    // --- Distance3D ---

    #[test]
    fn distance3d_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.0, e1);
        let x2 = store.alloc(4.0, e2);
        let y2 = store.alloc(6.0, e2);
        let z2 = store.alloc(3.0, e2);

        let c = Distance3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2, 5.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Coincident3D ---

    #[test]
    fn coincident3d_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.5, e1);
        let x2 = store.alloc(4.0, e2);
        let y2 = store.alloc(5.0, e2);
        let z2 = store.alloc(6.0, e2);

        let c = Coincident3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Fixed3D ---

    #[test]
    fn fixed3d_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(1.5, e);
        let y = store.alloc(2.5, e);
        let z = store.alloc(3.5, e);

        let c = Fixed3D::new(cid(0), e, x, y, z, [1.0, 2.0, 3.0]);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- PointOnPlane ---

    #[test]
    fn point_on_plane_contracts() {
        let pe = eid(0);
        let ple = eid(1);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, pe);
        let py = store.alloc(2.0, pe);
        let pz = store.alloc(0.5, pe);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.0, ple);
        let ny = store.alloc(0.0, ple);
        let nz = store.alloc(1.0, ple);

        let c = PointOnPlane::new(cid(0), pe, px, py, pz, ple, p0x, p0y, p0z, nx, ny, nz);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Coplanar ---

    #[test]
    fn coplanar_contracts() {
        let ple = eid(0);
        let pe1 = eid(1);
        let pe2 = eid(2);
        let mut store = ParamStore::new();

        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.3, ple);
        let ny = store.alloc(0.5, ple);
        let nz = store.alloc(1.0, ple);

        let px1 = store.alloc(1.0, pe1);
        let py1 = store.alloc(2.0, pe1);
        let pz1 = store.alloc(0.5, pe1);
        let px2 = store.alloc(3.0, pe2);
        let py2 = store.alloc(4.0, pe2);
        let pz2 = store.alloc(1.0, pe2);

        let c = Coplanar::new(
            cid(0),
            ple,
            p0x,
            p0y,
            p0z,
            nx,
            ny,
            nz,
            &[(pe1, px1, py1, pz1), (pe2, px2, py2, pz2)],
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Parallel3D ---

    #[test]
    fn parallel3d_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.5, e1);
        let l1_z2 = store.alloc(0.3, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(1.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(2.0, e2);
        let l2_y2 = store.alloc(1.5, e2);
        let l2_z2 = store.alloc(0.7, e2);

        let c = Parallel3D::new(
            cid(0),
            e1,
            l1_x1,
            l1_y1,
            l1_z1,
            l1_x2,
            l1_y2,
            l1_z2,
            e2,
            l2_x1,
            l2_y1,
            l2_z1,
            l2_x2,
            l2_y2,
            l2_z2,
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Perpendicular3D ---

    #[test]
    fn perpendicular3d_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.3, e1);
        let l1_z2 = store.alloc(0.0, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(0.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(-0.3, e2);
        let l2_y2 = store.alloc(1.0, e2);
        let l2_z2 = store.alloc(0.5, e2);

        let c = Perpendicular3D::new(
            cid(0),
            e1,
            l1_x1,
            l1_y1,
            l1_z1,
            l1_x2,
            l1_y2,
            l1_z2,
            e2,
            l2_x1,
            l2_y1,
            l2_z1,
            l2_x2,
            l2_y2,
            l2_z2,
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    // --- Coaxial ---

    #[test]
    fn coaxial_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let p1x = store.alloc(0.0, e1);
        let p1y = store.alloc(0.0, e1);
        let p1z = store.alloc(0.0, e1);
        let d1x = store.alloc(1.0, e1);
        let d1y = store.alloc(0.2, e1);
        let d1z = store.alloc(0.3, e1);
        let p2x = store.alloc(5.0, e2);
        let p2y = store.alloc(1.0, e2);
        let p2z = store.alloc(1.5, e2);
        let d2x = store.alloc(2.0, e2);
        let d2y = store.alloc(0.5, e2);
        let d2z = store.alloc(0.7, e2);

        let c = Coaxial::new(
            cid(0),
            e1,
            p1x,
            p1y,
            p1z,
            d1x,
            d1y,
            d1z,
            e2,
            p2x,
            p2y,
            p2z,
            d2x,
            d2y,
            d2z,
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }
}

// =========================================================================
// Assembly Constraint Contract Tests
// =========================================================================

mod assembly_constraints {
    use super::*;
    use solverang::assembly::*;

    /// Create a rigid body with given translation and identity rotation.
    fn make_body(store: &mut ParamStore, entity: EntityId, pos: [f64; 3]) -> (
        ParamId, ParamId, ParamId,
        ParamId, ParamId, ParamId, ParamId,
    ) {
        let tx = store.alloc(pos[0], entity);
        let ty = store.alloc(pos[1], entity);
        let tz = store.alloc(pos[2], entity);
        let qw = store.alloc(1.0, entity);
        let qx = store.alloc(0.0, entity);
        let qy = store.alloc(0.0, entity);
        let qz = store.alloc(0.0, entity);
        (tx, ty, tz, qw, qx, qy, qz)
    }

    /// Create a rigid body with slight rotation for non-trivial Jacobians.
    fn make_body_rotated(store: &mut ParamStore, entity: EntityId, pos: [f64; 3]) -> (
        ParamId, ParamId, ParamId,
        ParamId, ParamId, ParamId, ParamId,
    ) {
        let tx = store.alloc(pos[0], entity);
        let ty = store.alloc(pos[1], entity);
        let tz = store.alloc(pos[2], entity);
        let norm = (1.0_f64 + 0.01 + 0.04 + 0.09).sqrt();
        let qw = store.alloc(1.0 / norm, entity);
        let qx = store.alloc(0.1 / norm, entity);
        let qy = store.alloc(0.2 / norm, entity);
        let qz = store.alloc(0.3 / norm, entity);
        (tx, ty, tz, qw, qx, qy, qz)
    }

    // --- UnitQuaternion ---

    #[test]
    fn unit_quaternion_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let norm = (1.0_f64 + 0.04 + 0.09 + 0.01).sqrt();
        let qw = store.alloc(1.0 / norm, e);
        let qx = store.alloc(0.2 / norm, e);
        let qy = store.alloc(0.3 / norm, e);
        let qz = store.alloc(0.1 / norm, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    #[test]
    fn unit_quaternion_contracts_at_identity() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let qw = store.alloc(1.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        assert_contracts(&c, &store);

        // Verify residual is zero at identity
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-14);
    }

    // --- Mate ---

    #[test]
    fn mate_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body_rotated(&mut store, e1, [1.0, 2.0, 3.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body_rotated(&mut store, e2, [4.0, 5.0, 6.0]);

        let c = Mate::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [1.0, 0.5, -0.3],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [-0.5, 1.0, 0.2],
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);
    }

    #[test]
    fn mate_contracts_at_identity() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 0.0]);

        let c = Mate::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [1.0, 0.0, 0.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [1.0, 0.0, 0.0],
        );
        assert_contracts(&c, &store);

        // Verify residual is zero when both bodies coincide
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-12));
    }

    // --- CoaxialAssembly ---

    #[test]
    fn coaxial_assembly_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body_rotated(&mut store, e2, [2.0, 1.0, 0.5]);

        let c = CoaxialAssembly::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        );
        assert_contracts(&c, &store);
        // CoaxialAssembly uses FD internally, so accuracy check with wider tolerance
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);
    }

    // --- Insert ---

    #[test]
    fn insert_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 2.0, 3.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [4.0, 5.0, 6.0]);

        let c = Insert::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            2.0,
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-4);
    }

    // --- Gear ---

    #[test]
    fn gear_contracts() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        let theta1: f64 = 0.3;
        let b1_qw = store.alloc((theta1 / 2.0).cos(), e1);
        let b1_qx = store.alloc(0.0, e1);
        let b1_qy = store.alloc(0.0, e1);
        let b1_qz = store.alloc((theta1 / 2.0).sin(), e1);

        let theta2: f64 = 0.7;
        let b2_qw = store.alloc((theta2 / 2.0).cos(), e2);
        let b2_qx = store.alloc(0.0, e2);
        let b2_qy = store.alloc(0.0, e2);
        let b2_qz = store.alloc((theta2 / 2.0).sin(), e2);

        let c = Gear::new(
            cid(0),
            e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    #[test]
    fn gear_contracts_when_satisfied() {
        let e1 = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();

        // Body1: 30 deg about z, Body2: 60 deg (ratio=2)
        let theta1: f64 = std::f64::consts::PI / 6.0;
        let b1_qw = store.alloc((theta1 / 2.0).cos(), e1);
        let b1_qx = store.alloc(0.0, e1);
        let b1_qy = store.alloc(0.0, e1);
        let b1_qz = store.alloc((theta1 / 2.0).sin(), e1);

        let theta2: f64 = std::f64::consts::PI / 3.0;
        let b2_qw = store.alloc((theta2 / 2.0).cos(), e2);
        let b2_qx = store.alloc(0.0, e2);
        let b2_qy = store.alloc(0.0, e2);
        let b2_qz = store.alloc((theta2 / 2.0).sin(), e2);

        let c = Gear::new(
            cid(0),
            e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );
        assert_contracts(&c, &store);

        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "gear should be satisfied, residual={}", r[0]);
    }

    // --- RigidBody entity + UnitQuaternion combined contract ---

    #[test]
    fn rigid_body_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let tx = store.alloc(1.0, e);
        let ty = store.alloc(2.0, e);
        let tz = store.alloc(3.0, e);
        let qw = store.alloc(1.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let body = RigidBody::new(e, tx, ty, tz, qw, qx, qy, qz);
        assert_entity_contracts(&body, &store);
        assert_eq!(body.params().len(), 7);
    }
}

// =========================================================================
// 2D Entity Contract Tests
// =========================================================================

mod sketch2d_entities {
    use super::*;
    use solverang::sketch2d::*;

    #[test]
    fn point2d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(3.0, e);
        let y = store.alloc(4.0, e);

        let point = Point2D::new(e, x, y);
        assert_entity_contracts(&point, &store);
        assert_eq!(point.params().len(), 2);
        assert_eq!(point.name(), "Point2D");
        assert_eq!(point.id(), e);
    }

    #[test]
    fn line_segment2d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let x2 = store.alloc(10.0, e);
        let y2 = store.alloc(5.0, e);

        let line = LineSegment2D::new(e, x1, y1, x2, y2);
        assert_entity_contracts(&line, &store);
        assert_eq!(line.params().len(), 4);
        assert_eq!(line.name(), "LineSegment2D");
    }

    #[test]
    fn circle2d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let cx = store.alloc(1.0, e);
        let cy = store.alloc(2.0, e);
        let r = store.alloc(5.0, e);

        let circle = Circle2D::new(e, cx, cy, r);
        assert_entity_contracts(&circle, &store);
        assert_eq!(circle.params().len(), 3);
        assert_eq!(circle.name(), "Circle2D");
    }

    #[test]
    fn arc2d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let cx = store.alloc(0.0, e);
        let cy = store.alloc(0.0, e);
        let r = store.alloc(1.0, e);
        let a0 = store.alloc(0.0, e);
        let a1 = store.alloc(std::f64::consts::FRAC_PI_2, e);

        let arc = Arc2D::new(e, cx, cy, r, a0, a1);
        assert_entity_contracts(&arc, &store);
        assert_eq!(arc.params().len(), 5);
        assert_eq!(arc.name(), "Arc2D");
    }

    #[test]
    fn infinite_line2d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, e);
        let py = store.alloc(2.0, e);
        let dx = store.alloc(1.0, e);
        let dy = store.alloc(0.0, e);

        let line = InfiniteLine2D::new(e, px, py, dx, dy);
        assert_entity_contracts(&line, &store);
        assert_eq!(line.params().len(), 4);
        assert_eq!(line.name(), "InfiniteLine2D");
    }

    /// Verify that entity param IDs match the actual params stored.
    #[test]
    fn entity_params_are_valid_param_ids() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);

        let point = Point2D::new(e, x, y);

        // All param IDs should be readable from the store
        for &pid in point.params() {
            let val = store.get(pid);
            assert!(val.is_finite());
        }

        // Verify the IDs match what we allocated
        assert_eq!(point.params()[0], x);
        assert_eq!(point.params()[1], y);
    }

    /// Verify that shared params between entities and constraints are consistent.
    #[test]
    fn shared_params_consistency() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let p1 = Point2D::new(e0, x1, y1);
        let p2 = Point2D::new(e1, x2, y2);
        let line = LineSegment2D::new(eid(2), x1, y1, x2, y2);

        // Line endpoints share param IDs with points
        assert_eq!(line.params()[0], p1.params()[0]);
        assert_eq!(line.params()[1], p1.params()[1]);
        assert_eq!(line.params()[2], p2.params()[0]);
        assert_eq!(line.params()[3], p2.params()[1]);

        // A constraint over these shared params should reference valid IDs
        let c = solverang::sketch2d::DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        let constraint_params: HashSet<ParamId> = c.param_ids().iter().copied().collect();
        for &pid in p1.params() {
            assert!(constraint_params.contains(&pid));
        }
        for &pid in p2.params() {
            assert!(constraint_params.contains(&pid));
        }
    }
}

// =========================================================================
// 3D Entity Contract Tests
// =========================================================================

mod sketch3d_entities {
    use super::*;
    use solverang::sketch3d::*;

    #[test]
    fn point3d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);
        let z = store.alloc(3.0, e);

        let point = Point3D::new(e, x, y, z);
        assert_entity_contracts(&point, &store);
        assert_eq!(point.params().len(), 3);
        assert_eq!(point.name(), "Point3D");
    }

    #[test]
    fn line_segment3d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let z1 = store.alloc(0.0, e);
        let x2 = store.alloc(1.0, e);
        let y2 = store.alloc(2.0, e);
        let z2 = store.alloc(3.0, e);

        let seg = LineSegment3D::new(e, x1, y1, z1, x2, y2, z2);
        assert_entity_contracts(&seg, &store);
        assert_eq!(seg.params().len(), 6);
        assert_eq!(seg.name(), "LineSegment3D");
    }

    #[test]
    fn plane_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, e);
        let py = store.alloc(0.0, e);
        let pz = store.alloc(0.0, e);
        let nx = store.alloc(0.0, e);
        let ny = store.alloc(0.0, e);
        let nz = store.alloc(1.0, e);

        let plane = Plane::new(e, px, py, pz, nx, ny, nz);
        assert_entity_contracts(&plane, &store);
        assert_eq!(plane.params().len(), 6);
        assert_eq!(plane.name(), "Plane");
    }

    #[test]
    fn axis3d_entity_contracts() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, e);
        let py = store.alloc(2.0, e);
        let pz = store.alloc(3.0, e);
        let dx = store.alloc(0.0, e);
        let dy = store.alloc(0.0, e);
        let dz = store.alloc(1.0, e);

        let axis = Axis3D::new(e, px, py, pz, dx, dy, dz);
        assert_entity_contracts(&axis, &store);
        assert_eq!(axis.params().len(), 6);
        assert_eq!(axis.name(), "Axis3D");
    }
}

// =========================================================================
// Cross-cutting Contract Tests
// =========================================================================

/// Tests that validate contracts across multiple constraint types simultaneously.
mod cross_cutting {
    use super::*;

    /// A deliberately broken constraint to verify the validator catches bugs.
    /// This proves the validator works by checking negative cases.
    struct BrokenConstraint {
        id: ConstraintId,
        entity: EntityId,
        param: ParamId,
    }

    impl Constraint for BrokenConstraint {
        fn id(&self) -> ConstraintId { self.id }
        fn name(&self) -> &str { "Broken" }
        fn entity_ids(&self) -> &[EntityId] { std::slice::from_ref(&self.entity) }
        fn param_ids(&self) -> &[ParamId] { std::slice::from_ref(&self.param) }
        fn equation_count(&self) -> usize { 2 }  // Claims 2...
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param)]  // ...but returns only 1
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(5, self.param, 1.0)]  // Row 5 is out of range (eq_count=2)
        }
    }

    #[test]
    fn validator_catches_residual_count_mismatch() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let p = store.alloc(1.0, e);

        let c = BrokenConstraint { id: cid(0), entity: e, param: p };
        let violations = validate_constraint_contracts(&c, &store);

        let has_count_violation = violations.iter().any(|v| v.contract == "residual_count_matches");
        assert!(has_count_violation, "Should detect residual count mismatch");
    }

    #[test]
    fn validator_catches_jacobian_row_out_of_range() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let p = store.alloc(1.0, e);

        let c = BrokenConstraint { id: cid(0), entity: e, param: p };
        let violations = validate_constraint_contracts(&c, &store);

        let has_row_violation = violations.iter().any(|v| v.contract == "jacobian_row_valid");
        assert!(has_row_violation, "Should detect Jacobian row out of range");
    }

    /// A constraint with an invalid ParamId in the Jacobian.
    struct BadParamConstraint {
        id: ConstraintId,
        entity: EntityId,
        param: ParamId,
        bad_param: ParamId,
    }

    impl Constraint for BadParamConstraint {
        fn id(&self) -> ConstraintId { self.id }
        fn name(&self) -> &str { "BadParam" }
        fn entity_ids(&self) -> &[EntityId] { std::slice::from_ref(&self.entity) }
        fn param_ids(&self) -> &[ParamId] { std::slice::from_ref(&self.param) }
        fn equation_count(&self) -> usize { 1 }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param)]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            // References bad_param which is NOT in param_ids()
            vec![(0, self.bad_param, 1.0)]
        }
    }

    #[test]
    fn validator_catches_undeclared_param_in_jacobian() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let p = store.alloc(1.0, e);
        let bad_p = store.alloc(2.0, e);

        let c = BadParamConstraint {
            id: cid(0),
            entity: e,
            param: p,
            bad_param: bad_p,
        };
        let violations = validate_constraint_contracts(&c, &store);

        let has_param_violation = violations.iter().any(|v| v.contract == "jacobian_param_ids_valid");
        assert!(has_param_violation, "Should detect undeclared param in Jacobian");
    }

    /// A constraint with NaN in residuals.
    struct NanConstraint {
        id: ConstraintId,
        entity: EntityId,
        param: ParamId,
    }

    impl Constraint for NanConstraint {
        fn id(&self) -> ConstraintId { self.id }
        fn name(&self) -> &str { "NanConstraint" }
        fn entity_ids(&self) -> &[EntityId] { std::slice::from_ref(&self.entity) }
        fn param_ids(&self) -> &[ParamId] { std::slice::from_ref(&self.param) }
        fn equation_count(&self) -> usize { 1 }
        fn residuals(&self, _store: &ParamStore) -> Vec<f64> {
            vec![f64::NAN]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, f64::INFINITY)]
        }
    }

    #[test]
    fn validator_catches_nan_and_infinity() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let p = store.alloc(1.0, e);

        let c = NanConstraint { id: cid(0), entity: e, param: p };
        let violations = validate_constraint_contracts(&c, &store);

        let has_residual_nan = violations.iter().any(|v| v.contract == "residuals_finite");
        let has_jac_inf = violations.iter().any(|v| v.contract == "jacobian_values_finite");
        assert!(has_residual_nan, "Should detect NaN in residuals");
        assert!(has_jac_inf, "Should detect Infinity in Jacobian");
    }

    /// Verify that all real constraint types pass the validator (no false positives).
    #[test]
    fn no_false_positives_on_valid_constraints() {
        use solverang::sketch2d::*;

        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e1);
        let y2 = store.alloc(6.0, e1);

        // These should all pass with zero violations
        let constraints: Vec<Box<dyn Constraint>> = vec![
            Box::new(DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0)),
            Box::new(Coincident::new(cid(1), e0, e1, x1, y1, x2, y2)),
            Box::new(Horizontal::new(cid(2), e0, e1, y1, y2)),
            Box::new(Vertical::new(cid(3), e0, e1, x1, x2)),
            Box::new(Fixed::new(cid(4), e0, x1, y1, 5.0, 7.0)),
        ];

        for c in &constraints {
            let violations = validate_constraint_contracts(c.as_ref(), &store);
            assert!(
                violations.is_empty(),
                "False positive on '{}': {:?}",
                c.name(),
                violations.iter().map(|v| v.to_string()).collect::<Vec<_>>()
            );
        }
    }
}

// =========================================================================
// Stress tests: Constraints at boundary conditions
// =========================================================================

mod boundary_conditions {
    use super::*;
    use solverang::sketch2d::*;

    /// All constraints should maintain contracts with negative coordinates.
    #[test]
    fn negative_coordinates() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(-10.0, e0);
        let y1 = store.alloc(-20.0, e0);
        let x2 = store.alloc(-4.0, e1);
        let y2 = store.alloc(-6.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);

        let c = Coincident::new(cid(1), e0, e1, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }

    /// Constraints should handle identical points without panicking.
    #[test]
    fn identical_points() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(5.0, e0);
        let y1 = store.alloc(5.0, e0);
        let x2 = store.alloc(5.0, e1);
        let y2 = store.alloc(5.0, e1);

        // Coincident at same point: residuals should be zero
        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-14));

        // Distance at coincident points: should still satisfy contracts
        let c = DistancePtPt::new(cid(1), e0, e1, x1, y1, x2, y2, 0.0);
        assert_contracts(&c, &store);
    }

    /// Constraints should work with mixed positive/negative values.
    #[test]
    fn mixed_sign_coordinates() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(-5.0, e0);
        let y1 = store.alloc(3.0, e0);
        let x2 = store.alloc(7.0, e1);
        let y2 = store.alloc(-2.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);

        let c = Parallel::new(cid(1), e0, e1, x1, y1, x2, y2, x1, y1, x2, y2);
        assert_contracts(&c, &store);
        assert_jacobian_accurate(&c, &store, 1e-7, 1e-5);
    }
}
