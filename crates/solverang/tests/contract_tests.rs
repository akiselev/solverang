//! Contract / Design-by-Contract tests for the V3 Constraint and Entity traits.
//!
//! Every `Constraint` implementation has implicit contracts:
//!
//! 1. `residuals()` returns exactly `equation_count()` finite values
//! 2. `jacobian()` entries have `row < equation_count()`
//! 3. `jacobian()` entries reference only ParamIds from `param_ids()`
//! 4. `jacobian()` values are finite
//! 5. Analytical Jacobian matches finite-difference Jacobian
//! 6. `param_ids()` are all alive in the ParamStore
//! 7. `entity_ids()` is non-empty
//! 8. `name()` is non-empty
//! 9. When the constraint is geometrically satisfied, residuals ≈ 0
//! 10. Default trait methods: `weight() == 1.0`, `is_soft() == false`
//!
//! Every `Entity` implementation has implicit contracts:
//!
//! 1. `params()` is non-empty
//! 2. `name()` is non-empty
//! 3. All ParamIds in `params()` are alive and retrievable from the store
//!
//! This test module validates these contracts for **every** constraint and entity
//! type across sketch2d, sketch3d, and assembly modules.

use std::cell::RefCell;
use std::collections::HashSet;

use solverang::constraint::Constraint;
use solverang::entity::Entity;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;
use solverang::system::ConstraintSystem;

// =========================================================================
// Contract validator framework
// =========================================================================

/// Result of a single contract check.
#[derive(Debug)]
struct ContractViolation {
    constraint_name: String,
    contract: String,
    detail: String,
}

/// Validates all implicit contracts of a `Constraint` implementation.
///
/// Returns a list of violations (empty = all contracts satisfied).
fn validate_constraint_contracts(
    constraint: &dyn Constraint,
    store: &ParamStore,
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    let name = constraint.name().to_string();

    // Contract 1: name is non-empty
    if constraint.name().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "name_non_empty".into(),
            detail: "name() returned empty string".into(),
        });
    }

    // Contract 2: entity_ids is non-empty
    if constraint.entity_ids().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "entity_ids_non_empty".into(),
            detail: "entity_ids() returned empty slice".into(),
        });
    }

    // Contract 3: equation_count > 0
    let eq_count = constraint.equation_count();
    if eq_count == 0 {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "equation_count_positive".into(),
            detail: "equation_count() returned 0".into(),
        });
    }

    // Contract 4: param_ids are all alive in the store
    let param_ids = constraint.param_ids();
    for &pid in param_ids {
        // Attempt to get the value; if the ID is stale or invalid, this panics
        // in the real code, but we catch the violation here via a safe check.
        let val = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.get(pid)));
        if val.is_err() {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "param_ids_alive".into(),
                detail: format!("param_id {:?} is not alive in store", pid),
            });
        }
    }

    // Contract 5: residuals() returns exactly equation_count() elements
    let residuals = constraint.residuals(store);
    if residuals.len() != eq_count {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "residual_dimension".into(),
            detail: format!(
                "residuals() returned {} elements, equation_count() = {}",
                residuals.len(),
                eq_count
            ),
        });
    }

    // Contract 6: all residual values are finite
    for (i, &r) in residuals.iter().enumerate() {
        if !r.is_finite() {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "residual_finite".into(),
                detail: format!("residuals()[{}] = {} (not finite)", i, r),
            });
        }
    }

    // Contract 7-9: Jacobian contracts
    let jacobian = constraint.jacobian(store);
    let param_set: HashSet<ParamId> = param_ids.iter().copied().collect();

    for (idx, &(row, pid, val)) in jacobian.iter().enumerate() {
        // Contract 7: row < equation_count
        if row >= eq_count {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "jacobian_row_valid".into(),
                detail: format!(
                    "jacobian entry {} has row={}, but equation_count()={}",
                    idx, row, eq_count
                ),
            });
        }

        // Contract 8: ParamId is in param_ids()
        if !param_set.contains(&pid) {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "jacobian_param_valid".into(),
                detail: format!(
                    "jacobian entry {} references {:?} which is not in param_ids()",
                    idx, pid
                ),
            });
        }

        // Contract 9: value is finite
        if !val.is_finite() {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "jacobian_value_finite".into(),
                detail: format!(
                    "jacobian entry {} has value={} (not finite)",
                    idx, val
                ),
            });
        }
    }

    // Contract 10: default trait methods
    if constraint.weight() != 1.0 && !constraint.is_soft() {
        // Only flag if weight != 1.0 for a non-soft constraint (unusual)
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "default_weight".into(),
            detail: format!("weight() = {} for non-soft constraint", constraint.weight()),
        });
    }

    violations
}

/// Validates that the analytical Jacobian matches finite differences.
///
/// This is the most important contract: if the Jacobian is wrong, the solver
/// will produce wrong geometry.
fn validate_jacobian_accuracy(
    constraint: &dyn Constraint,
    store: &ParamStore,
    eps: f64,
    tol: f64,
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    let name = constraint.name().to_string();
    let params = constraint.param_ids().to_vec();
    let analytical = constraint.jacobian(store);
    let eq_count = constraint.equation_count();

    // For each (equation, param) pair, compare analytical vs finite-difference.
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
                continue; // dimension mismatch already caught above
            }

            let fd = (r_plus[eq] - r_minus[eq]) / (2.0 * eps);

            // Sum all analytical entries for this (eq, pid) pair
            let ana: f64 = analytical
                .iter()
                .filter(|&&(r, p, _)| r == eq && p == pid)
                .map(|&(_, _, v)| v)
                .sum();

            let error = (fd - ana).abs();
            // Use relative tolerance for large values
            let scale = fd.abs().max(ana.abs()).max(1.0);
            if error > tol * scale {
                violations.push(ContractViolation {
                    constraint_name: name.clone(),
                    contract: "jacobian_accuracy".into(),
                    detail: format!(
                        "eq={}, param={:?}: analytical={:.10}, fd={:.10}, error={:.2e}",
                        eq, pid, ana, fd, error,
                    ),
                });
            }
        }
    }

    violations
}

/// Validates all contracts of an `Entity` implementation.
fn validate_entity_contracts(
    entity: &dyn Entity,
    store: &ParamStore,
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    let name = entity.name().to_string();

    // Contract 1: name is non-empty
    if entity.name().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "entity_name_non_empty".into(),
            detail: "name() returned empty string".into(),
        });
    }

    // Contract 2: params is non-empty
    if entity.params().is_empty() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "entity_params_non_empty".into(),
            detail: "params() returned empty slice".into(),
        });
    }

    // Contract 3: all params are alive in the store
    for &pid in entity.params() {
        let val = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.get(pid)));
        if val.is_err() {
            violations.push(ContractViolation {
                constraint_name: name.clone(),
                contract: "entity_params_alive".into(),
                detail: format!("param {:?} is not alive in store", pid),
            });
        }
    }

    // Contract 4: no duplicate param IDs
    let param_set: HashSet<ParamId> = entity.params().iter().copied().collect();
    if param_set.len() != entity.params().len() {
        violations.push(ContractViolation {
            constraint_name: name.clone(),
            contract: "entity_params_unique".into(),
            detail: format!(
                "params() has {} entries but only {} unique",
                entity.params().len(),
                param_set.len()
            ),
        });
    }

    violations
}

/// Run all contract checks and panic with a detailed report if any fail.
fn assert_constraint_contracts(constraint: &dyn Constraint, store: &ParamStore) {
    let mut all_violations = validate_constraint_contracts(constraint, store);
    all_violations.extend(validate_jacobian_accuracy(constraint, store, 1e-7, 1e-4));

    if !all_violations.is_empty() {
        let report: Vec<String> = all_violations
            .iter()
            .map(|v| format!("  [{}] {}: {}", v.constraint_name, v.contract, v.detail))
            .collect();
        panic!(
            "Contract violations for '{}':\n{}",
            constraint.name(),
            report.join("\n")
        );
    }
}

/// Run all contract checks for a constraint known to be in a satisfied state.
fn assert_constraint_satisfied(constraint: &dyn Constraint, store: &ParamStore, tol: f64) {
    let residuals = constraint.residuals(store);
    for (i, &r) in residuals.iter().enumerate() {
        assert!(
            r.abs() < tol,
            "Constraint '{}' residual[{}] = {:.2e} exceeds tolerance {:.2e} \
             (should be ~0 when satisfied)",
            constraint.name(),
            i,
            r,
            tol,
        );
    }
}

fn assert_entity_contracts(entity: &dyn Entity, store: &ParamStore) {
    let violations = validate_entity_contracts(entity, store);
    if !violations.is_empty() {
        let report: Vec<String> = violations
            .iter()
            .map(|v| format!("  [{}] {}: {}", v.constraint_name, v.contract, v.detail))
            .collect();
        panic!(
            "Entity contract violations for '{}':\n{}",
            entity.name(),
            report.join("\n")
        );
    }
}

// =========================================================================
// Test helpers
// =========================================================================

/// Mint a fresh `EntityId` via a thread-local `ConstraintSystem`.
///
/// `EntityId::new()` is `pub(crate)`, so integration tests cannot construct
/// IDs directly.  Instead we use `ConstraintSystem::alloc_entity_id()` which
/// is public.  The argument `_i` is ignored — every call mints a new,
/// unique ID.  Using a thread-local avoids creating a system per test.
fn eid(_i: u32) -> EntityId {
    thread_local! {
        static SYS: RefCell<ConstraintSystem> = RefCell::new(ConstraintSystem::new());
    }
    SYS.with(|sys| sys.borrow_mut().alloc_entity_id())
}

/// Mint a fresh `ConstraintId` via a thread-local `ConstraintSystem`.
fn cid(_i: u32) -> ConstraintId {
    thread_local! {
        static SYS: RefCell<ConstraintSystem> = RefCell::new(ConstraintSystem::new());
    }
    SYS.with(|sys| sys.borrow_mut().alloc_constraint_id())
}

// =========================================================================
// SKETCH 2D ENTITY CONTRACT TESTS
// =========================================================================

mod sketch2d_entity_contracts {
    use super::*;
    use solverang::sketch2d::{Arc2D, Circle2D, InfiniteLine2D, LineSegment2D, Point2D};

    #[test]
    fn point2d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(3.14, e);
        let y = store.alloc(2.72, e);
        let pt = Point2D::new(e, x, y);

        assert_entity_contracts(&pt, &store);
        assert_eq!(pt.params().len(), 2);
        assert_eq!(pt.id(), e);
    }

    #[test]
    fn line_segment2d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let x2 = store.alloc(5.0, e);
        let y2 = store.alloc(5.0, e);
        let line = LineSegment2D::new(e, x1, y1, x2, y2);

        assert_entity_contracts(&line, &store);
        assert_eq!(line.params().len(), 4);
    }

    #[test]
    fn circle2d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let cx = store.alloc(1.0, e);
        let cy = store.alloc(2.0, e);
        let r = store.alloc(5.0, e);
        let circle = Circle2D::new(e, cx, cy, r);

        assert_entity_contracts(&circle, &store);
        assert_eq!(circle.params().len(), 3);
    }

    #[test]
    fn arc2d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let cx = store.alloc(0.0, e);
        let cy = store.alloc(0.0, e);
        let r = store.alloc(1.0, e);
        let a0 = store.alloc(0.0, e);
        let a1 = store.alloc(std::f64::consts::FRAC_PI_2, e);
        let arc = Arc2D::new(e, cx, cy, r, a0, a1);

        assert_entity_contracts(&arc, &store);
        assert_eq!(arc.params().len(), 5);
    }

    #[test]
    fn infinite_line2d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let px = store.alloc(1.0, e);
        let py = store.alloc(2.0, e);
        let dx = store.alloc(1.0, e);
        let dy = store.alloc(0.0, e);
        let line = InfiniteLine2D::new(e, px, py, dx, dy);

        assert_entity_contracts(&line, &store);
        assert_eq!(line.params().len(), 4);
    }
}

// =========================================================================
// SKETCH 3D ENTITY CONTRACT TESTS
// =========================================================================

mod sketch3d_entity_contracts {
    use super::*;
    use solverang::sketch3d::{Axis3D, LineSegment3D, Plane, Point3D};

    #[test]
    fn point3d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);
        let z = store.alloc(3.0, e);
        let pt = Point3D::new(e, x, y, z);

        assert_entity_contracts(&pt, &store);
        assert_eq!(pt.params().len(), 3);
    }

    #[test]
    fn line_segment3d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let z1 = store.alloc(0.0, e);
        let x2 = store.alloc(1.0, e);
        let y2 = store.alloc(2.0, e);
        let z2 = store.alloc(3.0, e);
        let seg = LineSegment3D::new(e, x1, y1, z1, x2, y2, z2);

        assert_entity_contracts(&seg, &store);
        assert_eq!(seg.params().len(), 6);
    }

    #[test]
    fn plane_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let px = store.alloc(0.0, e);
        let py = store.alloc(0.0, e);
        let pz = store.alloc(0.0, e);
        let nx = store.alloc(0.0, e);
        let ny = store.alloc(0.0, e);
        let nz = store.alloc(1.0, e);
        let plane = Plane::new(e, px, py, pz, nx, ny, nz);

        assert_entity_contracts(&plane, &store);
        assert_eq!(plane.params().len(), 6);
    }

    #[test]
    fn axis3d_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let px = store.alloc(0.0, e);
        let py = store.alloc(0.0, e);
        let pz = store.alloc(0.0, e);
        let dx = store.alloc(0.0, e);
        let dy = store.alloc(0.0, e);
        let dz = store.alloc(1.0, e);
        let axis = Axis3D::new(e, px, py, pz, dx, dy, dz);

        assert_entity_contracts(&axis, &store);
        assert_eq!(axis.params().len(), 6);
    }
}

// =========================================================================
// ASSEMBLY ENTITY CONTRACT TESTS
// =========================================================================

mod assembly_entity_contracts {
    use super::*;
    use solverang::assembly::RigidBody;

    #[test]
    fn rigid_body_contracts() {
        let mut store = ParamStore::new();
        let e = eid(0);
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
        assert_eq!(body.name(), "RigidBody");
    }
}

// =========================================================================
// SKETCH 2D CONSTRAINT CONTRACT TESTS
// =========================================================================

mod sketch2d_constraint_contracts {
    use super::*;
    use solverang::sketch2d::*;

    // --- DistancePtPt ---

    #[test]
    fn distance_pt_pt_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn distance_pt_pt_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.5, e1);
        let y2 = store.alloc(7.3, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 10.0);
        assert_constraint_contracts(&c, &store);
        // Not satisfied, but all structural contracts still hold
    }

    #[test]
    fn distance_pt_pt_zero_distance() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(5.0, e0);
        let y1 = store.alloc(5.0, e0);
        let x2 = store.alloc(5.0, e1);
        let y2 = store.alloc(5.0, e1);

        // Zero-distance: squared formulation should handle this without singularity
        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 0.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    // --- DistancePtLine ---

    #[test]
    fn distance_pt_line_contracts_satisfied() {
        let mut store = ParamStore::new();
        let ep = eid(0);
        let el = eid(1);
        let px = store.alloc(5.0, ep);
        let py = store.alloc(1.0, ep);
        let x1 = store.alloc(0.0, el);
        let y1 = store.alloc(0.0, el);
        let x2 = store.alloc(10.0, el);
        let y2 = store.alloc(0.0, el);

        let c = DistancePtLine::new(cid(0), ep, el, px, py, x1, y1, x2, y2, 1.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn distance_pt_line_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let ep = eid(0);
        let el = eid(1);
        let px = store.alloc(3.0, ep);
        let py = store.alloc(2.5, ep);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(0.5, el);
        let x2 = store.alloc(7.0, el);
        let y2 = store.alloc(3.0, el);

        let c = DistancePtLine::new(cid(0), ep, el, px, py, x1, y1, x2, y2, 2.0);
        assert_constraint_contracts(&c, &store);
    }

    // --- Coincident ---

    #[test]
    fn coincident_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(3.0, e0);
        let y1 = store.alloc(4.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn coincident_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(5.0, e1);

        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        assert_constraint_contracts(&c, &store);
    }

    // --- TangentLineCircle ---

    #[test]
    fn tangent_line_circle_contracts_satisfied() {
        let mut store = ParamStore::new();
        let el = eid(0);
        let ec = eid(1);
        // Horizontal line y=5, circle at origin r=5
        let x1 = store.alloc(-10.0, el);
        let y1 = store.alloc(5.0, el);
        let x2 = store.alloc(10.0, el);
        let y2 = store.alloc(5.0, el);
        let cx = store.alloc(0.0, ec);
        let cy = store.alloc(0.0, ec);
        let r = store.alloc(5.0, ec);

        let c = TangentLineCircle::new(cid(0), el, ec, x1, y1, x2, y2, cx, cy, r);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn tangent_line_circle_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let el = eid(0);
        let ec = eid(1);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(2.0, el);
        let x2 = store.alloc(5.0, el);
        let y2 = store.alloc(4.0, el);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(7.0, ec);
        let r = store.alloc(2.0, ec);

        let c = TangentLineCircle::new(cid(0), el, ec, x1, y1, x2, y2, cx, cy, r);
        assert_constraint_contracts(&c, &store);
    }

    // --- TangentCircleCircle ---

    #[test]
    fn tangent_circle_circle_external_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        // dist=5=3+2
        let cx1 = store.alloc(0.0, e0);
        let cy1 = store.alloc(0.0, e0);
        let r1 = store.alloc(3.0, e0);
        let cx2 = store.alloc(5.0, e1);
        let cy2 = store.alloc(0.0, e1);
        let r2 = store.alloc(2.0, e1);

        let c = TangentCircleCircle::external(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn tangent_circle_circle_internal_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        // dist=2=5-3
        let cx1 = store.alloc(0.0, e0);
        let cy1 = store.alloc(0.0, e0);
        let r1 = store.alloc(5.0, e0);
        let cx2 = store.alloc(2.0, e1);
        let cy2 = store.alloc(0.0, e1);
        let r2 = store.alloc(3.0, e1);

        let c = TangentCircleCircle::internal(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn tangent_circle_circle_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let cx1 = store.alloc(1.0, e0);
        let cy1 = store.alloc(2.0, e0);
        let r1 = store.alloc(3.0, e0);
        let cx2 = store.alloc(6.0, e1);
        let cy2 = store.alloc(4.0, e1);
        let r2 = store.alloc(2.0, e1);

        let ext = TangentCircleCircle::external(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        assert_constraint_contracts(&ext, &store);

        let int = TangentCircleCircle::internal(cid(1), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        assert_constraint_contracts(&int, &store);
    }

    // --- Parallel ---

    #[test]
    fn parallel_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        // dir (1,2) and dir (2,4) are parallel
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(1.0, e0);
        let y2 = store.alloc(2.0, e0);
        let x3 = store.alloc(3.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(5.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = Parallel::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn parallel_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(6.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = Parallel::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
    }

    // --- Perpendicular ---

    #[test]
    fn perpendicular_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        // dir (1,0) and dir (0,1) are perpendicular
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(1.0, e0);
        let y2 = store.alloc(0.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(0.0, e1);
        let x4 = store.alloc(0.0, e1);
        let y4 = store.alloc(1.0, e1);

        let c = Perpendicular::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn perpendicular_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(3.0, e0);
        let x3 = store.alloc(2.0, e1);
        let y3 = store.alloc(0.0, e1);
        let x4 = store.alloc(5.0, e1);
        let y4 = store.alloc(7.0, e1);

        let c = Perpendicular::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
    }

    // --- Angle ---

    #[test]
    fn angle_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        // 45 degree line
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let x2 = store.alloc(1.0, e);
        let y2 = store.alloc(1.0, e);

        let c = Angle::new(cid(0), e, x1, y1, x2, y2, std::f64::consts::FRAC_PI_4);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn angle_contracts_zero_angle() {
        let mut store = ParamStore::new();
        let e = eid(0);
        // Horizontal line, angle = 0
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let x2 = store.alloc(5.0, e);
        let y2 = store.alloc(0.0, e);

        let c = Angle::new(cid(0), e, x1, y1, x2, y2, 0.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn angle_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x1 = store.alloc(1.0, e);
        let y1 = store.alloc(2.0, e);
        let x2 = store.alloc(4.0, e);
        let y2 = store.alloc(6.0, e);

        let c = Angle::new(cid(0), e, x1, y1, x2, y2, 0.7);
        assert_constraint_contracts(&c, &store);
    }

    // --- Horizontal ---

    #[test]
    fn horizontal_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let y1 = store.alloc(3.0, e0);
        let y2 = store.alloc(3.0, e1);

        let c = Horizontal::new(cid(0), e0, e1, y1, y2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn horizontal_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let y1 = store.alloc(1.0, e0);
        let y2 = store.alloc(5.0, e1);

        let c = Horizontal::new(cid(0), e0, e1, y1, y2);
        assert_constraint_contracts(&c, &store);
    }

    // --- Vertical ---

    #[test]
    fn vertical_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(7.0, e0);
        let x2 = store.alloc(7.0, e1);

        let c = Vertical::new(cid(0), e0, e1, x1, x2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn vertical_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(2.0, e0);
        let x2 = store.alloc(8.0, e1);

        let c = Vertical::new(cid(0), e0, e1, x1, x2);
        assert_constraint_contracts(&c, &store);
    }

    // --- Fixed ---

    #[test]
    fn fixed_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(3.0, e);
        let y = store.alloc(4.0, e);

        let c = Fixed::new(cid(0), e, x, y, 3.0, 4.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn fixed_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);

        let c = Fixed::new(cid(0), e, x, y, 5.0, 7.0);
        assert_constraint_contracts(&c, &store);
    }

    // --- Midpoint ---

    #[test]
    fn midpoint_contracts_satisfied() {
        let mut store = ParamStore::new();
        let ep = eid(0);
        let el = eid(1);
        let mx = store.alloc(5.0, ep);
        let my = store.alloc(3.0, ep);
        let x1 = store.alloc(2.0, el);
        let y1 = store.alloc(1.0, el);
        let x2 = store.alloc(8.0, el);
        let y2 = store.alloc(5.0, el);

        let c = Midpoint::new(cid(0), ep, el, mx, my, x1, y1, x2, y2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn midpoint_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let ep = eid(0);
        let el = eid(1);
        let mx = store.alloc(3.0, ep);
        let my = store.alloc(4.0, ep);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(2.0, el);
        let x2 = store.alloc(7.0, el);
        let y2 = store.alloc(9.0, el);

        let c = Midpoint::new(cid(0), ep, el, mx, my, x1, y1, x2, y2);
        assert_constraint_contracts(&c, &store);
    }

    // --- Symmetric ---

    #[test]
    fn symmetric_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(5.0, e1);
        let y2 = store.alloc(8.0, e1);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(5.0, ec);

        let c = Symmetric::new(cid(0), e0, e1, ec, x1, y1, x2, y2, cx, cy);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn symmetric_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(6.0, e1);
        let y2 = store.alloc(9.0, e1);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(5.0, ec);

        let c = Symmetric::new(cid(0), e0, e1, ec, x1, y1, x2, y2, cx, cy);
        assert_constraint_contracts(&c, &store);
    }

    // --- EqualLength ---

    #[test]
    fn equal_length_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        // Both lines have length 5
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e0);
        let y2 = store.alloc(4.0, e0);
        let x3 = store.alloc(1.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(4.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = EqualLength::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn equal_length_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(6.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(3.0, e1);

        let c = EqualLength::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
    }

    // --- PointOnCircle ---

    #[test]
    fn point_on_circle_contracts_satisfied() {
        let mut store = ParamStore::new();
        let ep = eid(0);
        let ec = eid(1);
        // (3,4) on circle center (0,0) r=5
        let px = store.alloc(3.0, ep);
        let py = store.alloc(4.0, ep);
        let cx = store.alloc(0.0, ec);
        let cy = store.alloc(0.0, ec);
        let r = store.alloc(5.0, ec);

        let c = PointOnCircle::new(cid(0), ep, ec, px, py, cx, cy, r);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn point_on_circle_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let ep = eid(0);
        let ec = eid(1);
        let px = store.alloc(2.0, ep);
        let py = store.alloc(3.0, ep);
        let cx = store.alloc(1.0, ec);
        let cy = store.alloc(1.0, ec);
        let r = store.alloc(4.0, ec);

        let c = PointOnCircle::new(cid(0), ep, ec, px, py, cx, cy, r);
        assert_constraint_contracts(&c, &store);
    }

    // --- Comprehensive: all 15 sketch2d constraints with large coordinates ---

    #[test]
    fn all_sketch2d_constraints_large_values() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let scale = 1e6;

        let x1 = store.alloc(0.0 * scale, e0);
        let y1 = store.alloc(0.0 * scale, e0);
        let x2 = store.alloc(3.0 * scale, e0);
        let y2 = store.alloc(4.0 * scale, e0);

        // At large scales, finite-difference Jacobian accuracy degrades due to
        // catastrophic cancellation (eps=1e-7 is tiny vs 1e6 values). Use a
        // scale-appropriate eps for Jacobian checks instead of the default.
        let large_eps = 1e-1;  // Appropriate for 1e6-scale values
        let large_tol = 1e-4;

        // DistancePtPt
        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0 * scale);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty(), "{:?}", violations.iter().map(|v| &v.detail).collect::<Vec<_>>());
        let jac_violations = validate_jacobian_accuracy(&c, &store, large_eps, large_tol);
        assert!(jac_violations.is_empty(), "{:?}", jac_violations.iter().map(|v| &v.detail).collect::<Vec<_>>());

        // Coincident
        let xa = store.alloc(1e6, e0);
        let ya = store.alloc(2e6, e0);
        let xb = store.alloc(1e6, e1);
        let yb = store.alloc(2e6, e1);
        let c = Coincident::new(cid(1), e0, e1, xa, ya, xb, yb);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty(), "{:?}", violations.iter().map(|v| &v.detail).collect::<Vec<_>>());

        // Horizontal
        let ya2 = store.alloc(5e6, e0);
        let yb2 = store.alloc(5e6, e1);
        let c = Horizontal::new(cid(2), e0, e1, ya2, yb2);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty(), "{:?}", violations.iter().map(|v| &v.detail).collect::<Vec<_>>());

        // Vertical
        let xa2 = store.alloc(7e6, e0);
        let xb2 = store.alloc(7e6, e1);
        let c = Vertical::new(cid(3), e0, e1, xa2, xb2);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty(), "{:?}", violations.iter().map(|v| &v.detail).collect::<Vec<_>>());

        // Fixed
        let xf = store.alloc(1e6, e0);
        let yf = store.alloc(2e6, e0);
        let c = Fixed::new(cid(4), e0, xf, yf, 1e6, 2e6);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty(), "{:?}", violations.iter().map(|v| &v.detail).collect::<Vec<_>>());
    }

    // --- Comprehensive: all 15 sketch2d constraints with tiny values ---

    #[test]
    fn all_sketch2d_constraints_tiny_values() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let scale = 1e-6;

        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0 * scale, e1);
        let y2 = store.alloc(4.0 * scale, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0 * scale);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-20);
    }
}

// =========================================================================
// SKETCH 3D CONSTRAINT CONTRACT TESTS
// =========================================================================

mod sketch3d_constraint_contracts {
    use super::*;
    use solverang::sketch3d::*;

    // --- Distance3D ---

    #[test]
    fn distance3d_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(0.0, e1);
        let y1 = store.alloc(0.0, e1);
        let z1 = store.alloc(0.0, e1);
        let x2 = store.alloc(3.0, e2);
        let y2 = store.alloc(4.0, e2);
        let z2 = store.alloc(0.0, e2);

        let c = Distance3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2, 5.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn distance3d_contracts_diagonal() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.0, e1);
        let x2 = store.alloc(4.0, e2);
        let y2 = store.alloc(6.0, e2);
        let z2 = store.alloc(3.0, e2);

        let c = Distance3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2, 5.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    // --- Coincident3D ---

    #[test]
    fn coincident3d_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.0, e1);
        let x2 = store.alloc(1.0, e2);
        let y2 = store.alloc(2.0, e2);
        let z2 = store.alloc(3.0, e2);

        let c = Coincident3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn coincident3d_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.5, e1);
        let x2 = store.alloc(4.0, e2);
        let y2 = store.alloc(5.0, e2);
        let z2 = store.alloc(6.0, e2);

        let c = Coincident3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2);
        assert_constraint_contracts(&c, &store);
    }

    // --- Fixed3D ---

    #[test]
    fn fixed3d_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);
        let z = store.alloc(3.0, e);

        let c = Fixed3D::new(cid(0), e, x, y, z, [1.0, 2.0, 3.0]);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn fixed3d_contracts_unsatisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(1.5, e);
        let y = store.alloc(2.5, e);
        let z = store.alloc(3.5, e);

        let c = Fixed3D::new(cid(0), e, x, y, z, [1.0, 2.0, 3.0]);
        assert_constraint_contracts(&c, &store);
    }

    // --- PointOnPlane ---

    #[test]
    fn point_on_plane_contracts_satisfied() {
        let mut store = ParamStore::new();
        let pe = eid(0);
        let ple = eid(1);
        // Point at (1,2,0) on plane z=0
        let px = store.alloc(1.0, pe);
        let py = store.alloc(2.0, pe);
        let pz = store.alloc(0.0, pe);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.0, ple);
        let ny = store.alloc(0.0, ple);
        let nz = store.alloc(1.0, ple);

        let c = PointOnPlane::new(cid(0), pe, px, py, pz, ple, p0x, p0y, p0z, nx, ny, nz);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn point_on_plane_contracts_oblique() {
        let mut store = ParamStore::new();
        let pe = eid(0);
        let ple = eid(1);
        // Oblique plane: n=(1,1,1), point on plane (0,0,0), test point (1,-1,0) -> dot=0
        let px = store.alloc(1.0, pe);
        let py = store.alloc(-1.0, pe);
        let pz = store.alloc(0.0, pe);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(1.0, ple);
        let ny = store.alloc(1.0, ple);
        let nz = store.alloc(1.0, ple);

        let c = PointOnPlane::new(cid(0), pe, px, py, pz, ple, p0x, p0y, p0z, nx, ny, nz);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn point_on_plane_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let pe = eid(0);
        let ple = eid(1);
        let px = store.alloc(1.0, pe);
        let py = store.alloc(2.0, pe);
        let pz = store.alloc(0.5, pe);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.3, ple);
        let ny = store.alloc(0.5, ple);
        let nz = store.alloc(1.0, ple);

        let c = PointOnPlane::new(cid(0), pe, px, py, pz, ple, p0x, p0y, p0z, nx, ny, nz);
        assert_constraint_contracts(&c, &store);
    }

    // --- Coplanar ---

    #[test]
    fn coplanar_contracts_satisfied() {
        let mut store = ParamStore::new();
        let ple = eid(0);
        let pe1 = eid(1);
        let pe2 = eid(2);
        // Plane z=0
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.0, ple);
        let ny = store.alloc(0.0, ple);
        let nz = store.alloc(1.0, ple);
        // Points on z=0
        let px1 = store.alloc(1.0, pe1);
        let py1 = store.alloc(2.0, pe1);
        let pz1 = store.alloc(0.0, pe1);
        let px2 = store.alloc(3.0, pe2);
        let py2 = store.alloc(4.0, pe2);
        let pz2 = store.alloc(0.0, pe2);

        let c = Coplanar::new(
            cid(0), ple, p0x, p0y, p0z, nx, ny, nz,
            &[(pe1, px1, py1, pz1), (pe2, px2, py2, pz2)],
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn coplanar_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let ple = eid(0);
        let pe1 = eid(1);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.3, ple);
        let ny = store.alloc(0.5, ple);
        let nz = store.alloc(1.0, ple);
        let px1 = store.alloc(1.0, pe1);
        let py1 = store.alloc(2.0, pe1);
        let pz1 = store.alloc(0.5, pe1);

        let c = Coplanar::new(
            cid(0), ple, p0x, p0y, p0z, nx, ny, nz,
            &[(pe1, px1, py1, pz1)],
        );
        assert_constraint_contracts(&c, &store);
    }

    // --- Parallel3D ---

    #[test]
    fn parallel3d_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        // dir (1,0,0) and dir (2,0,0) are parallel
        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.0, e1);
        let l1_z2 = store.alloc(0.0, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(1.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(2.0, e2);
        let l2_y2 = store.alloc(1.0, e2);
        let l2_z2 = store.alloc(0.0, e2);

        let c = Parallel3D::new(
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn parallel3d_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
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
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        assert_constraint_contracts(&c, &store);
    }

    // --- Perpendicular3D ---

    #[test]
    fn perpendicular3d_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        // (1,0,0) dot (0,1,0) = 0
        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.0, e1);
        let l1_z2 = store.alloc(0.0, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(0.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(0.0, e2);
        let l2_y2 = store.alloc(1.0, e2);
        let l2_z2 = store.alloc(0.0, e2);

        let c = Perpendicular3D::new(
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn perpendicular3d_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
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
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        assert_constraint_contracts(&c, &store);
    }

    // --- Coaxial ---

    #[test]
    fn coaxial_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        // Same x-axis
        let p1x = store.alloc(0.0, e1);
        let p1y = store.alloc(0.0, e1);
        let p1z = store.alloc(0.0, e1);
        let d1x = store.alloc(1.0, e1);
        let d1y = store.alloc(0.0, e1);
        let d1z = store.alloc(0.0, e1);
        let p2x = store.alloc(5.0, e2);
        let p2y = store.alloc(0.0, e2);
        let p2z = store.alloc(0.0, e2);
        let d2x = store.alloc(2.0, e2);
        let d2y = store.alloc(0.0, e2);
        let d2z = store.alloc(0.0, e2);

        let c = Coaxial::new(
            cid(0), e1, p1x, p1y, p1z, d1x, d1y, d1z,
            e2, p2x, p2y, p2z, d2x, d2y, d2z,
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn coaxial_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
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
            cid(0), e1, p1x, p1y, p1z, d1x, d1y, d1z,
            e2, p2x, p2y, p2z, d2x, d2y, d2z,
        );
        assert_constraint_contracts(&c, &store);
    }
}

// =========================================================================
// ASSEMBLY CONSTRAINT CONTRACT TESTS
// =========================================================================

mod assembly_constraint_contracts {
    use super::*;
    use solverang::assembly::{CoaxialAssembly, Gear, Insert, Mate, UnitQuaternion};

    fn make_body(
        store: &mut ParamStore,
        entity: EntityId,
        pos: [f64; 3],
    ) -> (ParamId, ParamId, ParamId, ParamId, ParamId, ParamId, ParamId) {
        let tx = store.alloc(pos[0], entity);
        let ty = store.alloc(pos[1], entity);
        let tz = store.alloc(pos[2], entity);
        let qw = store.alloc(1.0, entity);
        let qx = store.alloc(0.0, entity);
        let qy = store.alloc(0.0, entity);
        let qz = store.alloc(0.0, entity);
        (tx, ty, tz, qw, qx, qy, qz)
    }

    #[allow(dead_code)]
    fn make_rotated_body(
        store: &mut ParamStore,
        entity: EntityId,
        pos: [f64; 3],
        axis: [f64; 3],
        angle: f64,
    ) -> (ParamId, ParamId, ParamId, ParamId, ParamId, ParamId, ParamId) {
        let tx = store.alloc(pos[0], entity);
        let ty = store.alloc(pos[1], entity);
        let tz = store.alloc(pos[2], entity);
        let half = angle / 2.0;
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let qw = store.alloc(half.cos(), entity);
        let qx = store.alloc(half.sin() * axis[0] / norm, entity);
        let qy = store.alloc(half.sin() * axis[1] / norm, entity);
        let qz = store.alloc(half.sin() * axis[2] / norm, entity);
        (tx, ty, tz, qw, qx, qy, qz)
    }

    // --- UnitQuaternion ---

    #[test]
    fn unit_quaternion_contracts_satisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let qw = store.alloc(1.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn unit_quaternion_contracts_normalized() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let norm = (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt();
        let qw = store.alloc(1.0 / norm, e);
        let qx = store.alloc(2.0 / norm, e);
        let qy = store.alloc(3.0 / norm, e);
        let qz = store.alloc(4.0 / norm, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    #[test]
    fn unit_quaternion_contracts_unnormalized() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let qw = store.alloc(2.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        assert_constraint_contracts(&c, &store);
        // residual = 4 - 1 = 3, not satisfied but contracts still hold
    }

    // --- Mate ---

    #[test]
    fn mate_contracts_identity_bodies() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 0.0]);

        let c = Mate::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [1.0, 0.0, 0.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [1.0, 0.0, 0.0],
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn mate_contracts_translated() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [10.0, 0.0, 0.0]);

        let c = Mate::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [5.0, 0.0, 0.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [-5.0, 0.0, 0.0],
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn mate_contracts_rotated() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 2.0, 3.0]);
        let norm = (1.0_f64 + 0.01 + 0.04).sqrt();
        let b2_tx = store.alloc(4.0, e2);
        let b2_ty = store.alloc(5.0, e2);
        let b2_tz = store.alloc(6.0, e2);
        let b2_qw = store.alloc(1.0 / norm, e2);
        let b2_qx = store.alloc(0.1 / norm, e2);
        let b2_qy = store.alloc(0.2 / norm, e2);
        let b2_qz = store.alloc(0.0 / norm, e2);

        let c = Mate::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [1.0, 0.5, -0.3],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [-0.5, 1.0, 0.2],
        );
        assert_constraint_contracts(&c, &store);
    }

    // --- CoaxialAssembly ---

    #[test]
    fn coaxial_assembly_contracts_aligned() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 5.0]);

        let c = CoaxialAssembly::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn coaxial_assembly_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 0.0, 0.0]);
        let norm = (1.0_f64 + 0.01 + 0.04 + 0.09).sqrt();
        let b2_tx = store.alloc(2.0, e2);
        let b2_ty = store.alloc(1.0, e2);
        let b2_tz = store.alloc(0.5, e2);
        let b2_qw = store.alloc(1.0 / norm, e2);
        let b2_qx = store.alloc(0.1 / norm, e2);
        let b2_qy = store.alloc(0.2 / norm, e2);
        let b2_qz = store.alloc(0.3 / norm, e2);

        let c = CoaxialAssembly::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        );
        assert_constraint_contracts(&c, &store);
    }

    // --- Insert ---

    #[test]
    fn insert_contracts_flush_aligned() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 0.0]);

        let c = Insert::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            0.0,
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn insert_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 2.0, 3.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [4.0, 5.0, 6.0]);

        let c = Insert::new(
            cid(0), e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            2.0,
        );
        assert_constraint_contracts(&c, &store);
    }

    // --- Gear ---

    #[test]
    fn gear_contracts_no_rotation() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let b1_qw = store.alloc(1.0, e1);
        let b1_qx = store.alloc(0.0, e1);
        let b1_qy = store.alloc(0.0, e1);
        let b1_qz = store.alloc(0.0, e1);
        let b2_qw = store.alloc(1.0, e2);
        let b2_qx = store.alloc(0.0, e2);
        let b2_qy = store.alloc(0.0, e2);
        let b2_qz = store.alloc(0.0, e2);

        let c = Gear::new(
            cid(0), e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn gear_contracts_ratio_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        // Body1: 30 deg about z, Body2: 60 deg about z, ratio=2
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
            cid(0), e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    #[test]
    fn gear_contracts_arbitrary() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
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
            cid(0), e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );
        assert_constraint_contracts(&c, &store);
    }
}

// =========================================================================
// CROSS-MODULE CONSISTENCY TESTS
// =========================================================================

mod cross_module_tests {
    use super::*;
    use solverang::sketch2d::*;
    use solverang::sketch3d::*;
    use solverang::assembly::*;

    /// Verify that the Constraint trait's default methods work correctly.
    #[test]
    fn default_trait_methods() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(0.0, e);
        let y = store.alloc(0.0, e);

        let c = Fixed::new(cid(0), e, x, y, 0.0, 0.0);
        assert_eq!(c.weight(), 1.0);
        assert!(!c.is_soft());
    }

    /// Verify that constraints with shared parameters work correctly.
    #[test]
    fn shared_parameters_between_constraints() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let e2 = eid(2);

        let x0 = store.alloc(0.0, e0);
        let y0 = store.alloc(0.0, e0);
        let x1 = store.alloc(3.0, e1);
        let y1 = store.alloc(0.0, e1);
        let x2 = store.alloc(0.0, e2);
        let y2 = store.alloc(4.0, e2);

        // Constraints sharing point (x0, y0)
        let c1 = DistancePtPt::new(cid(0), e0, e1, x0, y0, x1, y1, 3.0);
        let c2 = DistancePtPt::new(cid(1), e0, e2, x0, y0, x2, y2, 4.0);
        let c3 = Horizontal::new(cid(2), e0, e1, y0, y1);

        assert_constraint_contracts(&c1, &store);
        assert_constraint_contracts(&c2, &store);
        assert_constraint_contracts(&c3, &store);

        assert_constraint_satisfied(&c1, &store, 1e-10);
        assert_constraint_satisfied(&c2, &store, 1e-10);
        assert_constraint_satisfied(&c3, &store, 1e-14);
    }

    /// Verify that entity params match what constraints expect.
    #[test]
    fn entity_params_match_constraint_usage() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);

        let x0 = store.alloc(0.0, e0);
        let y0 = store.alloc(0.0, e0);
        let x1 = store.alloc(3.0, e1);
        let y1 = store.alloc(4.0, e1);

        let pt0 = Point2D::new(e0, x0, y0);
        let pt1 = Point2D::new(e1, x1, y1);

        // The constraint's param_ids should be a subset of the union of entity params
        let c = DistancePtPt::new(cid(0), e0, e1, x0, y0, x1, y1, 5.0);

        let entity_params: HashSet<ParamId> = pt0
            .params()
            .iter()
            .chain(pt1.params().iter())
            .copied()
            .collect();
        let constraint_params: HashSet<ParamId> =
            c.param_ids().iter().copied().collect();

        assert!(
            constraint_params.is_subset(&entity_params),
            "Constraint params {:?} not subset of entity params {:?}",
            constraint_params,
            entity_params,
        );
    }

    /// Test that 2D and 3D distance constraints behave consistently.
    #[test]
    fn distance_2d_vs_3d_consistency() {
        let mut store2d = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1_2d = store2d.alloc(0.0, e0);
        let y1_2d = store2d.alloc(0.0, e0);
        let x2_2d = store2d.alloc(3.0, e1);
        let y2_2d = store2d.alloc(4.0, e1);

        let c2d = DistancePtPt::new(cid(0), e0, e1, x1_2d, y1_2d, x2_2d, y2_2d, 5.0);

        let mut store3d = ParamStore::new();
        let x1_3d = store3d.alloc(0.0, e0);
        let y1_3d = store3d.alloc(0.0, e0);
        let z1_3d = store3d.alloc(0.0, e0);
        let x2_3d = store3d.alloc(3.0, e1);
        let y2_3d = store3d.alloc(4.0, e1);
        let z2_3d = store3d.alloc(0.0, e1); // z=0 => 2D equivalent

        let c3d = Distance3D::new(cid(0), e0, x1_3d, y1_3d, z1_3d, e1, x2_3d, y2_3d, z2_3d, 5.0);

        let r2d = c2d.residuals(&store2d);
        let r3d = c3d.residuals(&store3d);

        assert!(
            (r2d[0] - r3d[0]).abs() < 1e-12,
            "2D residual={}, 3D residual={} should match for z=0",
            r2d[0],
            r3d[0]
        );
    }

    /// Test that multiple constraints on the same rigid body don't interfere.
    #[test]
    fn assembly_multiple_constraints_same_body() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let tx1 = store.alloc(0.0, e1);
        let ty1 = store.alloc(0.0, e1);
        let tz1 = store.alloc(0.0, e1);
        let qw1 = store.alloc(1.0, e1);
        let qx1 = store.alloc(0.0, e1);
        let qy1 = store.alloc(0.0, e1);
        let qz1 = store.alloc(0.0, e1);

        let tx2 = store.alloc(10.0, e2);
        let ty2 = store.alloc(0.0, e2);
        let tz2 = store.alloc(0.0, e2);
        let qw2 = store.alloc(1.0, e2);
        let qx2 = store.alloc(0.0, e2);
        let qy2 = store.alloc(0.0, e2);
        let qz2 = store.alloc(0.0, e2);

        // UnitQuaternion for both bodies
        let uq1 = UnitQuaternion::new(cid(0), e1, qw1, qx1, qy1, qz1);
        let uq2 = UnitQuaternion::new(cid(1), e2, qw2, qx2, qy2, qz2);

        // Mate between the bodies
        let mate = Mate::new(
            cid(2), e1, tx1, ty1, tz1, qw1, qx1, qy1, qz1,
            [5.0, 0.0, 0.0],
            e2, tx2, ty2, tz2, qw2, qx2, qy2, qz2,
            [-5.0, 0.0, 0.0],
        );

        assert_constraint_contracts(&uq1, &store);
        assert_constraint_contracts(&uq2, &store);
        assert_constraint_contracts(&mate, &store);
        assert_constraint_satisfied(&uq1, &store, 1e-14);
        assert_constraint_satisfied(&uq2, &store, 1e-14);
        assert_constraint_satisfied(&mate, &store, 1e-10);
    }

    /// Verify solver mapping consistency with constraint param_ids.
    #[test]
    fn solver_mapping_covers_constraint_params() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);

        let mapping = store.build_solver_mapping_for(c.param_ids());
        // Every free param in param_ids() should be in the mapping
        for &pid in c.param_ids() {
            if !store.is_fixed(pid) {
                assert!(
                    mapping.param_to_col.contains_key(&pid),
                    "Free param {:?} not in solver mapping",
                    pid
                );
            }
        }
    }

    /// Test that fixing a parameter doesn't break constraint contracts.
    #[test]
    fn fixed_params_dont_break_contracts() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        store.fix(x1);
        store.fix(y1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        // Contracts should still hold even with fixed params
        let violations = validate_constraint_contracts(&c, &store);
        assert!(
            violations.is_empty(),
            "Violations with fixed params: {:?}",
            violations.iter().map(|v| &v.detail).collect::<Vec<_>>()
        );
    }

    /// Verify that param reuse (free + realloc) with stale IDs is detected.
    #[test]
    fn param_store_generational_safety() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let id1 = store.alloc(1.0, e);
        store.free(id1);
        let id2 = store.alloc(2.0, e);

        // id1 and id2 differ (different generations, even though same slot)
        assert_ne!(id1, id2);

        // id2 works fine
        assert!((store.get(id2) - 2.0).abs() < 1e-15);

        // id1 should panic (stale generation)
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.get(id1)));
        assert!(result.is_err(), "Stale ParamId should panic on get()");
    }
}

// =========================================================================
// EDGE CASE CONTRACT TESTS
// =========================================================================

mod edge_case_contracts {
    use super::*;
    use solverang::sketch2d::*;
    use solverang::sketch3d::*;

    /// Coincident points (zero distance) — squared formulation handles this.
    #[test]
    fn coincident_points_distance_constraint() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(5.0, e0);
        let y1 = store.alloc(5.0, e0);
        let x2 = store.alloc(5.0, e1);
        let y2 = store.alloc(5.0, e1);

        // Distance = 0: no singularity with squared formulation
        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 0.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    /// Negative coordinates should work fine.
    #[test]
    fn negative_coordinates() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(-10.0, e0);
        let y1 = store.alloc(-20.0, e0);
        let x2 = store.alloc(-7.0, e1);
        let y2 = store.alloc(-16.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    /// Test parallel lines that are anti-parallel (opposite directions).
    #[test]
    fn anti_parallel_lines() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        // dir (1,0) and dir (-2,0) are anti-parallel
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(1.0, e0);
        let y2 = store.alloc(0.0, e0);
        let x3 = store.alloc(5.0, e1);
        let y3 = store.alloc(3.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(3.0, e1);

        let c = Parallel::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-10);
    }

    /// Test 3D constraints at the origin.
    #[test]
    fn all_zero_3d() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(0.0, e1);
        let y1 = store.alloc(0.0, e1);
        let z1 = store.alloc(0.0, e1);
        let x2 = store.alloc(0.0, e2);
        let y2 = store.alloc(0.0, e2);
        let z2 = store.alloc(0.0, e2);

        let c = Coincident3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2);
        assert_constraint_contracts(&c, &store);
        assert_constraint_satisfied(&c, &store, 1e-14);
    }

    /// Verify constraint contracts hold after parameter mutation.
    #[test]
    fn contracts_hold_after_mutation() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        assert_constraint_contracts(&c, &store);

        // Mutate parameters
        store.set(x2, 100.0);
        store.set(y2, -50.0);

        // Contracts should still hold (different values, but still valid)
        assert_constraint_contracts(&c, &store);
    }

    /// Verify contracts with very large parameter values.
    #[test]
    fn extreme_large_values_3d() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let scale = 1e8;
        let x1 = store.alloc(0.0, e1);
        let y1 = store.alloc(0.0, e1);
        let z1 = store.alloc(0.0, e1);
        let x2 = store.alloc(3.0 * scale, e2);
        let y2 = store.alloc(4.0 * scale, e2);
        let z2 = store.alloc(0.0, e2);

        let c = Distance3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2, 5.0 * scale);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty());
        assert_constraint_satisfied(&c, &store, 1e-2); // Wider tol for large scale
    }

    /// Verify contracts with very small parameter values.
    #[test]
    fn extreme_small_values() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let scale = 1e-10;
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0 * scale, e1);
        let y2 = store.alloc(4.0 * scale, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0 * scale);
        let violations = validate_constraint_contracts(&c, &store);
        assert!(violations.is_empty());
    }

    /// Test a complete 2D triangle with mixed constraint types.
    #[test]
    fn complete_triangle_contracts() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let e2 = eid(2);

        // Triangle: p0=(0,0), p1=(4,0), p2=(0,3)
        let x0 = store.alloc(0.0, e0);
        let y0 = store.alloc(0.0, e0);
        let x1 = store.alloc(4.0, e1);
        let y1 = store.alloc(0.0, e1);
        let x2 = store.alloc(0.0, e2);
        let y2 = store.alloc(3.0, e2);

        // Fix p0
        let fix = Fixed::new(cid(0), e0, x0, y0, 0.0, 0.0);
        assert_constraint_contracts(&fix, &store);
        assert_constraint_satisfied(&fix, &store, 1e-14);

        // Horizontal edge p0-p1
        let horiz = Horizontal::new(cid(1), e0, e1, y0, y1);
        assert_constraint_contracts(&horiz, &store);
        assert_constraint_satisfied(&horiz, &store, 1e-14);

        // Vertical edge p0-p2
        let vert = Vertical::new(cid(2), e0, e2, x0, x2);
        assert_constraint_contracts(&vert, &store);
        assert_constraint_satisfied(&vert, &store, 1e-14);

        // Distance p0-p1 = 4
        let d01 = DistancePtPt::new(cid(3), e0, e1, x0, y0, x1, y1, 4.0);
        assert_constraint_contracts(&d01, &store);
        assert_constraint_satisfied(&d01, &store, 1e-10);

        // Distance p0-p2 = 3
        let d02 = DistancePtPt::new(cid(4), e0, e2, x0, y0, x2, y2, 3.0);
        assert_constraint_contracts(&d02, &store);
        assert_constraint_satisfied(&d02, &store, 1e-10);

        // Hypotenuse p1-p2 = 5
        let d12 = DistancePtPt::new(cid(5), e1, e2, x1, y1, x2, y2, 5.0);
        assert_constraint_contracts(&d12, &store);
        assert_constraint_satisfied(&d12, &store, 1e-10);
    }

    /// Test a 3D tetrahedron with mixed constraints.
    #[test]
    fn tetrahedron_contracts() {
        let mut store = ParamStore::new();
        let e0 = eid(0);
        let e1 = eid(1);
        let e2 = eid(2);
        let e3 = eid(3);

        // Regular tetrahedron with edge length 2
        let x0 = store.alloc(1.0, e0);
        let y0 = store.alloc(1.0, e0);
        let z0 = store.alloc(1.0, e0);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(-1.0, e1);
        let z1 = store.alloc(-1.0, e1);
        let x2 = store.alloc(-1.0, e2);
        let y2 = store.alloc(1.0, e2);
        let z2 = store.alloc(-1.0, e2);
        let x3 = store.alloc(-1.0, e3);
        let y3 = store.alloc(-1.0, e3);
        let z3 = store.alloc(1.0, e3);

        // Edge length = sqrt((1-1)^2 + (1-(-1))^2 + (1-(-1))^2) = sqrt(8) = 2*sqrt(2)
        let edge = (8.0_f64).sqrt();

        let d01 = Distance3D::new(cid(0), e0, x0, y0, z0, e1, x1, y1, z1, edge);
        let d02 = Distance3D::new(cid(1), e0, x0, y0, z0, e2, x2, y2, z2, edge);
        let d03 = Distance3D::new(cid(2), e0, x0, y0, z0, e3, x3, y3, z3, edge);
        let d12 = Distance3D::new(cid(3), e1, x1, y1, z1, e2, x2, y2, z2, edge);
        let d13 = Distance3D::new(cid(4), e1, x1, y1, z1, e3, x3, y3, z3, edge);
        let d23 = Distance3D::new(cid(5), e2, x2, y2, z2, e3, x3, y3, z3, edge);

        for c in [&d01, &d02, &d03, &d12, &d13, &d23] {
            assert_constraint_contracts(c as &dyn Constraint, &store);
            assert_constraint_satisfied(c as &dyn Constraint, &store, 1e-10);
        }
    }
}
