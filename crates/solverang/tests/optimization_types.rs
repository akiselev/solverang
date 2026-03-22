//! Tests for optimization foundation types (M1).

use solverang::optimization::multiplier_store::{MultiplierId, MultiplierStore};
use solverang::optimization::{
    ObjectiveId, OptimizationConfig, OptimizationResult, OptimizationStatus,
};
use solverang::ConstraintId;

#[test]
fn objective_id_equality() {
    let a = ObjectiveId::new(0, 0);
    let b = ObjectiveId::new(0, 0);
    let c = ObjectiveId::new(0, 1); // different generation
    let d = ObjectiveId::new(1, 0); // different index

    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
}

#[test]
fn objective_id_debug() {
    let id = ObjectiveId::new(5, 2);
    assert_eq!(format!("{:?}", id), "Objective(5g2)");
}

#[test]
fn objective_id_hashable() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ObjectiveId::new(0, 0));
    set.insert(ObjectiveId::new(1, 0));
    set.insert(ObjectiveId::new(0, 0)); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn multiplier_id_creation() {
    let cid = ConstraintId::new(3, 1);
    let mid = MultiplierId::new(cid, 0);
    assert_eq!(mid.constraint_id, cid);
    assert_eq!(mid.equation_row, 0);
}

#[test]
fn multiplier_store_set_get() {
    let mut store = MultiplierStore::new();
    assert!(store.is_empty());

    let cid = ConstraintId::new(0, 0);
    let mid = MultiplierId::new(cid, 0);

    store.set(mid, 3.14);
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(mid), Some(3.14));
}

#[test]
fn multiplier_store_get_nonexistent() {
    let store = MultiplierStore::new();
    let cid = ConstraintId::new(99, 0);
    let mid = MultiplierId::new(cid, 0);
    assert_eq!(store.get(mid), None);
}

#[test]
fn multiplier_store_clear() {
    let mut store = MultiplierStore::new();
    let cid = ConstraintId::new(0, 0);
    store.set(MultiplierId::new(cid, 0), 1.0);
    store.set(MultiplierId::new(cid, 1), 2.0);
    assert_eq!(store.len(), 2);

    store.clear();
    assert!(store.is_empty());
    assert_eq!(store.get(MultiplierId::new(cid, 0)), None);
}

#[test]
fn multiplier_store_lambda_for_constraint() {
    let mut store = MultiplierStore::new();
    let c0 = ConstraintId::new(0, 0);
    let c1 = ConstraintId::new(1, 0);

    // Constraint 0 has 3 equation rows
    store.set(MultiplierId::new(c0, 0), 1.0);
    store.set(MultiplierId::new(c0, 1), 2.0);
    store.set(MultiplierId::new(c0, 2), 3.0);

    // Constraint 1 has 1 equation row
    store.set(MultiplierId::new(c1, 0), -5.0);

    let lambda0 = store.lambda_for_constraint(c0).unwrap();
    assert_eq!(lambda0, vec![1.0, 2.0, 3.0]);

    let lambda1 = store.lambda_for_constraint(c1).unwrap();
    assert_eq!(lambda1, vec![-5.0]);

    // Non-existent constraint
    let c99 = ConstraintId::new(99, 0);
    assert!(store.lambda_for_constraint(c99).is_none());
}

#[test]
fn optimization_config_defaults() {
    let config = OptimizationConfig::default();
    assert_eq!(
        config.algorithm,
        solverang::OptimizationAlgorithm::Auto
    );
    assert!(config.outer_tolerance > 0.0);
    assert!(config.inner_tolerance > 0.0);
    assert!(config.rho_init > 0.0);
    assert!(config.lbfgs_memory > 0);
}

#[test]
fn optimization_result_status_variants() {
    // Test status enum variants (not_implemented is pub(crate))
    let result = OptimizationResult {
        objective_value: f64::NAN,
        status: OptimizationStatus::NotImplemented,
        outer_iterations: 0,
        inner_iterations: 0,
        kkt_residual: solverang::KktResidual {
            primal: f64::INFINITY,
            dual: f64::INFINITY,
            complementarity: f64::INFINITY,
        },
        multipliers: MultiplierStore::new(),
        constraint_violations: Vec::new(),
        duration: std::time::Duration::ZERO,
    };
    assert_eq!(result.status, OptimizationStatus::NotImplemented);
    assert!(!result.status.is_converged());
    assert!(result.objective_value.is_nan());
    assert!(result.multipliers.is_empty());
}

#[test]
fn optimization_status_is_converged() {
    assert!(OptimizationStatus::Converged.is_converged());
    assert!(!OptimizationStatus::MaxIterationsReached.is_converged());
    assert!(!OptimizationStatus::Infeasible.is_converged());
    assert!(!OptimizationStatus::Diverged.is_converged());
    assert!(!OptimizationStatus::NotImplemented.is_converged());
}

#[test]
fn kkt_residual_tolerance_check() {
    let kkt = solverang::KktResidual {
        primal: 1e-8,
        dual: 1e-7,
        complementarity: 1e-9,
    };
    assert!(kkt.is_within_tolerance(1e-6, 1e-6, 1e-6));
    assert!(!kkt.is_within_tolerance(1e-9, 1e-6, 1e-6)); // primal too tight
}
