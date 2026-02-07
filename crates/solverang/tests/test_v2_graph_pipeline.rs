//! Integration tests for the graph analysis and solve pipeline modules.
//!
//! Tests cover:
//! - Union-Find incremental component tracking
//! - IncrementalGraph entity/constraint management and dirty tracking
//! - Dulmage-Mendelsohn decomposition (under/well/over-determined classification)
//! - Structural and numerical redundancy detection
//! - DOF diagnostics
//! - Pipeline solver selection heuristics
//! - Pipeline component extraction and Problem trait compliance
//! - Pipeline warm start regularization
//! - Full pipeline solve orchestration

#![cfg(feature = "geometry")]

use solverang::geometry::constraint::{Constraint, Nonlinearity};
use solverang::geometry::params::{ConstraintId, EntityId};
use solverang::graph::diagnostics::{ConstraintStatus, DOFAnalysis};
use solverang::graph::dm::dm_decompose;
use solverang::graph::incremental::{ConstraintMeta, EntityMeta, IncrementalGraph};
use solverang::graph::redundancy::{detect_numerical_redundancy, detect_structural_redundancy};
use solverang::graph::union_find::{ComponentId, IncrementalUnionFind};
use solverang::pipeline::extract::extract_component;
use solverang::pipeline::select::{estimate_sparsity, select_solver, SolverSelection};
use solverang::pipeline::solve::{solve_dirty_components, ComponentSolveResult, SolvePipeline};
use solverang::pipeline::warm_start::{
    degrees_of_freedom, is_underconstrained, regularize_jacobian, regularize_residuals,
    DEFAULT_LAMBDA,
};
use solverang::Problem;

// ============================================================================
// Helper: Mock constraints
// ============================================================================

/// A linear constraint: sum of deps params = target.
/// Residual: sum(params[deps]) - target
/// Jacobian: all partial derivatives = 1.0
struct MockLinearConstraint {
    id: ConstraintId,
    deps: Vec<usize>,
    target: f64,
}

impl Constraint for MockLinearConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "MockLinearConstraint"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let sum: f64 = self.deps.iter().map(|&i| params[i]).sum();
        vec![sum - self.target]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        self.deps.iter().map(|&i| (0, i, 1.0)).collect()
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

/// A quadratic (nonlinear) constraint: sum of (params[deps])^2 = target.
/// Residual: sum(params[i]^2) - target
/// Jacobian: d/d(params[i]) = 2 * params[i]
struct MockQuadraticConstraint {
    id: ConstraintId,
    deps: Vec<usize>,
    target: f64,
}

impl Constraint for MockQuadraticConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "MockQuadraticConstraint"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let sum: f64 = self.deps.iter().map(|&i| params[i] * params[i]).sum();
        vec![sum - self.target]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        self.deps
            .iter()
            .map(|&i| (0, i, 2.0 * params[i]))
            .collect()
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

// Helper: create an EntityMeta
fn make_entity(id: usize, param_count: usize) -> EntityMeta {
    EntityMeta {
        id: EntityId(id),
        param_count,
        fixed_param_count: 0,
    }
}

fn make_entity_fixed(id: usize, param_count: usize, fixed_count: usize) -> EntityMeta {
    EntityMeta {
        id: EntityId(id),
        param_count,
        fixed_param_count: fixed_count,
    }
}

// Helper: create a ConstraintMeta
fn make_constraint(id: usize, entity_deps: Vec<usize>, equation_count: usize) -> ConstraintMeta {
    ConstraintMeta {
        id: ConstraintId(id),
        equation_count,
        entity_deps: entity_deps.into_iter().map(EntityId).collect(),
        param_deps: vec![],
    }
}

// ============================================================================
// Graph Union-Find Integration (5+ tests)
// ============================================================================

#[test]
fn test_union_find_geometric_components() {
    // Two separate triangles: {0,1,2} and {3,4,5}
    let mut uf = IncrementalUnionFind::new(6);

    // Triangle 1: 0-1, 1-2, 2-0
    uf.add_edge(0, 1);
    uf.add_edge(1, 2);
    uf.add_edge(2, 0);

    // Triangle 2: 3-4, 4-5, 5-3
    uf.add_edge(3, 4);
    uf.add_edge(4, 5);
    uf.add_edge(5, 3);

    assert_eq!(uf.component_count(), 2);

    // Nodes within the same triangle are connected
    assert!(uf.connected(0, 1));
    assert!(uf.connected(0, 2));
    assert!(uf.connected(1, 2));

    assert!(uf.connected(3, 4));
    assert!(uf.connected(3, 5));
    assert!(uf.connected(4, 5));

    // Nodes across triangles are NOT connected
    assert!(!uf.connected(0, 3));
    assert!(!uf.connected(1, 4));
    assert!(!uf.connected(2, 5));

    // Check components map
    let comps = uf.components();
    assert_eq!(comps.len(), 2);
    let mut sizes: Vec<usize> = comps.values().map(|v| v.len()).collect();
    sizes.sort();
    assert_eq!(sizes, vec![3, 3]);
}

#[test]
fn test_union_find_merge_components() {
    // Start with 3 components: {0,1}, {2,3}, {4,5}
    let mut uf = IncrementalUnionFind::new(6);
    uf.add_edge(0, 1);
    uf.add_edge(2, 3);
    uf.add_edge(4, 5);
    assert_eq!(uf.component_count(), 3);

    // Merge the first two by connecting node 1 to node 2
    uf.add_edge(1, 2);
    assert_eq!(uf.component_count(), 2);

    // {0,1,2,3} should all be connected now
    assert!(uf.connected(0, 3));
    assert!(uf.connected(1, 2));

    // {4,5} is still separate
    assert!(!uf.connected(0, 4));
}

#[test]
fn test_union_find_rebuild_after_removal() {
    // Chain: 0-1-2-3
    let mut uf = IncrementalUnionFind::new(4);
    uf.add_edge(0, 1);
    uf.add_edge(1, 2);
    uf.add_edge(2, 3);
    assert_eq!(uf.component_count(), 1);

    // Remove the middle edge (1,2) — splits into {0,1} and {2,3}
    uf.remove_edge(1, 2);
    assert_eq!(uf.component_count(), 2);

    assert!(uf.connected(0, 1));
    assert!(uf.connected(2, 3));
    assert!(!uf.connected(0, 2));
    assert!(!uf.connected(1, 3));
}

#[test]
fn test_union_find_single_node() {
    let mut uf = IncrementalUnionFind::new(1);
    assert_eq!(uf.component_count(), 1);
    assert_eq!(uf.find(0), 0);
    assert_eq!(uf.edge_count(), 0);

    let comps = uf.components();
    assert_eq!(comps.len(), 1);
    assert_eq!(comps.values().next().unwrap().len(), 1);
}

#[test]
fn test_union_find_all_connected() {
    // Connect all 5 nodes into a single component
    let mut uf = IncrementalUnionFind::new(5);
    uf.add_edge(0, 1);
    uf.add_edge(1, 2);
    uf.add_edge(2, 3);
    uf.add_edge(3, 4);

    assert_eq!(uf.component_count(), 1);

    // All pairs are connected
    for i in 0..5 {
        for j in 0..5 {
            assert!(uf.connected(i, j));
        }
    }
}

#[test]
fn test_union_find_cycle_remove_preserves_connectivity() {
    // Create a cycle: 0-1-2-0, so removing one edge keeps it connected
    let mut uf = IncrementalUnionFind::new(3);
    uf.add_edge(0, 1);
    uf.add_edge(1, 2);
    uf.add_edge(2, 0);
    assert_eq!(uf.component_count(), 1);

    // Removing one edge from a cycle should NOT split the component
    uf.remove_edge(0, 1);
    assert_eq!(uf.component_count(), 1);
    assert!(uf.connected(0, 1)); // Still connected via 0-2-1
}

// ============================================================================
// Incremental Graph Integration (5+ tests)
// ============================================================================

#[test]
fn test_incremental_add_entities_constraints() {
    let mut graph = IncrementalGraph::new();

    // Add three entities (points with 2 params each)
    graph.add_entity(make_entity(0, 2));
    graph.add_entity(make_entity(1, 2));
    graph.add_entity(make_entity(2, 2));

    assert_eq!(graph.entity_count(), 3);
    assert_eq!(graph.version(), 3); // Three add_entity calls

    // Take initial dirty set (all entities are new => dirty)
    let dirty = graph.take_dirty();
    assert!(!dirty.is_empty());

    // After take, dirty is cleared
    assert!(graph.take_dirty().is_empty());

    // Add a constraint connecting entity 0 and 1
    graph.add_constraint(make_constraint(0, vec![0, 1], 1));

    assert_eq!(graph.constraint_count(), 1);
    assert_eq!(graph.component_count(), 2); // {0,1} and {2}

    // Dirty should include the affected component
    let dirty2 = graph.take_dirty();
    assert!(!dirty2.is_empty());
}

#[test]
fn test_incremental_remove_constraint() {
    let mut graph = IncrementalGraph::new();
    graph.add_entity(make_entity(0, 2));
    graph.add_entity(make_entity(1, 2));
    graph.add_constraint(make_constraint(0, vec![0, 1], 1));

    assert_eq!(graph.component_count(), 1);

    // Remove the constraint
    graph.remove_constraint(ConstraintId(0));

    assert_eq!(graph.component_count(), 2); // Split back to two components
    assert_eq!(graph.constraint_count(), 0);

    // Dirty should reflect the split
    let dirty = graph.take_dirty();
    assert!(!dirty.is_empty());
}

#[test]
fn test_incremental_dirty_detection() {
    let mut graph = IncrementalGraph::new();
    graph.add_entity(make_entity(0, 2));
    graph.add_entity(make_entity(1, 2));
    graph.add_entity(make_entity(2, 2));

    // Component 1: entities 0 and 1
    graph.add_constraint(make_constraint(0, vec![0, 1], 1));
    // Entity 2 stays isolated

    // Clear dirty state
    graph.take_dirty();

    // Modify only one component by marking entity 0 dirty
    graph.mark_dirty_by_entity(EntityId(0));

    let dirty = graph.take_dirty();
    assert_eq!(dirty.len(), 1); // Only the component containing entity 0

    // Verify entity 2's component was NOT marked dirty
    // (take_dirty already drained, so nothing should remain)
    assert!(graph.take_dirty().is_empty());
}

#[test]
fn test_incremental_entity_metadata() {
    let mut graph = IncrementalGraph::new();
    let meta = EntityMeta {
        id: EntityId(42),
        param_count: 5,
        fixed_param_count: 2,
    };
    graph.add_entity(meta);

    let retrieved = graph.get_entity(EntityId(42)).unwrap();
    assert_eq!(retrieved.param_count, 5);
    assert_eq!(retrieved.fixed_param_count, 2);
    assert_eq!(retrieved.id, EntityId(42));
}

#[test]
fn test_incremental_constraint_metadata() {
    let mut graph = IncrementalGraph::new();
    graph.add_entity(make_entity(0, 2));
    graph.add_entity(make_entity(1, 2));

    let cmeta = ConstraintMeta {
        id: ConstraintId(99),
        equation_count: 3,
        entity_deps: vec![EntityId(0), EntityId(1)],
        param_deps: vec![0, 1, 2, 3],
    };
    graph.add_constraint(cmeta);

    let retrieved = graph.get_constraint(ConstraintId(99)).unwrap();
    assert_eq!(retrieved.equation_count, 3);
    assert_eq!(retrieved.entity_deps.len(), 2);
    assert_eq!(retrieved.param_deps, vec![0, 1, 2, 3]);
}

#[test]
fn test_incremental_component_dof() {
    let mut graph = IncrementalGraph::new();

    // Two 2D points (4 params total), one distance constraint (1 equation)
    graph.add_entity(make_entity(0, 2));
    graph.add_entity(make_entity(1, 2));
    graph.add_constraint(make_constraint(0, vec![0, 1], 1));

    let comp = graph.component_of(EntityId(0)).unwrap();
    let dof = graph.component_dof(comp);
    // 4 free params - 1 equation = 3 DOF
    assert_eq!(dof, 3);
}

#[test]
fn test_incremental_component_dof_with_fixed() {
    let mut graph = IncrementalGraph::new();

    // Entity 0: 2 params, both fixed
    graph.add_entity(make_entity_fixed(0, 2, 2));
    // Entity 1: 2 params, free
    graph.add_entity(make_entity(1, 2));
    graph.add_constraint(make_constraint(0, vec![0, 1], 1));

    let comp = graph.component_of(EntityId(0)).unwrap();
    let dof = graph.component_dof(comp);
    // (4 total - 2 fixed) - 1 equation = 1 DOF
    assert_eq!(dof, 1);
}

// ============================================================================
// DM Decomposition Integration (5+ tests)
// ============================================================================

#[test]
fn test_dm_well_constrained() {
    // 3 constraints, 3 variables, each constraint touches exactly one variable
    // This is a perfect matching = well-constrained
    let edges = vec![(0, 0), (1, 1), (2, 2)];
    let dm = dm_decompose(3, 3, &edges);

    assert!(dm.under.is_empty());
    assert!(dm.over.is_empty());
    assert!(!dm.well.is_empty());

    let total_well_eqs: usize = dm.well.iter().map(|b| b.equation_count()).sum();
    let total_well_vars: usize = dm.well.iter().map(|b| b.variable_count()).sum();
    assert_eq!(total_well_eqs, 3);
    assert_eq!(total_well_vars, 3);
    assert!(dm.unmatched_constraints.is_empty());
}

#[test]
fn test_dm_underconstrained() {
    // 2 constraints, 4 variables => under-constrained
    let edges = vec![(0, 0), (0, 1), (1, 2), (1, 3)];
    let dm = dm_decompose(2, 4, &edges);

    assert!(!dm.under.is_empty());
    // Under-determined block should have excess variables
    assert!(dm.under.variable_count() > dm.under.equation_count());
}

#[test]
fn test_dm_overconstrained() {
    // 4 constraints, 2 variables => over-constrained
    let edges = vec![
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ];
    let dm = dm_decompose(4, 2, &edges);

    assert!(!dm.over.is_empty());
    // Over-determined block should have excess constraints
    assert!(dm.over.equation_count() > dm.over.variable_count());
    assert!(!dm.unmatched_constraints.is_empty());
}

#[test]
fn test_dm_mixed() {
    // Construct a system with:
    // - Variables 0,1 well-constrained by constraints 0,1
    // - Variable 2 under-constrained (no constraint)
    // - Constraint 2 redundant (also touches var 0,1) => over
    //
    // 3 constraints, 3 variables
    // c0 -> v0
    // c1 -> v1
    // c2 -> v0, v1 (redundant)
    let edges = vec![(0, 0), (1, 1), (2, 0), (2, 1)];
    let dm = dm_decompose(3, 3, &edges);

    // Variable 2 has no constraint edges => under-determined
    assert!(
        dm.under.variable_count() > 0 || dm.over.equation_count() > 0,
        "Mixed system should have under- or over-determined parts"
    );

    // The unmatched constraint (c2) makes the over-determined part nonempty
    assert!(!dm.unmatched_constraints.is_empty());
}

#[test]
fn test_dm_disconnected() {
    // Two independent well-constrained blocks
    // Block 1: c0 <-> v0, c1 <-> v1
    // Block 2: c2 <-> v2, c3 <-> v3
    let edges = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
    let dm = dm_decompose(4, 4, &edges);

    assert!(dm.under.is_empty());
    assert!(dm.over.is_empty());
    assert!(!dm.well.is_empty());

    let total_well_eqs: usize = dm.well.iter().map(|b| b.equation_count()).sum();
    let total_well_vars: usize = dm.well.iter().map(|b| b.variable_count()).sum();
    assert_eq!(total_well_eqs, 4);
    assert_eq!(total_well_vars, 4);
}

#[test]
fn test_dm_larger_well_constrained() {
    // 6x6 system with a dense block — all constraints touch multiple variables
    let edges = vec![
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 4),
        (4, 4),
        (4, 5),
        (5, 5),
        (5, 0),
    ];
    let dm = dm_decompose(6, 6, &edges);

    // Should be fully well-constrained
    assert!(dm.under.is_empty());
    assert!(dm.over.is_empty());
    assert!(dm.unmatched_constraints.is_empty());
}

// ============================================================================
// Redundancy Detection Integration (3+ tests)
// ============================================================================

#[test]
fn test_structural_redundancy() {
    // 3 constraints, 2 variables => one structurally redundant
    let ids = vec![ConstraintId(0), ConstraintId(1), ConstraintId(2)];
    let edges = vec![
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1), // c2 is redundant
    ];

    let redundant = detect_structural_redundancy(&ids, 3, 2, &edges);
    assert!(
        !redundant.is_empty(),
        "Should detect at least one structurally redundant constraint"
    );
}

#[test]
fn test_numerical_redundancy() {
    // Two constraints with identical Jacobian rows (numerically equivalent)
    let ids = vec![ConstraintId(0), ConstraintId(1)];
    let jacobian_entries = vec![
        (0, 0, 1.0),
        (0, 1, 2.0), // Row 0: [1, 2]
        (1, 0, 1.0),
        (1, 1, 2.0), // Row 1: [1, 2] — duplicate!
    ];

    let redundant = detect_numerical_redundancy(&jacobian_entries, 2, 2, &ids, 1e-10);
    assert!(
        !redundant.is_empty(),
        "Duplicate Jacobian rows should be numerically redundant"
    );
    // The later row (index 1) should be flagged
    assert!(redundant.contains(&ConstraintId(1)));
}

#[test]
fn test_no_redundancy() {
    // Well-constrained: 2 constraints, 2 variables, independent
    let ids = vec![ConstraintId(0), ConstraintId(1)];

    // Structural check
    let edges = vec![(0, 0), (1, 1)];
    let struct_redundant = detect_structural_redundancy(&ids, 2, 2, &edges);
    assert!(
        struct_redundant.is_empty(),
        "No structural redundancy in a perfect matching"
    );

    // Numerical check (identity Jacobian)
    let jacobian_entries = vec![(0, 0, 1.0), (1, 1, 1.0)];
    let num_redundant = detect_numerical_redundancy(&jacobian_entries, 2, 2, &ids, 1e-10);
    assert!(
        num_redundant.is_empty(),
        "Identity Jacobian rows are independent"
    );
}

#[test]
fn test_numerical_redundancy_linear_combination() {
    // Row 2 is a linear combination of rows 0 and 1
    let ids = vec![ConstraintId(0), ConstraintId(1), ConstraintId(2)];
    let jacobian_entries = vec![
        (0, 0, 1.0),
        (0, 1, 0.0),
        (1, 0, 0.0),
        (1, 1, 1.0),
        (2, 0, 2.0),
        (2, 1, 3.0), // 2*row0 + 3*row1
    ];

    let redundant = detect_numerical_redundancy(&jacobian_entries, 3, 2, &ids, 1e-10);
    assert!(redundant.contains(&ConstraintId(2)));
}

// ============================================================================
// DOF Diagnostics (additional tests beyond unit tests)
// ============================================================================

#[test]
fn test_dof_analysis_well_constrained() {
    let analysis = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
    assert!(analysis.is_well_constrained());
    assert_eq!(analysis.dof, 0);
    assert_eq!(analysis.free_variables, 2);
    assert_eq!(analysis.status, ConstraintStatus::WellConstrained);
}

#[test]
fn test_dof_analysis_under_constrained() {
    let analysis = DOFAnalysis::new(ComponentId(0), 6, 0, 2);
    assert!(analysis.is_under_constrained());
    assert_eq!(analysis.dof, 4);
    assert_eq!(analysis.free_variables, 6);
}

#[test]
fn test_dof_analysis_over_constrained() {
    let analysis = DOFAnalysis::new(ComponentId(0), 4, 0, 8);
    assert!(analysis.is_over_constrained());
    assert_eq!(analysis.dof, -4);
}

#[test]
fn test_dof_analysis_from_free() {
    let analysis = DOFAnalysis::from_free(ComponentId(1), 5, 5);
    assert!(analysis.is_well_constrained());
    assert_eq!(analysis.dof, 0);
    assert_eq!(analysis.free_variables, 5);
    assert_eq!(analysis.total_equations, 5);
}

// ============================================================================
// Pipeline Solver Selection (5+ tests)
// ============================================================================

#[test]
fn test_select_nr_for_linear() {
    // Linear constraints should always select Newton-Raphson
    let selection = select_solver(10, 10, Nonlinearity::Linear, 0.5);
    assert_eq!(selection, SolverSelection::NewtonRaphson);
}

#[test]
fn test_select_robust_for_nonlinear() {
    // Moderate nonlinearity, medium-sized system => Robust
    let selection = select_solver(20, 20, Nonlinearity::Moderate, 0.5);
    assert_eq!(selection, SolverSelection::Robust);
}

#[test]
fn test_select_auto_for_small() {
    // Trivial system (<=2 variables) => NewtonRaphson regardless of nonlinearity
    let selection = select_solver(2, 2, Nonlinearity::High, 0.5);
    assert_eq!(selection, SolverSelection::NewtonRaphson);

    let selection = select_solver(1, 1, Nonlinearity::Moderate, 0.5);
    assert_eq!(selection, SolverSelection::NewtonRaphson);
}

#[test]
fn test_select_sparse_for_large() {
    // Large variable count (>100) with sparse Jacobian (<10% fill) => Sparse
    let selection = select_solver(200, 200, Nonlinearity::Moderate, 0.05);
    assert_eq!(selection, SolverSelection::Sparse);
}

#[test]
fn test_select_lm_for_moderate() {
    // High nonlinearity, non-trivial system => LevenbergMarquardt
    let selection = select_solver(10, 10, Nonlinearity::High, 0.5);
    assert_eq!(selection, SolverSelection::LevenbergMarquardt);
}

#[test]
fn test_select_sparsity_estimate() {
    // 10x10 matrix with 20 non-zeros: sparsity = 20/100 = 0.2
    let sp = estimate_sparsity(10, 10, 20);
    assert!((sp - 0.2).abs() < 1e-10);

    // Empty matrix degenerate case
    let sp = estimate_sparsity(0, 0, 0);
    assert_eq!(sp, 1.0);
}

// ============================================================================
// Pipeline Component Extraction (5+ tests)
// ============================================================================

#[test]
fn test_extract_single_component() {
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 5.0,
    })];

    let params = vec![2.0, 3.0];
    let fixed = vec![false, false];

    let component = extract_component(&constraints, &[0], &params, &fixed);

    // Must implement Problem correctly
    assert_eq!(component.variable_count(), 2);
    assert_eq!(component.residual_count(), 1);
    assert!(component.name().contains("component"));
}

#[test]
fn test_extract_component_residuals() {
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 5.0,
    })];

    let params = vec![2.0, 3.0];
    let fixed = vec![false, false];

    let component = extract_component(&constraints, &[0], &params, &fixed);

    // Gather free variable indices
    let free = component.free_variable_indices().to_vec();

    // Build x in local order matching the component's mapping
    let x: Vec<f64> = free.iter().map(|&gi| params[gi]).collect();
    let residuals = component.residuals(&x);

    // Expected residual: (2 + 3) - 5 = 0
    assert_eq!(residuals.len(), 1);
    assert!((residuals[0]).abs() < 1e-10);
}

#[test]
fn test_extract_component_jacobian() {
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 3.0,
    })];

    let params = vec![1.0, 2.0];
    let fixed = vec![false, false];

    let component = extract_component(&constraints, &[0], &params, &fixed);

    let free = component.free_variable_indices().to_vec();
    let x: Vec<f64> = free.iter().map(|&gi| params[gi]).collect();
    let jac = component.jacobian(&x);

    // Should have 2 entries (one per free variable), all with value 1.0
    assert_eq!(jac.len(), 2);
    for (row, col, val) in &jac {
        assert_eq!(*row, 0);
        assert!(*col < 2);
        assert!((val - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_extract_free_variables() {
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1, 2],
        target: 6.0,
    })];

    let params = vec![1.0, 2.0, 3.0];
    // Only params 0 and 2 are free; param 1 is fixed
    let fixed = vec![false, true, false];

    let component = extract_component(&constraints, &[0], &params, &fixed);

    // Only 2 free variables
    assert_eq!(component.variable_count(), 2);

    let free = component.free_variable_indices();
    assert!(free.contains(&0));
    assert!(free.contains(&2));
    assert!(!free.contains(&1));
}

#[test]
fn test_extract_with_fixed_params() {
    let constraints: Vec<Box<dyn Constraint>> = vec![
        Box::new(MockLinearConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 5.0,
        }),
        Box::new(MockLinearConstraint {
            id: ConstraintId(1),
            deps: vec![1, 2],
            target: 7.0,
        }),
    ];

    let params = vec![1.0, 2.0, 3.0];
    // Fix param 0
    let fixed = vec![true, false, false];

    let component = extract_component(&constraints, &[0, 1], &params, &fixed);

    // Params 1 and 2 are free
    assert_eq!(component.variable_count(), 2);
    // 2 equations (one per constraint)
    assert_eq!(component.residual_count(), 2);

    // Jacobian should only reference local columns 0 and 1 (mapped from global 1 and 2)
    let free = component.free_variable_indices().to_vec();
    let x: Vec<f64> = free.iter().map(|&gi| params[gi]).collect();
    let jac = component.jacobian(&x);
    for (_, col, _) in &jac {
        assert!(*col < 2, "Column index should be in local [0, 2)");
    }
}

#[test]
fn test_extract_component_initial_point() {
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 5.0,
    })];

    let params = vec![10.0, 20.0];
    let fixed = vec![false, false];

    let component = extract_component(&constraints, &[0], &params, &fixed);
    let x0 = component.initial_point(1.0);

    // Initial point should pull from current params for warm start
    assert_eq!(x0.len(), 2);
    // Both 10.0 and 20.0 should appear (order depends on hash iteration)
    let sum: f64 = x0.iter().sum();
    assert!((sum - 30.0).abs() < 1e-10);
}

// ============================================================================
// Pipeline Warm Start (3+ tests)
// ============================================================================

#[test]
fn test_warm_start_regularization() {
    let original = vec![1.0, 2.0];
    let x = vec![0.5, 1.5, 2.5];
    let x0 = vec![0.0, 1.0, 2.0];
    let lambda = 1e-3;

    let augmented = regularize_residuals(&original, &x, &x0, lambda);

    // 2 original + 3 regularization terms
    assert_eq!(augmented.len(), 5);
    assert_eq!(augmented[0], 1.0);
    assert_eq!(augmented[1], 2.0);

    // Regularization: lambda * (x[i] - x0[i])
    assert!((augmented[2] - 1e-3 * 0.5).abs() < 1e-12);
    assert!((augmented[3] - 1e-3 * 0.5).abs() < 1e-12);
    assert!((augmented[4] - 1e-3 * 0.5).abs() < 1e-12);
}

#[test]
fn test_warm_start_jacobian() {
    let original = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)];
    let lambda = 1e-4;

    let augmented = regularize_jacobian(&original, 2, 3, lambda);

    // 3 original + 3 diagonal = 6
    assert_eq!(augmented.len(), 6);

    // Original entries preserved
    assert_eq!(augmented[0], (0, 0, 2.0));
    assert_eq!(augmented[1], (0, 1, 1.0));
    assert_eq!(augmented[2], (1, 1, 3.0));

    // Regularization diagonal: rows start at n_original_equations=2
    assert_eq!(augmented[3], (2, 0, lambda));
    assert_eq!(augmented[4], (3, 1, lambda));
    assert_eq!(augmented[5], (4, 2, lambda));
}

#[test]
fn test_warm_start_underconstrained() {
    // Verify helper functions for detecting under-constrained systems
    assert!(is_underconstrained(2, 5)); // 5 vars > 2 eqs
    assert!(!is_underconstrained(5, 5)); // Equal
    assert!(!is_underconstrained(8, 3)); // Over-constrained

    assert_eq!(degrees_of_freedom(3, 5), 2);
    assert_eq!(degrees_of_freedom(5, 5), 0);
    assert_eq!(degrees_of_freedom(8, 3), -5);
}

#[test]
fn test_warm_start_default_lambda_magnitude() {
    // Verify that DEFAULT_LAMBDA is small enough to not dominate constraint residuals
    let original = vec![0.0]; // Constraint satisfied
    let x = vec![1.0];
    let x0 = vec![0.0];

    let augmented = regularize_residuals(&original, &x, &x0, DEFAULT_LAMBDA);
    assert_eq!(augmented.len(), 2);
    // Regularization term should be very small
    assert!(augmented[1].abs() < 1e-5);
}

// ============================================================================
// Full Pipeline Solve (5+ tests)
// ============================================================================

#[test]
fn test_pipeline_solve_single_component() {
    // Single linear constraint: x + y = 5, fix x=1, solve for y
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 5.0,
    })];

    let mut params = vec![1.0, 1.0];
    let fixed = vec![true, false];

    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

    assert_eq!(results.len(), 1);
    assert!(results[0].success, "Linear constraint should converge");
    assert!(results[0].residual_norm < 1e-6);

    // x stays 1.0, y should be 4.0
    assert!((params[0] - 1.0).abs() < 1e-10);
    assert!((params[1] - 4.0).abs() < 1e-3);
}

#[test]
fn test_pipeline_solve_two_components() {
    // Two independent groups, each with a linear constraint
    let constraints: Vec<Box<dyn Constraint>> = vec![
        Box::new(MockLinearConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 3.0,
        }),
        Box::new(MockLinearConstraint {
            id: ConstraintId(1),
            deps: vec![2, 3],
            target: 7.0,
        }),
    ];

    let mut params = vec![0.0, 0.0, 0.0, 0.0];
    let fixed = vec![true, false, true, false]; // Fix params 0 and 2

    let results = solve_dirty_components(
        &constraints,
        &mut params,
        &fixed,
        &[vec![0], vec![1]], // Two separate dirty components
    );

    assert_eq!(results.len(), 2);
    assert!(results[0].success);
    assert!(results[1].success);

    // params[0]=0 fixed => params[1]=3
    assert!((params[1] - 3.0).abs() < 1e-3);
    // params[2]=0 fixed => params[3]=7
    assert!((params[3] - 7.0).abs() < 1e-3);
}

#[test]
fn test_pipeline_solve_linear() {
    // Linear constraint should converge in very few iterations
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 10.0,
    })];

    let mut params = vec![3.0, 0.0];
    let fixed = vec![true, false]; // Fix param 0 at 3

    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    // Linear should converge in 1-2 iterations
    assert!(
        results[0].iterations <= 3,
        "Linear constraint should converge quickly, got {} iterations",
        results[0].iterations
    );
    // y should be 7
    assert!((params[1] - 7.0).abs() < 1e-6);
}

#[test]
fn test_pipeline_solve_nonlinear() {
    // Nonlinear constraint: x^2 + y^2 = 25
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockQuadraticConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 25.0,
    })];

    let mut params = vec![3.0, 4.5]; // Close to solution (3,4)
    let fixed = vec![true, false]; // Fix x=3, solve for y

    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

    assert_eq!(results.len(), 1);
    assert!(results[0].success, "Quadratic constraint should converge");

    // x=3 fixed, so y^2 = 25 - 9 = 16, y = 4 (from nearby initial guess)
    assert!((params[1] - 4.0).abs() < 1e-3, "y should be ~4.0, got {}", params[1]);
}

#[test]
fn test_solve_pipeline_struct() {
    let pipeline = SolvePipeline::new();

    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 9.0,
    })];

    let mut params = vec![2.0, 0.0];
    let fixed = vec![true, false];

    let results = pipeline.solve(&constraints, &mut params, &fixed, &[vec![0]]);

    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    assert!((params[1] - 7.0).abs() < 1e-3);
}

#[test]
fn test_solve_pipeline_default_trait() {
    let pipeline = SolvePipeline::default();

    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 4.0,
    })];

    let mut params = vec![1.0, 1.0];
    let fixed = vec![true, false];

    let results = pipeline.solve(&constraints, &mut params, &fixed, &[vec![0]]);
    assert!(results[0].success);
    assert!((params[1] - 3.0).abs() < 1e-3);
}

#[test]
fn test_pipeline_solver_selection_linear_uses_nr() {
    // Verify that linear constraints cause Newton-Raphson selection
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 5.0,
    })];

    let mut params = vec![1.0, 1.0];
    let fixed = vec![false, false];

    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

    assert_eq!(results[0].solver_used, SolverSelection::NewtonRaphson);
}

#[test]
fn test_pipeline_solver_selection_nonlinear_moderate() {
    // Moderate nonlinearity, >2 variables => Robust
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockQuadraticConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1, 2, 3, 4],
        target: 10.0,
    })];

    let mut params = vec![1.0; 5];
    let fixed = vec![false; 5];

    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

    assert_eq!(results[0].solver_used, SolverSelection::Robust);
}

#[test]
fn test_pipeline_empty_component_skipped() {
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0],
        target: 5.0,
    })];

    let mut params = vec![1.0];
    let fixed = vec![false];

    // Pass an empty constraint group; should be skipped
    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![]]);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_component_solve_result_fields() {
    let result = ComponentSolveResult {
        component_id: 3,
        success: true,
        iterations: 7,
        residual_norm: 1e-9,
        solver_used: SolverSelection::LevenbergMarquardt,
    };

    assert_eq!(result.component_id, 3);
    assert!(result.success);
    assert_eq!(result.iterations, 7);
    assert!(result.residual_norm < 1e-8);
    assert_eq!(result.solver_used, SolverSelection::LevenbergMarquardt);

    // Verify Debug impl works
    let debug = format!("{:?}", result);
    assert!(debug.contains("LevenbergMarquardt"));
}

// ============================================================================
// End-to-end: graph + pipeline integration
// ============================================================================

#[test]
fn test_graph_to_pipeline_end_to_end() {
    // Simulate a full workflow:
    // 1. Build an IncrementalGraph
    // 2. Add entities and constraints
    // 3. Take dirty components
    // 4. Determine constraint groups
    // 5. Solve them via the pipeline

    let mut graph = IncrementalGraph::new();

    // Two 2D points
    graph.add_entity(make_entity(0, 2)); // Point at params [0, 1]
    graph.add_entity(make_entity(1, 2)); // Point at params [2, 3]

    // One constraint connecting them
    graph.add_constraint(ConstraintMeta {
        id: ConstraintId(0),
        equation_count: 1,
        entity_deps: vec![EntityId(0), EntityId(1)],
        param_deps: vec![0, 1, 2, 3],
    });

    // Take dirty components
    let dirty = graph.take_dirty();
    assert!(!dirty.is_empty());

    // Build constraint for the solve pipeline
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 5.0,
    })];

    let mut params = vec![1.0, 1.0];
    let fixed = vec![true, false]; // Fix param 0

    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

    assert!(results[0].success);
    assert!((params[1] - 4.0).abs() < 1e-3);

    // After solving, dirty set should be empty (we already drained it)
    assert!(graph.take_dirty().is_empty());
}

#[test]
fn test_dof_analysis_integration_with_graph() {
    // Use the IncrementalGraph to compute DOF, then verify with DOFAnalysis

    let mut graph = IncrementalGraph::new();
    // 3 free 2D points: 6 free params
    graph.add_entity(make_entity(0, 2));
    graph.add_entity(make_entity(1, 2));
    graph.add_entity(make_entity(2, 2));

    // 3 distance constraints: 3 equations
    graph.add_constraint(make_constraint(0, vec![0, 1], 1));
    graph.add_constraint(make_constraint(1, vec![1, 2], 1));
    graph.add_constraint(make_constraint(2, vec![0, 2], 1));

    let comp = graph.component_of(EntityId(0)).unwrap();
    let dof = graph.component_dof(comp);
    assert_eq!(dof, 3); // 6 params - 3 equations = 3 DOF

    // Verify with DOFAnalysis
    let analysis = DOFAnalysis::new(comp, 6, 0, 3);
    assert_eq!(analysis.dof, 3);
    assert!(analysis.is_under_constrained());
    assert_eq!(analysis.status, ConstraintStatus::UnderConstrained);
}

#[test]
fn test_dm_decomposition_with_redundancy_pipeline() {
    // Build a system, run DM decomposition, then detect redundancy

    // 3 constraints, 2 variables => over-constrained, at least 1 redundant
    let n_constraints = 3;
    let n_variables = 2;
    let ids = vec![ConstraintId(0), ConstraintId(1), ConstraintId(2)];
    let edges = vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];

    // DM decomposition
    let dm = dm_decompose(n_constraints, n_variables, &edges);
    assert!(!dm.over.is_empty(), "Should have over-determined block");

    // Structural redundancy
    let struct_redundant = detect_structural_redundancy(&ids, n_constraints, n_variables, &edges);
    assert!(!struct_redundant.is_empty());

    // Numerical redundancy: rows 0 and 2 are identical
    let jacobian_entries = vec![
        (0, 0, 1.0),
        (0, 1, 2.0),
        (1, 0, 0.0),
        (1, 1, 1.0),
        (2, 0, 1.0),
        (2, 1, 2.0), // Same as row 0
    ];
    let num_redundant =
        detect_numerical_redundancy(&jacobian_entries, n_constraints, n_variables, &ids, 1e-10);
    assert!(!num_redundant.is_empty());
}

#[test]
fn test_warm_start_with_pipeline_solve() {
    // Verify that warm start (regularization) produces reasonable augmented residuals
    // then solve a system that starts near the solution

    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockLinearConstraint {
        id: ConstraintId(0),
        deps: vec![0, 1],
        target: 10.0,
    })];

    // Start close to solution
    let mut params = vec![5.0, 4.9]; // target: 5 + y = 10
    let fixed = vec![true, false];

    let component = extract_component(&constraints, &[0], &params, &fixed);
    let x0 = component.initial_point(1.0);

    // Get current residuals
    let res = component.residuals(&x0);

    // Apply regularization
    let augmented = regularize_residuals(&res, &x0, &x0, DEFAULT_LAMBDA);
    // Original residual length + n_variables
    assert_eq!(augmented.len(), res.len() + x0.len());
    // Regularization terms should be zero when x == x0
    for i in res.len()..augmented.len() {
        assert!(augmented[i].abs() < 1e-15);
    }

    // Now actually solve
    let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);
    assert!(results[0].success);
    assert!((params[1] - 5.0).abs() < 1e-3);
}
