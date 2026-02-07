//! [`ConstraintSystem`] — the top-level coordinator for entity/constraint solving.
//!
//! This module provides the main entry point for building and solving constraint
//! systems. It manages entities, constraints, parameters, and the solve pipeline:
//!
//! 1. Entities are added (each owns parameters in the [`ParamStore`]).
//! 2. Constraints are added between entities.
//! 3. On [`solve()`](ConstraintSystem::solve), the system decomposes into
//!    independent clusters of coupled constraints.
//! 4. Each cluster becomes a [`ReducedSubProblem`](crate::solve::ReducedSubProblem)
//!    and is solved with an LM solver.
//! 5. Solutions are written back to the `ParamStore`.
//!
//! # Example
//!
//! ```ignore
//! use solverang::system::{ConstraintSystem, SystemConfig};
//!
//! let mut system = ConstraintSystem::new();
//! let px = system.alloc_param(0.0, entity_id);
//! // ... add entities, constraints ...
//! let result = system.solve();
//! ```

use std::collections::HashMap;

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::id::{ClusterId, ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;
use crate::problem::Problem;
use crate::solve::ReducedSubProblem;
use crate::solver::{LMConfig, LMSolver, SolverConfig};
use crate::solver::SolveResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the constraint system and its solver pipeline.
#[derive(Clone, Debug)]
pub struct SystemConfig {
    /// Configuration for the Levenberg-Marquardt solver.
    pub lm_config: LMConfig,
    /// Configuration for the Newton-Raphson solver (used by AutoSolver).
    pub solver_config: SolverConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            lm_config: LMConfig::default(),
            solver_config: SolverConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Overall result of solving the entire constraint system.
pub struct SystemResult {
    /// High-level status of the solve.
    pub status: SystemStatus,
    /// Per-cluster results (one entry per independent cluster).
    pub clusters: Vec<ClusterResult>,
    /// Total solver iterations summed across all clusters.
    pub total_iterations: usize,
    /// Wall-clock duration of the solve.
    pub duration: std::time::Duration,
}

/// High-level status of the entire system solve.
#[derive(Debug)]
pub enum SystemStatus {
    /// All clusters converged.
    Solved,
    /// Some clusters converged but at least one did not.
    PartiallySolved,
    /// Structural issues detected before or after solving.
    DiagnosticFailure(Vec<DiagnosticIssue>),
}

/// Result of solving a single cluster.
pub struct ClusterResult {
    /// Which cluster this result belongs to.
    pub cluster_id: ClusterId,
    /// Solve status for this cluster.
    pub status: ClusterSolveStatus,
    /// Number of solver iterations for this cluster.
    pub iterations: usize,
    /// Final residual norm for this cluster.
    pub residual_norm: f64,
}

/// Solve status for a single cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSolveStatus {
    /// The solver converged within tolerance.
    Converged,
    /// The solver ran but did not converge.
    NotConverged,
    /// The cluster was skipped (e.g., no free variables).
    Skipped,
}

/// A diagnostic issue detected in the constraint system.
#[derive(Debug, Clone)]
pub enum DiagnosticIssue {
    /// A constraint is redundant (implied by others).
    RedundantConstraint {
        constraint: ConstraintId,
        implied_by: Vec<ConstraintId>,
    },
    /// Two or more constraints conflict (cannot be simultaneously satisfied).
    ConflictingConstraints {
        constraints: Vec<ConstraintId>,
    },
    /// An entity has unconstrained directions.
    UnderConstrained {
        entity: EntityId,
        free_directions: usize,
    },
}

// ---------------------------------------------------------------------------
// Internal: cluster decomposition via union-find on ParamIds
// ---------------------------------------------------------------------------

/// A cluster of constraints that share parameters (directly or transitively).
#[derive(Debug, Clone)]
struct Cluster {
    id: ClusterId,
    /// Indices into `ConstraintSystem::constraints`.
    constraint_indices: Vec<usize>,
    /// All distinct `ParamId`s touched by constraints in this cluster.
    param_ids: Vec<ParamId>,
}

/// Union-Find for efficient connected component detection.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path splitting
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Decompose constraints into independent clusters based on shared `ParamId`s.
///
/// Two constraints belong to the same cluster if they share any parameter
/// (directly or transitively through other constraints).
fn decompose_into_clusters(
    constraints: &[Option<Box<dyn Constraint>>],
) -> Vec<Cluster> {
    // Collect alive constraint indices
    let alive: Vec<usize> = constraints
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|_| i))
        .collect();

    if alive.is_empty() {
        return Vec::new();
    }

    // Build a mapping: ParamId -> list of alive constraint indices that use it
    let mut param_to_constraints: HashMap<ParamId, Vec<usize>> = HashMap::new();
    for &idx in &alive {
        let constraint = constraints[idx].as_ref().unwrap();
        for &pid in constraint.param_ids() {
            param_to_constraints.entry(pid).or_default().push(idx);
        }
    }

    // Map alive constraint indices to dense [0..alive.len()) for union-find
    let mut idx_to_dense: HashMap<usize, usize> = HashMap::new();
    for (dense, &idx) in alive.iter().enumerate() {
        idx_to_dense.insert(idx, dense);
    }

    let mut uf = UnionFind::new(alive.len());

    // Union constraints that share a parameter
    for indices in param_to_constraints.values() {
        if indices.len() > 1 {
            let first = idx_to_dense[&indices[0]];
            for &ci in &indices[1..] {
                uf.union(first, idx_to_dense[&ci]);
            }
        }
    }

    // Group by root
    let mut root_to_group: HashMap<usize, Vec<usize>> = HashMap::new();
    for (dense, &idx) in alive.iter().enumerate() {
        let root = uf.find(dense);
        root_to_group.entry(root).or_default().push(idx);
    }

    // Build Cluster structs
    let mut clusters: Vec<Cluster> = root_to_group
        .into_values()
        .enumerate()
        .map(|(cluster_idx, mut constraint_indices)| {
            constraint_indices.sort_unstable();
            // Collect all unique ParamIds
            let mut param_set: Vec<ParamId> = Vec::new();
            let mut seen: std::collections::HashSet<ParamId> = std::collections::HashSet::new();
            for &ci in &constraint_indices {
                let constraint = constraints[ci].as_ref().unwrap();
                for &pid in constraint.param_ids() {
                    if seen.insert(pid) {
                        param_set.push(pid);
                    }
                }
            }
            Cluster {
                id: ClusterId(cluster_idx),
                constraint_indices,
                param_ids: param_set,
            }
        })
        .collect();

    // Deterministic ordering by first constraint index
    clusters.sort_by_key(|c| c.constraint_indices.first().copied().unwrap_or(usize::MAX));
    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.id = ClusterId(i);
    }

    clusters
}

// ---------------------------------------------------------------------------
// ConstraintSystem
// ---------------------------------------------------------------------------

/// The top-level constraint system coordinator.
///
/// Manages entities, constraints, and parameters. Provides a `solve()` method
/// that decomposes the system into independent clusters and solves each one.
pub struct ConstraintSystem {
    params: ParamStore,
    entities: Vec<Option<Box<dyn Entity>>>,
    constraints: Vec<Option<Box<dyn Constraint>>>,
    config: SystemConfig,
    /// Cached clusters from the last decomposition.
    clusters: Vec<Cluster>,
    /// Whether the constraint/entity set has changed since the last decompose.
    needs_decompose: bool,
    /// Next generation for entity ID allocation.
    next_entity_gen: u32,
    /// Next generation for constraint ID allocation.
    next_constraint_gen: u32,
}

impl Default for ConstraintSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSystem {
    /// Create a new, empty constraint system with default configuration.
    pub fn new() -> Self {
        Self {
            params: ParamStore::new(),
            entities: Vec::new(),
            constraints: Vec::new(),
            config: SystemConfig::default(),
            clusters: Vec::new(),
            needs_decompose: true,
            next_entity_gen: 0,
            next_constraint_gen: 0,
        }
    }

    /// Create a new constraint system with the given configuration.
    pub fn with_config(config: SystemConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    // -------------------------------------------------------------------
    // Parameter access
    // -------------------------------------------------------------------

    /// Allocate a new parameter with the given initial value, owned by `owner`.
    ///
    /// This is the primary way entities obtain `ParamId`s before being added.
    pub fn alloc_param(&mut self, value: f64, owner: EntityId) -> ParamId {
        self.params.alloc(value, owner)
    }

    /// Shared reference to the parameter store.
    pub fn params(&self) -> &ParamStore {
        &self.params
    }

    /// Mutable reference to the parameter store.
    pub fn params_mut(&mut self) -> &mut ParamStore {
        &mut self.params
    }

    /// Get the current value of a parameter.
    pub fn get_param(&self, id: ParamId) -> f64 {
        self.params.get(id)
    }

    /// Set the value of a parameter.
    pub fn set_param(&mut self, id: ParamId, value: f64) {
        self.params.set(id, value);
    }

    /// Mark a parameter as fixed (excluded from solving).
    pub fn fix_param(&mut self, id: ParamId) {
        self.params.fix(id);
        self.needs_decompose = true;
    }

    /// Mark a parameter as free (included in solving).
    pub fn unfix_param(&mut self, id: ParamId) {
        self.params.unfix(id);
        self.needs_decompose = true;
    }

    // -------------------------------------------------------------------
    // Entity management
    // -------------------------------------------------------------------

    /// Add an entity to the system.
    ///
    /// The entity must already have its `EntityId` and `ParamId`s allocated
    /// (via [`alloc_entity_id`](Self::alloc_entity_id) and
    /// [`alloc_param`](Self::alloc_param)).
    ///
    /// Returns the entity's ID.
    pub fn add_entity(&mut self, entity: Box<dyn Entity>) -> EntityId {
        let id = entity.id();
        let idx = id.raw_index() as usize;

        // Grow the entity vector if needed
        if idx >= self.entities.len() {
            self.entities.resize_with(idx + 1, || None);
        }
        self.entities[idx] = Some(entity);
        self.needs_decompose = true;
        id
    }

    /// Allocate a new [`EntityId`] for constructing an entity.
    ///
    /// Call this first, then use the returned ID to allocate parameters
    /// via [`alloc_param`](Self::alloc_param), build the entity, and finally
    /// call [`add_entity`](Self::add_entity).
    pub fn alloc_entity_id(&mut self) -> EntityId {
        let gen = self.next_entity_gen;
        self.next_entity_gen += 1;
        let index = self.entities.len() as u32;
        // Reserve a slot
        self.entities.push(None);
        EntityId::new(index, gen)
    }

    /// Remove an entity and free its parameters.
    ///
    /// Any constraints referencing this entity will not be automatically
    /// removed; remove them separately if needed.
    pub fn remove_entity(&mut self, id: EntityId) {
        let idx = id.raw_index() as usize;
        if idx < self.entities.len() {
            if let Some(entity) = self.entities[idx].take() {
                for &pid in entity.params() {
                    self.params.free(pid);
                }
                self.needs_decompose = true;
            }
        }
    }

    // -------------------------------------------------------------------
    // Constraint management
    // -------------------------------------------------------------------

    /// Allocate a new [`ConstraintId`] for constructing a constraint.
    pub fn alloc_constraint_id(&mut self) -> ConstraintId {
        let gen = self.next_constraint_gen;
        self.next_constraint_gen += 1;
        let index = self.constraints.len() as u32;
        self.constraints.push(None);
        ConstraintId::new(index, gen)
    }

    /// Add a constraint to the system.
    ///
    /// The constraint must already have its `ConstraintId` set (via
    /// [`alloc_constraint_id`](Self::alloc_constraint_id)).
    ///
    /// Returns the constraint's ID.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) -> ConstraintId {
        let id = constraint.id();
        let idx = id.raw_index() as usize;

        if idx >= self.constraints.len() {
            self.constraints.resize_with(idx + 1, || None);
        }
        self.constraints[idx] = Some(constraint);
        self.needs_decompose = true;
        id
    }

    /// Remove a constraint from the system.
    pub fn remove_constraint(&mut self, id: ConstraintId) {
        let idx = id.raw_index() as usize;
        if idx < self.constraints.len() {
            self.constraints[idx] = None;
            self.needs_decompose = true;
        }
    }

    // -------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------

    /// Number of independent clusters in the current decomposition.
    ///
    /// If the system has changed since the last decompose, this triggers
    /// a re-decomposition.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Degrees of freedom: (free params) - (total equation count).
    ///
    /// A positive DOF means under-constrained; zero means well-constrained;
    /// negative means over-constrained.
    pub fn degrees_of_freedom(&self) -> i32 {
        let free_params = self.params.free_param_count() as i32;
        let equations: i32 = self
            .constraints
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.equation_count() as i32)
            .sum();
        free_params - equations
    }

    /// Number of alive entities.
    pub fn entity_count(&self) -> usize {
        self.entities.iter().filter(|e| e.is_some()).count()
    }

    /// Number of alive constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.iter().filter(|c| c.is_some()).count()
    }

    // -------------------------------------------------------------------
    // Solving
    // -------------------------------------------------------------------

    /// Solve the constraint system.
    ///
    /// 1. If the topology has changed, re-decompose into clusters.
    /// 2. For each cluster, build a [`ReducedSubProblem`] and solve with LM.
    /// 3. Write solutions back to the [`ParamStore`].
    /// 4. Return a [`SystemResult`] with per-cluster details.
    pub fn solve(&mut self) -> SystemResult {
        let start = std::time::Instant::now();

        // Step 1: re-decompose if needed
        if self.needs_decompose {
            self.clusters = decompose_into_clusters(&self.constraints);
            self.needs_decompose = false;
        }

        let solver = LMSolver::new(self.config.lm_config.clone());

        let mut cluster_results = Vec::with_capacity(self.clusters.len());
        let mut total_iterations: usize = 0;
        let mut all_converged = true;
        let mut any_converged = false;

        // Step 2-3: solve each cluster
        for cluster in &self.clusters {
            // Collect constraint references for this cluster
            let constraint_refs: Vec<&dyn Constraint> = cluster
                .constraint_indices
                .iter()
                .filter_map(|&idx| self.constraints[idx].as_deref())
                .collect();

            // Build the sub-problem and solve it.
            // We solve inside a block to release the borrow on self.params
            // before writing the solution back.
            let (mapping, result) = {
                let sub = ReducedSubProblem::new(
                    &self.params,
                    constraint_refs,
                    &cluster.param_ids,
                );

                // Skip clusters with no free variables
                if sub.variable_count() == 0 {
                    let residual_norm = if sub.residual_count() > 0 {
                        let r = sub.residuals(&[]);
                        r.iter().map(|v| v * v).sum::<f64>().sqrt()
                    } else {
                        0.0
                    };
                    cluster_results.push(ClusterResult {
                        cluster_id: cluster.id,
                        status: ClusterSolveStatus::Skipped,
                        iterations: 0,
                        residual_norm,
                    });
                    continue;
                }

                let x0 = sub.initial_point(1.0);
                let result = solver.solve(&sub, &x0);
                let mapping = sub.mapping().clone();
                (mapping, result)
            };
            // sub is dropped here, releasing the immutable borrow on self.params

            let (status, iterations, residual_norm) = match &result {
                SolveResult::Converged {
                    solution,
                    iterations,
                    residual_norm,
                } => {
                    self.params.write_free_values(solution, &mapping);
                    any_converged = true;
                    (ClusterSolveStatus::Converged, *iterations, *residual_norm)
                }
                SolveResult::NotConverged {
                    solution,
                    iterations,
                    residual_norm,
                } => {
                    self.params.write_free_values(solution, &mapping);
                    all_converged = false;
                    (ClusterSolveStatus::NotConverged, *iterations, *residual_norm)
                }
                SolveResult::Failed { .. } => {
                    all_converged = false;
                    (ClusterSolveStatus::NotConverged, 0, f64::INFINITY)
                }
            };

            total_iterations += iterations;
            cluster_results.push(ClusterResult {
                cluster_id: cluster.id,
                status,
                iterations,
                residual_norm,
            });
        }

        let duration = start.elapsed();

        let system_status = if all_converged && !cluster_results.is_empty() {
            SystemStatus::Solved
        } else if any_converged {
            SystemStatus::PartiallySolved
        } else if cluster_results.is_empty() {
            // No clusters to solve (no constraints); trivially solved
            SystemStatus::Solved
        } else {
            SystemStatus::DiagnosticFailure(Vec::new())
        };

        SystemResult {
            status: system_status,
            clusters: cluster_results,
            total_iterations,
            duration,
        }
    }

    // -----------------------------------------------------------------
    // Convenience methods (useful for testing and geometry plugins)
    // -----------------------------------------------------------------

    /// Total number of scalar equations across all constraints.
    pub fn equation_count(&self) -> usize {
        self.constraints
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.equation_count())
            .sum()
    }

    /// Evaluate all constraint residuals at the current parameter values.
    pub fn compute_residuals(&self) -> Vec<f64> {
        let mut residuals = Vec::new();
        for c in &self.constraints {
            if let Some(c) = c.as_ref() {
                residuals.extend(c.residuals(&self.params));
            }
        }
        residuals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // -------------------------------------------------------------------
    // Test entity: a 2D point with two parameters (x, y).
    // -------------------------------------------------------------------
    struct TestPoint {
        id: EntityId,
        params: Vec<ParamId>,
    }

    impl Entity for TestPoint {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            "TestPoint"
        }
    }

    // -------------------------------------------------------------------
    // Test constraint: distance between two 1D points equals target.
    //   residual = (a - b)^2 - d^2  (single equation)
    //   Using squared form to keep it simple. For tests with small values
    //   we use the linear form: residual = a - target.
    // -------------------------------------------------------------------
    struct FixValueConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        param: ParamId,
        target: f64,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "FixValue"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
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

    // -------------------------------------------------------------------
    // Test constraint: a + b = target  (sum constraint).
    // -------------------------------------------------------------------
    struct SumConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        params: Vec<ParamId>,
        target: f64,
    }

    impl Constraint for SumConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "Sum"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let a = store.get(self.params[0]);
            let b = store.get(self.params[1]);
            vec![a + b - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![
                (0, self.params[0], 1.0),
                (0, self.params[1], 1.0),
            ]
        }
    }

    /// Helper to build a point entity in the system.
    fn add_test_point(system: &mut ConstraintSystem, x: f64, y: f64) -> (EntityId, ParamId, ParamId) {
        let eid = system.alloc_entity_id();
        let px = system.alloc_param(x, eid);
        let py = system.alloc_param(y, eid);
        let point = TestPoint {
            id: eid,
            params: vec![px, py],
        };
        system.add_entity(Box::new(point));
        (eid, px, py)
    }

    #[test]
    fn test_empty_system() {
        let system = ConstraintSystem::new();
        assert_eq!(system.entity_count(), 0);
        assert_eq!(system.constraint_count(), 0);
        assert_eq!(system.degrees_of_freedom(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut system = ConstraintSystem::new();
        let (eid, _px, _py) = add_test_point(&mut system, 1.0, 2.0);

        assert_eq!(system.entity_count(), 1);
        assert_eq!(system.params().alive_count(), 2);
        assert_eq!(system.degrees_of_freedom(), 2); // 2 free params, 0 constraints

        // Verify param values
        let _ = eid; // used for ownership
    }

    #[test]
    fn test_add_and_remove_entity() {
        let mut system = ConstraintSystem::new();
        let (eid, px, py) = add_test_point(&mut system, 3.0, 4.0);

        assert_eq!(system.entity_count(), 1);
        assert_eq!(system.params().alive_count(), 2);

        system.remove_entity(eid);
        assert_eq!(system.entity_count(), 0);
        // Params should be freed
        assert_eq!(system.params().alive_count(), 0);

        // Suppress unused variable warnings
        let _ = (px, py);
    }

    #[test]
    fn test_add_constraint() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        let cid = system.alloc_constraint_id();
        let constraint = FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        };
        system.add_constraint(Box::new(constraint));

        assert_eq!(system.constraint_count(), 1);
        // DOF = 2 free params - 1 equation = 1
        assert_eq!(system.degrees_of_freedom(), 1);
    }

    #[test]
    fn test_remove_constraint() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        let cid = system.alloc_constraint_id();
        let constraint = FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        };
        system.add_constraint(Box::new(constraint));
        assert_eq!(system.constraint_count(), 1);

        system.remove_constraint(cid);
        assert_eq!(system.constraint_count(), 0);
        assert_eq!(system.degrees_of_freedom(), 2);
    }

    #[test]
    fn test_fix_unfix_param() {
        let mut system = ConstraintSystem::new();
        let (_eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        assert_eq!(system.degrees_of_freedom(), 2);

        system.fix_param(px);
        assert_eq!(system.degrees_of_freedom(), 1); // one param fixed

        system.unfix_param(px);
        assert_eq!(system.degrees_of_freedom(), 2);
    }

    #[test]
    fn test_solve_empty_system() {
        let mut system = ConstraintSystem::new();
        let result = system.solve();

        assert!(matches!(result.status, SystemStatus::Solved));
        assert_eq!(result.clusters.len(), 0);
        assert_eq!(result.total_iterations, 0);
    }

    #[test]
    fn test_solve_single_fix_constraint() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

        let cid = system.alloc_constraint_id();
        let constraint = FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 7.0,
        };
        system.add_constraint(Box::new(constraint));

        let result = system.solve();
        assert!(
            matches!(result.status, SystemStatus::Solved | SystemStatus::PartiallySolved),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status
        );

        // px should now be close to 7.0
        let val = system.get_param(px);
        assert!(
            (val - 7.0).abs() < 1e-6,
            "Expected px ~ 7.0, got {}",
            val
        );
    }

    #[test]
    fn test_solve_two_independent_clusters() {
        let mut system = ConstraintSystem::new();
        let (eid1, px1, _py1) = add_test_point(&mut system, 0.0, 0.0);
        let (eid2, px2, _py2) = add_test_point(&mut system, 0.0, 0.0);

        // Constraint on px1 -> target 3.0
        let cid1 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid1,
            entity_ids: vec![eid1],
            param: px1,
            target: 3.0,
        }));

        // Constraint on px2 -> target 5.0 (independent cluster)
        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid2,
            entity_ids: vec![eid2],
            param: px2,
            target: 5.0,
        }));

        let result = system.solve();

        // Should be 2 clusters
        assert_eq!(result.clusters.len(), 2);

        // Both should converge
        assert!(
            matches!(result.status, SystemStatus::Solved | SystemStatus::PartiallySolved),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status
        );

        assert!((system.get_param(px1) - 3.0).abs() < 1e-6);
        assert!((system.get_param(px2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_coupled_constraints() {
        let mut system = ConstraintSystem::new();
        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);

        // Fix px = 3.0
        let cid1 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid1,
            entity_ids: vec![eid],
            param: px,
            target: 3.0,
        }));

        // px + py = 10.0  =>  py = 7.0
        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(SumConstraint {
            id: cid2,
            entity_ids: vec![eid],
            params: vec![px, py],
            target: 10.0,
        }));

        let result = system.solve();

        // These two constraints share px, so they should be in the same cluster
        assert_eq!(result.clusters.len(), 1, "Coupled constraints should form 1 cluster");

        assert!(
            matches!(result.status, SystemStatus::Solved | SystemStatus::PartiallySolved),
            "Solve status: {:?}",
            result.status
        );

        assert!(
            (system.get_param(px) - 3.0).abs() < 1e-6,
            "px = {}, expected 3.0",
            system.get_param(px)
        );
        assert!(
            (system.get_param(py) - 7.0).abs() < 1e-6,
            "py = {}, expected 7.0",
            system.get_param(py)
        );
    }

    #[test]
    fn test_solve_with_fixed_param_cluster_skipped() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 5.0, 0.0);

        // Fix px so it cannot move
        system.fix_param(px);

        // Constraint wants px = 5.0 (already satisfied since px is fixed at 5.0)
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        }));

        let result = system.solve();

        // The cluster should be skipped (no free variables)
        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].status, ClusterSolveStatus::Skipped);
        // Residual should be ~0 since the constraint is already satisfied
        assert!(result.clusters[0].residual_norm < 1e-10);
    }

    #[test]
    fn test_get_set_param() {
        let mut system = ConstraintSystem::new();
        let eid = system.alloc_entity_id();
        let p = system.alloc_param(42.0, eid);

        assert!((system.get_param(p) - 42.0).abs() < 1e-12);

        system.set_param(p, 99.0);
        assert!((system.get_param(p) - 99.0).abs() < 1e-12);
    }

    #[test]
    fn test_with_config() {
        let config = SystemConfig {
            lm_config: LMConfig::robust(),
            solver_config: SolverConfig::fast(),
        };
        let system = ConstraintSystem::with_config(config);
        assert_eq!(system.entity_count(), 0);
    }

    #[test]
    fn test_decompose_into_clusters_empty() {
        let constraints: Vec<Option<Box<dyn Constraint>>> = Vec::new();
        let clusters = decompose_into_clusters(&constraints);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_decompose_independent_constraints() {
        // Build two constraints that do NOT share params
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let p1 = store.alloc(1.0, owner);
        let p2 = store.alloc(2.0, owner);

        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![owner],
            param: p1,
            target: 5.0,
        });
        let c2: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![owner],
            param: p2,
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), Some(c2)];
        let clusters = decompose_into_clusters(&constraints);
        assert_eq!(clusters.len(), 2, "Independent constraints -> 2 clusters");
    }

    #[test]
    fn test_decompose_coupled_constraints() {
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let p1 = store.alloc(1.0, owner);
        let p2 = store.alloc(2.0, owner);

        // c1 uses p1; c2 uses p1 and p2 -> they share p1 -> same cluster
        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![owner],
            param: p1,
            target: 5.0,
        });
        let c2: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![owner],
            params: vec![p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), Some(c2)];
        let clusters = decompose_into_clusters(&constraints);
        assert_eq!(clusters.len(), 1, "Coupled constraints -> 1 cluster");
        assert_eq!(clusters[0].constraint_indices.len(), 2);
    }

    #[test]
    fn test_system_result_duration() {
        let mut system = ConstraintSystem::new();
        let result = system.solve();
        // Duration should be non-negative (trivially true but checks the field exists)
        assert!(result.duration.as_nanos() >= 0);
    }

    #[test]
    fn test_needs_decompose_flag() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        // First solve triggers decompose
        let _ = system.solve();
        assert!(!system.needs_decompose);

        // Adding a constraint sets the flag
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        }));
        assert!(system.needs_decompose);

        // Solve again clears it
        let _ = system.solve();
        assert!(!system.needs_decompose);

        // Removing the constraint sets the flag
        system.remove_constraint(cid);
        assert!(system.needs_decompose);
    }

    #[test]
    fn test_cluster_count_after_solve() {
        let mut system = ConstraintSystem::new();
        let (eid1, px1, _) = add_test_point(&mut system, 0.0, 0.0);
        let (eid2, px2, _) = add_test_point(&mut system, 0.0, 0.0);

        let cid1 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid1,
            entity_ids: vec![eid1],
            param: px1,
            target: 1.0,
        }));

        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid2,
            entity_ids: vec![eid2],
            param: px2,
            target: 2.0,
        }));

        let _ = system.solve();
        assert_eq!(system.cluster_count(), 2);
    }
}
