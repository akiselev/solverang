//! [`ConstraintSystem`] — the top-level coordinator for entity/constraint solving.
//!
//! This module provides the main entry point for building and solving constraint
//! systems. It manages entities, constraints, parameters, and the solve pipeline:
//!
//! 1. Entities are added (each owns parameters in the [`ParamStore`]).
//! 2. Constraints are added between entities.
//! 3. On [`solve()`](ConstraintSystem::solve), the system delegates to a
//!    [`SolvePipeline`] which decomposes into independent clusters, analyzes,
//!    reduces, and solves each one.
//! 4. Solutions are written back to the `ParamStore`.
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

use crate::constraint::Constraint;
use crate::dataflow::{ChangeTracker, SolutionCache};
use crate::entity::Entity;
use crate::id::{ConstraintId, EntityId, ParamId};
use crate::optimization::{
    InequalityFn, MultiplierStore, Objective, OptimizationConfig, OptimizationResult,
    OptimizationStatus,
};
use crate::param::ParamStore;
use crate::pipeline::SolvePipeline;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the constraint system and its solver pipeline.
#[derive(Clone, Debug)]
pub struct SystemConfig {
    /// Configuration for the Levenberg-Marquardt solver.
    pub lm_config: crate::solver::LMConfig,
    /// Configuration for the Newton-Raphson solver (used by AutoSolver).
    pub solver_config: crate::solver::SolverConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            lm_config: crate::solver::LMConfig::default(),
            solver_config: crate::solver::SolverConfig::default(),
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
    pub cluster_id: crate::id::ClusterId,
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
    ConflictingConstraints { constraints: Vec<ConstraintId> },
    /// An entity has unconstrained directions.
    UnderConstrained {
        entity: EntityId,
        free_directions: usize,
    },
}

// ---------------------------------------------------------------------------
// ConstraintSystem
// ---------------------------------------------------------------------------

/// The top-level constraint system coordinator.
///
/// Manages entities, constraints, and parameters. Provides a `solve()` method
/// that delegates to a [`SolvePipeline`] which decomposes the system into
/// independent clusters, analyzes, reduces, and solves each one.
pub struct ConstraintSystem {
    params: ParamStore,
    entities: Vec<Option<Box<dyn Entity>>>,
    constraints: Vec<Option<Box<dyn Constraint>>>,
    config: SystemConfig,
    pipeline: SolvePipeline,
    change_tracker: ChangeTracker,
    solution_cache: SolutionCache,
    /// Per-slot generation counters for entity IDs.
    entity_generations: Vec<u32>,
    /// Free list of reusable entity slots.
    entity_free_list: Vec<u32>,
    /// Per-slot generation counters for constraint IDs.
    constraint_generations: Vec<u32>,
    /// Free list of reusable constraint slots.
    constraint_free_list: Vec<u32>,
    // --- Optimization extension ---
    /// Objective function to minimize (None = constraint-satisfaction only).
    objective: Option<Box<dyn Objective>>,
    /// Inequality constraints h(x) ≤ 0.
    inequalities: Vec<Option<Box<dyn InequalityFn>>>,
    /// Configuration for optimization solvers.
    opt_config: OptimizationConfig,
    /// Multipliers from the last optimization solve.
    last_multipliers: MultiplierStore,
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
            pipeline: SolvePipeline::default(),
            change_tracker: ChangeTracker::new(),
            solution_cache: SolutionCache::new(),
            entity_generations: Vec::new(),
            entity_free_list: Vec::new(),
            constraint_generations: Vec::new(),
            constraint_free_list: Vec::new(),
            objective: None,
            inequalities: Vec::new(),
            opt_config: OptimizationConfig::default(),
            last_multipliers: MultiplierStore::new(),
        }
    }

    /// Create a new constraint system with the given configuration.
    pub fn with_config(config: SystemConfig) -> Self {
        let mut s = Self::new();
        s.config = config;
        s
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
        self.change_tracker.mark_param_dirty(id);
    }

    /// Mark a parameter as fixed (excluded from solving).
    pub fn fix_param(&mut self, id: ParamId) {
        self.params.fix(id);
        self.change_tracker.mark_param_dirty(id);
        self.pipeline.invalidate();
    }

    /// Mark a parameter as free (included in solving).
    pub fn unfix_param(&mut self, id: ParamId) {
        self.params.unfix(id);
        self.change_tracker.mark_param_dirty(id);
        self.pipeline.invalidate();
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
        self.change_tracker.mark_entity_added(id);
        id
    }

    /// Allocate a new [`EntityId`] for constructing an entity.
    ///
    /// Call this first, then use the returned ID to allocate parameters
    /// via [`alloc_param`](Self::alloc_param), build the entity, and finally
    /// call [`add_entity`](Self::add_entity).
    pub fn alloc_entity_id(&mut self) -> EntityId {
        if let Some(index) = self.entity_free_list.pop() {
            let gen = self.entity_generations[index as usize] + 1;
            self.entity_generations[index as usize] = gen;
            // Clear the slot for reuse
            self.entities[index as usize] = None;
            EntityId::new(index, gen)
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(None);
            self.entity_generations.push(0);
            EntityId::new(index, 0)
        }
    }

    /// Remove an entity and free its parameters.
    ///
    /// Any constraints referencing this entity will not be automatically
    /// removed; remove them separately if needed.
    pub fn remove_entity(&mut self, id: EntityId) {
        let idx = id.raw_index() as usize;
        if idx < self.entities.len()
            && idx < self.entity_generations.len()
            && self.entity_generations[idx] == id.generation
        {
            if let Some(entity) = self.entities[idx].take() {
                for &pid in entity.params() {
                    self.params.free(pid);
                }
                self.entity_free_list.push(idx as u32);
                self.change_tracker.mark_entity_removed(id);
                self.pipeline.invalidate();
            }
        }
    }

    // -------------------------------------------------------------------
    // Constraint management
    // -------------------------------------------------------------------

    /// Allocate a new [`ConstraintId`] for constructing a constraint.
    pub fn alloc_constraint_id(&mut self) -> ConstraintId {
        if let Some(index) = self.constraint_free_list.pop() {
            let gen = self.constraint_generations[index as usize] + 1;
            self.constraint_generations[index as usize] = gen;
            self.constraints[index as usize] = None;
            ConstraintId::new(index, gen)
        } else {
            let index = self.constraints.len() as u32;
            self.constraints.push(None);
            self.constraint_generations.push(0);
            ConstraintId::new(index, 0)
        }
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
        self.change_tracker.mark_constraint_added(id);
        self.pipeline.invalidate();
        id
    }

    /// Remove a constraint from the system.
    pub fn remove_constraint(&mut self, id: ConstraintId) {
        let idx = id.raw_index() as usize;
        if idx < self.constraints.len()
            && idx < self.constraint_generations.len()
            && self.constraint_generations[idx] == id.generation
        {
            if self.constraints[idx].is_some() {
                self.constraints[idx] = None;
                self.constraint_free_list.push(idx as u32);
                self.change_tracker.mark_constraint_removed(id);
                self.pipeline.invalidate();
            }
        }
    }

    // -------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------

    /// Number of independent clusters in the current decomposition.
    pub fn cluster_count(&self) -> usize {
        self.pipeline.cluster_count()
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
    /// Delegates to the [`SolvePipeline`] which handles decomposition,
    /// analysis, reduction, per-cluster solving, and post-processing.
    pub fn solve(&mut self) -> SystemResult {
        let start = std::time::Instant::now();
        let mut result = self.pipeline.run(
            &self.constraints,
            &self.entities,
            &mut self.params,
            &self.config,
            &mut self.change_tracker,
            &mut self.solution_cache,
        );
        result.duration = start.elapsed();
        result
    }

    /// Solve only clusters affected by parameter changes since the last solve.
    /// Falls back to full solve on structural changes.
    pub fn solve_incremental(&mut self) -> SystemResult {
        // Same as solve() -- the pipeline already handles incremental logic.
        self.solve()
    }

    /// Project a drag displacement onto the constraint manifold.
    pub fn drag(&mut self, displacements: &[(ParamId, f64)]) -> crate::solve::drag::DragResult {
        use crate::solve::drag::{apply_drag, project_drag};

        // Build constraint refs and mapping for affected params.
        let constraint_refs: Vec<&dyn Constraint> = self
            .constraints
            .iter()
            .filter_map(|c| c.as_deref())
            .collect();
        let mapping = self.params.build_solver_mapping();

        let result = project_drag(
            &constraint_refs,
            &self.params,
            &mapping,
            displacements,
            1e-10,
        );

        apply_drag(&mut self.params, &mapping, &result);

        // Mark dragged params dirty for subsequent solve.
        for &(pid, _) in displacements {
            self.change_tracker.mark_param_dirty(pid);
        }

        result
    }

    /// Analyze redundancy in the constraint system.
    pub fn analyze_redundancy(&self) -> crate::graph::redundancy::RedundancyAnalysis {
        let constraint_refs: Vec<(usize, &dyn Constraint)> = self
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_deref().map(|c| (i, c as &dyn Constraint)))
            .collect();
        let mapping = self.params.build_solver_mapping();
        crate::graph::redundancy::analyze_redundancy(
            &constraint_refs,
            &self.params,
            &mapping,
            1e-10,
        )
    }

    /// Analyze degrees of freedom per entity.
    pub fn analyze_dof(&self) -> crate::graph::dof::DofAnalysis {
        let entity_refs: Vec<&dyn Entity> =
            self.entities.iter().filter_map(|e| e.as_deref()).collect();
        let constraint_refs: Vec<(usize, &dyn Constraint)> = self
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_deref().map(|c| (i, c as &dyn Constraint)))
            .collect();
        let mapping = self.params.build_solver_mapping();
        crate::graph::dof::analyze_dof(&entity_refs, &constraint_refs, &self.params, &mapping)
    }

    /// Run full diagnostics (redundancy + DOF analysis).
    pub fn diagnose(&self) -> Vec<DiagnosticIssue> {
        let mut issues = Vec::new();

        let redundancy = self.analyze_redundancy();
        for r in &redundancy.redundant {
            issues.push(DiagnosticIssue::RedundantConstraint {
                constraint: r.id,
                implied_by: vec![],
            });
        }
        for g in &redundancy.conflicts {
            issues.push(DiagnosticIssue::ConflictingConstraints {
                constraints: g.constraint_ids.clone(),
            });
        }

        let dof = self.analyze_dof();
        for e in &dof.entities {
            if e.dof > 0 {
                issues.push(DiagnosticIssue::UnderConstrained {
                    entity: e.entity_id,
                    free_directions: e.dof,
                });
            }
        }

        issues
    }

    // -------------------------------------------------------------------
    // Optimization
    // -------------------------------------------------------------------

    /// Set the objective function to minimize.
    ///
    /// Only one objective is supported at a time. Setting a new objective
    /// replaces the previous one.
    pub fn set_objective(&mut self, objective: Box<dyn Objective>) {
        self.objective = Some(objective);
    }

    /// Remove the objective function (revert to constraint-satisfaction only).
    pub fn clear_objective(&mut self) {
        self.objective = None;
    }

    /// Whether an objective function is set.
    pub fn has_objective(&self) -> bool {
        self.objective.is_some()
    }

    /// Add an inequality constraint h(x) ≤ 0.
    ///
    /// The inequality must already have its `ConstraintId` set (via
    /// [`alloc_constraint_id`](Self::alloc_constraint_id) — shares ID space
    /// with equality constraints).
    pub fn add_inequality(&mut self, inequality: Box<dyn InequalityFn>) -> ConstraintId {
        let id = inequality.id();
        let idx = id.raw_index() as usize;

        if idx >= self.inequalities.len() {
            self.inequalities.resize_with(idx + 1, || None);
        }
        self.inequalities[idx] = Some(inequality);
        id
    }

    /// Set optimization configuration.
    pub fn set_opt_config(&mut self, config: OptimizationConfig) {
        self.opt_config = config;
    }

    /// Get the current optimization configuration.
    pub fn opt_config(&self) -> &OptimizationConfig {
        &self.opt_config
    }

    /// Run constrained optimization: `min f(x) s.t. constraints`.
    ///
    /// Requires an objective to be set via [`set_objective`](Self::set_objective).
    /// Existing [`Constraint`] objects serve as equality constraints (`g(x) = 0`).
    /// [`InequalityFn`] objects serve as inequality constraints (`h(x) ≤ 0`).
    ///
    /// # Algorithm Selection
    ///
    /// - No constraints → BFGS (unconstrained)
    /// - Equality constraints only → ALM (with BFGS inner loop)
    /// - Auto → selects based on problem structure
    pub fn optimize(&mut self) -> OptimizationResult {
        let objective = match &self.objective {
            Some(obj) => obj.as_ref(),
            None => {
                return OptimizationResult {
                    objective_value: f64::NAN,
                    status: OptimizationStatus::Infeasible,
                    outer_iterations: 0,
                    inner_iterations: 0,
                    kkt_residual: crate::optimization::KktResidual {
                        primal: f64::INFINITY,
                        dual: f64::INFINITY,
                        complementarity: f64::INFINITY,
                    },
                    multipliers: MultiplierStore::new(),
                    constraint_violations: Vec::new(),
                    duration: std::time::Duration::ZERO,
                };
            }
        };

        // Classify: check if we have equality constraints
        let eq_constraints: Vec<&dyn Constraint> = self
            .constraints
            .iter()
            .filter_map(|c| c.as_deref())
            .collect();

        let has_equalities = !eq_constraints.is_empty();

        // Algorithm selection
        use crate::optimization::OptimizationAlgorithm;
        let algorithm = match self.opt_config.algorithm {
            OptimizationAlgorithm::Auto => {
                if has_equalities {
                    OptimizationAlgorithm::Alm
                } else {
                    OptimizationAlgorithm::Bfgs
                }
            }
            other => other,
        };

        let result = match algorithm {
            OptimizationAlgorithm::Bfgs | OptimizationAlgorithm::Auto => {
                crate::solver::BfgsSolver::solve(objective, &mut self.params, &self.opt_config)
            }
            OptimizationAlgorithm::Alm => crate::solver::AlmSolver::solve(
                objective,
                &eq_constraints,
                &mut self.params,
                &self.opt_config,
            ),
        };

        // Store multipliers for post-solve access
        self.last_multipliers = MultiplierStore::new();
        for (mid, val) in result.multipliers.iter() {
            self.last_multipliers.set(mid, val);
        }

        result
    }

    /// Get the Lagrange multipliers from the last optimization solve for a
    /// specific constraint.
    ///
    /// Returns `None` if no optimization has been run or if the constraint
    /// has no multipliers.
    pub fn multiplier(&self, constraint_id: ConstraintId) -> Option<Vec<f64>> {
        self.last_multipliers.lambda_for_constraint(constraint_id)
    }

    /// Get the full multiplier store from the last optimization solve.
    pub fn multipliers(&self) -> &MultiplierStore {
        &self.last_multipliers
    }

    // -------------------------------------------------------------------
    // Pipeline
    // -------------------------------------------------------------------

    /// Set a custom pipeline for this system.
    pub fn set_pipeline(&mut self, pipeline: SolvePipeline) {
        self.pipeline = pipeline;
    }

    /// Access the change tracker.
    pub fn change_tracker(&self) -> &ChangeTracker {
        &self.change_tracker
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
    use crate::solver::{LMConfig, SolverConfig};

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
            vec![(0, self.params[0], 1.0), (0, self.params[1], 1.0)]
        }
    }

    /// Helper to build a point entity in the system.
    fn add_test_point(
        system: &mut ConstraintSystem,
        x: f64,
        y: f64,
    ) -> (EntityId, ParamId, ParamId) {
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
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status
        );

        // px should now be close to 7.0
        let val = system.get_param(px);
        assert!((val - 7.0).abs() < 1e-6, "Expected px ~ 7.0, got {}", val);
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
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
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
        assert_eq!(
            result.clusters.len(),
            1,
            "Coupled constraints should form 1 cluster"
        );

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
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
    fn test_system_result_duration() {
        let mut system = ConstraintSystem::new();
        let result = system.solve();
        // Duration should be non-negative (trivially true but checks the field exists)
        let _duration = result.duration;
    }

    #[test]
    fn test_structural_change_triggers_redecompose() {
        let mut system = ConstraintSystem::new();
        let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

        // First solve works fine
        let _ = system.solve();

        // Adding a constraint is a structural change
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        }));

        // The change tracker should have structural changes
        assert!(system.change_tracker().has_structural_changes());

        // Solve again should succeed (triggers re-decompose)
        let result = system.solve();
        assert!(matches!(
            result.status,
            SystemStatus::Solved | SystemStatus::PartiallySolved
        ));

        // After solve, change tracker is cleared
        assert!(!system.change_tracker().has_any_changes());

        // Removing the constraint is also a structural change
        system.remove_constraint(cid);
        assert!(system.change_tracker().has_structural_changes());
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
