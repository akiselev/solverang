//! Pluggable solve pipeline.
//!
//! Each phase can be independently swapped with a custom implementation.
//!
//! ```text
//! Decompose → Analyze → Reduce → Solve → PostProcess
//! ```
//!
//! The [`SolvePipeline`] struct orchestrates the full pipeline, caching
//! decomposition results and supporting incremental re-solves via the
//! [`ChangeTracker`].

pub mod analyze;
pub mod decompose;
pub mod traits;
pub mod types;
pub mod post_process;
pub mod reduce;
pub mod solve_phase;

#[cfg(test)]
mod minpack_bridge_tests;
#[cfg(test)]
mod incremental_tests;
#[cfg(test)]
mod error_path_tests;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use types::{ClusterData, ClusterAnalysis, ReducedCluster, ClusterSolution};
pub use traits::{Decompose, Analyze, Reduce, SolveCluster, PostProcess};

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use std::collections::{HashMap, HashSet};

use crate::constraint::Constraint;
use crate::dataflow::{ChangeTracker, SolutionCache};
use crate::entity::Entity;
use crate::id::{ClusterId, ParamId};
use crate::param::ParamStore;
use crate::system::{
    ClusterResult, ClusterSolveStatus, DiagnosticIssue, SystemConfig, SystemResult, SystemStatus,
};

use self::analyze::DefaultAnalyze;
use self::decompose::DefaultDecompose;
use self::post_process::{DefaultPostProcess, collect_diagnostics};
use self::reduce::DefaultReduce;
use self::solve_phase::DefaultSolve;

// ---------------------------------------------------------------------------
// SolvePipeline
// ---------------------------------------------------------------------------

/// Orchestrates the full `Decompose -> Analyze -> Reduce -> Solve -> PostProcess`
/// pipeline, caching cluster decomposition between solves.
///
/// The pipeline supports incremental solving: when only parameter values change
/// (no structural changes), it re-uses the cached decomposition and only
/// re-solves dirty clusters.
pub struct SolvePipeline {
    decompose: Box<dyn Decompose>,
    analyze: Box<dyn Analyze>,
    reduce: Box<dyn Reduce>,
    solve: Box<dyn SolveCluster>,
    post_process: Box<dyn PostProcess>,
    /// Cached clusters from last decomposition.
    cached_clusters: Vec<ClusterData>,
    /// Whether decomposition cache is valid.
    clusters_valid: bool,
}

impl Default for SolvePipeline {
    fn default() -> Self {
        Self {
            decompose: Box::new(DefaultDecompose),
            analyze: Box::new(DefaultAnalyze::default()),
            reduce: Box::new(DefaultReduce),
            solve: Box::new(DefaultSolve),
            post_process: Box::new(DefaultPostProcess),
            cached_clusters: Vec::new(),
            clusters_valid: false,
        }
    }
}

impl SolvePipeline {
    /// Invalidate the cached decomposition, forcing a re-decompose on the
    /// next [`run`](Self::run) call.
    pub fn invalidate(&mut self) {
        self.clusters_valid = false;
    }

    /// Number of clusters in the cached decomposition.
    ///
    /// Returns 0 if no decomposition has been performed yet.
    pub fn cluster_count(&self) -> usize {
        self.cached_clusters.len()
    }

    /// Run the full pipeline.
    ///
    /// If structural changes are detected via the `tracker`, or the cached
    /// decomposition is invalid, the system is re-decomposed. Otherwise the
    /// cached decomposition is re-used and only dirty clusters are re-solved.
    ///
    /// Solutions are written back to `store` and the change tracker is
    /// cleared at the end.
    pub fn run(
        &mut self,
        constraints: &[Option<Box<dyn Constraint>>],
        entities: &[Option<Box<dyn Entity>>],
        store: &mut ParamStore,
        config: &SystemConfig,
        tracker: &mut ChangeTracker,
        cache: &mut SolutionCache,
    ) -> SystemResult {
        let start = std::time::Instant::now();

        // -----------------------------------------------------------------
        // (a) Decompose if needed
        // -----------------------------------------------------------------
        let structural_change = tracker.has_structural_changes() || !self.clusters_valid;
        if structural_change {
            self.cached_clusters =
                self.decompose.decompose(constraints, entities, store);
            self.clusters_valid = true;
            cache.invalidate_all();
        }

        // -----------------------------------------------------------------
        // (b) Build param_to_cluster map
        // -----------------------------------------------------------------
        let mut param_to_cluster: HashMap<ParamId, ClusterId> = HashMap::new();
        for cluster in &self.cached_clusters {
            for &pid in &cluster.param_ids {
                param_to_cluster.insert(pid, cluster.id);
            }
        }

        // -----------------------------------------------------------------
        // (c) Determine dirty clusters
        // -----------------------------------------------------------------
        let dirty_clusters: HashSet<ClusterId> = if structural_change {
            self.cached_clusters.iter().map(|c| c.id).collect()
        } else {
            tracker.compute_dirty_clusters(&param_to_cluster)
        };

        // -----------------------------------------------------------------
        // (d) Process each cluster
        // -----------------------------------------------------------------
        let mut cluster_results = Vec::with_capacity(self.cached_clusters.len());
        let mut all_diagnostics: Vec<DiagnosticIssue> = Vec::new();

        for cluster in &self.cached_clusters {
            if !dirty_clusters.contains(&cluster.id) {
                // Clean cluster: skip with cached residual.
                let cached_residual = cache
                    .get(&cluster.id)
                    .map(|c| c.residual_norm)
                    .unwrap_or(0.0);
                cluster_results.push(ClusterResult {
                    cluster_id: cluster.id,
                    status: ClusterSolveStatus::Skipped,
                    iterations: 0,
                    residual_norm: cached_residual,
                });
                continue;
            }

            // Phase 2: Analyze (immutable borrow of store).
            let analysis =
                self.analyze
                    .analyze(cluster, constraints, entities, store);

            // Phase 3: Reduce (may temporarily fix eliminated params in store).
            let reduced = self.reduce.reduce(cluster, constraints, store);

            // Phase 4: Solve (immutable borrow of store).
            let warm_start = cache.get(&cluster.id).map(|c| c.solution.as_slice());
            let solution = self.solve.solve_cluster(
                &reduced,
                &analysis,
                constraints,
                store,
                warm_start,
                config,
            );

            // -- Write solution back to store --

            // Write solution param_values back to store.
            for &(pid, val) in &solution.param_values {
                store.set(pid, val);
            }

            // Write numerical_solution back via mapping if present.
            if let (Some(mapping), Some(nums)) =
                (&solution.mapping, &solution.numerical_solution)
            {
                store.write_free_values(nums, mapping);
            }

            // Cache the solution.
            let cached_solution = solution
                .numerical_solution
                .clone()
                .unwrap_or_default();
            cache.store(
                cluster.id,
                cached_solution,
                solution.residual_norm,
                solution.iterations,
            );

            // Un-fix params that were temporarily fixed during the reduce
            // phase so they remain free for subsequent pipeline runs.
            for &(pid, _) in &reduced.eliminated_params {
                store.unfix(pid);
            }

            // Post-process.
            let result =
                self.post_process
                    .post_process(&solution, &analysis, cluster);
            cluster_results.push(result);

            // Collect diagnostics from analysis.
            all_diagnostics.extend(collect_diagnostics(&analysis));
        }

        // -----------------------------------------------------------------
        // (e) Aggregate results
        // -----------------------------------------------------------------
        let mut converged_count = 0usize;
        let mut not_converged_count = 0usize;
        let mut skipped_count = 0usize;
        let mut total_iterations = 0usize;

        for cr in &cluster_results {
            total_iterations += cr.iterations;
            match cr.status {
                ClusterSolveStatus::Converged => converged_count += 1,
                ClusterSolveStatus::NotConverged => not_converged_count += 1,
                ClusterSolveStatus::Skipped => skipped_count += 1,
            }
        }

        let status = if !all_diagnostics.is_empty() && not_converged_count > 0 {
            SystemStatus::DiagnosticFailure(all_diagnostics)
        } else if not_converged_count == 0 {
            // All clusters either converged or were skipped.
            SystemStatus::Solved
        } else if converged_count > 0 || skipped_count > 0 {
            // Mixed: some converged/skipped, some failed.
            SystemStatus::PartiallySolved
        } else {
            // All clusters failed to converge.
            SystemStatus::DiagnosticFailure(all_diagnostics)
        };

        tracker.clear();

        SystemResult {
            status,
            clusters: cluster_results,
            total_iterations,
            duration: start.elapsed(),
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`SolvePipeline`] with custom phase
/// implementations.
///
/// Any phase left unset will use its default implementation.
///
/// # Example
///
/// ```ignore
/// use solverang::pipeline::{PipelineBuilder, SolvePipeline};
/// use solverang::pipeline::analyze::NoopAnalyze;
///
/// let pipeline = PipelineBuilder::new()
///     .analyze(NoopAnalyze)
///     .build();
/// ```
pub struct PipelineBuilder {
    decompose: Option<Box<dyn Decompose>>,
    analyze: Option<Box<dyn Analyze>>,
    reduce: Option<Box<dyn Reduce>>,
    solve: Option<Box<dyn SolveCluster>>,
    post_process: Option<Box<dyn PostProcess>>,
}

impl PipelineBuilder {
    /// Create a new builder with all phases unset (defaults will be used).
    pub fn new() -> Self {
        Self {
            decompose: None,
            analyze: None,
            reduce: None,
            solve: None,
            post_process: None,
        }
    }

    /// Set a custom decomposition phase.
    pub fn decompose(mut self, d: impl Decompose + 'static) -> Self {
        self.decompose = Some(Box::new(d));
        self
    }

    /// Set a custom analysis phase.
    pub fn analyze(mut self, a: impl Analyze + 'static) -> Self {
        self.analyze = Some(Box::new(a));
        self
    }

    /// Set a custom reduction phase.
    pub fn reduce(mut self, r: impl Reduce + 'static) -> Self {
        self.reduce = Some(Box::new(r));
        self
    }

    /// Set a custom solve phase.
    pub fn solve(mut self, s: impl SolveCluster + 'static) -> Self {
        self.solve = Some(Box::new(s));
        self
    }

    /// Set a custom post-processing phase.
    pub fn post_process(mut self, p: impl PostProcess + 'static) -> Self {
        self.post_process = Some(Box::new(p));
        self
    }

    /// Build the [`SolvePipeline`], filling defaults for any unset phases.
    pub fn build(self) -> SolvePipeline {
        SolvePipeline {
            decompose: self
                .decompose
                .unwrap_or_else(|| Box::new(DefaultDecompose)),
            analyze: self
                .analyze
                .unwrap_or_else(|| Box::new(DefaultAnalyze::default())),
            reduce: self
                .reduce
                .unwrap_or_else(|| Box::new(DefaultReduce)),
            solve: self
                .solve
                .unwrap_or_else(|| Box::new(DefaultSolve)),
            post_process: self
                .post_process
                .unwrap_or_else(|| Box::new(DefaultPostProcess)),
            cached_clusters: Vec::new(),
            clusters_valid: false,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // -----------------------------------------------------------------------
    // Test entity: a 2D point with two parameters (x, y).
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Test constraint: fix a single parameter to a target value.
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Test constraint: a + b = target (sum constraint).
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_pipeline_constructs_without_panicking() {
        let _pipeline = SolvePipeline::default();
    }

    #[test]
    fn builder_new_build_produces_valid_pipeline() {
        let _pipeline = PipelineBuilder::new().build();
    }

    #[test]
    fn builder_with_custom_phases_overrides_defaults() {
        use crate::pipeline::analyze::NoopAnalyze;
        use crate::pipeline::reduce::NoopReduce;
        use crate::pipeline::solve_phase::NumericalOnlySolve;
        use crate::pipeline::post_process::DiagnosticPostProcess;

        let _pipeline = PipelineBuilder::new()
            .analyze(NoopAnalyze)
            .reduce(NoopReduce)
            .solve(NumericalOnlySolve)
            .post_process(DiagnosticPostProcess)
            .build();
    }

    // -----------------------------------------------------------------------
    // End-to-end tests
    // -----------------------------------------------------------------------

    #[test]
    fn end_to_end_pipeline_solves_simple_system() {
        let mut store = ParamStore::new();
        let eid1 = EntityId::new(0, 0);
        let eid2 = EntityId::new(1, 0);

        let px1 = store.alloc(0.0, eid1);
        let py1 = store.alloc(0.0, eid1);
        let px2 = store.alloc(0.0, eid2);
        let py2 = store.alloc(0.0, eid2);

        let point1 = TestPoint {
            id: eid1,
            params: vec![px1, py1],
        };
        let point2 = TestPoint {
            id: eid2,
            params: vec![px2, py2],
        };

        let entities: Vec<Option<Box<dyn Entity>>> = vec![
            Some(Box::new(point1)),
            Some(Box::new(point2)),
        ];

        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![eid1],
            param: px1,
            target: 3.0,
        });
        let c2: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![eid1],
            param: py1,
            target: 4.0,
        });
        let c3: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(2, 0),
            entity_ids: vec![eid2],
            param: px2,
            target: 7.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> =
            vec![Some(c1), Some(c2), Some(c3)];

        let config = SystemConfig::default();
        let mut tracker = ChangeTracker::new();
        let mut solution_cache = SolutionCache::new();
        let mut pipeline = SolvePipeline::default();

        // Mark structural changes so decomposition runs.
        tracker.mark_entity_added(eid1);
        tracker.mark_entity_added(eid2);
        tracker.mark_constraint_added(ConstraintId::new(0, 0));
        tracker.mark_constraint_added(ConstraintId::new(1, 0));
        tracker.mark_constraint_added(ConstraintId::new(2, 0));

        let result = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut solution_cache,
        );

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status,
        );

        assert!(
            (store.get(px1) - 3.0).abs() < 1e-6,
            "px1 = {}, expected 3.0",
            store.get(px1),
        );
        assert!(
            (store.get(py1) - 4.0).abs() < 1e-6,
            "py1 = {}, expected 4.0",
            store.get(py1),
        );
        assert!(
            (store.get(px2) - 7.0).abs() < 1e-6,
            "px2 = {}, expected 7.0",
            store.get(px2),
        );
    }

    #[test]
    fn pipeline_skips_clean_clusters_on_second_run() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(0, 0);
        let px = store.alloc(0.0, eid);

        let point = TestPoint {
            id: eid,
            params: vec![px],
        };
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(Box::new(point))];

        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        });
        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];

        let config = SystemConfig::default();
        let mut tracker = ChangeTracker::new();
        let mut cache = SolutionCache::new();
        let mut pipeline = SolvePipeline::default();

        // First run: structural changes trigger decomposition.
        tracker.mark_entity_added(eid);
        tracker.mark_constraint_added(ConstraintId::new(0, 0));

        let result1 = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );
        assert!(matches!(
            result1.status,
            SystemStatus::Solved | SystemStatus::PartiallySolved
        ));

        // Second run: no changes, clusters should be skipped.
        let result2 = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );
        assert!(matches!(result2.status, SystemStatus::Solved));
        assert_eq!(result2.clusters.len(), 1);
        assert_eq!(result2.clusters[0].status, ClusterSolveStatus::Skipped);
        assert_eq!(result2.total_iterations, 0);
    }

    #[test]
    fn pipeline_invalidate_forces_redecompose() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(0, 0);
        let px = store.alloc(0.0, eid);

        let point = TestPoint {
            id: eid,
            params: vec![px],
        };
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(Box::new(point))];

        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![eid],
            param: px,
            target: 5.0,
        });
        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];

        let config = SystemConfig::default();
        let mut tracker = ChangeTracker::new();
        let mut cache = SolutionCache::new();
        let mut pipeline = SolvePipeline::default();

        // First run with structural change.
        tracker.mark_entity_added(eid);
        let _ = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );

        // Reset the param value so the solver actually has work to do.
        store.set(px, 0.0);

        // Invalidate and run again: should re-decompose and re-solve.
        pipeline.invalidate();
        let result = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );
        assert!(matches!(
            result.status,
            SystemStatus::Solved | SystemStatus::PartiallySolved
        ));
        // Verify the param was re-solved to the correct value.
        assert!(
            (store.get(px) - 5.0).abs() < 1e-6,
            "px = {}, expected 5.0 after invalidation + re-solve",
            store.get(px),
        );
    }

    #[test]
    fn end_to_end_coupled_constraints() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(0, 0);
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let point = TestPoint {
            id: eid,
            params: vec![px, py],
        };
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(Box::new(point))];

        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![eid],
            param: px,
            target: 3.0,
        });
        let c2: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![eid],
            params: vec![px, py],
            target: 10.0,
        });
        let constraints: Vec<Option<Box<dyn Constraint>>> =
            vec![Some(c1), Some(c2)];

        let config = SystemConfig::default();
        let mut tracker = ChangeTracker::new();
        let mut cache = SolutionCache::new();
        let mut pipeline = SolvePipeline::default();

        tracker.mark_entity_added(eid);
        tracker.mark_constraint_added(ConstraintId::new(0, 0));
        tracker.mark_constraint_added(ConstraintId::new(1, 0));

        let result = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Solve status: {:?}",
            result.status,
        );
        assert!(
            (store.get(px) - 3.0).abs() < 1e-6,
            "px = {}, expected 3.0",
            store.get(px),
        );
        assert!(
            (store.get(py) - 7.0).abs() < 1e-6,
            "py = {}, expected 7.0",
            store.get(py),
        );
    }

    #[test]
    fn empty_system_returns_solved() {
        let mut store = ParamStore::new();
        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![];
        let entities: Vec<Option<Box<dyn Entity>>> = vec![];
        let config = SystemConfig::default();
        let mut tracker = ChangeTracker::new();
        let mut cache = SolutionCache::new();
        let mut pipeline = SolvePipeline::default();

        let result = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );

        assert!(matches!(result.status, SystemStatus::Solved));
        assert_eq!(result.clusters.len(), 0);
        assert_eq!(result.total_iterations, 0);
    }

    // -----------------------------------------------------------------------
    // Regression: cascading elimination through the full pipeline
    // -----------------------------------------------------------------------

    /// Regression test for the reduce phase coupling bug: when the
    /// EliminateReducer analytically solves a constraint involving a param
    /// that participates in other constraints, the remaining constraints
    /// should see the updated value and cascade further eliminations.
    ///
    /// This end-to-end test verifies that:
    /// 1. Cascading elimination works across linear constraints.
    /// 2. Eliminated params are unfixed after solve (not permanently fixed).
    /// 3. A second pipeline run produces the same correct result.
    #[test]
    fn end_to_end_cascading_elimination() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(0, 0);
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let point = TestPoint {
            id: eid,
            params: vec![px, py],
        };
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(Box::new(point))];

        // C0: px = 3.0 (eliminable)
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![eid],
            param: px,
            target: 3.0,
        });
        // C1: px + py = 10.0 (becomes eliminable after px is determined)
        let c1: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![eid],
            params: vec![px, py],
            target: 10.0,
        });
        let constraints: Vec<Option<Box<dyn Constraint>>> =
            vec![Some(c0), Some(c1)];

        let config = SystemConfig::default();
        let mut tracker = ChangeTracker::new();
        let mut cache = SolutionCache::new();
        let mut pipeline = SolvePipeline::default();

        tracker.mark_entity_added(eid);
        tracker.mark_constraint_added(ConstraintId::new(0, 0));
        tracker.mark_constraint_added(ConstraintId::new(1, 0));

        let result = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );

        assert!(
            matches!(result.status, SystemStatus::Solved),
            "Expected Solved, got {:?}",
            result.status,
        );
        assert!(
            (store.get(px) - 3.0).abs() < 1e-6,
            "px = {}, expected 3.0",
            store.get(px),
        );
        assert!(
            (store.get(py) - 7.0).abs() < 1e-6,
            "py = {}, expected 7.0",
            store.get(py),
        );

        // Params should NOT be permanently fixed after solve.
        assert!(
            !store.is_fixed(px),
            "px should not remain fixed after solve"
        );
        assert!(
            !store.is_fixed(py),
            "py should not remain fixed after solve"
        );

        // Second run with the same system should still produce correct
        // results (verifies that un-fixing works properly).
        pipeline.invalidate();
        let result2 = pipeline.run(
            &constraints,
            &entities,
            &mut store,
            &config,
            &mut tracker,
            &mut cache,
        );
        assert!(
            matches!(result2.status, SystemStatus::Solved),
            "Second run: expected Solved, got {:?}",
            result2.status,
        );
        assert!(
            (store.get(px) - 3.0).abs() < 1e-6,
            "Second run: px = {}, expected 3.0",
            store.get(px),
        );
        assert!(
            (store.get(py) - 7.0).abs() < 1e-6,
            "Second run: py = {}, expected 7.0",
            store.get(py),
        );
    }
}
