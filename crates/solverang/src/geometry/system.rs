//! Constraint system for geometric constraint solving (v2).
//!
//! This version wraps a flat ParameterStore and supports all geometric primitives
//! (not just points) as first-class entities.

use super::params::{ParameterStore, ConstraintId, EntityHandle};
use super::entity::EntityKind;
use super::constraint::Constraint;
use crate::problem::Problem;

/// A system of geometric constraints to be solved.
///
/// The constraint system manages:
/// - A flat parameter store (all entities represented as parameter ranges)
/// - Fixed/free parameter flags
/// - Geometric constraints operating on parameters
///
/// It implements the `Problem` trait for use with solverang solvers.
pub struct ConstraintSystem {
    /// The flat parameter store containing all entity parameters.
    params: ParameterStore,
    /// The constraints in the system.
    constraints: Vec<Box<dyn Constraint>>,
    /// Name of the system for debugging.
    name: String,
    /// Next constraint ID to allocate.
    next_constraint_id: usize,
}

impl Default for ConstraintSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSystem {
    /// Create a new empty constraint system.
    pub fn new() -> Self {
        Self {
            params: ParameterStore::new(),
            constraints: Vec::new(),
            name: String::from("ConstraintSystem"),
            next_constraint_id: 0,
        }
    }

    /// Create a constraint system with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            params: ParameterStore::new(),
            constraints: Vec::new(),
            name: name.into(),
            next_constraint_id: 0,
        }
    }

    /// Get a reference to the parameter store.
    pub fn params(&self) -> &ParameterStore {
        &self.params
    }

    /// Get a mutable reference to the parameter store.
    pub fn params_mut(&mut self) -> &mut ParameterStore {
        &mut self.params
    }

    /// Set the name of this constraint system.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    // === Entity convenience methods (delegate to params) ===

    /// Add a 2D point.
    pub fn add_point_2d(&mut self, x: f64, y: f64) -> EntityHandle {
        self.params.add_point_2d(x, y)
    }

    /// Add a 3D point.
    pub fn add_point_3d(&mut self, x: f64, y: f64, z: f64) -> EntityHandle {
        self.params.add_point_3d(x, y, z)
    }

    /// Add a 2D circle (center x, y, radius).
    pub fn add_circle_2d(&mut self, cx: f64, cy: f64, r: f64) -> EntityHandle {
        self.params.add_circle_2d(cx, cy, r)
    }

    /// Add a 2D line (two endpoints).
    pub fn add_line_2d(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) -> EntityHandle {
        self.params.add_line_2d(x1, y1, x2, y2)
    }

    /// Add a 2D arc (center, radius, start angle, end angle).
    pub fn add_arc_2d(&mut self, cx: f64, cy: f64, r: f64, start_angle: f64, end_angle: f64) -> EntityHandle {
        self.params.add_arc_2d(cx, cy, r, start_angle, end_angle)
    }

    /// Add a 2D cubic Bezier curve (4 control points).
    pub fn add_cubic_bezier_2d(&mut self, points: [[f64; 2]; 4]) -> EntityHandle {
        self.params.add_cubic_bezier_2d(points)
    }

    /// Add a 2D ellipse (center, semi-major, semi-minor, rotation).
    pub fn add_ellipse_2d(&mut self, cx: f64, cy: f64, rx: f64, ry: f64, rotation: f64) -> EntityHandle {
        self.params.add_ellipse_2d(cx, cy, rx, ry, rotation)
    }

    /// Add a scalar parameter.
    pub fn add_scalar(&mut self, value: f64) -> EntityHandle {
        self.params.add_scalar(value)
    }

    /// Add an entity with custom kind and values.
    pub fn add_entity(&mut self, kind: EntityKind, values: &[f64]) -> EntityHandle {
        self.params.add_entity(kind, values)
    }

    // === Fix/free operations ===

    /// Fix all parameters of an entity (make them driven/immutable).
    pub fn fix_entity(&mut self, handle: &EntityHandle) {
        self.params.fix_entity(handle);
    }

    /// Fix a single parameter by index.
    pub fn fix_param(&mut self, idx: usize) {
        self.params.fix_param(idx);
    }

    /// Free all parameters of an entity (make them solvable).
    pub fn free_entity(&mut self, handle: &EntityHandle) {
        self.params.free_entity(handle);
    }

    /// Free a single parameter by index.
    pub fn free_param(&mut self, idx: usize) {
        self.params.free_param(idx);
    }

    // === Constraint operations ===

    /// Allocate and return a new constraint ID.
    pub fn next_constraint_id(&mut self) -> ConstraintId {
        let id = ConstraintId(self.next_constraint_id);
        self.next_constraint_id += 1;
        id
    }

    /// Add a constraint to the system.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) {
        self.constraints.push(constraint);
    }

    /// Number of constraints in the system.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    // === Entity/param inspection ===

    /// Get all entity handles sorted by EntityId.
    pub fn handles(&self) -> Vec<EntityHandle> {
        self.params.entity_handles()
    }

    /// Check if a specific parameter is fixed (driven/immutable).
    pub fn is_param_fixed(&self, idx: usize) -> bool {
        self.params.is_fixed(idx)
    }

    // === System statistics ===

    /// Total number of equations from all constraints.
    pub fn equation_count(&self) -> usize {
        self.constraints.iter().map(|c| c.equation_count()).sum()
    }

    /// Number of entities in the system.
    pub fn entity_count(&self) -> usize {
        self.params.entity_count()
    }

    /// Total number of free (solvable) parameters.
    pub fn variable_count(&self) -> usize {
        self.params.free_param_count()
    }

    /// Degrees of freedom (DOF) = variables - equations.
    ///
    /// - DOF > 0: underconstrained (multiple solutions)
    /// - DOF = 0: exactly constrained (unique solution if consistent)
    /// - DOF < 0: overconstrained (may be inconsistent)
    pub fn degrees_of_freedom(&self) -> i32 {
        self.variable_count() as i32 - self.equation_count() as i32
    }

    /// Check if the system is well-constrained (DOF = 0).
    pub fn is_well_constrained(&self) -> bool {
        self.degrees_of_freedom() == 0
    }

    /// Check if the system is underconstrained (DOF > 0).
    pub fn is_underconstrained(&self) -> bool {
        self.degrees_of_freedom() > 0
    }

    /// Check if the system is overconstrained (DOF < 0).
    pub fn is_overconstrained(&self) -> bool {
        self.degrees_of_freedom() < 0
    }

    // === Value access ===

    /// Get the current values of all free parameters.
    pub fn current_values(&self) -> Vec<f64> {
        self.params.current_free_values()
    }

    /// Set the free parameter values from a compressed vector.
    pub fn set_values(&mut self, values: &[f64]) {
        self.params.set_free_values(values);
    }
}

impl Problem for ConstraintSystem {
    fn name(&self) -> &str {
        &self.name
    }

    fn residual_count(&self) -> usize {
        self.equation_count()
    }

    fn variable_count(&self) -> usize {
        self.variable_count()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // 1. Build free_indices from params
        let free_indices = self.params.free_indices();

        // 2. Create temp values: copy all params, then scatter x into free slots
        let mut temp_values = self.params.values().to_vec();
        for (i, &global_idx) in free_indices.iter().enumerate() {
            temp_values[global_idx] = x[i];
        }

        // 3. Evaluate all constraints against temp values
        let mut all_residuals = Vec::with_capacity(self.equation_count());
        for constraint in &self.constraints {
            all_residuals.extend(constraint.residuals(&temp_values));
        }

        all_residuals
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        // 1. Same temp values as residuals
        let free_indices = self.params.free_indices();
        let mut temp_values = self.params.values().to_vec();
        for (i, &global_idx) in free_indices.iter().enumerate() {
            temp_values[global_idx] = x[i];
        }

        // 2. Build a map: global_param_idx → free_variable_idx (or None if fixed)
        let mut global_to_free = vec![None; self.params.param_count()];
        for (free_idx, &global_idx) in free_indices.iter().enumerate() {
            global_to_free[global_idx] = Some(free_idx);
        }

        // 3. Evaluate all constraint jacobians and remap columns
        let mut all_entries = Vec::new();
        let mut row_offset = 0;

        for constraint in &self.constraints {
            let entries = constraint.jacobian(&temp_values);

            for (local_row, global_col, val) in entries {
                // Remap: global_col → free_col (skip fixed params)
                if let Some(free_col) = global_to_free.get(global_col).and_then(|&x| x) {
                    all_entries.push((row_offset + local_row, free_col, val));
                }
                // If global_col is fixed, we skip this entry entirely
            }

            row_offset += constraint.equation_count();
        }

        all_entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        self.current_values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::params::ConstraintId;

    // Test-only distance constraint
    struct TestDistanceConstraint {
        id: ConstraintId,
        p1_start: usize,
        p2_start: usize,
        dim: usize,
        target: f64,
        deps: Vec<usize>,
    }

    impl TestDistanceConstraint {
        fn new(id: ConstraintId, p1: &EntityHandle, p2: &EntityHandle, target: f64) -> Self {
            let dim = p1.kind.param_count().min(p2.kind.param_count());
            let mut deps = Vec::new();
            for i in 0..dim {
                deps.push(p1.param(i));
                deps.push(p2.param(i));
            }
            Self {
                id,
                p1_start: p1.params.start,
                p2_start: p2.params.start,
                dim,
                target,
                deps,
            }
        }
    }

    impl Constraint for TestDistanceConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }

        fn name(&self) -> &'static str {
            "TestDistance"
        }

        fn equation_count(&self) -> usize {
            1
        }

        fn dependencies(&self) -> &[usize] {
            &self.deps
        }

        fn residuals(&self, params: &[f64]) -> Vec<f64> {
            let mut sum_sq = 0.0;
            for i in 0..self.dim {
                let diff = params[self.p1_start + i] - params[self.p2_start + i];
                sum_sq += diff * diff;
            }
            let dist = sum_sq.sqrt();
            vec![dist - self.target]
        }

        fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
            let mut sum_sq = 0.0;
            for i in 0..self.dim {
                let diff = params[self.p1_start + i] - params[self.p2_start + i];
                sum_sq += diff * diff;
            }
            let dist = sum_sq.sqrt().max(1e-10);

            let mut entries = Vec::new();
            for i in 0..self.dim {
                let diff = params[self.p1_start + i] - params[self.p2_start + i];
                let partial = diff / dist;
                entries.push((0, self.p1_start + i, partial));
                entries.push((0, self.p2_start + i, -partial));
            }
            entries
        }
    }

    #[test]
    fn test_system_creation() {
        let system = ConstraintSystem::new();
        assert_eq!(system.entity_count(), 0);
        assert_eq!(system.constraint_count(), 0);
        assert_eq!(system.variable_count(), 0);
    }

    #[test]
    fn test_add_entities() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(0.0, 0.0);
        let p2 = system.add_point_2d(10.0, 0.0);
        let circle = system.add_circle_2d(5.0, 5.0, 3.0);

        assert_eq!(system.entity_count(), 3);
        assert_eq!(system.params().param_count(), 7); // 2 + 2 + 3
        assert_eq!(system.variable_count(), 7); // All free

        assert_eq!(p1.kind, EntityKind::Point2D);
        assert_eq!(p2.kind, EntityKind::Point2D);
        assert_eq!(circle.kind, EntityKind::Circle2D);
    }

    #[test]
    fn test_fix_and_free() {
        let mut system = ConstraintSystem::new();
        let p = system.add_point_2d(1.0, 2.0);

        assert_eq!(system.variable_count(), 2);

        system.fix_entity(&p);
        assert_eq!(system.variable_count(), 0);

        system.free_entity(&p);
        assert_eq!(system.variable_count(), 2);

        // Fix only x coordinate
        system.fix_param(p.param(0));
        assert_eq!(system.variable_count(), 1);
    }

    #[test]
    fn test_dof_calculation() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(0.0, 0.0);
        let p2 = system.add_point_2d(10.0, 0.0);

        // 4 variables, 0 equations
        assert_eq!(system.degrees_of_freedom(), 4);
        assert!(system.is_underconstrained());

        // Add distance constraint (1 equation)
        let id = system.next_constraint_id();
        let constraint = TestDistanceConstraint::new(id, &p1, &p2, 10.0);
        system.add_constraint(Box::new(constraint));

        // 4 variables, 1 equation
        assert_eq!(system.degrees_of_freedom(), 3);

        // Fix first point (removes 2 variables)
        system.fix_entity(&p1);
        assert_eq!(system.degrees_of_freedom(), 1);

        // Add another distance constraint
        let p3 = system.add_point_2d(5.0, 5.0);
        let id2 = system.next_constraint_id();
        let constraint2 = TestDistanceConstraint::new(id2, &p2, &p3, 5.0);
        system.add_constraint(Box::new(constraint2));

        // 2 + 2 variables (p2, p3), 2 equations
        assert_eq!(system.degrees_of_freedom(), 2);
    }

    #[test]
    fn test_problem_trait_residuals() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(0.0, 0.0);
        let p2 = system.add_point_2d(3.0, 4.0);

        system.fix_entity(&p1);

        let id = system.next_constraint_id();
        let constraint = TestDistanceConstraint::new(id, &p1, &p2, 5.0);
        system.add_constraint(Box::new(constraint));

        // Test Problem trait
        assert_eq!(system.residual_count(), 1);
        assert_eq!(system.variable_count(), 2); // Only p2's x,y

        // At solution (3, 4), distance is exactly 5
        let x = vec![3.0, 4.0];
        let residuals = system.residuals(&x);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);

        // At a different point, residual should be non-zero
        let x2 = vec![1.0, 1.0];
        let residuals2 = system.residuals(&x2);
        let expected_dist = ((1.0 * 1.0 + 1.0 * 1.0) as f64).sqrt();
        assert!((residuals2[0] - (expected_dist - 5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_problem_trait_jacobian() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(0.0, 0.0);
        let p2 = system.add_point_2d(3.0, 4.0);

        system.fix_entity(&p1);

        let id = system.next_constraint_id();
        let constraint = TestDistanceConstraint::new(id, &p1, &p2, 5.0);
        system.add_constraint(Box::new(constraint));

        let x = vec![3.0, 4.0];
        let jac = system.jacobian(&x);

        // Should have 2 entries (partial wrt p2.x and p2.y)
        // Partials wrt p1 are filtered out because p1 is fixed
        assert_eq!(jac.len(), 2);

        // At (3,4) with distance 5: ∂dist/∂x = 3/5 = 0.6, ∂dist/∂y = 4/5 = 0.8
        let mut found_x = false;
        let mut found_y = false;
        for (row, col, val) in jac {
            assert_eq!(row, 0); // Single equation
            if col == 0 {
                // Free variable 0 is p2.x
                assert!((val - 0.6).abs() < 1e-10);
                found_x = true;
            } else if col == 1 {
                // Free variable 1 is p2.y
                assert!((val - 0.8).abs() < 1e-10);
                found_y = true;
            }
        }
        assert!(found_x && found_y);
    }

    #[test]
    fn test_jacobian_column_remapping() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(0.0, 0.0); // params 0, 1
        let p2 = system.add_point_2d(3.0, 4.0); // params 2, 3

        // Fix p1.x (param 0), so free params are: [1, 2, 3]
        system.fix_param(p1.param(0));

        let id = system.next_constraint_id();
        let constraint = TestDistanceConstraint::new(id, &p1, &p2, 5.0);
        system.add_constraint(Box::new(constraint));

        assert_eq!(system.variable_count(), 3); // p1.y, p2.x, p2.y

        let x = vec![0.0, 3.0, 4.0]; // p1.y=0, p2.x=3, p2.y=4
        let jac = system.jacobian(&x);

        // Should have 3 entries:
        // - (0, 0, ∂/∂p1.y) where col 0 maps to global param 1
        // - (0, 1, ∂/∂p2.x) where col 1 maps to global param 2
        // - (0, 2, ∂/∂p2.y) where col 2 maps to global param 3
        assert_eq!(jac.len(), 3);

        // Columns should be 0, 1, 2 (remapped from global 1, 2, 3)
        // Note: entry order depends on constraint iteration, so sort before comparing
        let mut cols: Vec<_> = jac.iter().map(|(_, col, _)| *col).collect();
        cols.sort();
        assert_eq!(cols, vec![0, 1, 2]);
    }

    #[test]
    fn test_current_and_set_values() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(1.0, 2.0);
        let _p2 = system.add_point_2d(3.0, 4.0);

        system.fix_entity(&p1);

        let values = system.current_values();
        assert_eq!(values, vec![3.0, 4.0]); // Only p2

        system.set_values(&[10.0, 20.0]);
        let new_values = system.current_values();
        assert_eq!(new_values, vec![10.0, 20.0]);

        // p1 should remain unchanged
        assert_eq!(system.params().get_value(p1.param(0)), 1.0);
        assert_eq!(system.params().get_value(p1.param(1)), 2.0);
    }

    #[test]
    fn test_well_constrained() {
        let mut system = ConstraintSystem::new();

        let p1 = system.add_point_2d(0.0, 0.0);
        let p2 = system.add_point_2d(10.0, 0.0);

        system.fix_entity(&p1);

        // 2 variables, 0 equations
        assert!(system.is_underconstrained());

        let id1 = system.next_constraint_id();
        let c1 = TestDistanceConstraint::new(id1, &p1, &p2, 10.0);
        system.add_constraint(Box::new(c1));

        // 2 variables, 1 equation
        assert!(system.is_underconstrained());

        // Add a second constraint to make it exactly constrained
        // (In reality, we'd need a different constraint type, but for testing DOF...)
        let p3 = system.add_point_2d(5.0, 5.0);
        system.fix_entity(&p3);
        let id2 = system.next_constraint_id();
        let c2 = TestDistanceConstraint::new(id2, &p2, &p3, 5.0);
        system.add_constraint(Box::new(c2));

        // 2 variables, 2 equations
        assert!(system.is_well_constrained());
        assert!(!system.is_underconstrained());
        assert!(!system.is_overconstrained());
    }

    #[test]
    fn test_mixed_entity_types() {
        let mut system = ConstraintSystem::new();

        let _point = system.add_point_2d(0.0, 0.0);
        let circle = system.add_circle_2d(10.0, 10.0, 5.0);
        let line = system.add_line_2d(0.0, 0.0, 20.0, 0.0);

        assert_eq!(system.entity_count(), 3);
        assert_eq!(system.params().param_count(), 2 + 3 + 4); // 9 total

        // Fix the circle's radius only
        system.fix_param(circle.param(2));
        assert_eq!(system.variable_count(), 8); // 9 - 1

        // Fix the entire line
        system.fix_entity(&line);
        assert_eq!(system.variable_count(), 4); // 9 - 1 - 4
    }
}
