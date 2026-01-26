//! Constraint system for geometric constraint solving.

use crate::problem::Problem;
use super::point::Point;
use super::constraints::GeometricConstraint;

/// A system of geometric constraints to be solved.
///
/// The constraint system manages:
/// - A collection of points (variables)
/// - Fixed point markers
/// - Geometric constraints between points
///
/// It implements the `Problem` trait for use with solverang solvers.
///
/// # Example
///
/// ```rust
/// use solverang::geometry::{ConstraintSystem, Point2D};
/// use solverang::geometry::constraints::{DistanceConstraint, FixedConstraint};
///
/// let mut system = ConstraintSystem::<2>::new();
///
/// // Add three points
/// let p0 = system.add_point(Point2D::new(0.0, 0.0));
/// let p1 = system.add_point(Point2D::new(5.0, 0.0));
/// let p2 = system.add_point(Point2D::new(2.0, 3.0));
///
/// // Fix the first point
/// system.fix_point(p0);
///
/// // Add constraints
/// system.add_constraint(Box::new(DistanceConstraint::<2>::new(p0, p1, 10.0)));
/// system.add_constraint(Box::new(DistanceConstraint::<2>::new(p1, p2, 8.0)));
/// system.add_constraint(Box::new(DistanceConstraint::<2>::new(p2, p0, 6.0)));
///
/// // Check degrees of freedom
/// println!("DOF: {}", system.degrees_of_freedom());
/// ```
pub struct ConstraintSystem<const D: usize> {
    /// The points in the system.
    points: Vec<Point<D>>,
    /// Whether each point is fixed (true) or free (false).
    fixed: Vec<bool>,
    /// The constraints in the system.
    constraints: Vec<Box<dyn GeometricConstraint<D>>>,
    /// Name of the system for debugging.
    name: String,
}

impl<const D: usize> Default for ConstraintSystem<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ConstraintSystem<D> {
    /// Create a new empty constraint system.
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            fixed: Vec::new(),
            constraints: Vec::new(),
            name: String::from("ConstraintSystem"),
        }
    }

    /// Create a constraint system with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            points: Vec::new(),
            fixed: Vec::new(),
            constraints: Vec::new(),
            name: name.into(),
        }
    }

    /// Add a free (movable) point to the system.
    ///
    /// Returns the index of the added point.
    pub fn add_point(&mut self, point: Point<D>) -> usize {
        let idx = self.points.len();
        self.points.push(point);
        self.fixed.push(false);
        idx
    }

    /// Add a fixed (immovable) point to the system.
    ///
    /// Returns the index of the added point.
    pub fn add_point_fixed(&mut self, point: Point<D>) -> usize {
        let idx = self.points.len();
        self.points.push(point);
        self.fixed.push(true);
        idx
    }

    /// Mark a point as fixed.
    pub fn fix_point(&mut self, index: usize) {
        if let Some(f) = self.fixed.get_mut(index) {
            *f = true;
        }
    }

    /// Mark a point as free.
    pub fn free_point(&mut self, index: usize) {
        if let Some(f) = self.fixed.get_mut(index) {
            *f = false;
        }
    }

    /// Check if a point is fixed.
    pub fn is_fixed(&self, index: usize) -> bool {
        self.fixed.get(index).copied().unwrap_or(false)
    }

    /// Get a point by index.
    pub fn get_point(&self, index: usize) -> Option<&Point<D>> {
        self.points.get(index)
    }

    /// Get a mutable reference to a point by index.
    pub fn get_point_mut(&mut self, index: usize) -> Option<&mut Point<D>> {
        self.points.get_mut(index)
    }

    /// Set a point's position.
    pub fn set_point(&mut self, index: usize, point: Point<D>) {
        if let Some(p) = self.points.get_mut(index) {
            *p = point;
        }
    }

    /// Get all points.
    pub fn points(&self) -> &[Point<D>] {
        &self.points
    }

    /// Number of points in the system.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of free (non-fixed) points.
    pub fn free_point_count(&self) -> usize {
        self.fixed.iter().filter(|&&f| !f).count()
    }

    /// Number of fixed points.
    pub fn fixed_point_count(&self) -> usize {
        self.fixed.iter().filter(|&&f| f).count()
    }

    /// Add a constraint to the system.
    pub fn add_constraint(&mut self, constraint: Box<dyn GeometricConstraint<D>>) {
        self.constraints.push(constraint);
    }

    /// Number of constraints in the system.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Total number of equations from all constraints.
    pub fn equation_count(&self) -> usize {
        self.constraints.iter().map(|c| c.equation_count()).sum()
    }

    /// Total number of scalar variables (D per free point).
    pub fn total_variable_count(&self) -> usize {
        self.free_point_count() * D
    }

    /// Degrees of freedom (DOF) = variables - equations.
    ///
    /// - DOF > 0: underconstrained (multiple solutions)
    /// - DOF = 0: exactly constrained (unique solution if consistent)
    /// - DOF < 0: overconstrained (may be inconsistent)
    pub fn degrees_of_freedom(&self) -> i32 {
        self.total_variable_count() as i32 - self.equation_count() as i32
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

    /// Build mapping from free point indices to variable indices.
    fn build_free_point_map(&self) -> Vec<Option<usize>> {
        let mut var_idx = 0;
        self.fixed
            .iter()
            .map(|&is_fixed| {
                if is_fixed {
                    None
                } else {
                    let idx = var_idx;
                    var_idx += 1;
                    Some(idx)
                }
            })
            .collect()
    }

    /// Extract current values as a flat variable vector.
    ///
    /// Only includes free (non-fixed) point coordinates.
    pub fn current_values(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.total_variable_count());
        for (i, point) in self.points.iter().enumerate() {
            if !self.fixed.get(i).copied().unwrap_or(false) {
                for k in 0..D {
                    values.push(point.get(k));
                }
            }
        }
        values
    }

    /// Update points from a flat variable vector.
    ///
    /// Only updates free (non-fixed) points.
    pub fn set_values(&mut self, values: &[f64]) {
        let mut val_idx = 0;
        for (i, point) in self.points.iter_mut().enumerate() {
            if !self.fixed.get(i).copied().unwrap_or(false) {
                for k in 0..D {
                    if let Some(&v) = values.get(val_idx) {
                        point.set(k, v);
                    }
                    val_idx += 1;
                }
            }
        }
    }

    /// Evaluate all constraint residuals.
    pub fn evaluate_residuals(&self) -> Vec<f64> {
        let mut residuals = Vec::with_capacity(self.equation_count());
        for constraint in &self.constraints {
            residuals.extend(constraint.residuals(&self.points));
        }
        residuals
    }

    /// Compute the full Jacobian as sparse triplets.
    ///
    /// Column indices are remapped to only include free point coordinates.
    pub fn compute_jacobian(&self) -> Vec<(usize, usize, f64)> {
        let free_map = self.build_free_point_map();

        let mut all_entries = Vec::new();
        let mut row_offset = 0;

        for constraint in &self.constraints {
            let entries = constraint.jacobian(&self.points);

            for (local_row, col, val) in entries {
                // col is point_idx * D + coord
                let point_idx = col / D;
                let coord = col % D;

                // Skip fixed points
                if let Some(Some(free_idx)) = free_map.get(point_idx) {
                    let new_col = free_idx * D + coord;
                    all_entries.push((row_offset + local_row, new_col, val));
                }
            }

            row_offset += constraint.equation_count();
        }

        all_entries
    }

    /// Clear all constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Clear all points and constraints.
    pub fn clear(&mut self) {
        self.points.clear();
        self.fixed.clear();
        self.constraints.clear();
    }
}

impl<const D: usize> Problem for ConstraintSystem<D> {
    fn name(&self) -> &str {
        &self.name
    }

    fn residual_count(&self) -> usize {
        self.equation_count()
    }

    fn variable_count(&self) -> usize {
        self.total_variable_count()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Create a temporary system with updated values
        let mut temp_points = self.points.clone();

        // Update free points from x
        let mut val_idx = 0;
        for (i, point) in temp_points.iter_mut().enumerate() {
            if !self.fixed.get(i).copied().unwrap_or(false) {
                for k in 0..D {
                    if let Some(&v) = x.get(val_idx) {
                        point.set(k, v);
                    }
                    val_idx += 1;
                }
            }
        }

        // Evaluate constraints with temporary points
        let mut residuals = Vec::with_capacity(self.equation_count());
        for constraint in &self.constraints {
            residuals.extend(constraint.residuals(&temp_points));
        }
        residuals
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Create temporary points with updated values
        let mut temp_points = self.points.clone();

        let mut val_idx = 0;
        for (i, point) in temp_points.iter_mut().enumerate() {
            if !self.fixed.get(i).copied().unwrap_or(false) {
                for k in 0..D {
                    if let Some(&v) = x.get(val_idx) {
                        point.set(k, v);
                    }
                    val_idx += 1;
                }
            }
        }

        // Build Jacobian with temporary points
        let free_map = self.build_free_point_map();

        let mut all_entries = Vec::new();
        let mut row_offset = 0;

        for constraint in &self.constraints {
            let entries = constraint.jacobian(&temp_points);

            for (local_row, col, val) in entries {
                let point_idx = col / D;
                let coord = col % D;

                if let Some(Some(free_idx)) = free_map.get(point_idx) {
                    let new_col = free_idx * D + coord;
                    all_entries.push((row_offset + local_row, new_col, val));
                }
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
    use crate::geometry::point::{Point2D, Point3D};
    use crate::geometry::constraints::DistanceConstraint;

    #[test]
    fn test_system_creation() {
        let system = ConstraintSystem::<2>::new();
        assert_eq!(system.point_count(), 0);
        assert_eq!(system.constraint_count(), 0);
    }

    #[test]
    fn test_add_points() {
        let mut system = ConstraintSystem::<2>::new();
        let p0 = system.add_point(Point2D::new(0.0, 0.0));
        let p1 = system.add_point_fixed(Point2D::new(10.0, 0.0));

        assert_eq!(p0, 0);
        assert_eq!(p1, 1);
        assert_eq!(system.point_count(), 2);
        assert_eq!(system.free_point_count(), 1);
        assert_eq!(system.fixed_point_count(), 1);
        assert!(!system.is_fixed(0));
        assert!(system.is_fixed(1));
    }

    #[test]
    fn test_fix_free_point() {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(Point2D::new(0.0, 0.0));

        assert!(!system.is_fixed(0));
        system.fix_point(0);
        assert!(system.is_fixed(0));
        system.free_point(0);
        assert!(!system.is_fixed(0));
    }

    #[test]
    fn test_dof_calculation() {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(Point2D::new(0.0, 0.0)); // 2 DOF
        system.add_point(Point2D::new(5.0, 0.0)); // 2 DOF
        system.add_point(Point2D::new(2.0, 3.0)); // 2 DOF
        // Total: 6 DOF

        assert_eq!(system.total_variable_count(), 6);
        assert_eq!(system.degrees_of_freedom(), 6);

        // Add distance constraint (1 equation)
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 5.0)));
        assert_eq!(system.degrees_of_freedom(), 5);

        // Fix a point (removes 2 DOF)
        system.fix_point(0);
        assert_eq!(system.degrees_of_freedom(), 3);
    }

    #[test]
    fn test_values_extraction() {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(Point2D::new(1.0, 2.0));
        system.add_point_fixed(Point2D::new(3.0, 4.0));
        system.add_point(Point2D::new(5.0, 6.0));

        let values = system.current_values();
        // Only free points: (1,2) and (5,6)
        assert_eq!(values, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_values_update() {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(Point2D::new(0.0, 0.0));
        system.add_point_fixed(Point2D::new(10.0, 10.0));
        system.add_point(Point2D::new(0.0, 0.0));

        system.set_values(&[1.0, 2.0, 3.0, 4.0]);

        // Point 0 should be updated
        assert_eq!(system.get_point(0).map(|p| p.get(0)), Some(1.0));
        assert_eq!(system.get_point(0).map(|p| p.get(1)), Some(2.0));

        // Point 1 is fixed, should not change
        assert_eq!(system.get_point(1).map(|p| p.get(0)), Some(10.0));
        assert_eq!(system.get_point(1).map(|p| p.get(1)), Some(10.0));

        // Point 2 should be updated
        assert_eq!(system.get_point(2).map(|p| p.get(0)), Some(3.0));
        assert_eq!(system.get_point(2).map(|p| p.get(1)), Some(4.0));
    }

    #[test]
    fn test_problem_trait() {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point_fixed(Point2D::new(0.0, 0.0));
        system.add_point(Point2D::new(3.0, 4.0)); // distance = 5

        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 5.0)));

        // Test Problem trait
        assert_eq!(system.residual_count(), 1);
        assert_eq!(system.variable_count(), 2); // Only point 1's x,y

        let x = vec![3.0, 4.0];
        let residuals = system.residuals(&x);
        assert!(residuals[0].abs() < 1e-10);

        let jac = system.jacobian(&x);
        assert_eq!(jac.len(), 2); // 2 non-zero entries
    }

    #[test]
    fn test_3d_system() {
        let mut system = ConstraintSystem::<3>::new();
        system.add_point_fixed(Point3D::new(0.0, 0.0, 0.0));
        system.add_point(Point3D::new(1.0, 2.0, 2.0)); // distance = 3

        system.add_constraint(Box::new(DistanceConstraint::<3>::new(0, 1, 3.0)));

        assert_eq!(system.variable_count(), 3); // Only point 1's x,y,z

        let x = vec![1.0, 2.0, 2.0];
        let residuals = system.residuals(&x);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_clear() {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(Point2D::new(0.0, 0.0));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 0, 0.0)));

        system.clear();
        assert_eq!(system.point_count(), 0);
        assert_eq!(system.constraint_count(), 0);
    }
}
