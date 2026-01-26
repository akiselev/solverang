//! Lowering implementations for geometric constraints.
//!
//! This module provides [`Lowerable`] implementations for geometric constraint types,
//! enabling JIT compilation of constraint systems.

use super::lower::{Lowerable, LoweringContext, OpcodeEmitter};

#[cfg(feature = "geometry")]
use crate::geometry::constraints::{
    AngleConstraint, CoincidentConstraint, DistanceConstraint, FixedConstraint,
    HorizontalConstraint, VerticalConstraint,
};

#[cfg(feature = "geometry")]
use crate::geometry::point::MIN_EPSILON;

/// Lowering implementation for DistanceConstraint.
///
/// Residual: sqrt((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + ...) - target_distance
/// Jacobian: d/d(p_i[k]) = +-((p2[k] - p1[k]) / dist)
#[cfg(feature = "geometry")]
impl Lowerable for DistanceConstraint<2> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let residual_idx = ctx.current_residual;

        // Load point coordinates
        let p1_x = emitter.load_var(ctx.var_index(self.point1, 0));
        let p1_y = emitter.load_var(ctx.var_index(self.point1, 1));
        let p2_x = emitter.load_var(ctx.var_index(self.point2, 0));
        let p2_y = emitter.load_var(ctx.var_index(self.point2, 1));

        // Compute differences
        let dx = emitter.sub(p2_x, p1_x);
        let dy = emitter.sub(p2_y, p1_y);

        // Compute squared distances
        let dx2 = emitter.square(dx);
        let dy2 = emitter.square(dy);

        // Compute distance
        let dist_sq = emitter.add(dx2, dy2);
        let dist = emitter.safe_distance(dist_sq, MIN_EPSILON);

        // Compute residual: dist - target
        let target = emitter.const_f64(self.target);
        let residual = emitter.sub(dist, target);

        emitter.store_residual(residual_idx, residual);
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        emitter.set_residual_index(ctx.current_residual);

        // Load point coordinates
        let p1_x = emitter.load_var(ctx.var_index(self.point1, 0));
        let p1_y = emitter.load_var(ctx.var_index(self.point1, 1));
        let p2_x = emitter.load_var(ctx.var_index(self.point2, 0));
        let p2_y = emitter.load_var(ctx.var_index(self.point2, 1));

        // Compute differences
        let dx = emitter.sub(p2_x, p1_x);
        let dy = emitter.sub(p2_y, p1_y);

        // Compute distance
        let dx2 = emitter.square(dx);
        let dy2 = emitter.square(dy);
        let dist_sq = emitter.add(dx2, dy2);
        let dist = emitter.safe_distance(dist_sq, MIN_EPSILON);

        // Jacobian: d(residual)/d(p1.x) = -dx/dist
        // Jacobian: d(residual)/d(p1.y) = -dy/dist
        // Jacobian: d(residual)/d(p2.x) = dx/dist
        // Jacobian: d(residual)/d(p2.y) = dy/dist

        let dx_over_dist = emitter.div(dx, dist);
        let dy_over_dist = emitter.div(dy, dist);
        let neg_dx_over_dist = emitter.neg(dx_over_dist);
        let neg_dy_over_dist = emitter.neg(dy_over_dist);

        emitter.store_jacobian_current(ctx.var_index(self.point1, 0), neg_dx_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point1, 1), neg_dy_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 0), dx_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 1), dy_over_dist);
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }
}

/// Lowering implementation for 3D DistanceConstraint.
#[cfg(feature = "geometry")]
impl Lowerable for DistanceConstraint<3> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let residual_idx = ctx.current_residual;

        // Load point coordinates
        let p1_x = emitter.load_var(ctx.var_index(self.point1, 0));
        let p1_y = emitter.load_var(ctx.var_index(self.point1, 1));
        let p1_z = emitter.load_var(ctx.var_index(self.point1, 2));
        let p2_x = emitter.load_var(ctx.var_index(self.point2, 0));
        let p2_y = emitter.load_var(ctx.var_index(self.point2, 1));
        let p2_z = emitter.load_var(ctx.var_index(self.point2, 2));

        // Compute differences
        let dx = emitter.sub(p2_x, p1_x);
        let dy = emitter.sub(p2_y, p1_y);
        let dz = emitter.sub(p2_z, p1_z);

        // Compute squared distances
        let dx2 = emitter.square(dx);
        let dy2 = emitter.square(dy);
        let dz2 = emitter.square(dz);

        // Compute distance
        let dist_sq_xy = emitter.add(dx2, dy2);
        let dist_sq = emitter.add(dist_sq_xy, dz2);
        let dist = emitter.safe_distance(dist_sq, MIN_EPSILON);

        // Compute residual: dist - target
        let target = emitter.const_f64(self.target);
        let residual = emitter.sub(dist, target);

        emitter.store_residual(residual_idx, residual);
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        emitter.set_residual_index(ctx.current_residual);

        // Load point coordinates
        let p1_x = emitter.load_var(ctx.var_index(self.point1, 0));
        let p1_y = emitter.load_var(ctx.var_index(self.point1, 1));
        let p1_z = emitter.load_var(ctx.var_index(self.point1, 2));
        let p2_x = emitter.load_var(ctx.var_index(self.point2, 0));
        let p2_y = emitter.load_var(ctx.var_index(self.point2, 1));
        let p2_z = emitter.load_var(ctx.var_index(self.point2, 2));

        // Compute differences
        let dx = emitter.sub(p2_x, p1_x);
        let dy = emitter.sub(p2_y, p1_y);
        let dz = emitter.sub(p2_z, p1_z);

        // Compute distance
        let dx2 = emitter.square(dx);
        let dy2 = emitter.square(dy);
        let dz2 = emitter.square(dz);
        let dist_sq_xy = emitter.add(dx2, dy2);
        let dist_sq = emitter.add(dist_sq_xy, dz2);
        let dist = emitter.safe_distance(dist_sq, MIN_EPSILON);

        // Jacobian entries
        let dx_over_dist = emitter.div(dx, dist);
        let dy_over_dist = emitter.div(dy, dist);
        let dz_over_dist = emitter.div(dz, dist);
        let neg_dx_over_dist = emitter.neg(dx_over_dist);
        let neg_dy_over_dist = emitter.neg(dy_over_dist);
        let neg_dz_over_dist = emitter.neg(dz_over_dist);

        emitter.store_jacobian_current(ctx.var_index(self.point1, 0), neg_dx_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point1, 1), neg_dy_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point1, 2), neg_dz_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 0), dx_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 1), dy_over_dist);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 2), dz_over_dist);
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }
}

/// Lowering implementation for CoincidentConstraint.
///
/// Residuals: p2[k] - p1[k] = 0 for each dimension k
/// Jacobian: constant -1 and +1 entries
#[cfg(feature = "geometry")]
impl Lowerable for CoincidentConstraint<2> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;

        // For each dimension, residual = p2[k] - p1[k]
        for k in 0..2 {
            let p1_k = emitter.load_var(ctx.var_index(self.point1, k));
            let p2_k = emitter.load_var(ctx.var_index(self.point2, k));
            let residual = emitter.sub(p2_k, p1_k);
            emitter.store_residual(base_residual + k as u32, residual);
        }
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;

        // Jacobian is constant: d(residual_k)/d(p1[k]) = -1, d(residual_k)/d(p2[k]) = 1
        let neg_one = emitter.const_f64(-1.0);
        let one = emitter.const_f64(1.0);

        for k in 0..2 {
            emitter.set_residual_index(base_residual + k as u32);
            emitter.store_jacobian_current(ctx.var_index(self.point1, k), neg_one);
            emitter.store_jacobian_current(ctx.var_index(self.point2, k), one);
        }
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }
}

#[cfg(feature = "geometry")]
impl Lowerable for CoincidentConstraint<3> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;

        for k in 0..3 {
            let p1_k = emitter.load_var(ctx.var_index(self.point1, k));
            let p2_k = emitter.load_var(ctx.var_index(self.point2, k));
            let residual = emitter.sub(p2_k, p1_k);
            emitter.store_residual(base_residual + k as u32, residual);
        }
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;
        let neg_one = emitter.const_f64(-1.0);
        let one = emitter.const_f64(1.0);

        for k in 0..3 {
            emitter.set_residual_index(base_residual + k as u32);
            emitter.store_jacobian_current(ctx.var_index(self.point1, k), neg_one);
            emitter.store_jacobian_current(ctx.var_index(self.point2, k), one);
        }
    }

    fn residual_count(&self) -> usize {
        3
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }
}

/// Lowering implementation for FixedConstraint.
///
/// Residuals: p[k] - target[k] = 0 for each dimension k
/// Jacobian: constant 1 entries
#[cfg(feature = "geometry")]
impl Lowerable for FixedConstraint<2> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;

        for k in 0..2 {
            let p_k = emitter.load_var(ctx.var_index(self.point, k));
            let target_k = emitter.const_f64(self.target.get(k));
            let residual = emitter.sub(p_k, target_k);
            emitter.store_residual(base_residual + k as u32, residual);
        }
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;
        let one = emitter.const_f64(1.0);

        for k in 0..2 {
            emitter.set_residual_index(base_residual + k as u32);
            emitter.store_jacobian_current(ctx.var_index(self.point, k), one);
        }
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point]
    }
}

#[cfg(feature = "geometry")]
impl Lowerable for FixedConstraint<3> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;

        for k in 0..3 {
            let p_k = emitter.load_var(ctx.var_index(self.point, k));
            let target_k = emitter.const_f64(self.target.get(k));
            let residual = emitter.sub(p_k, target_k);
            emitter.store_residual(base_residual + k as u32, residual);
        }
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base_residual = ctx.current_residual;
        let one = emitter.const_f64(1.0);

        for k in 0..3 {
            emitter.set_residual_index(base_residual + k as u32);
            emitter.store_jacobian_current(ctx.var_index(self.point, k), one);
        }
    }

    fn residual_count(&self) -> usize {
        3
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point]
    }
}

/// Lowering implementation for HorizontalConstraint.
///
/// Residual: p2.y - p1.y = 0
/// Jacobian: d/d(p1.y) = -1, d/d(p2.y) = 1
#[cfg(feature = "geometry")]
impl Lowerable for HorizontalConstraint {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let p1_y = emitter.load_var(ctx.var_index(self.point1, 1));
        let p2_y = emitter.load_var(ctx.var_index(self.point2, 1));
        let residual = emitter.sub(p2_y, p1_y);
        emitter.store_residual(ctx.current_residual, residual);
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        emitter.set_residual_index(ctx.current_residual);
        let neg_one = emitter.const_f64(-1.0);
        let one = emitter.const_f64(1.0);
        emitter.store_jacobian_current(ctx.var_index(self.point1, 1), neg_one);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 1), one);
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }
}

/// Lowering implementation for VerticalConstraint.
///
/// Residual: p2.x - p1.x = 0
/// Jacobian: d/d(p1.x) = -1, d/d(p2.x) = 1
#[cfg(feature = "geometry")]
impl Lowerable for VerticalConstraint {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let p1_x = emitter.load_var(ctx.var_index(self.point1, 0));
        let p2_x = emitter.load_var(ctx.var_index(self.point2, 0));
        let residual = emitter.sub(p2_x, p1_x);
        emitter.store_residual(ctx.current_residual, residual);
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        emitter.set_residual_index(ctx.current_residual);
        let neg_one = emitter.const_f64(-1.0);
        let one = emitter.const_f64(1.0);
        emitter.store_jacobian_current(ctx.var_index(self.point1, 0), neg_one);
        emitter.store_jacobian_current(ctx.var_index(self.point2, 0), one);
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }
}

/// Lowering implementation for AngleConstraint.
///
/// Residual: dy * cos(theta) - dx * sin(theta) = 0
/// where dx = p2.x - p1.x, dy = p2.y - p1.y
#[cfg(feature = "geometry")]
impl Lowerable for AngleConstraint {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let p1_x = emitter.load_var(ctx.var_index(self.line_start, 0));
        let p1_y = emitter.load_var(ctx.var_index(self.line_start, 1));
        let p2_x = emitter.load_var(ctx.var_index(self.line_end, 0));
        let p2_y = emitter.load_var(ctx.var_index(self.line_end, 1));

        let dx = emitter.sub(p2_x, p1_x);
        let dy = emitter.sub(p2_y, p1_y);

        let cos_theta = emitter.const_f64(self.angle.cos());
        let sin_theta = emitter.const_f64(self.angle.sin());

        // residual = dy * cos(theta) - dx * sin(theta)
        let dy_cos = emitter.mul(dy, cos_theta);
        let dx_sin = emitter.mul(dx, sin_theta);
        let residual = emitter.sub(dy_cos, dx_sin);

        emitter.store_residual(ctx.current_residual, residual);
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        emitter.set_residual_index(ctx.current_residual);

        let cos_theta = emitter.const_f64(self.angle.cos());
        let sin_theta = emitter.const_f64(self.angle.sin());
        let neg_cos_theta = emitter.const_f64(-self.angle.cos());
        let neg_sin_theta = emitter.const_f64(-self.angle.sin());

        // d/d(p1.x) = sin(theta)
        // d/d(p1.y) = -cos(theta)
        // d/d(p2.x) = -sin(theta)
        // d/d(p2.y) = cos(theta)

        emitter.store_jacobian_current(ctx.var_index(self.line_start, 0), sin_theta);
        emitter.store_jacobian_current(ctx.var_index(self.line_start, 1), neg_cos_theta);
        emitter.store_jacobian_current(ctx.var_index(self.line_end, 0), neg_sin_theta);
        emitter.store_jacobian_current(ctx.var_index(self.line_end, 1), cos_theta);
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.line_start, self.line_end]
    }
}

#[cfg(all(test, feature = "geometry"))]
mod tests {
    use super::*;
    use crate::geometry::point::Point2D;
    use crate::jit::{CompiledConstraints, JITCompiler, lower_constraints};

    #[test]
    fn test_lower_distance_constraint() {
        let constraints = vec![DistanceConstraint::<2>::new(0, 1, 5.0)];

        let cc = lower_constraints(&constraints, 4, 2);

        assert_eq!(cc.n_residuals, 1);
        assert_eq!(cc.n_vars, 4);
        assert!(!cc.residual_ops.is_empty());
        assert!(!cc.jacobian_ops.is_empty());
    }

    #[test]
    fn test_lower_coincident_constraint() {
        let constraints = vec![CoincidentConstraint::<2>::new(0, 1)];

        let cc = lower_constraints(&constraints, 4, 2);

        assert_eq!(cc.n_residuals, 2);
        assert_eq!(cc.n_vars, 4);
    }

    #[test]
    fn test_lower_fixed_constraint() {
        let constraints = vec![FixedConstraint::<2>::new(0, Point2D::new(1.0, 2.0))];

        let cc = lower_constraints(&constraints, 2, 2);

        assert_eq!(cc.n_residuals, 2);
        assert_eq!(cc.n_vars, 2);
    }

    #[test]
    fn test_lower_horizontal_constraint() {
        let constraints = vec![HorizontalConstraint::new(0, 1)];

        let cc = lower_constraints(&constraints, 4, 2);

        assert_eq!(cc.n_residuals, 1);
        assert_eq!(cc.n_vars, 4);
    }

    #[test]
    fn test_jit_distance_correctness() {
        let constraints = vec![DistanceConstraint::<2>::new(0, 1, 5.0)];

        let cc = lower_constraints(&constraints, 4, 2);

        let mut compiler = JITCompiler::new().expect("compiler creation failed");
        let jit_fn = compiler.compile(&cc).expect("compilation failed");

        // Test point: (0, 0) and (3, 4) -> distance = 5, residual = 0
        let vars = [0.0, 0.0, 3.0, 4.0];
        let mut residuals = [0.0];

        jit_fn.evaluate_residuals(&vars, &mut residuals);

        assert!(
            residuals[0].abs() < 1e-10,
            "residual should be 0, got {}",
            residuals[0]
        );

        // Test point: (0, 0) and (6, 8) -> distance = 10, residual = 5
        let vars2 = [0.0, 0.0, 6.0, 8.0];
        jit_fn.evaluate_residuals(&vars2, &mut residuals);

        assert!(
            (residuals[0] - 5.0).abs() < 1e-10,
            "residual should be 5, got {}",
            residuals[0]
        );
    }
}
