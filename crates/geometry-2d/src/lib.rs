//! Reusable two-dimensional geometric constraint vocabulary.
//!
//! Points, segments, and circles carry only Solverang variable IDs. CAD,
//! illustration, and PCB consumers retain their own entity identity and policy.

#![forbid(unsafe_code)]

use solverang::{
    Constraint, ConstraintError, ConstraintId, ConstraintModel, ConstraintOutput, Relation,
    VariableId, VariableValues,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point2 {
    pub x: VariableId,
    pub y: VariableId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment2 {
    pub start: Point2,
    pub end: Point2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Circle2 {
    pub center: Point2,
    pub radius: VariableId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis2 {
    X,
    Y,
}

#[derive(Default)]
pub struct Geometry2d {
    model: ConstraintModel,
}

impl Geometry2d {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_point(&mut self, x: f64, y: f64) -> Result<Point2, ConstraintError> {
        Ok(Point2 {
            x: self.model.add_variable(x)?,
            y: self.model.add_variable(y)?,
        })
    }

    pub fn add_segment(&self, start: Point2, end: Point2) -> Segment2 {
        Segment2 { start, end }
    }

    pub fn add_circle(&mut self, center: Point2, radius: f64) -> Result<Circle2, ConstraintError> {
        if radius < 0.0 {
            return Err(ConstraintError::InvalidDefinition(
                "circle radius must be non-negative".into(),
            ));
        }
        Ok(Circle2 {
            center,
            radius: self.model.add_variable(radius)?,
        })
    }

    pub fn constrain(
        &mut self,
        constraint: impl Constraint + 'static,
    ) -> Result<ConstraintId, ConstraintError> {
        self.model.add_constraint(constraint)
    }

    pub fn model(&self) -> &ConstraintModel {
        &self.model
    }

    pub fn into_model(self) -> ConstraintModel {
        self.model
    }
}

pub struct FixedPoint {
    pub point: Point2,
    pub value: [f64; 2],
}

impl Constraint for FixedPoint {
    fn kind(&self) -> &'static str {
        "fixed-point-2d"
    }
    fn residual_count(&self) -> usize {
        2
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        out.set_residual(0, values.get(self.point.x)? - self.value[0])?;
        out.set_residual(1, values.get(self.point.y)? - self.value[1])?;
        out.set_derivative(0, self.point.x, 1.0)?;
        out.set_derivative(1, self.point.y, 1.0)
    }
}

pub struct FixedCoordinate {
    pub point: Point2,
    pub axis: Axis2,
    pub value: f64,
}

impl Constraint for FixedCoordinate {
    fn kind(&self) -> &'static str {
        "fixed-coordinate-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let variable = match self.axis {
            Axis2::X => self.point.x,
            Axis2::Y => self.point.y,
        };
        out.set_residual(0, values.get(variable)? - self.value)?;
        out.set_derivative(0, variable, 1.0)
    }
}

pub struct Coincident {
    pub left: Point2,
    pub right: Point2,
}

impl Constraint for Coincident {
    fn kind(&self) -> &'static str {
        "coincident-2d"
    }
    fn residual_count(&self) -> usize {
        2
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        for (row, left, right) in [
            (0, self.left.x, self.right.x),
            (1, self.left.y, self.right.y),
        ] {
            out.set_residual(row, values.get(left)? - values.get(right)?)?;
            out.set_derivative(row, left, 1.0)?;
            out.set_derivative(row, right, -1.0)?;
        }
        Ok(())
    }
}

pub struct Distance {
    pub left: Point2,
    pub right: Point2,
    pub distance: f64,
}

impl Constraint for Distance {
    fn kind(&self) -> &'static str {
        "distance-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        validate_nonnegative(self.distance, "distance")?;
        let dx = values.get(self.left.x)? - values.get(self.right.x)?;
        let dy = values.get(self.left.y)? - values.get(self.right.y)?;
        out.set_residual(0, dx * dx + dy * dy - self.distance * self.distance)?;
        set_pair_derivatives(out, self.left, self.right, 2.0 * dx, 2.0 * dy)
    }
}

pub struct MinimumDistance {
    pub left: Point2,
    pub right: Point2,
    pub distance: f64,
}

impl Constraint for MinimumDistance {
    fn kind(&self) -> &'static str {
        "minimum-distance-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn relation(&self, _: usize) -> Relation {
        Relation::LessOrEqual
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        validate_nonnegative(self.distance, "minimum distance")?;
        let dx = values.get(self.left.x)? - values.get(self.right.x)?;
        let dy = values.get(self.left.y)? - values.get(self.right.y)?;
        out.set_residual(0, self.distance * self.distance - dx * dx - dy * dy)?;
        set_pair_derivatives(out, self.left, self.right, -2.0 * dx, -2.0 * dy)
    }
}

pub struct Horizontal(pub Segment2);

impl Constraint for Horizontal {
    fn kind(&self) -> &'static str {
        "horizontal-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        out.set_residual(0, values.get(self.0.end.y)? - values.get(self.0.start.y)?)?;
        out.set_derivative(0, self.0.end.y, 1.0)?;
        out.set_derivative(0, self.0.start.y, -1.0)
    }
}

pub struct Vertical(pub Segment2);

impl Constraint for Vertical {
    fn kind(&self) -> &'static str {
        "vertical-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        out.set_residual(0, values.get(self.0.end.x)? - values.get(self.0.start.x)?)?;
        out.set_derivative(0, self.0.end.x, 1.0)?;
        out.set_derivative(0, self.0.start.x, -1.0)
    }
}

pub struct Parallel {
    pub left: Segment2,
    pub right: Segment2,
}

impl Constraint for Parallel {
    fn kind(&self) -> &'static str {
        "parallel-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let [ux, uy] = segment_vector(values, self.left)?;
        let [vx, vy] = segment_vector(values, self.right)?;
        out.set_residual(0, ux * vy - uy * vx)?;
        for (variable, derivative) in [
            (self.left.start.x, -vy),
            (self.left.start.y, vx),
            (self.left.end.x, vy),
            (self.left.end.y, -vx),
            (self.right.start.x, uy),
            (self.right.start.y, -ux),
            (self.right.end.x, -uy),
            (self.right.end.y, ux),
        ] {
            out.set_derivative(0, variable, derivative)?;
        }
        Ok(())
    }
}

pub struct Perpendicular {
    pub left: Segment2,
    pub right: Segment2,
}

impl Constraint for Perpendicular {
    fn kind(&self) -> &'static str {
        "perpendicular-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let [ux, uy] = segment_vector(values, self.left)?;
        let [vx, vy] = segment_vector(values, self.right)?;
        out.set_residual(0, ux * vx + uy * vy)?;
        for (variable, derivative) in [
            (self.left.start.x, -vx),
            (self.left.start.y, -vy),
            (self.left.end.x, vx),
            (self.left.end.y, vy),
            (self.right.start.x, -ux),
            (self.right.start.y, -uy),
            (self.right.end.x, ux),
            (self.right.end.y, uy),
        ] {
            out.set_derivative(0, variable, derivative)?;
        }
        Ok(())
    }
}

pub struct PointOnLine {
    pub point: Point2,
    pub line: Segment2,
}

impl Constraint for PointOnLine {
    fn kind(&self) -> &'static str {
        "point-on-line-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let ax = values.get(self.line.start.x)?;
        let ay = values.get(self.line.start.y)?;
        let bx = values.get(self.line.end.x)?;
        let by = values.get(self.line.end.y)?;
        let px = values.get(self.point.x)?;
        let py = values.get(self.point.y)?;
        out.set_residual(0, (bx - ax) * (py - ay) - (by - ay) * (px - ax))?;
        for (variable, derivative) in [
            (self.line.start.x, by - py),
            (self.line.start.y, px - bx),
            (self.line.end.x, py - ay),
            (self.line.end.y, ax - px),
            (self.point.x, ay - by),
            (self.point.y, bx - ax),
        ] {
            out.set_derivative(0, variable, derivative)?;
        }
        Ok(())
    }
}

pub struct PointOnCircle {
    pub point: Point2,
    pub circle: Circle2,
}

impl Constraint for PointOnCircle {
    fn kind(&self) -> &'static str {
        "point-on-circle-2d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let dx = values.get(self.point.x)? - values.get(self.circle.center.x)?;
        let dy = values.get(self.point.y)? - values.get(self.circle.center.y)?;
        let radius = values.get(self.circle.radius)?;
        out.set_residual(0, dx * dx + dy * dy - radius * radius)?;
        set_pair_derivatives(out, self.point, self.circle.center, 2.0 * dx, 2.0 * dy)?;
        out.set_derivative(0, self.circle.radius, -2.0 * radius)
    }
}

fn segment_vector(
    values: VariableValues<'_>,
    segment: Segment2,
) -> Result<[f64; 2], ConstraintError> {
    Ok([
        values.get(segment.end.x)? - values.get(segment.start.x)?,
        values.get(segment.end.y)? - values.get(segment.start.y)?,
    ])
}

fn set_pair_derivatives(
    out: &mut ConstraintOutput<'_>,
    left: Point2,
    right: Point2,
    dx: f64,
    dy: f64,
) -> Result<(), ConstraintError> {
    out.set_derivative(0, left.x, dx)?;
    out.set_derivative(0, left.y, dy)?;
    out.set_derivative(0, right.x, -dx)?;
    out.set_derivative(0, right.y, -dy)
}

fn validate_nonnegative(value: f64, name: &str) -> Result<(), ConstraintError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ConstraintError::InvalidDefinition(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solverang::{
        ConstraintSolveConfig, ConstraintStatus, solve_constraints, verify_constraint_jacobian,
    };

    #[test]
    fn cad_and_vector_profile_uses_the_same_neutral_constraints() {
        let mut geometry = Geometry2d::new();
        let a = geometry.add_point(0.1, -0.1).unwrap();
        let b = geometry.add_point(3.8, 0.2).unwrap();
        let c = geometry.add_point(4.2, 2.8).unwrap();
        geometry
            .constrain(FixedPoint {
                point: a,
                value: [0.0, 0.0],
            })
            .unwrap();
        geometry
            .constrain(FixedCoordinate {
                point: b,
                axis: Axis2::Y,
                value: 0.0,
            })
            .unwrap();
        geometry
            .constrain(FixedCoordinate {
                point: c,
                axis: Axis2::X,
                value: 4.0,
            })
            .unwrap();
        geometry
            .constrain(Distance {
                left: a,
                right: b,
                distance: 4.0,
            })
            .unwrap();
        geometry
            .constrain(Distance {
                left: b,
                right: c,
                distance: 3.0,
            })
            .unwrap();
        geometry
            .constrain(Vertical(geometry.add_segment(b, c)))
            .unwrap();
        assert!(
            verify_constraint_jacobian(geometry.model(), geometry.model().initial_values(), 1.0e-6,)
                .unwrap() < 1.0e-7
        );
        let report =
            solve_constraints(geometry.model(), &ConstraintSolveConfig::default()).unwrap();
        assert_eq!(report.status, ConstraintStatus::Solved);
        assert!(report.residual_norm < 1.0e-8);
    }

    #[test]
    fn autopcb_clearance_is_a_generic_inequality() {
        let mut geometry = Geometry2d::new();
        let obstacle = geometry.add_point(0.0, 0.0).unwrap();
        let route = geometry.add_point(0.25, 0.0).unwrap();
        geometry
            .constrain(FixedPoint {
                point: obstacle,
                value: [0.0, 0.0],
            })
            .unwrap();
        geometry
            .constrain(FixedCoordinate {
                point: route,
                axis: Axis2::Y,
                value: 0.0,
            })
            .unwrap();
        geometry
            .constrain(MinimumDistance {
                left: route,
                right: obstacle,
                distance: 1.0,
            })
            .unwrap();
        let report =
            solve_constraints(geometry.model(), &ConstraintSolveConfig::default()).unwrap();
        assert_eq!(report.status, ConstraintStatus::Underconstrained);
        assert!(report.values[route.x.0].abs() >= 1.0 - 1.0e-7);
    }
}
