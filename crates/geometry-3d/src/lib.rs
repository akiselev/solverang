//! Reusable three-dimensional geometric constraint vocabulary.
//!
//! The crate does not own CAD topology, assembly identity, units, or commit
//! policy. Consumers map their entities to these variable-only primitives.

#![forbid(unsafe_code)]

use solverang::{
    Constraint, ConstraintError, ConstraintId, ConstraintModel, ConstraintOutput, Relation,
    VariableId, VariableValues,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point3 {
    pub x: VariableId,
    pub y: VariableId,
    pub z: VariableId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment3 {
    pub start: Point3,
    pub end: Point3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane3 {
    pub normal: [f64; 3],
    pub offset: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis3 {
    X,
    Y,
    Z,
}

#[derive(Default)]
pub struct Geometry3d {
    model: ConstraintModel,
}

impl Geometry3d {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_point(&mut self, x: f64, y: f64, z: f64) -> Result<Point3, ConstraintError> {
        Ok(Point3 {
            x: self.model.add_variable(x)?,
            y: self.model.add_variable(y)?,
            z: self.model.add_variable(z)?,
        })
    }

    pub fn add_segment(&self, start: Point3, end: Point3) -> Segment3 {
        Segment3 { start, end }
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
    pub point: Point3,
    pub value: [f64; 3],
}

impl Constraint for FixedPoint {
    fn kind(&self) -> &'static str {
        "fixed-point-3d"
    }
    fn residual_count(&self) -> usize {
        3
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        for (row, variable, target) in [
            (0, self.point.x, self.value[0]),
            (1, self.point.y, self.value[1]),
            (2, self.point.z, self.value[2]),
        ] {
            out.set_residual(row, values.get(variable)? - target)?;
            out.set_derivative(row, variable, 1.0)?;
        }
        Ok(())
    }
}

pub struct FixedCoordinate {
    pub point: Point3,
    pub axis: Axis3,
    pub value: f64,
}

impl Constraint for FixedCoordinate {
    fn kind(&self) -> &'static str {
        "fixed-coordinate-3d"
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
            Axis3::X => self.point.x,
            Axis3::Y => self.point.y,
            Axis3::Z => self.point.z,
        };
        out.set_residual(0, values.get(variable)? - self.value)?;
        out.set_derivative(0, variable, 1.0)
    }
}

pub struct Coincident {
    pub left: Point3,
    pub right: Point3,
}

impl Constraint for Coincident {
    fn kind(&self) -> &'static str {
        "coincident-3d"
    }
    fn residual_count(&self) -> usize {
        3
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        for (row, left, right) in [
            (0, self.left.x, self.right.x),
            (1, self.left.y, self.right.y),
            (2, self.left.z, self.right.z),
        ] {
            out.set_residual(row, values.get(left)? - values.get(right)?)?;
            out.set_derivative(row, left, 1.0)?;
            out.set_derivative(row, right, -1.0)?;
        }
        Ok(())
    }
}

pub struct Distance {
    pub left: Point3,
    pub right: Point3,
    pub distance: f64,
}

impl Constraint for Distance {
    fn kind(&self) -> &'static str {
        "distance-3d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        validate_distance(self.distance)?;
        let delta = point_delta(values, self.left, self.right)?;
        out.set_residual(0, dot(delta, delta) - self.distance * self.distance)?;
        set_pair_derivatives(out, self.left, self.right, delta.map(|value| 2.0 * value))
    }
}

pub struct MinimumDistance {
    pub left: Point3,
    pub right: Point3,
    pub distance: f64,
}

impl Constraint for MinimumDistance {
    fn kind(&self) -> &'static str {
        "minimum-distance-3d"
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
        validate_distance(self.distance)?;
        let delta = point_delta(values, self.left, self.right)?;
        out.set_residual(0, self.distance * self.distance - dot(delta, delta))?;
        set_pair_derivatives(out, self.left, self.right, delta.map(|value| -2.0 * value))
    }
}

pub struct PointOnPlane {
    pub point: Point3,
    pub plane: Plane3,
}

impl Constraint for PointOnPlane {
    fn kind(&self) -> &'static str {
        "point-on-plane-3d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        if self.plane.normal.iter().any(|value| !value.is_finite())
            || !self.plane.offset.is_finite()
            || dot(self.plane.normal, self.plane.normal) == 0.0
        {
            return Err(ConstraintError::InvalidDefinition(
                "plane must have a finite nonzero normal and finite offset".into(),
            ));
        }
        let point = [
            values.get(self.point.x)?,
            values.get(self.point.y)?,
            values.get(self.point.z)?,
        ];
        out.set_residual(0, dot(self.plane.normal, point) - self.plane.offset)?;
        for (variable, derivative) in point_variables(self.point)
            .into_iter()
            .zip(self.plane.normal)
        {
            out.set_derivative(0, variable, derivative)?;
        }
        Ok(())
    }
}

pub struct Perpendicular {
    pub left: Segment3,
    pub right: Segment3,
}

impl Constraint for Perpendicular {
    fn kind(&self) -> &'static str {
        "perpendicular-3d"
    }
    fn residual_count(&self) -> usize {
        1
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let u = point_delta(values, self.left.end, self.left.start)?;
        let v = point_delta(values, self.right.end, self.right.start)?;
        out.set_residual(0, dot(u, v))?;
        add_vector_derivative(out, 0, self.left.start, v.map(|value| -value))?;
        add_vector_derivative(out, 0, self.left.end, v)?;
        add_vector_derivative(out, 0, self.right.start, u.map(|value| -value))?;
        add_vector_derivative(out, 0, self.right.end, u)
    }
}

pub struct Parallel {
    pub left: Segment3,
    pub right: Segment3,
}

impl Constraint for Parallel {
    fn kind(&self) -> &'static str {
        "parallel-3d"
    }
    fn residual_count(&self) -> usize {
        3
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        out: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError> {
        let u = point_delta(values, self.left.end, self.left.start)?;
        let v = point_delta(values, self.right.end, self.right.start)?;
        let residual = cross(u, v);
        for (row, value) in residual.into_iter().enumerate() {
            out.set_residual(row, value)?;
        }
        let basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (axis, unit) in basis.into_iter().enumerate() {
            add_column(
                out,
                self.left.start,
                axis,
                cross(unit, v).map(|value| -value),
            )?;
            add_column(out, self.left.end, axis, cross(unit, v))?;
            add_column(
                out,
                self.right.start,
                axis,
                cross(u, unit).map(|value| -value),
            )?;
            add_column(out, self.right.end, axis, cross(u, unit))?;
        }
        Ok(())
    }
}

fn point_variables(point: Point3) -> [VariableId; 3] {
    [point.x, point.y, point.z]
}

fn point_delta(
    values: VariableValues<'_>,
    left: Point3,
    right: Point3,
) -> Result<[f64; 3], ConstraintError> {
    Ok([
        values.get(left.x)? - values.get(right.x)?,
        values.get(left.y)? - values.get(right.y)?,
        values.get(left.z)? - values.get(right.z)?,
    ])
}

fn set_pair_derivatives(
    out: &mut ConstraintOutput<'_>,
    left: Point3,
    right: Point3,
    derivative: [f64; 3],
) -> Result<(), ConstraintError> {
    add_vector_derivative(out, 0, left, derivative)?;
    add_vector_derivative(out, 0, right, derivative.map(|value| -value))
}

fn add_vector_derivative(
    out: &mut ConstraintOutput<'_>,
    row: usize,
    point: Point3,
    derivative: [f64; 3],
) -> Result<(), ConstraintError> {
    for (variable, value) in point_variables(point).into_iter().zip(derivative) {
        out.set_derivative(row, variable, value)?;
    }
    Ok(())
}

fn add_column(
    out: &mut ConstraintOutput<'_>,
    point: Point3,
    axis: usize,
    derivative: [f64; 3],
) -> Result<(), ConstraintError> {
    let variable = point_variables(point)[axis];
    for (row, value) in derivative.into_iter().enumerate() {
        out.set_derivative(row, variable, value)?;
    }
    Ok(())
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn validate_distance(distance: f64) -> Result<(), ConstraintError> {
    if distance.is_finite() && distance >= 0.0 {
        Ok(())
    } else {
        Err(ConstraintError::InvalidDefinition(
            "distance must be finite and non-negative".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solverang::{
        ConstraintSolveConfig, ConstraintStatus, solve_constraints, verify_constraint_jacobian,
    };

    #[test]
    fn cadabra_candidate_uses_the_neutral_point_plane_contract() {
        let mut geometry = Geometry3d::new();
        let point = geometry.add_point(1.2, 1.8, 4.0).unwrap();
        geometry
            .constrain(FixedCoordinate {
                point,
                axis: Axis3::X,
                value: 1.0,
            })
            .unwrap();
        geometry
            .constrain(FixedCoordinate {
                point,
                axis: Axis3::Y,
                value: 2.0,
            })
            .unwrap();
        geometry
            .constrain(PointOnPlane {
                point,
                plane: Plane3 {
                    normal: [0.0, 0.0, 1.0],
                    offset: 3.0,
                },
            })
            .unwrap();
        assert!(
            verify_constraint_jacobian(geometry.model(), geometry.model().initial_values(), 1.0e-6,)
                .unwrap() < 1.0e-8
        );
        let report =
            solve_constraints(geometry.model(), &ConstraintSolveConfig::default()).unwrap();
        assert_eq!(report.status, ConstraintStatus::Solved);
        assert!((report.values[point.z.0] - 3.0).abs() < 1.0e-8);
    }
}
