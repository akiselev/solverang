use serde::{Deserialize, Serialize};

use crate::ConstraintError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VariableId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstraintId(pub usize);

/// Interpretation of a signed residual supplied by a constraint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Relation {
    #[default]
    Equal,
    /// Satisfied when the raw residual is less than or equal to zero.
    LessOrEqual,
    /// Satisfied when the raw residual is greater than or equal to zero.
    GreaterOrEqual,
}

#[derive(Clone, Copy)]
pub struct VariableValues<'a> {
    values: &'a [f64],
}

impl<'a> VariableValues<'a> {
    pub fn get(self, id: VariableId) -> Result<f64, ConstraintError> {
        self.values
            .get(id.0)
            .copied()
            .ok_or(ConstraintError::UnknownVariable(id.0))
    }
}

pub struct ConstraintOutput<'a> {
    residual: &'a mut [f64],
    jacobian: &'a mut [f64],
    columns: usize,
}

impl ConstraintOutput<'_> {
    pub fn set_residual(&mut self, row: usize, value: f64) -> Result<(), ConstraintError> {
        let residual = self.residual.get_mut(row).ok_or_else(|| {
            ConstraintError::InvalidDefinition("residual row is out of range".into())
        })?;
        *residual = value;
        Ok(())
    }

    pub fn set_derivative(
        &mut self,
        row: usize,
        variable: VariableId,
        value: f64,
    ) -> Result<(), ConstraintError> {
        if variable.0 >= self.columns {
            return Err(ConstraintError::UnknownVariable(variable.0));
        }
        let index = row
            .checked_mul(self.columns)
            .and_then(|offset| offset.checked_add(variable.0))
            .ok_or_else(|| ConstraintError::InvalidDefinition("Jacobian index overflow".into()))?;
        let derivative = self.jacobian.get_mut(index).ok_or_else(|| {
            ConstraintError::InvalidDefinition("Jacobian row is out of range".into())
        })?;
        *derivative += value;
        Ok(())
    }
}

/// A consumer-neutral residual block. Domain crates provide concrete vocabulary.
pub trait Constraint: Send + Sync {
    fn kind(&self) -> &'static str;
    fn residual_count(&self) -> usize;
    fn relation(&self, _row: usize) -> Relation {
        Relation::Equal
    }
    fn evaluate(
        &self,
        values: VariableValues<'_>,
        output: &mut ConstraintOutput<'_>,
    ) -> Result<(), ConstraintError>;
}

struct ConstraintEntry {
    id: ConstraintId,
    constraint: Box<dyn Constraint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelEvaluation {
    pub residuals: Vec<f64>,
    pub jacobian: Vec<f64>,
    pub row_constraints: Vec<ConstraintId>,
    pub row_kinds: Vec<&'static str>,
}

#[derive(Default)]
pub struct ConstraintModel {
    initial_values: Vec<f64>,
    constraints: Vec<ConstraintEntry>,
}

impl ConstraintModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_variable(&mut self, initial_value: f64) -> Result<VariableId, ConstraintError> {
        if !initial_value.is_finite() {
            return Err(ConstraintError::NonFinite);
        }
        let id = VariableId(self.initial_values.len());
        self.initial_values.push(initial_value);
        Ok(id)
    }

    pub fn add_constraint(
        &mut self,
        constraint: impl Constraint + 'static,
    ) -> Result<ConstraintId, ConstraintError> {
        if constraint.residual_count() == 0 {
            return Err(ConstraintError::InvalidDefinition(
                "a constraint must contribute at least one residual".into(),
            ));
        }
        let id = ConstraintId(self.constraints.len());
        self.constraints.push(ConstraintEntry {
            id,
            constraint: Box::new(constraint),
        });
        Ok(id)
    }

    pub fn variable_count(&self) -> usize {
        self.initial_values.len()
    }

    pub fn residual_count(&self) -> usize {
        self.constraints
            .iter()
            .map(|entry| entry.constraint.residual_count())
            .sum()
    }

    pub fn initial_values(&self) -> &[f64] {
        &self.initial_values
    }

    pub fn evaluate(&self, values: &[f64]) -> Result<ModelEvaluation, ConstraintError> {
        if values.len() != self.variable_count() {
            return Err(ConstraintError::DimensionMismatch {
                operation: "constraint-model values",
                expected: self.variable_count(),
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ConstraintError::NonFinite);
        }
        let rows = self.residual_count();
        let columns = self.variable_count();
        let mut residuals = vec![0.0; rows];
        let mut jacobian = vec![
            0.0;
            rows.checked_mul(columns).ok_or_else(|| {
                ConstraintError::InvalidDefinition("model Jacobian dimensions overflow".into())
            })?
        ];
        let mut row_constraints = Vec::with_capacity(rows);
        let mut row_kinds = Vec::with_capacity(rows);
        let mut row_offset = 0;
        for entry in &self.constraints {
            let count = entry.constraint.residual_count();
            let residual_slice = &mut residuals[row_offset..row_offset + count];
            let jacobian_slice =
                &mut jacobian[row_offset * columns..(row_offset + count) * columns];
            let mut output = ConstraintOutput {
                residual: residual_slice,
                jacobian: jacobian_slice,
                columns,
            };
            entry
                .constraint
                .evaluate(VariableValues { values }, &mut output)?;
            for local_row in 0..count {
                apply_relation(
                    entry.constraint.relation(local_row),
                    &mut residual_slice[local_row],
                    &mut jacobian_slice[local_row * columns..(local_row + 1) * columns],
                );
                row_constraints.push(entry.id);
                row_kinds.push(entry.constraint.kind());
            }
            row_offset += count;
        }
        if residuals
            .iter()
            .chain(&jacobian)
            .any(|value| !value.is_finite())
        {
            return Err(ConstraintError::NonFinite);
        }
        Ok(ModelEvaluation {
            residuals,
            jacobian,
            row_constraints,
            row_kinds,
        })
    }
}

fn apply_relation(relation: Relation, residual: &mut f64, derivatives: &mut [f64]) {
    let inactive = match relation {
        Relation::Equal => false,
        Relation::LessOrEqual => *residual <= 0.0,
        Relation::GreaterOrEqual => *residual >= 0.0,
    };
    if inactive {
        *residual = 0.0;
        derivatives.fill(0.0);
    }
}
