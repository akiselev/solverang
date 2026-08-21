use methodus::{
    EvaluationContext, LeastSquaresConfig, LeastSquaresOperator, NumericError, solve_least_squares,
    verify_least_squares_jacobian,
};
use serde::{Deserialize, Serialize};

use crate::{ConstraintError, ConstraintId, ConstraintModel, ModelEvaluation};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintSolveConfig {
    pub numerical: LeastSquaresConfig,
    pub rank_tolerance: f64,
    pub conflict_tolerance: f64,
}

impl Default for ConstraintSolveConfig {
    fn default() -> Self {
        Self {
            numerical: LeastSquaresConfig::default(),
            rank_tolerance: 1.0e-9,
            conflict_tolerance: 1.0e-8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintStatus {
    Solved,
    Underconstrained,
    NotConverged,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintSolveReport {
    pub values: Vec<f64>,
    pub status: ConstraintStatus,
    pub residual_norm: f64,
    pub degrees_of_freedom: usize,
    pub conflicting_constraints: Vec<ConstraintId>,
    pub numerical: methodus::LeastSquaresReport,
}

pub fn solve_constraints(
    model: &ConstraintModel,
    config: &ConstraintSolveConfig,
) -> Result<ConstraintSolveReport, ConstraintError> {
    validate_config(config)?;
    let context = EvaluationContext::reproducible();
    let numerical =
        solve_least_squares(model, &context, model.initial_values(), &config.numerical)?;
    let evaluation = model.evaluate(&numerical.variables)?;
    let residual_norm = l2(&evaluation.residuals);
    let rank = matrix_rank(
        &evaluation.jacobian,
        evaluation.residuals.len(),
        model.variable_count(),
        config.rank_tolerance,
    );
    let degrees_of_freedom = model.variable_count().saturating_sub(rank);
    let conflicting_constraints = conflicts(&evaluation, config.conflict_tolerance);
    let status = if !numerical.converged || !conflicting_constraints.is_empty() {
        ConstraintStatus::NotConverged
    } else if degrees_of_freedom > 0 {
        ConstraintStatus::Underconstrained
    } else {
        ConstraintStatus::Solved
    };
    Ok(ConstraintSolveReport {
        values: numerical.variables.clone(),
        status,
        residual_norm,
        degrees_of_freedom,
        conflicting_constraints,
        numerical,
    })
}

/// Maximum discrepancy between supplied derivatives and centered differences
/// after equality/inequality activation.
pub fn verify_constraint_jacobian(
    model: &ConstraintModel,
    values: &[f64],
    epsilon: f64,
) -> Result<f64, ConstraintError> {
    verify_least_squares_jacobian(model, &EvaluationContext::reproducible(), values, epsilon)
        .map_err(|error| {
            ConstraintError::InvalidDefinition(format!("Jacobian verification failed: {error}"))
        })
}

impl LeastSquaresOperator for ConstraintModel {
    fn variable_count(&self) -> usize {
        self.variable_count()
    }

    fn residual_count(&self) -> usize {
        self.residual_count()
    }

    fn residual(
        &self,
        _: &EvaluationContext,
        variables: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let evaluation = self.evaluate(variables).map_err(as_numeric)?;
        require_length(
            "constraint residual output",
            output.len(),
            evaluation.residuals.len(),
        )?;
        output.copy_from_slice(&evaluation.residuals);
        Ok(())
    }

    fn jacobian(
        &self,
        _: &EvaluationContext,
        variables: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let evaluation = self.evaluate(variables).map_err(as_numeric)?;
        require_length(
            "constraint Jacobian output",
            output.len(),
            evaluation.jacobian.len(),
        )?;
        output.copy_from_slice(&evaluation.jacobian);
        Ok(())
    }
}

fn validate_config(config: &ConstraintSolveConfig) -> Result<(), ConstraintError> {
    if !config.rank_tolerance.is_finite()
        || config.rank_tolerance <= 0.0
        || !config.conflict_tolerance.is_finite()
        || config.conflict_tolerance < 0.0
    {
        return Err(ConstraintError::InvalidDefinition(
            "diagnostic tolerances must be finite and rank tolerance must be positive".into(),
        ));
    }
    Ok(())
}

fn as_numeric(error: ConstraintError) -> NumericError {
    NumericError::Operator {
        message: error.to_string(),
    }
}

fn require_length(operation: &str, actual: usize, expected: usize) -> Result<(), NumericError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NumericError::DimensionMismatch {
            operation: operation.into(),
            expected,
            actual,
        })
    }
}

fn conflicts(evaluation: &ModelEvaluation, tolerance: f64) -> Vec<ConstraintId> {
    let mut result = Vec::new();
    for (row, residual) in evaluation.residuals.iter().enumerate() {
        let id = evaluation.row_constraints[row];
        if residual.abs() > tolerance && result.last() != Some(&id) {
            result.push(id);
        }
    }
    result
}

fn matrix_rank(matrix: &[f64], rows: usize, columns: usize, tolerance: f64) -> usize {
    let mut matrix = matrix.to_vec();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).max_by(|left, right| {
            matrix[*left * columns + column]
                .abs()
                .total_cmp(&matrix[*right * columns + column].abs())
        }) else {
            break;
        };
        if matrix[pivot * columns + column].abs() <= tolerance {
            continue;
        }
        for index in 0..columns {
            matrix.swap(rank * columns + index, pivot * columns + index);
        }
        let diagonal = matrix[rank * columns + column];
        for index in column..columns {
            matrix[rank * columns + index] /= diagonal;
        }
        for row in 0..rows {
            if row == rank {
                continue;
            }
            let factor = matrix[row * columns + column];
            for index in column..columns {
                matrix[row * columns + index] -= factor * matrix[rank * columns + index];
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
