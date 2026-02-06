//! Sub-problem extraction from constraint system components.
//!
//! This module provides functionality to extract a sub-problem from a component
//! of the constraint system. A sub-problem is a view into the parent system that
//! implements the Problem trait, allowing it to be solved by any solver.

use crate::problem::Problem;
use std::collections::HashMap;

/// A view of a single component as a Problem.
///
/// This struct borrows the parent constraint system and provides a Problem
/// implementation that operates on only the constraints and variables within
/// a specific component.
pub struct ComponentProblem<'a> {
    /// Human-readable name for this component
    name: String,
    /// The parent constraint set (borrowed)
    constraints: &'a [Box<dyn crate::geometry::constraint::Constraint>],
    /// Indices of constraints in this component
    constraint_indices: Vec<usize>,
    /// Global param indices that are free variables in this component
    pub free_var_indices: Vec<usize>,
    /// Map from global param index to local variable index
    global_to_local: HashMap<usize, usize>,
    /// Full parameter values (copied snapshot, will be mutated during solve)
    params: Vec<f64>,
    /// Total equations in this component
    n_equations: usize,
}

impl<'a> ComponentProblem<'a> {
    /// Get the free variable indices for this component.
    pub fn free_variable_indices(&self) -> &[usize] {
        &self.free_var_indices
    }
}

impl<'a> Problem for ComponentProblem<'a> {
    fn name(&self) -> &str {
        &self.name
    }

    fn residual_count(&self) -> usize {
        self.n_equations
    }

    fn variable_count(&self) -> usize {
        self.free_var_indices.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // 1. Copy params, scatter x into free slots
        let mut p = self.params.clone();
        for (local_i, &global_i) in self.free_var_indices.iter().enumerate() {
            p[global_i] = x[local_i];
        }

        // 2. Evaluate constraints
        let mut residuals = Vec::with_capacity(self.n_equations);
        for &ci in &self.constraint_indices {
            residuals.extend(self.constraints[ci].residuals(&p));
        }
        residuals
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        // 1. Copy params, scatter x into free slots
        let mut p = self.params.clone();
        for (local_i, &global_i) in self.free_var_indices.iter().enumerate() {
            p[global_i] = x[local_i];
        }

        // 2. Evaluate constraint Jacobians and remap columns
        let mut entries = Vec::new();
        let mut row_offset = 0;
        for &ci in &self.constraint_indices {
            let c = &self.constraints[ci];
            for (local_row, global_col, val) in c.jacobian(&p) {
                // Only include entries for free variables (not fixed)
                if let Some(&local_col) = self.global_to_local.get(&global_col) {
                    entries.push((row_offset + local_row, local_col, val));
                }
                // If global_col is fixed, skip (it's not in our variable set)
            }
            row_offset += c.equation_count();
        }
        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        // Use current parameter values as initial guess (warm start)
        self.free_var_indices
            .iter()
            .map(|&gi| self.params[gi])
            .collect()
    }
}

/// Extract a component problem from a constraint system.
///
/// # Arguments
///
/// * `constraints` - All constraints in the system
/// * `constraint_indices` - Indices of constraints in this component
/// * `params` - Full parameter vector
/// * `fixed` - Fixed flags for all parameters
///
/// # Returns
///
/// A ComponentProblem that can be passed to any solver.
pub fn extract_component<'a>(
    constraints: &'a [Box<dyn crate::geometry::constraint::Constraint>],
    constraint_indices: &[usize],
    params: &[f64],
    fixed: &[bool],
) -> ComponentProblem<'a> {
    // 1. Collect all param indices referenced by the given constraints
    let mut param_set = std::collections::HashSet::new();
    for &ci in constraint_indices {
        for &pi in constraints[ci].dependencies() {
            param_set.insert(pi);
        }
    }

    // 2. Filter to only free (unfixed) params
    let free_var_indices: Vec<usize> = param_set
        .into_iter()
        .filter(|&pi| pi < fixed.len() && !fixed[pi])
        .collect();

    // 3. Build global_to_local map
    let global_to_local: HashMap<usize, usize> = free_var_indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    // 4. Count total equations
    let n_equations: usize = constraint_indices
        .iter()
        .map(|&ci| constraints[ci].equation_count())
        .sum();

    // 5. Create a snapshot of parameters
    let params_snapshot = params.to_vec();

    ComponentProblem {
        name: format!("component_{}", constraint_indices.first().unwrap_or(&0)),
        constraints,
        constraint_indices: constraint_indices.to_vec(),
        free_var_indices,
        global_to_local,
        params: params_snapshot,
        n_equations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::constraint::{Constraint, Nonlinearity};
    use crate::geometry::params::ConstraintId;

    /// Mock constraint for testing.
    struct MockConstraint {
        id: ConstraintId,
        deps: Vec<usize>,
        n_equations: usize,
    }

    impl Constraint for MockConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }

        fn name(&self) -> &'static str {
            "MockConstraint"
        }

        fn equation_count(&self) -> usize {
            self.n_equations
        }

        fn dependencies(&self) -> &[usize] {
            &self.deps
        }

        fn residuals(&self, params: &[f64]) -> Vec<f64> {
            // Simple residual: sum of parameter values
            let sum: f64 = self.deps.iter().map(|&i| params[i]).sum();
            vec![sum; self.n_equations]
        }

        fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
            // Jacobian: all 1.0 for all dependencies
            let mut entries = Vec::new();
            for row in 0..self.n_equations {
                for &col in &self.deps {
                    entries.push((row, col, 1.0));
                }
            }
            entries
        }

        fn nonlinearity_hint(&self) -> Nonlinearity {
            Nonlinearity::Linear
        }
    }

    #[test]
    fn test_extract_component() {
        // Create two mock constraints
        let constraints: Vec<Box<dyn Constraint>> = vec![
            Box::new(MockConstraint {
                id: ConstraintId(0),
                deps: vec![0, 1],
                n_equations: 1,
            }),
            Box::new(MockConstraint {
                id: ConstraintId(1),
                deps: vec![1, 2],
                n_equations: 2,
            }),
        ];

        let params = vec![1.0, 2.0, 3.0, 4.0];
        let fixed = vec![true, false, false, true]; // Only params 1 and 2 are free

        let component = extract_component(&constraints, &[0, 1], &params, &fixed);

        // Should have 2 free variables (params 1 and 2)
        assert_eq!(component.variable_count(), 2);

        // Should have 1 + 2 = 3 equations
        assert_eq!(component.residual_count(), 3);

        // Free variables should be params 1 and 2
        let free_vars = component.free_variable_indices();
        assert!(free_vars.contains(&1));
        assert!(free_vars.contains(&2));
    }

    #[test]
    fn test_component_problem_residuals() {
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            n_equations: 1,
        })];

        let params = vec![1.0, 2.0];
        let fixed = vec![false, false];

        let component = extract_component(&constraints, &[0], &params, &fixed);

        // Evaluate at x = [1.0, 2.0]
        let residuals = component.residuals(&[1.0, 2.0]);

        // Should be sum of params = 1.0 + 2.0 = 3.0
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_component_problem_jacobian() {
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            n_equations: 1,
        })];

        let params = vec![1.0, 2.0];
        let fixed = vec![false, false];

        let component = extract_component(&constraints, &[0], &params, &fixed);

        // Get Jacobian at x = [1.0, 2.0]
        let jac = component.jacobian(&[1.0, 2.0]);

        // Should have 2 entries (one for each variable)
        assert_eq!(jac.len(), 2);

        // Both should be 1.0 (partial derivatives)
        for (row, col, val) in jac {
            assert_eq!(row, 0); // Only one equation
            assert!(col < 2); // Two variables
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_component_with_fixed_params() {
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1, 2],
            n_equations: 1,
        })];

        let params = vec![1.0, 2.0, 3.0];
        let fixed = vec![true, false, false]; // First param is fixed

        let component = extract_component(&constraints, &[0], &params, &fixed);

        // Should have 2 free variables (params 1 and 2)
        assert_eq!(component.variable_count(), 2);

        // Jacobian should only have columns for free variables
        let jac = component.jacobian(&[2.0, 3.0]);

        // Should have entries only for columns 0 and 1 (local indices for global params 1 and 2)
        for (_, col, _) in jac {
            assert!(col < 2); // Only 2 free variables
        }
    }

    #[test]
    fn test_initial_point() {
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(MockConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            n_equations: 1,
        })];

        let params = vec![5.0, 7.0];
        let fixed = vec![false, false];

        let component = extract_component(&constraints, &[0], &params, &fixed);

        let x0 = component.initial_point(1.0);

        // Should return current parameter values
        assert_eq!(x0.len(), 2);
        // Order depends on the HashSet iteration, so check both are present
        assert!(x0.contains(&5.0) || x0.contains(&7.0));
    }
}
