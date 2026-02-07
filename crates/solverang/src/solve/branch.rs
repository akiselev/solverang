//! Branch selection for multi-solution constraint systems.
//!
//! Many geometric constraint systems admit multiple valid solutions. For
//! example, a point constrained by two distances (circle-circle intersection)
//! typically has two solutions. This module provides strategies for selecting
//! the "best" solution from several solver runs started at different initial
//! points.
//!
//! # Strategies
//!
//! - [`BranchStrategy::ClosestToPrevious`] — pick the converged solution whose
//!   L2 distance to a reference configuration is smallest. This is the natural
//!   choice for interactive editing where the user expects continuity.
//!
//! - [`BranchStrategy::SmallestResidual`] — pick the converged solution with
//!   the smallest residual norm. Useful for batch solving where only accuracy
//!   matters.

use crate::solver::SolveResult;

/// Strategy for selecting among multiple solution branches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BranchStrategy {
    /// Pick the solution closest to the previous configuration (L2 norm).
    #[default]
    ClosestToPrevious,
    /// Pick the solution with smallest residual norm.
    SmallestResidual,
}

/// Select the best solution from multiple solver runs with different initial
/// points.
///
/// Many geometric constraint systems have multiple valid solutions (e.g., a
/// circle-line intersection has two points). This function helps select the
/// one that is closest to the user's intent.
///
/// # Arguments
///
/// * `results` - Results from multiple solver runs (possibly different
///   starting points).
/// * `previous_values` - The reference configuration (typically the state
///   before the most recent edit). Used by `ClosestToPrevious`.
/// * `strategy` - Which selection criterion to apply.
///
/// # Returns
///
/// The index (into `results`) of the best converged solution, or `None` if
/// no result converged.
pub fn select_branch(
    results: &[SolveResult],
    previous_values: &[f64],
    strategy: BranchStrategy,
) -> Option<usize> {
    match strategy {
        BranchStrategy::ClosestToPrevious => select_closest(results, previous_values),
        BranchStrategy::SmallestResidual => select_smallest_residual(results),
    }
}

/// Find the converged result closest to `previous` in L2 norm.
fn select_closest(results: &[SolveResult], previous: &[f64]) -> Option<usize> {
    let mut best_index: Option<usize> = None;
    let mut best_dist = f64::INFINITY;

    for (i, result) in results.iter().enumerate() {
        if let SolveResult::Converged { solution, .. } = result {
            debug_assert_eq!(
                solution.len(),
                previous.len(),
                "select_closest: solution and previous must have the same length"
            );
            if solution.len() != previous.len() {
                continue;
            }
            let dist_sq: f64 = solution
                .iter()
                .zip(previous.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist_sq < best_dist {
                best_dist = dist_sq;
                best_index = Some(i);
            }
        }
    }

    best_index
}

/// Find the converged result with the smallest residual norm.
fn select_smallest_residual(results: &[SolveResult]) -> Option<usize> {
    let mut best_index: Option<usize> = None;
    let mut best_residual = f64::INFINITY;

    for (i, result) in results.iter().enumerate() {
        if let SolveResult::Converged { residual_norm, .. } = result {
            if *residual_norm < best_residual {
                best_residual = *residual_norm;
                best_index = Some(i);
            }
        }
    }

    best_index
}

/// Generate multiple initial points for branch exploration.
///
/// Perturbs the given initial point to explore different solution branches.
/// The perturbations are deterministic (based on index) so that results are
/// reproducible.
///
/// # Arguments
///
/// * `initial` - The base initial point.
/// * `perturbation_scale` - Magnitude of the perturbations.
/// * `num_branches` - How many perturbed starting points to generate
///   (in addition to the unperturbed original, which is always included
///   as the first element).
///
/// # Returns
///
/// A vector of initial points. The first element is always the unperturbed
/// `initial`. Subsequent elements are deterministic perturbations.
pub fn generate_branch_starts(
    initial: &[f64],
    perturbation_scale: f64,
    num_branches: usize,
) -> Vec<Vec<f64>> {
    let n = initial.len();
    let mut starts = Vec::with_capacity(num_branches + 1);

    // Always include the unperturbed initial point.
    starts.push(initial.to_vec());

    for branch in 0..num_branches {
        let mut perturbed = initial.to_vec();
        for j in 0..n {
            // Deterministic perturbation using a simple hash-like scheme.
            // Alternate sign based on (branch + j) parity, scale by a
            // varying factor to explore different directions.
            let sign = if (branch + j) % 2 == 0 { 1.0 } else { -1.0 };
            let factor = 1.0 + ((branch as f64 + 1.0) * (j as f64 + 1.0)).sin();
            perturbed[j] += sign * perturbation_scale * factor;
        }
        starts.push(perturbed);
    }

    starts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolveResult;

    #[test]
    fn test_select_closest_to_previous() {
        let results = vec![
            SolveResult::Converged {
                solution: vec![10.0, 0.0],
                iterations: 5,
                residual_norm: 1e-12,
            },
            SolveResult::Converged {
                solution: vec![1.0, 1.0],
                iterations: 5,
                residual_norm: 1e-12,
            },
            SolveResult::Converged {
                solution: vec![5.0, 5.0],
                iterations: 5,
                residual_norm: 1e-12,
            },
        ];

        let previous = vec![1.5, 1.5];
        let best = select_branch(&results, &previous, BranchStrategy::ClosestToPrevious);
        assert_eq!(best, Some(1)); // [1.0, 1.0] is closest to [1.5, 1.5]
    }

    #[test]
    fn test_select_smallest_residual() {
        let results = vec![
            SolveResult::Converged {
                solution: vec![1.0],
                iterations: 5,
                residual_norm: 1e-6,
            },
            SolveResult::Converged {
                solution: vec![2.0],
                iterations: 5,
                residual_norm: 1e-12,
            },
            SolveResult::Converged {
                solution: vec![3.0],
                iterations: 5,
                residual_norm: 1e-9,
            },
        ];

        let best = select_branch(&results, &[], BranchStrategy::SmallestResidual);
        assert_eq!(best, Some(1)); // residual 1e-12 is smallest
    }

    #[test]
    fn test_no_converged_results() {
        let results = vec![
            SolveResult::NotConverged {
                solution: vec![1.0],
                iterations: 100,
                residual_norm: 0.5,
            },
            SolveResult::Failed {
                error: crate::solver::SolveError::SingularJacobian,
            },
        ];

        let best = select_branch(&results, &[0.0], BranchStrategy::ClosestToPrevious);
        assert_eq!(best, None);

        let best = select_branch(&results, &[0.0], BranchStrategy::SmallestResidual);
        assert_eq!(best, None);
    }

    #[test]
    fn test_empty_results() {
        let results: Vec<SolveResult> = vec![];
        assert_eq!(
            select_branch(&results, &[0.0], BranchStrategy::ClosestToPrevious),
            None
        );
    }

    #[test]
    fn test_single_converged_result() {
        let results = vec![SolveResult::Converged {
            solution: vec![42.0],
            iterations: 3,
            residual_norm: 1e-15,
        }];

        let best = select_branch(&results, &[0.0], BranchStrategy::ClosestToPrevious);
        assert_eq!(best, Some(0));
    }

    #[test]
    fn test_generate_branch_starts_includes_original() {
        let initial = vec![1.0, 2.0, 3.0];
        let starts = generate_branch_starts(&initial, 0.5, 3);

        assert_eq!(starts.len(), 4); // 1 original + 3 perturbed
        assert_eq!(starts[0], initial);
    }

    #[test]
    fn test_generate_branch_starts_perturbed() {
        let initial = vec![0.0, 0.0];
        let starts = generate_branch_starts(&initial, 1.0, 2);

        // Perturbed points should differ from the original.
        for start in &starts[1..] {
            let differs = start.iter().zip(initial.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
            assert!(differs, "perturbed point should differ from original");
        }
    }

    #[test]
    fn test_generate_branch_starts_deterministic() {
        let initial = vec![1.0, 2.0];
        let starts1 = generate_branch_starts(&initial, 0.5, 4);
        let starts2 = generate_branch_starts(&initial, 0.5, 4);

        assert_eq!(starts1, starts2);
    }

    #[test]
    fn test_generate_branch_starts_zero_branches() {
        let initial = vec![5.0];
        let starts = generate_branch_starts(&initial, 1.0, 0);

        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0], initial);
    }

    #[test]
    fn test_mixed_converged_and_failed() {
        let results = vec![
            SolveResult::Failed {
                error: crate::solver::SolveError::SingularJacobian,
            },
            SolveResult::Converged {
                solution: vec![5.0, 5.0],
                iterations: 10,
                residual_norm: 1e-10,
            },
            SolveResult::NotConverged {
                solution: vec![3.0, 3.0],
                iterations: 100,
                residual_norm: 0.1,
            },
        ];

        let previous = vec![0.0, 0.0];

        // Only index 1 converged.
        let best = select_branch(&results, &previous, BranchStrategy::ClosestToPrevious);
        assert_eq!(best, Some(1));

        let best = select_branch(&results, &previous, BranchStrategy::SmallestResidual);
        assert_eq!(best, Some(1));
    }
}
