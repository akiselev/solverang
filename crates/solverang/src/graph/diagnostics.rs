use super::union_find::ComponentId;

/// DOF (degrees of freedom) analysis for a component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DOFAnalysis {
    /// The component being analyzed.
    pub component: ComponentId,
    /// Total number of parameters in the component.
    pub total_variables: usize,
    /// Number of fixed (non-variable) parameters.
    pub fixed_variables: usize,
    /// Number of free (variable) parameters.
    pub free_variables: usize,
    /// Total number of scalar equations from all constraints.
    pub total_equations: usize,
    /// Degrees of freedom: free_variables - total_equations.
    /// - DOF = 0: well-constrained (unique solution)
    /// - DOF > 0: under-constrained (infinite solutions)
    /// - DOF < 0: over-constrained (no solution or redundant)
    pub dof: i32,
    /// Classification of constraint status.
    pub status: ConstraintStatus,
}

/// Classification of a component's constraint status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintStatus {
    /// DOF = 0: System is well-constrained with a unique solution.
    WellConstrained,
    /// DOF > 0: System has degrees of freedom (under-constrained).
    UnderConstrained,
    /// DOF < 0: System has more constraints than DOF (over-constrained).
    OverConstrained,
}

impl DOFAnalysis {
    /// Create a new DOF analysis.
    pub fn new(
        component: ComponentId,
        total_variables: usize,
        fixed_variables: usize,
        total_equations: usize,
    ) -> Self {
        let free_variables = total_variables.saturating_sub(fixed_variables);
        let dof = free_variables as i32 - total_equations as i32;

        let status = match dof.cmp(&0) {
            std::cmp::Ordering::Equal => ConstraintStatus::WellConstrained,
            std::cmp::Ordering::Greater => ConstraintStatus::UnderConstrained,
            std::cmp::Ordering::Less => ConstraintStatus::OverConstrained,
        };

        Self {
            component,
            total_variables,
            fixed_variables,
            free_variables,
            total_equations,
            dof,
            status,
        }
    }

    /// Create a DOF analysis with explicit free variables count.
    /// This is useful when the free variables are pre-computed.
    pub fn from_free(
        component: ComponentId,
        free_variables: usize,
        total_equations: usize,
    ) -> Self {
        let dof = free_variables as i32 - total_equations as i32;

        let status = match dof.cmp(&0) {
            std::cmp::Ordering::Equal => ConstraintStatus::WellConstrained,
            std::cmp::Ordering::Greater => ConstraintStatus::UnderConstrained,
            std::cmp::Ordering::Less => ConstraintStatus::OverConstrained,
        };

        Self {
            component,
            total_variables: free_variables, // Approximation when fixed not tracked separately
            fixed_variables: 0,
            free_variables,
            total_equations,
            dof,
            status,
        }
    }

    /// Check if the component is well-constrained.
    pub fn is_well_constrained(&self) -> bool {
        self.status == ConstraintStatus::WellConstrained
    }

    /// Check if the component is under-constrained.
    pub fn is_under_constrained(&self) -> bool {
        self.status == ConstraintStatus::UnderConstrained
    }

    /// Check if the component is over-constrained.
    pub fn is_over_constrained(&self) -> bool {
        self.status == ConstraintStatus::OverConstrained
    }

    /// Get a human-readable status message.
    pub fn status_message(&self) -> String {
        match self.status {
            ConstraintStatus::WellConstrained => {
                format!(
                    "Component {} is well-constrained: {} free variables, {} equations, DOF = 0",
                    self.component.0, self.free_variables, self.total_equations
                )
            }
            ConstraintStatus::UnderConstrained => {
                format!(
                    "Component {} is under-constrained: {} free variables, {} equations, DOF = +{}",
                    self.component.0, self.free_variables, self.total_equations, self.dof
                )
            }
            ConstraintStatus::OverConstrained => {
                format!(
                    "Component {} is over-constrained: {} free variables, {} equations, DOF = {}",
                    self.component.0, self.free_variables, self.total_equations, self.dof
                )
            }
        }
    }

    /// Get a short status string.
    pub fn status_short(&self) -> &'static str {
        match self.status {
            ConstraintStatus::WellConstrained => "well-constrained",
            ConstraintStatus::UnderConstrained => "under-constrained",
            ConstraintStatus::OverConstrained => "over-constrained",
        }
    }
}

impl std::fmt::Display for DOFAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status_message())
    }
}

impl ConstraintStatus {
    /// Get a human-readable name for the status.
    pub fn name(&self) -> &'static str {
        match self {
            ConstraintStatus::WellConstrained => "well-constrained",
            ConstraintStatus::UnderConstrained => "under-constrained",
            ConstraintStatus::OverConstrained => "over-constrained",
        }
    }

    /// Check if this status indicates a solvable system.
    pub fn is_solvable(&self) -> bool {
        match self {
            ConstraintStatus::WellConstrained => true,
            ConstraintStatus::UnderConstrained => true, // Can solve with least-squares
            ConstraintStatus::OverConstrained => false, // May have no solution
        }
    }
}

impl std::fmt::Display for ConstraintStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_well_constrained() {
        // 4 total variables, 2 fixed, 2 free, 2 equations -> DOF = 0
        let analysis = DOFAnalysis::new(ComponentId(0), 4, 2, 2);

        assert_eq!(analysis.free_variables, 2);
        assert_eq!(analysis.dof, 0);
        assert_eq!(analysis.status, ConstraintStatus::WellConstrained);
        assert!(analysis.is_well_constrained());
        assert!(!analysis.is_under_constrained());
        assert!(!analysis.is_over_constrained());
    }

    #[test]
    fn test_under_constrained() {
        // 6 total, 2 fixed, 4 free, 2 equations -> DOF = +2
        let analysis = DOFAnalysis::new(ComponentId(1), 6, 2, 2);

        assert_eq!(analysis.free_variables, 4);
        assert_eq!(analysis.dof, 2);
        assert_eq!(analysis.status, ConstraintStatus::UnderConstrained);
        assert!(!analysis.is_well_constrained());
        assert!(analysis.is_under_constrained());
        assert!(!analysis.is_over_constrained());
    }

    #[test]
    fn test_over_constrained() {
        // 4 total, 1 fixed, 3 free, 5 equations -> DOF = -2
        let analysis = DOFAnalysis::new(ComponentId(2), 4, 1, 5);

        assert_eq!(analysis.free_variables, 3);
        assert_eq!(analysis.dof, -2);
        assert_eq!(analysis.status, ConstraintStatus::OverConstrained);
        assert!(!analysis.is_well_constrained());
        assert!(!analysis.is_under_constrained());
        assert!(analysis.is_over_constrained());
    }

    #[test]
    fn test_all_fixed() {
        // 4 total, 4 fixed, 0 free, 0 equations -> DOF = 0 (well-constrained trivially)
        let analysis = DOFAnalysis::new(ComponentId(3), 4, 4, 0);

        assert_eq!(analysis.free_variables, 0);
        assert_eq!(analysis.dof, 0);
        assert_eq!(analysis.status, ConstraintStatus::WellConstrained);
    }

    #[test]
    fn test_no_constraints() {
        // 4 total, 0 fixed, 4 free, 0 equations -> DOF = +4 (under-constrained)
        let analysis = DOFAnalysis::new(ComponentId(4), 4, 0, 0);

        assert_eq!(analysis.free_variables, 4);
        assert_eq!(analysis.dof, 4);
        assert_eq!(analysis.status, ConstraintStatus::UnderConstrained);
    }

    #[test]
    fn test_from_free() {
        let analysis = DOFAnalysis::from_free(ComponentId(5), 10, 8);

        assert_eq!(analysis.free_variables, 10);
        assert_eq!(analysis.total_equations, 8);
        assert_eq!(analysis.dof, 2);
        assert_eq!(analysis.status, ConstraintStatus::UnderConstrained);
    }

    #[test]
    fn test_status_message() {
        let well = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
        assert!(well.status_message().contains("well-constrained"));
        assert!(well.status_message().contains("DOF = 0"));

        let under = DOFAnalysis::new(ComponentId(1), 6, 2, 2);
        assert!(under.status_message().contains("under-constrained"));
        assert!(under.status_message().contains("DOF = +2"));

        let over = DOFAnalysis::new(ComponentId(2), 4, 1, 5);
        assert!(over.status_message().contains("over-constrained"));
        assert!(over.status_message().contains("DOF = -2"));
    }

    #[test]
    fn test_status_short() {
        let well = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
        assert_eq!(well.status_short(), "well-constrained");

        let under = DOFAnalysis::new(ComponentId(1), 6, 2, 2);
        assert_eq!(under.status_short(), "under-constrained");

        let over = DOFAnalysis::new(ComponentId(2), 4, 1, 5);
        assert_eq!(over.status_short(), "over-constrained");
    }

    #[test]
    fn test_display_dof_analysis() {
        let analysis = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
        let display = format!("{}", analysis);
        assert!(display.contains("well-constrained"));
        assert!(display.contains("Component 0"));
    }

    #[test]
    fn test_constraint_status_name() {
        assert_eq!(ConstraintStatus::WellConstrained.name(), "well-constrained");
        assert_eq!(ConstraintStatus::UnderConstrained.name(), "under-constrained");
        assert_eq!(ConstraintStatus::OverConstrained.name(), "over-constrained");
    }

    #[test]
    fn test_constraint_status_is_solvable() {
        assert!(ConstraintStatus::WellConstrained.is_solvable());
        assert!(ConstraintStatus::UnderConstrained.is_solvable());
        assert!(!ConstraintStatus::OverConstrained.is_solvable());
    }

    #[test]
    fn test_display_constraint_status() {
        let status = ConstraintStatus::WellConstrained;
        assert_eq!(format!("{}", status), "well-constrained");

        let status = ConstraintStatus::UnderConstrained;
        assert_eq!(format!("{}", status), "under-constrained");

        let status = ConstraintStatus::OverConstrained;
        assert_eq!(format!("{}", status), "over-constrained");
    }

    #[test]
    fn test_edge_case_zero_everything() {
        let analysis = DOFAnalysis::new(ComponentId(0), 0, 0, 0);
        assert_eq!(analysis.free_variables, 0);
        assert_eq!(analysis.dof, 0);
        assert_eq!(analysis.status, ConstraintStatus::WellConstrained);
    }

    #[test]
    fn test_edge_case_only_equations() {
        // 0 variables but 5 equations -> over-constrained
        let analysis = DOFAnalysis::new(ComponentId(0), 0, 0, 5);
        assert_eq!(analysis.free_variables, 0);
        assert_eq!(analysis.dof, -5);
        assert_eq!(analysis.status, ConstraintStatus::OverConstrained);
    }

    #[test]
    fn test_saturating_sub() {
        // More fixed than total (shouldn't happen in practice, but should be safe)
        let analysis = DOFAnalysis::new(ComponentId(0), 2, 10, 0);
        assert_eq!(analysis.free_variables, 0); // Saturates to 0
        assert_eq!(analysis.dof, 0);
    }

    #[test]
    fn test_large_positive_dof() {
        let analysis = DOFAnalysis::new(ComponentId(0), 1000, 0, 10);
        assert_eq!(analysis.dof, 990);
        assert!(analysis.is_under_constrained());
    }

    #[test]
    fn test_large_negative_dof() {
        let analysis = DOFAnalysis::new(ComponentId(0), 10, 0, 1000);
        assert_eq!(analysis.dof, -990);
        assert!(analysis.is_over_constrained());
    }

    #[test]
    fn test_equality() {
        let a1 = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
        let a2 = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
        assert_eq!(a1, a2);

        let a3 = DOFAnalysis::new(ComponentId(1), 4, 2, 2);
        assert_ne!(a1, a3); // Different component
    }

    #[test]
    fn test_clone() {
        let analysis = DOFAnalysis::new(ComponentId(0), 4, 2, 2);
        let cloned = analysis.clone();
        assert_eq!(analysis, cloned);
    }
}
