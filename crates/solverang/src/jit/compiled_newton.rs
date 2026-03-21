//! Compiled Newton step for small systems (N < 30).
//!
//! This module compiles the entire Newton iteration step — residual evaluation,
//! dense Jacobian assembly, LU decomposition, and back-substitution — into a
//! single native function via Cranelift.

/// A compiled Newton step function.
///
/// Performs one complete Newton iteration: evaluate F(x) and J(x), solve
/// J * delta = -F for delta, and compute x_new = x + delta.
pub struct CompiledNewtonStep {
    /// Native function pointer.
    ///
    /// Signature: fn(vars_in: *const f64, vars_out: *mut f64, scratch: *mut f64) -> f64
    /// Returns the residual norm ||F(x)||.
    step_fn: unsafe extern "C" fn(*const f64, *mut f64, *mut f64) -> f64,

    /// Number of variables (= number of residuals for square systems).
    n: usize,
}

impl CompiledNewtonStep {
    pub(crate) fn new(
        step_fn: unsafe extern "C" fn(*const f64, *mut f64, *mut f64) -> f64,
        n: usize,
    ) -> Self {
        Self { step_fn, n }
    }

    /// Perform one Newton step.
    ///
    /// - `vars_in`: current variable values (length n)
    /// - `vars_out`: updated variable values (length n)
    /// - `scratch`: scratch buffer (length >= m + m*n + n)
    ///
    /// Returns the residual norm ||F(vars_in)||.
    ///
    /// # Singular Jacobians
    ///
    /// The compiled LU decomposition does not use pivoting. If the Jacobian
    /// is singular or near-singular, the result will contain NaN/infinity.
    /// Callers should check `norm.is_finite()` after each step and abort
    /// if it returns false.
    pub fn evaluate(&self, vars_in: &[f64], vars_out: &mut [f64], scratch: &mut [f64]) -> f64 {
        let n = self.n;
        debug_assert!(vars_in.len() >= n);
        debug_assert!(vars_out.len() >= n);
        debug_assert!(scratch.len() >= n + n * n + n);

        let norm =
            unsafe { (self.step_fn)(vars_in.as_ptr(), vars_out.as_mut_ptr(), scratch.as_mut_ptr()) };

        // If LU decomposition hit a singular pivot, norm will be NaN.
        // In that case, copy vars_in to vars_out unchanged so the caller
        // sees no progress rather than corrupted data.
        if !norm.is_finite() {
            vars_out[..n].copy_from_slice(&vars_in[..n]);
        }

        norm
    }

    /// Get the system size.
    pub fn system_size(&self) -> usize {
        self.n
    }
}

impl std::fmt::Debug for CompiledNewtonStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledNewtonStep")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}
