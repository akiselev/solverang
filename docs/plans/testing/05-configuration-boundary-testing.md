# Plan 05: Configuration Boundary Testing

## Goal

Solver and system configurations contain numeric parameters with implicit valid
ranges. Invalid or extreme configurations can cause infinite loops, NaN
propagation, division by zero, or nonsensical behavior. This plan audits every
configuration surface in both the legacy solver layer and the V3 constraint
system layer, documents boundary expectations, and specifies proptest strategies
to fuzz the entire configuration space.

---

## Part A: Legacy Solver Configurations

### A.1 `SolverConfig` (Newton-Raphson)

Source: `crates/solverang/src/solver/config.rs`

| Parameter | Type | Default | Valid Range | Boundary Behavior |
|-----------|------|---------|-------------|-------------------|
| `max_iterations` | `usize` | 200 | 0..=usize::MAX | 0: immediate NotConverged |
| `tolerance` | `f64` | 1e-8 | >0 | 0.0: never satisfies convergence check; NaN: all comparisons false; Inf: always converges on first check |
| `line_search` | `bool` | true | N/A | false: full Newton step always taken |
| `armijo_c` | `f64` | 1e-4 | (0, 1) | 0.0: any step accepted; 1.0: requires exact linear decrease; >1: impossible to satisfy |
| `backtrack_factor` | `f64` | 0.5 | (0, 1) | 0.0: step becomes zero; 1.0: step never shrinks (stalls); >1: step grows (diverges) |
| `max_line_search_iterations` | `usize` | 20 | 0..=usize::MAX | 0: line search never runs, full step always taken |
| `min_step_size` | `f64` | 1e-12 | >=0 | 0.0: line search never terminates by step size; Inf: first backtrack terminates |
| `svd_tolerance` | `f64` | 1e-10 | >=0 | 0.0: no singular values rejected; NaN: all rejected; Inf: all rejected |

```rust
#[test]
fn test_nr_max_iterations_zero() {
    let config = SolverConfig { max_iterations: 0, ..Default::default() };
    let solver = Solver::new(config);
    let result = solver.solve(&Rosenbrock::new(), &[0.0, 0.0]);
    assert!(!result.is_converged());
}

#[test]
fn test_nr_tolerance_nan_terminates() {
    let config = SolverConfig { tolerance: f64::NAN, ..Default::default() };
    let solver = Solver::new(config);
    let result = solver.solve(&LinearProblem::identity(2), &[0.0, 0.0]);
    // NaN comparisons always false; must not infinite-loop
    assert!(!result.is_converged());
}

#[test]
fn test_nr_tolerance_infinity_converges_immediately() {
    let config = SolverConfig { tolerance: f64::INFINITY, ..Default::default() };
    let solver = Solver::new(config);
    let result = solver.solve(&Rosenbrock::new(), &[0.0, 0.0]);
    assert!(result.is_converged());
}

#[test]
fn test_nr_backtrack_factor_one_stalls() {
    let config = SolverConfig {
        backtrack_factor: 1.0,
        max_line_search_iterations: 10,
        ..Default::default()
    };
    let solver = Solver::new(config);
    let result = solver.solve(&Rosenbrock::new(), &[0.0, 0.0]);
    // Should exhaust line search iterations, not infinite-loop
}

#[test]
fn test_nr_backtrack_factor_greater_than_one() {
    let config = SolverConfig {
        backtrack_factor: 2.0,
        max_line_search_iterations: 5,
        ..Default::default()
    };
    let solver = Solver::new(config);
    let _ = solver.solve(&Rosenbrock::new(), &[0.0, 0.0]);
    // Must not panic or infinite-loop
}
```

### A.2 `LMConfig` (Levenberg-Marquardt)

Source: `crates/solverang/src/solver/lm_config.rs`

| Parameter | Type | Default | Valid Range | Boundary Behavior |
|-----------|------|---------|-------------|-------------------|
| `ftol` | `f64` | 1e-12 | >=0 | 0.0: relative reduction never satisfied; NaN: always false |
| `xtol` | `f64` | 1e-12 | >=0 | Same as ftol |
| `gtol` | `f64` | 1e-12 | >=0 | Same as ftol |
| `stepbound` | `f64` | 100.0 | >0 | 0.0: zero trust region; Inf: unconstrained step; negative: undefined |
| `patience` | `usize` | 200 | >0 | 0: max_fev = 0, no evaluations allowed |
| `scale_diag` | `bool` | true | N/A | false: no diagonal rescaling |

```rust
#[test]
fn test_lm_patience_zero() {
    let config = LMConfig { patience: 0, ..Default::default() };
    let solver = LMSolver::new(config);
    let result = solver.solve(&Rosenbrock::new(), &[0.0, 0.0]);
    // patience * (n+1) = 0 function evaluations allowed
    assert!(!result.is_converged());
}

#[test]
fn test_lm_stepbound_zero() {
    let config = LMConfig { stepbound: 0.0, ..Default::default() };
    let solver = LMSolver::new(config);
    let result = solver.solve(&LinearProblem::identity(2), &[1.0, 1.0]);
    // Zero trust region: solver cannot move; must not panic
}

#[test]
fn test_lm_all_tolerances_nan() {
    let config = LMConfig {
        ftol: f64::NAN, xtol: f64::NAN, gtol: f64::NAN,
        ..Default::default()
    };
    let solver = LMSolver::new(config);
    let result = solver.solve(&LinearProblem::identity(2), &[1.0, 1.0]);
    // All convergence checks fail; should exhaust function evaluations
    assert!(!result.is_converged());
}

#[test]
fn test_lm_stepbound_negative() {
    let config = LMConfig { stepbound: -10.0, ..Default::default() };
    let solver = LMSolver::new(config);
    let _ = solver.solve(&LinearProblem::identity(2), &[1.0, 1.0]);
    // Must not panic
}
```

### A.3 `SparseSolverConfig`

| Parameter | Type | Default | Valid Range | Boundary Behavior |
|-----------|------|---------|-------------|-------------------|
| `max_iterations` | `usize` | 200 | >0 | 0: no iterations |
| `tolerance` | `f64` | 1e-10 | >0 | Same issues as NR |

### A.4 `ParallelSolverConfig`

| Parameter | Type | Default | Valid Range | Boundary Behavior |
|-----------|------|---------|-------------|-------------------|
| `num_threads` | `Option<usize>` | None | >0 or None | Some(0): Rayon may panic or use 1 thread |

---

## Part B: V3 System Configuration

### B.1 `SystemConfig`

Source: `crates/solverang/src/system.rs`

`SystemConfig` wraps both `LMConfig` and `SolverConfig`. When solving through the
V3 `ConstraintSystem`, the LM config is used for the pipeline's numerical solve
phase, and the solver config is used by AutoSolver or Newton-Raphson fallback.

```rust
#[test]
fn test_system_config_lm_boundaries() {
    // Zero patience through SystemConfig
    let config = SystemConfig {
        lm_config: LMConfig { patience: 0, ..Default::default() },
        solver_config: SolverConfig::default(),
    };
    let mut system = ConstraintSystem::with_config(config);
    // Add a simple 1-constraint system
    let eid = system.alloc_entity_id();
    let px = system.alloc_param(0.0, eid);
    system.add_entity(Box::new(/* TestPoint */));
    let cid = system.alloc_constraint_id();
    system.add_constraint(Box::new(/* FixValue { target: 5.0 } */));
    let result = system.solve();
    // Should not panic; may not converge
}

#[test]
fn test_system_config_tolerance_nan_through_pipeline() {
    let config = SystemConfig {
        lm_config: LMConfig { ftol: f64::NAN, xtol: f64::NAN, gtol: f64::NAN,
                               ..Default::default() },
        solver_config: SolverConfig { tolerance: f64::NAN, ..Default::default() },
    };
    let mut system = ConstraintSystem::with_config(config);
    // ... add constraints ...
    let result = system.solve();
    // Must terminate without panic
}

#[test]
fn test_system_config_fast_vs_robust_presets() {
    let fast_config = SystemConfig {
        lm_config: LMConfig::fast(),
        solver_config: SolverConfig::fast(),
    };
    let robust_config = SystemConfig {
        lm_config: LMConfig::robust(),
        solver_config: SolverConfig::robust(),
    };
    // Both should produce valid SystemConfigs and solve a simple system
    for config in [fast_config, robust_config] {
        let mut system = ConstraintSystem::with_config(config);
        // ... add well-behaved constraints ...
        let result = system.solve();
        assert!(matches!(
            result.status,
            SystemStatus::Solved | SystemStatus::PartiallySolved
        ));
    }
}
```

### B.2 SVD Tolerance in `graph/dof.rs` and `graph/redundancy.rs`

Both `analyze_dof()` and `analyze_redundancy()` use an SVD tolerance parameter to
determine the numerical rank of the Jacobian. This tolerance is currently
hardcoded or passed as a function argument (1e-10).

| Tolerance Value | Effect on DOF/Redundancy |
|-----------------|--------------------------|
| 0.0 | All singular values treated as nonzero; rank = min(m,n); no redundancy detected |
| 1e-20 | Very tight; near-singular directions not detected |
| 1e-10 (default) | Normal behavior |
| 1.0 | Very loose; many directions appear rank-deficient; false redundancy |
| Inf | All singular values below Inf; rank = 0; everything appears redundant |
| NaN | Comparison undefined; behavior depends on implementation |

```rust
#[test]
fn test_dof_svd_tolerance_zero() {
    // With tolerance=0, all singular values are nonzero, so rank = min(m,n).
    // A known-redundant system should NOT report redundancy.
    let (entities, constraints, store, mapping) = build_redundant_system();
    let analysis = analyze_redundancy(&constraints, &store, &mapping, 0.0);
    // Should find rank == equation_count (no redundancy detected)
}

#[test]
fn test_dof_svd_tolerance_large() {
    // With tolerance=1.0, many singular values are rejected.
    let (entities, constraints, store, mapping) = build_well_constrained_system();
    let analysis = analyze_redundancy(&constraints, &store, &mapping, 1.0);
    // May falsely report redundancy
}

#[test]
fn test_dof_analysis_with_nan_tolerance_does_not_panic() {
    let (entities, constraints, store, mapping) = build_simple_system();
    let _ = analyze_redundancy(&constraints, &store, &mapping, f64::NAN);
    // Must not panic
}
```

### B.3 Closed-Form Solver Tolerances

Source: `crates/solverang/src/solve/closed_form.rs`

The circle-circle intersection solver (`solve_two_distances`) computes a
discriminant to determine whether two circles intersect. There is an implicit
tolerance for "nearly tangent" circles.

```rust
#[test]
fn test_circle_circle_nearly_tangent() {
    // Two circles that are exactly tangent (discriminant = 0).
    // Should produce 1 valid solution branch, not NaN.
    let (pattern, constraints, store) = build_tangent_circle_circle();
    let result = solve_pattern(&pattern, &constraints, &store);
    assert!(result.unwrap().solved);
}

#[test]
fn test_circle_circle_no_intersection() {
    // Two circles that do not intersect (negative discriminant).
    let (pattern, constraints, store) = build_non_intersecting_circles();
    let result = solve_pattern(&pattern, &constraints, &store);
    assert!(!result.unwrap().solved);
}

#[test]
fn test_circle_circle_nearly_no_intersection() {
    // Circles separated by epsilon more than sum of radii.
    // Discriminant is a very small negative number.
    let (pattern, constraints, store) = build_nearly_non_intersecting_circles();
    let result = solve_pattern(&pattern, &constraints, &store);
    // Should not produce NaN coordinates
    if result.as_ref().unwrap().solved {
        for (_, v) in &result.unwrap().values {
            assert!(v.is_finite());
        }
    }
}
```

### B.4 `DragResult` SVD Tolerance

Source: `crates/solverang/src/solve/drag.rs`

The `project_drag()` function uses an SVD tolerance to identify the null space
of the constraint Jacobian. The tolerance passed is currently 1e-10 in
`ConstraintSystem::drag()`.

```rust
#[test]
fn test_drag_svd_tolerance_zero() {
    // With tolerance=0, null space is empty for any non-zero singular values.
    // Drag should be completely absorbed (preservation_ratio = 0).
    let result = project_drag_with_tolerance(&constraints, &store, &mapping, &displacements, 0.0);
    assert!(result.preservation_ratio <= f64::EPSILON);
}

#[test]
fn test_drag_svd_tolerance_very_large() {
    // With very large tolerance, entire space is "null" -- drag passes through unmodified.
    let result = project_drag_with_tolerance(&constraints, &store, &mapping, &displacements, 1e10);
    assert!((result.preservation_ratio - 1.0).abs() < 1e-6);
}
```

---

## Part C: Sketch2D Constraint Boundary Values

### C.1 Geometric Parameter Boundaries

The sketch2d constraints use squared formulations (`dx^2 + dy^2 - d^2`).
Test degenerate geometric values.

| Constraint | Boundary Value | Expected Behavior |
|-----------|---------------|-------------------|
| `DistancePtPt` | `distance = 0.0` (coincident) | Squared formulation `d^2 = 0`, residual = `dx^2+dy^2`, well-defined |
| `DistancePtPt` | `distance < 0` (invalid) | `target_sq = d^2` is positive, so constraint works but is semantically wrong |
| `DistancePtPt` | `distance = f64::MAX` | `target_sq` overflows; residual is `Inf` |
| `DistancePtLine` | Line endpoints coincident | Division by zero in line direction? |
| `Angle` | `angle = 0.0` (parallel) | Should behave like `Parallel` |
| `Angle` | `angle = PI` (anti-parallel) | Should be equivalent |
| `Angle` | `angle = PI/2` | Should behave like `Perpendicular` |
| `Angle` | `angle = 2*PI` | Equivalent to `angle = 0` |
| `Angle` | `angle = NaN` | Residuals should not propagate to valid-looking values |
| `PointOnCircle` | `radius = 0` | Point must equal circle center |
| `TangentCircleCircle` | Both radii = 0 | Two point-circles; tangency degenerates |
| `TangentLineCircle` | Circle radius = 0 | Degenerates to point-on-line |
| `EqualLength` | Both segments have zero length | 0 == 0; trivially satisfied |

```rust
#[test]
fn test_distance_zero_squared_formulation() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_point(1.0, 2.0);
    let p1 = b.add_point(1.5, 2.5);
    b.constrain_distance(p0, p1, 0.0);
    let mut system = b.build();
    let result = system.solve();
    // Points should become coincident
    assert!(matches!(result.status, SystemStatus::Solved | SystemStatus::PartiallySolved));
}

#[test]
fn test_distance_negative_value() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(1.0, 0.0);
    b.constrain_distance(p0, p1, -5.0);
    let mut system = b.build();
    let result = system.solve();
    // target_sq = 25.0, so it will solve for distance=5.0
    // This is arguably a bug; document the behavior
}

#[test]
fn test_angle_zero_vs_pi() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(1.0, 0.0);
    let p2 = b.add_point(2.0, 0.0);
    let p3 = b.add_point(3.0, 0.0);
    b.constrain_angle(/* line 0-1, line 2-3, angle = 0.0 */);
    let mut system = b.build();
    let result = system.solve();
    // Lines should be parallel
}

#[test]
fn test_coincident_points_for_angle_constraint() {
    // When two points defining a line are coincident, the line direction is
    // undefined. The angle constraint should handle this gracefully.
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_point(1.0, 1.0);
    let p1 = b.add_point(1.0, 1.0); // coincident with p0
    let p2 = b.add_point(0.0, 0.0);
    let p3 = b.add_point(1.0, 0.0);
    b.constrain_angle(/* line p0-p1, line p2-p3, angle = PI/4 */);
    let mut system = b.build();
    // Should not produce NaN or panic
    let _ = system.solve();
}
```

### C.2 Sketch3D Constraint Boundaries

| Constraint | Boundary Value | Expected Behavior |
|-----------|---------------|-------------------|
| `Distance3D` | `distance = 0.0` | Same as 2D: squared formulation handles it |
| `PointOnPlane` | Plane normal = zero vector | Degenerate plane; constraint is vacuous |
| `Coplanar` | All points coincident | Trivially coplanar; residual = 0 |
| `Coaxial` | Axis direction = zero vector | Degenerate axis |

### C.3 Assembly Constraint Boundaries

| Constraint | Boundary Value | Expected Behavior |
|-----------|---------------|-------------------|
| `UnitQuaternion` | All components zero | Residual = `-1`; solver should recover to unit quaternion |
| `UnitQuaternion` | Very large quaternion | Residual = `qw^2+...+qz^2 - 1 >> 0` |
| `Mate` | Bodies coincident | Zero translation residual |
| `Gear` | Ratio = 0 | One body does not rotate |
| `Gear` | Ratio = Inf | Undefined behavior; should fail gracefully |
| `Gear` | Ratio < 0 | Reverse rotation |

```rust
#[test]
fn test_unit_quaternion_from_zero() {
    // Start with qw=qx=qy=qz=0. UnitQuaternion constraint should drive
    // the system toward a unit quaternion.
    let mut system = ConstraintSystem::new();
    let eid = system.alloc_entity_id();
    let qw = system.alloc_param(0.0, eid);
    let qx = system.alloc_param(0.0, eid);
    let qy = system.alloc_param(0.0, eid);
    let qz = system.alloc_param(0.0, eid);
    // ... add RigidBody entity and UnitQuaternion constraint ...
    let result = system.solve();
    let norm_sq = system.get_param(qw).powi(2)
        + system.get_param(qx).powi(2)
        + system.get_param(qy).powi(2)
        + system.get_param(qz).powi(2);
    assert!((norm_sq - 1.0).abs() < 1e-6, "Quaternion not normalized: {}", norm_sq);
}
```

---

## Part D: Proptest Strategies

### D.1 Legacy Config Strategies

```rust
use proptest::prelude::*;

fn solver_config_strategy() -> impl Strategy<Value = SolverConfig> {
    (
        0usize..1000,                       // max_iterations
        prop_oneof![                        // tolerance
            Just(0.0), Just(f64::NAN), Just(f64::INFINITY),
            Just(f64::NEG_INFINITY), Just(f64::MIN_POSITIVE),
            Just(f64::EPSILON), Just(-1.0),
            1e-15f64..1e-1,
        ],
        any::<bool>(),                      // line_search
        prop_oneof![                        // armijo_c
            Just(0.0), Just(1.0), Just(-1.0), Just(2.0),
            0.0001f64..0.5,
        ],
        prop_oneof![                        // backtrack_factor
            Just(0.0), Just(1.0), Just(2.0),
            0.01f64..0.99,
        ],
        0usize..100,                        // max_line_search_iterations
        prop_oneof![                        // min_step_size
            Just(0.0), Just(f64::INFINITY),
            1e-20f64..1.0,
        ],
        prop_oneof![                        // svd_tolerance
            Just(0.0), Just(f64::NAN), Just(f64::INFINITY),
            1e-15f64..1e-1,
        ],
    ).prop_map(|(max_iter, tol, ls, armijo, bt, max_ls, min_step, svd_tol)| {
        SolverConfig {
            max_iterations: max_iter,
            tolerance: tol,
            line_search: ls,
            armijo_c: armijo,
            backtrack_factor: bt,
            max_line_search_iterations: max_ls,
            min_step_size: min_step,
            svd_tolerance: svd_tol,
        }
    })
}

fn lm_config_strategy() -> impl Strategy<Value = LMConfig> {
    (
        prop_oneof![Just(0.0), Just(f64::NAN), 1e-15f64..1e-1],  // ftol
        prop_oneof![Just(0.0), Just(f64::NAN), 1e-15f64..1e-1],  // xtol
        prop_oneof![Just(0.0), Just(f64::NAN), 1e-15f64..1e-1],  // gtol
        prop_oneof![Just(0.0), Just(-10.0), Just(f64::INFINITY), 0.1f64..1000.0], // stepbound
        0usize..500,                                                // patience
        any::<bool>(),                                              // scale_diag
    ).prop_map(|(ftol, xtol, gtol, stepbound, patience, scale_diag)| {
        LMConfig { ftol, xtol, gtol, stepbound, patience, scale_diag }
    })
}
```

### D.2 V3 SystemConfig Strategy

```rust
fn system_config_strategy() -> impl Strategy<Value = SystemConfig> {
    (lm_config_strategy(), solver_config_strategy()).prop_map(|(lm, solver)| {
        SystemConfig {
            lm_config: lm,
            solver_config: solver,
        }
    })
}
```

### D.3 SVD Tolerance Strategy

```rust
fn svd_tolerance_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::MIN_POSITIVE),
        Just(1e-15),
        Just(1e-10),  // default
        Just(1e-5),
        Just(1.0),
        1e-15f64..1.0,
    ]
}
```

### D.4 No-Panic Property Tests

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_nr_solver_no_panic_with_any_config(config in solver_config_strategy()) {
        let problem = LinearProblem::identity(2);
        let solver = Solver::new(config);
        let _ = solver.solve(&problem, &[0.0, 0.0]);
    }

    #[test]
    fn prop_lm_solver_no_panic_with_any_config(config in lm_config_strategy()) {
        let problem = LinearProblem::identity(2);
        let solver = LMSolver::new(config);
        let _ = solver.solve(&problem, &[0.0, 0.0]);
    }

    #[test]
    fn prop_v3_system_no_panic_with_any_config(config in system_config_strategy()) {
        let mut system = ConstraintSystem::with_config(config);
        let eid = system.alloc_entity_id();
        let px = system.alloc_param(0.0, eid);
        // ... add entity and simple FixValue constraint ...
        let _ = system.solve();
    }
}
```

### D.5 Geometric Boundary Proptest

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_sketch2d_distance_any_value(
        distance in prop_oneof![
            Just(0.0), Just(-1.0), Just(f64::NAN),
            Just(f64::INFINITY), Just(f64::MIN_POSITIVE),
            0.001f64..1000.0,
        ],
    ) {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_point(1.0, 0.0);
        b.constrain_distance(p0, p1, distance);
        let mut system = b.build();
        // Must not panic
        let _ = system.solve();
    }

    #[test]
    fn prop_sketch2d_angle_any_value(
        angle in prop_oneof![
            Just(0.0), Just(std::f64::consts::PI),
            Just(std::f64::consts::FRAC_PI_2),
            Just(f64::NAN), Just(f64::INFINITY),
            Just(-std::f64::consts::PI),
            -10.0f64..10.0,
        ],
    ) {
        // ... build angle constraint with arbitrary angle value ...
        // Must not panic
    }
}
```

---

## Part E: Suggested Validation

### E.1 `Config::validate()` Methods

Add validation at construction time for both legacy and V3 configs:

```rust
impl SolverConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.tolerance.is_nan() || self.tolerance < 0.0 {
            return Err(ConfigError::InvalidTolerance(self.tolerance));
        }
        if self.backtrack_factor <= 0.0 || self.backtrack_factor >= 1.0 {
            return Err(ConfigError::InvalidBacktrackFactor(self.backtrack_factor));
        }
        if self.armijo_c <= 0.0 || self.armijo_c >= 1.0 {
            return Err(ConfigError::InvalidArmijoC(self.armijo_c));
        }
        if self.svd_tolerance.is_nan() || self.svd_tolerance < 0.0 {
            return Err(ConfigError::InvalidSvdTolerance(self.svd_tolerance));
        }
        Ok(())
    }
}

impl LMConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, val) in [("ftol", self.ftol), ("xtol", self.xtol), ("gtol", self.gtol)] {
            if val.is_nan() || val < 0.0 {
                return Err(ConfigError::InvalidTolerance(val));
            }
        }
        if self.stepbound <= 0.0 || !self.stepbound.is_finite() {
            return Err(ConfigError::InvalidStepbound(self.stepbound));
        }
        if self.patience == 0 {
            return Err(ConfigError::ZeroPatience);
        }
        Ok(())
    }
}

impl SystemConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.lm_config.validate()?;
        self.solver_config.validate()?;
        Ok(())
    }
}
```

### E.2 Sketch2D Builder Validation

```rust
impl Sketch2DBuilder {
    /// Validate geometric parameter sanity before building the system.
    pub fn validate(&self) -> Result<(), GeometryError> {
        // Check for negative distances, NaN values, etc.
    }
}
```

---

## File Organization

```
crates/solverang/tests/
  config_boundary_tests/
    mod.rs
    legacy_solver_config.rs    -- NR and LM config boundary tests
    system_config.rs           -- V3 SystemConfig boundary tests
    svd_tolerance.rs           -- DOF/redundancy/drag SVD tolerance
    closed_form_tolerance.rs   -- Circle-circle discriminant boundaries
    sketch2d_boundaries.rs     -- Geometric value boundaries
    sketch3d_boundaries.rs     -- 3D geometric value boundaries
    assembly_boundaries.rs     -- Quaternion, gear ratio boundaries
    proptest_configs.rs        -- Proptest strategies and no-panic tests
```

## Estimated Effort

| Task | Time |
|------|------|
| Legacy solver config boundary tests (NR + LM) | 3-4 hours |
| V3 SystemConfig boundary tests | 2-3 hours |
| SVD tolerance boundary tests (dof, redundancy, drag) | 2-3 hours |
| Closed-form solver tolerance tests | 2 hours |
| Sketch2D geometric boundary tests (16 constraints) | 4-5 hours |
| Sketch3D geometric boundary tests (8 constraints) | 2-3 hours |
| Assembly boundary tests (quaternion, gear, mate) | 2-3 hours |
| Proptest strategies and no-panic fuzzing | 3-4 hours |
| Config validation methods (optional) | 2-3 hours |
| **Total** | **~22-30 hours** |
