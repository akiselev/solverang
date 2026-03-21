# Integration Patterns: Constraint Solving and Optimization in CAD Systems

How optimization and nonlinear constraint solving interact in practice,
with recommendations for Solverang.

---

## Part 1: CAD Optimization Use Cases

### 1.1 Shape Optimization (Minimize Weight Subject to Stress Constraints)

**Mathematical formulation:**

```
minimize    Volume(x)          (or mass = rho * Volume)
subject to  sigma_max(x) <= sigma_allow   (stress constraint)
            g_i(x) >= 0                    (geometric feasibility)
            x_L <= x <= x_U                (design variable bounds)
```

Where `x` is a vector of geometric parameters (fillet radii, wall thicknesses,
cross-section dimensions, spline control point positions) and `sigma_max(x)` is
obtained from FEA.

**Constraint types:** Nonlinear inequality (stress), linear/nonlinear equality
(geometric compatibility), bound constraints.

**Typical problem size:** 5--200 design variables, 1--50 constraints, but each
function evaluation requires a full FEA solve (expensive).

**Required solver features:** Gradient-based optimization (SQP, augmented
Lagrangian), sensitivity analysis (adjoint methods for FEA derivatives),
bound handling, merit function for infeasible iterates.

**Key insight:** The inner loop is an FEA solve (a *different* kind of
constraint satisfaction -- equilibrium equations), while the outer loop is NLP.
This is inherently bilevel: the optimizer varies geometry, the FEA solver
enforces physics.

---

### 1.2 Topology Optimization (Optimal Material Distribution)

**Mathematical formulation (SIMP method):**

```
minimize    c(rho) = U^T K(rho) U        (compliance / strain energy)
subject to  K(rho) U = F                  (equilibrium)
            sum(rho_e * v_e) <= V_max     (volume fraction)
            0 < rho_min <= rho_e <= 1     (density bounds)
```

The Young's modulus is interpolated as `E_e = E_min + rho_e^p (E_solid - E_min)`
with penalization power `p > 1` (typically p=3) to push densities toward 0 or 1.
`E_min` is a small regularization value (typically 1e-9 * E_solid) that prevents
singular stiffness matrices when rho_e approaches zero.

**Constraint types:** Linear volume constraint, PDE constraint (equilibrium),
bound constraints on density variables. Manufacturing constraints (minimum
feature size, symmetry, draw direction) add additional inequality constraints.

**Typical problem size:** 10,000--1,000,000+ density variables (one per mesh
element). Industrial problems commonly use ~1M elements.

**Required solver features:** Large-scale optimizer (MMA -- Method of Moving
Asymptotes, or OC -- Optimality Criteria), sparse linear algebra, sensitivity
filtering, continuation on penalization parameter.

**Relevance to Solverang:** Low for direct implementation (requires FEA
backend), but the mathematical structure -- large-scale bound-constrained
optimization with a single linear constraint -- is a useful benchmark.

---

### 1.3 Mechanism Optimization (Linkage Synthesis)

**Mathematical formulation:**

```
minimize    sum_{i=1}^{N} ||P_coupler(theta_i; x) - P_target_i||^2
subject to  Grashof condition: s + l <= p + q
            Assembly constraints (loop closure)
            Sequence / order constraints
            x_L <= x <= x_U
```

Where `x = [a, b, c, d, x0, y0, ...]` are link lengths and pivot positions,
`theta_i` are input crank angles, and `P_coupler` is the coupler point position
computed from forward kinematics.

**Constraint types:** Nonlinear equality (loop closure / kinematic constraints),
nonlinear inequality (Grashof, assembly, collision avoidance), bounds.

**Typical problem size:** 4--20 design variables, 5--30 constraints, N=6--50
target path points.

**Required solver features:** This is a constrained nonlinear least-squares
problem. The objective has residual structure (sum of squared point deviations),
so Gauss-Newton/LM structure can be exploited. Constraints require SQP or
augmented Lagrangian. Multiple local minima are common -- multi-start or global
methods help.

**Relevance to Solverang:** HIGH. The inner kinematics solve is geometric
constraint satisfaction (exactly what Solverang does). The outer optimization
varies mechanism parameters. This is a natural first use case.

---

### 1.4 Tolerance Optimization (Minimize Manufacturing Cost)

**Mathematical formulation:**

```
minimize    sum_i Cost_i(t_i)             (manufacturing cost)
subject to  sigma_assembly(t) <= sigma_max (assembly quality)
            t_i_min <= t_i <= t_i_max      (feasible tolerance range)
            RSS or worst-case stackup <= gap_allow
```

Where `t_i` are individual part tolerances and `Cost_i(t_i)` is typically
modeled as `C_i / t_i^alpha` (tighter tolerance = higher cost, with
`alpha ~ 1--2`).

**Constraint types:** Nonlinear inequality (stackup analysis, which may involve
Monte Carlo or RSS formulas), bounds.

**Typical problem size:** 10--100 tolerance variables, 5--50 stackup
constraints.

**Required solver features:** Bound-constrained optimization (L-BFGS-B),
possibly stochastic objective evaluation if using Monte Carlo stackup.

**Relevance to Solverang:** MEDIUM. Tolerance stackup analysis requires a
geometric constraint solver to compute assembly configurations at extreme
tolerance values. Solverang's decomposition and sensitivity analysis could feed
directly into tolerance optimization.

---

### 1.5 Assembly Optimization (Minimize Interference/Clearance)

**Mathematical formulation:**

```
minimize    sum max(0, penetration_ij(x))^2   (minimize interference)
   or       -min_ij clearance_ij(x)            (maximize minimum clearance)
subject to  assembly_constraints(x) = 0         (mates, coaxiality, etc.)
            x_L <= x <= x_U
```

**Constraint types:** Nonlinear equality (assembly constraints -- the geometric
constraint system), nonlinear inequality (clearance/interference), bounds.

**Typical problem size:** 6 DOF per rigid body, 10--500 parts, 50--3000
variables, 100--5000 constraint equations.

**Required solver features:** Constrained optimization with the assembly
constraint solver as an equality constraint evaluator. Spatial queries for
interference detection. SQP or augmented Lagrangian.

**Relevance to Solverang:** HIGH. The assembly constraint solver IS Solverang's
constraint system. Adding an objective function on top of it is the natural
extension path described in `docs/notes/optimization.md`.

---

### 1.6 Parametric Design Optimization

**Mathematical formulation:**

```
minimize    f(p)                           (performance metric)
subject to  sketch_constraints(p, x) = 0  (parametric CAD constraints)
            g(p) <= 0                      (design rules)
            p_L <= p <= p_U
```

Where `p` are design parameters (dimensions in a sketch) and `x` are the
derived geometric positions computed by the constraint solver.

**Constraint types:** The sketch constraints are an inner equality system
solved by the geometric constraint solver. Design rules are outer inequality
constraints.

**Typical problem size:** 3--50 design parameters, 50--500 inner constraint
equations (solved implicitly).

**Required solver features:** Bilevel structure: outer optimizer varies `p`,
inner solver satisfies sketch constraints to determine `x(p)`. Sensitivity
`dx/dp` via implicit differentiation:

```
J_x * dx/dp = -J_p   (from differentiating sketch_constraints(p,x) = 0)
```

This requires the Jacobian from the constraint solver, which Solverang already
computes.

**Relevance to Solverang:** HIGHEST. This is the most natural use case.
Solverang already solves `sketch_constraints(p, x) = 0`. Adding parametric
optimization means wrapping an outer optimizer around the existing solver.

---

### 1.7 Path Planning / Motion Optimization

**Mathematical formulation:**

```
minimize    sum ||q_{i+1} - q_i||^2       (minimize total motion)
   or       T_total                         (minimize time)
subject to  kinematic_constraints(q_i) = 0  (joint limits, workspace)
            obstacle_avoidance(q_i) >= d_min
            q_min <= q_i <= q_max
```

**Typical problem size:** 50--500 waypoint variables (discretized trajectory),
dense constraint Jacobian.

**Required solver features:** SQP or interior point, warm-starting between
time steps, sparse structure exploitation.

**Relevance to Solverang:** LOW priority for initial implementation, but the
kinematic constraint inner loop maps well to Solverang's architecture.

---

### 1.8 Fit Optimization (Curves/Surfaces to Point Clouds)

**Mathematical formulation:**

```
minimize    sum_i ||S(u_i, v_i; P) - Q_i||^2   (point deviation)
subject to  0 <= u_i <= 1, 0 <= v_i <= 1        (parameter bounds)
            continuity(S, S_adj) = 0              (G1/G2 between patches)
```

Where `P` are control point positions (and weights for NURBS), `(u_i, v_i)`
are parameter values for each data point, and `Q_i` are measured points.

**Constraint types:** Bound constraints on parameters, equality constraints
for inter-patch continuity.

**Typical problem size:** 50--5000 control point coordinates, 100--100,000
data points, making the residual vector very large. The problem has
exploitable least-squares structure.

**Required solver features:** Levenberg-Marquardt with parameter bounds
(bounded LM), alternating optimization between parameter values and control
points, sparse Jacobian handling.

**Relevance to Solverang:** HIGH. Solverang's LM solver already handles
nonlinear least-squares. Adding bound constraints and structuring the
alternating optimization is a natural extension. The Constraint trait's
`residuals()` + `jacobian()` maps directly to fitting residuals.

---

## Part 2: Integration Patterns in Existing CAD Systems

### 2.1 SolidWorks Simulation (Optimization Studies)

**Architecture:** SolidWorks uses a layered approach:

1. **Design Study Manager** defines design variables (CAD dimensions), constraints
   (simulation outputs like max stress, displacement), and objectives (minimize
   mass, maximize factor of safety).
2. **Parametric sweep** evaluates the design space via Design of Experiments (DoE).
3. **Optimization engine** uses response surface methodology (RSM) or direct
   iterative convergence. The optimizer calls the CAD rebuilder + FEA solver as
   a black box.

**Key pattern:** *Black-box optimization over a parametric rebuild + simulation
pipeline.* The constraint solver (sketch solver) and the optimizer are
completely decoupled. The optimizer only sees parameter-to-output mappings.

**Strengths:** Simple, works with any simulation type.
**Weaknesses:** No gradient information (finite differences or surrogate-based),
slow convergence, cannot exploit problem structure.

### 2.2 CATIA (Generative Design / Knowledge Advisor)

**Architecture:** CATIA Knowledge Advisor allows users to define engineering
rules and optimization criteria using a scripting language. The generative
design workflow:

1. Define functional requirements (loads, boundary conditions).
2. Define design space (bounding volume, keep-in/keep-out zones).
3. Run topology optimization (level-set method in current CATIA V5/V6
   Functional Generative Design; SIMP was used in earlier releases and some
   partner tools).
4. Reconstruct smooth CAD geometry from the optimized field.

**Key pattern:** *Separate optimization engine with CAD reconstruction
post-processing.* The geometric constraint solver handles sketch-level design,
while optimization operates at a higher level (structural or topological).

### 2.3 Siemens NX (Design Optimization)

**Architecture:** NX provides two integrated approaches:

1. **Design Space Explorer** -- parametric optimization that varies CAD dimensions
   while respecting design constraints. Uses surrogate models and trust-region
   schemes.
2. **Topology Optimizer** -- manufacturing-aware topology optimization with
   constraints for draft angles, minimum thickness, symmetry, and additive
   manufacturing support.

**Key pattern:** *Trust-region optimization with manufacturing constraints
embedded in the topology optimizer.* The constraint solver handles parametric
design, while a separate optimizer handles structural optimization. The two
interact through the parametric model -- changing a dimension triggers a
constraint re-solve, mesh regeneration, and FEA re-solve.

### 2.4 FreeCAD + Optimization Plugins

**Architecture:** FreeCAD's approach is modular and plugin-based:

- **BESO** (Bidirectional Evolutionary Structural Optimization) via external
  Python scripts calling CalculiX for FEA.
- **ToOptix** addon for topology optimization.
- **FEMbyGEN** for generative design with additive manufacturing constraints.
- **CADO** -- full toolchain from CAD input to optimized CAD output using
  OpenCASCADE for voxelization and ToPy for topology optimization.

**Key pattern:** *Loosely coupled: separate tools for constraint solving (FreeCAD
sketcher), FEA (CalculiX), and optimization (Python scripts).* Communication
is file-based or through FreeCAD's internal API.

### 2.5 OpenSCAD + Optimization Approaches

**Architecture:** OpenSCAD is script-based with no built-in constraint solver
or optimizer. Optimization is achieved through:

- External Python scripts that parametrically generate `.scad` files with
  varying parameters.
- Gradient-free optimization (Bayesian, evolutionary) wrapping the OpenSCAD
  rebuild-and-evaluate loop.
- AI/ML-based approaches using the textual representation for design space
  exploration.

**Key pattern:** *Pure black-box optimization over a procedural geometry
generator.* No constraint solver involvement at all -- geometry is defined
procedurally, not declaratively.

### 2.6 Spatial CDS (Constraint Design Solver)

**Architecture:** CDS, the commercial geometric constraint solver SDK from
Spatial (Dassault Systemes), provides three solving modes that illustrate the
optimization-constraint interplay:

1. **Update mode** -- standard constraint satisfaction (F(x) = 0).
2. **Interactive dragging** -- projects user displacement onto the constraint
   manifold's null space (Solverang's `drag.rs` implements exactly this).
3. **Simulation mode** -- drives geometries toward target values while
   satisfying constraints (a form of constrained optimization).

CDS's simulation mode is the closest commercial analog to what Solverang should
target: *minimize deviation from targets subject to geometric constraints*.

### 2.7 Summary of Integration Patterns

| Pattern | Examples | Coupling | Gradient Info |
|---------|----------|----------|---------------|
| Black-box wrapper | SolidWorks, OpenSCAD | Loose | None (FD/surrogate) |
| Separate engine + reconstruction | CATIA, FreeCAD+BESO | Loose | Within FEA only |
| Trust-region with parametric model | Siemens NX | Medium | Surrogate-based |
| Integrated solver modes | Spatial CDS | Tight | Jacobian available |
| Unified trait (proposed) | Solverang | Tight | Full Jacobian + AD |

---

## Part 3: Solver Interplay

### 3.1 SQP Inner Loop Uses QP Solver

Sequential Quadratic Programming (SQP) solves constrained optimization by
iterating:

```
At iterate x_k:
  1. Approximate objective as quadratic: q(d) = grad_f^T d + 0.5 d^T H_k d
  2. Linearize constraints: c(x_k) + J_c d = 0, g(x_k) + J_g d <= 0
  3. Solve QP subproblem for step d_k
  4. Update: x_{k+1} = x_k + alpha_k d_k  (with line search)
  5. Update H_k via BFGS or exact Hessian
```

The inner QP solver is itself a constraint solver (active-set or interior-point
method for quadratic programs). Two variants:

- **IQP (inequality QP):** The QP solver internally manages the active set.
  More robust, standard in SNOPT.
- **EQP (equality QP):** The outer SQP manages the active set, passing only
  active constraints to the inner solver. Faster when the active set is
  stable.

**Relevance to Solverang:** Solverang's Newton-Raphson solver already solves
linearized equation systems (the core of the EQP approach). Adding a merit
function and active-set management converts NR into an SQP outer loop.

### 3.2 Optimization as Constraint: Epigraph Reformulation

Any optimization objective can be reformulated as a constraint:

```
minimize f(x)  <==>  minimize t  subject to  f(x) <= t
```

This "epigraph" trick converts an optimization problem into a feasibility
problem with one extra variable. More practically:

```
minimize f(x) subject to c(x) = 0
   <==>
Find (x, lambda) such that:
   grad_f(x) + J_c(x)^T lambda = 0   (stationarity -- KKT)
   c(x) = 0                           (feasibility)
```

This KKT system is a square nonlinear system -- exactly what Solverang's
Newton-Raphson solver handles. The variables are `(x, lambda)` and the
equations are the KKT conditions.

**Relevance to Solverang:** For equality-constrained optimization, the KKT
approach converts optimization into root-finding, which Solverang already does.
This is the simplest integration pattern and should be implemented first.

### 3.3 Bilevel Optimization: Outer Optimization, Inner Constraint Satisfaction

The most common CAD optimization pattern is bilevel:

```
Outer: minimize f(p)
       subject to g(p) <= 0

Inner: given p, solve C(p, x) = 0 for x(p)
       (geometric constraint satisfaction)
```

The outer optimizer sees `f(p) = f_hat(p, x(p))` where `x(p)` is implicitly
defined by the constraint system. Gradients are obtained via the implicit
function theorem:

```
df/dp = partial_f/partial_p + (partial_f/partial_x)(dx/dp)

where dx/dp = -J_x^{-1} J_p
      (from differentiating C(p, x(p)) = 0)
```

`J_x` is the constraint Jacobian (which Solverang already computes and
factorizes during solving). `J_p` is the Jacobian with respect to design
parameters (a subset of the full Jacobian when parameters appear in
constraints).

**Relevance to Solverang:** This is the highest-leverage integration pattern.
The inner solver is already built. Computing `dx/dp` requires one extra linear
solve per design parameter (or one adjoint solve for all parameters if
`partial_f/partial_x` is available).

### 3.4 Sensitivity Analysis: Constraint Lagrange Multipliers Inform Optimization

When Solverang solves a constraint system, the Lagrange multipliers (or their
numerical equivalents from the LM/NR solve) carry optimization-relevant
information:

- **Multiplier magnitude** indicates how much the objective would improve per
  unit relaxation of that constraint. A large multiplier on a distance
  constraint means "this constraint is expensive -- consider relaxing it."
- **Shadow prices** for manufacturing: the multiplier on a tolerance constraint
  tells you the marginal cost of tightening that tolerance.
- **Redundancy detection** (which Solverang already does via
  `analyze_redundancy()`) identifies constraints that have zero multipliers --
  they don't affect the solution and could be removed.

Concretely, after solving `min ||F(x)||^2` via LM, the pseudo-multipliers
are available from:

```
lambda = -(J^T J)^{-1} J^T r
```

where `J` is the Jacobian at the solution and `r` is the residual. These
approximate the Lagrange multipliers of the underlying constraint system.

### 3.5 Warm-Starting Optimization from Constraint Solution

When the constraint system is solved incrementally (parameter drag, animation,
design iteration), previous solutions provide excellent warm starts:

- **Parameter continuation:** When a design parameter changes by a small
  amount `dp`, the new solution is approximately `x_new ~= x_old + (dx/dp) dp`.
  Solverang's incremental solve and solution cache already support this.
- **Active set prediction:** For inequality constraints, the active set at the
  previous solution is likely similar to the active set at the new solution.
  Warm-starting the active-set QP solver saves iterations.
- **Hessian reuse:** The BFGS approximation or the JTJ approximation from LM
  at the previous solution is a good initial Hessian for the new problem.

Solverang's `SolutionCache` and `ChangeTracker` provide the infrastructure for
warm-starting. The pipeline's incremental solve already skips unchanged
clusters.

> **REVIEW NOTE:** In bilevel optimization the inner constraint solver tolerance
> directly affects outer loop convergence. If the inner solve converges only
> loosely, the implicit gradient `dx/dp = -J_x^{-1} J_p` is inaccurate, which
> can prevent the outer optimizer from converging (the "inexact oracle" problem).
> Practical mitigation: tighten the inner solver tolerance progressively as the
> outer loop converges (e.g., inner_tol = max(outer_residual * 1e-2, abs_tol_min)).
> This interaction must be exposed in the `ParametricOptimizer` API.

---

## Part 4: Recommended Architecture for Solverang

### 4.1 Which Use Cases to Target First

**Tier 1 (natural fit, implement now):**

1. **Parametric design optimization** -- `minimize f(p) subject to
   sketch_constraints(p, x) = 0`. This is the most common CAD optimization
   pattern and maps directly to Solverang's architecture.

2. **Fit optimization** -- `minimize sum ||residuals||^2 subject to bounds`.
   Solverang's LM solver already minimizes sum-of-squares. Adding bound
   constraints (L-BFGS-B or bounded LM) completes this use case.

3. **Soft constraint optimization** -- `minimize sum w_i * c_i(x)^2 + f(x)`.
   The existing `Constraint::weight()` mechanism approximates this. Making it
   first-class (as proposed in `optimization.md`) is the cleanest path.

**Tier 2 (moderate extension):**

4. **Mechanism synthesis** -- constrained nonlinear least-squares with
   inequality constraints. Requires SQP or augmented Lagrangian.

5. **Assembly clearance optimization** -- assembly constraints as equalities,
   clearance as inequality objective. Natural extension of assembly solver.

6. **Tolerance optimization** -- sensitivity analysis from the constraint solver
   feeds directly into cost optimization.

**Tier 3 (requires external integration):**

7. **Shape optimization** -- needs FEA backend.
8. **Topology optimization** -- needs FEA + large-scale optimizer.
9. **Path planning** -- needs collision detection + trajectory discretization.

### 4.2 Most Natural Integration Pattern

Based on Solverang's architecture, the recommended integration follows the
**KKT-as-root-finding** pattern for equality-constrained problems and the
**bilevel implicit differentiation** pattern for general problems.

**Pattern A: KKT system as extended constraint system (equality-constrained)**

For problems with only equality constraints (the common sketch case):

```rust
// Conceptual -- not literal API
system.add_objective(MinimizeWeight { ... });
// Under the hood, this creates an extended system:
//   original constraints  C(x) = 0
//   stationarity         grad_f + J_C^T lambda = 0
// Solved as one big Newton system on (x, lambda)
```

This leverages the existing Newton-Raphson solver directly. The extended
KKT Newton system is:

```
[ H_L   J_C^T ] [dx     ]   [ -(grad_f + J_C^T lambda) ]
[ J_C   0     ] [dlambda] = [ -C(x)                     ]
```

Row 1 is the stationarity condition (gradient of Lagrangian w.r.t. x = 0).
Row 2 is the feasibility condition (constraint residual = 0).
`H_L` is the Hessian of the Lagrangian (approximated via BFGS or Gauss-Newton).

> **REVIEW NOTE:** This KKT system is singular when `J_C` is rank-deficient
> (redundant constraints). Solverang already provides `analyze_redundancy()` --
> call it before constructing the extended system and either remove redundant
> constraints, apply Tikhonov regularization to the (2,2) zero block, or fall
> back to the penalty/ALM formulation when rank deficiency is detected.

**Pattern B: Bilevel with implicit differentiation (parametric optimization)**

For design optimization where some parameters are "design variables" and
others are "state variables":

```rust
// Outer: BFGS/L-BFGS optimizer varies design params
// Inner: Solverang solves sketch constraints for state params
// Gradients: implicit differentiation using Jacobian from inner solve

fn evaluate_and_differentiate(design_params: &[f64]) -> (f64, Vec<f64>) {
    system.set_design_params(design_params);
    system.solve();  // inner constraint satisfaction
    let obj = compute_objective(&system);
    let grad = implicit_differentiate(&system, design_params);
    (obj, grad)
}
```

**Pattern C: Penalty/augmented Lagrangian (inequality constraints)**

For inequality constraints, use an augmented Lagrangian approach:

```
L_A(x, lambda, mu) = f(x) + sum lambda_i c_i(x)
                            + (mu/2) sum c_i(x)^2
                            + sum mu/2 * max(0, g_j(x) + lambda_j/mu)^2
```

Each sub-problem (fixed lambda, mu) is an unconstrained optimization that
Solverang can solve. Outer loop updates multipliers and penalty parameter.

### 4.3 Solverang's Unique Advantages

1. **Full Jacobian availability.** Unlike black-box CAD optimizers (SolidWorks,
   OpenSCAD), Solverang computes exact sparse Jacobians for all constraints.
   This enables gradient-based optimization with exact derivatives, eliminating
   the need for finite differences or surrogate models.

2. **Decomposition-aware optimization.** Solverang's graph decomposition into
   independent clusters means optimization can be parallelized: clusters that
   don't share design variables can be optimized independently. Few CAD-focused
   constraint solvers expose this structure to an external optimizer; most treat
   the constraint system as a monolithic black box from the optimizer's
   perspective.

   > **REVIEW NOTE:** This advantage degrades when an objective function couples
   > variables from multiple clusters (e.g., total mass, end-effector error). In
   > that case decomposition collapses to one cluster for optimization, though
   > constraint-only clusters that are unaffected by the objective can still be
   > solved independently first. See `00_synthesis.md` Section 3.2 for the
   > handling strategy.

3. **Incremental solve + warm-starting.** The `SolutionCache` and
   `ChangeTracker` infrastructure means that when the optimizer changes one
   parameter, only affected clusters are re-solved. This makes each optimization
   iteration fast.

4. **Implicit differentiation is inexpensive.** After solving `C(x) = 0`, the
   factorized Jacobian `J_x` is already available. Computing `dx/dp = -J_x^{-1}
   J_p` requires one back-substitution per design parameter -- much cheaper than
   re-solving the constraint system. `J_p` (the constraint Jacobian w.r.t. design
   parameters) must be evaluated separately; see the note below on how design
   parameters interact with the current `fix_param` mechanism.

   > **REVIEW NOTE:** The current codebase fixes design parameters via
   > `ParamStore::fix()`, which excludes them from the solver Jacobian columns.
   > Computing `J_p` therefore requires either: (a) temporarily unfixing design
   > parameters to include them as Jacobian columns, (b) a separate differentiation
   > pass that evaluates constraint partials with respect to fixed parameters, or
   > (c) treating design parameters as a distinct parameter class in Phase 3.
   > This gap must be addressed in the `DesignVariable` concept design.

5. **Pluggable pipeline.** The pipeline is customizable via
   `ConstraintSystem::set_pipeline()`, which replaces the default `SolvePipeline`.
   The `Reduce` phase could be extended to detect optimization opportunities (soft
   constraints, objectives) and the `SolveCluster` phase could be replaced with an
   optimization-aware solver.

6. **Inequality support already exists.** The `SlackVariableTransform` in
   `constraints/inequality.rs` converts inequality constraints to equalities
   via slack variables. This is the foundation for constrained optimization
   -- the slack variables become part of the KKT system.

7. **Compile-time AD potential.** Rust's type system enables compile-time
   automatic differentiation (via operator overloading with dual numbers or
   via Enzyme at the LLVM level). This would allow users to write objective
   functions in plain Rust and get exact gradients automatically, without
   manually implementing `gradient()`. The `ad-trait` crate (Li et al.,
   arXiv:2504.15976, 2025) provides a fast, flexible Rust-native AD library
   that could serve as the runtime AD backend while Enzyme provides the
   compile-time path.

### 4.4 Phased Rollout Plan

#### Phase 1: Foundation (Objective + KKT + ALM)

**Goal:** Enable `minimize f(x) subject to sketch_constraints(x) = 0`.

<!-- Decision: Phase 1 includes BOTH BFGS/L-BFGS (unconstrained) AND ALM (equality-constrained, reusing NR/LM inner loop). Both require only first derivatives. This document and 00_synthesis.md are now aligned on Phase 1 scope. -->

**Changes:**

1. Add `Objective` trait to the constraint system (named `Objective` to match
   `00_synthesis.md`, which also requires `fn id() -> ObjectiveId` and
   `fn name() -> &str`):
   ```rust
   pub trait Objective: Send + Sync {
       fn id(&self) -> ObjectiveId;
       fn name(&self) -> &str;
       fn param_ids(&self) -> &[ParamId];
       fn value(&self, store: &ParamStore) -> f64;
       fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;
   }
   ```

2. Add `ConstraintSystem::set_objective()` and `ConstraintSystem::optimize()`
   (names per `00_synthesis.md`; `set_objective` replaces a single active
   objective rather than accumulating multiple).

3. Implement the KKT-as-root-finding approach: extend the constraint system
   with stationarity conditions and solve via existing Newton-Raphson. See the
   corrected block layout in Section 4.2 Pattern A.

4. Implement the ALM outer loop (reuses existing LM as inner solver).

5. Implement BFGS/L-BFGS solver for unconstrained problems (gradient-only,
   no Hessian needed).

6. Add `ParamStore::set_bounds()` for parameter bounds.

**Deliverables:**
- `Objective` trait + `ObjectiveId`
- `MultiplierStore` for dual variables
- BFGS/L-BFGS solver (unconstrained, gradient-only)
- ALM solver (equality-constrained, outer loop + existing LM inner loop)
- KKT system builder
- Parameter bounds
- Tests: Rosenbrock unconstrained (BFGS), Rosenbrock constrained to circle (ALM),
  equality-constrained sketch optimization, Hock-Schittkowski #1

**Estimated scope:** ~1500 lines of new code.

#### Phase 2: Bounds + L-BFGS-B + ConstraintKind

**Goal:** Enable inequality constraints and bound-constrained optimization.

<!-- Decision: Phase 2 = Bounds + L-BFGS-B. SQP is in Phase 3 because it requires a QP subproblem solver. This document and 00_synthesis.md are aligned. -->

**Changes:**

1. Add `ConstraintKind` enum (`Equality`, `Inequality`, `Penalty`) as proposed
   in `00_synthesis.md` (Section 1 trait hierarchy), replacing the existing
   `is_soft()` / `weight()` ad-hoc mechanism in `constraint/mod.rs`.

2. Add variable bounds to `ParamStore` (lower/upper per parameter).

3. Implement L-BFGS-B solver for large-scale bound-constrained problems.

4. Implement BFGS for small-to-medium unconstrained problems.

5. Extend the pipeline's `SolveCluster` phase to handle objectives: when a
   cluster has an attached objective, dispatch to the appropriate solver.

**Deliverables:**
- `ConstraintKind` enum
- Variable bounds in `ParamStore`
- L-BFGS-B solver
- BFGS solver
- Pipeline integration for objective-bearing clusters
- Tests: box-constrained Rosenbrock, bounded curve fitting

**Estimated scope:** ~1500 lines.

#### Phase 3: SQP + Implicit Differentiation + Bilevel

**Goal:** Enable general inequality constraints (SQP) and parametric design
optimization (outer optimizer + inner constraint solver).

<!-- Decision: SQP is in Phase 3 because it requires a QP subproblem solver. Both this document and 00_synthesis.md agree. -->

**Changes:**

1. Implement SQP solver that reuses the existing Newton-Raphson as the EQP
   inner solver. The SQP outer loop manages the active set and merit function.
   Depends on a Rust QP solver (Clarabel.rs or OSQP).

2. Add `DesignVariable` concept: mark certain parameters as design variables
   that the outer optimizer controls, while the inner solver determines the
   remaining state variables. Clarify how this relates to the existing
   `fix_param()` mechanism (design variables need to be excluded from the
   inner solve but included in Jacobian differentiation w.r.t. `p`).

3. Implement implicit differentiation: after inner solve, compute
   `dx/dp = -J_x^{-1} J_p` using the already-factorized Jacobian.

4. Implement `ParametricOptimizer` that wraps the constraint system:
   ```rust
   pub struct ParametricOptimizer {
       system: ConstraintSystem,
       objective: Box<dyn Objective>,
       design_params: Vec<ParamId>,
       optimizer: Box<dyn Optimizer>,  // BFGS, L-BFGS, SQP, etc.
   }
   ```

5. Exploit decomposition: only re-solve clusters affected by changed design
   variables, using the `ChangeTracker`.

**Deliverables:**
- SQP solver (active-set variant, reusing NR as EQP inner solver)
- Implicit differentiation module
- `ParametricOptimizer` struct
- Design variable concept in `ParamStore`
- Tests: mechanism synthesis, parametric bracket optimization, linkage
  dimension optimization, Hock-Schittkowski suite

**Estimated scope:** ~2500 lines.

#### Phase 4: Validation + Advanced Features

**Goal:** Production readiness and advanced use cases.

**Changes:**

1. Standard optimization test suite: Hock-Schittkowski problems #1-#116,
   constrained Rosenbrock variants, mechanism synthesis benchmarks, COPS
   benchmark subset. "30+ problems" is sufficient for initial validation;
   full CUTEst coverage is a long-term goal. Specify pass/fail criteria
   (e.g., convergence to within 1% of known optimal within 500 major iterations)
   before claiming production readiness.

2. Augmented Lagrangian solver as an alternative to SQP for problems with many
   inequality constraints and when exact Hessians are unavailable.

   <!-- Decision: ALM is in Phase 1 (along with BFGS), not Phase 4. Phase 1 delivers both unconstrained (BFGS) and equality-constrained (ALM) optimization. This Phase 4 item is superseded. -->

3. Multi-objective support via weighted sum or epsilon-constraint method.

4. Sensitivity report: after optimization, report which constraints are active,
   their multipliers, and the marginal cost of relaxation. The `MultiplierStore`
   (Phase 1 deliverable in `00_synthesis.md`) provides the dual variables needed.

5. Integration with compile-time AD (Enzyme or dual-number based) for automatic
   gradient computation of user-defined objectives. The `ad-trait` crate
   (arXiv:2504.15976, 2025) is a Rust-native AD option worth evaluating alongside
   Enzyme.

6. Investigate convergence tolerance interaction between inner and outer loops in
   bilevel optimization: loose inner tolerance produces inexact implicit gradients
   for the outer loop. Strategies include tightening inner tolerance progressively
   or using inexact oracle variants.

**Deliverables:**
- Optimization test suite (Hock-Schittkowski #1-#30 + mechanism benchmarks as
  initial target; expand toward 100+ for production claim)
- Augmented Lagrangian solver
- Sensitivity reporting
- AD integration prototype

**Estimated scope:** ~3000 lines.

---

## Appendix: Solver Algorithm Selection Matrix

Given a problem classification, which solver to use:

| Problem Type | Algorithm | Solverang Component |
|---|---|---|
| F(x) = 0 (square) | Newton-Raphson | `solver::Solver` (exists) |
| min \|\|F(x)\|\|^2 | Levenberg-Marquardt | `solver::LMSolver` (exists) |
| min f(x) s.t. c(x)=0 | KKT + Newton-Raphson, ALM | Phase 1 (reuse existing NR/LM) |
| min f(x) unconstrained | BFGS | Phase 2 (new) |
| min f(x) + bounds | L-BFGS-B | Phase 2 (new) |
| min \|\|F(x)\|\|^2 + bounds | Bounded LM | Phase 2 (extend existing LM) |
| Large-scale + bounds | L-BFGS-B or MMA | Phase 2 (new) |
| min f(x) s.t. c(x)=0, g(x)<=0 | SQP | Phase 3 (new, reuses NR as EQP inner) |
| min f(p), inner C(p,x)=0 | Bilevel + implicit diff | Phase 3 (new, reuses existing solver) |
| Many inequalities, no exact Hessian | ALM | Phase 4 (alternative to SQP) |

---

## References

- Bettig & Hoffmann, "Geometric Constraint Solving in Parametric CAD," ASME J. Computing and Information Science in Engineering, 2011.
- Gill & Wong, "Sequential Quadratic Programming Methods," UCSD CCOM, 2012.
- SNOPT: Gill, Murray, Saunders, "SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization," SIAM Review 47(1), 2005.
- Wang et al., "Fitting B-spline Curves to Point Clouds by Curvature-Based Squared Distance Minimization," ACM Trans. Graphics, 2006.
- Sleesongsom & Bureerat, "Optimal Synthesis of Four-Bar Linkage Path Generation through Evolutionary Computation," Computational Intelligence and Neuroscience, 2018.
- Spatial CDS documentation, https://www.spatial.com/solutions/3d-modeling/constraint-design-solver
- SolidWorks Optimization Design Study, https://help.solidworks.com/2022/English/SolidWorks/sldworks/c_Using_Optimization_Module.htm
- Li et al., "ad-trait: A Fast and Flexible Automatic Differentiation Library in Rust," arXiv:2504.15976, 2025.
- Comet-FEniCS, "Topology optimization using the SIMP method," https://comet-fenics.readthedocs.io/
