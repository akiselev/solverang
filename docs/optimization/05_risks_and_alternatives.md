# Risks, Limitations, and Alternatives for Optimization Extension

Skeptic/Red Team analysis of extending Solverang's constraint solver to support optimization.

## Table of Contents

1. [Runtime AD vs Compile-Time Symbolic Differentiation](#1-runtime-ad-vs-compile-time-symbolic-differentiation)
2. [Non-Smooth Objectives](#2-non-smooth-objectives)
3. [Global Optimization](#3-global-optimization)
4. [Scaling Limits](#4-scaling-limits)
5. [Degenerate Cases](#5-degenerate-cases)
6. [Fundamental Limits of the Macro Approach](#6-fundamental-limits-of-the-macro-approach)
7. [Alternative Architectures](#7-alternative-architectures)
8. [Two-Year Architecture Evolution Prediction](#8-two-year-architecture-evolution-prediction)
9. [Recommendations](#9-recommendations)

---

## 1. Runtime AD vs Compile-Time Symbolic Differentiation

### What Solverang Does Today

The `#[auto_jacobian]` macro performs compile-time symbolic differentiation. It:
1. Parses a Rust expression into an `Expr` AST (see `crates/macros/src/expr.rs`)
2. Calls `Expr::differentiate()` to symbolically compute partial derivatives
3. Applies `Expr::simplify()` to reduce the result
4. Generates Rust code via `Expr::to_tokens()` that evaluates the derivative at runtime
5. Also generates JIT opcode streams via `Expr::to_opcode_tokens()` for Cranelift compilation

This is neither pure compile-time symbolic math nor runtime AD. It is **compile-time code generation of derivative evaluation code**. The distinction matters.

### Comparison Table

| Criterion | Solverang Macro (Symbolic Codegen) | Forward-Mode AD (Dual Numbers) | Reverse-Mode AD (Tape-Based) |
|---|---|---|---|
| **Differentiation timing** | Compile time | Runtime | Runtime |
| **Code generated** | Explicit derivative expressions | Generic over number type | Records operations, replays backward |
| **Control flow** | Cannot handle | Natural support | Natural support |
| **Expression size** | O(n^2) Jacobian, O(n^3) Hessian for product chains; exponential only for deep recursive compositions (unusual in CAD) | O(n) per directional derivative | O(1) backward passes for full gradient |
| **Second derivatives (Hessian)** | Requires differentiating derivatives (AST^2 blowup) | Nested duals, O(n^2) cost | Mixed forward-over-reverse, O(n) per Hessian-vector product |
| **Memory at compile time** | AST size can explode for deep expressions | Zero | Zero |
| **Runtime overhead** | Zero dispatch overhead; static code | Dual number arithmetic overhead (~2x per op) | Tape allocation, indirection, cache misses |
| **Sparsity exploitation** | Automatic (skips zero derivatives) | Requires seeding strategy | Requires graph coloring for sparse Jacobians |
| **Composability** | One residual at a time, no cross-residual sharing | Composable with any Rust code | Composable with any Rust code |
| **Debugging** | Readable generated code | Standard debugger works | Tape inspection required |
| **User effort** | Write expression in restricted subset | Change numeric type to Dual | Wrap code in tape context |

### Concrete Cases Where Macros Fail

**Case 1: Data-Dependent Branching**

A constraint whose behavior depends on the current geometry:

```rust
// This CANNOT be handled by #[auto_jacobian]
#[residual]
fn residual(&self, x: &[f64]) -> f64 {
    let d = ((x[0] - x[2]).powi(2) + (x[1] - x[3]).powi(2)).sqrt();
    if d < self.threshold {
        // Near-field: exact distance
        d - self.target
    } else {
        // Far-field: approximate
        (x[0] - x[2]).abs() + (x[1] - x[3]).abs() - self.target
    }
}
```

The macro rejects this at the expression-parsing stage. The `if/else` construct
is a valid Rust tail expression and passes the initial body check, but
`parse_expr` in `parse.rs` hits its catch-all branch for `SynExpr::If` and
returns: "Unsupported expression type: If { ... }". No `if`, `match`, `for`,
`while`, or `loop` expression is in the recognized set.
A runtime AD system handles this naturally by tracing whichever branch executes.

**Case 2: Iterative Sub-Computations**

```rust
// Newton iteration to find closest point on implicit curve
#[residual]
fn residual(&self, x: &[f64]) -> f64 {
    let mut t = self.initial_t;
    for _ in 0..10 {
        let f = curve_eval(t, x);
        let df = curve_deriv(t, x);
        t -= f / df;
    }
    point_to_curve_distance(x, t) - self.target
}
```

The macro cannot parse loops. Runtime AD differentiates through them by unrolling
the computation trace. This pattern arises in projection constraints,
closest-point problems, and any constraint involving implicit geometry.

**Case 3: Expression Size Explosion**

The current `differentiate()` method clones sub-expressions. Consider a chain of
multiplications:

```rust
// f = x[0] * x[1] * x[2] * ... * x[n-1]
```

The derivative with respect to `x[k]` via the product rule is the product of all
other variables. But the AST representation grows as O(n) per variable, and there
are n variables, so the total generated Jacobian code is O(n^2). For a 100-variable
product, this generates ~10,000 terms. The `simplify()` pass is single-pass and
does not perform common subexpression elimination (CSE), so identical
sub-expressions are recomputed.

With Hessians the situation is worse: differentiating the already-expanded
derivative expressions gives O(n^3) AST nodes for the same product example.

**Case 4: Dynamic Variable Counts**

```rust
// Number of variables depends on runtime configuration
fn residual(&self, x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..self.n_points {
        sum += (x[2*i] - self.targets[i].x).powi(2)
             + (x[2*i+1] - self.targets[i].y).powi(2);
    }
    sum.sqrt() - self.total_target
}
```

The macro needs to see concrete index expressions (`x[0]`, `x[1]`, etc.) at
compile time. It cannot handle `x[2*i]` where `i` is a loop variable. Runtime AD
handles this with no special effort.

**Case 5: External Library Calls**

```rust
#[residual]
fn residual(&self, x: &[f64]) -> f64 {
    let curve = nurbs::BSplineCurve::evaluate(x);
    curve.curvature_at(0.5) - self.target_curvature
}
```

The macro cannot see into opaque function calls. It rejects anything that is not
in its recognized set of operations (`sqrt`, `sin`, `cos`, `tan`, `atan2`, `abs`,
`ln`, `exp`, `powi`, `powf`). Tape-based AD can record through foreign code if it
is written generically over the number type.

### Verdict

The macro approach is excellent for the constraint-solver's current sweet spot:
moderate-complexity, analytically-expressible residuals with static structure.
For optimization objectives that involve branching, iteration, or dynamic
structure, a runtime AD fallback is not optional -- it is necessary.

---

## 2. Non-Smooth Objectives

### What the Architecture Handles

The `Expr` AST includes `Abs`, and `differentiate()` computes its derivative as
`sign(e) * de` (specifically, `e / |e| * de`). This is correct almost
everywhere but **undefined at e = 0**. The generated code will produce `0/0 = NaN`
at that point.

The current solvers (Newton-Raphson and Levenberg-Marquardt) both check for
non-finite values and bail out:
- Newton-Raphson returns `SolveError::NonFiniteResiduals` or
  `SolveError::NonFiniteJacobian`
- LM adapter returns `None` from `residuals()` or `jacobian()`, triggering
  `TerminationReason::User`

So non-smooth objectives **fail with a cryptic error in practice**: the solver
stops with `SolveError::NonFiniteJacobian` (an explicit named error), but
without causal context. The user sees "non-finite Jacobian" rather than "your
objective is non-differentiable at this point because Abs evaluated at zero."

### Catalog of Non-Smooth Scenarios

| Objective Type | Example | Current Behavior | What Should Happen |
|---|---|---|---|
| **L1 norm** | `\|x[0]\| + \|x[1]\|` | NaN at zero crossing | Subgradient method or smooth approximation |
| **Minimax** | `max(f1(x), f2(x))` | Cannot express in macro | Epigraph reformulation or bundle method |
| **Indicator function** | `if g(x) < 0 { INFINITY } else { 0 }` | Macro rejects if/else | Penalty method or interior point |
| **Piecewise linear** | `max(0, g(x))` (hinge) | Cannot express | Smooth approximation: `log(1 + exp(k*g))` |
| **Euclidean norm** | `sqrt(dx^2 + dy^2)` | NaN gradient at origin | Handled by regularization: `sqrt(dx^2 + dy^2 + eps)` |
| **Rank function** | `rank(J(x))` | Not expressible | Entirely out of scope |

### What Is In Scope

1. **Smooth approximations of non-smooth objectives.** The user writes
   `softmax` or `log-sum-exp` instead of `max`. The macro handles these fine.
   This should be documented as the recommended approach.

2. **Huber loss** can be expressed as a piecewise function, but the macro
   cannot handle it directly. A pre-built `HuberResidual` wrapper that
   switches between quadratic and linear at runtime (not through the macro)
   is feasible.

3. **Regularized norms.** Instead of `sqrt(dx^2 + dy^2)`, use
   `sqrt(dx^2 + dy^2 + 1e-12)`. This is already a common pattern in CAD
   constraints (the `ClearanceConstraint.gradient_2d` returns zeros when
   distance < 1e-10, which is the same idea).

### What Is NOT In Scope

1. **L1-regularized optimization** (LASSO) -- requires proximal operators or
   coordinate descent, neither of which fits the residual/Jacobian framework.

2. **Mixed-integer optimization** -- discrete variables are fundamentally
   outside the continuous solver's reach.

3. **Semidefinite programming** -- the `Problem` trait has no concept of
   matrix-valued constraints.

4. **Complementarity constraints** (`g(x) >= 0, h(x) >= 0, g(x)*h(x) = 0`)
   -- these arise in contact mechanics and are notoriously hard. The slack
   variable transform does not handle them.

### Risk

The main risk is that users will write non-smooth objectives, get cryptic
`NonFiniteJacobian` errors, and blame the solver. The mitigation is:
- Better error messages that detect and report non-differentiable points
- A library of smooth approximation utilities
- Documentation that explicitly states the smoothness requirement

---

## 3. Global Optimization

### Is This Out of Scope?

For a CAD constraint solver, global optimization is **partially in scope**.
Consider: a distance constraint `dist(A, B) = 5` with two points initially at
the origin. There are infinitely many solutions (a circle of radius 5). The
solver finds one based on the initial guess. But if we add an optimization
objective "minimize total wire length," there may be a unique global optimum
that a local solver misses entirely.

Real CAD scenarios where global optimization matters:
- **Packing problems**: place N components in minimum area
- **Topology optimization**: which configuration of constraints minimizes
  objective?
- **Multi-stable mechanisms**: find all equilibrium positions

### Should the Architecture Accommodate It?

**Yes, but only at the orchestration level, not the solver level.** The
architecture should support:

1. **Multi-start**: Run the local solver from multiple initial points, keep
   the best. This requires no solver changes -- just a loop around
   `solver.solve()`. The `initial_point(factor)` method already hints at this
   pattern (factors of 1.0, 10.0, 100.0 are MINPACK convention).

2. **Basin-hopping**: Local optimization + random perturbation. Again, just
   orchestration.

3. **Interval arithmetic for bounds**: Could verify that a local optimum is
   global within a region. This would require a parallel interval-arithmetic
   evaluation of the objective, which the macro could potentially generate
   (same AST, different codegen backend).

What the architecture should NOT try to do:
- Genetic algorithms or particle swarm -- these need only function
  evaluations, not Jacobians. They can call `problem.residuals()` without
  any integration into the solver framework.
- Branch-and-bound for mixed-integer -- fundamentally different paradigm.

### Risk

The risk is designing the optimization extension so tightly coupled to
gradient-based methods that adding multi-start or stochastic methods later
requires rearchitecting. The mitigation is keeping the `Problem` trait as the
universal interface and adding optimization as a layer above, not a
modification of, the existing solver infrastructure.

---

## 4. Scaling Limits

### Problem Size vs Approach Feasibility

| Problem Size (variables) | Macro Jacobian | Dense NR | Sparse NR | LM (dense) | LM (sparse) | JIT Solver |
|---|---|---|---|---|---|---|
| **1-10** | Excellent | Excellent | Overkill | Excellent | Overkill | Unnecessary |
| **10-100** | Good | Good | Good | Good | Good | Marginal benefit |
| **100-1,000** | Compile time concern | O(n^3) per step | Good if sparse | O(mn^2) | Good | Significant benefit |
| **1,000-10,000** | Compile time PROBLEM | Infeasible | Feasible if sparse | Infeasible | Feasible | Necessary |
| **10,000-100,000** | Infeasible | Infeasible | Feasible with structure | Infeasible | Feasible with structure | Critical |
| **100,000+** | Infeasible | Infeasible | Requires specialized methods | Infeasible | Requires specialized methods | Helpful but not sufficient |

### Specific Scaling Bottlenecks

**Macro Compile Time**: The symbolic differentiation runs at compile time
inside `rustc`. Each variable in a residual adds one `differentiate()` call
that traverses the entire AST. For a residual touching k variables with AST
depth d, the differentiation cost is O(k * 2^d) in the worst case (binary
tree of multiplications). The simplification pass is O(n) where n is the
output AST size, but it does not perform CSE.

Concrete numbers: A residual `(x[0]-x[2])^2 + (x[1]-x[3])^2 - d^2` produces
~50 AST nodes. Each of its 4 partial derivatives produces ~30 nodes after
simplification. This is fine. But a sum-of-squares objective over 100 points
(`sum_{i=0}^{99} (x[2i] - t_i)^2 + (x[2i+1] - t_i)^2`) touches 200
variables and produces 200 derivative expressions -- each trivial, but the
total generated code is ~6,000 lines. This starts to hurt compile times.

The real problem is that the macro processes **one residual at a time** (see
`find_residual_method` which returns the first `#[residual]`). For
optimization with a single scalar objective over many variables, this means
the entire objective must be one expression. That expression may be very
large.

**Jacobian Assembly**: The current `Problem::jacobian()` returns
`Vec<(usize, usize, f64)>` -- a heap-allocated vector of triplets. For a
sparse Jacobian with nnz entries, this allocates nnz * 24 bytes per
evaluation. At 10,000 iterations on a problem with 50,000 non-zeros, that is
12 GB of allocation churn. The JIT solver presumably avoids this.

**Linear Algebra**: Newton-Raphson uses `nalgebra::DMatrix` (dense). The
`solve_linear` method tries LU first, then falls back to SVD. Both are O(n^3)
for an n x n system. For n > 1,000, this dominates runtime.
Levenberg-Marquardt uses the `levenberg-marquardt` crate which also appears
to be dense. Neither solver has sparse linear algebra support in the hot path
today, despite the `sparse_solver.rs` file existing.

**Hessian for Optimization**: Newton's method for optimization requires the
Hessian (n x n matrix of second derivatives). With the macro approach, this
means differentiating each Jacobian entry -- squaring the AST explosion
problem. For a problem with n = 100 variables, the Hessian has 10,000
entries, each requiring a symbolic derivative of an already-differentiated
expression. This is likely infeasible for the macro approach beyond n ~ 50.

### Recommendation

For the optimization extension:
- n < 50: Macro-generated Hessians are feasible. Use Newton's method.
- 50 < n < 500: Use BFGS (approximate Hessian from gradient history).
  Requires only first derivatives (Jacobian), which the macro handles well.
- 500 < n < 10,000: Use L-BFGS (limited-memory BFGS). Still only needs
  gradients. Sparse structure is critical.
- n > 10,000: Consider CG (conjugate gradient) methods that need only
  Hessian-vector products, which can be computed via forward-mode AD (dual
  numbers) in O(n) without forming the full Hessian.

---

## 5. Degenerate Cases

### Catalog

| Degenerate Case | Example | Current Handling | Risk Level | Proposed Handling |
|---|---|---|---|---|
| **Rank-deficient Hessian** | Minimize `x^2 + y^2` subject to `x + y = 1` at the minimum: Hessian of Lagrangian is rank-deficient in constrained directions | NR uses SVD fallback with `svd_tolerance = 1e-10` | Medium | Projected Newton method; regularize Hessian with `max(H, epsilon * I)` |
| **Redundant inequalities** | `x >= 0` AND `x >= -1` (second is always implied) | SlackVariableTransform adds a slack for each, making the system larger than needed | Low | Preprocessing pass to detect and remove redundant bounds |
| **Unbounded problem** | Minimize `x` with no constraints | Solver iterates to infinity, returns NotConverged | High | Detect unbounded descent direction; return explicit error |
| **Flat objective** | Minimize `0*x` (zero gradient everywhere) | Zero Jacobian triggers `SingularJacobian` error | Medium | Detect zero gradient and report "objective has no preferred direction" |
| **Saddle points** | Minimize `x^2 - y^2` | Newton converges to saddle (0,0) | High | Check second-order conditions; negative curvature detection |
| **Infeasible constraints** | `x = 1` AND `x = 2` | Solver oscillates, returns NotConverged with large residual | Medium | Redundancy analysis already detects conflicts; use `DiagnosticIssue::ConflictingConstraints` |
| **Degenerate constraint qualification** | All constraint gradients parallel at solution | LICQ violated; Lagrange multipliers not unique | High | Detect via SVD of constraint Jacobian; warn user |
| **Near-zero slack** | Slack variable s -> 0 means inequality is active; Jacobian entry -2s -> 0 | Jacobian becomes singular at active inequality boundary | Critical | Switch to active-set method when slack < threshold; or use interior-point with log-barrier |
| **Cycling in active set** | Inequality constraints alternately active/inactive | Solver oscillates without converging | Medium | Anti-cycling rules (Bland's rule analog); or use interior-point instead of active-set |

### The Slack Variable Problem in Detail

The current `SlackVariableTransform` converts `g(x) >= 0` to `g(x) - s^2 = 0`.
This has a critical flaw for optimization: when an inequality becomes active
(the optimal point is on the boundary), `s -> 0` and the Jacobian entry for
the slack variable is `-2s -> 0`. This makes the Jacobian rank-deficient
precisely at the solution.

Concretely:

```
Extended Jacobian at s = 0:
[ ... dg/dx_j ... | -2*0 ]  =  [ ... dg/dx_j ... | 0 ]
```

The zero column means the solver cannot determine how to move the slack
variable. LM will regularize this away (the damping term saves it).
Newton-Raphson tries LU decomposition first; if the matrix is square, LU
fails on the zero column and NR falls back to SVD pseudoinverse
(`newton_raphson.rs` lines 183-190). SVD does NOT fail on a rank-deficient
matrix -- it returns a minimum-norm step that zeros out the degenerate
direction. This means NR silently continues with a step that ignores the
active constraint boundary, rather than reporting `SingularJacobian`. The
behavior is worse than a clean error: NR may iterate indefinitely without
making progress on the slack variable, appearing to converge on the original
variables while the slack component stagnates.

**Alternative**: Use log-barrier or interior-point formulation instead of
slack variables for optimization:
- Log-barrier: minimize `f(x) - mu * sum(ln(g_i(x)))` for decreasing mu
- Interior-point: directly solve the KKT system with proper scaling

Both avoid the rank-deficiency at active constraints.

### The Saddle Point Problem

For unconstrained optimization via Newton's method, the update direction is
`d = -H^{-1} * grad`. If the Hessian `H` has negative eigenvalues
(indicating a saddle point or local maximum), Newton converges *toward* the
saddle rather than away from it. The current solver infrastructure has no
mechanism to detect or handle this because it was designed for root-finding
(find F(x) = 0), not optimization (find min f(x)).

Adding optimization requires adding:
1. Hessian eigenvalue checks (at least for the minimum eigenvalue)
2. Modified Newton directions when negative curvature is detected
   (e.g., trust-region methods, or adding `tau * I` to make H positive definite)

### LM Termination Mapping Is Fragile

The `LMSolver::convert_termination` method (`levenberg_marquardt.rs` lines
172-188) maps `TerminationReason::User(msg)` to error types by string
matching:

```rust
if msg.contains("residual") { ... NonFiniteResiduals }
else if msg.contains("jacobian") { ... NonFiniteJacobian }
else { NotConverged }
```

The same pattern repeats for `TerminationReason::Numerical` and
`TerminationReason::WrongDimensions`. If the upstream `levenberg-marquardt`
crate changes its message text in a minor version update, these branches
silently misclassify the termination. An optimization outer loop (ALM, SQP)
relying on the inner LM solver's error type to distinguish "non-finite
Jacobian" from "not converged" could take the wrong recovery path with no
visible failure.

**Mitigation**: Replace string matching with a flag set by the adapter.
`LMProblemAdapter` already knows whether it returned `None` from `residuals()`
or `jacobian()`. Store that information as an enum field and read it after
`lm.minimize()` returns, rather than parsing the crate's message string.

### `NoImprovementPossible` May Report False Convergence

`levenberg_marquardt.rs` lines 212-220: when the LM step size drops below
machine epsilon, the crate returns `TerminationReason::NoImprovementPossible`.
The current adapter reports this as `Converged` if
`residual_norm < self.config.ftol.sqrt()`. With the default `ftol = 1e-8`,
this reports convergence when `residual_norm < 1e-4` -- four orders of
magnitude coarser than the configured tolerance.

For optimization inner loops, a stalled LM solver reporting `Converged` causes
the outer multiplier update to proceed on an inaccurate inner solution,
degrading convergence of the outer algorithm.

**Mitigation**: Map `NoImprovementPossible` to `NotConverged` unconditionally,
or use a separate `Stalled` variant. Let the outer algorithm decide whether
stalling is acceptable.

---

## 6. Fundamental Limits of the Macro Approach

### What the Macro Cannot Do, By Design

1. **No control flow.** The parser in `parse.rs` explicitly rejects everything
   that is not an expression: no `if`, no `match`, no `for`, no `while`, no
   `loop`, no `return`. The `extract_return_expr` function requires the
   function body to end with a single expression.

2. **No closures or function calls.** The `parse_function_call` and
   `parse_method_call` functions have a hard-coded whitelist of recognized
   operations. Any call not in the list produces a compile error.

3. **No variable-exponent powers.** `Pow(base, exp)` requires `exp` to be
   a compile-time constant (`parse_constant_expr` rejects non-literal
   exponents). This means `x.powf(y)` where `y` is a variable is not
   supported.

4. **No shared sub-expressions.** Let bindings are expanded by
   `expand_bindings`, which substitutes each binding into the final
   expression. If the same sub-expression appears multiple times, it is
   duplicated in the AST and differentiated independently. There is no DAG
   representation, only a tree.

5. **No multiple residuals with shared structure.** The macro generates
   one `jacobian_entries` method for one `#[residual]` function. If an
   optimization objective shares sub-expressions with constraint residuals,
   there is no way to amortize the computation.

6. **No higher-order derivatives without AST explosion.** The
   `differentiate` method returns a new `Expr`. Differentiating that result
   gives the second derivative, but the expression size roughly doubles
   each time (product rule applied to already-expanded products). There is
   no provision for mixed partials or Hessian-vector products.

7. **No dynamic variable count.** The macro can handle struct-field-indexed
   array accesses (`x[self.idx_a]`): these are recorded as variable
   references with a runtime-evaluated index and generate valid code like
   `x[self.idx_a]`. A single generic constraint struct can therefore serve
   multiple entity pairs. What is impossible is a loop-variable index
   (`x[2*i]` where `i` is a loop variable), because the macro cannot
   determine at compile time which state-vector slots are involved. This
   means dynamic variable counts (number of points determined at runtime)
   are not supported, but parameterized constraints over a fixed number of
   entity slots work fine.

### The CSE Problem

Consider a distance residual:
```
f(x) = sqrt((x[2]-x[0])^2 + (x[3]-x[1])^2) - self.target
```

The derivative w.r.t. x[0] is:
```
df/dx[0] = -(x[2]-x[0]) / sqrt((x[2]-x[0])^2 + (x[3]-x[1])^2)
```

The derivative w.r.t. x[1] is:
```
df/dx[1] = -(x[3]-x[1]) / sqrt((x[2]-x[0])^2 + (x[3]-x[1])^2)
```

Both share the `sqrt(...)` denominator. The generated code computes it
twice. For 4 variables, this is 4x redundancy on the most expensive
sub-expression. For n variables sharing a common sub-expression, the
redundancy is n-fold.

The `simplify()` pass handles algebraic identities (0+x = x, 1*x = x, etc.)
but does not perform common sub-expression elimination. Adding CSE to the
macro would require:
1. Converting the tree AST to a DAG
2. Hash-consing or structural equality checking
3. Emitting let-bindings in the generated code for shared nodes

This is doable but non-trivial, and it changes the codegen from a simple
recursive `to_tokens()` to a topological-sort-based code emitter.

**The Abs case is guaranteed double-evaluation.** The differentiation rule for
`Abs` in `expr.rs` always clones `e` twice: once as the numerator of the sign
function `e / |e|` and once as the argument of the outer `Abs`. At runtime,
`e` is evaluated twice even if it appears only once in the source expression.
For an L1 norm `|f1(x)| + |f2(x)| + ...`, every term doubles the cost of
evaluating its sub-expression in the Jacobian. Unlike the distance example
where CSE is optional, the Abs double-evaluation is structural and requires
CSE to fix.

### The Hessian Wall

For optimization, we need the Hessian (matrix of second partial derivatives)
or at least Hessian-vector products. The current architecture offers three
paths, all problematic:

1. **Symbolic second differentiation via macro**: Differentiate the Jacobian
   expressions. AST size explosion makes this impractical for n > ~30
   variables.

2. **Finite-difference Hessian**: Perturb each variable, recompute
   Jacobian, divide by perturbation. Costs O(n) Jacobian evaluations per
   Hessian. Numerically noisy. Requires n+1 Jacobian evaluations per
   Newton step.

3. **Runtime AD for Hessian only**: Use dual numbers or a tape to
   differentiate the macro-generated Jacobian code. This is a hybrid
   approach: the macro generates exact first derivatives, and runtime AD
   computes second derivatives from those. Practical but requires a second
   AD system.

The most pragmatic path is (3): use macro-generated Jacobians as the source
of truth, and apply forward-mode AD (dual numbers) to them for
Hessian-vector products when needed. This avoids the AST explosion while
preserving the macro's benefits for first derivatives.

---

## 7. Alternative Architectures

### Alternative A: Pure Runtime AD (Dual Numbers)

**Approach**: Make the `Problem` trait generic over the number type
`T: Float`. Constraints implement `residuals<T>(&self, x: &[T]) -> Vec<T>`.
For evaluation, call with `T = f64`. For Jacobian, call with
`T = DualNumber` (one call per variable for forward mode, or one call total
for reverse mode).

**Advantages**:
- Supports all control flow, loops, function calls
- No compile-time AST processing
- Hessian-vector products come naturally (nested duals)
- Users write normal Rust code

**Disadvantages**:
- 2-3x slower per evaluation (dual number arithmetic overhead)
- Requires all math to be generic over `T`, including external libraries
- Reverse-mode AD in Rust requires either a tape (runtime overhead) or
  compiler plugin (does not exist in stable Rust)
- Loses the zero-overhead guarantee of the current approach

**Why it might be better**: For optimization specifically, the cost of
computing Hessians via symbolic macros is prohibitive. Dual numbers compute
exact Hessian-vector products in O(n) time with O(1) implementation effort.
The 2x slowdown per evaluation is irrelevant if it avoids O(n^2) symbolic
differentiation.

### Alternative B: Expression Graph with Runtime CSE

**Approach**: Instead of a compile-time AST, build an expression DAG at
runtime. Each constraint registers its computation graph. The system performs
CSE across all constraints, then computes Jacobians (and Hessians) from
the shared graph.

**Advantages**:
- Handles dynamic structure (variable number of constraints)
- CSE across constraints and objectives
- Can compute Hessians efficiently via reverse-mode on the graph
- Graph can be JIT-compiled (extends existing Cranelift infrastructure)

**Disadvantages**:
- Indirection overhead for graph traversal
- More complex implementation
- Users define constraints via a graph-building API instead of plain Rust
  expressions (worse ergonomics)
- Similar to what CasADi does -- proven but complex

**Why it might be better**: This is the approach used by production
optimization libraries (CasADi, JAX). It naturally supports Hessians,
cross-constraint CSE, and sparsity detection. If Solverang ever targets
problems with 10,000+ variables, this architecture is probably necessary.

### Alternative C: Hybrid (Current Path, Extended)

**Approach**: Keep the macro for first derivatives. Add optional runtime
AD for second derivatives. Add BFGS/L-BFGS to avoid needing Hessians for
most problems.

**Advantages**:
- Minimal disruption to existing architecture
- Zero-overhead first derivatives for the common case
- BFGS sidesteps the Hessian problem for moderate-size problems
- Can incrementally add runtime AD for specific use cases

**Disadvantages**:
- Two differentiation systems to maintain
- Users must understand when each applies
- Large-scale optimization still requires runtime AD or finite differences
- Architectural complexity accumulates over time

**This is the path of least resistance and probably the right near-term
choice.** But it should be adopted with eyes open about its ceiling.

### Alternative D: Source-to-Source Transformation (Enzyme-style)

**Approach**: Use LLVM-level AD (Enzyme) to differentiate arbitrary Rust
code at the IR level.

**Advantages**:
- Handles all Rust code, including control flow and library calls
- Reverse-mode AD with minimal overhead
- Exact derivatives, no approximation
- Works at LLVM IR level, so sees through all abstractions

**Disadvantages**:
- Enzyme for Rust is experimental and not on stable
- Requires nightly Rust compiler
- Build system complexity
- Not suitable for a library that should work on stable Rust

**Why it matters**: In 2-3 years, if Enzyme stabilizes for Rust, it could
replace both the macro system and any runtime AD with a single, zero-overhead,
all-powerful AD system. The architecture should be designed so this could
be swapped in without changing user-facing APIs.

---

## 8. Two-Year Architecture Evolution Prediction

### What This Architecture Will NOT Be Able To Do in 2 Years

1. **Large-scale topology optimization.** Problems with 100,000+ design
   variables require adjoint methods, multigrid solvers, and matrix-free
   linear algebra. None of these fit the current dense-matrix, explicit-Jacobian
   architecture.

2. **Mixed continuous-discrete optimization.** Choosing which constraints to
   apply (constraint selection), integer numbers of entities, or discrete
   topology choices require branch-and-bound or MILP solvers, which are an
   entirely different class of algorithm.

3. **PDE-constrained optimization.** If Solverang's CAD kernel ever needs to
   optimize shapes under physical simulation constraints (stress, thermal,
   fluid), this requires PDE solvers coupled with adjoints. Out of scope.

4. **Robust optimization under uncertainty.** Optimizing worst-case
   performance under parameter uncertainty requires sampling, scenario
   decomposition, or semi-infinite programming.

5. **Real-time re-optimization.** If the user drags a point and expects the
   optimization objective to be re-minimized in < 16ms (60fps), the solver
   must support warm-starting from the previous solution and exploit
   incremental structure. The `drag()` method on `ConstraintSystem` is a
   step toward this, but optimization warm-starting is harder (active set
   changes, trust region radius adaptation).

6. **Differentiable optimization layers.** Computing derivatives *through*
   the solver (e.g., for learning constraint parameters) requires implicit
   differentiation of the KKT conditions. This is a research topic, and the
   current architecture has no path to it.

### What It WILL Be Able To Do

1. **Constrained optimization of CAD sketches** with < 500 variables, smooth
   objectives, and bound/inequality constraints. This covers the vast
   majority of 2D sketch optimization use cases.

2. **Design rule checking** via the inequality constraint infrastructure
   already in place.

3. **Multi-objective scalarization** (weighted sum, epsilon-constraint) for
   balancing competing objectives.

4. **Parametric sweeps** (optimize for each value of a parameter) by
   warm-starting from the previous solution.

### Timeline Prediction

- **Month 0-3**: BFGS/L-BFGS optimizer (unconstrained) and ALM outer loop
  with existing LM as inner solver (equality-constrained). Both use
  macro-generated gradients only. Log-barrier for inequalities (not
  SlackVariableTransform). Works for n < 200.

<!-- Decision: Phase 1 = BFGS (unconstrained) + ALM (equality-constrained). Log-barrier for inequalities, not SlackVariableTransform. Aligned with 00_synthesis.md. -->

- **Month 3-6**: Interior-point method replaces slack variables for better
  behavior at active constraints. Auto-detection of optimization vs
  constraint-satisfaction mode.

- **Month 6-12**: Runtime AD (forward-mode dual numbers) as optional backend
  for constraints that the macro cannot handle. BFGS handles most cases
  without Hessians. Scaling tested up to n = 1,000.

- **Month 12-18**: Expression graph for cross-constraint CSE and efficient
  Hessian computation. JIT compilation of the graph. This is where the
  architecture starts to diverge from "macro does everything."

- **Month 18-24**: Multi-start framework. Sensitivity analysis. The macro
  approach is still used for simple constraints but is no longer the only
  differentiation path.

---

## 9. Recommendations

### For Each Identified Risk

| Risk | Severity | Recommendation |
|---|---|---|
| **Macro cannot handle control flow** | High for optimization, low for constraints | Add runtime AD fallback (forward-mode dual numbers) behind a trait method `supports_ad()`. Keep macro as the fast path. |
| **Expression size explosion** | Medium | Add CSE pass to `simplify()`. Short-term: document the limitation and recommend splitting large objectives into multiple residuals. |
| **No Hessian support** | High for optimization | Use BFGS/L-BFGS as the primary optimization algorithm (avoids Hessians). Add Hessian-vector products via dual numbers for trust-region methods later. |
| **Non-smooth objectives produce cryptic errors** | High | Add smoothness detection: check for `Abs` nodes in the AST and warn at compile time. Add smooth approximation utilities (`smooth_abs(x, eps)`, `log_sum_exp`, `softmax`). Improve runtime error messages to include causal context. |
| **Slack variable rank deficiency** | Critical for optimization | Replace `SlackVariableTransform` with log-barrier interior-point method for optimization problems. Keep slack variables as a simple option for pure feasibility problems. |
| **Saddle point convergence** | High | Any optimization Newton method must check Hessian positive-definiteness (or use trust-region). BFGS maintains positive-definiteness by construction, which is another argument for BFGS as the default. |
| **Unbounded problems** | Medium | Add gradient norm and step size monitoring. If the step size grows without bound for K consecutive iterations, report "possibly unbounded." |
| **Scaling beyond n = 500** | Medium (2-year horizon) | Use sparse linear algebra (the `sparse_solver.rs` already exists). L-BFGS for optimization. Plan for expression graph architecture at n > 5,000. |
| **Architecture lock-in** | Low but important | Define optimization through the `Problem` trait extension, not new traits. Keep the `residuals`/`jacobian` interface as the universal contract. Any optimizer should work with any differentiable `Problem`. |
| **LM termination message fragility** | Medium | Replace string-matching in `convert_termination` with a flag field in `LMProblemAdapter`. The adapter already knows why it returned `None`; record the reason as an enum field and read it after `lm.minimize()` returns. |
| **LM `NoImprovementPossible` false convergence** | Medium for optimization | Map `NoImprovementPossible` to `NotConverged` unconditionally, or introduce a `Stalled` variant in `SolveResult`. Never report a stalled solver as converged; let the outer algorithm decide whether stalling is acceptable. |

### The Single Most Important Recommendation

**Do not extend the macro to generate Hessians.** The AST explosion makes
this unviable beyond toy problems. Instead, adopt BFGS as the primary
optimization algorithm (which needs only gradients) and add runtime
forward-mode AD for the rare cases where exact Hessian-vector products
are needed. This keeps the macro doing what it does well (first derivatives
of moderate expressions) while avoiding its fundamental limitation
(combinatorial blowup of higher derivatives).

### The Second Most Important Recommendation

**Replace slack variables with interior-point for optimization.** The
`SlackVariableTransform` is clever for constraint satisfaction but
pathological for optimization (rank deficiency at active constraints,
poor scaling, no convergence rate guarantees). A log-barrier interior-point
method is the standard approach for constrained optimization and should be
the default when inequalities are present in an optimization context.
