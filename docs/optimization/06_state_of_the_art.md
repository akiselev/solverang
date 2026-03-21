# State of the Art: Optimization Solvers and Recent Research

This document surveys the current landscape of optimization solver implementations,
recent academic research (2022-2025), and opportunities for Solverang to push
beyond existing tools.

---

## Part 1: State-of-the-Art Implementations

### 1.1 IPOPT (Interior Point OPTimizer)

**Repository**: https://github.com/coin-or/Ipopt
**License**: Eclipse Public License (EPL)
**Language**: C++ with interfaces to C, Fortran, Python, Julia, AMPL

**Algorithm approach**: Primal-dual interior point method with a filter
line-search strategy (Fletcher-Leyffer). The algorithm converts inequality
constraints to equalities via slack variables, then applies a log-barrier
relaxation, producing a sequence of barrier subproblems solved by Newton's
method on the KKT system. Global convergence of each barrier subproblem is
enforced by a filter line-search that tracks the trade-off between optimality
and feasibility, accepting steps that improve one without worsening the other
too much. The barrier parameter mu is updated either monotonically
(Fiacco-McCormick) or adaptively (default).

**Derivative requirements**: Requires first derivatives (gradient of objective,
Jacobian of constraints). Second derivatives (Hessian of the Lagrangian) are
optional but strongly recommended for performance. When exact Hessians are not
provided, IPOPT falls back to a limited-memory quasi-Newton (L-BFGS) update.

**Sparsity exploitation**: The core of each iteration is solving a sparse
symmetric indefinite linear system (the augmented KKT system). IPOPT interfaces
with multiple sparse direct solvers:
- MA27, MA57, MA77, MA86, MA97 (HSL)
- MUMPS (open source)
- Pardiso (Intel MKL or academic)
- SPRAL (open source, multi-threaded)

Inertia correction is applied by adding delta*I to the Hessian block when the
factorization reveals incorrect inertia, ensuring descent properties.

**Problem types**: General nonlinear programs (NLPs) with equality and inequality
constraints, including nonconvex problems. Not designed for integer variables or
combinatorial structure.

**Performance characteristics**: Excellent for large-scale problems (millions of
variables) when the Hessian is sparse and a good sparse linear solver is
available. Typically 20-80 iterations. Sensitive to choice of linear solver.
The filter line search provides robustness against infeasible starting points.

**API design**: Callback-based (TNLP interface). User provides:
- `eval_f(x)` -- objective
- `eval_grad_f(x)` -- gradient of objective
- `eval_g(x)` -- constraint values
- `eval_jac_g(x)` -- Jacobian of constraints (sparse triplet)
- `eval_h(x, lambda)` -- Hessian of Lagrangian (sparse triplet, optional)

Sparsity structure is declared once at initialization; only values change
between iterations.

**Comparison with Solverang**: IPOPT's callback architecture maps well to
Solverang's `Problem` trait. Solverang's compile-time symbolic differentiation
could automatically provide exact Hessians via the `#[auto_jacobian]` macro
(extended to `#[auto_hessian]`), eliminating the need for quasi-Newton
fallbacks. Solverang's Cranelift JIT would compile the entire evaluation
pipeline to native code, potentially matching or exceeding IPOPT's interpreted
callback performance. Solverang's graph decomposition could feed IPOPT-style
solves on independent subproblems in parallel.

**Key references**:
- Wachter, Biegler: "On the implementation of an interior-point filter
  line-search algorithm for large-scale nonlinear programming",
  Mathematical Programming 106(1), 2006.

---

### 1.2 SNOPT (Sparse Nonlinear OPTimizer)

**Website**: https://ccom.ucsd.edu/~optimizers/solvers/snopt/
**License**: Commercial (academic licenses available)
**Language**: Fortran with C/C++/Python interfaces

**Algorithm approach**: Sparse sequential quadratic programming (SQP) with a
smooth augmented Lagrangian merit function. Each major iteration solves a
quadratic programming (QP) subproblem using the reduced-Hessian solver SQOPT.
The Hessian of the Lagrangian is approximated via a limited-memory quasi-Newton
(L-BFGS) update, avoiding the need for exact second derivatives.

The algorithm operates in major/minor iterations:
- Major iterations update the primal-dual iterate by solving the QP subproblem.
- Minor iterations solve the QP via an active-set method on the reduced space.
- A smooth augmented Lagrangian merit function with explicit infeasibility
  handling controls globalization.

**Derivative requirements**: Requires first derivatives (gradient of objective,
Jacobian of constraints). Exact Hessians are NOT used; instead, a limited-memory
quasi-Newton approximation is maintained. This is the key architectural
distinction from IPOPT.

**Sparsity exploitation**: Exploits sparsity in the constraint Jacobian. The
reduced-Hessian approach means that the QP subproblem involves a dense matrix of
dimension equal to the number of superbasic variables (degrees of freedom at the
current iterate). This is efficient when the number of active degrees of freedom
is moderate (up to ~2000), even if the total number of variables and constraints
is very large.

**Problem types**: Large-scale NLPs where many variables enter linearly or where
there are relatively few degrees of freedom at the solution (many active
constraints). Particularly strong when the problem has a "narrow" feasible
region.

**Performance characteristics**: Very efficient when the number of nonlinear
variables or superbasic variables is small relative to the total problem size.
Less efficient than IPOPT for problems with many nonlinear degrees of freedom.
The quasi-Newton approach avoids Hessian computation but sacrifices superlinear
convergence in the full space.

**API design**: Fortran-style with workspace arrays. Modern wrappers (pyOptSparse,
AMPL) provide cleaner interfaces. Problem structure is specified via sparsity
patterns and callback functions.

**Comparison with Solverang**: SNOPT's SQP approach is relevant for Solverang's
planned optimization support. The reduced-Hessian technique could be valuable
for CAD problems where many constraints are linear (e.g., coincidence, alignment)
and only a few are nonlinear (e.g., tangency, distance). Solverang's symbolic
differentiation could provide exact Jacobians cheaply, improving the quasi-Newton
Hessian approximations.

**Key references**:
- Gill, Murray, Saunders: "SNOPT: An SQP Algorithm for Large-Scale Constrained
  Optimization", SIAM Review 47(1), 2005.

---

### 1.3 Artelys Knitro

**Website**: https://www.artelys.com/solvers/knitro/
**License**: Commercial
**Language**: C/C++ with Python, Java, Julia, MATLAB, R interfaces

**Algorithm approach**: Knitro offers four NLP algorithms:
1. **Interior/Direct (IPM)**: Interior-point with direct linear algebra.
2. **Interior/CG**: Interior-point with conjugate gradient (iterative) for
   the linear system, suitable for very large-scale problems.
3. **Active Set (SQP)**: Sequential quadratic programming with active-set
   management.
4. **Active Set (SQCQP)**: SQP for problems with quadratic constraints.

Knitro's distinguishing feature is the **crossover capability**: an
interior-point solution can be "cleaned up" by switching to an active-set method,
yielding an exact active set, precise multiplier estimates, and basis information.
This hybrid approach captures the strengths of both paradigm families.

Additionally, Knitro provides 2 MINLP algorithms for problems with integer
variables: (1) nonlinear branch-and-bound (NLPBB) and (2) the hybrid
Quesada-Grossman method for convex MINLP.

**Derivative requirements**: Flexible. Accepts exact first and second
derivatives, finite-difference approximations, or quasi-Newton (BFGS/L-BFGS)
Hessian approximations. Automatic differentiation is not built in but interfaces
with AD tools.

**Sparsity exploitation**: Full sparse matrix support. For the interior-point
algorithms, exploits sparsity in the KKT system. For the active-set algorithms,
exploits sparsity in the QP subproblems. Supports user-provided sparsity
patterns.

**Problem types**: NLP, QCQP, MINLP, MPEC (mathematical programs with
equilibrium constraints). Handles nonconvex problems, integer variables, and
complementarity constraints.

**Performance characteristics**: Consistently ranks among the top solvers in
benchmarks (Mittelmann, CUTEst). The ability to choose among four algorithms
and crossover between them provides robustness across diverse problem types.
Version 15.1 (2024) introduced significant performance improvements.

**API design**: Rich API with problem construction, callback evaluation, and
extensive tuning parameters. Supports both modeler interfaces (AMPL, GAMS) and
direct API calls.

**Comparison with Solverang**: Knitro's multi-algorithm approach validates
Solverang's `AutoSolver` concept. The crossover between interior-point and
active-set methods is a technique Solverang could adopt: solve with an
IPM for robustness, then refine with an active-set polish for exact
constraint identification. The MINLP support is relevant for CAD problems
with discrete choices (e.g., selecting between tangent configurations).

---

### 1.4 NLopt

**Repository**: https://github.com/stevengj/nlopt
**License**: MIT/LGPL (algorithm-dependent)
**Language**: C with wrappers for C++, Python, Julia, Rust, and many others

**Algorithm approach**: NLopt is a meta-library that provides a unified interface
to many optimization algorithms. It does not implement a single solver but
instead wraps and unifies:

- **Local gradient-based**: MMA (Method of Moving Asymptotes), SLSQP, CCSAQ,
  L-BFGS, TNEWTON (truncated Newton), VAR1/VAR2 (variable-metric methods).
- **Local derivative-free**: COBYLA, BOBYQA, NEWUOA, Nelder-Mead, Sbplx (subplex),
  PRAXIS.
- **Global gradient-based**: StoGO (stochastic global optimization), AGS
  (adaptive global search), MLSL (multi-level single-linkage).
- **Global derivative-free**: DIRECT, DIRECT-L, CRS (controlled random search),
  ISRES (improved stochastic ranking evolution strategy), ESCH (evolutionary).

**Derivative requirements**: Depends on the chosen algorithm. Gradient-based
methods require first derivatives. Derivative-free methods require only
function evaluations. No algorithm in NLopt uses second derivatives directly.

**Sparsity exploitation**: None. NLopt treats all problems as dense. This is a
fundamental limitation for large-scale problems.

**Problem types**: Unconstrained, bound-constrained, and nonlinearly constrained
(equality and inequality) optimization. No integer variable support.

**Performance characteristics**: The strength is breadth, not depth. NLopt makes
it easy to try many algorithms on the same problem. Individual algorithms are
reference implementations, generally not competitive with specialized solvers
(IPOPT, SNOPT, Knitro) for large-scale problems. COBYLA is particularly useful
as a robust derivative-free option for small problems.

**API design**: Minimalist C API. Create an optimizer with `nlopt_create(algorithm,
n)`, set bounds, add constraints, set objective, call `nlopt_optimize`. The Rust
wrapper (nlopt crate) provides a thin safe wrapper around this C API.

**Comparison with Solverang**: NLopt's multi-algorithm approach validates the
concept of a solver with multiple backends. Solverang's architecture could
subsume NLopt's role by offering a similar algorithm-selection interface but
with native Rust implementations, automatic differentiation, and sparsity
exploitation that NLopt lacks. Specific algorithms worth porting: MMA for
convex-approximation methods and COBYLA for derivative-free fallback.

> **REVIEW NOTE**: WORHP (We Optimize Really Huge Problems, https://worhp.de)
> is absent from this survey. It is a commercial SQP/IPM solver used operationally
> by ESA for flight-dynamics applications, supports general NLP (equality +
> inequality constraints), and is a named CasADi backend (Section 1.6). It is
> comparable in scope to SNOPT and Knitro. Its architecture (sparse SQP with
> BFGS/SR1 Hessian updates, native MPS I/O) is worth a brief entry in Part 1
> for completeness, particularly given Solverang's aerospace-adjacent use cases.

---

### 1.5 Ceres Solver (Google)

**Repository**: https://github.com/ceres-solver/ceres-solver
**Website**: http://ceres-solver.org/
**License**: Apache-2.0
**Language**: C++

**Algorithm approach**: Primarily a nonlinear least-squares solver using
Levenberg-Marquardt and Dogleg trust-region methods. Also supports general
unconstrained optimization via L-BFGS and nonlinear conjugate gradient.

The key innovation is the **CostFunction / AutoDiffCostFunction** framework:
- Users write a templated `operator()` functor.
- Ceres instantiates it with `Jet<double, N>` types (dual numbers) to compute
  exact derivatives via forward-mode AD.
- `DynamicAutoDiffCostFunction` handles variable-size parameter blocks by
  evaluating in strides.

**Derivative requirements**: Three options per cost function:
1. `AutoDiffCostFunction` -- automatic differentiation via Jet types.
2. `NumericDiffCostFunction` -- finite differences.
3. `AnalyticCostFunction` -- hand-coded Jacobians.
Different cost functions in the same problem can use different strategies.

**Sparsity exploitation**: Sophisticated structure exploitation:
- **Schur complement**: Bundle adjustment problems have specific sparsity where
  point parameters can be eliminated, leaving a smaller camera-only system.
  Ceres provides `SPARSE_SCHUR`, `DENSE_SCHUR`, and `ITERATIVE_SCHUR` solvers.
- **Elimination ordering**: Constrained Approximate Minimum Degree (CAMD)
  ordering minimizes fill-in during sparse factorization.
- **Nested dissection**: Fill-reducing ordering for the Schur complement.
- **Sparse direct solvers**: Integrates SuiteSparse (CHOLMOD), Accelerate
  (Apple), and Eigen's sparse Cholesky.

**Problem types**: Nonlinear least-squares (primary use case), bounds constraints,
general unconstrained optimization. No equality/inequality constraint support
beyond bounds.

**Performance characteristics**: State-of-the-art for large-scale bundle
adjustment (tens of thousands of cameras, millions of 3D points). Threaded
Jacobian evaluation and linear solvers. Robust loss functions (Huber, Cauchy,
etc.) for outlier handling.

**API design**: Problem-building API. Users add `ResidualBlock` objects, each
with a `CostFunction`, optional `LossFunction`, and parameter block pointers.
The `Problem` object manages the graph of residuals and parameters.

**Comparison with Solverang**: Ceres is the closest architectural relative to
Solverang in the optimization ecosystem. Both use a residual-block / parameter-
block decomposition. Key differences:
- Ceres uses forward-mode AD via Jet types at runtime; Solverang uses compile-time
  symbolic differentiation via proc macros + JIT compilation. Solverang's approach
  can be faster for large systems because it generates optimized native code.
- Ceres has Schur complement solvers; Solverang has graph-based decomposition.
  These are complementary techniques.
- Ceres lacks general constrained optimization; Solverang plans full NLP support
  via the optimization extension.
- Solverang's `#[auto_jacobian]` is analogous to `AutoDiffCostFunction` but
  operates at compile time, enabling JIT optimization.

**Key references**:
- Agarwal et al.: "Ceres Solver" (http://ceres-solver.org/).

---

### 1.6 CasADi

**Repository**: https://github.com/casadi/casadi
**Website**: https://web.casadi.org/
**License**: LGPL-3.0
**Language**: C++ core with Python, MATLAB/Octave frontends

**Algorithm approach**: CasADi is not a solver but a **symbolic computation
framework** for constructing optimization problems and computing their
derivatives. It provides:
- Symbolic expression graphs using compressed column storage (CCS) sparse matrices.
- Two symbolic types: **SX** (scalar-valued, elementwise operations) and **MX**
  (matrix-valued, can wrap external functions).
- Forward and reverse mode AD on expression graphs.
- Code generation: C code output that, compiled with -O2, runs 4-10x faster than
  CasADi's built-in virtual machine interpreter.

CasADi then interfaces with external solvers:
- NLP: IPOPT, SNOPT, Knitro, WORHP, BONMIN, CONOPT
- QP: qpOASES, OOQP, CPLEX, Gurobi, OSQP
- ODE/DAE: SUNDIALS (CVODES, IDAS)

**Derivative requirements**: Computes all required derivatives automatically
from the symbolic graph. Supports first and second derivatives, including the
Hessian of the Lagrangian needed by interior-point and SQP methods.

**Sparsity exploitation**: Central design principle. "Everything is a sparse
matrix." CasADi propagates sparsity patterns through the expression graph and
generates derivative code that only computes structurally nonzero entries. This
is one of CasADi's most important features for large-scale optimization.

**Problem types**: NLPs, QPs, ODEs, DAEs, optimal control problems, moving
horizon estimation. Through solver backends, supports convex and nonconvex
problems.

**Performance characteristics**: The symbolic expression graph enables aggressive
sparsity exploitation and efficient code generation. The overhead of symbolic
construction is amortized over many solves. For optimal control problems
(discretized with direct collocation), CasADi + IPOPT is the standard toolchain.

**API design**: Symbolic construction API. Users build expressions, define
decision variables, objective, and constraints, then call a solver. The
"everything-is-a-sparse-matrix" paradigm unifies scalars, vectors, and matrices.

**Comparison with Solverang**: CasADi's symbolic approach is closest to
Solverang's `#[auto_jacobian]` macro system. Key parallels:
- Both build expression graphs and differentiate symbolically.
- Both exploit sparsity in derivative computations.
- CasADi generates C code; Solverang generates Cranelift IR and JIT-compiles.
- CasADi's symbolic graph is constructed at runtime; Solverang's is
  constructed at compile time (proc macro), enabling more optimization.
Solverang could adopt CasADi's "everything-is-sparse" philosophy more deeply,
particularly for Hessian computation where sparsity patterns are crucial.

**Key references**:
- Andersson et al.: "CasADi: a software framework for nonlinear optimization
  and optimal control", Mathematical Programming Computation 11(1), 2019.

---

### 1.7 Optim.jl (Julia)

**Repository**: https://github.com/JuliaNLSolvers/Optim.jl
**License**: MIT
**Language**: Julia

**Algorithm approach**: Pure Julia implementation of classical optimization
algorithms:
- **Derivative-free**: Nelder-Mead, simulated annealing, particle swarm.
- **First-order**: Steepest descent, conjugate gradient (Hager-Zhang,
  Fletcher-Reeves), BFGS, L-BFGS, accelerated gradient.
- **Second-order**: Newton's method with line search, Newton with trust region.
- **Constrained**: Interior-point Newton (IPNewton) supporting box constraints
  and nonlinear inequality/equality constraints via the Optim constraints
  interface; Fminbox (wrapping unconstrained methods for box constraints).

**Derivative requirements**: Flexible. Supports analytical gradients/Hessians,
finite differences, and automatic differentiation via Julia's ForwardDiff.jl
and ReverseDiff.jl ecosystem.

**Sparsity exploitation**: Limited built-in support, but Julia's multiple
dispatch allows users to plug in sparse matrix types. The SparseDiffTools.jl
package provides efficient sparse Jacobian and Hessian computation via
coloring algorithms.

**Problem types**: Unconstrained, box-constrained, and (with extensions)
nonlinearly constrained optimization. Part of the broader JuliaNLSolvers
family which includes NLsolve.jl for equation solving and LsqFit.jl for
curve fitting.

**Performance characteristics**: Julia's JIT compilation means the optimizer
and objective are compiled together, enabling function inlining, SIMD, and
other optimizations. This "whole-program optimization" is unique among
optimization libraries and yields excellent performance for small-to-medium
problems.

**API design**: Functional API. `optimize(f, x0, method)` with optional gradient
and Hessian arguments. Clean separation between objective specification and
solver selection. Custom preconditioners via dispatch.

**Comparison with Solverang**: Optim.jl validates the JIT-compilation approach.
Julia achieves zero-overhead AD + optimization because the JIT compiler sees
both the objective and the optimizer. Solverang's Cranelift JIT for residual/
Jacobian evaluation achieves a similar effect. The key difference is that
Solverang generates specialized code at compile time (proc macro) rather than
relying on runtime JIT, which means lower startup overhead for repeated solves.

---

### 1.8 scipy.optimize

**Website**: https://docs.scipy.org/doc/scipy/reference/optimize.html
**License**: BSD
**Language**: Python (with Fortran/C backends)

**Algorithm approach**: scipy.optimize.minimize provides a unified interface to
multiple algorithms:
- **Unconstrained**: Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, Newton-CG,
  trust-ncg, trust-krylov, trust-exact.
- **Constrained**: COBYLA, SLSQP, trust-constr.
- **Least-squares**: `least_squares` with trust-region reflective and dogbox.
- **Root-finding**: `root` with hybr (MINPACK), lm (Levenberg-Marquardt),
  Broyden, Anderson, Krylov.

The `trust-constr` method (introduced in SciPy 1.1) is the most versatile
constrained optimizer, implementing a trust-region SQP method with Byrd-Omojokun
and barrier methods internally.

**Derivative requirements**: Accepts analytical gradients/Hessians, finite
differences, or `LinearOperator` objects for Hessian-vector products. No
built-in AD support (users typically combine with JAX or autograd).

**Sparsity exploitation**: The `trust-constr` and `least_squares` methods
accept sparse Jacobians and Hessians. SLSQP and L-BFGS-B do not exploit
sparsity.

**Problem types**: Unconstrained, bound-constrained, and general nonlinearly
constrained optimization. Least-squares with bounds. Root-finding.

**Performance characteristics**: Reference implementation quality. Good for
prototyping and small-medium problems. The Fortran backends (L-BFGS-B, SLSQP)
are efficient but the Python overhead limits performance for problems requiring
many function evaluations.

**API design**: Functional API with `minimize(fun, x0, method, jac, hess,
constraints, bounds, options)`. OptimizeResult object returns solution,
convergence info, and diagnostics.

**Comparison with Solverang**: scipy.optimize serves as the baseline reference
implementation. Solverang's native Rust implementation with JIT compilation
should significantly outperform scipy for function-evaluation-heavy problems.
The `trust-constr` algorithm's Byrd-Omojokun approach is a proven constrained
optimization strategy worth studying for Solverang's IPM/SQP implementation.

---

### 1.9 JAX/Optax/JAXopt

**Repositories**:
- JAX: https://github.com/google/jax
- Optax: https://github.com/google-deepmind/optax
- JAXopt: https://github.com/google/jaxopt (active development ceased; a subset
  of features -- losses, projections, L-BFGS -- ported to Optax; scientific
  computing optimizers were NOT fully transferred; minimal releases continue)

**License**: Apache-2.0
**Language**: Python

**Algorithm approach**: JAX provides a transformable numerical computing
framework with:
- `jax.grad` / `jax.jacobian` / `jax.hessian` -- automatic differentiation.
- `jax.jit` -- JIT compilation to XLA (GPU/TPU/CPU).
- `jax.vmap` -- vectorized mapping for batched computation.

Optax provides gradient transformation chains: Adam, RMSProp, AdaGrad, LAMB,
etc. These are "gradient processors" that compose via `optax.chain`.

JAXopt (now partially merged into Optax) provided classical optimizers:
- L-BFGS, gradient descent, proximal gradient.
- **Differentiable optimization**: solutions are differentiable with respect
  to their inputs via implicit differentiation (Argmin theorem) or unrolled
  differentiation.
- Batchable: `jax.vmap` vectorizes across problem instances.
- Hardware-accelerated: GPU/TPU execution.

**Derivative requirements**: JAX computes all derivatives automatically via
source-to-source transformation and XLA compilation. Forward and reverse mode.

**Sparsity exploitation**: Limited. JAX operates on dense arrays by default.
Sparse support is experimental (jax.experimental.sparse). This is a significant
limitation compared to classical NLP solvers.

**Problem types**: Primarily unconstrained optimization (ML training). JAXopt
adds proximal operators for simple constraints. Not designed for general NLP
with nonlinear equality/inequality constraints.

**Performance characteristics**: Exceptional for GPU-parallel problems with
regular structure (e.g., neural network training). The composition of JIT +
AD + vmap enables impressive throughput. Less suited for irregular sparse
problems common in CAD constraint solving.

**API design**: Functional/compositional. Users compose gradient transformations
and apply them in a training loop. JAXopt provides `solver.run(init_params)`
returning `(params, state)`.

**Comparison with Solverang**: JAX's philosophy of "transformable functions"
resonates with Solverang's design. Both compile computation graphs for
efficient execution. Key differences:
- JAX targets GPU/TPU; Solverang targets CPU (Cranelift).
- JAX is dense-array oriented; Solverang exploits sparsity.
- JAX provides runtime AD; Solverang provides compile-time symbolic
  differentiation.
- JAXopt's differentiable optimization (implicit differentiation of solver
  solutions) is an advanced feature Solverang could adopt for sensitivity
  analysis and bilevel optimization.

---

### 1.10 Clarabel.rs

**Repository**: https://github.com/oxfordcontrol/Clarabel.rs
**License**: Apache-2.0
**Language**: Rust (native)

**Algorithm approach**: Interior-point method for convex conic optimization
using a novel homogeneous embedding. Unlike solvers based on the standard
homogeneous self-dual embedding (HSDE), Clarabel handles quadratic objectives
directly without epigraphical reformulation, making it significantly faster
for QPs.

**Derivative requirements**: Not applicable (problem is specified as matrices
and vectors in standard conic form). No function callbacks.

**Sparsity exploitation**: Full sparse matrix support using compressed sparse
column (CSC) format. The KKT system is solved via sparse LDL factorization.

**Problem types**: Linear programs (LPs), quadratic programs (QPs),
second-order cone programs (SOCPs), semidefinite programs (SDPs), exponential
cones, power cones, and generalized power cones.

**Performance characteristics**: Competitive with commercial conic solvers
(MOSEK, Gurobi). Infeasibility detection via homogeneous embedding. Recent
GPU-accelerated version (CuClarabel, December 2024) extends the solver to
CUDA with mixed-precision linear system solvers.

**API design**: Rust-native API. Users construct the problem in standard conic
form: `min 0.5 x'Px + q'x s.t. Ax + s = b, s in K` where K is a Cartesian
product of cones. The solver returns primal/dual solution, status, and
iteration count.

**Comparison with Solverang**: Clarabel.rs is the most relevant Rust-native
reference implementation. Its conic programming framework could serve as the
QP/SOCP subproblem solver within Solverang's SQP implementation. The LDL
factorization code and sparse matrix handling could inform Solverang's
sparse solver development. Clarabel.rs supports faer as an optional linear
solver backend (`--features faer-sparse`), validating faer as a production-grade
sparse linear algebra choice for Rust-native IPM implementations.

**Key references**:
- Goulart, Chen: "Clarabel: An interior-point solver for conic programs with
  quadratic objectives", arXiv:2405.12762, 2024.

---

### 1.11 OSQP (Operator Splitting QP Solver)

**Repository**: https://github.com/osqp/osqp
**Website**: https://osqp.org/
**License**: Apache-2.0
**Language**: C with Rust interface (osqp crate)

**Algorithm approach**: ADMM (Alternating Direction Method of Multipliers)
applied to convex quadratic programs. Uses a single matrix factorization in
the setup phase; all subsequent operations are matrix-vector multiplications,
making it extremely efficient for parametric problems where only the
data changes.

**Derivative requirements**: Not applicable (QP formulation, no callbacks).

**Sparsity exploitation**: Custom sparse linear algebra routines. Exploits
structure in the ADMM splitting. Warm-starting support enables caching the
matrix factorization across related solves.

**Problem types**: Convex quadratic programs: `min 0.5 x'Px + q'x s.t.
l <= Ax <= u`.

**Performance characteristics**: Very fast for small-to-medium QPs, especially
in parametric settings (e.g., model predictive control) where warm-starting
is effective. First-order method, so convergence to high accuracy is slower
than interior-point methods, but low-accuracy solutions are extremely fast.

**API design**: C API with `osqp_setup`, `osqp_solve`, `osqp_update_*`.
The Rust wrapper provides a safe, ergonomic interface.

**Comparison with Solverang**: OSQP could serve as the QP subproblem solver
in Solverang's SQP method. The ADMM approach with warm-starting is ideal
for the sequence of related QP subproblems that arise in SQP iterations.
The Rust wrapper is already available on crates.io.

---

### 1.12 Rust-Native Optimization Libraries

#### argmin

**Repository**: https://github.com/argmin-rs/argmin
**License**: MIT/Apache-2.0
**Language**: Pure Rust

**Algorithm approach**: Framework for implementing optimization algorithms:
- Line search methods: Backtracking, More-Thuente, Hager-Zhang.
- Trust region methods: Cauchy point, Dogleg, Steihaug.
- Quasi-Newton: BFGS, L-BFGS, DFP, SR1-TrustRegion.
- Gauss-Newton: with and without line search.
- Steepest descent, conjugate gradient (Fletcher-Reeves, Polak-Ribiere).
- Nelder-Mead, simulated annealing, particle swarm.

**Key design**: Type-agnostic via Rust generics. Works with nalgebra, ndarray,
or custom types. Observation/logging and checkpointing support. The framework
separates the "solver core" (iteration logic) from the "math backend" (linear
algebra operations) via trait bounds.

**Comparison with Solverang**: argmin is the closest Rust ecosystem peer. Its
strength is the clean trait-based architecture; its weakness is the lack of
sparsity exploitation, JIT compilation, and problem decomposition. Solverang
already surpasses argmin in these areas. However, argmin's line search and
trust region implementations are well-tested and could serve as reference
implementations or even be integrated.

#### nlopt (Rust wrapper)

**Crate**: https://crates.io/crates/nlopt
**Wrapper for**: NLopt C library

Provides access to all NLopt algorithms from Rust. Thin wrapper with Rust
safety guarantees.

#### good_lp

**Repository**: https://github.com/rust-or/good_lp
**Purpose**: Mixed-integer linear programming in Rust.
**Backends**: CBC (default), HiGHS, CPLEX, SCIP, Clarabel, microlp (pure
Rust). Gurobi is NOT a named cargo feature; it is accessible only via the
external lp-solvers file-based bridge (writes an LP file, calls a binary).

#### faer

**Repository**: https://github.com/sarah-quinones/faer-rs
**Purpose**: High-performance linear algebra in pure Rust.
**Relevance**: Solverang already uses faer for sparse matrix operations. faer
provides sparse Cholesky (LLT, LDLT, Bunch-Kaufman), LU, and QR
factorizations with performance competitive with Eigen, making it suitable
as the linear algebra backend for IPM and SQP implementations. Clarabel.rs
optionally uses faer as its linear solver backend (`--features faer-sparse`).

#### HiGHS

**Repository**: https://github.com/ERGO-Code/HiGHS
**Rust crate**: https://crates.io/crates/highs
**License**: MIT
**Purpose**: Open-source LP/MIP solver written in C++.

The leading open-source LP and MIP solver, used as a backend in Julia's JuMP,
SciPy (since 1.11), and OR-Tools. Provides dual simplex, interior-point, and
branch-and-bound. The `highs` Rust crate provides safe bindings.

**Relevance**: Relevant as an alternative to Clarabel/OSQP for LP subproblems,
and is already used by good_lp. Its interior-point engine could serve as an
LP oracle for decomposition-based approaches.

#### DiffSol / DiffSL

**Repository**: https://github.com/martinjrobins/diffsol
**Crate**: https://crates.io/crates/diffsol
**License**: MIT
**Purpose**: ODE/DAE solver in pure Rust with JIT-compiled residual evaluation.

DiffSol is the most architecturally similar published Rust project to
Solverang. It provides:
- A domain-specific language (DiffSL) compiled with Cranelift or LLVM JIT.
- Enzyme AD for automatic Jacobian computation at the IR level.
- BDF (backward differentiation formulae) and Runge-Kutta solvers.
- Python/R bindings via DiffSL compiled to WebAssembly.

**Key distinction from Solverang**: DiffSol targets ODE/DAE integration (time
evolution problems) rather than algebraic constraint solving. Its symbolic
analysis happens at DSL-parse time rather than Rust proc-macro time, and
relies on Enzyme (runtime LLVM AD) rather than compile-time symbolic
differentiation. The architectures converge on the same execution model but
serve different problem domains.

**Comparison with Solverang**: The existence of DiffSol confirms the
Cranelift-JIT + AD approach is sound and practical for performance-critical
Rust numerical code. Solverang's advantage in the constraint-solving domain is
compile-time sparsity detection, which DiffSol does not perform.

---

## Part 2: Recent Research (2022-2025)

### 2.1 Advances in SQP Methods

#### Unified Funnel Restoration SQP (Vanaret & Leyffer, 2024-2025)

A significant advance in SQP methodology is the **Uno** solver, which implements
a unified framework for Lagrange-Newton methods. The key innovation is the
**funnel** globalization strategy:

- A monotonically decreasing upper bound (the "funnel") on constraint violation
  replaces the traditional filter.
- Infeasible QP subproblems are handled by a feasibility restoration strategy.
- The framework unifies filter SQP, funnel SQP, and interior-point methods
  through four common algorithmic ingredients (arXiv:2406.13454; a revised
  submission, November 2025, expanded to eight interchangeable building blocks).
- Uno implements presets that mimic existing solvers: `filtersqp` mimics
  filterSQP, `ipopt` mimics IPOPT.

The solver is competitive with filterSQP, IPOPT, SNOPT, MINOS, LANCELOT,
LOQO, and CONOPT on CUTEst benchmarks.

**Repository**: https://github.com/cvanaret/Uno (MIT license, C++)

**Relevance for Solverang**: The modular building-block architecture directly
maps to Solverang's trait-based design. Solverang could implement the funnel
SQP as a composition of interchangeable components: `GlobalizationStrategy`
(filter/funnel), `SubproblemSolver` (SQP/barrier), `ConstraintRelaxation`
(slack/penalty), and `LinearSolver` (direct/iterative).

**References**:
- Kiessling, Leyffer, Vanaret: "A unified funnel restoration SQP algorithm",
  Mathematical Programming, 2025. arXiv:2409.09208.
- Vanaret, Leyffer: "Implementing a unified solver for nonlinearly constrained
  optimization", under review at Mathematical Programming Computation (first
  submitted June 2024, revised November 2025). arXiv:2406.13454.

#### OpenSQP (2024)

A modular, reconfigurable SQP algorithm in Python that enables users to
modify key components: merit functions, line search algorithms, Hessian
approximations, and QP solvers. The standard configuration uses an augmented
Lagrangian merit function and BFGS Hessian approximation. Benchmarked on
CUTEst, competitive with SLSQP, SNOPT, and IPOPT.

**Reference**: "OpenSQP: A Reconfigurable Open-Source SQP Algorithm in Python
for Nonlinear Optimization", arXiv:2512.05392, December 2024.

---

### 2.2 Interior Point Method Improvements

#### "Interior Point Methods in the Year 2025" (Gondzio, 2025)

This comprehensive survey (published in EURO Journal on Computational
Optimization, February 2025) provides a state-of-the-art assessment:

- **Complexity improvements**: Ongoing efforts to reduce both iteration
  complexity and per-iteration computational cost, with new conditioning
  measures providing alternative complexity perspectives.
- **Sparse linear algebra**: Efficient implementation for nonsymmetric cones
  via low-rank and sparsity properties. Augmented linear systems can be made
  sparse and quasidefinite after static regularization, enabling sparse LDL
  factorization.
- **Krylov solver stopping criteria**: New criteria for iterative linear
  solvers within IPMs (SIAM Journal on Scientific Computing, April 2023).
- **Decomposition integration**: IPMs provide natural integration with
  decomposition algorithms, cutting plane methods, and column generation.

**Reference**: Gondzio: "Interior point methods in the year 2025",
EURO Journal on Computational Optimization, 2025.

#### GPU-Accelerated Interior Point Methods

Significant progress in GPU-accelerated optimization:

- **CuClarabel** (Chen, Tse, Nobel, Goulart, Boyd, Dec 2024): GPU
  implementation of Clarabel for conic optimization. Uses mixed parallel
  computing strategy processing linear constraints first, then other conic
  constraints in parallel. Mixed-precision linear system solvers achieve
  additional acceleration without compromising accuracy. arXiv:2412.19027.

- **cuPDLP** (Applegate et al., 2024): GPU-based primal-dual hybrid gradient
  for LP, with practical enhancements including adaptive restarts,
  preconditioning, and Halpern-type acceleration. Published in Operations
  Research, 2024. Performance comparable to commercial LP solvers (CPLEX,
  Gurobi).

- **GPU topology optimization**: 3D problems with 67M elements and 201M design
  variables solved in 9-23 hours using GPU-based methods.

#### Newton-CG Barrier-Augmented Lagrangian (He, Jiang, Zhang, 2023)

A method for finding approximate second-order stationary points (SOSP) of
general nonconvex conic optimization. Achieves operation complexity of
O-tilde(epsilon^{-7/2} min{n, epsilon^{-3/4}}) under constraint qualification.
First complexity analysis for finding SOSP of general nonconvex conic problems.

**Reference**: He, Jiang, Zhang: "A Newton-CG based barrier-augmented Lagrangian
method for general nonconvex conic optimization", Computational Optimization
and Applications, 2024.

---

### 2.3 Augmented Lagrangian Methods

#### ALADIN (Augmented Lagrangian Alternating Direction Inexact Newton)

ALADIN addresses distributed nonconvex optimization by combining SQP and
augmented Lagrangian ideas:

- **Convergence**: Superlinear or quadratic convergence rate with appropriate
  Hessian approximations, contrasting with ADMM's linear rate.
- **Architecture**: Two-loop structure where each agent solves a local NLP,
  then a coupled QP with equality constraints coordinates agents.
- **Practical performance**: Reduces iterations by ~10x compared to ADMM on
  power systems benchmarks.

Recent variants (2023-2025):
- **Consensus ALADIN (C-ALADIN)**: Framework for distributed optimization with
  local convergence guarantees for nonconvex problems (arXiv:2306.05662, 2023).
- **Flexible ALADIN**: Random polling variant with rigorous convergence analysis
  for both convex and nonconvex problems (March 2025).
- **ALADIN-beta**: Extension to mathematical programs with complementarity
  constraints (MPCC), 2024.

**Relevance for Solverang**: ALADIN's distributed decomposition approach maps
directly to Solverang's existing `ParallelSolver` architecture. For large CAD
assemblies that decompose into coupled subsystems (e.g., a gearbox where each
gear is a subproblem coupled through contact constraints), ALADIN's
coordination strategy could replace the current independent-component parallel
solve with a coupled-component parallel solve.

**Reference**: Houska, Frasch, Diehl: "An Augmented Lagrangian Based Algorithm
for Distributed Non-Convex Optimization", SIAM Journal on Optimization 26(2),
2016.

#### Exact Augmented Lagrangian Duality for Nonconvex MINLP (2024)

Recent work establishes exact duality results for augmented Lagrangian
relaxations of nonconvex mixed-integer nonlinear programs, opening new
algorithmic possibilities for MINLP solvers.

**Reference**: "Exact Augmented Lagrangian Duality for Nonconvex Mixed-Integer
Nonlinear Optimization", Optimization Online, July 2024.

---

### 2.4 Automatic Differentiation for Optimization

#### Enzyme AD (MIT, 2021-2025)

Enzyme performs AD at the LLVM IR level, after standard compiler optimizations:

- **Post-optimization AD**: By differentiating optimized LLVM IR rather than
  source code, Enzyme generates faster derivatives than traditional
  source-to-source or operator-overloading tools.
- **Language-agnostic**: Works with any LLVM-targeting language: C, C++,
  Fortran, Julia, Rust, Swift, MLIR.
- **GPU support**: First fully automatic reverse-mode AD tool for GPU kernels.

**Rust integration progress (2024-2025)**:
- The `#[autodiff]` attribute (`#![feature(autodiff)]`) is available in nightly
  rustc as of 2025 but remains experimental and unstable.
- Backend integration PR merged (building Enzyme from source on Tier 1 targets).
- GSoC 2025 project worked on improving Rust-Enzyme reliability and compile
  times; CI for Enzyme added to Rust's CI pipeline.
- First paper utilizing `std::autodiff` in Rust published, demonstrating
  significantly faster compilation times compared to JAX.
- Stabilization is planned but no timeline has been announced; a full RFC is
  required before stabilization.

**Rust Project Goal (2024H2)**: "Expose experimental LLVM features for automatic
differentiation and GPU offloading."

**Relevance for Solverang**: When `#[autodiff]` stabilizes in Rust, Solverang
could offer two AD pathways:
1. **Compile-time symbolic AD** via `#[auto_jacobian]` (current) -- for
   constraint systems where the expression structure is known at compile time.
2. **LLVM-level AD** via `#[autodiff]` (future) -- for user-defined cost
   functions where symbolic analysis is impractical.
This dual approach would be unique in the optimization ecosystem.

**References**:
- Moses et al.: "Reverse-mode automatic differentiation and optimization of
  GPU kernels via Enzyme", SC '21.
- "Compiler Optimizations for Higher-Order Automatic Differentiation",
  PPoPP Workshop on Differentiable Parallel Programming, 2025.

#### TapeFlow: Streaming Gradient Tapes (2024)

Introduces streaming gradient tape construction during forward pass, reducing
memory requirements for reverse-mode AD. Published at CGO 2024.

**Reference**: "TapeFlow: Streaming Gradient Tapes in Automatic
Differentiation", IEEE/ACM CGO, 2024.

---

### 2.5 Structure-Exploiting Optimization

#### Sparse Variable Projection (2024)

A VarPro scheme that jointly exploits separability and sparsity in robotic
perception problems (SLAM, structure-from-motion):

- For problems where some variables appear linearly, the method eliminates
  them via a closed-form solution, constructing a matrix-free Schur complement
  operator for the reduced problem.
- Handles gauge symmetries (global shift/rotation invariance) that are common
  in perception problems.
- 2x-35x faster than state-of-the-art methods (GTSAM, g2o) on SLAM/SfM
  benchmarks while preserving sparsity and accuracy.
- More memory efficient because the reduced problem preserves sparsity and
  uses iterative linear solvers.

**Relevance for Solverang**: CAD constraint systems often have separable
structure: some parameters appear linearly (e.g., point positions in distance
constraints) while others appear nonlinearly (e.g., angles in rotation
constraints). Variable projection could eliminate the linear parameters, solving
a smaller nonlinear system. Solverang's symbolic differentiation can detect
linearity at compile time.

**Reference**: "Sparse Variable Projection in Robotic Perception: Exploiting
Separable Structure for Efficient Nonlinear Optimization",
arXiv:2512.07969, December 2024.

#### Partially Separable Structure Exploitation (2023-2024)

Structured derivative-free methods exploit coordinate partially separable
structure (associated with sparsity) to solve large problems otherwise
intractable:

- Random pattern search algorithms designed for partially separable functions.
- Improvements make it possible to solve large problems that are intractable
  by other derivative-free methods.

**Reference**: "Exploiting Problem Structure in Derivative Free Optimization",
ACM Transactions on Mathematical Software, 2021 (with 2023 extensions).

---

### 2.6 Optimization on Manifolds (Riemannian Optimization)

#### Core Advances (2023-2024)

- **Preconditioned metrics on product manifolds**: General framework for
  optimization on product manifolds with preconditioned metrics, with
  applications to CCA and truncated SVD. (arXiv:2306.08873, 2023)
- **Proximal gradient on manifolds**: Extension of the proximal gradient method
  to composite optimization on Riemannian manifolds, enabling L1-type
  regularization in manifold settings. (Mathematics, 2024)
- **Riemannian Frank-Wolfe**: First-order methods for constrained optimization
  on manifolds with global, non-asymptotic convergence. (2023)
- **Nonsmooth Riemannian Optimization (NRO)**: Handles nonsmooth objectives
  (e.g., from L1 regularization) on manifolds.
- **Constrained Riemannian Optimization (CRO)**: Optimization with additional
  constraints beyond the manifold structure.

#### Toolboxes

- **Manopt** (MATLAB), **Pymanopt** (Python), **Manopt.jl** (Julia):
  Trust-regions, conjugate gradient on Stiefel, Grassmann, and other manifolds.

**Relevance for Solverang**: CAD constraint solving naturally involves manifold
structure:
- **SO(3)**: 3D rotation constraints (e.g., angular alignment).
- **SE(3)**: Rigid body transformations (position + orientation).
- **S^1**: 2D angular constraints (perpendicularity, tangency).
Formulating solver updates on these manifolds instead of in Euclidean space
with added normalization constraints could improve robustness and convergence.
For example, instead of representing a rotation as a quaternion with a unit-norm
constraint, solve directly on SO(3) using Riemannian gradient descent.

---

### 2.7 Mixed-Integer Nonlinear Programming (MINLP)

#### Machine Learning for MINLP (2024)

- **Learning to Optimize**: Differentiable correction layers generate integer
  outputs while preserving gradient information for end-to-end training.
  Learning-based approaches produce high-quality solutions for parametric
  MINLPs extremely quickly. (OpenReview, 2024)
- **ML-enhanced branch-and-bound**: Reinforcement learning improves branching
  decisions, node selection, and cutting plane generation without compromising
  global optimality guarantees. (2024)
- **Kriging-assisted differential evolution**: Surrogate-based optimization
  for mixed-integer variables with constraint handling. (Liu et al., 2024)

**Relevance for Solverang**: MINLP arises in CAD when design choices involve
discrete decisions: selecting among tangent configurations (internal vs.
external tangency), choosing fillet types, or deciding constraint satisfaction
strategies for over-constrained systems. Solverang's branch solver
(`solve/branch.rs`) already handles some discrete choices; a more systematic
MINLP framework could generalize this.

---

### 2.8 Robust Optimization Under Uncertainty

#### Distributionally Robust Optimization (DRO, 2024-2025)

- **Data-driven DRO**: Wasserstein ambiguity sets provide probabilistic
  robustness guarantees with finite-sample data. (Acta Numerica, 2024)
- **Decision-dependent uncertainty**: Uncertainty sets whose parameters
  depend on decisions, relevant for design problems where manufacturing
  tolerances depend on chosen geometry. (2024-2025)
- **Adjustable robust optimization**: Multi-stage decision-making under
  uncertainty with recourse. (2025)

**Relevance for Solverang**: Tolerance analysis in CAD is a natural application.
Instead of solving for nominal dimensions, solve for dimensions that are robust
to manufacturing variations. This could be formulated as a minimax problem:
find design parameters that minimize worst-case constraint violation over a
tolerance set.

---

### 2.9 Machine Learning for Solver Algorithm Selection

#### Algorithm Selection and Solver Portfolios

Research on using ML to choose the best solver or algorithm configuration
for a given problem instance is active but primarily focused on combinatorial
optimization (SAT, MIP). For continuous NLP:

- Knitro's automatic algorithm selection is a practical example: runtime
  heuristics choose among four algorithms.
- Uno's modular framework enables systematic comparison of algorithm
  combinations.
- Recent work on transfer learning for hyperparameter tuning of optimization
  algorithms shows promise.

**Relevance for Solverang**: Solverang's `AutoSolver` already performs basic
algorithm selection (NR vs. LM based on problem dimensions). This could be
extended with learned features: Jacobian sparsity pattern, condition number
estimates, problem decomposition structure, and convergence history from
previous solves could inform algorithm and parameter selection.

---

### 2.10 GPU-Accelerated Optimization

#### Key Developments (2023-2025)

- **NVIDIA cuOpt**: Open-sourced GPU-accelerated optimization for routing
  and scheduling problems. (2025)
- **cuPDLP**: Primal-dual hybrid gradient for LP on GPU, competitive with
  commercial solvers. Now integrated into HiGHS. (Operations Research, 2024)
- **CuClarabel**: GPU conic solver with mixed-precision, 2024.
  (arXiv:2412.19027)
- **cuHALLaR**: GPU-accelerated low-rank SDP solver. (2025)
- **GPU topology optimization**: 67M elements solved in ~9 hours. (2024)

The trend is clear: first-order methods (PDLP, ADMM) are natural fits for GPU
parallelism because they rely on matrix-vector products rather than sparse
factorizations. Interior-point methods require more complex GPU-parallel
linear algebra but are catching up.

**Relevance for Solverang**: For typical CAD constraint systems (10-10,000
variables), GPU acceleration offers marginal benefit because the problems are
too small to saturate GPU parallelism. However, for topology optimization and
shape optimization (millions of variables), GPU support would be valuable.
Solverang's Cranelift JIT could potentially target GPU backends in the future,
though this is a long-term goal.

---

## Part 3: Beyond State of the Art

### 3.1 Compile-Time Symbolic Differentiation + JIT

Solverang's `#[auto_jacobian]` macro system occupies a distinctive architectural
position. The specific combination of Rust proc-macro compile-time symbolic
analysis feeding a Cranelift JIT for algebraic constraint solving is novel in
the published ecosystem. The closest parallel is **DiffSol** (diffsol crate,
`martinjrobins/diffsol`), which uses a compiled DSL with Cranelift/LLVM JIT +
Enzyme AD, but targets ODE integration rather than algebraic constraint systems.
SymX (C++, `InteractiveComputerGraphics/SymX`) also combines symbolic
differentiation with JIT for FEM-style nonlinear optimization. Here is how the
pieces compare:

| System | Differentiation | Compilation | Result |
|--------|----------------|-------------|--------|
| IPOPT | User-provided callbacks | None | Interpreted callbacks |
| Ceres | Runtime forward-mode AD (Jets) | None | Some overhead vs. analytical (benchmark-dependent) |
| CasADi | Runtime expression graph + AD | C codegen (offline) | 4-10x over VM interpreter (per CasADi docs), but offline |
| JAX | Runtime source-to-source AD | XLA JIT (GPU/CPU) | Fast but dense-only |
| Julia/Optim.jl | Runtime AD (ForwardDiff) | Julia JIT | Good, whole-program opt |
| Enzyme | Post-optimization LLVM AD | LLVM (native) | Fast; TapeFlow (CGO 2024) is 1.3-2.5x faster on tested HW |
| DiffSol | Runtime Enzyme AD on DSL | Cranelift or LLVM JIT | Fast; targets ODE not constraint systems |
| **Solverang** | **Compile-time symbolic AD** | **Cranelift JIT (planned)** | **Zero-overhead AD + native; AOT functions today** |

Solverang's approach has several unique advantages:
1. **Symbolic sparsity detection**: The proc macro can analyze the expression
   structure and determine the exact sparsity pattern of the Jacobian (and
   Hessian) at compile time. No runtime sparsity detection needed.
2. **Dead-code elimination**: The symbolic expression graph enables aggressive
   elimination of structurally zero entries before code generation.
3. **Fused evaluation**: Residuals and Jacobian entries can be generated in
   a single pass, sharing common subexpressions.
4. **Ahead-of-time + JIT**: The symbolic analysis happens at compile time
   (proc macro), but code generation can be either AOT (as Rust functions)
   or JIT (via Cranelift), giving flexibility.

**Opportunities**:
- Extend to **Hessian computation**: The same symbolic analysis that produces
  Jacobians can produce Hessians of the Lagrangian, enabling second-order
  optimization methods without user effort.
- **Sparsity-aware Hessian assembly**: Propagate sparsity patterns through
  the Lagrangian multiplication to compute only structurally nonzero Hessian
  entries.
- **Automatic problem classification**: The symbolic structure reveals
  whether constraints are linear, quadratic, or general nonlinear, enabling
  automatic selection of specialized algorithms.

### 3.2 CAD-Specific Optimization

Current optimization solvers are domain-agnostic. Solverang has the opportunity
to exploit CAD-specific structure:

#### Constraint Type Exploitation

CAD constraint systems have a characteristic structure that no general optimizer
exploits:
- **Linear constraints** (coincidence, horizontal, vertical, midpoint):
  Can be eliminated symbolically, reducing problem dimension.
- **Quadratic constraints** (distance, radius, angle): Can be handled by
  QP/QCQP subproblem solvers.
- **Trigonometric constraints** (tangency, perpendicularity via angles):
  Require general NLP treatment.

Solverang's symbolic analysis can classify constraints at compile time and
route them to specialized handlers. This "tiered" approach--eliminate linear,
solve quadratic subproblems, iterate on nonlinear--is more efficient than
treating all constraints uniformly.

#### Incremental Re-solve

CAD editing is inherently incremental: the user modifies one dimension or
constraint, and the system must re-solve quickly. Optimization solvers rarely
support this. Solverang's opportunities:
- **Warm-starting from previous solution**: Interior-point and SQP methods
  accept warm starts, but the barrier parameter / active set must be managed
  carefully.
- **Incremental factorization**: When only a few rows/columns of the Jacobian
  change, the factorization can be updated rather than recomputed.
  Rank-k updates to Cholesky/LDL factorizations are O(nk) vs. O(n^2) or
  O(n^3) for refactorization.
- **Sensitivity-based prediction**: Compute dx/dp (sensitivity of the solution
  with respect to parameters) and use it to predict the new solution when a
  parameter changes, providing an excellent warm start.
- **Dataflow tracking**: Solverang's existing `dataflow` module tracks
  which parameters are dirty, enabling selective re-evaluation.

#### Drag Solving

Interactive dragging requires solving the constraint system at 60+ FPS.
Current approaches:
- Solve in least-squares sense with penalty on deviation from drag target.
- Project the drag direction onto the null space of the Jacobian (DOF
  manifold).

Solverang's optimization extension enables a more principled approach:
- **Minimize displacement from current configuration** subject to constraints.
- This is a QP (quadratic objective + linearized constraints) that can be
  solved in microseconds with OSQP or Clarabel.
- The null-space projection becomes the unconstrained case of this QP.

#### Topology and Shape Optimization

CAD-integrated topology and shape optimization is an active research area:
- **Parametric level set methods** for shape optimization with CAD integration.
- **Automatic CAD model reconstruction** from topology optimization results.
- Solverang's constraint framework could express manufacturing constraints
  (minimum feature size, draft angles, overhang limits) as optimization
  constraints, enabling direct manufacturing-aware topology optimization.

### 3.3 Exploiting Geometric Structure

#### Manifold-Aware Solving

Geometric constraints naturally live on manifolds:

| Geometric Entity | Natural Manifold | Dimension |
|-----------------|-----------------|-----------|
| 2D direction | S^1 (unit circle) | 1 |
| 3D rotation | SO(3) | 3 |
| Rigid body pose | SE(3) | 6 |
| Unit quaternion | S^3 | 3 (tangent) |
| Symmetric positive definite matrix | SPD(n) | n(n+1)/2 |

Current approach: Represent on ambient Euclidean space + normalization
constraints (e.g., ||q|| = 1 for quaternions). This adds constraints and
introduces singularities.

Manifold-aware approach: Parameterize updates on the tangent space, retract
to the manifold after each step. Benefits:
- Fewer variables (no normalization constraints).
- No singularities from redundant parameterizations.
- Better conditioning (the Hessian reflects the intrinsic geometry).

Solverang could implement this as a `ManifoldParameterization` trait:
```
trait ManifoldParameterization {
    fn tangent_dimension(&self) -> usize;
    fn ambient_dimension(&self) -> usize;
    fn plus(x: &[f64], delta: &[f64]) -> Vec<f64>;  // retraction
    fn minus(x: &[f64], y: &[f64]) -> Vec<f64>;     // inverse retraction
    fn lift_jacobian(x: &[f64]) -> SparseMatrix;     // J_ambient = J_tangent * lift
}
```
This is analogous to Ceres Solver's `Manifold` interface (formerly
`LocalParameterization`, deprecated in Ceres 2.1.0 and removed in 2.2.0) but
integrated with Solverang's JIT compilation for zero-overhead retraction.

#### Decomposition-Aware Optimization

Solverang's existing graph decomposition finds independent components and
solves them in parallel. For optimization, this extends to:
- **Block-diagonal Hessian detection**: Independent components contribute
  independent diagonal blocks to the Hessian.
- **Bordered block-diagonal structure**: Coupled components produce a
  bordered block-diagonal Hessian that can be solved efficiently via
  Schur complement.
- **Tree-structured coupling**: Many CAD assemblies have hierarchical
  (tree-like) coupling where ALADIN-style coordination is efficient.

### 3.4 Hybrid Symbolic-Numeric Methods

Solverang's symbolic differentiation opens the door to symbolic preprocessing
that no purely numeric solver can perform:

#### Symbolic Constraint Elimination

Given the symbolic form of constraints, Solverang can:
1. **Detect and eliminate linear dependencies**: If constraints form a
   linear subsystem, solve it symbolically to eliminate variables.
2. **Substitute closed-form solutions**: If a constraint has the form
   `x_i = expr(x_j, ...)`, substitute directly, reducing problem dimension.
3. **Recognize special structures**: Detect when a subsystem is solvable
   in closed form (e.g., two-circle intersection, line-circle intersection)
   and use geometric algorithms instead of iterative solvers.

Solverang's `reduce` module already implements some of these techniques.
The optimization extension should preserve and extend this capability.

#### Resultant Theory for Polynomial Systems

For constraint systems that are polynomial (distance, angle, area constraints),
classical algebraic geometry techniques can find all solutions:
- **Sylvester resultants**: Eliminate variables from polynomial systems.
- **Groebner bases**: Compute the variety of the constraint system.
- **Homotopy continuation**: Track solution paths from a known system to the
  target system (guarantees finding all solutions).

These techniques are complementary to iterative optimization: they find all
solutions (not just the nearest local minimum), which is valuable for:
- Detecting alternative configurations in CAD (e.g., both tangent circles).
- Verifying that the optimization converged to the global minimum.
- Computing the number of valid configurations.

Recent work on hybrid symbolic-numeric computation using resultant theory
(2024) extends these methods to transcendental terms, enabling application
to a broader class of CAD constraints.

#### Interval Arithmetic for Global Verification

Interval methods can verify that an optimization solution is indeed a global
minimum within a region, or prove that no solution exists. This provides
guarantees that iterative solvers cannot:
- Verify well-constrained status (unique solution within tolerance).
- Detect over-constrained configurations (no solution exists).
- Bound the solution sensitivity to parameter perturbations.

---

## Summary: Most Promising Techniques for Solverang

### Immediate Priorities (Next Implementation Phase)

1. **Hessian computation via symbolic AD**: Extend `#[auto_jacobian]` to
   `#[auto_hessian]`. The infrastructure (proc macro, opcode emitter,
   Cranelift backend) already exists; it needs to emit second-derivative
   opcodes. This unlocks full Newton SQP and interior-point methods.

2. **SQP with OSQP/Clarabel QP subproblems**: Implement an SQP outer loop
   that uses existing Rust conic/QP solvers for the subproblems. This
   avoids reimplementing QP solvers from scratch. The Uno-style modular
   architecture guides the design.

3. **Interior-point method**: Implement a filter line-search IPM following
   IPOPT's proven approach, using faer for sparse KKT factorization. The
   filter mechanism is well-understood and robust.

4. **Augmented Lagrangian outer loop**: Simple to implement, robust for
   nonconvex problems, and can reuse the existing Newton/LM solvers as
   subproblem solvers. Good stepping stone before full SQP/IPM.

### Medium-Term Enhancements

5. **Manifold parameterizations for SO(3)/SE(3)**: Improve robustness and
   reduce problem dimension for 3D constraint systems.

6. **Variable projection for separable problems**: Automatically detect
   and exploit linear separability in constraint systems.

7. **Incremental warm-starting**: Sensitivity-based solution prediction for
   interactive CAD editing.

8. **ALADIN-style distributed optimization**: For coupled-component
   assemblies where independent parallel solving is insufficient.

### Long-Term Research Directions

9. **Enzyme `#[autodiff]` integration**: When stabilized in Rust, provide
   a second AD pathway for user-defined cost functions.

10. **GPU-accelerated solvers**: For topology/shape optimization with
    millions of variables. Evaluate Cranelift GPU backend or wgpu compute
    shaders.

11. **ML-guided algorithm selection**: Train a classifier on problem
    features (sparsity, decomposition structure, constraint types) to
    select the best solver configuration.

12. **Hybrid symbolic-numeric solving**: Integrate resultant/Groebner
    methods for polynomial subsystems with iterative methods for the
    general case.

---

## Reference Summary

### Solver Implementations
- IPOPT: https://github.com/coin-or/Ipopt
- SNOPT: https://ccom.ucsd.edu/~optimizers/solvers/snopt/
- Knitro: https://www.artelys.com/solvers/knitro/
- NLopt: https://github.com/stevengj/nlopt
- Ceres Solver: https://github.com/ceres-solver/ceres-solver
- CasADi: https://github.com/casadi/casadi
- Optim.jl: https://github.com/JuliaNLSolvers/Optim.jl
- scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Optax: https://github.com/google-deepmind/optax
- Clarabel.rs: https://github.com/oxfordcontrol/Clarabel.rs
- OSQP: https://github.com/osqp/osqp
- argmin: https://github.com/argmin-rs/argmin
- Uno: https://github.com/cvanaret/Uno
- faer: https://github.com/sarah-quinones/faer-rs
- HiGHS: https://github.com/ERGO-Code/HiGHS (highs crate)
- DiffSol: https://github.com/martinjrobins/diffsol

### Key Papers (2022-2025)
- Gondzio: "Interior point methods in the year 2025", EURO J. Comp. Opt., 2025.
- Kiessling, Leyffer, Vanaret: "A unified funnel restoration SQP algorithm", Math. Prog., 2025. arXiv:2409.09208.
- Vanaret, Leyffer: "Implementing a unified solver for nonlinearly constrained optimization", under review at Math. Prog. Comp. arXiv:2406.13454 (2024).
- Chen et al.: "CuClarabel: GPU Acceleration for a Conic Optimization Solver", arXiv:2412.19027, 2024.
- He, Jiang, Zhang: "Newton-CG based barrier-augmented Lagrangian method for nonconvex conic optimization", Comp. Opt. Appl., 2024.
- Houska, Frasch, Diehl: "An Augmented Lagrangian Based Algorithm for Distributed Non-Convex Optimization", SIAM J. Opt., 2016 (ALADIN foundation).
- Moses et al.: "Reverse-mode AD and optimization of GPU kernels via Enzyme", SC '21.
- "Sparse Variable Projection in Robotic Perception", arXiv:2512.07969, 2024.
- "OpenSQP: A Reconfigurable Open-Source SQP Algorithm", arXiv:2512.05392, 2024.
- Goulart, Chen: "Clarabel: An interior-point solver for conic programs with quadratic objectives", arXiv:2405.12762, 2024.
- Andersson et al.: "CasADi: a software framework for nonlinear optimization and optimal control", Math. Prog. Comp., 2019.
- Gill, Murray, Saunders: "SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization", SIAM Review 47(1), 2005.
- Wachter, Biegler: "On the implementation of an interior-point filter line-search algorithm for large-scale NLP", Math. Prog. 106(1), 2006.
