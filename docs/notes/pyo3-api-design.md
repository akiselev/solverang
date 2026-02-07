# PyO3 Python API Design Exploration for Solverang

## Overview

This document explores multiple API designs for exposing solverang's Rust solver
to Python via PyO3. The goal is a **pythonic, fast** binding where Python is a
thin layer over the Rust solver but provides an ergonomic API that feels native
to Python users.

### Key Design Tensions

1. **Rust traits vs Python duck-typing** -- `Problem` is a trait; Python has no
   equivalent. We must choose how users define problems.
2. **Const generics vs runtime dimensions** -- `ConstraintSystem<const D: usize>`
   cannot be generic in PyO3. We must monomorphize or use enum dispatch.
3. **Builder pattern ownership** -- Rust builders consume `self`; Python objects
   are reference-counted. These are fundamentally incompatible.
4. **GIL + callbacks** -- If residual/jacobian are Python callables, we can't
   release the GIL during solve. Pure-Rust problem types can release the GIL.
5. **Copy overhead** -- Every `Vec<f64>` crossing the boundary is a copy unless
   we use numpy arrays with `rust-numpy`.

### Crate Layout (Common to All Designs)

```
crates/
  solverang/          # existing pure-Rust library (unchanged)
  solverang-python/   # new crate with PyO3 bindings
    Cargo.toml        # cdylib, depends on solverang + pyo3 + numpy
    pyproject.toml    # maturin build config
    src/
      lib.rs          # #[pymodule] entry point
      problem.rs      # Problem wrappers
      solver.rs       # Solver wrappers
      result.rs       # Result/error types
      geometry.rs     # Geometry bindings
    solverang/        # Python package directory
      __init__.py     # Re-exports, pure-Python helpers
      _solverang.pyi  # Type stubs for IDE support
      py.typed        # PEP 561 marker
```

Build/publish:
```toml
# Cargo.toml
[lib]
name = "_solverang"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module", "abi3-py39"] }
numpy = "0.23"
solverang = { path = "../solverang", features = ["geometry", "parallel", "sparse"] }
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "solverang"
requires-python = ">=3.9"
dependencies = ["numpy>=1.20"]
```

---

## Design A: Callback-Based (Maximum Flexibility)

Users define problems by passing Python callables for residuals and jacobian.
This is the most flexible approach -- any Python code can define a problem.

### Python API

```python
import numpy as np
import solverang as sr

# Define a problem with callables
def residuals(x):
    return [x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]]

def jacobian(x):
    # Return sparse triplets: list of (row, col, value)
    return [
        (0, 0, 2*x[0]), (0, 1, 2*x[1]),
        (1, 0, 1.0),    (1, 1, -1.0),
    ]

problem = sr.Problem(
    residuals=residuals,
    jacobian=jacobian,
    num_residuals=2,
    num_variables=2,
    name="unit circle intersection",
)

# Solve with auto-selected solver
result = sr.solve(problem, x0=[0.5, 0.5])

print(result.solution)       # numpy array
print(result.converged)      # bool
print(result.iterations)     # int
print(result.residual_norm)  # float

# Or with explicit solver choice + config
result = sr.solve(
    problem,
    x0=[0.5, 0.5],
    solver="levenberg-marquardt",
    tolerance=1e-12,
    max_iterations=500,
)

# Jacobian-free (auto finite-difference)
problem = sr.Problem(
    residuals=residuals,
    num_residuals=2,
    num_variables=2,
)
```

### Rust Implementation

```rust
#[pyclass(frozen)]
struct PyProblem {
    name: String,
    residual_count: usize,
    variable_count: usize,
    residuals_fn: PyObject,
    jacobian_fn: Option<PyObject>,  // None => finite difference
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (*, residuals, num_residuals, num_variables, jacobian=None, name=None))]
    fn new(
        residuals: PyObject,
        num_residuals: usize,
        num_variables: usize,
        jacobian: Option<PyObject>,
        name: Option<String>,
    ) -> Self {
        Self {
            name: name.unwrap_or_else(|| "unnamed".into()),
            residual_count: num_residuals,
            variable_count: num_variables,
            residuals_fn: residuals,
            jacobian_fn: jacobian,
        }
    }
}

impl Problem for PyProblem {
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        Python::with_gil(|py| {
            let array = PyArray1::from_slice(py, x);
            self.residuals_fn
                .call1(py, (array,))
                .unwrap()
                .extract::<Vec<f64>>(py)
                .unwrap()
        })
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        match &self.jacobian_fn {
            Some(jac_fn) => Python::with_gil(|py| {
                let array = PyArray1::from_slice(py, x);
                jac_fn.call1(py, (array,))
                    .unwrap()
                    .extract::<Vec<(usize, usize, f64)>>(py)
                    .unwrap()
            }),
            None => finite_difference_jacobian(self, x),
        }
    }
    // ...
}

/// Top-level solve function
#[pyfunction]
#[pyo3(signature = (problem, x0, *, solver=None, tolerance=None, max_iterations=None))]
fn solve(
    py: Python<'_>,
    problem: &PyProblem,
    x0: Vec<f64>,
    solver: Option<&str>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PySolveResult> {
    // Cannot release GIL here -- residuals/jacobian are Python callables
    let result = match solver.unwrap_or("auto") {
        "newton-raphson" | "nr" => { /* ... */ },
        "levenberg-marquardt" | "lm" => { /* ... */ },
        "auto" => { /* ... */ },
        other => return Err(PyValueError::new_err(format!("unknown solver: {other}"))),
    };
    Ok(PySolveResult::from(result))
}
```

### Pros

- **Most flexible**: any Python function can define a problem
- **Familiar pattern**: similar to scipy.optimize APIs
- **Auto finite-difference**: users can omit jacobian for prototyping
- **Simple mental model**: just pass functions
- **Easy to integrate with existing Python code** (sympy, autograd, jax)

### Cons

- **Cannot release the GIL** during solve -- each residual/jacobian evaluation
  requires reacquiring the GIL to call back into Python. This means other Python
  threads are blocked during solve.
- **Per-call overhead**: every iteration crosses the Python/Rust boundary twice
  (residuals + jacobian). For problems that converge in 5-10 iterations this is
  negligible; for 500+ iterations with cheap residuals, overhead dominates.
- **No JIT acceleration**: Python callables cannot be JIT-compiled by solverang's
  Cranelift-based JIT.
- **Sparse triplet format for jacobian** is not the most natural Python API
  (users might expect a dense 2D array or scipy.sparse).

### Performance Characteristics

- Overhead per iteration: ~0.1-0.5ms for GIL acquire + Python call + extract
- For typical geometric constraint problems (10-50 iterations, <100 vars): overhead is <5% of total time
- For large problems (1000+ vars, 200+ iterations): overhead can be 20-50% of total time
- Cannot leverage rayon parallel solver with Python callbacks

---

## Design B: Pre-built Problem Types (Maximum Performance)

Instead of callbacks, expose specific problem types as Rust-native `#[pyclass]`
structs. Users compose problems from these pre-built types. The solve loop stays
entirely in Rust -- no GIL contention, no callback overhead.

### Python API

```python
import solverang as sr

# Geometry: build constraint system with fluent API
system = (sr.ConstraintSystem2D()
    .add_point(0.0, 0.0)           # p0
    .add_point(10.0, 0.0)          # p1
    .add_point(5.0, 1.0)           # p2
    .fix_point(0)
    .fix_point(1)
    .add_distance(0, 1, 10.0)
    .add_distance(1, 2, 8.0)
    .add_distance(2, 0, 6.0)
)

result = system.solve()  # entire solve in Rust, GIL released
print(result.points)     # [(0.0, 0.0), (10.0, 0.0), (x, y)]

# Generic problem from expression strings (compiled to native code via JIT)
problem = sr.ExpressionProblem(
    variables=["x", "y"],
    residuals=[
        "x^2 + y^2 - 1",
        "x - y",
    ],
)
result = sr.solve(problem, x0=[0.5, 0.5])

# Batch solve: solve many instances with different parameters
systems = [make_system(params) for params in parameter_sweep]
results = sr.solve_batch(systems)  # parallel, GIL released for entire batch
```

### Rust Implementation

```rust
/// 2D Constraint System -- fully Rust-native, no Python callbacks
#[pyclass]
struct PyConstraintSystem2D {
    points: Vec<[f64; 2]>,
    fixed: Vec<bool>,
    constraints: Vec<ConstraintSpec>,  // enum of all constraint types
    name: Option<String>,
}

#[derive(Clone)]
enum ConstraintSpec {
    Distance { p1: usize, p2: usize, target: f64 },
    Horizontal { p1: usize, p2: usize },
    Vertical { p1: usize, p2: usize },
    Angle { p1: usize, p2: usize, radians: f64 },
    Coincident { p1: usize, p2: usize },
    Parallel { l1s: usize, l1e: usize, l2s: usize, l2e: usize },
    Perpendicular { l1s: usize, l1e: usize, l2s: usize, l2e: usize },
    PointOnLine { point: usize, start: usize, end: usize },
    PointOnCircle { point: usize, center: usize, radius: f64 },
    Midpoint { mid: usize, start: usize, end: usize },
    EqualLength { l1s: usize, l1e: usize, l2s: usize, l2e: usize },
    // ... etc for all constraint types
}

#[pymethods]
impl PyConstraintSystem2D {
    #[new]
    fn new() -> Self { /* ... */ }

    fn add_point<'a>(mut slf: PyRefMut<'a, Self>, x: f64, y: f64) -> PyRefMut<'a, Self> {
        slf.points.push([x, y]);
        slf.fixed.push(false);
        slf
    }

    fn fix_point<'a>(mut slf: PyRefMut<'a, Self>, index: usize) -> PyRefMut<'a, Self> {
        slf.fixed[index] = true;
        slf
    }

    fn add_distance<'a>(
        mut slf: PyRefMut<'a, Self>, p1: usize, p2: usize, target: f64,
    ) -> PyRefMut<'a, Self> {
        slf.constraints.push(ConstraintSpec::Distance { p1, p2, target });
        slf
    }

    // ... more constraint methods

    fn solve(&self, py: Python<'_>) -> PyResult<PyGeometryResult> {
        // Build the Rust ConstraintSystem from stored specs
        let system = self.build_system()?;
        let initial = system.current_values();

        // Release GIL -- entire solve runs in pure Rust
        let result = py.allow_threads(|| {
            let solver = LMSolver::new(LMConfig::default());
            solver.solve(&system, &initial)
        });

        Ok(PyGeometryResult::from(result, &self.points))
    }
}
```

### Pros

- **Maximum performance**: entire solve loop in Rust, GIL released
- **Parallel-safe**: can use rayon, can run multiple solves concurrently from
  Python threads
- **JIT-compatible**: pre-built constraint types can implement `Lowerable` for
  Cranelift JIT compilation
- **Batch operations**: can solve thousands of systems in parallel with one Python call
- **Method chaining** works naturally with `PyRefMut`

### Cons

- **Less flexible**: users can only use constraint types we've pre-built
- **Cannot express arbitrary math** without the expression string approach
- **Larger Rust-side API surface**: every constraint type needs its own spec enum
  variant and Python method
- **Two-phase build**: users construct specs, then `solve()` converts to real
  Rust types. Validation happens late.
- **Expression-string approach** requires parsing and is error-prone compared to
  real Python code

### Performance Characteristics

- Zero per-iteration overhead from Python
- GIL fully released during solve
- For geometric problems: 10-100x faster than Design A for problems with many iterations
- Batch solve can saturate all CPU cores

---

## Design C: Hybrid (Recommended)

Combine Designs A and B: provide pre-built problem types for maximum performance,
but also accept Python callables for maximum flexibility. The user chooses their
tradeoff.

### Python API

```python
import numpy as np
import solverang as sr

# ─── Path 1: Pre-built geometry (fast, GIL-free) ───

system = sr.ConstraintSystem2D("Triangle")
p0 = system.add_point(0.0, 0.0, fixed=True)
p1 = system.add_point(10.0, 0.0, fixed=True)
p2 = system.add_point(5.0, 1.0)  # initial guess

system.constrain_distance(p0, p1, 10.0)
system.constrain_distance(p1, p2, 8.0)
system.constrain_distance(p2, p0, 6.0)

result = system.solve()           # GIL released, pure Rust
assert result.converged
print(result.points[2])           # (x, y) of the apex

# ─── Path 2: Custom problem with callables (flexible) ───

result = sr.solve(
    residuals=lambda x: [x[0]**2 - 2.0],
    x0=[1.0],
    # jacobian auto-computed via finite differences
)
print(result.x)  # ~[1.4142...]

# ─── Path 3: Numpy-native for larger problems ───

def rosenbrock_residuals(x):
    """Rosenbrock function as residual system."""
    r = np.empty(2 * (len(x) - 1))
    r[0::2] = 10.0 * (x[1:] - x[:-1]**2)
    r[1::2] = 1.0 - x[:-1]
    return r

result = sr.solve(
    residuals=rosenbrock_residuals,
    x0=np.zeros(100),
    solver="lm",
    tolerance=1e-10,
)

# ─── Configuration ───

config = sr.SolverConfig(
    tolerance=1e-12,
    max_iterations=500,
    solver="lm",        # or "nr", "auto", "robust"
)

result = sr.solve(problem, x0=x0, config=config)

# ─── Result API ───

result.x              # numpy array: solution vector
result.solution       # alias for result.x
result.converged      # bool
result.iterations     # int
result.residual_norm  # float
result.success        # alias for converged (scipy compat)

# Raise on failure (like requests.raise_for_status())
result.raise_on_failure()  # raises sr.SolverError if not converged

# Result is truthy when converged
if result:
    print("solved!")
```

### Rust Implementation

```rust
// ─── Module Structure ───

#[pymodule]
mod _solverang {
    #[pymodule_export]
    use super::{
        PyConstraintSystem2D, PyConstraintSystem3D,
        PySolveResult, PySolverConfig,
        solve, solve_system,
    };

    #[pymodule]
    mod exceptions {
        use super::*;
        // Custom exception hierarchy
    }
}

// ─── Result Type ───

#[pyclass(frozen, name = "SolveResult")]
struct PySolveResult {
    solution: Vec<f64>,
    converged: bool,
    iterations: usize,
    residual_norm: f64,
    error_message: Option<String>,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.solution)
    }

    #[getter]
    fn solution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.x(py)
    }

    #[getter]
    fn converged(&self) -> bool { self.converged }

    #[getter]
    fn success(&self) -> bool { self.converged }  // scipy compat

    #[getter]
    fn iterations(&self) -> usize { self.iterations }

    #[getter]
    fn residual_norm(&self) -> f64 { self.residual_norm }

    fn raise_on_failure(&self) -> PyResult<()> {
        if !self.converged {
            let msg = self.error_message.as_deref()
                .unwrap_or("solver did not converge");
            Err(SolverError::new_err(msg))
        } else {
            Ok(())
        }
    }

    fn __bool__(&self) -> bool { self.converged }

    fn __repr__(&self) -> String {
        if self.converged {
            format!("SolveResult(converged=True, iterations={}, residual_norm={:.2e})",
                    self.iterations, self.residual_norm)
        } else {
            format!("SolveResult(converged=False, iterations={}, residual_norm={:.2e})",
                    self.iterations, self.residual_norm)
        }
    }
}

impl From<SolveResult> for PySolveResult {
    fn from(r: SolveResult) -> Self {
        match r {
            SolveResult::Converged { solution, iterations, residual_norm } =>
                Self { solution, converged: true, iterations, residual_norm,
                       error_message: None },
            SolveResult::NotConverged { solution, iterations, residual_norm } =>
                Self { solution, converged: false, iterations, residual_norm,
                       error_message: Some("max iterations exceeded".into()) },
            SolveResult::Failed { error } =>
                Self { solution: vec![], converged: false, iterations: 0,
                       residual_norm: f64::NAN,
                       error_message: Some(error.to_string()) },
        }
    }
}

// ─── Top-level solve() ───

#[pyfunction]
#[pyo3(signature = (
    residuals=None, x0=None, *,
    jacobian=None, num_residuals=None, num_variables=None,
    problem=None, solver=None, config=None,
    tolerance=None, max_iterations=None,
))]
fn solve(
    py: Python<'_>,
    residuals: Option<PyObject>,
    x0: Option<Vec<f64>>,
    jacobian: Option<PyObject>,
    num_residuals: Option<usize>,
    num_variables: Option<usize>,
    problem: Option<&PyProblem>,
    solver: Option<&str>,
    config: Option<&PySolverConfig>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PySolveResult> {
    // Build problem from either explicit Problem or callables
    // Dispatch to appropriate solver
    // ...
}
```

### Geometry API (Imperative Style)

```rust
#[pyclass(name = "ConstraintSystem2D")]
struct PyConstraintSystem2D {
    name: String,
    points: Vec<[f64; 2]>,
    fixed: Vec<bool>,
    constraints: Vec<ConstraintSpec>,
}

#[pymethods]
impl PyConstraintSystem2D {
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(name: Option<String>) -> Self {
        Self {
            name: name.unwrap_or_else(|| "unnamed".into()),
            points: Vec::new(),
            fixed: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a point, return its index. Optionally fix it.
    #[pyo3(signature = (x, y, *, fixed=false))]
    fn add_point(&mut self, x: f64, y: f64, fixed: bool) -> usize {
        let idx = self.points.len();
        self.points.push([x, y]);
        self.fixed.push(fixed);
        idx
    }

    fn fix_point(&mut self, index: usize) { self.fixed[index] = true; }

    fn constrain_distance(&mut self, p1: usize, p2: usize, distance: f64) {
        self.constraints.push(ConstraintSpec::Distance { p1, p2, target: distance });
    }

    fn constrain_horizontal(&mut self, p1: usize, p2: usize) {
        self.constraints.push(ConstraintSpec::Horizontal { p1, p2 });
    }

    fn constrain_vertical(&mut self, p1: usize, p2: usize) {
        self.constraints.push(ConstraintSpec::Vertical { p1, p2 });
    }

    fn constrain_angle(&mut self, p1: usize, p2: usize, degrees: f64) {
        self.constraints.push(ConstraintSpec::Angle {
            p1, p2, radians: degrees.to_radians()
        });
    }

    fn constrain_perpendicular(&mut self, l1_start: usize, l1_end: usize,
                                l2_start: usize, l2_end: usize) {
        self.constraints.push(ConstraintSpec::Perpendicular {
            l1s: l1_start, l1e: l1_end, l2s: l2_start, l2e: l2_end
        });
    }

    fn constrain_parallel(&mut self, l1_start: usize, l1_end: usize,
                           l2_start: usize, l2_end: usize) {
        self.constraints.push(ConstraintSpec::Parallel {
            l1s: l1_start, l1e: l1_end, l2s: l2_start, l2e: l2_end
        });
    }

    fn constrain_coincident(&mut self, p1: usize, p2: usize) {
        self.constraints.push(ConstraintSpec::Coincident { p1, p2 });
    }

    fn constrain_midpoint(&mut self, mid: usize, start: usize, end: usize) {
        self.constraints.push(ConstraintSpec::Midpoint { mid, start, end });
    }

    fn constrain_point_on_line(&mut self, point: usize, start: usize, end: usize) {
        self.constraints.push(ConstraintSpec::PointOnLine { point, start, end });
    }

    fn constrain_point_on_circle(&mut self, point: usize, center: usize, radius: f64) {
        self.constraints.push(ConstraintSpec::PointOnCircle { point, center, radius });
    }

    fn constrain_equal_length(&mut self, l1_start: usize, l1_end: usize,
                               l2_start: usize, l2_end: usize) {
        self.constraints.push(ConstraintSpec::EqualLength {
            l1s: l1_start, l1e: l1_end, l2s: l2_start, l2e: l2_end
        });
    }

    // ─── Informational ───

    #[getter]
    fn num_points(&self) -> usize { self.points.len() }

    #[getter]
    fn num_constraints(&self) -> usize { self.constraints.len() }

    #[getter]
    fn degrees_of_freedom(&self) -> isize {
        let free_vars: usize = self.fixed.iter()
            .filter(|&&f| !f).count() * 2;  // 2 coords per free point
        free_vars as isize - self.constraints.len() as isize
    }

    // ─── Solve ───

    #[pyo3(signature = (*, solver=None, tolerance=None, max_iterations=None))]
    fn solve(
        &self, py: Python<'_>,
        solver: Option<&str>, tolerance: Option<f64>, max_iterations: Option<usize>,
    ) -> PyResult<PyGeometryResult> {
        let system = self.build_rust_system()?;
        let initial = system.current_values();

        // GIL released -- pure Rust solve
        let result = py.allow_threads(|| {
            let solver = LMSolver::new(LMConfig::default());
            solver.solve(&system, &initial)
        });

        PyGeometryResult::from_solve(result, &self.points)
    }

    fn __repr__(&self) -> String {
        format!("ConstraintSystem2D('{}', points={}, constraints={}, dof={})",
                self.name, self.points.len(), self.constraints.len(),
                self.degrees_of_freedom())
    }
}
```

### Pros

- **Best of both worlds**: fast path for pre-built types, flexible path for custom problems
- **Users choose their tradeoff**: prototype with callbacks, deploy with pre-built types
- **Geometry API is fully GIL-free** when solving
- **Familiar to scipy users** (callback path) and CAD users (geometry path)
- **Single `solve()` entry point** with multiple dispatch based on arguments

### Cons

- **Larger API surface** than either A or B alone
- **Two different mental models** for defining problems
- **`solve()` function has many optional parameters** -- could be confusing
- **Imperative geometry API loses Rust builder's fluent chaining** (methods return
  `None` in Python, not `self`)

---

## Design D: Protocol-Based (Most Pythonic)

Use Python protocols (duck-typing via `__dunder__` methods) instead of explicit
classes. Any Python object that implements the right methods can be used as a
problem. This is the most "Pythonic" approach.

### Python API

```python
import numpy as np
import solverang as sr

# Any object with the right methods works as a "problem"
class CircleIntersection:
    """Find where unit circle meets y=x."""

    @property
    def num_variables(self) -> int:
        return 2

    @property
    def num_residuals(self) -> int:
        return 2

    def residuals(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])

    def jacobian(self, x: np.ndarray) -> list[tuple[int, int, float]]:
        return [(0, 0, 2*x[0]), (0, 1, 2*x[1]), (1, 0, 1.0), (1, 1, -1.0)]

result = sr.solve(CircleIntersection(), x0=[0.5, 0.5])

# Also works with a simple dataclass
from dataclasses import dataclass

@dataclass
class QuadraticProblem:
    target: float

    @property
    def num_variables(self): return 1

    @property
    def num_residuals(self): return 1

    def residuals(self, x):
        return [x[0]**2 - self.target]

result = sr.solve(QuadraticProblem(target=2.0), x0=[1.0])
print(result.x[0])  # ~1.4142

# Dict-based for quick one-offs
result = sr.solve({
    "residuals": lambda x: [x[0]**2 - 2.0],
    "num_variables": 1,
    "num_residuals": 1,
}, x0=[1.0])

# The geometry API also uses protocols for custom constraints
class MyCustomConstraint:
    """Custom constraint: sum of coordinates = target."""
    def __init__(self, points, target):
        self.points = points
        self.target = target

    @property
    def num_residuals(self):
        return 1

    def residuals(self, all_coords):
        total = sum(all_coords[p*2] + all_coords[p*2+1] for p in self.points)
        return [total - self.target]

    def jacobian(self, all_coords):
        return [(0, p*2, 1.0) for p in self.points] + \
               [(0, p*2+1, 1.0) for p in self.points]

system = sr.ConstraintSystem2D("custom")
system.add_point(0.0, 0.0)
system.add_point(5.0, 5.0)
system.add_custom_constraint(MyCustomConstraint([0, 1], target=10.0))
```

### Rust Implementation

```rust
/// Extract a Problem from any Python object that implements the protocol
fn extract_problem(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<DynPyProblem> {
    // Check if it's a dict
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return DynPyProblem::from_dict(py, dict);
    }

    // Check for required attributes/methods
    let residuals_fn = obj.getattr("residuals")
        .map_err(|_| PyTypeError::new_err(
            "problem must have a 'residuals' method"
        ))?;

    let num_residuals: usize = obj.getattr("num_residuals")
        .and_then(|attr| attr.extract())
        .map_err(|_| PyTypeError::new_err(
            "problem must have a 'num_residuals' property"
        ))?;

    let num_variables: usize = obj.getattr("num_variables")
        .and_then(|attr| attr.extract())
        .map_err(|_| PyTypeError::new_err(
            "problem must have a 'num_variables' property"
        ))?;

    let jacobian_fn = obj.getattr("jacobian").ok();

    let name = obj.getattr("name")
        .and_then(|attr| attr.extract::<String>())
        .unwrap_or_else(|_| obj.get_type().name().unwrap_or("unnamed").into());

    Ok(DynPyProblem {
        name,
        residual_count: num_residuals,
        variable_count: num_variables,
        residuals_fn: residuals_fn.unbind(),
        jacobian_fn: jacobian_fn.map(|f| f.unbind()),
    })
}

#[pyfunction]
#[pyo3(signature = (problem, x0, **kwargs))]
fn solve(
    py: Python<'_>,
    problem: &Bound<'_, PyAny>,  // Accept any Python object
    x0: Vec<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySolveResult> {
    let problem = extract_problem(py, problem)?;
    // ... solve with extracted problem
}
```

### Pros

- **Most Pythonic**: follows Python's "duck typing" philosophy
- **No inheritance required**: any object with the right shape works
- **Dict-based problems** for quick prototyping in REPL/notebooks
- **Custom constraints** can be Python objects mixed with built-in Rust constraints
- **Familiar to numpy/scipy users** who expect protocol-based APIs

### Cons

- **Same GIL limitations** as Design A for custom problems
- **Runtime type errors** instead of construction-time errors (no static checking)
- **Harder to document**: "implement these methods" is less discoverable than
  "inherit from this class"
- **Protocol extraction overhead** on each `solve()` call (minor, one-time)
- **Mixed Rust/Python constraints** requires careful interop for the geometry path

---

## Design E: Dataclass + Decorator (Most Concise)

Use Python decorators and dataclass-like patterns to minimize boilerplate. This
is the "magic" approach -- concise but potentially surprising.

### Python API

```python
import solverang as sr
import numpy as np

# Decorator-based problem definition
@sr.problem(variables=["x", "y"])
def circle_line(x, y):
    """Find intersection of unit circle and y=x."""
    return [
        x**2 + y**2 - 1,
        x - y,
    ]

result = circle_line.solve(x0=[0.5, 0.5])

# With explicit jacobian
@sr.problem(variables=["x", "y"])
def circle_line(x, y):
    return [x**2 + y**2 - 1, x - y]

@circle_line.jacobian
def circle_line_jac(x, y):
    return [
        [2*x, 2*y],
        [1.0, -1.0],
    ]

result = circle_line.solve(x0=[0.5, 0.5])

# Parametric problems via classes
@sr.problem
class Rosenbrock:
    a: float = 1.0
    b: float = 100.0

    def residuals(self, x, y):
        return [self.a - x, self.b * (y - x**2)]

problem = Rosenbrock(a=1.0, b=100.0)
result = problem.solve(x0=[0.0, 0.0])

# Even more concise: expression strings
result = sr.solve_equations(
    ["x^2 + y^2 = 1", "x = y"],
    x0={"x": 0.5, "y": 0.5},
)

# Geometry with context manager
with sr.Sketch("Rectangle") as s:
    p0 = s.point(0, 0, fixed=True)
    p1 = s.point(10, 0)
    p2 = s.point(10, 5)
    p3 = s.point(0, 5)

    s.horizontal(p0, p1)
    s.vertical(p1, p2)
    s.horizontal(p2, p3)
    s.vertical(p3, p0)
    s.distance(p0, p1, 10.0)
    s.distance(p1, p2, 5.0)

result = s.solve()
```

### Rust + Python Implementation

The `@sr.problem` decorator would be **pure Python** wrapping the Rust bindings:

```python
# solverang/__init__.py

from ._solverang import _solve, SolveResult, ConstraintSystem2D

import inspect
from functools import wraps

def problem(fn=None, *, variables=None):
    """Decorator to create a solvable problem from a function."""
    def decorator(fn):
        sig = inspect.signature(fn)
        var_names = variables or list(sig.parameters.keys())
        n = len(var_names)

        class WrappedProblem:
            @property
            def num_variables(self):
                return n

            @property
            def num_residuals(self):
                # Call once with dummy to determine output size
                dummy = [0.0] * n
                return len(fn(*dummy))

            def residuals(self, x):
                return fn(*x[:n])

            def solve(self, x0, **kwargs):
                return _solve(self, x0, **kwargs)

        wrapper = WrappedProblem()
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
```

### Pros

- **Minimal boilerplate**: defining a problem is just writing a function
- **Pythonic naming convention**: `x0={"x": 0.5}` with named variables
- **Decorator pattern** is familiar to Flask/FastAPI/pytest users
- **Context manager sketch** is natural for imperative geometry construction
- **Expression strings** could be JIT-compiled for zero Python overhead

### Cons

- **"Magic" behavior** can be confusing -- function becomes an object
- **Decorator introspection** to determine `num_residuals` requires a dummy call
- **Variable unpacking** (`fn(*x[:n])`) adds overhead vs passing a slice
- **Two implementations**: decorators in Python, solve loop in Rust. More places
  for bugs.
- **Expression string parsing** is a whole new feature (potential security concern
  if not sandboxed)
- **Naming collisions**: `solve` is both a top-level function and a method on
  decorated problems

---

## Comparison Matrix

| Criterion                     | A: Callback | B: Pre-built | C: Hybrid | D: Protocol | E: Decorator |
|-------------------------------|:-----------:|:------------:|:---------:|:-----------:|:------------:|
| **Performance (GIL-free)**    | No          | Yes          | Both      | No          | No*          |
| **Custom problems**           | Yes         | No           | Yes       | Yes         | Yes          |
| **Geometry support**          | Manual      | Native       | Native    | Native      | Native       |
| **API surface size**          | Small       | Large        | Medium    | Small       | Small        |
| **Discoverability**           | Good        | Good         | Good      | Fair        | Fair         |
| **scipy familiarity**         | High        | Low          | High      | High        | Medium       |
| **Boilerplate**               | Low         | Low          | Low       | Lowest      | Lowest       |
| **Type safety**               | Medium      | High         | Medium    | Low         | Low          |
| **JIT potential**             | No          | Yes          | Partial   | No          | Strings only |
| **Batch/parallel solve**      | No          | Yes          | Partial   | No          | No           |
| **Publish complexity**        | Low         | Medium       | Medium    | Low         | Low          |

*Design E with expression strings could be GIL-free if compiled to native code.

---

## Recommendation: Design C (Hybrid) with Protocol Extraction from D

The recommended approach combines the **hybrid architecture of Design C** with
the **protocol-based problem extraction from Design D** and the **convenience
decorators from Design E** as a pure-Python layer.

### Layered Architecture

```
Layer 3: Pure Python convenience (decorators, context managers)
  │  solverang/__init__.py
  │  @sr.problem decorator, Sketch context manager
  │
Layer 2: PyO3 bindings (thin, fast)
  │  solverang/_solverang.so
  │  PyConstraintSystem2D/3D, solve(), PySolveResult
  │
Layer 1: Rust solver (unchanged)
     solverang crate
     Problem trait, all solvers, geometry, JIT
```

### Core Principles

1. **`solve()` accepts anything problem-shaped** (protocol extraction from D).
   A `PyProblem`, a dict, or any object with `residuals`/`num_variables`/
   `num_residuals`.

2. **Pre-built geometry types are first-class** and release the GIL. These are
   the performance path.

3. **The `@sr.problem` decorator** is pure Python sugar -- it creates an object
   that satisfies the protocol. Zero Rust complexity for this feature.

4. **Result objects are rich** with numpy arrays, truthiness, `raise_on_failure()`,
   and good `__repr__`.

5. **Config uses keyword arguments** on `solve()` for simple cases and a
   `SolverConfig` object for advanced cases. No need to pre-create config objects
   for common usage.

### Minimal Initial Scope

For a first release, implement only:

1. `solve(residuals=..., x0=...)` -- callback path
2. `solve(problem, x0=...)` -- protocol path
3. `ConstraintSystem2D` -- geometry with all 2D constraints
4. `ConstraintSystem3D` -- geometry with all 3D constraints
5. `SolveResult` -- rich result type
6. `SolverConfig` -- optional config object
7. Custom exceptions: `SolverError`, `ConvergenceError`, `DimensionError`

Defer to later:
- `@sr.problem` decorator (pure Python, can add without Rust changes)
- Expression string problems
- JIT compilation from Python
- Batch/parallel solve from Python
- Inequality constraints

---

## Technical Details for Implementation

### Error Handling Strategy

```rust
use pyo3::create_exception;

// Exception hierarchy
create_exception!(_solverang, SolverError, pyo3::exceptions::PyException);
create_exception!(_solverang, ConvergenceError, SolverError);
create_exception!(_solverang, DimensionError, SolverError);
create_exception!(_solverang, SingularJacobianError, SolverError);

impl From<SolveError> for PyErr {
    fn from(err: SolveError) -> PyErr {
        match err {
            SolveError::SingularJacobian =>
                SingularJacobianError::new_err(err.to_string()),
            SolveError::DimensionMismatch { .. } =>
                DimensionError::new_err(err.to_string()),
            SolveError::MaxIterationsExceeded(_) =>
                ConvergenceError::new_err(err.to_string()),
            SolveError::NoEquations | SolveError::NoVariables =>
                DimensionError::new_err(err.to_string()),
            _ => SolverError::new_err(err.to_string()),
        }
    }
}
```

### Const Generic Handling (2D/3D)

Use enum dispatch internally, expose as separate Python types:

```rust
// Internal enum for runtime dimension dispatch
enum ConstraintSystemInner {
    TwoD(solverang::geometry::ConstraintSystem<2>),
    ThreeD(solverang::geometry::ConstraintSystem<3>),
}

// But expose as separate Python classes for clear API
#[pyclass(name = "ConstraintSystem2D")]
struct PyConstraintSystem2D { /* ... */ }

#[pyclass(name = "ConstraintSystem3D")]
struct PyConstraintSystem3D { /* ... */ }

// Shared implementation via macro to avoid duplication
macro_rules! impl_constraint_system {
    ($py_name:ident, $dim:literal, $point_type:ident) => {
        #[pymethods]
        impl $py_name {
            // ... shared methods
        }
    }
}

impl_constraint_system!(PyConstraintSystem2D, 2, Point2D);
impl_constraint_system!(PyConstraintSystem3D, 3, Point3D);
```

### Numpy Integration

```rust
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, IntoPyArray};

#[pymethods]
impl PySolveResult {
    /// Solution as numpy array (zero-copy when possible)
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        // This copies since we own the Vec -- but it's a single copy at the end,
        // not per-iteration
        PyArray1::from_vec(py, self.solution.clone())
    }
}

// Accept numpy arrays as input (zero-copy read)
#[pyfunction]
fn solve(
    py: Python<'_>,
    problem: &Bound<'_, PyAny>,
    x0: PyReadonlyArray1<'_, f64>,  // zero-copy from numpy
) -> PyResult<PySolveResult> {
    let x0_slice = x0.as_slice()?;
    // ...
}
```

### Thread Safety Design

```rust
// Frozen classes for thread safety (recommended for free-threaded Python 3.13+)
#[pyclass(frozen)]
struct PySolveResult { /* immutable fields */ }

// Mutable geometry system uses interior mutability
#[pyclass]
struct PyConstraintSystem2D {
    // No Mutex needed -- PyO3 handles &mut self borrow checking at runtime
    points: Vec<[f64; 2]>,
    fixed: Vec<bool>,
    constraints: Vec<ConstraintSpec>,
}
```

### Publishing

```bash
# Development
cd crates/solverang-python
maturin develop --release

# Build wheels for distribution
maturin build --release

# Cross-platform CI with GitHub Actions
# Use PyO3/maturin-action for automated wheel builds

# Publish
maturin publish  # or: uv publish target/wheels/*
```

### Type Stubs (`.pyi`)

```python
# solverang/_solverang.pyi
from typing import Optional, Sequence, Union
import numpy as np
import numpy.typing as npt

class SolveResult:
    @property
    def x(self) -> npt.NDArray[np.float64]: ...
    @property
    def solution(self) -> npt.NDArray[np.float64]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def success(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual_norm(self) -> float: ...
    def raise_on_failure(self) -> None: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...

class SolverConfig:
    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 200,
        solver: str = "auto",
    ) -> None: ...

class ConstraintSystem2D:
    def __init__(self, name: Optional[str] = None) -> None: ...
    def add_point(self, x: float, y: float, *, fixed: bool = False) -> int: ...
    def fix_point(self, index: int) -> None: ...
    def constrain_distance(self, p1: int, p2: int, distance: float) -> None: ...
    def constrain_horizontal(self, p1: int, p2: int) -> None: ...
    def constrain_vertical(self, p1: int, p2: int) -> None: ...
    def constrain_angle(self, p1: int, p2: int, degrees: float) -> None: ...
    def constrain_parallel(self, l1_start: int, l1_end: int,
                           l2_start: int, l2_end: int) -> None: ...
    def constrain_perpendicular(self, l1_start: int, l1_end: int,
                                l2_start: int, l2_end: int) -> None: ...
    def constrain_coincident(self, p1: int, p2: int) -> None: ...
    def constrain_midpoint(self, mid: int, start: int, end: int) -> None: ...
    def constrain_point_on_line(self, point: int, start: int, end: int) -> None: ...
    def constrain_point_on_circle(self, point: int, center: int, radius: float) -> None: ...
    def constrain_equal_length(self, l1_start: int, l1_end: int,
                                l2_start: int, l2_end: int) -> None: ...
    @property
    def num_points(self) -> int: ...
    @property
    def num_constraints(self) -> int: ...
    @property
    def degrees_of_freedom(self) -> int: ...
    def solve(
        self,
        *,
        solver: Optional[str] = None,
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> SolveResult: ...
    def __repr__(self) -> str: ...

def solve(
    problem: object = ...,
    x0: Union[Sequence[float], npt.NDArray[np.float64]] = ...,
    *,
    residuals: Optional[object] = None,
    jacobian: Optional[object] = None,
    num_residuals: Optional[int] = None,
    num_variables: Optional[int] = None,
    solver: Optional[str] = None,
    config: Optional[SolverConfig] = None,
    tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
) -> SolveResult: ...

class SolverError(Exception): ...
class ConvergenceError(SolverError): ...
class DimensionError(SolverError): ...
class SingularJacobianError(SolverError): ...
```

---

## Open Questions

1. **Should `solve()` accept both positional `residuals` and protocol objects?**
   Having one function do both is convenient but complex. Alternative: two
   functions `solve(problem, x0)` and `solve_function(residuals, x0)`.

2. **Should geometry results return point objects or tuples?** Returning
   `[(0.0, 0.0), (10.0, 0.0)]` is simple; returning `Point2D` objects with
   methods is richer but heavier.

3. **How to handle the jacobian format?** Sparse triplets `[(row, col, val)]`
   are efficient but unfamiliar. Dense `list[list[float]]` is familiar but
   wasteful. Support both and auto-detect?

4. **Should we support scipy.sparse matrices** for jacobian input/output?
   This would add a runtime dependency but would be familiar to scientific
   Python users.

5. **Version 1 scope**: Is the geometry path enough for v1, or do we need the
   callback path from day one? The geometry path is simpler to implement and
   doesn't have the GIL complications.
