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

## Design F: Expression Graph via Operator Overloading (Best of All Worlds)

Python's operator overloading (`__add__`, `__mul__`, `__pow__`, etc.) builds a
**Rust-side expression tree** -- not a Python AST, but actual Rust `Expr` nodes
stored in `#[pyclass]` objects. When `solve()` is called, the tree is:

1. **Symbolically differentiated** to compute the Jacobian automatically
2. **Lowered** to `ConstraintOp` opcodes via `OpcodeEmitter`
3. **JIT-compiled** to native code via Cranelift
4. **Solved entirely in Rust** with the GIL released

This is the same approach used by SymPy, PyTorch, JAX, and TensorFlow: Python
code *describes* computation but never *executes* it. All actual math happens in
Rust/native code.

**Key insight**: solverang already has all the infrastructure for this:
- `Expr` enum in the macro crate (with symbolic differentiation)
- `ConstraintOp` opcodes + `OpcodeEmitter` (register-based IR)
- `Lowerable` trait (expression → opcodes)
- `JITCompiler` (opcodes → native code via Cranelift)

We just need a **runtime** `Expr` type (the macro crate's is compile-time only)
and PyO3 operator overloads to construct it from Python.

### Python API

```python
import solverang as sr

# ─── Create symbolic variables ───
x, y = sr.variables("x y")
# or: x, y = sr.variables(2)
# or: x = sr.Variable("x"); y = sr.Variable("y")

# ─── Build expressions with normal Python operators ───
# These DO NOT compute anything -- they build a Rust-side expression graph
r1 = x**2 + y**2 - 1.0       # unit circle
r2 = x - y                    # line y = x

# ─── Solve: expression tree → differentiate → JIT compile → solve ───
result = sr.solve(
    residuals=[r1, r2],
    x0=[0.5, 0.5],
)
# Jacobian is computed automatically via symbolic differentiation.
# Entire solve runs in Rust with GIL released.
# JIT-compiled to native code -- no Python callbacks at all.

print(result.x)  # [0.7071..., 0.7071...]

# ─── Math functions ───
r = sr.sqrt(x**2 + y**2) - 1.0     # module-level functions
r = (x**2 + y**2).sqrt() - 1.0     # or method syntax

# Full set: sqrt, sin, cos, tan, atan2, abs, pow
r = sr.sin(x) * sr.cos(y)
r = sr.atan2(y, x) - 0.7854
r = abs(x - y)                      # Python's abs() works too

# ─── Expressions are inspectable ───
print(r1)           # "x**2 + y**2 - 1"
print(r1.diff(x))   # "2*x"  (symbolic derivative)
print(r1.variables)  # [Variable("x"), Variable("y")]

# ─── Constants and parameters ───
a = sr.Parameter("a", value=1.0)    # named constant, can be changed
b = sr.Parameter("b", value=100.0)

# Rosenbrock function
r1 = a - x
r2 = b * (y - x**2)

result = sr.solve(residuals=[r1, r2], x0=[0.0, 0.0])

# Change parameter and re-solve (re-uses JIT-compiled code if structure unchanged)
a.value = 2.0
result = sr.solve(residuals=[r1, r2], x0=[0.0, 0.0])

# ─── Works with geometry system too ───
system = sr.ConstraintSystem2D("custom")
p0 = system.add_point(0.0, 0.0, fixed=True)
p1 = system.add_point(5.0, 5.0)

# Access point coordinates as symbolic expressions
px, py = system.coords(p1)  # returns (Expr, Expr) bound to point 1

# Add a custom expression-based constraint alongside built-in ones
system.constrain_distance(p0, p1, 7.0)
system.add_residual(px + py - 10.0)  # custom: x1 + y1 = 10

result = system.solve()

# ─── Vectorized operations ───
xs = sr.variables("x", count=100)  # x_0, x_1, ..., x_99
residuals = []
for i in range(99):
    residuals.append(10.0 * (xs[i+1] - xs[i]**2))  # Rosenbrock
    residuals.append(1.0 - xs[i])

result = sr.solve(residuals=residuals, x0=[0.0]*100)

# ─── Equation syntax sugar ───
x, y = sr.variables("x y")

# Use == to create a residual (lhs - rhs = 0)
eq1 = sr.eq(x**2 + y**2, 1.0)    # x^2 + y^2 - 1 = 0
eq2 = sr.eq(x, y)                 # x - y = 0

result = sr.solve(equations=[eq1, eq2], x0=[0.5, 0.5])
```

### Rust Data Structures

```rust
/// Runtime expression tree -- mirrors the macro crate's Expr but is
/// constructable at runtime from Python operator overloads.
///
/// This lives in the main solverang crate (not the macro crate) so it
/// can implement Lowerable and integrate with the JIT pipeline.
#[derive(Clone, Debug)]
pub enum RuntimeExpr {
    Var(u32),                                    // variable index
    Const(f64),                                  // literal constant
    Param { id: u32, value: Arc<AtomicU64> },    // mutable parameter (shared)
    Neg(Box<RuntimeExpr>),
    Add(Box<RuntimeExpr>, Box<RuntimeExpr>),
    Sub(Box<RuntimeExpr>, Box<RuntimeExpr>),
    Mul(Box<RuntimeExpr>, Box<RuntimeExpr>),
    Div(Box<RuntimeExpr>, Box<RuntimeExpr>),
    Pow(Box<RuntimeExpr>, f64),                  // constant exponent
    Sqrt(Box<RuntimeExpr>),
    Sin(Box<RuntimeExpr>),
    Cos(Box<RuntimeExpr>),
    Tan(Box<RuntimeExpr>),
    Atan2(Box<RuntimeExpr>, Box<RuntimeExpr>),
    Abs(Box<RuntimeExpr>),
}

impl RuntimeExpr {
    /// Symbolic differentiation with respect to variable `var_idx`.
    /// Reuses the same algorithm as the macro crate's Expr::differentiate.
    pub fn differentiate(&self, var_idx: u32) -> RuntimeExpr { /* ... */ }

    /// Algebraic simplification (constant folding, identity elimination).
    pub fn simplify(&self) -> RuntimeExpr { /* ... */ }

    /// Collect all variable indices referenced in this expression.
    pub fn variables(&self) -> BTreeSet<u32> { /* ... */ }

    /// Lower this expression to ConstraintOp opcodes.
    pub fn emit(&self, emitter: &mut OpcodeEmitter) -> Reg { /* ... */ }

    /// Evaluate directly (interpreted, no JIT). Useful for debugging.
    pub fn evaluate(&self, vars: &[f64]) -> f64 { /* ... */ }
}
```

### Lowering RuntimeExpr to Opcodes

```rust
impl RuntimeExpr {
    /// Recursively emit opcodes for this expression, returning the register
    /// holding the result.
    pub fn emit(&self, emitter: &mut OpcodeEmitter) -> Reg {
        match self {
            RuntimeExpr::Var(idx) => emitter.load_var(*idx),
            RuntimeExpr::Const(v) => emitter.const_f64(*v),
            RuntimeExpr::Param { value, .. } => {
                // Load parameter's current value as a constant
                let bits = value.load(Ordering::Relaxed);
                emitter.const_f64(f64::from_bits(bits))
            }
            RuntimeExpr::Neg(inner) => {
                let r = inner.emit(emitter);
                emitter.neg(r)
            }
            RuntimeExpr::Add(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.add(ra, rb)
            }
            RuntimeExpr::Sub(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.sub(ra, rb)
            }
            RuntimeExpr::Mul(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.mul(ra, rb)
            }
            RuntimeExpr::Div(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.div(ra, rb)
            }
            RuntimeExpr::Pow(base, exp) => {
                let rb = base.emit(emitter);
                if *exp == 2.0 {
                    emitter.square(rb)  // x^2 → mul(x, x)
                } else if *exp == 0.5 {
                    emitter.sqrt(rb)    // x^0.5 → sqrt(x)
                } else {
                    // General power: expand as exp(exp * ln(base))
                    // or handle specific integer cases
                    todo!("general power")
                }
            }
            RuntimeExpr::Sqrt(inner) => {
                let r = inner.emit(emitter);
                emitter.sqrt(r)
            }
            RuntimeExpr::Sin(inner) => {
                let r = inner.emit(emitter);
                emitter.sin(r)
            }
            RuntimeExpr::Cos(inner) => {
                let r = inner.emit(emitter);
                emitter.cos(r)
            }
            RuntimeExpr::Atan2(y, x) => {
                let ry = y.emit(emitter);
                let rx = x.emit(emitter);
                emitter.atan2(ry, rx)
            }
            RuntimeExpr::Abs(inner) => {
                let r = inner.emit(emitter);
                emitter.abs(r)
            }
            RuntimeExpr::Tan(inner) => {
                // tan(x) = sin(x) / cos(x)
                let r = inner.emit(emitter);
                let s = emitter.sin(r);
                let c = emitter.cos(r);
                emitter.div(s, c)
            }
        }
    }
}
```

### Problem Construction from Expression Graphs

```rust
/// A Problem built entirely from RuntimeExpr trees.
/// Implements Problem trait -- residuals and jacobians are computed via
/// JIT-compiled native code. No Python callbacks.
pub struct ExprProblem {
    name: String,
    num_vars: usize,
    residual_exprs: Vec<RuntimeExpr>,
    jacobian_exprs: Vec<Vec<(u32, RuntimeExpr)>>,  // sparse: (col, d_residual/d_var)
    // Optionally JIT-compiled for maximum performance:
    jit_fn: Option<JITFunction>,
}

impl ExprProblem {
    pub fn new(
        name: String,
        num_vars: usize,
        residuals: Vec<RuntimeExpr>,
    ) -> Self {
        // Auto-differentiate each residual w.r.t. each variable it references
        let jacobian_exprs: Vec<Vec<(u32, RuntimeExpr)>> = residuals.iter()
            .map(|r| {
                r.variables().into_iter()
                    .map(|var_idx| {
                        let deriv = r.differentiate(var_idx).simplify();
                        (var_idx, deriv)
                    })
                    .filter(|(_, d)| !matches!(d, RuntimeExpr::Const(v) if *v == 0.0))
                    .collect()
            })
            .collect();

        let mut problem = Self {
            name,
            num_vars,
            residual_exprs: residuals,
            jacobian_exprs,
            jit_fn: None,
        };

        // Try to JIT compile (falls back to interpreted if platform unsupported)
        problem.try_jit_compile();
        problem
    }

    fn try_jit_compile(&mut self) {
        if !jit_available() { return; }

        let mut emitter = OpcodeEmitter::new();

        // Emit residual opcodes
        for (i, expr) in self.residual_exprs.iter().enumerate() {
            let reg = expr.emit(&mut emitter);
            emitter.store_residual(i as u32, reg);
        }
        let residual_ops = emitter.take_ops();

        // Emit jacobian opcodes
        let mut emitter = OpcodeEmitter::new();
        let mut pattern = Vec::new();
        for (row, jac_row) in self.jacobian_exprs.iter().enumerate() {
            for (col, deriv_expr) in jac_row {
                let reg = deriv_expr.emit(&mut emitter);
                let idx = pattern.len() as u32;
                emitter.store_jacobian_indexed(idx, reg);
                pattern.push(JacobianEntry { row: row as u32, col: *col });
            }
        }
        let jacobian_ops = emitter.take_ops();

        let compiled = CompiledConstraints {
            residual_ops,
            jacobian_ops,
            n_residuals: self.residual_exprs.len(),
            n_vars: self.num_vars,
            jacobian_nnz: pattern.len(),
            jacobian_pattern: pattern,
            max_register: emitter.max_register(),
        };

        match JITCompiler::new().and_then(|c| c.compile(&compiled)) {
            Ok(jit_fn) => self.jit_fn = Some(jit_fn),
            Err(_) => {} // fall back to interpreted evaluation
        }
    }
}

impl Problem for ExprProblem {
    fn name(&self) -> &str { &self.name }
    fn residual_count(&self) -> usize { self.residual_exprs.len() }
    fn variable_count(&self) -> usize { self.num_vars }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        if let Some(ref jit) = self.jit_fn {
            let mut out = vec![0.0; self.residual_exprs.len()];
            unsafe { jit.evaluate_residuals(x, &mut out); }
            out
        } else {
            // Interpreted fallback
            self.residual_exprs.iter()
                .map(|expr| expr.evaluate(x))
                .collect()
        }
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        if let Some(ref jit) = self.jit_fn {
            let mut values = vec![0.0; self.jacobian_exprs.iter().map(|r| r.len()).sum()];
            unsafe { jit.evaluate_jacobian(x, &mut values); }
            jit.jacobian_to_coo(&values)
        } else {
            // Interpreted fallback
            let mut triplets = Vec::new();
            for (row, jac_row) in self.jacobian_exprs.iter().enumerate() {
                for (col, deriv) in jac_row {
                    let val = deriv.evaluate(x);
                    if val != 0.0 {
                        triplets.push((row, *col as usize, val));
                    }
                }
            }
            triplets
        }
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.num_vars]
    }
}
```

### PyO3 Bindings: The `Expr` PyClass

```rust
/// Python-visible expression node. Each instance wraps a Rust RuntimeExpr.
/// All operator overloads return new PyExpr instances (immutable expression DAG).
#[pyclass(frozen, name = "Expr")]
#[derive(Clone)]
struct PyExpr {
    inner: RuntimeExpr,
    name: Option<String>,  // for display: "x", "y", etc.
}

#[pymethods]
impl PyExpr {
    // ─── Arithmetic operators (build tree, don't compute) ───

    fn __add__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Add(
            Box::new(self.inner.clone()),
            Box::new(other.into_expr()),
        ), name: None }
    }

    fn __radd__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Add(
            Box::new(other.into_expr()),
            Box::new(self.inner.clone()),
        ), name: None }
    }

    fn __sub__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Sub(
            Box::new(self.inner.clone()),
            Box::new(other.into_expr()),
        ), name: None }
    }

    fn __rsub__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Sub(
            Box::new(other.into_expr()),
            Box::new(self.inner.clone()),
        ), name: None }
    }

    fn __mul__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Mul(
            Box::new(self.inner.clone()),
            Box::new(other.into_expr()),
        ), name: None }
    }

    fn __rmul__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Mul(
            Box::new(other.into_expr()),
            Box::new(self.inner.clone()),
        ), name: None }
    }

    fn __truediv__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Div(
            Box::new(self.inner.clone()),
            Box::new(other.into_expr()),
        ), name: None }
    }

    fn __rtruediv__(&self, other: ExprOrFloat) -> Self {
        PyExpr { inner: RuntimeExpr::Div(
            Box::new(other.into_expr()),
            Box::new(self.inner.clone()),
        ), name: None }
    }

    fn __pow__(&self, exp: f64, _modulo: Option<PyObject>) -> Self {
        PyExpr { inner: RuntimeExpr::Pow(
            Box::new(self.inner.clone()), exp,
        ), name: None }
    }

    fn __neg__(&self) -> Self {
        PyExpr { inner: RuntimeExpr::Neg(
            Box::new(self.inner.clone()),
        ), name: None }
    }

    fn __abs__(&self) -> Self {
        PyExpr { inner: RuntimeExpr::Abs(
            Box::new(self.inner.clone()),
        ), name: None }
    }

    // ─── Math methods ───

    fn sqrt(&self) -> Self {
        PyExpr { inner: RuntimeExpr::Sqrt(Box::new(self.inner.clone())), name: None }
    }

    fn sin(&self) -> Self {
        PyExpr { inner: RuntimeExpr::Sin(Box::new(self.inner.clone())), name: None }
    }

    fn cos(&self) -> Self {
        PyExpr { inner: RuntimeExpr::Cos(Box::new(self.inner.clone())), name: None }
    }

    fn tan(&self) -> Self {
        PyExpr { inner: RuntimeExpr::Tan(Box::new(self.inner.clone())), name: None }
    }

    // ─── Symbolic differentiation ───

    fn diff(&self, var: &PyExpr) -> PyResult<Self> {
        match &var.inner {
            RuntimeExpr::Var(idx) => Ok(PyExpr {
                inner: self.inner.differentiate(*idx).simplify(),
                name: None,
            }),
            _ => Err(PyValueError::new_err("can only differentiate w.r.t. a variable")),
        }
    }

    // ─── Inspection ───

    #[getter]
    fn variables(&self) -> Vec<u32> {
        self.inner.variables().into_iter().collect()
    }

    fn __repr__(&self) -> String {
        // Pretty-print the expression tree
        format_expr(&self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Accept either a PyExpr or a plain float from Python.
/// This lets users write `x + 1.0` without explicit wrapping.
#[derive(FromPyObject)]
enum ExprOrFloat {
    Expr(PyExpr),
    Float(f64),
}

impl ExprOrFloat {
    fn into_expr(self) -> RuntimeExpr {
        match self {
            ExprOrFloat::Expr(e) => e.inner,
            ExprOrFloat::Float(v) => RuntimeExpr::Const(v),
        }
    }
}
```

### Module-Level Functions

```rust
/// Create named symbolic variables.
/// Usage: x, y = sr.variables("x y")
///        xs = sr.variables("x", count=5)  -> [x_0, x_1, ..., x_4]
#[pyfunction]
#[pyo3(signature = (names, *, count=None))]
fn variables(names: &str, count: Option<usize>) -> Vec<PyExpr> {
    match count {
        Some(n) => (0..n).map(|i| PyExpr {
            inner: RuntimeExpr::Var(i as u32),
            name: Some(format!("{}_{}", names.trim(), i)),
        }).collect(),
        None => names.split_whitespace().enumerate().map(|(i, name)| PyExpr {
            inner: RuntimeExpr::Var(i as u32),
            name: Some(name.to_string()),
        }).collect(),
    }
}

/// Module-level math functions that work on expressions.
#[pyfunction]
fn sqrt(expr: ExprOrFloat) -> PyExpr {
    PyExpr { inner: RuntimeExpr::Sqrt(Box::new(expr.into_expr())), name: None }
}

#[pyfunction]
fn sin(expr: ExprOrFloat) -> PyExpr {
    PyExpr { inner: RuntimeExpr::Sin(Box::new(expr.into_expr())), name: None }
}

// ... cos, tan, atan2, abs similarly

/// Create a residual from an equation: sr.eq(lhs, rhs) → lhs - rhs
#[pyfunction]
fn eq(lhs: ExprOrFloat, rhs: ExprOrFloat) -> PyExpr {
    PyExpr {
        inner: RuntimeExpr::Sub(
            Box::new(lhs.into_expr()),
            Box::new(rhs.into_expr()),
        ),
        name: None,
    }
}

/// Top-level solve that handles expression-based problems.
#[pyfunction]
#[pyo3(signature = (*, residuals=None, equations=None, x0, solver=None,
                    tolerance=None, max_iterations=None))]
fn solve(
    py: Python<'_>,
    residuals: Option<Vec<PyExpr>>,
    equations: Option<Vec<PyExpr>>,
    x0: Vec<f64>,
    solver: Option<&str>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PySolveResult> {
    let exprs = residuals.or(equations)
        .ok_or_else(|| PyValueError::new_err("must provide residuals or equations"))?;

    let rust_exprs: Vec<RuntimeExpr> = exprs.into_iter()
        .map(|e| e.inner)
        .collect();

    let num_vars = x0.len();

    // Build problem: differentiate + (optionally) JIT compile
    let problem = ExprProblem::new("python_expr".into(), num_vars, rust_exprs);

    // Solve with GIL released -- no Python callbacks needed!
    let result = py.allow_threads(|| {
        let solver = AutoSolver::new();
        solver.solve(&problem, &x0)
    });

    Ok(PySolveResult::from(result))
}
```

### The Complete Pipeline (What Happens at `solve()` Time)

```
Python: x, y = sr.variables("x y")         # → Var(0), Var(1)
Python: r = x**2 + y**2 - 1.0              # → Sub(Add(Pow(Var(0),2), Pow(Var(1),2)), Const(1))
Python: sr.solve(residuals=[r, x-y], ...)   # triggers:

  ┌──────────────────────────────────────────────────┐
  │ 1. DIFFERENTIATE (symbolic, in Rust)             │
  │    d(r1)/dx = 2*x    d(r1)/dy = 2*y             │
  │    d(r2)/dx = 1      d(r2)/dy = -1              │
  ├──────────────────────────────────────────────────┤
  │ 2. LOWER to opcodes (OpcodeEmitter)              │
  │    LoadVar r0, 0          ; x                    │
  │    LoadVar r1, 1          ; y                    │
  │    Mul    r2, r0, r0      ; x^2                  │
  │    Mul    r3, r1, r1      ; y^2                  │
  │    Add    r4, r2, r3      ; x^2 + y^2            │
  │    LoadConst r5, 1.0                             │
  │    Sub    r6, r4, r5      ; x^2 + y^2 - 1        │
  │    StoreResidual 0, r6                           │
  │    Sub    r7, r0, r1      ; x - y                │
  │    StoreResidual 1, r7                           │
  ├──────────────────────────────────────────────────┤
  │ 3. JIT COMPILE (Cranelift → native x86/ARM)     │
  │    fn(vars: *const f64, residuals: *mut f64)     │
  │    fn(vars: *const f64, jacobian: *mut f64)      │
  ├──────────────────────────────────────────────────┤
  │ 4. SOLVE (GIL released, pure Rust)              │
  │    Newton-Raphson / Levenberg-Marquardt          │
  │    Calls JIT-compiled native code each iteration │
  │    No Python interaction whatsoever              │
  └──────────────────────────────────────────────────┘
        ↓
  PySolveResult { x: [0.7071, 0.7071], converged: True, ... }
```

### Advanced: Parameter Sweep (Zero Recompilation)

```python
import solverang as sr

x, y = sr.variables("x y")
r = sr.Parameter("r", value=1.0)  # mutable parameter

residuals = [
    x**2 + y**2 - r**2,   # circle of radius r
    x - y,                  # line y = x
]

# First solve: compiles the expression graph
result1 = sr.solve(residuals=residuals, x0=[0.5, 0.5])

# Change parameter -- expression structure unchanged, reuses compiled code
r.value = 2.0
result2 = sr.solve(residuals=residuals, x0=[1.0, 1.0])

r.value = 5.0
result3 = sr.solve(residuals=residuals, x0=[3.0, 3.0])

# All three solves use the same JIT-compiled native code.
# Only the parameter value is different each time.
```

The `Parameter` type uses `Arc<AtomicU64>` (storing f64 bits) so the compiled
code can load the current parameter value at each iteration without
recompilation. This is critical for parameter sweeps, optimization loops, and
interactive applications.

### Integration with Geometry System

```python
import solverang as sr

system = sr.ConstraintSystem2D("custom shape")
p0 = system.add_point(0.0, 0.0, fixed=True)
p1 = system.add_point(3.0, 0.0)
p2 = system.add_point(3.0, 4.0)

# Built-in constraints (already fast, already Lowerable)
system.constrain_distance(p0, p1, 5.0)

# Custom constraint via expression: hypotenuse = 5
x1, y1 = system.coords(p1)  # symbolic refs to point 1's coordinates
x2, y2 = system.coords(p2)  # symbolic refs to point 2's coordinates

system.add_residual(sr.sqrt((x2 - x1)**2 + (y2 - y1)**2) - 3.0)
system.add_residual(y1)  # p1 on x-axis

result = system.solve()
```

When `system.coords(p1)` is called, it returns `PyExpr` objects with
`RuntimeExpr::Var(idx)` where `idx` maps to the correct position in the
constraint system's flat variable array. This allows expression-based custom
constraints to be mixed freely with built-in geometric constraints. Both are
lowered to the same opcode stream, JIT-compiled together, and solved as a
single problem.

### Pros

- **Custom math with zero Python overhead**: user writes Python expressions,
  but solve runs entirely in Rust with GIL released
- **Automatic Jacobians**: symbolic differentiation is exact (no finite
  differences, no user-supplied jacobian)
- **JIT compilation**: expressions are compiled to native code via Cranelift,
  matching hand-written Rust performance
- **Inspectable**: users can print expressions, check derivatives, debug
  symbolically
- **Parameter sweeps**: change constants without recompilation
- **Composable with geometry**: expression constraints mix with built-in
  constraints in the same solve
- **Familiar pattern**: similar to SymPy, PyTorch, JAX expression building
- **Infrastructure already exists**: `ConstraintOp`, `OpcodeEmitter`,
  `JITCompiler`, `Lowerable`, symbolic differentiation

### Cons

- **No control flow**: expressions can't contain `if/else`, loops, or
  conditionals (same limitation as JAX's tracing). A `max(a, b)` function
  provides some workaround.
- **Expression tree bloat**: complex expressions create large Rust-side object
  graphs. A 1000-variable Rosenbrock has ~4000 expression nodes. This is fine
  for construction but uses more memory than a callback.
- **Constant exponents only**: `x**y` where both are variables isn't supported
  (would need `exp(y * ln(x))` which requires adding `Exp` and `Ln` opcodes).
  `x**2`, `x**0.5`, `x**(-1)` all work fine.
- **New Rust code needed**: `RuntimeExpr` type, differentiation, simplification,
  lowering -- about 500-800 lines of Rust. However, the algorithms already exist
  in the macro crate and can be adapted.
- **Debugging opacity**: when something goes wrong numerically, the user can't
  step through the computation with a Python debugger (it's running as native
  code). Need good error messages and `expr.evaluate(x)` for manual checking.
- **`__pow__` signature restriction**: Python's `__pow__` takes 3 args
  (base, exp, mod). The exp must be extractable as a constant `f64` at
  expression-build time. `x ** y` where `y` is a `PyExpr` would need special
  handling.

### What New Rust Code Is Needed

| Component | Lines (est.) | Notes |
|-----------|-------------|-------|
| `RuntimeExpr` enum | ~50 | Mirrors macro crate `Expr` |
| `RuntimeExpr::differentiate()` | ~120 | Port from macro crate, adapt for runtime |
| `RuntimeExpr::simplify()` | ~100 | Port from macro crate |
| `RuntimeExpr::emit()` | ~80 | Lower to `ConstraintOp` via `OpcodeEmitter` |
| `RuntimeExpr::evaluate()` | ~60 | Interpreted fallback |
| `RuntimeExpr::display()` | ~50 | Pretty-printing |
| `ExprProblem` impl | ~100 | `Problem` trait impl with JIT |
| PyO3 `PyExpr` bindings | ~200 | Operator overloads, methods |
| `variables()`, `solve()`, math fns | ~100 | Module-level Python API |
| **Total** | **~860** | |

Most of this is mechanical porting from the macro crate's `Expr` (which already
has differentiation, simplification, and code generation). The main work is
adapting it from compile-time `TokenStream` generation to runtime opcode
emission.

---

## Comparison Matrix

| Criterion                     | A: Callback | B: Pre-built | C: Hybrid | D: Protocol | E: Decorator | **F: Expr Graph** |
|-------------------------------|:-----------:|:------------:|:---------:|:-----------:|:------------:|:-----------------:|
| **Performance (GIL-free)**    | No          | Yes          | Both      | No          | No*          | **Yes**           |
| **Custom problems**           | Yes         | No           | Yes       | Yes         | Yes          | **Yes**           |
| **Auto Jacobian**             | No          | Yes          | Partial   | No          | No           | **Yes**           |
| **JIT-compiled**              | No          | Yes          | Partial   | No          | No           | **Yes**           |
| **Geometry support**          | Manual      | Native       | Native    | Native      | Native       | **Native+custom** |
| **API surface size**          | Small       | Large        | Medium    | Small       | Small        | **Medium**        |
| **Discoverability**           | Good        | Good         | Good      | Fair        | Fair         | **Good**          |
| **scipy familiarity**         | High        | Low          | High      | High        | Medium       | **Medium**        |
| **Boilerplate**               | Low         | Low          | Low       | Lowest      | Lowest       | **Lowest**        |
| **Type safety**               | Medium      | High         | Medium    | Low         | Low          | **High**          |
| **Control flow**              | Yes         | N/A          | Yes       | Yes         | Yes          | **No**            |
| **Batch/parallel solve**      | No          | Yes          | Partial   | No          | No           | **Yes**           |
| **Publish complexity**        | Low         | Medium       | Medium    | Low         | Low          | **Medium**        |
| **New Rust code**             | ~200 LOC    | ~400 LOC     | ~600 LOC  | ~300 LOC    | ~100 LOC     | **~860 LOC**      |

*Design E with expression strings could be GIL-free if compiled to native code.

Design F uniquely achieves **both** custom user-defined math **and** full
GIL-free JIT-compiled performance. It is the only design where users write
arbitrary math in Python but get Rust-native execution speed.

---

## Recommendation: Design F (Expression Graph) + B (Pre-built Geometry)

The expression graph approach (Design F) is the clear winner for the core
problem-definition API. It is the **only design that gives users both custom
math and GIL-free JIT-compiled performance**. Combined with pre-built geometry
types (Design B) for the common case, this gives an API that is:

- **As flexible as callbacks** (user writes arbitrary math)
- **As fast as hand-written Rust** (JIT-compiled, GIL released)
- **Jacobian-free** (automatic symbolic differentiation)
- **Composable** (expression constraints mix with geometry constraints)

### Layered Architecture

```
Layer 3: Pure Python convenience (optional, later)
  │  solverang/__init__.py
  │  @sr.problem decorator, Sketch context manager
  │
Layer 2: PyO3 bindings
  │  solverang/_solverang.so
  │  PyExpr (operator overloads), variables(), solve()
  │  PyConstraintSystem2D/3D (pre-built geometry)
  │  PySolveResult, SolverConfig, exceptions
  │
Layer 1: Rust solver + RuntimeExpr
     solverang crate
     RuntimeExpr → differentiate → lower → JIT compile → solve
     Problem trait, all solvers, geometry, JIT (existing)
```

### Core Principles

1. **Expressions are the primary API**. Users build math with Python operators;
   the result is a Rust-side expression tree that gets JIT-compiled.

2. **Jacobians are always automatic**. Users never write jacobian functions.
   Symbolic differentiation produces exact, sparse jacobians.

3. **The GIL is always released during solve**. Whether using expressions or
   pre-built geometry, the entire solve loop runs in Rust.

4. **Pre-built geometry types exist for convenience**, not necessity. Users
   *could* build all geometric constraints from expressions, but
   `constrain_distance()` is more ergonomic for the common case.

5. **Expression constraints compose with geometry constraints**. A single
   `ConstraintSystem2D` can have both built-in distance constraints and custom
   expression-based constraints, all JIT-compiled together.

### Minimal Initial Scope

For a first release:

1. `RuntimeExpr` in the solverang crate (differentiate, simplify, emit, evaluate)
2. `ExprProblem` implementing `Problem` with JIT compilation
3. PyO3 `Expr` class with operator overloads
4. `variables()`, `solve()`, `eq()`, math functions (`sqrt`, `sin`, etc.)
5. `ConstraintSystem2D` with `add_residual(expr)` support
6. `SolveResult` -- rich result type
7. Custom exceptions

Defer to later:
- `ConstraintSystem3D` (same pattern as 2D)
- `Parameter` type for mutable constants
- `@sr.problem` decorator
- Callback fallback path (Design A) for control-flow-heavy problems
- Batch/parallel solve from Python
- Expression caching and structural hashing

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

1. **Should `RuntimeExpr` live in the main `solverang` crate or the Python crate?**
   Putting it in the main crate means it can be used from pure Rust too (e.g.,
   runtime-defined problems from config files). But it adds a dependency on JIT
   infrastructure. Recommendation: main crate, behind a `runtime-expr` feature
   flag.

2. **Variable-exponent powers**: `x**y` where both are expressions needs `Exp`
   and `Ln` opcodes in `ConstraintOp`. These don't exist yet. Should we add them
   to the opcode set, or restrict `**` to constant exponents? Constant-only is
   simpler and covers 95% of use cases.

3. **Expression deduplication/CSE**: Should we perform common subexpression
   elimination before lowering? For `d = sqrt(dx^2 + dy^2)`, the distance
   computation appears in both the residual and its derivative. CSE would reduce
   redundant computation but adds complexity. The JIT compiler may handle some
   of this already.

4. **Callback fallback**: Should v1 also include the callback path (Design A)
   for problems that need control flow? Or should we ship expressions-only first
   and add callbacks later? Callbacks are simpler to implement but create a
   "two-class" API.

5. **How should `system.coords(p1)` work internally?** The expressions need
   variable indices that map into the constraint system's flat variable array.
   This coupling means expression-based constraints must be aware of the
   geometry system's variable layout. Need a clean abstraction boundary.

6. **Error messages for unsupported operations**: If a user writes
   `sr.solve(residuals=[x if x > 0 else -x], ...)`, the Python `if` evaluates
   eagerly and bypasses the expression graph. We can't catch this at build time.
   Should we document this clearly, or try to provide runtime diagnostics?

7. **Thread safety of expression trees**: `PyExpr` is `#[pyclass(frozen)]` so
   the expression tree is immutable and can be shared across threads. But
   `Parameter` has interior mutability (`AtomicU64`). Is this the right model,
   or should parameter changes create a new expression tree?
