# Plan 5: Modeling DSL Layer

## Status: PROPOSAL
## Priority: Medium (user experience — makes all other plans accessible)
## Depends on: Plan A (ProblemBase), Plan 4 (Dispatch); benefits from Plans 1-3
## Feature flag: `model`

---

## 1. Motivation

Currently, defining a problem requires implementing a Rust trait with manual residual
and Jacobian functions:

```rust
impl Problem for MyProblem {
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] - 1.0, x[0] + x[1] * x[1] - 1.0]
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, 0, 2.0 * x[0]), (0, 1, 1.0),
            (1, 0, 1.0), (1, 1, 2.0 * x[1]),
        ]
    }
    // ... 5 more methods
}
```

This is verbose, error-prone (manual Jacobians!), and requires Rust expertise. A
modeling DSL lets users express problems naturally:

```rust
let model = Model::new()
    .var("x", Continuous::unbounded())
    .var("y", Continuous::unbounded())
    .constraint("circle", |m| m.x * m.x + m.y * m.y - 1.0)
    .constraint("line", |m| m.x - m.y)
    .initial(vec![0.5, 0.5])
    .build();

let result = solverang::solve(&model);
```

The DSL also enables automatic problem classification, Jacobian generation, and
solver selection — tying together all other plans.

## 2. Design

### 2.1 Expression Tree (Core IR)

The DSL is built on a typed expression tree that represents mathematical expressions
symbolically. This enables:
- Automatic differentiation (symbolic Jacobian generation)
- Problem classification (linearity detection, etc.)
- JIT compilation (via existing Cranelift infrastructure)
- Pretty-printing for diagnostics

```rust
/// A symbolic expression over problem variables.
#[derive(Clone, Debug)]
pub enum Expr {
    /// A variable reference: x_i.
    Var(VarId),
    /// A constant value.
    Constant(f64),
    /// Integer constant (for discrete expressions).
    IntConstant(i64),
    /// Binary operation.
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation.
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Conditional expression: if cond then a else b.
    Conditional {
        condition: Box<BoolExpr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// A named parameter (not a variable, but a constant that can change).
    Parameter(ParamId),
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Pow, Atan2, Min, Max,
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    Neg, Abs, Sqrt, Sin, Cos, Tan, Asin, Acos, Atan,
    Exp, Ln, Floor, Ceil, Round,
}

/// A boolean expression for logical constraints.
#[derive(Clone, Debug)]
pub enum BoolExpr {
    Comparison { left: Box<Expr>, op: CompareOp, right: Box<Expr> },
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    Not(Box<BoolExpr>),
    BoolVar(VarId),  // For boolean decision variables
}

#[derive(Clone, Copy, Debug)]
pub enum CompareOp { Eq, Neq, Lt, Le, Gt, Ge }
```

### 2.2 Variable & Parameter IDs

```rust
/// Opaque identifier for a variable in a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarId(usize);

/// Opaque identifier for a parameter in a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(usize);
```

### 2.3 Model Builder

```rust
/// A high-level model builder for defining optimization/constraint problems.
pub struct Model {
    /// Variable definitions.
    variables: Vec<ModelVariable>,
    /// Constraint definitions.
    constraints: Vec<ModelConstraint>,
    /// Objective function (if any).
    objective: Option<(OptimizationSense, Expr)>,
    /// Parameters (constants that can be changed without rebuilding).
    parameters: Vec<ModelParameter>,
    /// Initial variable values.
    initial_values: Vec<f64>,
    /// Problem name.
    name: String,
}

pub struct ModelVariable {
    pub id: VarId,
    pub name: String,
    pub domain: VariableDomain,
    pub initial: Option<f64>,
}

pub struct ModelConstraint {
    pub name: String,
    pub kind: ConstraintKind,
}

pub enum ConstraintKind {
    /// expr = 0 (equality).
    Equality(Expr),
    /// expr >= 0 (inequality).
    Inequality(Expr),
    /// Boolean expression must be true (logical/discrete).
    Logical(BoolExpr),
    /// All values in scope must be different.
    AllDifferent(Vec<VarId>),
    /// Global constraint from CSP module.
    Global(GlobalConstraintExpr),
}

pub struct ModelParameter {
    pub id: ParamId,
    pub name: String,
    pub value: f64,
}
```

### 2.4 Builder API

```rust
impl Model {
    pub fn new() -> Self { ... }

    /// Name the model.
    pub fn name(mut self, name: impl Into<String>) -> Self { ... }

    /// Add a continuous variable with optional bounds.
    pub fn continuous(mut self, name: &str, lower: f64, upper: f64) -> (Self, VarId) { ... }

    /// Add an unbounded continuous variable.
    pub fn var(mut self, name: &str) -> (Self, VarId) { ... }

    /// Add an integer variable.
    pub fn integer(mut self, name: &str, lower: i64, upper: i64) -> (Self, VarId) { ... }

    /// Add a boolean (0/1) variable.
    pub fn boolean(mut self, name: &str) -> (Self, VarId) { ... }

    /// Add a parameter.
    pub fn param(mut self, name: &str, value: f64) -> (Self, ParamId) { ... }

    /// Add an equality constraint: expr = 0.
    pub fn subject_to(mut self, name: &str, expr: Expr) -> Self { ... }

    /// Add an inequality constraint: expr >= 0.
    pub fn such_that(mut self, name: &str, expr: Expr) -> Self { ... }

    /// Set the objective: minimize expr.
    pub fn minimize(mut self, expr: Expr) -> Self { ... }

    /// Set the objective: maximize expr.
    pub fn maximize(mut self, expr: Expr) -> Self { ... }

    /// Set initial values.
    pub fn initial(mut self, values: Vec<f64>) -> Self { ... }

    /// Compile the model into a solvable problem.
    ///
    /// This analyzes the model, generates Jacobians (via symbolic
    /// differentiation of the expression tree), classifies the problem,
    /// and returns a compiled form ready for solving.
    pub fn build(self) -> CompiledModel { ... }
}
```

### 2.5 Operator Overloading for Ergonomic Expression Building

```rust
/// A variable handle that supports arithmetic operators.
///
/// This is the key to the DSL's ergonomics: `x * x + y - 1.0` builds
/// an expression tree automatically.
#[derive(Clone)]
pub struct Var {
    id: VarId,
}

impl std::ops::Add for Var { type Output = Expr; ... }
impl std::ops::Sub for Var { type Output = Expr; ... }
impl std::ops::Mul for Var { type Output = Expr; ... }
impl std::ops::Div for Var { type Output = Expr; ... }
impl std::ops::Neg for Var { type Output = Expr; ... }

impl std::ops::Add<f64> for Var { type Output = Expr; ... }
impl std::ops::Add<Var> for f64 { type Output = Expr; ... }
// ... all combinations of Var, Expr, f64

impl Var {
    pub fn sqrt(self) -> Expr { ... }
    pub fn sin(self) -> Expr { ... }
    pub fn cos(self) -> Expr { ... }
    pub fn pow(self, exp: impl Into<Expr>) -> Expr { ... }
    pub fn abs(self) -> Expr { ... }
}

// Same ops for Expr
impl std::ops::Add for Expr { ... }
// etc.
```

### 2.6 Compiled Model

```rust
/// A model compiled into a form that implements the appropriate problem traits.
///
/// The compilation step:
/// 1. Analyzes expression trees for linearity, convexity, sparsity
/// 2. Generates Jacobian functions via symbolic differentiation
/// 3. Classifies the problem (ProblemClass from Plan 4)
/// 4. Chooses the appropriate trait implementation
pub struct CompiledModel {
    /// The classification of this problem.
    classification: ProblemClass,
    /// Compiled residual evaluator.
    residual_evaluator: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>,
    /// Compiled Jacobian evaluator.
    jacobian_evaluator: Box<dyn Fn(&[f64]) -> Vec<(usize, usize, f64)> + Send + Sync>,
    /// Optional objective evaluator.
    objective_evaluator: Option<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    /// Optional objective gradient evaluator.
    objective_gradient: Option<Box<dyn Fn(&[f64]) -> Vec<(usize, f64)> + Send + Sync>>,
    /// Variable metadata.
    variables: Vec<ModelVariable>,
    /// Constraint metadata.
    constraints: Vec<ModelConstraint>,
    /// Initial values.
    initial_values: Vec<f64>,
    /// Problem name.
    name: String,
}

// CompiledModel implements the appropriate problem traits:
impl ProblemBase for CompiledModel { ... }
impl Problem for CompiledModel { ... }  // when all-continuous
impl OptimizationProblem for CompiledModel { ... }  // when has objective
// etc.
```

### 2.7 Symbolic Differentiation

```rust
/// Differentiate an expression with respect to a variable.
///
/// Returns the symbolic derivative as a new expression tree.
/// This is the same approach as the #[auto_jacobian] macro but applied
/// to the expression tree IR instead of Rust AST.
pub fn differentiate(expr: &Expr, var: VarId) -> Expr {
    match expr {
        Expr::Var(id) => {
            if *id == var { Expr::Constant(1.0) } else { Expr::Constant(0.0) }
        }
        Expr::Constant(_) => Expr::Constant(0.0),
        Expr::Binary { op, left, right } => {
            let dl = differentiate(left, var);
            let dr = differentiate(right, var);
            match op {
                BinaryOp::Add => dl + dr,
                BinaryOp::Mul => dl * right.as_ref().clone() + left.as_ref().clone() * dr,
                // ... product rule, quotient rule, chain rule
            }
        }
        Expr::Unary { op, operand } => {
            let inner_d = differentiate(operand, var);
            match op {
                UnaryOp::Sin => inner_d * operand.as_ref().clone().cos(),
                UnaryOp::Sqrt => inner_d / (Expr::Constant(2.0) * operand.as_ref().clone().sqrt()),
                // ... chain rule for each function
            }
        }
        // ...
    }
}

/// Simplify an expression tree (constant folding, identity removal).
pub fn simplify(expr: Expr) -> Expr {
    match expr {
        Expr::Binary { op: BinaryOp::Add, left, right }
            if matches!(left.as_ref(), Expr::Constant(c) if *c == 0.0) => {
            simplify(*right)
        }
        Expr::Binary { op: BinaryOp::Mul, left, right }
            if matches!(left.as_ref(), Expr::Constant(c) if *c == 1.0) => {
            simplify(*right)
        }
        // ... more simplification rules
        _ => expr,
    }
}
```

## 3. Usage Examples

### 3.1 Simple Nonlinear System

```rust
use solverang::model::*;

let (model, x) = Model::new().var("x");
let (model, y) = model.var("y");

let model = model
    .subject_to("circle", x * x + y * y - 1.0)
    .subject_to("line", x - y)
    .initial(vec![0.5, 0.5])
    .build();

let result = solverang::solve(&model);
```

### 3.2 Constrained Optimization

```rust
let (model, x) = Model::new().continuous("x", -10.0, 10.0);
let (model, y) = model.continuous("y", -10.0, 10.0);

let model = model
    .minimize((x - 1.0).pow(2.0) + (y - 2.0).pow(2.0))  // min distance to (1,2)
    .subject_to("budget", 3.0 * x + 2.0 * y - 12.0)      // 3x + 2y <= 12
    .such_that("non_neg_x", x)                              // x >= 0
    .such_that("non_neg_y", y)                              // y >= 0
    .build();

let result = solverang::solve(&model);
```

### 3.3 Mixed-Integer Problem

```rust
let (model, n) = Model::new().integer("widgets", 0, 100);
let (model, cost) = model.var("cost");

let model = model
    .minimize(cost)
    .subject_to("pricing", cost - (50.0 * n.as_expr() - 0.5 * n.as_expr().pow(2.0)))
    .such_that("min_production", n.as_expr() - 10.0)
    .build();

let result = solverang::solve(&model);
```

### 3.4 CSP (Discrete)

```rust
let mut model = Model::new();
let mut queens = Vec::new();
for i in 0..8 {
    let (m, q) = model.integer(&format!("q{}", i), 0, 7);
    model = m;
    queens.push(q);
}

// All queens in different columns
model = model.all_different("columns", &queens);

// No two queens on same diagonal
for i in 0..8 {
    for j in (i+1)..8 {
        let diff = (j - i) as f64;
        model = model
            .such_that(&format!("diag_{i}_{j}_a"),
                       queens[i].as_expr() - queens[j].as_expr() + diff)
            .such_that(&format!("diag_{i}_{j}_b"),
                       queens[j].as_expr() - queens[i].as_expr() + diff);
    }
}

let model = model.build();
let result = solverang::solve(&model);
```

## 4. Relationship to Existing `#[auto_jacobian]` Macro

The `solverang_macros` crate already does symbolic differentiation on Rust AST in
procedural macros. The modeling DSL does the same thing but at **runtime** on expression
trees. The two approaches complement each other:

| Aspect              | `#[auto_jacobian]` macro      | Modeling DSL                    |
|---------------------|-------------------------------|---------------------------------|
| When                | Compile time                  | Runtime                         |
| Input               | Rust source code              | Expression tree                 |
| Output              | Rust functions                | Closure / JIT-compiled function |
| Performance         | Zero overhead (native code)   | Small overhead (virtual dispatch)|
| Flexibility         | Fixed at compile time         | Can change at runtime           |
| User experience     | Requires Rust trait impl      | Builder API, operator overload  |
| JIT integration     | Separate                      | Natural (expr tree -> Cranelift)|

The DSL's expression tree can optionally be lowered to JIT-compiled code via the existing
`jit/` module infrastructure, bridging the performance gap.

## 5. File Layout

```
crates/solverang/src/
├── model/                         # NEW (feature: "model")
│   ├── mod.rs                     # Module exports
│   ├── expr.rs                    # Expr, BoolExpr, BinaryOp, UnaryOp
│   ├── var.rs                     # VarId, Var (operator overloading)
│   ├── builder.rs                 # Model builder API
│   ├── constraint.rs              # ConstraintKind, ModelConstraint
│   ├── compile.rs                 # CompiledModel, compilation pipeline
│   ├── differentiate.rs           # Symbolic differentiation
│   ├── simplify.rs                # Expression simplification
│   ├── analyze.rs                 # Linearity/convexity analysis
│   └── eval.rs                    # Expression evaluation (interpreter)
```

## 6. Implementation Phases

### Phase 1: Expression tree + differentiation + continuous problems
- `Expr`, operator overloading for `Var` and `Expr`
- `differentiate()` and `simplify()`
- `Model` builder for continuous equality constraints
- `CompiledModel` implementing `Problem`
- Test: solve Rosenbrock, circle+line via DSL

### Phase 2: Inequality and optimization support
- `ConstraintKind::Inequality`
- `Model::minimize()` / `maximize()`
- `CompiledModel` implementing `OptimizationProblem`
- Integration with Plan 1 solvers

### Phase 3: Discrete variables and CSP
- `Model::integer()`, `Model::boolean()`
- `Model::all_different()`
- `CompiledModel` implementing `DiscreteProblem`
- Integration with Plan 3 CSP solver

### Phase 4: Analysis and classification
- Linearity detection (is every expression linear in variables?)
- Sparsity analysis (which variables appear in which constraints?)
- Automatic `ProblemClass` generation for Plan 4 dispatch
- Convexity detection (composition rules)

### Phase 5: JIT compilation
- Lower `Expr` tree to existing `ConstraintOp` opcodes
- JIT-compile via existing Cranelift infrastructure
- Benchmark: interpreted vs JIT vs native (macro) Jacobian

### Phase 6: Parameters and re-solving
- `Model::param()` for changeable constants
- Re-solve with different parameter values without recompilation
- Useful for sensitivity analysis, parametric studies

## 7. Interaction with Concurrent Work

| Active Work Area        | Impact                                                  |
|-------------------------|---------------------------------------------------------|
| `#[auto_jacobian]` macro| Complementary. Macro = compile-time, DSL = runtime.     |
|                         | DSL's `differentiate()` uses same math (chain rule etc).|
| JIT module              | Phase 5 lowers Expr -> ConstraintOp -> Cranelift.       |
|                         | Reuses existing `jit/lower.rs` and `jit/cranelift.rs`.  |
| Geometry module         | Future: geometry constraints expressed as Expr trees.    |
|                         | `DistanceConstraint` = `(dx*dx + dy*dy).sqrt() - d`.    |
| Plan 4 (Dispatch)       | `CompiledModel` carries its `ProblemClass`, dispatch     |
|                         | uses it directly without re-classification.              |
| Plan 1 (Optimization)   | Phase 2 generates OptimizationProblem from model.        |
| Plan 3 (CSP)            | Phase 3 generates DiscreteProblem from model.            |

## 8. Design Decisions & Rationale

**Q: Why not a string-based DSL (like "x^2 + y - 1 = 0")?**
A: String parsing is fragile, gives poor error messages, and loses IDE support.
Operator overloading in Rust gives type-safe expression building with compile-time
error detection. The expression tree is the same IR either way.

**Q: Why runtime differentiation instead of extending the proc macro?**
A: The proc macro works at compile time on static expressions. The DSL enables:
1. Problems defined at runtime (from config files, user input, APIs)
2. Parametric problems where structure changes with parameters
3. Dynamic constraint addition/removal
4. Integration with non-Rust frontends (future: Python bindings, WASM)

**Q: Why not use an existing symbolic math crate?**
A: We need a small, focused expression tree optimized for:
1. Jacobian generation (not general CAS)
2. JIT compilation via Cranelift (specific lowering)
3. CSP constraint representation
4. Integration with solverang's type system

A general CAS (like `symbolica`) would be over-engineered for this use case.

**Q: How does parameter support work?**
A: Parameters are stored in the `CompiledModel` and can be updated without
rebuilding. The expression evaluator/JIT code reads parameter values from a
shared vector, similar to how variables work but without solver modification.

## 9. Open Questions

1. **Multi-dimensional variables**: Should the DSL support vector/matrix variables
   natively? E.g., `model.vector("x", 3)` creating x[0], x[1], x[2]. This is
   syntactic sugar but significantly improves usability for geometric problems.

2. **Constraint naming vs anonymous**: Should constraints always be named? Named
   constraints help diagnostics but add verbosity. Could use auto-generated names.

3. **Serialization**: Should `Model` (pre-compilation) be serializable? This would
   enable saving/loading problem definitions, useful for benchmarking and testing.

4. **Python bindings**: The DSL's runtime expression tree is a natural fit for
   PyO3 bindings. Should this be planned from the start (data structure design)?

## 10. Acceptance Criteria

- [ ] Expression tree with operator overloading compiles and is ergonomic
- [ ] Symbolic differentiation produces correct Jacobians (verified against numeric)
- [ ] `Model::build()` produces a working `Problem` implementation
- [ ] Simple nonlinear system solvable via DSL with same accuracy as manual impl
- [ ] Expression simplification reduces tree size for common patterns
- [ ] At least one optimization problem solvable via DSL (requires Plan 1)
- [ ] At least one CSP solvable via DSL (requires Plan 3)
