# Plan 3: Finite-Domain Constraint Satisfaction (CSP) Module

## Status: PROPOSAL (v2 — TheorySolver hook for Plan 6 added)
## Priority: Medium (opens entirely new problem class)
## Depends on: Plan A (ProblemBase, VariableDomain)
## Feature flag: `csp`

---

## Revision Notes (v2)

**Core design unchanged.** `CSPSolver` remains a typed solver for `DiscreteProblem`.

**Composition hook (Plan 6)**: The CSP solver needs an extension point for Plan 6's
DPLL(T) integration. Specifically, the backtracking search loop must accept an optional
`TheorySolver` callback that is called after each variable assignment. If the theory
solver reports a conflict, the CSP solver backtracks and learns the conflict as a nogood.
This is the "T" in DPLL(T). Implementation detail:

```rust
impl CSPSolver {
    /// Solve a pure discrete problem (no theory).
    pub fn solve(&self, problem: &dyn DiscreteProblem) -> CSPResult { ... }

    /// Solve with a theory solver for hybrid problems (Plan 6).
    /// After each discrete assignment, checks theory consistency.
    pub fn solve_with_theory(
        &self,
        problem: &dyn DiscreteProblem,
        theory: &mut dyn TheorySolver,
    ) -> CSPResult { ... }
}
```

The CSP solver's search loop already has assignment/backtrack points — the theory
hook plugs in at those points without changing the core algorithm.

---

## 1. Motivation

Finite-domain constraint satisfaction is a fundamentally different paradigm from
continuous solving. Problems like Sudoku, graph coloring, scheduling, timetabling,
and configuration cannot be expressed as F(x) = 0 with gradients. They require:

- **Discrete variables** with finite domains (e.g., {1..9}, {Red, Green, Blue})
- **Constraint propagation** (arc consistency) to prune domains before search
- **Backtracking search** with intelligent variable/value ordering
- **Global constraints** (alldifferent, cumulative) with specialized propagators

This plan adds a CSP module that runs **alongside** the continuous solver, not as a
replacement. The two systems can later be composed for hybrid problems (Plan 4).

## 2. Core Abstractions

### 2.1 `DiscreteProblem` Trait

```rust
/// A constraint satisfaction problem over finite domains.
///
/// Variables have discrete, finite domains. Constraints are predicates
/// that must all be satisfied simultaneously. There is no objective function
/// (pure feasibility); for optimization over discrete variables, see Plan 2.
pub trait DiscreteProblem: ProblemBase {
    /// Initial domain for each variable.
    fn domains(&self) -> Vec<FiniteDomain>;

    /// Number of constraints.
    fn constraint_count(&self) -> usize;

    /// Get constraint by index.
    fn constraint(&self, index: usize) -> &dyn CSPConstraint;

    /// All constraints.
    fn constraints(&self) -> Vec<&dyn CSPConstraint>;
}
```

### 2.2 `FiniteDomain`

```rust
/// A finite set of possible values for a discrete variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteDomain {
    /// Sorted, unique values in the domain.
    values: Vec<i64>,
}

impl FiniteDomain {
    /// Create a domain from a range [lower, upper] inclusive.
    pub fn range(lower: i64, upper: i64) -> Self { ... }

    /// Create a domain from explicit values.
    pub fn from_values(values: impl IntoIterator<Item = i64>) -> Self { ... }

    /// Boolean domain: {0, 1}.
    pub fn boolean() -> Self { Self::from_values([0, 1]) }

    /// Number of values in the domain.
    pub fn size(&self) -> usize { self.values.len() }

    /// Is this domain empty (unsatisfiable)?
    pub fn is_empty(&self) -> bool { self.values.is_empty() }

    /// Is this domain a singleton (variable fixed)?
    pub fn is_fixed(&self) -> bool { self.values.len() == 1 }

    /// Remove a value from the domain. Returns true if the value was present.
    pub fn remove(&mut self, value: i64) -> bool { ... }

    /// Restrict domain to values in a given set.
    pub fn intersect(&mut self, other: &FiniteDomain) { ... }

    /// Restrict to values satisfying a predicate.
    pub fn filter(&mut self, pred: impl Fn(i64) -> bool) { ... }

    /// The fixed value if this is a singleton domain.
    pub fn fixed_value(&self) -> Option<i64> { ... }

    /// Minimum and maximum values.
    pub fn bounds(&self) -> Option<(i64, i64)> { ... }

    /// Iterator over values.
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ { ... }
}
```

### 2.3 `CSPConstraint` Trait

```rust
/// A constraint in a finite-domain CSP.
pub trait CSPConstraint: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Indices of variables this constraint involves.
    fn scope(&self) -> &[usize];

    /// Check if the constraint is satisfied by a complete or partial assignment.
    ///
    /// `assignment` maps variable index -> assigned value.
    /// Returns `None` if not all variables in scope are assigned (cannot check).
    fn is_satisfied(&self, assignment: &PartialAssignment) -> Option<bool>;

    /// Propagate this constraint: remove inconsistent values from domains.
    ///
    /// Returns `PropagationResult` indicating what changed.
    /// This is the core of constraint propagation (arc consistency).
    fn propagate(&self, domains: &mut [FiniteDomain]) -> PropagationResult;
}

/// Result of constraint propagation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PropagationResult {
    /// No domains changed.
    NoChange,
    /// Some values were removed from domains.
    Changed {
        /// Which variables had their domains reduced.
        changed_variables: Vec<usize>,
    },
    /// A domain became empty — this constraint cannot be satisfied.
    Contradiction,
}

/// A partial assignment of values to variables.
#[derive(Clone, Debug)]
pub struct PartialAssignment {
    values: Vec<Option<i64>>,
}

impl PartialAssignment {
    pub fn new(size: usize) -> Self { ... }
    pub fn assign(&mut self, var: usize, value: i64) { ... }
    pub fn unassign(&mut self, var: usize) { ... }
    pub fn get(&self, var: usize) -> Option<i64> { ... }
    pub fn is_complete(&self) -> bool { ... }
}
```

## 3. Built-in Constraints

### 3.1 Table Constraint (universal)

```rust
/// Extensional constraint: explicitly lists allowed/forbidden tuples.
pub struct TableConstraint {
    scope: Vec<usize>,
    tuples: Vec<Vec<i64>>,
    is_support: bool,  // true = allowed tuples, false = forbidden tuples
}
```

### 3.2 Common Global Constraints

```rust
/// All variables in scope must take distinct values.
pub struct AllDifferent { scope: Vec<usize> }

/// A binary relation: var_a op var_b (+ optional offset).
pub struct BinaryRelation {
    var_a: usize,
    var_b: usize,
    relation: Relation,
    offset: i64,  // var_a `relation` var_b + offset
}

pub enum Relation { Eq, Neq, Lt, Le, Gt, Ge }

/// Linear constraint: sum(coefficients[i] * vars[i]) `relation` rhs.
pub struct LinearConstraint {
    vars: Vec<usize>,
    coefficients: Vec<i64>,
    relation: Relation,
    rhs: i64,
}

/// Element constraint: vars[index_var] = value_var.
pub struct Element {
    array_vars: Vec<usize>,
    index_var: usize,
    value_var: usize,
}
```

### 3.3 Propagators

Each global constraint has a specialized propagator:

| Constraint    | Propagation Algorithm          | Complexity     |
|---------------|--------------------------------|----------------|
| AllDifferent  | Matching-based (Regin 1994)    | O(n * d)       |
| BinaryRelation| Arc consistency (AC-3)         | O(d)           |
| Linear        | Bounds consistency             | O(n)           |
| Table         | GAC via STR (simple tabular)   | O(tuples * arity) |
| Element       | Domain filtering               | O(d)           |

The initial implementation can use simpler (weaker) propagation and be upgraded:

- Phase 1: AC-3 for binary, forward checking for n-ary
- Phase 2: Full GAC for AllDifferent, bounds consistency for Linear
- Phase 3: STR for Table constraints

## 4. CSP Solver

### 4.1 Backtracking Search with Propagation

```rust
pub struct CSPSolver {
    /// Variable ordering heuristic.
    variable_ordering: VariableOrdering,
    /// Value ordering heuristic.
    value_ordering: ValueOrdering,
    /// Propagation level.
    propagation: PropagationLevel,
    /// Maximum solutions to find (0 = find all).
    max_solutions: usize,
    /// Maximum backtracks before giving up.
    max_backtracks: usize,
}

pub enum VariableOrdering {
    /// Smallest domain first (dom/wdeg if available).
    SmallestDomain,
    /// Most constrained variable (most constraints involving unassigned neighbors).
    MostConstrained,
    /// First unassigned variable (simplest, for debugging).
    Lexicographic,
    /// dom/wdeg: domain size divided by weighted degree.
    DomOverWDeg,
}

pub enum ValueOrdering {
    /// Ascending order.
    Ascending,
    /// Value that leads to least domain reductions in neighbors.
    LeastConstraining,
    /// Random (for randomized restart).
    Random { seed: u64 },
}

pub enum PropagationLevel {
    /// No propagation (pure backtracking).
    None,
    /// Forward checking: propagate only the most recently assigned variable.
    ForwardChecking,
    /// Maintaining arc consistency: full propagation after each assignment.
    MAC,
}
```

### 4.2 Solver Algorithm (MAC — Maintaining Arc Consistency)

```
solve(domains, assignment):
    if propagate_all(domains) == Contradiction:
        return Failure
    if assignment.is_complete():
        return Solution(assignment)

    var = select_variable(domains, assignment)   // VariableOrdering
    for value in order_values(var, domains):     // ValueOrdering
        saved_domains = domains.clone()          // checkpoint
        domains[var] = {value}
        assignment.assign(var, value)

        if propagate_all(domains) != Contradiction:
            result = solve(domains, assignment)
            if result == Solution:
                return result

        domains = saved_domains                  // restore
        assignment.unassign(var)

    return Failure
```

### 4.3 Result Type

```rust
#[derive(Clone, Debug)]
pub enum CSPResult {
    /// Found a satisfying assignment.
    Satisfiable {
        assignment: Vec<i64>,
        backtracks: usize,
        propagations: usize,
    },
    /// Proven unsatisfiable.
    Unsatisfiable {
        backtracks: usize,
        propagations: usize,
    },
    /// Found multiple solutions.
    AllSolutions {
        solutions: Vec<Vec<i64>>,
        backtracks: usize,
    },
    /// Hit resource limits.
    ResourceLimit {
        partial_solutions: Vec<Vec<i64>>,
        backtracks: usize,
    },
}
```

## 5. File Layout

```
crates/solverang/src/
├── csp/                           # NEW (feature: "csp")
│   ├── mod.rs                     # Module exports
│   ├── problem.rs                 # DiscreteProblem trait
│   ├── domain.rs                  # FiniteDomain
│   ├── assignment.rs              # PartialAssignment
│   ├── constraint.rs              # CSPConstraint trait, PropagationResult
│   ├── constraints/               # Built-in constraint types
│   │   ├── mod.rs
│   │   ├── all_different.rs
│   │   ├── binary_relation.rs
│   │   ├── linear.rs
│   │   ├── table.rs
│   │   └── element.rs
│   ├── propagation.rs             # AC-3, forward checking, MAC
│   ├── solver.rs                  # CSPSolver (backtracking + propagation)
│   ├── ordering.rs                # Variable and value ordering heuristics
│   └── result.rs                  # CSPResult
```

## 6. Implementation Phases

### Phase 1: Core framework + basic constraints
- `FiniteDomain`, `PartialAssignment`, `CSPConstraint` trait
- `BinaryRelation`, `AllDifferent` (simple propagation)
- Forward checking solver
- Test: N-Queens, Sudoku

### Phase 2: Arc consistency
- AC-3 propagation engine
- MAC solver (maintaining arc consistency)
- `DomOverWDeg` variable ordering
- Test: graph coloring, Latin squares

### Phase 3: Global constraint propagators
- AllDifferent with matching-based propagation
- Linear with bounds consistency
- Table with STR
- Test: Harder scheduling instances

### Phase 4: Optimization over CSP (integration with Plan 2)
- CSP with optimization objective (COP — Constraint Optimization Problem)
- Branch and bound over discrete domains
- Interface: `ConstraintOptimizationProblem` combining `DiscreteProblem` + objective

### Phase 5: Randomized restarts and learning
- Random value ordering for diversification
- Restart strategies (geometric, Luby)
- Nogood learning (simple clause recording)

## 7. Integration Points

### 7.1 Decomposition Reuse

The existing `decomposition.rs` uses union-find to detect independent components.
The CSP solver can reuse this:

```rust
// CSP constraints define a scope (variable indices) — same as constraint graph edges
impl DecomposableProblem for DiscreteProblemWrapper {
    fn constraint_graph(&self) -> Vec<(usize, usize)> {
        self.constraints().iter().enumerate().flat_map(|(c_idx, c)| {
            c.scope().iter().map(move |&v_idx| (c_idx, v_idx))
        }).collect()
    }
}
```

Independent CSP components can be solved in parallel using existing `ParallelSolver`
patterns.

### 7.2 Hybrid Continuous+Discrete

Some problems have both continuous and discrete aspects (e.g., geometric constraint
satisfaction with discrete choices). The dispatcher (Plan 4) will route:

- Pure continuous part -> NR/LM solvers
- Pure discrete part -> CSP solver
- Coupled parts -> hybrid solver (future work)

### 7.3 Connection to Geometry

Interesting future application: geometric problems with discrete choices:
- "This point must lie on one of these three circles" (disjunctive constraint)
- "These parts must be placed in one of N orientations" (discrete + continuous)

## 8. Interaction with Concurrent Work

| Active Work Area      | Impact                                                    |
|-----------------------|-----------------------------------------------------------|
| Existing solvers      | None. CSP is a separate module with its own solver.       |
| Decomposition         | Reuses union-find infrastructure for CSP decomposition.   |
| Parallel solver       | CSP components can be solved in parallel (rayon).         |
| Geometry module       | Future hybrid: discrete geometry choices.                 |
| Plan A (ProblemBase)  | CSP variables use VariableDomain::Enumerated.             |
| Plan 4 (Dispatcher)   | Dispatcher routes discrete problems to CSP solver.        |

## 9. Benchmarks & Validation

- **N-Queens** (N = 4, 8, 12, 20, 50): Classic CSP benchmark
- **Sudoku**: 9x9 with varying difficulty
- **Graph coloring**: Petersen graph, random graphs
- **Latin squares**: Order 5, 10
- **Comparison**: Against simple brute-force to validate propagation effectiveness

## 10. Open Questions

1. **Domain representation**: `Vec<i64>` is simple but can be slow for large ranges.
   Should we support interval representation + bitsets for common cases?

2. **Constraint learning**: Nogood recording (clause learning from SAT) dramatically
   improves performance on hard instances. Include in Phase 1 or defer?

3. **SAT encoding**: Should we support automatic CSP-to-SAT translation for problems
   where a SAT solver would be more efficient? This is common in CP solvers.

4. **Custom propagators**: Should users be able to define custom propagators, or only
   use built-in constraint types?

## 11. Acceptance Criteria

- [ ] `FiniteDomain`, `CSPConstraint`, `DiscreteProblem` traits compile
- [ ] CSP solver finds all 92 solutions to 8-Queens
- [ ] Sudoku puzzle solved with propagation (not just brute force)
- [ ] `AllDifferent` constraint with basic propagation
- [ ] Unsatisfiable problems correctly detected
- [ ] MAC solver outperforms forward checking on graph coloring
