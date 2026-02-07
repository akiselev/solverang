# Plan 2: Integer Variable Support with Branch-and-Bound

## Status: PROPOSAL
## Priority: Medium-High (unlocks scheduling, assignment, routing)
## Depends on: Plan A (ProblemBase, VariableDomain), Plan 1 (OptimizationProblem)
## Feature flag: `mixed-integer`

---

## 1. Motivation

Many practical problems mix continuous and integer variables:

- **Facility location**: Which facilities to open (binary) + how much to ship (continuous)
- **VLSI placement**: Grid-aligned placement (integer) + routing (continuous)
- **Scheduling**: Task assignment (integer) + timing (continuous)
- **Engineering design**: Number of components (integer) + dimensions (continuous)

Solverang's continuous solver is a natural LP/NLP relaxation engine. Wrapping it in a
branch-and-bound tree gives basic mixed-integer nonlinear programming (MINLP) capability.

## 2. Core Concept

Branch-and-bound (B&B) solves MIP by:

1. **Relax** integer variables to continuous, solve the relaxation
2. If the relaxed solution is integer-feasible, it's optimal for this node
3. Otherwise, **branch** on a fractional integer variable: create two child nodes
   with tighter bounds (x_i <= floor(x*_i) and x_i >= ceil(x*_i))
4. **Bound**: prune nodes whose relaxation objective is worse than the best known
   integer-feasible solution (incumbent)
5. Repeat until the tree is exhausted or the gap is small enough

## 3. Design

### 3.1 `MixedIntegerProblem` Trait

```rust
/// A problem with both continuous and integer/binary variables.
///
/// Builds on OptimizationProblem by adding integrality constraints.
/// The relaxation (ignoring integrality) must be solvable as a
/// standard OptimizationProblem.
pub trait MixedIntegerProblem: OptimizationProblem {
    /// Which variables must take integer values.
    ///
    /// Returns indices into the variable array. Variables not listed
    /// here are treated as continuous.
    fn integer_variables(&self) -> &[usize];

    /// Which variables are binary (0-1).
    ///
    /// Binary variables are a special case of integer with bounds [0, 1].
    /// Solvers may exploit this for tighter relaxations.
    fn binary_variables(&self) -> Vec<usize> {
        // Default: check integer variables with [0,1] bounds
        self.integer_variables().iter().filter(|&&i| {
            match &self.variables()[i].domain {
                VariableDomain::Boolean => true,
                VariableDomain::Integer { lower, upper } => *lower == 0 && *upper == 1,
                _ => false,
            }
        }).copied().collect()
    }

    /// Tolerance for considering a variable "integer".
    fn integrality_tolerance(&self) -> f64 { 1e-6 }
}
```

### 3.2 Branch-and-Bound Solver

```rust
pub struct BranchAndBoundSolver {
    /// Solver for continuous relaxations at each node.
    relaxation_solver: Box<dyn RelaxationSolver>,
    /// Branching strategy.
    branching: BranchingStrategy,
    /// Node selection strategy.
    node_selection: NodeSelection,
    /// Maximum nodes to explore.
    max_nodes: usize,
    /// Relative optimality gap tolerance.
    gap_tolerance: f64,
    /// Time limit.
    time_limit: Option<Duration>,
}

pub enum BranchingStrategy {
    /// Branch on the most fractional variable.
    MostFractional,
    /// Branch on the first fractional variable.
    FirstFractional,
    /// Strong branching: solve both children, pick best bound improvement.
    /// Expensive but produces smaller trees.
    Strong { candidates: usize },
    /// Reliability branching: hybrid of pseudocost and strong branching.
    Reliability { min_reliable: usize },
}

pub enum NodeSelection {
    /// Best-first: explore node with best (lowest) bound.
    BestFirst,
    /// Depth-first: explore deepest node (good for finding incumbents fast).
    DepthFirst,
    /// Best-estimate: weighted combination of bound and depth.
    BestEstimate { depth_weight: f64 },
}
```

### 3.3 B&B Tree Node

```rust
/// A node in the branch-and-bound tree.
struct BBNode {
    /// ID of this node.
    id: usize,
    /// Parent node ID (None for root).
    parent: Option<usize>,
    /// Depth in the tree.
    depth: usize,
    /// Additional variable bounds imposed at this node.
    /// These are the branching decisions accumulated on the path from root.
    bound_changes: Vec<BoundChange>,
    /// Relaxation bound (objective value of the continuous relaxation).
    relaxation_bound: Option<f64>,
    /// Status of this node.
    status: NodeStatus,
}

struct BoundChange {
    variable: usize,
    bound_type: BoundType,
    value: f64,
}

enum BoundType { Lower, Upper }

enum NodeStatus {
    Pending,
    Solved { bound: f64, solution: Vec<f64> },
    Pruned(PruneReason),
    IntegerFeasible { objective: f64, solution: Vec<f64> },
}

enum PruneReason {
    BoundExceedsIncumbent,
    Infeasible,
    IntegerFeasible,
}
```

### 3.4 Relaxation Interface

The continuous relaxation at each B&B node is solved using existing solvers via a
transform that applies the node's bound changes:

```rust
/// Wraps an OptimizationProblem with additional bound constraints from B&B node.
struct NodeRelaxation<'a, P: MixedIntegerProblem> {
    problem: &'a P,
    bound_changes: &'a [BoundChange],
}

impl<P: MixedIntegerProblem> OptimizationProblem for NodeRelaxation<'_, P> {
    // Delegates to inner problem but adds bound constraints
    fn inequality_count(&self) -> usize {
        self.problem.inequality_count() + self.bound_changes.len()
    }
    // ...
}
```

This is then solved via `PenaltyTransform` or `AugmentedLagrangianSolver` from Plan 1,
which ultimately delegates to existing NR/LM solvers.

### 3.5 Result Type

```rust
#[derive(Clone, Debug)]
pub enum MIPResult {
    /// Proven optimal within gap tolerance.
    Optimal {
        solution: Vec<f64>,
        objective: f64,
        gap: f64,           // relative optimality gap
        nodes_explored: usize,
        iterations_total: usize,
    },
    /// Found a feasible solution but could not prove optimality.
    Feasible {
        solution: Vec<f64>,
        objective: f64,
        gap: f64,
        nodes_explored: usize,
    },
    /// Proven infeasible (no integer-feasible solution exists).
    Infeasible {
        nodes_explored: usize,
    },
    /// Hit resource limits.
    ResourceLimit {
        best_solution: Option<Vec<f64>>,
        best_objective: Option<f64>,
        gap: Option<f64>,
        nodes_explored: usize,
        reason: ResourceLimitReason,
    },
    /// Solver error.
    Failed { error: SolveError },
}

pub enum ResourceLimitReason {
    MaxNodes,
    TimeLimit,
}
```

## 4. File Layout

```
crates/solverang/src/
├── mixed_integer/                 # NEW (feature: "mixed-integer")
│   ├── mod.rs                     # Module exports
│   ├── problem.rs                 # MixedIntegerProblem trait
│   ├── solver.rs                  # BranchAndBoundSolver
│   ├── node.rs                    # BBNode, NodeStatus, BoundChange
│   ├── branching.rs               # Branching strategies
│   ├── node_selection.rs          # Node selection strategies
│   ├── relaxation.rs              # NodeRelaxation wrapper
│   ├── result.rs                  # MIPResult
│   └── heuristics.rs              # Rounding heuristics, feasibility pump
```

## 5. Implementation Phases

### Phase 1: Basic B&B framework
- `MixedIntegerProblem` trait
- `BranchAndBoundSolver` with `MostFractional` branching and `BestFirst` selection
- `NodeRelaxation` wrapping into `OptimizationProblem` -> `Problem`
- Simple test: min x s.t. x in {0, 1, 2, 3, 4, 5}

### Phase 2: Warm-starting and heuristics
- Pass previous node's solution as initial point to child node's relaxation
- Simple rounding heuristic: round fractional integers, check feasibility
- Feasibility pump: alternate between rounding and projecting onto constraints

### Phase 3: Advanced branching
- Pseudocost tracking for branching variable selection
- Strong branching with configurable candidate limit
- Reliability branching hybrid

### Phase 4: Parallelism
- Parallel node exploration (each node's relaxation is independent)
- Integrate with existing `ParallelSolver` infrastructure
- Shared incumbent with atomic updates

### Phase 5: Cutting planes (advanced, future)
- Gomory fractional cuts from LP relaxation
- Lift-and-project cuts
- Problem-specific cuts (user-defined callback)

## 6. Performance Considerations

- **Warm starting**: The relaxation at a child node differs from its parent by only one
  bound change. The parent's solution is an excellent initial point. This is critical
  for performance — cold-starting every node from scratch would be very slow.

- **Relaxation solver choice**: For LP-like subproblems, an interior-point or simplex
  solver (Plan 1, Strategy C) would be much faster than NR/LM. For MINLP, the existing
  nonlinear solvers are appropriate.

- **Node storage**: For large trees, store only bound changes relative to parent, not
  full problem copies. Reconstruct full bounds by walking the path from root.

## 7. Interaction with Concurrent Work

| Active Work Area      | Impact                                                    |
|-----------------------|-----------------------------------------------------------|
| Existing solvers      | Used directly: NR/LM solve relaxations at each node.     |
| Plan 1 (Optimization) | Direct dependency: B&B wraps OptimizationProblem.         |
| Plan A (ProblemBase)  | Uses VariableDomain to identify integer/binary variables. |
| Parallel solver       | Future: parallel node exploration reuses rayon infra.     |
| Decomposition         | Opportunity: decompose into independent integer and       |
|                       | continuous subproblems for faster solving.                |
| Sparse solver         | Large relaxations may benefit from sparse operations.     |

## 8. Scope Boundaries

This plan explicitly does NOT include:
- Full LP/simplex solver (would use existing NR/LM or external crate)
- Presolve/preprocessing (probing, coefficient reduction)
- Cut generation beyond basic Gomory (MIP cuts are a deep topic)
- Indicator constraints (if binary=1, then constraint active)

These are valuable future extensions but each is a substantial effort. The basic B&B
framework enables them without requiring them upfront.

## 9. Open Questions

1. **External LP solver integration**: Should we optionally wrap an external LP solver
   (e.g., `minilp`, `good_lp`) for faster relaxations? Or is converting to NR/LM
   residuals sufficient for v1?

2. **Callback API**: Should users be able to register callbacks for custom branching,
   lazy constraint generation, or heuristics? Common in mature MIP solvers (Gurobi, SCIP).

3. **Presolve**: Even basic presolve (fixing variables, removing redundant constraints)
   can dramatically reduce tree size. Worth including in v1?

## 10. Acceptance Criteria

- [ ] `MixedIntegerProblem` trait defined
- [ ] `BranchAndBoundSolver` solves simple knapsack-type problems correctly
- [ ] Pruning works: provably optimal solutions found without exhaustive enumeration
- [ ] Warm-starting reduces total iteration count vs cold-starting
- [ ] At least one problem with both integer and continuous variables solved correctly
- [ ] Resource limits (max nodes, time) respected
