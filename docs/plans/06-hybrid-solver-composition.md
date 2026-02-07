# Plan 6: Hybrid Solver Composition

## Status: PROPOSAL
## Priority: High (this is the answer to "how do solvers work together?")
## Depends on: Plan A (ProblemBase), Plan 4 (VariablePartition), Plan 1 (optimization), Plan 3 (CSP)
## Feature flag: `hybrid`

---

## 1. The Problem

A PCB placement tool has a problem like this:

- **Discrete**: which component goes in which slot (100 choices)
- **Continuous**: exact (x, y) positions satisfying clearance constraints
- **Coupling**: "if component A is in slot 3, it must be within 5mm of component B"

The discrete part is a CSP. The continuous part is an NLP. Neither solver can handle the
whole problem alone, and they can't solve their parts independently because the coupling
constraints connect discrete choices to continuous positions.

This is not a hypothetical — it's the core challenge in CAD, VLSI, scheduling, and
engineering design. And it's the real answer to "do these plans integrate into a single
API?" The answer is: **they compose through a protocol, not through a shared trait.**

## 2. Existing Composition Pattern

The codebase already has solver composition in `ParallelSolver`:

```
Full Problem
    │
    ▼ decompose() (union-find)
    │
    ├── Component A ──→ NR/LM solve ──→ partial solution
    ├── Component B ──→ NR/LM solve ──→ partial solution
    └── Component C ──→ NR/LM solve ──→ partial solution
                                            │
                                            ▼ merge_results()
                                        Full Solution
```

This is **horizontal decomposition**: independent sub-problems solved in parallel.
It works because the components share no variables.

Hybrid problems need **vertical decomposition**: a master solver drives the process
and calls sub-solvers that communicate results back. The sub-problems are NOT
independent — the discrete choices affect which continuous constraints exist.

## 3. Composition Patterns

Three well-studied patterns from operations research and formal verification:

### 3.1 Pattern A: Separable (trivial case)

When `VariablePartition::is_separable()` is true, there are no coupling constraints.
Solve each part independently:

```
Mixed Problem
    │
    ▼ partition_variables()
    │
    ├── Continuous sub-problem ──→ AutoSolver ──→ continuous solution
    │
    └── Discrete sub-problem ──→ CSPSolver ──→ discrete solution
                                                    │
                                                    ▼ merge
                                                Full Solution
```

This is just `ParallelSolver` generalized to different solver types. Easy.

### 3.2 Pattern B: Benders Decomposition (master/sub)

When there ARE coupling constraints, one solver drives and the other checks:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Master (discrete):                                         │
│  "Component A in slot 3, component B in slot 7"             │
│       │                                                     │
│       ▼                                                     │
│  Sub-problem (continuous):                                  │
│  "Given those slots, find positions satisfying clearance"   │
│       │                                                     │
│       ├── Feasible → accept, return full solution            │
│       │                                                     │
│       └── Infeasible → generate Benders cut:                │
│           "slot 3 AND slot 7 is infeasible"                 │
│           Add this constraint to master and re-solve         │
│                                                             │
│  Repeat until master is infeasible (proven impossible)      │
│  or sub-problem is feasible (found solution)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The key communication protocol:

```rust
/// Result of checking a sub-problem given a master's decisions.
pub enum SubproblemResult {
    /// The continuous sub-problem is feasible given these discrete decisions.
    Feasible {
        continuous_solution: Vec<f64>,
    },
    /// The continuous sub-problem is infeasible. Here's why.
    Infeasible {
        /// A constraint (over discrete variables) that the master must satisfy
        /// to avoid this infeasibility. This is the "Benders cut."
        cut: BendersCut,
    },
}

/// A constraint learned from sub-problem infeasibility.
/// Added to the master problem to prune infeasible discrete combinations.
pub struct BendersCut {
    /// The discrete variable assignments that caused infeasibility.
    /// The master must avoid this combination (or a superset of it).
    pub conflicting_assignments: Vec<(usize, i64)>,  // (var_index, value)
}
```

### 3.3 Pattern C: DPLL(T) / SMT-style (theory checking)

More fine-grained than Benders: the discrete solver checks with the continuous
solver **incrementally** as it builds partial assignments, not just at the end:

```
CSP Solver (drives the search)
    │
    ├── Assign discrete var 0 = 3
    │   └── Check: is this still potentially feasible?
    │       └── ContinuousTheory::check(partial_assignment) → Ok
    │
    ├── Assign discrete var 1 = 7
    │   └── Check: still feasible?
    │       └── ContinuousTheory::check(partial_assignment) → Conflict!
    │           "var 0 = 3 AND var 1 = 7 → infeasible because constraint C5"
    │           → CSP learns this as a nogood and backtracks
    │
    ├── Assign discrete var 1 = 2 (backtracked)
    │   └── Check: feasible?
    │       └── ContinuousTheory::check(partial_assignment) → Ok
    │   ...
    │
    └── All discrete vars assigned, theory check passes
        └── ContinuousTheory::solve_full() → continuous solution
        └── Return combined solution
```

This is more efficient than Benders because it detects infeasibility **early** (during
search, not after a full discrete solution). It requires:

```rust
/// A "theory solver" that checks consistency of continuous constraints
/// against (partial) discrete assignments.
///
/// This is the interface between the CSP solver (Plan 3) and the
/// continuous solver (existing NR/LM). The CSP solver calls this
/// after each discrete assignment to check if the remaining continuous
/// constraints are still satisfiable.
pub trait TheorySolver: Send + Sync {
    /// Check consistency of a partial discrete assignment.
    ///
    /// Given the current discrete decisions, is the continuous sub-problem
    /// (potentially) feasible?
    ///
    /// - `Ok(())`: consistent so far (search can continue)
    /// - `Err(conflict)`: inconsistent — returns the minimal set of
    ///   discrete assignments that cause the conflict
    fn check(&mut self, assignment: &[(usize, i64)]) -> Result<(), Conflict>;

    /// After all discrete variables are assigned, find the full continuous solution.
    fn solve_full(&mut self, assignment: &[(usize, i64)]) -> Option<Vec<f64>>;
}

/// A set of discrete assignments that jointly cause infeasibility.
pub struct Conflict {
    /// The minimal subset of assignments that conflict.
    /// The CSP solver learns "NOT (var_a = val_a AND var_b = val_b AND ...)"
    pub assignments: Vec<(usize, i64)>,
    /// Human-readable explanation.
    pub reason: String,
}
```

## 4. The `HybridSolver`

```rust
/// Solver for problems with both continuous and discrete variables.
///
/// Uses Plan 4's VariablePartition to analyze the problem structure,
/// then selects a composition strategy:
///
/// - Separable → solve independently in parallel
/// - Coupled with small discrete space → Benders decomposition
/// - Coupled with large discrete space → DPLL(T) / theory-guided search
pub struct HybridSolver {
    /// Strategy for composing solvers.
    strategy: HybridStrategy,
    /// Continuous sub-solver.
    continuous_solver: AutoSolver,
    /// CSP sub-solver (Plan 3).
    #[cfg(feature = "csp")]
    csp_solver: CSPSolver,
    /// Max outer iterations for Benders.
    max_iterations: usize,
}

pub enum HybridStrategy {
    /// Automatically select based on problem structure.
    Auto,
    /// Force Benders decomposition.
    Benders,
    /// Force theory-guided search (DPLL(T)).
    TheoryGuided,
}
```

### 4.1 How the HybridSolver Composes Existing Solvers

```rust
impl HybridSolver {
    pub fn solve(&self, problem: &dyn HybridProblem) -> HybridResult {
        let partition = ProblemClassifier::partition_variables(problem);

        if partition.is_separable() {
            return self.solve_separable(problem, &partition);
        }

        match self.strategy {
            HybridStrategy::Auto => {
                // Small discrete space → Benders (simpler, fewer theory calls)
                // Large discrete space → DPLL(T) (prunes earlier)
                let discrete_space_size = self.estimate_discrete_space(problem, &partition);
                if discrete_space_size < 10_000 {
                    self.solve_benders(problem, &partition)
                } else {
                    self.solve_theory_guided(problem, &partition)
                }
            }
            HybridStrategy::Benders => self.solve_benders(problem, &partition),
            HybridStrategy::TheoryGuided => self.solve_theory_guided(problem, &partition),
        }
    }

    fn solve_separable(&self, problem: &dyn HybridProblem, partition: &VariablePartition) -> HybridResult {
        // Extract continuous sub-problem → solve with AutoSolver
        // Extract discrete sub-problem → solve with CSPSolver
        // Merge solutions
    }

    fn solve_benders(&self, problem: &dyn HybridProblem, partition: &VariablePartition) -> HybridResult {
        let mut cuts: Vec<BendersCut> = Vec::new();

        for iteration in 0..self.max_iterations {
            // 1. Solve master (discrete) with accumulated cuts
            let discrete_result = self.solve_master(problem, partition, &cuts);
            let discrete_assignment = match discrete_result {
                Some(a) => a,
                None => return HybridResult::Infeasible, // master infeasible → no solution exists
            };

            // 2. Solve sub-problem (continuous) given discrete decisions
            let continuous_problem = problem.continuous_subproblem(&discrete_assignment);
            let continuous_result = self.continuous_solver.solve(&continuous_problem, &continuous_problem.initial_point(1.0));

            match continuous_result {
                SolveResult::Converged { solution, residual_norm, .. } if residual_norm < 1e-6 => {
                    // Feasible! Combine and return.
                    return HybridResult::Solved {
                        continuous: solution,
                        discrete: discrete_assignment,
                        iterations: iteration + 1,
                    };
                }
                _ => {
                    // Infeasible. Generate cut and continue.
                    let cut = problem.generate_cut(&discrete_assignment);
                    cuts.push(cut);
                }
            }
        }

        HybridResult::MaxIterations { cuts_generated: cuts.len() }
    }
}
```

### 4.2 The `HybridProblem` Trait

This is the trait users implement to define a problem spanning both paradigms:

```rust
/// A problem with both continuous and discrete variables, connected
/// by coupling constraints.
///
/// This trait defines how to decompose the full problem into sub-problems
/// that individual solvers can handle.
pub trait HybridProblem: ProblemBase {
    /// Given a discrete assignment, construct the continuous sub-problem.
    ///
    /// The discrete variables are fixed to the given values. The returned
    /// Problem contains only the continuous variables and constraints
    /// that remain after fixing the discrete decisions.
    fn continuous_subproblem(&self, discrete_assignment: &[(usize, i64)]) -> Box<dyn Problem>;

    /// Construct the discrete sub-problem (CSP).
    ///
    /// Returns the pure discrete constraints plus any accumulated cuts.
    fn discrete_subproblem(&self, cuts: &[BendersCut]) -> Box<dyn DiscreteProblem>;

    /// Given an infeasible continuous sub-problem, explain which discrete
    /// assignments caused the infeasibility.
    ///
    /// This generates a "cut" — a constraint that the master solver learns
    /// to avoid re-exploring the same infeasible region.
    fn generate_cut(&self, discrete_assignment: &[(usize, i64)]) -> BendersCut;

    /// Create a theory solver for incremental consistency checking.
    ///
    /// Used by the DPLL(T) strategy. The theory solver checks whether
    /// a partial discrete assignment is still compatible with continuous
    /// feasibility.
    fn theory_solver(&self) -> Box<dyn TheorySolver>;
}
```

## 5. Concrete Example: PCB Placement

```rust
struct PCBPlacement {
    components: Vec<Component>,
    slots: Vec<Slot>,
    clearance_rules: Vec<ClearanceRule>,
}

impl HybridProblem for PCBPlacement {
    fn continuous_subproblem(&self, assignment: &[(usize, i64)]) -> Box<dyn Problem> {
        // assignment tells us which slot each component is in.
        // Build a geometric constraint system (existing ConstraintSystem<2>!)
        // with distance/clearance constraints based on assigned slots.
        let mut system = ConstraintSystem::<2>::new();
        for (comp_idx, &slot_idx) in assignment {
            // Place component at slot's center as initial point,
            // add clearance constraints to neighbors
            // ...
        }
        Box::new(system)
    }

    fn discrete_subproblem(&self, cuts: &[BendersCut]) -> Box<dyn DiscreteProblem> {
        // "Assign each component to exactly one slot"
        // plus accumulated cuts from Benders iterations
        // ...
    }

    fn generate_cut(&self, assignment: &[(usize, i64)]) -> BendersCut {
        // Analyze which pair of components was too close
        // Return: "component A and component B can't both be in these slots"
        // ...
    }
}

// Usage:
let solver = HybridSolver::new();
let result = solver.solve(&pcb_problem);
```

## 6. How This Relates to the Other Plans

```
                        Plan 6: HybridSolver
                        (composition protocol)
                       /          |          \
                      /           |           \
            ┌────────▼──┐  ┌─────▼──────┐  ┌──▼──────────┐
            │ Continuous │  │  CSP       │  │ Optimization│
            │ Problem    │  │  Solver    │  │ Solver      │
            │ (existing) │  │  (Plan 3)  │  │ (Plan 1)    │
            └────────────┘  └────────────┘  └─────────────┘
                  │                                │
            NR/LM/Sparse                    Penalty/AugLag
            (existing)                      (Plan 1)

Plan A: ProblemBase     → metadata for classification
Plan 4: Classifier      → VariablePartition for decomposition
Plan 2: B&B             → used for MIP sub-problems (integer + continuous objective)
Plan 5: DSL             → builds HybridProblem from model definition
```

The composition is NOT through a shared `solve()` interface. It's through:

1. **`HybridProblem` trait**: defines how to decompose into typed sub-problems
2. **`TheorySolver` trait**: defines how continuous solver reports back to discrete solver
3. **`BendersCut`**: the communication format between sub-problem and master
4. **`VariablePartition`**: the structural analysis that determines decomposition

Each solver keeps its own typed API. They communicate through these protocols.

## 7. File Layout

```
crates/solverang/src/
├── hybrid/                        # NEW (feature: "hybrid")
│   ├── mod.rs
│   ├── problem.rs                 # HybridProblem trait
│   ├── solver.rs                  # HybridSolver
│   ├── benders.rs                 # Benders decomposition loop
│   ├── theory.rs                  # TheorySolver trait, Conflict
│   ├── dpll_t.rs                  # DPLL(T) integration with CSP solver
│   └── result.rs                  # HybridResult
```

## 8. Implementation Phases

### Phase 1: Separable case + Benders skeleton
- `HybridProblem` trait
- `HybridSolver::solve_separable()` (independent sub-problems)
- `HybridSolver::solve_benders()` with basic cut generation
- Test: simple problem with 2 discrete choices + continuous constraints

### Phase 2: Theory solver interface
- `TheorySolver` trait
- `ContinuousTheory` impl using existing NR/LM
- Basic `check()` that solves full NLP and reports feasible/infeasible

### Phase 3: DPLL(T) integration with CSP (requires Plan 3)
- Modify CSP solver to call `TheorySolver::check()` at each assignment
- Nogood learning from `Conflict`
- Test: N-Queens variant with continuous position constraints

### Phase 4: Warm-starting and cut strengthening
- Pass previous continuous solution as initial point for next iteration
- Minimal conflict analysis (find smallest infeasible subset)
- Performance benchmarks

## 9. Open Questions

1. **Cut generation quality**: The naive cut is "this exact assignment is infeasible."
   Better cuts exclude larger regions ("any assignment where A and B are in adjacent slots").
   How much effort to invest in cut strengthening?

2. **Partial continuous feasibility**: In DPLL(T), checking full NLP feasibility after
   every discrete assignment is expensive. Can we use a cheaper check (bounds propagation,
   relaxation) and only call the full NLP at the end?

3. **Objective optimization in hybrid**: Benders naturally handles optimization (the
   sub-problem also returns an objective bound). But DPLL(T) is a feasibility procedure.
   For optimization, we'd need iterative tightening — which is essentially branch-and-bound
   over the theory solver.

4. **User ergonomics**: `HybridProblem` requires implementing 4 methods including
   `generate_cut()`, which is non-trivial. Can the DSL (Plan 5) generate these
   automatically from a declarative model?

## 10. Acceptance Criteria

- [ ] `HybridProblem` trait compiles with clear documentation
- [ ] Separable case correctly solves continuous + discrete parts independently
- [ ] Benders loop finds solution for simple coupled problem
- [ ] `TheorySolver` trait enables incremental consistency checking
- [ ] At least one hybrid problem solved that neither CSP nor NLP can handle alone
- [ ] Integration with existing `ConstraintSystem<2>` as continuous sub-problem
