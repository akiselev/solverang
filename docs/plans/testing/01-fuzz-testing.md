# Plan 01: Fuzz Testing

## Goal

Fuzz testing is the highest-priority testing addition for solverang. The crate has two
distinct attack surfaces that demand fuzzing:

1. **Legacy JIT module** (`src/jit/cranelift.rs`): Contains `unsafe` code using
   `std::mem::transmute` to convert raw pointers into function pointers. A malformed
   constraint description that produces bad machine code could cause a segfault or
   memory corruption.

2. **V3 generational ID system** (`src/id.rs`, `src/param/store.rs`, `src/system.rs`):
   While the V3 architecture contains no `unsafe` code, the generational index pattern
   that prevents use-after-free relies on correct generation counter management across
   free-list allocation/deallocation cycles. Bugs here manifest as silent data corruption
   (accessing wrong parameter values via stale IDs) rather than crashes, making them
   extremely hard to find with conventional testing.

Beyond these, the V3 architecture introduces 21k lines of algorithmic code spanning a
5-phase pipeline, graph algorithms, symbolic reduction passes, and closed-form analytical
solvers. All of these must handle arbitrary `f64` inputs (NaN, infinity, subnormals,
extreme magnitudes) and degenerate constraint topologies without panicking.

## Priority Ranking

Generational ID and `ParamStore` free-list bugs are the **new #1 fuzz priority**, above
even JIT unsafe code. Rationale: JIT bugs produce immediate crashes (segfaults) that are
easy to detect; stale-ID bugs produce silent wrong answers that can propagate through the
entire pipeline before manifesting as incorrect geometry.

| Priority | Target | Risk |
|----------|--------|------|
| **Critical** | V3 `ConstraintSystem` (generational IDs + free-list) | Stale ID = silent data corruption |
| **Critical** | `ParamStore` (alloc/free cycles, stale `ParamId`) | Wrong param values, panics on freed slots |
| **High** | V3 pipeline (5-phase with caching, incremental re-solve) | State machine bugs across phases |
| **High** | Closed-form solvers (circle-circle, Newton, polar) | Math edge cases with no solution |
| **High** | JIT compiler (legacy `unsafe` code) | Segfault, memory corruption |
| **Medium** | Graph algorithms (decompose, DOF, redundancy, patterns) | Index panics on degenerate graphs |
| **Medium** | Reduce passes (substitute, merge, eliminate) | Symbolic transforms corrupt state |
| **Medium** | Legacy solver with arbitrary inputs | Solver divergence, NaN propagation |
| **Low** | Sparse matrix operations | Out-of-bounds, duplicate entries |
| **Low** | Jacobian computation (legacy) | Dimension mismatches, non-finite values |

## Tool Selection

**Recommendation: `cargo-fuzz`** (wraps libFuzzer)

| Criterion | `cargo-fuzz` | `afl.rs` |
|-----------|-------------|----------|
| Setup complexity | Low (one command) | Medium (requires AFL++ install) |
| Structured fuzzing | `arbitrary` crate integration | Requires manual |
| CI integration | Simple (`cargo fuzz run -- -max_total_time=N`) | More complex |
| Coverage feedback | Built-in (SanitizerCoverage) | Built-in (AFL instrumentation) |
| Rust ecosystem support | First-class | Good but less common |

`cargo-fuzz` is the standard in the Rust ecosystem. If deeper exploration is needed
later, `afl.rs` can be added as a complement -- they find different bugs due to
different mutation strategies.

## Setup Steps

### 1. Install cargo-fuzz

```bash
cargo install cargo-fuzz
```

### 2. Initialize fuzz directory

```bash
cd crates/solverang
cargo fuzz init
```

This creates:

```
crates/solverang/fuzz/
├── Cargo.toml
├── fuzz_targets/
│   └── (targets go here)
└── corpus/
    └── (seed inputs go here)
```

### 3. Configure fuzz/Cargo.toml

```toml
[package]
name = "solverang-fuzz"
version = "0.0.0"
publish = false
edition = "2021"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
arbitrary = { version = "1", features = ["derive"] }
solverang = { path = "..", features = ["geometry", "jit", "sparse", "parallel", "macros"] }

# --- V3 targets (Critical + High priority) ---

[[bin]]
name = "fuzz_v3_constraint_system"
path = "fuzz_targets/fuzz_v3_constraint_system.rs"
test = false
doc = false

[[bin]]
name = "fuzz_param_store"
path = "fuzz_targets/fuzz_param_store.rs"
test = false
doc = false

[[bin]]
name = "fuzz_pipeline"
path = "fuzz_targets/fuzz_pipeline.rs"
test = false
doc = false

[[bin]]
name = "fuzz_closed_form_solver"
path = "fuzz_targets/fuzz_closed_form_solver.rs"
test = false
doc = false

[[bin]]
name = "fuzz_graph_algorithms"
path = "fuzz_targets/fuzz_graph_algorithms.rs"
test = false
doc = false

[[bin]]
name = "fuzz_reduce"
path = "fuzz_targets/fuzz_reduce.rs"
test = false
doc = false

# --- Legacy targets ---

[[bin]]
name = "fuzz_jit_compiler"
path = "fuzz_targets/fuzz_jit_compiler.rs"
test = false
doc = false

[[bin]]
name = "fuzz_solver_inputs"
path = "fuzz_targets/fuzz_solver_inputs.rs"
test = false
doc = false

[[bin]]
name = "fuzz_sparse_ops"
path = "fuzz_targets/fuzz_sparse_ops.rs"
test = false
doc = false

[[bin]]
name = "fuzz_jacobian"
path = "fuzz_targets/fuzz_jacobian.rs"
test = false
doc = false

[[bin]]
name = "fuzz_constraint_system_legacy"
path = "fuzz_targets/fuzz_constraint_system_legacy.rs"
test = false
doc = false
```

### 4. Update .gitignore

Add to the workspace `.gitignore`:

```
# Fuzz corpus and artifacts
**/fuzz/corpus/
**/fuzz/artifacts/
```

## Fuzz Targets

### Target 1: V3 `ConstraintSystem` (CRITICAL)

**File:** `fuzz/fuzz_targets/fuzz_v3_constraint_system.rs`

This is the single most important fuzz target. It exercises the generational ID
system, free-list allocation, and the full add/remove/solve lifecycle. The key
bugs we are hunting:

- **Stale `EntityId`**: Remove entity, add new entity in same slot, use old ID
- **Stale `ConstraintId`**: Same pattern for constraints
- **Stale `ParamId`**: Entity removed (params freed), constraint still holds old `ParamId`
- **Double free**: Remove same entity/constraint twice
- **Use after structural change**: Solve, then mutate, then access stale results

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::system::ConstraintSystem;
use solverang::id::{EntityId, ConstraintId, ParamId};
use solverang::entity::Entity;
use solverang::constraint::Constraint;
use solverang::param::ParamStore;

#[derive(Debug, Arbitrary)]
enum SystemOp {
    AddPoint { x: f64, y: f64 },
    AddCircle { cx: f64, cy: f64, r: f64 },
    RemoveEntity { idx: u8 },
    AddDistance { e1: u8, e2: u8, dist: f64 },
    AddCoincident { e1: u8, e2: u8 },
    AddFixed { e: u8 },
    AddHorizontal { e1: u8, e2: u8 },
    RemoveConstraint { idx: u8 },
    Solve,
    SolveIncremental,
    SetParam { entity_idx: u8, param_offset: u8, value: f64 },
    FixParam { entity_idx: u8, param_offset: u8 },
    UnfixParam { entity_idx: u8, param_offset: u8 },
    QueryDof,
    QueryClusterCount,
    RemoveAndReaddEntity { idx: u8, x: f64, y: f64 },
    Diagnose,
}

fuzz_target!(|ops: Vec<SystemOp>| {
    if ops.len() > 100 { return; }

    let mut system = ConstraintSystem::new();
    let mut entity_ids: Vec<EntityId> = Vec::new();
    let mut constraint_ids: Vec<ConstraintId> = Vec::new();

    for op in ops {
        match op {
            SystemOp::AddPoint { x, y } => {
                if entity_ids.len() >= 30 { continue; }
                let x = x.clamp(-1e6, 1e6);
                let y = y.clamp(-1e6, 1e6);
                if !x.is_finite() || !y.is_finite() { continue; }

                // Allocate entity, params, add -- the full lifecycle
                let eid = system.alloc_entity_id();
                let px = system.alloc_param(x, eid);
                let py = system.alloc_param(y, eid);
                // Build a minimal Entity impl and add it
                // (would use sketch2d::Point2D in real target)
                entity_ids.push(eid);
            }
            SystemOp::RemoveEntity { idx } => {
                if entity_ids.is_empty() { continue; }
                let i = (idx as usize) % entity_ids.len();
                let eid = entity_ids[i];
                system.remove_entity(eid);
                // Intentionally do NOT remove from entity_ids -- tests stale ID handling
            }
            SystemOp::RemoveAndReaddEntity { idx, x, y } => {
                // This specifically tests free-list slot reuse + generation bumps
                if entity_ids.is_empty() { continue; }
                let i = (idx as usize) % entity_ids.len();
                let old_eid = entity_ids[i];
                system.remove_entity(old_eid);

                let x = x.clamp(-1e6, 1e6);
                let y = y.clamp(-1e6, 1e6);
                if !x.is_finite() || !y.is_finite() { continue; }

                let new_eid = system.alloc_entity_id();
                let px = system.alloc_param(x, new_eid);
                let py = system.alloc_param(y, new_eid);
                entity_ids[i] = new_eid;
                // old_eid should now be stale (different generation)
            }
            SystemOp::RemoveConstraint { idx } => {
                if constraint_ids.is_empty() { continue; }
                let i = (idx as usize) % constraint_ids.len();
                system.remove_constraint(constraint_ids[i]);
            }
            SystemOp::Solve => {
                let _ = system.solve();
            }
            SystemOp::SolveIncremental => {
                let _ = system.solve_incremental();
            }
            SystemOp::QueryDof => {
                let _ = system.degrees_of_freedom();
            }
            SystemOp::QueryClusterCount => {
                let _ = system.cluster_count();
            }
            SystemOp::Diagnose => {
                let _ = system.diagnose();
            }
            _ => { /* other ops wired similarly */ }
        }
    }
});
```

**Key behaviors to stress:**

- **Generational IDs** (`id.rs`): Remove entity, add new entity in same slot, use stale ID
  to access params. The system should either panic deterministically or return an error,
  never silently return wrong data.
- **ParamStore free-list** (`param/store.rs`): Alloc/dealloc cycles. A freed `ParamId`
  with generation N should not be usable after a new param with generation N+1 is allocated
  in the same slot.
- **ChangeTracker** (`dataflow/tracker.rs`): Interleave structural changes (add/remove
  entities and constraints) with value changes (set_param). Verify incremental solve
  correctly identifies dirty clusters.
- **SolutionCache** (`dataflow/cache.rs`): Cache entries must be invalidated on structural
  changes. Stale cache entries should not corrupt subsequent solves.

### Target 2: `ParamStore` Free-List (CRITICAL)

**File:** `fuzz/fuzz_targets/fuzz_param_store.rs`

Isolated fuzzing of `ParamStore` with rapid alloc/free cycles to stress the free-list
and generation counter logic.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::param::ParamStore;
use solverang::id::{EntityId, ParamId};

#[derive(Debug, Arbitrary)]
enum StoreOp {
    Alloc { value: f64 },
    Free { idx: u8 },
    Get { idx: u8 },
    Set { idx: u8, value: f64 },
    Fix { idx: u8 },
    Unfix { idx: u8 },
    BuildMapping,
    ExtractFreeValues,
    Snapshot,
    FreeAndRealloc { idx: u8, new_value: f64 },
    UseStaleId { idx: u8 },
}

fuzz_target!(|ops: Vec<StoreOp>| {
    if ops.len() > 200 { return; }

    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let mut live_ids: Vec<ParamId> = Vec::new();
    let mut stale_ids: Vec<ParamId> = Vec::new();

    for op in ops {
        match op {
            StoreOp::Alloc { value } => {
                if live_ids.len() >= 50 { continue; }
                let v = if value.is_finite() { value.clamp(-1e6, 1e6) } else { 0.0 };
                let id = store.alloc(v, owner);
                live_ids.push(id);
            }
            StoreOp::Free { idx } => {
                if live_ids.is_empty() { continue; }
                let i = (idx as usize) % live_ids.len();
                let id = live_ids.remove(i);
                store.free(id);
                stale_ids.push(id); // Track for stale-ID testing
            }
            StoreOp::Get { idx } => {
                if live_ids.is_empty() { continue; }
                let i = (idx as usize) % live_ids.len();
                let _ = store.get(live_ids[i]); // Must not panic for live IDs
            }
            StoreOp::Set { idx, value } => {
                if live_ids.is_empty() { continue; }
                let i = (idx as usize) % live_ids.len();
                let v = if value.is_finite() { value } else { 0.0 };
                store.set(live_ids[i], v);
            }
            StoreOp::FreeAndRealloc { idx, new_value } => {
                // Free then immediately realloc in same slot
                if live_ids.is_empty() { continue; }
                let i = (idx as usize) % live_ids.len();
                let old_id = live_ids[i];
                store.free(old_id);
                stale_ids.push(old_id);
                let v = if new_value.is_finite() { new_value.clamp(-1e6, 1e6) } else { 0.0 };
                let new_id = store.alloc(v, owner);
                live_ids[i] = new_id;
                // old_id.index == new_id.index but old_id.generation != new_id.generation
            }
            StoreOp::UseStaleId { idx } => {
                // Deliberately access a freed ID -- should panic
                if stale_ids.is_empty() { continue; }
                let i = (idx as usize) % stale_ids.len();
                let stale = stale_ids[i];
                let _ = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| store.get(stale))
                );
                // Must either panic or return an error, never silently succeed
                // with wrong-generation data
            }
            StoreOp::BuildMapping => {
                let _ = store.build_solver_mapping();
            }
            StoreOp::ExtractFreeValues => {
                let mapping = store.build_solver_mapping();
                let _ = store.extract_free_values(&mapping);
            }
            StoreOp::Snapshot => {
                let _ = store.snapshot();
            }
            _ => {}
        }
    }
});
```

### Target 3: V3 Pipeline (HIGH)

**File:** `fuzz/fuzz_targets/fuzz_pipeline.rs`

The 5-phase pipeline (`Decompose -> Analyze -> Reduce -> Solve -> PostProcess`) should
handle any constraint graph topology without panicking. This target focuses on:

- Over-constrained, under-constrained, and contradictory systems
- Empty systems and single-constraint systems
- Structural changes between solves (incremental vs full re-decompose)
- Pipeline caching correctness (cached cluster reuse)
- Custom pipeline phase combinations via `PipelineBuilder`

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Debug, Arbitrary)]
struct PipelineFuzzInput {
    /// Number of points (entities) to create (1-15).
    num_points: u8,
    /// Initial coordinates for each point.
    coords: Vec<(f64, f64)>,
    /// Constraints to apply between points.
    constraints: Vec<PipelineConstraint>,
    /// Operations to perform after initial solve.
    mutations: Vec<PipelineMutation>,
}

#[derive(Debug, Arbitrary)]
enum PipelineConstraint {
    Distance { p1: u8, p2: u8, dist: f64 },
    Coincident { p1: u8, p2: u8 },
    Fixed { p: u8 },
    Horizontal { p1: u8, p2: u8 },
    Vertical { p1: u8, p2: u8 },
}

#[derive(Debug, Arbitrary)]
enum PipelineMutation {
    AddConstraint(PipelineConstraint),
    RemoveLastConstraint,
    MoveParam { point: u8, dx: f64, dy: f64 },
    SolveAgain,
    SolveIncremental,
    InvalidatePipeline,
}

fuzz_target!(|input: PipelineFuzzInput| {
    let num_points = (input.num_points as usize).clamp(1, 15);
    if input.coords.len() < num_points { return; }
    if input.constraints.len() > 40 { return; }
    if input.mutations.len() > 30 { return; }

    let mut system = solverang::ConstraintSystem::new();

    // Build entities
    // ... (allocate entity IDs, params, add entities)

    // Add initial constraints
    // ... (wire up constraints with clamped finite values)

    // First solve -- must not panic
    let _ = system.solve();

    // Apply mutations: add/remove constraints, move params, re-solve
    for mutation in &input.mutations {
        match mutation {
            PipelineMutation::SolveAgain => { let _ = system.solve(); }
            PipelineMutation::SolveIncremental => { let _ = system.solve_incremental(); }
            // ... handle other mutations
            _ => {}
        }
    }
});
```

### Target 4: Closed-Form Solvers (HIGH)

**File:** `fuzz/fuzz_targets/fuzz_closed_form_solver.rs`

`solve/closed_form.rs` contains analytical solvers for matched patterns. These have
precise mathematical edge cases that must be handled gracefully:

- **Circle-circle intersection** (`TwoDistances`): Two coincident circles (infinite
  intersections), non-intersecting circles (no solution), one circle inside the other
  (no solution), tangent circles (one solution), zero-radius circles.
- **Scalar Newton step** (`ScalarSolve`): Zero Jacobian entry (division by zero),
  near-zero Jacobian (huge step), NaN/infinity in residual.
- **Polar-to-cartesian** (`DistanceAngle`): Zero distance, angle at boundary values
  (0, pi, -pi, 2*pi), negative distance.
- **Horizontal/Vertical** (`HorizontalVertical`): NaN coordinates, extreme values.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::solve::closed_form::{solve_pattern, ClosedFormResult};
use solverang::graph::pattern::{MatchedPattern, PatternKind};
use solverang::param::ParamStore;
use solverang::id::{EntityId, ParamId};

#[derive(Debug, Arbitrary)]
struct ClosedFormFuzzInput {
    pattern_kind: u8,
    param_values: Vec<f64>,
    target_values: Vec<f64>,
}

fuzz_target!(|input: ClosedFormFuzzInput| {
    if input.param_values.is_empty() || input.param_values.len() > 10 { return; }

    // Build a ParamStore with the fuzzed values
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let param_ids: Vec<ParamId> = input.param_values.iter()
        .map(|&v| {
            let v = if v.is_finite() { v.clamp(-1e8, 1e8) } else { 0.0 };
            store.alloc(v, owner)
        })
        .collect();

    // Build MatchedPattern and corresponding test constraints
    let kind = match input.pattern_kind % 4 {
        0 => PatternKind::ScalarSolve,
        1 => PatternKind::TwoDistances,
        2 => PatternKind::HorizontalVertical,
        _ => PatternKind::DistanceAngle,
    };

    let pattern = MatchedPattern {
        kind,
        entity_ids: vec![EntityId::new(0, 0)],
        constraint_indices: vec![0],
        param_ids: param_ids.clone(),
    };

    // Build minimal constraint(s) matching the pattern
    // ... (construct appropriate constraints from fuzz input)

    // Call solve_pattern -- must return Some/None, never panic
    // let _ = solve_pattern(&pattern, &constraints, &store);
});
```

### Target 5: Graph Algorithms (MEDIUM)

**File:** `fuzz/fuzz_targets/fuzz_graph_algorithms.rs`

Fuzzes the constraint graph (`graph/bipartite.rs`), cluster decomposition
(`graph/decompose.rs`), DOF analysis (`graph/dof.rs`), redundancy detection
(`graph/redundancy.rs`), and pattern matching (`graph/pattern.rs`) with
degenerate inputs:

- Empty graphs (no entities, no constraints)
- Disconnected components (many isolated entities)
- Self-referencing constraints (entity constrained against itself)
- Duplicate constraints (same constraint added twice)
- Extremely dense graphs (every entity connected to every other)
- Single-entity systems
- Constraints referencing non-existent entities

The DOF analysis (`graph/dof.rs`) uses SVD via `nalgebra::DMatrix`, which should be
tested with:
- Rank-deficient Jacobians (more equations than independent directions)
- All-zero Jacobians (no constraints actually constrain anything)
- Very large condition numbers (near-singular Jacobians)

The redundancy analysis (`graph/redundancy.rs`) uses incremental rank testing and
should be tested with:
- Known redundant constraint sets (e.g., three distance constraints on a 2-DOF point)
- Conflicting constraints (e.g., distance=5 and distance=10 between same points)
- Near-redundant constraints (numerically close but not exactly dependent)

### Target 6: Reduce Passes (MEDIUM)

**File:** `fuzz/fuzz_targets/fuzz_reduce.rs`

The three symbolic reduction passes transform the constraint system before numerical
solving. Each can potentially corrupt the problem if the transformation logic has bugs:

- **`reduce/substitute.rs`** (fixed-parameter elimination): If a parameter is marked
  fixed, all constraints referencing it should have that parameter substituted with its
  constant value. Edge cases: all params fixed (nothing to solve), no params fixed
  (nothing to substitute), constraint becomes trivially satisfied after substitution.

- **`reduce/merge.rs`** (coincident parameter merging): When a coincident constraint
  enforces `param_a == param_b`, one is replaced by the other everywhere. Edge cases:
  chains of merges (a==b, b==c => a==c), self-merge (a==a), merge creates a cycle in
  the substitution map.

- **`reduce/eliminate.rs`** (trivial constraint detection): Single-equation constraints
  with exactly one free parameter can be solved analytically. Edge cases: zero Jacobian
  (unsolvable), multiple free params (not trivial), constraint already satisfied.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Debug, Arbitrary)]
struct ReduceFuzzInput {
    num_params: u8,
    fixed_params: Vec<u8>,
    coincident_pairs: Vec<(u8, u8)>,
    single_eq_constraints: Vec<(u8, f64)>,
}

fuzz_target!(|input: ReduceFuzzInput| {
    let n = (input.num_params as usize).clamp(1, 20);
    // Build a ParamStore with n params, mark some fixed
    // Build coincident constraints for merge testing
    // Build single-equation constraints for eliminate testing
    // Run each reduce pass -- must not panic
    // Verify: param count after reduction <= param count before
});
```

### Target 7: JIT Compiler (HIGH -- Legacy)

**File:** `fuzz/fuzz_targets/fuzz_jit_compiler.rs`

This target exercises the legacy `unsafe` JIT code path. It builds arbitrary constraint
systems using the legacy `ConstraintSystem<2>` API and evaluates residuals and Jacobians,
which go through the Cranelift JIT when the `jit` feature is enabled.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::geometry::{ConstraintSystem, Point2D};
use solverang::geometry::constraints::*;

#[derive(Debug, Arbitrary)]
struct FuzzConstraints {
    points: Vec<(f64, f64)>,
    constraint_ops: Vec<ConstraintOp>,
}

#[derive(Debug, Arbitrary)]
enum ConstraintOp {
    Distance { p1: u8, p2: u8, dist: f64 },
    Coincident { p1: u8, p2: u8 },
    Horizontal { p1: u8, p2: u8 },
    Vertical { p1: u8, p2: u8 },
    Fixed { p: u8 },
}

fuzz_target!(|input: FuzzConstraints| {
    if input.points.is_empty() || input.points.len() > 20 { return; }
    if input.constraint_ops.is_empty() || input.constraint_ops.len() > 30 { return; }

    let num_points = input.points.len();
    let mut system = ConstraintSystem::<2>::new();

    for &(x, y) in &input.points {
        let x = x.clamp(-1e6, 1e6);
        let y = y.clamp(-1e6, 1e6);
        if !x.is_finite() || !y.is_finite() { return; }
        system.add_point(Point2D::new(x, y));
    }

    for op in &input.constraint_ops {
        match op {
            ConstraintOp::Distance { p1, p2, dist } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j && dist.is_finite() && *dist > 0.0 {
                    system.add_constraint(Box::new(DistanceConstraint::<2>::new(i, j, *dist)));
                }
            }
            ConstraintOp::Coincident { p1, p2 } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j {
                    system.add_constraint(Box::new(CoincidentConstraint::<2>::new(i, j)));
                }
            }
            ConstraintOp::Horizontal { p1, p2 } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j {
                    system.add_constraint(Box::new(HorizontalConstraint::new(i, j)));
                }
            }
            ConstraintOp::Vertical { p1, p2 } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j {
                    system.add_constraint(Box::new(VerticalConstraint::new(i, j)));
                }
            }
            ConstraintOp::Fixed { p } => {
                let i = (*p as usize) % num_points;
                let pt = Point2D::new(
                    input.points[i].0.clamp(-1e6, 1e6),
                    input.points[i].1.clamp(-1e6, 1e6),
                );
                system.add_constraint(Box::new(FixedConstraint::<2>::new(i, pt)));
            }
        }
    }

    // Evaluate residuals and Jacobian -- these go through JIT if enabled
    let x = system.current_values();
    let _residuals = system.residuals(&x);
    let _jacobian = system.jacobian(&x);

    // Also try solving -- must not segfault
    use solverang::{LMSolver, LMConfig};
    let solver = LMSolver::new(LMConfig { max_iterations: 20, ..LMConfig::default() });
    let _result = solver.solve(&system, &x);
});
```

### Target 8: Legacy Solver with Arbitrary Inputs

**File:** `fuzz/fuzz_targets/fuzz_solver_inputs.rs`

Tests the legacy `Problem` trait solvers (Newton-Raphson, Levenberg-Marquardt) with
arbitrary `f64` inputs including NaN, infinity, subnormals, and extreme magnitudes.

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::{LMSolver, LMConfig, Problem, SolveResult};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    targets: Vec<f64>,
    x0: Vec<f64>,
    max_iterations: u16,
    patience: u8,
}

struct FuzzLinearProblem {
    target: Vec<f64>,
}

impl Problem for FuzzLinearProblem {
    fn name(&self) -> &str { "fuzz-linear" }
    fn residual_count(&self) -> usize { self.target.len() }
    fn variable_count(&self) -> usize { self.target.len() }
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.target).map(|(a, b)| a - b).collect()
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        (0..self.target.len()).map(|i| (i, i, 1.0)).collect()
    }
    fn initial_point(&self, f: f64) -> Vec<f64> { vec![f; self.target.len()] }
}

fuzz_target!(|input: FuzzInput| {
    if input.targets.is_empty() || input.targets.len() > 20 { return; }
    if input.x0.len() != input.targets.len() { return; }

    let problem = FuzzLinearProblem { target: input.targets };
    let config = LMConfig {
        max_iterations: input.max_iterations as usize,
        patience: input.patience as usize,
        ..LMConfig::default()
    };
    let solver = LMSolver::new(config);

    // Must not panic -- any result variant is acceptable
    let _result = solver.solve(&problem, &input.x0);
});
```

### Target 9: Sparse Matrix Operations (LOW)

**File:** `fuzz/fuzz_targets/fuzz_sparse_ops.rs`

Feeds arbitrary sparse triplet `(row, col, value)` entries into the sparse Jacobian
path. Tests out-of-bounds indices, duplicate entries, empty matrices, and extreme values.

### Target 10: Jacobian Computation (LOW)

**File:** `fuzz/fuzz_targets/fuzz_jacobian.rs`

Tests `jacobian_dense()`, `verify_jacobian()`, and finite-difference computation with
arbitrary problem dimensions and evaluation points. Verifies no panics from dimension
mismatches or non-finite values.

### Target 11: Legacy Constraint System Builder (LOW)

**File:** `fuzz/fuzz_targets/fuzz_constraint_system_legacy.rs`

Exercises the legacy `ConstraintSystemBuilder<2>` API with arbitrary combinations of
points, constraints, and fixed/free designations. Tests that the system never panics
regardless of constraint configuration.

## Structured Fuzzing with `arbitrary`

Instead of interpreting raw bytes, derive `Arbitrary` for structured inputs.
This dramatically improves fuzzing efficiency -- the fuzzer explores meaningful
input variations instead of wasting time on structurally invalid byte sequences.

### V3-specific structured types

```rust
/// Structured operations on the V3 ConstraintSystem.
#[derive(Debug, Arbitrary)]
enum V3Operation {
    /// Sketch2D builder operations (high-level API).
    Sketch2D(Sketch2DOp),
    /// Sketch3D operations.
    Sketch3D(Sketch3DOp),
    /// Assembly operations.
    Assembly(AssemblyOp),
    /// Raw system operations (low-level API).
    Raw(RawSystemOp),
}

#[derive(Debug, Arbitrary)]
enum Sketch2DOp {
    AddPoint { x: f64, y: f64 },
    AddFixedPoint { x: f64, y: f64 },
    AddLine { x1: f64, y1: f64, x2: f64, y2: f64 },
    AddCircle { cx: f64, cy: f64, r: f64 },
    ConstrainDistance { e1: u8, e2: u8, dist: f64 },
    ConstrainCoincident { e1: u8, e2: u8 },
    ConstrainHorizontal { e1: u8, e2: u8 },
    ConstrainVertical { e1: u8, e2: u8 },
    ConstrainParallel { l1: u8, l2: u8 },
    ConstrainPerpendicular { l1: u8, l2: u8 },
    ConstrainAngle { l1: u8, l2: u8, angle: f64 },
    ConstrainMidpoint { p: u8, l: u8 },
    ConstrainSymmetric { p1: u8, p2: u8, axis: u8 },
    ConstrainEqualLength { l1: u8, l2: u8 },
    ConstrainPointOnCircle { p: u8, c: u8 },
    ConstrainTangentLineCircle { l: u8, c: u8 },
    ConstrainTangentCircleCircle { c1: u8, c2: u8 },
    BuildAndSolve,
}

#[derive(Debug, Arbitrary)]
enum RawSystemOp {
    AllocEntity,
    AllocParam { value: f64, entity_idx: u8 },
    AddEntity { idx: u8 },
    RemoveEntity { idx: u8 },
    AllocConstraint,
    AddConstraint { idx: u8 },
    RemoveConstraint { idx: u8 },
    Solve,
}
```

### Legacy structured types

```rust
#[derive(Debug, Arbitrary)]
struct SolverFuzzInput {
    problem_type: ProblemType,
    dimensions: u8,
    values: Vec<f64>,
    config: FuzzConfig,
}

#[derive(Debug, Arbitrary)]
enum ProblemType {
    Linear,
    Quadratic,
    Coupled,
    Rosenbrock,
}

#[derive(Debug, Arbitrary)]
struct FuzzConfig {
    max_iterations: u16,
    tolerance: f64,
}
```

## Corpus Management

### Seed from existing tests

```bash
# Create seed corpus from existing test inputs
mkdir -p fuzz/corpus/fuzz_v3_constraint_system
mkdir -p fuzz/corpus/fuzz_param_store
mkdir -p fuzz/corpus/fuzz_pipeline
mkdir -p fuzz/corpus/fuzz_closed_form_solver
mkdir -p fuzz/corpus/fuzz_jit_compiler
mkdir -p fuzz/corpus/fuzz_solver_inputs

# Generate seeds programmatically from test problems
cargo test --features nist -- --nocapture generate_fuzz_seeds 2>/dev/null
```

For V3 targets, write a helper that serializes the operation sequences from existing
`system.rs`, `pipeline/mod.rs`, and `sketch2d/builder.rs` tests into the corpus
directory. This seeds the fuzzer with known-good operation patterns to mutate from.

### Corpus hygiene

```bash
# Minimize corpus after a fuzzing session
cargo fuzz cmin fuzz_v3_constraint_system
cargo fuzz cmin fuzz_param_store
cargo fuzz cmin fuzz_pipeline
cargo fuzz cmin fuzz_closed_form_solver
cargo fuzz cmin fuzz_jit_compiler
```

Commit the minimized corpus to the repository so CI runs start from a good baseline.

## CI Integration

### Time-boxed fuzz runs in nightly CI

```yaml
# In .github/workflows/nightly.yml
fuzz:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      target:
        # Critical targets get the most time
        - fuzz_v3_constraint_system
        - fuzz_param_store
        # High priority
        - fuzz_pipeline
        - fuzz_closed_form_solver
        - fuzz_jit_compiler
        # Medium priority
        - fuzz_graph_algorithms
        - fuzz_reduce
        # Legacy
        - fuzz_solver_inputs
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@nightly
    - run: cargo install cargo-fuzz
    - run: |
        cd crates/solverang
        # Critical targets get 20 min, others get 10 min
        if [[ "${{ matrix.target }}" == fuzz_v3_* ]] || \
           [[ "${{ matrix.target }}" == fuzz_param_store ]]; then
          TIMEOUT=1200
        else
          TIMEOUT=600
        fi
        cargo fuzz run ${{ matrix.target }} -- \
          -max_total_time=$TIMEOUT \
          -jobs=4 \
          -workers=4
    - uses: actions/upload-artifact@v4
      if: failure()
      with:
        name: fuzz-artifacts-${{ matrix.target }}
        path: crates/solverang/fuzz/artifacts/
```

Total CI time per nightly run: approximately 80 minutes across 8 targets.

## Triage Process

1. **Crash found**: `cargo fuzz` saves crashing input to `fuzz/artifacts/<target>/`
2. **Reproduce**: `cargo fuzz run <target> fuzz/artifacts/<target>/<crash-file>`
3. **Minimize**: `cargo fuzz tmin <target> fuzz/artifacts/<target>/<crash-file>`
4. **Diagnose**:
   - For JIT crashes: Run under ASAN/MSAN for memory safety bugs
   - For V3 stale-ID bugs: Check generation counters in ParamStore/ConstraintSystem
   - For pipeline bugs: Check ChangeTracker state and cached cluster validity
   - For closed-form bugs: Check edge case math (division by zero, sqrt of negative)
   - For graph bugs: Check index bounds and empty-graph handling
5. **Fix**: Patch the code
6. **Regression test**: Add minimized input as a regular `#[test]`:
   ```rust
   #[test]
   fn regression_crash_xyz() {
       let input = include_bytes!("../fuzz/regressions/crash_xyz");
       // Parse and run the input, assert no panic
   }
   ```
7. **Add to corpus**: Keep the minimized crash input in `fuzz/corpus/` so the fuzzer
   starts from this known-interesting input.

### V3-specific triage considerations

For V3 stale-ID bugs, the triage must verify:
- Is the generation counter being incremented correctly on slot reuse?
- Is the `entry()` / `entry_mut()` method correctly checking `alive && generation`?
- Is there a code path that bypasses the generation check (e.g., using `raw_index()`
  directly)?
- Does the `ChangeTracker` correctly classify the change as structural?

## Success Metrics

| Metric | Target |
|--------|--------|
| Fuzz targets implemented | 11 (6 V3 + 5 legacy) |
| Hours of fuzzing without new crashes | 100+ per target |
| V3 generational ID coverage | All alloc/free/reuse paths reached |
| JIT-specific coverage | All `unsafe` blocks reached |
| Pipeline phase coverage | All 5 phases exercised in every run |
| Corpus size per target | 500+ minimized inputs |
| CI fuzz time per nightly | ~80 minutes |

## Estimated Effort

| Task | Time |
|------|------|
| Install and initialize cargo-fuzz | 30 min |
| Write 6 V3 fuzz targets (system, paramstore, pipeline, closed-form, graph, reduce) | 8-10 hours |
| Write 5 legacy fuzz targets (JIT, solver, constraint system, sparse, jacobian) | 4-6 hours |
| Create seed corpus from existing V3 tests | 2 hours |
| Create seed corpus from existing legacy tests | 1 hour |
| CI integration | 1 hour |
| Initial fuzzing campaign (local) | 12+ hours of wall time |
| Document triage process | 1 hour |
| **Total active effort** | **~18-22 hours** |
