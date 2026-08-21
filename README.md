# Solverang

Solverang is a domain-neutral constraint engine. It owns variable and
constraint graphs, equality/inequality activation, candidate solving, rank and
degree-of-freedom analysis, conflict diagnostics, and reusable geometric
constraint vocabulary. Methodus supplies its numerical algorithms.

Consumers retain semantic authority:

- CADabra maps CAD entities and dimensional values into Solverang, then
  independently certifies candidates before committing geometry or topology.
- A vector editor maps document nodes, handles, and guides into the same 2-D
  primitives while retaining document and interaction policy.
- AutoPCB maps route control points, keep-outs, and clearance rules into the
  same 2-D primitives while retaining board topology and manufacturing rules.

Solverang does not contain CAD topology, document models, PCB rules, meshes,
scientific fields, or simulation algorithms.

## Packages

- `solverang`: consumer-neutral constraint graph, relations, solve
  orchestration, derivative checking, and diagnostics.
- `solverang-geometry-2d`: points, segments, circles, fixed/coincident,
  distance/clearance, horizontal/vertical, parallel/perpendicular, and
  point-on-curve constraints.
- `solverang-geometry-3d`: points, segments, planes, fixed/coincident,
  distance/clearance, point-on-plane, and parallel/perpendicular constraints.

The geometry crates carry only Solverang variable IDs and numeric targets.
Consumer entity identity, units, persistence, and authoritative acceptance stay
outside the crates.

## Example

```rust
use solverang::{ConstraintSolveConfig, ConstraintStatus, solve_constraints};
use solverang_geometry_2d::{Distance, FixedPoint, Geometry2d};

let mut geometry = Geometry2d::new();
let left = geometry.add_point(0.0, 0.0)?;
let right = geometry.add_point(2.5, 0.0)?;
geometry.constrain(FixedPoint { point: left, value: [0.0, 0.0] })?;
geometry.constrain(FixedPoint { point: right, value: [3.0, 0.0] })?;
geometry.constrain(Distance { left, right, distance: 3.0 })?;

let report = solve_constraints(geometry.model(), &ConstraintSolveConfig::default())?;
assert_eq!(report.status, ConstraintStatus::Solved);
# Ok::<(), solverang::ConstraintError>(())
```

## Authority boundary

A converged Solverang report is a finite-precision candidate, not proof of CAD
incidence, collision-free routing, manufacturability, or document validity.
Each consumer re-evaluates its own semantics and either accepts, repairs,
escalates, or refuses the candidate.

## Validation

```text
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --all-targets
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
cargo test --locked --workspace --doc
```
