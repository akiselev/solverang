# Solverang status

Updated: 2026-08-21
Branch: `master`
Milestone: generalized constraint-engine restoration

## Current role

Solverang owns domain-neutral constraint graphs, candidate solve orchestration,
and constraint diagnostics. It depends on Methodus for numerical algorithms.
It contains no simulation solver algorithms, CAD topology or certification,
vector-document model, PCB topology/manufacturing rules, scientific semantics,
mesh, or product policy.

The extraction dependency validated here is Methodus
`d5354abb4dfd197ba5fd66f3742f9820701e4c43`.

## Implemented surface

- stable `VariableId` and `ConstraintId` handles;
- extensible object-safe residual-block `Constraint` interface;
- equality, less-than-or-equal, and greater-than-or-equal relations;
- rectangular model evaluation with row provenance and analytic Jacobians;
- Methodus damped least-squares candidate solving;
- centered-difference derivative verification;
- rank, remaining-degree-of-freedom, and conflicting-constraint diagnostics;
- `solverang-geometry-2d` with shared CAD/vector/PCB-oriented primitives;
- `solverang-geometry-3d` with point/segment/plane primitives.

Acceptance cases exercise one CAD/vector profile, a generic PCB-clearance
inequality, and a 3-D point-on-plane candidate. These demonstrate dissimilar
consumers at the vocabulary boundary; they do not claim integration with an
external vector editor, AutoPCB, or CADabra yet.

## Extraction

The former numerical implementation moved into Methodus from shared Solverang
history. Solverang no longer exports Methodus traits or algorithms and no
compatibility facade remains. Historical broad sketch/assembly/pipeline code
was not restored wholesale; Git history remains the donor archive.

## Validation

Validated locally on 2026-08-21. Standalone CI now recreates the declared
sibling layout by checking out pinned Methodus `d29dae4` next to Solverang;
the previous workflow failed before formatting because `../methodus` was
absent from the isolated runner checkout.

- formatting and locked workspace/all-target checks passed;
- warnings-denied Clippy passed;
- 3 geometry acceptance tests passed, 0 failed;
- warnings-denied rustdoc and all workspace doctests passed;
- `git diff --check` passed.

## Next concrete work

1. Add consumer-owned identity adapters only with the first live CADabra,
   vector-editor, and AutoPCB integrations.
2. Add graph decomposition and incremental solving from representative editing
   sessions, preserving deterministic diagnostics.
3. Add curves, tangency, angle, rigid-frame, and assembly constraints only with
   independent derivative checks and dissimilar consumers.

Blockers: none.
