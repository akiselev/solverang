# AutoPCB Routing Integration Boundary

Date: 2026-08-19

Solverang should support AutoPCB as a continuous-refinement engine, not as a PCB route authority.

## Intended role

AutoPCB owns:

- discrete topology;
- layer and via-stack legality;
- route transactions;
- exact DRC and certificates;
- final route acceptance.

Solverang owns local continuous optimization problems after AutoPCB has selected a discrete route structure.

Examples:

- bend coordinate relaxation;
- length and skew tuning;
- differential-pair symmetry residuals;
- smooth clearance-margin objectives;
- local placement-route co-optimization;
- meander pitch and amplitude solving;
- via-position micro-adjustment inside a legal window.

## Boundary contract

The AutoPCB adapter should compile a verified route fragment into a Solverang problem:

```text
RouteFragment + RefinementWindow + ResidualSpec
    -> SolverangProblem
    -> CandidateGeometryDelta
    -> AutoPCB RouteTransaction
    -> verified commit or rejection
```

Solverang output must never be written directly to a PCB route.  It is a candidate delta only.

## Required Solverang support

The useful API surface for AutoPCB is:

1. deterministic solve configuration and seed recording;
2. residual diagnostics suitable for route-event logs;
3. sparse Jacobian support for many local variables;
4. bounded variables or projection hooks for legal geometric windows;
5. finite-difference verification for generated Jacobians;
6. an adapter-friendly problem/result serialization format.

## Non-goals

Solverang should not model:

- PCB rule precedence;
- copper primitive pair DRC;
- via-stack fabrication semantics;
- global net ordering;
- route acceptance.

Those remain AutoPCB route-kernel responsibilities.

## First implementation slice

Add an `autopcb-refinement` example problem once AutoPCB exposes typed route-fragment windows.  Until then, this document is the integration contract for keeping the two projects aligned.
