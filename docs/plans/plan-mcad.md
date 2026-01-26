# MCAD Geometric Kernel: High-Level Implementation Plan

## Overview

This plan defines a **six-layer geometric kernel** for MCAD (Mechanical CAD) with full B-Rep and NURBS support, built on top of the existing `atomic_solver` constraint system. The kernel is fully vertically integrated—we build all layers ourselves rather than depending on external topology libraries.

**Architecture approach:** Hybrid with Reference Implementation. We use OpenCASCADE (via `opencascade-rs`) as a test oracle for correctness verification, enabling AI agents to have hard pass/fail targets during development. Our implementation is independent but validated against known-good output.

**Key innovation:** Full differential-dataflow pipeline across all layers, enabling incremental updates not just for constraints (Layer 4-5) but experimentally for geometry and topology operations (Layers 1-3). This is a research direction—no production CAD kernel has attempted this.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|----------|-----------------|
| **Half-edge over winged-edge** | Half-edge provides O(1) queries for all navigation -> winged-edge requires orientation tracking during traversal -> half-edge is simpler to maintain invariants -> CGAL/OpenMesh/Fornjot all use half-edge variants -> more reference implementations for AI agents to learn from |
| **Integrate curvo for NURBS** | Curvo is actively maintained (777+ commits, MIT license) -> has working boolean ops on curves and intersection detection -> building NURBS from scratch is 3+ months -> curvo provides validated algorithms -> wrap in our API layer |
| **Full DD pipeline (research)** | DD excels at incremental computation -> Layers 4-5 (constraints, history) have proven DD benefit -> Layers 1-3 (geometry, topology, modeling) are novel DD application -> enables incremental boolean updates -> accept research risk for potential breakthrough |
| **OCC as test oracle** | OpenCASCADE is 24+ years mature with known-good output -> enables automated TDD (compare our output to OCC) -> AI agents get concrete pass/fail -> OCC is test dependency only, not runtime |
| **AP203 cc2a for STEP MVP** | AP203 cc2a is minimal conformance class (part + face + edge) -> AP242 is current but complex -> industry still uses AP203 for interop -> MVP is write-only, no parametric history |
| **Property tests + NIST + reference comparison** | Property tests catch edge cases automatically (proptest) -> NIST benchmarks provide certified accuracy targets -> OCC comparison catches algorithmic bugs -> three layers of verification for AI agent confidence |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| **truck-topology** | Explicitly rejected per user requirement. Building our own B-Rep for full control and DD integration research. |
| **Winged-edge B-Rep** | More complex traversal, error-prone updates, fewer reference implementations. Half-edge is industry standard. |
| **Build NURBS from scratch** | 3+ month effort for algorithms that curvo already implements well. Wrap curvo instead. |
| **Bottom-up sequential layers** | No visible output until Layer 6 complete (24+ months). Hybrid approach validates architecture earlier. |
| **DD for Layers 4-5 only** | User explicitly chose full DD pipeline as research direction. Accept higher risk for potential breakthrough. |
| **Fornjot integration** | Fornjot mainline paused (2024). Better to study design patterns and implement our own. |

### Constraints & Assumptions

**Technical:**
- Rust implementation (existing codebase commitment)
- Must integrate with Cadatomic event-sourcing architecture
- `atomic_solver` is Layer 4 foundation (already implemented)
- OpenCASCADE available via `opencascade-rs` for test oracle

**Test Infrastructure:**
- NIST StRD benchmarks (32 problems) - existing in atomic_solver
- NIST Manufacturing test cases - industrial parts with known dimensions
- ABC Dataset (1M CAD models) - curated subset for diversity testing
- CAx-IF STEP conformance tests - recommended practices for interop
- Property tests via proptest - invariant verification

**Dependencies:**
- `curvo` 0.x - NURBS curves/surfaces (Layer 1)
- `nalgebra` 0.33 - linear algebra
- `differential-dataflow` - incremental computation
- `opencascade-rs` - test oracle only

### Known Risks

| Risk | Mitigation | Anchor |
|------|------------|--------|
| **Full DD pipeline is research** | Accept novel territory. Fall back to DD for Layers 4-5 only if Layers 1-3 prove infeasible. Document findings. | N/A (research) |
| **Boolean operations are hard** | Use OCC reference comparison. Property tests for topology invariants. Extensive edge case testing. | N/A (new code) |
| **Curvo API mismatch** | Spike integration early (Phase 1.1). Build adapter layer between curvo types and our geometry API. | N/A (new code) |
| **Half-edge invariants** | Explicit invariant checks after every Euler operator. Property tests with proptest. | N/A (new code) |
| **STEP conformance complexity** | MVP is AP203 cc2a write-only. Resist scope creep to parametric history. | N/A (new code) |
| **OCC test oracle setup** | May require CMake/C++ toolchain for tests. Document setup clearly. Consider CI caching. | N/A (external) |

## Invisible Knowledge

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MCAD GEOMETRIC KERNEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Layer 6: I/O                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STEP Writer (AP203)  │  Tessellation  │  Future: IGES, Parasolid   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│   Layer 5: History & Regeneration (DD-based)                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Feature Tree  │  Dependency Graph  │  Incremental Regen  │  Undo   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│   Layer 4: Constraint Solving (atomic_solver)                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  2D Sketch  │  3D Assembly  │  DOF Analysis  │  Incremental Solve   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│   Layer 3: Modeling Operations                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Boolean  │  Fillet/Chamfer  │  Extrude  │  Revolve  │  Sweep/Loft  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│   Layer 2: Topology (Half-Edge B-Rep)                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Vertex  │  Edge  │  Wire  │  Face  │  Shell  │  Solid  │  Euler Ops│   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│   Layer 1: Geometry (curvo + custom)                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Point  │  Line  │  Arc  │  Circle  │  NURBS Curve  │  NURBS Surface│   │
│   │  Plane  │  Cylinder  │  Cone  │  Sphere  │  Torus  │  Intersection  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │     Differential Dataflow       │
                    │  (Incremental Update Pipeline)  │
                    └─────────────────────────────────┘
```

### Data Flow

```
User Edit (parameter change)
    │
    ▼
┌─────────────────┐
│ DD Input Stream │ ──── Change propagates through dataflow graph
└────────┬────────┘
         │
         ├──► Layer 1: Recompute affected geometry (NURBS eval)
         │
         ├──► Layer 2: Update topology if structure changed
         │
         ├──► Layer 3: Re-execute affected modeling ops (incremental boolean?)
         │
         ├──► Layer 4: Re-solve affected constraint components
         │
         ├──► Layer 5: Update feature tree, propagate downstream
         │
         ▼
┌─────────────────┐
│ DD Output Stream│ ──── Only changed geometry re-tessellated
└─────────────────┘
         │
         ▼
    Display Update
```

### Why This Structure

**Layer Separation:** Each layer has a distinct responsibility and can be tested in isolation. Layers 1-2 are pure geometry/topology (no application logic). Layer 3 transforms topology. Layer 4 solves constraints. Layer 5 manages history. Layer 6 serializes.

**DD Throughout:** Unlike traditional CAD (full regeneration on parameter change), DD enables computing only the delta. This is research territory for Layers 1-3 but has proven value for Layers 4-5.

**Topology Owns Geometry:** Following B-Rep convention, topological entities (Face, Edge, Vertex) hold references to geometric entities (Surface, Curve, Point). This is the inverse of atomic_solver's geometry-first approach—the kernel is a separate system that uses the solver as a component.

### Invariants

**Half-Edge B-Rep Invariants:**
- Every half-edge has a twin (twin.twin == self)
- Every half-edge has a next (forms closed loop per face)
- Every edge has exactly two half-edges
- Every face is bounded by at least one wire (outer loop)
- Inner wires (holes) have opposite orientation to outer wire
- Euler characteristic: V - E + F = 2 for closed solid

**DD Pipeline Invariants:**
- Changes propagate in topological sort order
- No cycles in dependency graph (feature tree is DAG)
- Downstream consumers see consistent snapshots

**Modeling Operation Invariants:**
- Boolean operations preserve manifold property (if inputs are manifold)
- Fillet/chamfer maintain tangent continuity (G1) with adjacent faces
- Extrude/revolve produce valid solids (closed, orientable)

### Tradeoffs

| Choice | Benefit | Cost |
|--------|---------|------|
| Full DD pipeline | Incremental updates, research breakthrough potential | Novel territory, may not work for Layer 3 |
| Build own B-Rep | Full control, DD integration, no external deps | More effort than using truck-topology |
| OCC as test oracle | Hard pass/fail for AI agents | C++ toolchain in test infra |
| Curvo for NURBS | Proven algorithms, faster start | API adapter layer needed |
| Half-edge | Simple invariants, O(1) queries | More memory than winged-edge |

---

## Test Suites for AI Agent TDD

### Available Test Infrastructure

| Suite | Source | Layer Coverage | AI Target Metric |
|-------|--------|----------------|------------------|
| **NIST StRD** | `atomic_solver/benches/nist_benchmarks.rs` | Layer 4 | Pass 32/32 problems |
| **NIST Manufacturing** | [NIST MBE/PMI](https://www.nist.gov/ctl/smart-connected-systems-division) | Layers 2-3, 6 | Round-trip fidelity |
| **ABC Dataset** | [Princeton](https://deep-geometry.github.io/abc-dataset/) | Layers 1-3 | Parse 10k models, 99% success |
| **CAx-IF STEP** | [cax-if.org](https://www.cax-if.org/cax/cax_testCases.php) | Layer 6 | Pass recommended practices |
| **OpenCASCADE reference** | `opencascade-rs` | Layers 1-3 | Match OCC output |
| **Property tests** | `proptest` crate | All layers | No failures on 10k runs |

### Test Types by Layer

| Layer | Property Tests | Reference Comparison | Benchmark Suite |
|-------|---------------|---------------------|-----------------|
| L1 Geometry | NURBS evaluation accuracy, curve/surface continuity | curvo baseline, OCC comparison | Curve intersection perf |
| L2 Topology | Euler characteristic, manifold property, half-edge invariants | OCC B-Rep structure | Large model traversal |
| L3 Modeling | Boolean result topology, fillet tangency | OCC boolean output | Boolean perf on complex |
| L4 Constraints | Convergence, DOF accuracy, solution stability | Existing NIST suite | Large assembly solving |
| L5 History | Dependency correctness, regen consistency | N/A (novel) | Incremental update latency |
| L6 I/O | STEP schema conformance, round-trip | CAx-IF test cases | Export throughput |

---

## Milestones (High-Level Phases)

This is a HIGH-LEVEL plan. Each phase will have detailed implementation plans created separately.

### Phase 0: Test Infrastructure Foundation

**Purpose:** Establish test oracle and benchmark infrastructure before implementation.

**Deliverables:**
- OpenCASCADE test harness via `opencascade-rs`
- NIST Manufacturing test model corpus
- ABC Dataset curated subset (10k models)
- CAx-IF STEP test case automation
- Property test framework patterns

**Acceptance Criteria:**
- Can invoke OCC and capture output for comparison
- Test models loaded and queryable
- CI pipeline runs reference tests

**Estimated Effort:** 2-3 weeks

---

### Phase 1: Geometry Layer (Layer 1)

**Purpose:** NURBS curves and surfaces, analytic surfaces, geometric operations.

**Crate:** `atomic_geometry` (new)

**Deliverables:**
- Curvo integration for NURBS evaluation
- Analytic surfaces: Plane, Cylinder, Cone, Sphere, Torus
- Curve operations: intersection, projection, offset
- Surface operations: intersection, normal evaluation
- Adapter layer between curvo and our API

**Test Targets:**
- NURBS evaluation matches curvo baseline
- Surface-surface intersection matches OCC output
- Property tests for continuity (G0, G1, G2)

**Dependencies:** Phase 0

**Estimated Effort:** 2-3 months

---

### Phase 2: Topology Layer (Layer 2)

**Purpose:** Half-edge B-Rep data structure with Euler operators.

**Crate:** `atomic_brep` (new)

**Deliverables:**
- Half-edge data structure: Vertex, HalfEdge, Edge, Wire, Face, Shell, Solid
- Euler operators: MEV, KEV, MEF, KEF, MEKR, KEMR, etc.
- Topology traversal and query API
- Invariant checking and validation
- Integration with Layer 1 geometry

**Test Targets:**
- Euler characteristic correct for all test solids
- Invariant checks pass after every operation
- Traversal matches OCC topology structure

**Dependencies:** Phase 1 (geometry for faces/edges)

**Estimated Effort:** 3-4 months

---

### Phase 3: Modeling Operations (Layer 3)

**Purpose:** Boolean operations, filleting, extrusion, sweep, loft.

**Crate:** `atomic_modeling` (new)

**Deliverables:**
- **3.1 Extrude/Revolve:** Linear extrusion, revolution about axis
- **3.2 Sweep/Loft:** Path sweep, multi-section loft
- **3.3 Boolean Operations:** Union, intersection, difference
- **3.4 Fillet/Chamfer:** Rolling-ball fillet, edge chamfer
- **3.5 Shell/Offset:** Hollow solid, offset surface

**Test Targets:**
- Boolean output topology matches OCC
- Fillet maintains G1 continuity (property test)
- Extrude/revolve produce valid closed solids
- NIST Manufacturing parts model correctly

**Dependencies:** Phase 2 (B-Rep for modification)

**Estimated Effort:** 6-8 months (Boolean is the hard part)

---

### Phase 4: Constraint Solving Extensions (Layer 4)

**Purpose:** Extend atomic_solver for 3D assembly constraints.

**Crate:** `atomic_solver` (extend), `ecad_solver` (extend)

**Deliverables:**
- Arc primitive + constraints (point-on-arc, arc-tangent, concentric)
- Plane primitive + constraints (coplanar, parallel, perpendicular, distance)
- Axis primitive + constraints (coaxial, angle)
- Assembly DOF model (6 DOF per rigid body)
- Assembly mates: coincident, align, insert
- Integration with Layer 2 B-Rep geometry

**Test Targets:**
- Existing NIST 32/32 maintained
- Arc constraint convergence tests
- Assembly DOF calculation matches commercial solvers
- Property tests for assembly solution stability

**Dependencies:** Phase 2 (need planes/axes from geometry)

**Estimated Effort:** 3-4 months

---

### Phase 5: History & Differential Dataflow (Layer 5)

**Purpose:** DD-based feature tree with incremental regeneration.

**Crate:** `atomic_history` (new), `ecad_solver` (DD activation)

**Deliverables:**
- Feature tree data structure (DAG)
- DD integration for dependency tracking
- Incremental regeneration (only recompute changed branches)
- Undo/redo via event replay
- **Research:** DD for Layer 1-3 incremental updates

**Test Targets:**
- Dependency tracking correctness (topological sort)
- Incremental regen produces same result as full regen
- DD throughput benchmarks (latency on parameter change)
- Research: DD boolean update feasibility

**Dependencies:** Phases 1-4 (need all layers to regenerate)

**Estimated Effort:** 4-6 months

---

### Phase 6: I/O Layer (Layer 6)

**Purpose:** STEP export and tessellation for display.

**Crate:** `atomic_step` (new), `atomic_tessellation` (new)

**Deliverables:**
- STEP AP203 cc2a writer (basic geometry export)
- Tessellation for display (triangulated mesh from B-Rep)
- Future: STEP AP242, IGES, native format

**Test Targets:**
- CAx-IF recommended practice tests pass
- STEP round-trip fidelity (write → read via OCC → compare)
- NIST Manufacturing parts export correctly
- Tessellation is watertight (no gaps)

**Dependencies:** Phases 2-3 (need valid B-Rep to export)

**Estimated Effort:** 3-4 months

---

### Phase 7: Integration & Hardening

**Purpose:** Full system integration, performance optimization, production readiness.

**Deliverables:**
- End-to-end workflow: sketch → features → assembly → STEP
- Performance optimization (hot paths, memory)
- Edge case hardening (numerical robustness)
- Documentation and examples

**Test Targets:**
- ABC Dataset 10k models processed without crash
- Large assembly performance (1000+ parts)
- DD incremental update latency targets

**Dependencies:** All phases

**Estimated Effort:** 3-4 months

---

## Milestone Dependencies

```
Phase 0 (Test Infra)
    │
    ▼
Phase 1 (Geometry) ───────────────┐
    │                             │
    ▼                             │
Phase 2 (Topology) ───────────────┤
    │                             │
    ├──────────────┬──────────────┤
    │              │              │
    ▼              ▼              ▼
Phase 3        Phase 4        Phase 6
(Modeling)   (Constraints)    (I/O)
    │              │              │
    └──────────────┼──────────────┘
                   │
                   ▼
             Phase 5 (History/DD)
                   │
                   ▼
             Phase 7 (Integration)
```

**Parallel Opportunities:**
- Phase 3 and Phase 4 can proceed in parallel after Phase 2
- Phase 6 can start during Phase 3 (export basic geometry)
- Within Phase 3, sub-phases 3.1-3.2 (extrude/sweep) can parallel 3.3-3.4 (boolean/fillet)

---

## Estimated Total Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0: Test Infra | 2-3 weeks | 3 weeks |
| Phase 1: Geometry | 2-3 months | 3.5 months |
| Phase 2: Topology | 3-4 months | 7 months |
| Phase 3: Modeling | 6-8 months | 14 months |
| Phase 4: Constraints | 3-4 months | (parallel with 3) |
| Phase 5: History/DD | 4-6 months | 20 months |
| Phase 6: I/O | 3-4 months | (parallel with 3-5) |
| Phase 7: Integration | 3-4 months | 24 months |

**Total: 20-26 months** for SolidWorks-comparable geometric kernel with full B-Rep, NURBS, and differential-dataflow-based incremental updates.

---

## Next Steps

1. **Create detailed Phase 0 plan** - Test infrastructure setup
2. **Create detailed Phase 1 plan** - Curvo integration and geometry layer
3. **Evaluate curvo spike** - Confirm API compatibility before committing
4. **Set up OCC test oracle** - Establish reference comparison baseline

Each phase will have its own detailed implementation plan with specific milestones, code changes, and test targets for AI agent execution.
