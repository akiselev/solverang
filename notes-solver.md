Critical Analysis: solverang Production Readiness

  Executive Summary

  Current State: A well-architected, numerically solid point-based 2D/3D constraint solver with good test
   coverage against standard benchmarks.

  Gap Assessment: To reach production-grade for Mechanical CAD, Electrical CAD, and Design Rule Checking,
   you need approximately 60-70% more work across multiple dimensions.

  ---
  What You Have (Strengths)
  ┌──────────────────────────────┬───────────┬────────────────────────┐
  │             Area             │  Status   │        Quality         │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Core numerics                │ Solid     │ Production-ready       │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Newton-Raphson + LM          │ Complete  │ Well-tuned             │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Sparse/Parallel solving      │ Complete  │ Good architecture      │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ JIT compilation              │ Complete  │ Nice differentiator    │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Point-to-point constraints   │ 15+ types │ Adequate for sketching │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ 2D/3D const generics         │ Clean     │ Excellent design       │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Inequality via slack vars    │ Basic     │ Functional             │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Test coverage (MINPACK/NIST) │ Excellent │ 27/27 + 32/32          │
  ├──────────────────────────────┼───────────┼────────────────────────┤
  │ Symbolic Jacobians           │ Complete  │ Good DX                │
  └──────────────────────────────┴───────────┴────────────────────────┘
  ---
  Critical Gaps by Domain

  1. Geometric Primitives (40% complete)

  What's Missing for MCAD:
  ┌─────────────────────────────┬───────────────────┬────────────┐
  │          Primitive          │    Importance     │ Complexity │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Arc/CircularArc             │ Critical          │ Medium     │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Ellipse/EllipticalArc       │ High              │ Medium     │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Spline (B-spline, Bezier)   │ Critical          │ High       │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ NURBS curves                │ Critical for MCAD │ Very High  │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Plane (3D)                  │ Critical          │ Low        │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Cylinder                    │ Critical          │ Medium     │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Cone                        │ High              │ Medium     │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Sphere (parametric surface) │ High              │ Medium     │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Torus                       │ Medium            │ High       │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ NURBS surfaces              │ Critical for MCAD │ Very High  │
  ├─────────────────────────────┼───────────────────┼────────────┤
  │ Swept/Lofted surfaces       │ High              │ Very High  │
  └─────────────────────────────┴───────────────────┴────────────┘
  What's Missing for ECAD:
  ┌───────────────────────┬────────────┬────────────┐
  │       Primitive       │ Importance │ Complexity │
  ├───────────────────────┼────────────┼────────────┤
  │ Polygon/Region        │ Critical   │ Medium     │
  ├───────────────────────┼────────────┼────────────┤
  │ Polyline (trace path) │ Critical   │ Low        │
  ├───────────────────────┼────────────┼────────────┤
  │ Copper pour region    │ Critical   │ High       │
  ├───────────────────────┼────────────┼────────────┤
  │ Padstack geometry     │ Critical   │ Medium     │
  ├───────────────────────┼────────────┼────────────┤
  │ Via geometry          │ Critical   │ Low        │
  ├───────────────────────┼────────────┼────────────┤
  │ Board outline         │ Critical   │ Low        │
  ├───────────────────────┼────────────┼────────────┤
  │ Keep-out zones        │ Critical   │ Medium     │
  └───────────────────────┴────────────┴────────────┘
  ---
  2. Constraint Types (35% complete for MCAD, 15% for ECAD)

  Missing MCAD Sketch Constraints:

  Critical:
  ├── Arc tangent to line
  ├── Arc tangent to arc
  ├── Smooth (G2 continuity)
  ├── Curvature-continuous (G3)
  ├── Concentric circles/arcs
  ├── Equal radius
  ├── Coradial
  ├── On-spline (point constrained to curve)
  └── Curve-curve intersection

  High Priority:
  ├── Construction geometry (reference-only)
  ├── Driven vs driving dimensions
  ├── Over-constraint detection
  └── Under-constraint DOF visualization

  Missing MCAD Assembly Constraints:

  Critical:
  ├── Mate (planar coincident)
  ├── Align (axial alignment)
  ├── Insert (cylindrical mate)
  ├── Orient (parallel/anti-parallel)
  ├── Angle between planes
  ├── Distance/Offset between planes
  ├── Tangent (surface-to-surface)
  └── Concentric (axis alignment)

  High Priority:
  ├── Gear ratio constraint
  ├── Rack-and-pinion
  ├── Cam/follower
  ├── Path constraint (point along curve)
  ├── Limit constraints (angle/distance ranges)
  └── Rigid group (weld constraint)

  Missing ECAD Constraints:

  Critical for PCB:
  ├── Net connectivity constraint
  ├── Layer assignment constraint
  ├── Trace-to-trace clearance (same net OK)
  ├── Trace-to-pad clearance
  ├── Pad-to-pad clearance
  ├── Component-to-component clearance
  ├── Component-to-board-edge clearance
  ├── Via-to-trace clearance
  ├── Via-to-via clearance
  └── Copper-to-edge clearance

  High Priority:
  ├── Differential pair spacing
  ├── Differential pair length matching
  ├── Matched length groups
  ├── Serpentine/meander constraints
  ├── Via-in-pad rules
  ├── Thermal relief rules
  ├── Annular ring minimum
  ├── Trace width rules (min/max by net class)
  └── Impedance constraints (trace + stackup)

  Schematic:
  ├── Pin-to-pin connection (net based)
  ├── Symbol alignment (grid snap)
  ├── Wire routing orthogonal/diagonal
  └── Bus member ordering

  ---
  3. Solver Architecture Gaps

  Missing for Production Quality:
  ┌───────────────────────────────┬────────────────────────────────────────────┬────────────┐
  │            Feature            │               Why It Matters               │ Complexity │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Multiple solution detection   │ A sketch can have flip states              │ High       │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Branch selection              │ Choose which solution branch user intended │ Medium     │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Constraint conflict diagnosis │ "These 3 constraints conflict"             │ High       │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ DOF analysis                  │ Show remaining freedoms visually           │ Medium     │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Redundancy detection          │ Identify duplicate/implied constraints     │ Medium     │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Constraint prioritization     │ Dimensions > geometric constraints         │ Low        │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Design tolerance              │ ±0.001" is different from numerical ε      │ Medium     │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ Parametric updates            │ Change one dim, solve incrementally        │ Medium     │
  ├───────────────────────────────┼────────────────────────────────────────────┼────────────┤
  │ History-based undo            │ Roll back constraint changes               │ Low        │
  └───────────────────────────────┴────────────────────────────────────────────┴────────────┘
  Missing for DRC (beyond slack variables):
  ┌─────────────────────────┬───────────────────────────────────────┐
  │         Feature         │            Why It Matters             │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Shape-to-shape distance │ Current: point-to-point only          │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Polygon containment     │ Is copper inside board outline?       │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Overlap detection       │ Do these copper regions intersect?    │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Clearance by net class  │ VCC needs 10mil, signals need 6mil    │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Layer-aware checking    │ Same-layer vs different-layer rules   │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Rule hierarchy          │ Net-specific overrides class defaults │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Incremental DRC         │ Re-check only affected region         │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Violation reporting     │ Location, rule violated, severity     │
  └─────────────────────────┴───────────────────────────────────────┘
  ---
  4. Integration & UX Gaps

  Missing for Interactive CAD:

  ├── Drag solving (solve while user drags)
  ├── Constraint inference (auto-detect intent)
  ├── Visual constraint display
  ├── Constraint manager (enable/disable/delete)
  ├── Diagnostic messages (human-readable errors)
  ├── Undo/redo integration
  ├── Persistent constraint storage
  └── Import/export (STEP AP214 constraints)

  ---
  Quantified Gap Assessment
  ┌─────────────────────────┬─────────────────┬────────────────────┬──────┐
  │        Category         │     Current     │ Production Target  │ Gap  │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ Primitives              │ 5 types         │ ~20 types          │ 75%  │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ 2D Constraints          │ 15 types        │ ~35 types          │ 57%  │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ 3D/Assembly Constraints │ 8 types         │ ~25 types          │ 68%  │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ ECAD Constraints        │ 1 type          │ ~20 types          │ 95%  │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ DRC Rules               │ Basic clearance │ Full rule engine   │ 90%  │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ Solver diagnostics      │ None            │ Full diagnosis     │ 100% │
  ├─────────────────────────┼─────────────────┼────────────────────┼──────┤
  │ Interactive features    │ None            │ Full interactivity │ 100% │
  └─────────────────────────┴─────────────────┴────────────────────┴──────┘
  Rough Effort Estimate (lines of code):
  ┌────────────────────────────────────────────┬───────────────┐
  │                 Component                  │ Estimated LOC │
  ├────────────────────────────────────────────┼───────────────┤
  │ Curve primitives (arc, spline, NURBS)      │ 3,000-5,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ Surface primitives (plane, cylinder, etc.) │ 2,000-4,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ Additional 2D constraints                  │ 2,000-3,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ Assembly constraints                       │ 3,000-5,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ ECAD constraints                           │ 4,000-6,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ Full DRC engine                            │ 5,000-10,000  │
  ├────────────────────────────────────────────┼───────────────┤
  │ Solver diagnostics                         │ 2,000-4,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ Interactive features                       │ 3,000-5,000   │
  ├────────────────────────────────────────────┼───────────────┤
  │ Total                                      │ 24,000-42,000 │
  └────────────────────────────────────────────┴───────────────┘
  Current codebase: ~15,000 LOC in solverang

  ---
  Recommended Prioritization

  Phase 1: Foundational Curves (Enables MCAD Sketching)

  1. Arc/circular arc with constraints
  2. Ellipse/elliptical arc
  3. Bezier/B-spline curves
  4. Tangent-to-curve constraints
  5. On-curve (point on spline) constraint

  Phase 2: ECAD Primitives & DRC

  1. Polygon/region primitives
  2. Shape-to-shape distance functions
  3. Net-aware constraint system
  4. Layer-aware rule checking
  5. Full DRC rule engine with reporting

  Phase 3: Assembly Constraints (Enables MCAD Assembly)

  1. Plane primitive with mate constraint
  2. Axis primitive with align/insert
  3. Rigid group/weld constraint
  4. Limit constraints (ranges)
  5. Motion constraints (gear, cam)

  Phase 4: Production Polish

  1. Constraint conflict diagnosis
  2. DOF analysis and visualization
  3. Multiple solution branch selection
  4. Incremental solving for parameter changes
  5. Full diagnostic/error reporting

  ---
  Architecture Recommendations

  1. Separate Geometric Kernel from Solver

  solverang (current)     - Pure numerical solving
       ↓
  geometric_kernel (new)      - Curves, surfaces, distance queries
       ↓
  cad_constraints (new)       - MCAD/ECAD constraint definitions
       ↓
  drc_engine (new)            - Rule-based DRC with spatial index

  2. Add Proper Distance Functions

  Your current ClearanceConstraint computes point-to-point distance. Real DRC needs:

  - Point-to-segment distance
  - Segment-to-segment distance
  - Point-to-polygon distance
  - Polygon-to-polygon distance (Minkowski sum or GJK)
  - Point-to-curve distance (arc, spline)
  - Curve-to-curve distance

  3. Add Constraint Graph for Diagnostics

  struct ConstraintGraph {
      nodes: Vec<Variable>,           // DOFs
      edges: Vec<ConstraintEdge>,     // Constraints

      fn find_conflicts(&self) -> Vec<ConflictSet>;
      fn compute_dof(&self) -> DofAnalysis;
      fn detect_redundancy(&self) -> Vec<RedundantSet>;
  }

  4. Consider Existing Libraries

  For curves/surfaces, you might integrate:
  - kurbo - 2D curves (arcs, beziers, paths)
  - flo_curves - Bezier manipulation
  - nalgebra - Linear algebra (you may already use)
  - parry3d - Collision detection (GJK, EPA algorithms)
  - geo - Computational geometry (polygon operations)

  ---
  Bottom Line

  You have a solid numerical core. The solver algorithms, sparse handling, JIT compilation, and testing
  infrastructure are production-quality.

  What's missing is the geometric and domain layer. The gap isn't in numerical methods—it's in:
  1. Higher-order curve/surface primitives
  2. Domain-specific constraint vocabularies (MCAD assembly, ECAD routing)
  3. Shape-based (not just point-based) distance queries
  4. Diagnostic/interactive features users expect

  For a minimal viable MCAD sketcher: Add arcs, splines, tangency constraints. (~4-6 weeks focused work)

  For a minimal viable ECAD DRC: Add polygon primitives, shape-to-shape distance, net-aware rules. (~6-8
  weeks)

  For production-grade both: The full gap is substantial—expect 6-12 months of focused development
  depending on team size.