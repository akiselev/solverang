Critical Assessment: solverang for Production MCAD

  Current State Summary
  ┌───────────────────────┬──────────────────┬───────────────────────────────┐
  │         Layer         │      Status      │          Assessment           │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ Numerical Core        │ Production-ready │ NR/LM/Sparse/JIT - solid      │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ 2D Sketch Constraints │ 80% complete     │ Missing arcs, splines, conics │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ 3D Primitives         │ 30% complete     │ Points/Lines/Spheres only     │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ Assembly Constraints  │ 0%               │ Not started                   │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ Sketch-on-Face        │ 0%               │ Not started                   │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ Solver Diagnostics    │ 10%              │ DOF count only                │
  ├───────────────────────┼──────────────────┼───────────────────────────────┤
  │ Differential Dataflow │ Not integrated   │ Infrastructure exists, unused │
  └───────────────────────┴──────────────────┴───────────────────────────────┘
  ---
  Critical Finding: Differential Dataflow is NOT Integrated

  The ecad_solver crate has differential-dataflow code but it's not in the active solving path:

  // dataflow.rs line 137-138:
  // For Phase 2: use a simplified approach...
  // This is not fully incremental like label propagation, but works for Phase 2.

  Reality:
  - SolverWorker uses synchronous BFS for component detection (O(V+E) per update)
  - Full recomputation after every structural change
  - The DD label propagation algorithm is implemented but not called
  - Parameter changes don't trigger re-solving at all

  ---
  Gap Analysis vs SolidWorks

  1. Missing Geometric Primitives
  ┌────────────────────────┬────────────┬───────────────┬──────────┐
  │       Primitive        │ SolidWorks │ solverang │ Priority │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Arc/Circular Arc       │ ✓          │ ✗             │ CRITICAL │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Ellipse/Elliptical Arc │ ✓          │ ✗             │ HIGH     │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Spline (B-spline)      │ ✓          │ ✗             │ CRITICAL │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ NURBS Curves           │ ✓          │ ✗             │ CRITICAL │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Conic Sections         │ ✓          │ ✗             │ MEDIUM   │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Plane (3D)             │ ✓          │ ✗             │ CRITICAL │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Axis (3D)              │ ✓          │ ✗             │ CRITICAL │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Cylinder               │ ✓          │ ✗             │ HIGH     │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Cone                   │ ✓          │ ✗             │ HIGH     │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ Torus                  │ ✓          │ ✗             │ MEDIUM   │
  ├────────────────────────┼────────────┼───────────────┼──────────┤
  │ NURBS Surface          │ ✓          │ ✗             │ HIGH     │
  └────────────────────────┴────────────┴───────────────┴──────────┘
  2. Missing Constraint Types

  Sketch Constraints (2D):
  ┌──────────────────┬────────┬───────────┬───────────────────────────────────┐
  │    Constraint    │ Status │ Equations │               Notes               │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Point-on-Arc     │ ✗      │ 1         │ Requires arc primitive            │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Point-on-Spline  │ ✗      │ 1         │ Requires spline primitive         │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Arc-Line Tangent │ ✗      │ 1         │ Requires arc primitive            │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Arc-Arc Tangent  │ ✗      │ 1         │ Requires arc primitive            │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Concentric       │ ✗      │ D-1       │ Same center for circles/arcs      │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Coradial         │ ✗      │ D         │ Same center + same radius         │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Smooth (G1)      │ ✗      │ D         │ Tangent continuity at join        │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Curvature (G2)   │ ✗      │ D+1       │ Curvature continuity              │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Equal Radius     │ ✗      │ 1         │ For circles/arcs                  │
  ├──────────────────┼────────┼───────────┼───────────────────────────────────┤
  │ Pierce (3D)      │ ✗      │ 2         │ Point at curve-plane intersection │
  └──────────────────┴────────┴───────────┴───────────────────────────────────┘
  Assembly Constraints (3D):
  ┌──────────────────────┬────────┬─────────────┬───────────────────────────────┐
  │      Constraint      │ Status │ DOF Removed │             Notes             │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Mate (Flush)         │ ✗      │ 1-3         │ Face-to-face, requires planes │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Align (Coaxial)      │ ✗      │ 2-4         │ Axis alignment, requires axes │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Insert               │ ✗      │ 4-5         │ Mate + align combined         │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Tangent Surface      │ ✗      │ 1           │ Requires surface primitives   │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Distance (Face-Face) │ ✗      │ 1           │ Requires planes               │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Angle (Plane-Plane)  │ ✗      │ 1           │ Requires planes               │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Gear Ratio           │ ✗      │ Links DOF   │ Coupled rotation              │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Cam/Follower         │ ✗      │ Links DOF   │ Trajectory following          │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Rack-Pinion          │ ✗      │ Links DOF   │ Rotational↔linear             │
  ├──────────────────────┼────────┼─────────────┼───────────────────────────────┤
  │ Path Constraint      │ ✗      │ Links DOF   │ Point along curve             │
  └──────────────────────┴────────┴─────────────┴───────────────────────────────┘
  3. Missing Solver Algorithms
  ┌──────────────────────────────────┬──────────────────────────────────────┬────────┐
  │            Algorithm             │               Purpose                │ Status │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Multiple Solution Detection      │ Find all valid configurations        │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Branch Selection                 │ Choose solution matching user intent │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Over-Constraint Detection        │ Identify conflicting constraints     │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Under-Constraint Analysis        │ Show remaining DOF per entity        │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Redundancy Detection             │ Find implied/duplicate constraints   │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Drag Solving (SQP)               │ Solve while user drags geometry      │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Dulmage-Mendelsohn Decomposition │ Structural analysis                  │ ✗      │
  ├──────────────────────────────────┼──────────────────────────────────────┼────────┤
  │ Incremental Jacobian Updates     │ Avoid full recomputation             │ ✗      │
  └──────────────────────────────────┴──────────────────────────────────────┴────────┘
  4. Missing System Features
  ┌──────────────────────────────┬────────────┬─────────────────┐
  │           Feature            │ SolidWorks │  solverang  │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Sketch-on-Face projection    │ ✓          │ ✗               │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Local coordinate systems     │ ✓          │ ✗ (global only) │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Part reference frames        │ ✓          │ ✗               │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Driving vs driven dimensions │ ✓          │ ✗               │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Construction geometry        │ ✓          │ ✗               │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Constraint status colors     │ ✓          │ ✗               │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ Conflict diagnosis messages  │ ✓          │ ✗               │
  ├──────────────────────────────┼────────────┼─────────────────┤
  │ DOF visualization            │ ✓          │ ✗               │
  └──────────────────────────────┴────────────┴─────────────────┘
  ---
  Architecture Gaps

  Current Architecture

  User → ecad_solver (sync BFS) → adapter → solverang (full solve)
                  ↓
          [DD infrastructure exists but unused]

  Required Architecture

  User → ecad_solver (DD incremental) → component solver (per-component)
                ↓                              ↓
       Structural analysis          Numerical solving (incremental)
       (DD label propagation)       (warm-start from previous solution)
                ↓                              ↓
       Diagnostics engine ←────────────────────┘
       (over/under-constrained, conflicts)

  ---
  Effort Estimation

  Phase 1: Core Curve Primitives (~4-6 weeks)

  - Arc with center/radius/angles parametrization
  - Arc constraints (tangent, concentric, point-on-arc)
  - Bezier/B-spline curves (cubic, quadratic)
  - Spline constraints (point-on-spline, tangent)
  - Enables: Full 2D sketch capability

  Phase 2: 3D Primitives & Planes (~4-6 weeks)

  - Plane primitive (normal + distance)
  - Axis primitive (point + direction)
  - Plane constraints (coplanar, parallel, perpendicular, distance, angle)
  - Point-on-plane, line-on-plane constraints
  - Sketch-on-face coordinate transformation API
  - Enables: 3D sketching on faces

  Phase 3: Assembly Constraints (~6-8 weeks)

  - Part reference frame abstraction (position + orientation)
  - Rigid body DOF model (6 DOF per floating component)
  - Mate constraint (face-to-face with offset)
  - Align constraint (axis alignment)
  - Insert constraint (combined mate + align)
  - Grounded component handling
  - Enables: Basic assembly positioning

  Phase 4: Solver Diagnostics (~4-6 weeks)

  - Over-constraint detection via Jacobian rank analysis
  - Under-constraint visualization (DOF per entity)
  - Conflict identification algorithm
  - Redundancy detection
  - Human-readable diagnostic messages
  - Enables: User-friendly constraint status

  Phase 5: Incremental Solving (~6-8 weeks)

  - Activate differential-dataflow in hot path
  - Incremental component detection
  - Warm-start solving from previous solution
  - Parameter change propagation
  - Drag solving mode (SQP integration)
  - Enables: Interactive performance

  Phase 6: Advanced Geometry (~8-12 weeks)

  - NURBS curves with full constraint support
  - Cylindrical/conical surface constraints
  - Surface-surface tangency
  - Advanced assembly (gear, cam, path)
  - Enables: Production MCAD parity

  ---
  Quick Wins vs Deep Work

  Quick Wins (< 2 weeks each)

  1. Arc primitive + basic constraints - Unlocks most real sketches
  2. Plane primitive + point-on-plane - Foundation for 3D
  3. Activate DD component detection - Already written, just not called
  4. DOF-per-entity reporting - Extend existing DOF calculation

  Deep Work (> 4 weeks each)

  1. Full spline support with G1/G2 continuity
  2. Complete assembly solver with DOF tracking
  3. Incremental numerical solving with warm-start
  4. Comprehensive diagnostic engine

  ---
  Recommended Priority Order

  1. Arc primitive + constraints          [CRITICAL - most sketches need arcs]
  2. Plane primitive + constraints        [CRITICAL - enables 3D workflows]
  3. Activate DD for component detection  [HIGH - already implemented]
  4. Over/under-constraint detection      [HIGH - user experience]
  5. Sketch-on-face transformation        [HIGH - core MCAD workflow]
  6. Basic assembly mates (face, axis)    [HIGH - assembly requirement]
  7. B-spline curves                      [MEDIUM - advanced sketching]
  8. Full drag solving                    [MEDIUM - interactivity]
  9. Advanced assembly (gear, cam)        [LOW - specialized use]
  10. NURBS surfaces                      [LOW - advanced modeling]

  ---
  Bottom Line

  You have ~35% of what's needed for SolidWorks-level MCAD.

  The numerical foundation is excellent. The gaps are:
  1. Geometric primitives - No arcs, splines, planes, surfaces
  2. Assembly framework - No rigid body model, no mates
  3. Incremental solving - DD infrastructure exists but isn't wired up
  4. Diagnostics - No help for users when constraints conflict

  For a minimal viable 3D MCAD sketcher: 3-4 months focused work
  For assembly solving: Additional 2-3 months
  For SolidWorks parity: 12-18 months total