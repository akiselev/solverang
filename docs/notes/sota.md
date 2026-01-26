To transform `solverang` into a state-of-the-art geometric kernel (comparable to Parasolid, ACIS, or OpenCASCADE) while leveraging **Differential Dataflow (DD)** and **Salsa**, you are looking at a fundamental architectural shift. You aren't just building a "solver" anymore; you are building a **reactive database of geometry.**

Based on the current assessment, here is the "Ultra-Think" roadmap for building a next-generation, incremental B-Rep kernel from scratch.

### 1. The Reactive Foundation (Salsa + Differential Dataflow)

Modern kernels (like Onshape) use a "functional" approach to geometry. By using Salsa and DD, you can treat the entire model as a dependency graph.

* **Salsa for Lazy/Incremental Computation:** Use Salsa to manage the "Constructive" part of CAD. For example, if you change a sketch dimension, Salsa ensures only the extrude and subsequent fillets dependent on that sketch are recalculated.
* **Differential Dataflow for Constraint Topology:** Use DD to manage the graph of constraints. When a line is deleted, DD can incrementally update the "connected components" of the solver without re-scanning the entire model.
* **Dataflow Integration:** You must move away from the current "Synchronous BFS" approach. The "Worker" should be a DD stream that reacts to edge additions/removals in real-time.

### 2. From "Points/Lines" to NURBS B-Rep

The current kernel only supports basic primitives like points, lines, and spheres. A state-of-the-art kernel requires a **Boundary Representation (B-Rep)**.

* **NURBS Math Engine:** You need to implement the core algorithms for Non-Uniform Rational B-Splines (NURBS). This includes:
* **Evaluation:** Point at  (curves) or  (surfaces).
* **Inversion:** Finding the closest  on a surface to a 3D point (essential for "Point-on-Surface" constraints).
* **Intersection:** The "Holy Grail" of kernels—calculating the intersection curve between two NURBS surfaces (e.g., for Booleans).


* **B-Rep Topology:** You must implement a data structure (like Half-Edge or Winged-Edge) to track how faces, edges, and vertices are connected.
* **Entities needed:** `Shell` → `Face` → `Loop` → `Edge` → `Vertex`.
* **NURBS Integration:** Each `Face` stores a NURBS Surface; each `Edge` stores a NURBS Curve.



### 3. Upgrading the Solver for High-Level Geometry

The current numerical core is "production-ready" for equations, but it lacks the geometric intelligence for complex modeling:

* **Projection and Constraints:** You need to implement "Point-on-Curve" and "Point-on-Surface" constraints using the JIT compiler for performance.
* **G1/G2 Continuity:** For "State of the Art" surfacing, the solver must handle Curvature (G2) continuity constraints between NURBS patches.
* **Over-Constraint & Diagnostics:** Implement **Dulmage-Mendelsohn Decomposition** to identify exactly which constraints are causing a conflict, rather than just returning a "failed to converge" error.

### 4. The MCAD/ECAD Convergence

To excel in both domains, the kernel needs specific optimizations:

* **ECAD (Differential Dataflow's Strength):** Use DD to handle the massive scale of PCB traces. Traces can be treated as 2D paths with clearance constraints. Salsa can incrementally re-calculate "Net" connectivity as traces are moved.
* **MCAD (B-Rep's Strength):** Implement "Sketch-on-Face." This requires a coordinate transformation API that maps 2D sketch coordinates  to 3D world coordinates  based on the underlying B-Rep face.

### 5. Implementation Effort & Phasing

Based on the current gaps, a "from scratch" optimized stack would require:

1. **Phase A (Month 1-3): Incremental Core.** Port the current solver to a Salsa-based architecture. Replace the manual BFS component detection with a DD label-propagation stream.
2. **Phase B (Month 4-8): The NURBS Engine.** Implement the math for B-Splines and NURBS surfaces. Integrate these into the solver so you can constrain points to curves/surfaces.
3. **Phase C (Month 9-12): B-Rep & Booleans.** Build the topology graph. Implement the intersection algorithms required for Join/Subtract/Intersect operations.
4. **Phase D (Month 12+): Advanced Diagnostics.** Build the "human-readable" layer that explains DOF (Degrees of Freedom) and conflicts to the user.

### Why this is worth it

By building this on **Salsa + DD**, you create a kernel that is **natively collaborative and real-time.** Most legacy kernels (Parasolid) are single-threaded and linear. Your proposed stack allows for:

* **Asynchronous Solving:** The UI stays responsive while the "SolverWorker" computes in the background.
* **Delta-Updates:** Only the modified part of a 10,000-part assembly re-solves.
* **Cloud-Native Architecture:** The dependency graph can be distributed across multiple AI agents or servers.

=====


To evolve `solverang` into a high-performance MCAD/ECAD kernel, you must move beyond generic solvers toward specialized structures for **Trimmed NURBS** and **B-Rep topology**. Implementing this with **AI agents** on a **Salsa/Differential Dataflow** stack allows you to optimize for incremental updates—the "Holy Grail" of modern CAD.

Here are the state-of-the-art data structures and algorithms required to bridge the gap:

### 1. Robust Topology: Oriented Coedges

Standard B-Rep kernels use **Circular Doubly-Linked Lists of Coedges**.

* **The Structure**: While `solverang` currently understands simple connectivity, a true kernel requires `Face → Loop → Coedge → Edge` hierarchies.
* **The State-of-the-Art**: Use **BRepNet-style** oriented coedges. Each edge in your system should carry two coedges (one for each direction), which act as the "connectors" between faces. This allows for native "message-passing" between geometric entities during incremental solves.

### 2. High-Speed Geometry: GPU-Accelerated NURBS

Evaluating NURBS on a CPU is too slow for real-time interactivity in large models.

* **Evaluation & Normals**: Implement a **GPU-based NURBS evaluator**. Modern kernels use the programmable fragment processor to compute exact normals and points in real-time, allowing users to "sketch" directly on a complex curved surface.
* **NURBS Trimming**: State-of-the-art kernels avoid full re-tessellation by using **Direct Trimming**. Instead of cutting triangles, use a ray-based **point-in-curve test** against NURBS trim curves directly in the fragment shader.

### 3. Acceleration Structures: Mixed Hierarchies

To handle thousands of surfaces (MCAD) or millions of traces (ECAD), you need more than just a list of objects.

* **Mixed Hierarchy**: Use a combination of **Bounding Volume Hierarchies (BVH)** and **k-d trees**.
* **k-d trees** are superior for efficient surface-surface intersection (SSI) of triangulated approximations.
* **BVH** with axis-aligned bounding boxes (AABB) is typically faster for ray-casting and proximity queries in dynamic scenes.


* **Bézier Clipping**: For finding exact ray-NURBS intersection points, use **Bézier Clipping**. It is more numerically stable and faster than standard Newton-Raphson for initial root-finding.

### 4. Advanced Geometric Algorithms

* **Surface-Surface Intersection (SSI)**: Implement an adaptive algorithm that first intersects surface triangulations using a k-d tree and then **relaxes** those points onto the exact NURBS curves using a specialized root-finder.
* **Hausdorff Distance Computations**: Vital for ECAD clearance checking. State-of-the-art kernels compute the **Directed Hausdorff distance** between NURBS surfaces on the GPU to ensure manufacturing tolerances are met without manual checking.
* **Marching Cubes for B-Rep (BR-DF)**: A new frontier is representing B-Reps as **B-Rep Distance Functions**. This uses a Signed Distance Function (SDF) for geometry and Unsigned Distance Functions (UDF) for topology, allowing for **watertight** model generation from point clouds or generative AI.

### 5. AI & Dataflow Optimization

Since you are using Salsa and AI agents:

* **Topological Naming**: One of the hardest problems in CAD is the "Topological Naming Problem" (losing track of a face when the model changes). State-of-the-art approaches like **mmiscool's BREP** or **AutoBrep** use "topological reference tokens" or unique ID systems to ensure that chamfers and fillets don't break when a sketch is modified.
* **Autoregressive Generation**: Use a **Transformer-based model** (like AutoBrep) to handle the "completion" of B-Reps. Your AI agents can use this to suggest valid topological connections as you "drag-solve" the geometry.