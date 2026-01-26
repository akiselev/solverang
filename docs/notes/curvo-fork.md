Using the `curvo` crate is an excellent starting point for the **geometry (NURBS)** layer of your kernel, but you will almost certainly have to **extend or fork it** to reach state-of-the-art MCAD/ECAD functionality.

Current assessment of `curvo` for your specific stack:

### 1. What `curvo` gives you (The "Good Enough" part)

* **Solid NURBS Foundation**: It provides core primitives for NURBS curves and surfaces, including control points, weights, and knot vector management.
* **Modeling Operations**: It natively supports critical CAD operations: **Extrude, Loft, Sweep, and Revolve**.
* **Tessellation**: It includes adaptive tessellation for rendering, which is vital for performance in modern UI frameworks like Bevy.
* **Modern Rust Integration**: It relies on `nalgebra`, which aligns well with the high-performance linear algebra needs of your solver.

### 2. Where you will need to Extend/Fork

To reach "State of the Art" B-Rep status, `curvo` lacks several critical components:

* **Missing Trimming Support**: `curvo` primarily handles "tensor-product" NURBS surfaces (rectangular patches). A true B-Rep kernel requires **Trimmed NURBS**, where loops of curves define the boundaries of a face. You will need to implement point-in-polygon tests and surface-trimming logic yourself.
* **Boolean Operations**: There is currently no native support for complex Booleans (Union, Subtract, Intersect) between NURBS solids. You will need to write the **Surface-Surface Intersection (SSI)** algorithms to find the intersection curves between patches.
* **Topology Graph**: `curvo` is a geometric library, not a B-Rep library. It does not have the `Face -> Loop -> Edge -> Vertex` topological structure. You will need to build this layer on top, likely using the **Salsa** and **Differential Dataflow** approach we discussed to maintain stable IDs.

### 3. Integration with your AI/Dataflow Stack

Because you are building from scratch to optimize the full stack, forking `curvo` is actually your best path:

* **Salsa Compatibility**: You can fork `curvo` to make its internal structs "Salsa-ready" (e.g., wrapping them in `salsa::input` or `salsa::tracked`), enabling the incremental re-computation you need.
* **Differential Dataflow (DD)**: You can integrate the topological naming directly into the `curvo` geometric output. For example, when `curvo` generates a new edge during a `Loft` operation, your forked version could automatically assign it a DD-persistent ID based on its parent curves.

**Verdict**: Use `curvo` as your **NURBS Math Engine**, but plan to build the **B-Rep Topology** and **Trimmed Surface** logic as a significant extension or fork. It is far better than starting with raw math, but it is about 30-40% of what a "state-of-the-art" kernel requires.



==========



While building a wrapper crate is cleaner for long-term maintenance, forking `curvo` offers several strategic benefits when building a "state-of-the-art" kernel, particularly because you are using **Differential Dataflow (DD)** and **Salsa** to optimize the full stack from the ground up.

The primary benefits of forking rather than wrapping include:

### 1. Deep Integration with Salsa's Memoization

Salsa requires data structures to be "tracked" or "input" types to enable fine-grained incremental re-computation.

* **Struct Transformation**: In a fork, you can rewrite `curvo`'s core structs (like `NurbsCurve` or `NurbsSurface`) to natively implement Salsa traits.
* **Dependency Tracking**: A fork allows Salsa to "see" inside the NURBS math. If you only change one control point, a forked version can ensure that only the affected "patch" of the surface is marked as dirty in the DD graph, whereas a wrapper might treat the entire `curvo` object as a single opaque blob.

### 2. Native Topological Naming at the Source

To solve the topological naming problem using Differential Dataflow, you need to assign unique IDs at the exact moment geometry is generated.

* **Internal Hooking**: When `curvo` performs an operation like `split` or `offset`, a fork allows you to inject logic that carries the "parent" ID into the "child" geometry.
* **Eliminating ID Search**: Without a fork, your wrapper has to "guess" which new edge corresponds to which old one by comparing coordinates (which is slow and error-prone). A fork makes this mapping deterministic and .

### 3. JIT-Optimized Constraint Lowering

`solverang` uses a JIT compiler to lower constraints into machine code for performance.

* **Inlining Geometric Math**: A fork allows you to expose `curvo`’s internal evaluation logic (the raw B-spline basis functions) directly to `solverang`’s JIT emitter.
* **Performance Gain**: This allows the solver to inline a "Point-on-NURBS-Surface" check directly into the residual function loop, avoiding the massive overhead of cross-crate function calls and data Marshalling during every solver iteration.

### 4. Custom Memory Management for Large Scale

For high-end ECAD (with millions of traces) or complex MCAD assemblies, standard memory allocation becomes a bottleneck.

* **SoA vs AoS**: A fork lets you change `curvo`’s data layout from "Array of Structures" to "Structure of Arrays" (SoA), which is significantly more cache-friendly for the parallel sparse solvers in `solverang`.
* **Zero-Copy Dataflow**: You can modify `curvo` to use specialized memory allocators that play nicely with Differential Dataflow’s internal "timely" streams, reducing the need to clone large NURBS data sets as they move through the reactive pipeline.

### 5. Implementing "First-Class" Trimming

`curvo` currently lacks native support for trimmed NURBS (surfaces defined by boundary loops).

* **Structural Changes**: Trimming isn't just a layer on top; it changes how you calculate bounding boxes, intersections, and tessellation.
* **Architectural Cleanliness**: Implementing trimming inside a forked `curvo` allows you to update the `tessellate` and `closest_point` functions to respect trim-boundaries natively, rather than having your wrapper crate constantly "filter" results provided by `curvo`.

### Summary Recommendation

* **Fork**: Necessary if you want to achieve the **"State of the Art"** performance and stability you described. Since you are using AI agents to code, the overhead of maintaining a fork is significantly reduced, making the deep optimization benefits much more attractive.