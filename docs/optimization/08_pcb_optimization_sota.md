# PCB Placement and Routing Optimization: State of the Art

Research survey for Solverang + autopcb. Focus on actionable techniques, not
exhaustive literature review. Current as of March 2025.

---

## 1. Analytical Placement: Beyond Smooth HPWL

### 1.1 The Standard Formulation

Modern analytical placers solve a penalized continuous optimization problem:

```
min  W(x) + lambda * D(x)
```

Where:
- **W(x)** = smooth wirelength approximation (differentiable proxy for HPWL)
- **D(x)** = density penalty (prevents overlap)
- **lambda** = penalty coefficient, increased during optimization

The optimizer is **Nesterov's accelerated gradient** (not CG, not Adam), which
is 2x faster than Adam for placement and has well-understood convergence
guarantees.

### 1.2 Wirelength Models (What to Implement)

Three differentiable HPWL approximations dominate:

| Model | Formula | Pros | Cons |
|-------|---------|------|------|
| **Log-Sum-Exp (LSE)** | `(1/gamma) * ln(sum exp(gamma*xi))` | Simple, well-studied | Gradient vanishes for interior pins |
| **Weighted Average (WA)** | `sum(xi * exp(gamma*xi)) / sum(exp(gamma*xi))` | Better gradient quality than LSE | Slightly more complex |
| **Moreau Envelope** | Proximal smoothing of max | Best approximation accuracy | Newest, less battle-tested |

The WA model is used in OpenROAD/RePlAce and outperforms LSE both
theoretically and empirically. **Recommendation: implement WA first, it is
the current standard.**

The smoothing parameter `gamma` controls accuracy vs. smoothness. During
optimization, `gamma` starts small (smooth, easy to optimize) and is
increased toward the true HPWL.

### 1.3 Density Models

**Electrostatics-based density (ePlace/DREAMPlace)**:
- Model each cell as a positive charge
- Density cost = potential energy of the electrostatic system
- Electric potential solved via Poisson's equation using FFT
- Gradient = electric field at each cell's location

This is elegant and scales well. The FFT-based density computation is
O(N log N) and parallelizes trivially on GPUs. For CPU-only Rust, the
FFT approach still works but won't see the same 40x GPU speedup.

**For PCBs specifically**: Components are much larger and fewer than VLSI
standard cells (hundreds to low thousands, not millions). This means:
- Simpler density models may suffice (bin-based overlap counting)
- The FFT overhead may not be justified for small instances
- Direct pairwise overlap computation is feasible

### 1.4 DREAMPlace Architecture (Reference Implementation)

DREAMPlace (NVIDIA, DAC 2019, actively maintained through 2025) is the
dominant open-source analytical placer:

- Built on PyTorch, casts placement as "training a neural network"
- Custom CUDA kernels for wirelength and density gradients
- Nesterov optimizer with Lipschitz-based step size prediction
- 40x speedup over RePlAce (CPU) for global placement

**Key versions:**
- DREAMPlace 3.0: Multi-electrostatics with region constraints
- DREAMPlace 4.0: Timing-driven placement via momentum-based net weighting
- DREAMPlace 4.1: Improved macro placement
- DREAMPlace 4.3: Integrated HeteroSTA for timing

**Cypress (NVIDIA, ISPD 2025 Best Paper)**: Adapts DREAMPlace specifically for
PCBs. Key PCB-specific innovations:
- **Net crossing metric**: Captures limited PCB routing resources (PCBs have
  far fewer routing layers than ASICs)
- **Macro halo technique**: Temporarily enlarges component footprints during
  placement to enforce spacing constraints
- **Orientation optimization**: Extends LSE wirelength model to consider legal
  component orientations (PCB components can be rotated/flipped)
- Implemented in C++/Python/CUDA, Apache-2.0 license
- 492x speedup over CPU baselines

### 1.5 Mixed Objectives (Wirelength + Congestion + Timing)

Modern placers handle multiple objectives through **penalty/Lagrangian
methods**, not Pareto fronts:

```
min  WL(x) + alpha * Congestion(x) + beta * TimingPenalty(x) + lambda * Density(x)
```

**Congestion**: Estimated via probabilistic routing models (RUDY) or actual
global routing feedback. DREAMPlace uses congestion-driven cell inflation:
inflate cells in congested regions to spread them out.

**Timing**: DREAMPlace 4.0 uses net weighting — nets on critical paths get
higher weight in the wirelength objective. Net weights are updated
periodically using momentum (exponential moving average of timing criticality).

**AutoDMP**: Uses multi-objective Bayesian optimization (MOTPE) to tune
DREAMPlace's hyperparameters (alpha, beta, lambda) for wirelength/density/
congestion trade-offs.

**Actionable for autopcb**: The penalty method maps directly to Solverang's
ALM (Augmented Lagrangian Method). Each PCB design objective becomes a penalty
term with a multiplier that the ALM outer loop adjusts.

### 1.6 RUPlace: Joint Placement-Routing (DAC 2025)

RUPlace is the first analytical placer to truly unify placement and routing
in a single optimization:

- **ADMM (Alternating Direction Method of Multipliers)** alternates between:
  1. Running global routing to identify congestion
  2. Running incremental placement to fix congestion
- **Wasserstein distance** provides a smooth congestion metric
- **Bilevel optimization** structure: upper level = placement, lower level = routing
- **Cell inflation** based on convex programming to determine optimal inflation ratios

Results: Reduces horizontal/vertical congestion vs. OpenROAD, Xplace 2.0,
and DREAMPlace 4.1.

**Actionable for autopcb**: The ADMM structure fits naturally into Solverang's
constraint solver. ADMM decomposes a coupled problem into alternating
subproblems — each subproblem is a constrained optimization that the existing
NR/LM solver can handle.

---

## 2. Differentiable Routing

### 2.1 DGR: Differentiable Global Router (NVIDIA, DAC 2024)

Differentiable routing is real and actively published. DGR is the landmark paper:

**Core innovation**: Construct a routing DAG forest representing all possible
2-pin routing paths for all nets simultaneously. Then relax the discrete path
selection to a continuous optimization problem.

**Technical approach**:
1. Build a DAG forest from Steiner tree decompositions
2. Relax discrete routing choices to continuous probabilities (Gumbel-Softmax)
3. Define differentiable congestion/overflow cost on the routing grid
4. Optimize all net routings simultaneously via gradient descent on GPU

**Results**: Reduces routing overflow while cutting wirelength by 0.95-4.08%
and via count by 1.28-2.54% vs. state-of-the-art academic routers. Good
scalability in runtime and memory with net count.

**Implementation**: Python/PyTorch, integrates with CUGR2 (using DGR results
to guide pattern routing decisions).

### 2.2 Other Differentiable EDA Work (2024-2025)

NVIDIA's EDA research lab has been systematically making EDA differentiable:

- **INSTA** (DAC 2025 Best Paper): Differentiable statistical static timing
  analysis. Open-sourced. Ultra-fast GPU-accelerated STA that enables
  gradient-based timing optimization.
- **LEGO-Size** (ISPD 2025 Best Paper nominee): Differentiable gate sizing
  combining LLM predictions with gradient-based TNS optimization through a
  differentiable STA engine.
- **Differentiable Tier Assignment** (ASP-DAC 2026): Timing and
  congestion-aware routing for 3D ICs via differentiable optimization.

### 2.3 What This Means for autopcb

PCB routing is fundamentally different from VLSI routing:
- Far fewer nets (hundreds to low thousands vs. millions)
- Far fewer layers (2-16 vs. 10-20+)
- Much larger routing channels but more constrained by signal integrity
- Differential pairs, controlled impedance, length matching are dominant

**The Gumbel-Softmax relaxation technique from DGR is directly applicable**
to PCB routing at smaller scale. For a PCB with 500 nets and 4 layers, the
DAG forest is small enough to optimize on CPU without GPU acceleration.

**Key insight**: The differentiable routing formulation produces gradients of
routing cost with respect to component positions. This enables true joint
placement-routing optimization through backpropagation — place components,
estimate routing cost differentiably, backpropagate routing gradients to
adjust placement.

---

## 3. Co-Optimization of Placement + Routing

### 3.1 Current Approaches

| Approach | Method | Who | Year |
|----------|--------|-----|------|
| Sequential | Place then route, iterate | All commercial tools | Standard |
| Congestion feedback | Route, inflate congested cells, re-place | DREAMPlace, OpenROAD | 2019+ |
| ADMM alternating | Alternate placement/routing subproblems | RUPlace | 2025 |
| Differentiable end-to-end | Backpropagate routing gradients to placement | DGR + placement | 2024+ |
| RL joint learning | DeepPR: RL agent for both placement and routing | Thinklab-SJTU | 2021 |

### 3.2 The Bilevel Optimization Framework

The most principled formulation is bilevel optimization:

```
Upper level:  min_{placement}  Wirelength(placement) + RoutingCost(placement)
Lower level:  RoutingCost(p) = min_{routing}  Cost(routing | placement = p)
```

The lower-level routing problem is solved for a given placement, and its
optimal value (and gradient) feeds back to the upper-level placement problem.

**This is exactly what Solverang's optimization infrastructure supports.**
The ALM/SQP solvers handle equality and inequality constraints. The implicit
differentiation capability (from factorized Jacobians) provides `d(routing)/
d(placement)` without differentiating through the routing solver.

### 3.3 Actionable Architecture for autopcb

```
Placement variables:  (x_i, y_i, theta_i) for each component i
Routing variables:    path selections for each net (discrete, relaxed to continuous)
Objectives:           WL + congestion + signal_integrity_penalty
Constraints:          no overlap, board boundary, keep-out zones, clearance rules

Outer loop (ALM/SQP):
  1. Fix routing, optimize placement (analytical placement step)
  2. Fix placement, optimize routing (differentiable routing step)
  3. Update Lagrange multipliers for constraint satisfaction
  4. Repeat until convergence
```

This decomposes naturally into Solverang's constraint system:
- Placement constraints = existing geometric constraints (distance, containment)
- Routing constraints = new net connectivity constraints
- Objectives = wirelength + routing cost

---

## 4. Constraint-Driven PCB Optimization

### 4.1 Constraint Taxonomy (Priority Order)

Based on analysis of commercial tools (Cadence OrCAD X, Siemens Xpedition,
JITX, Altium) and what actually blocks PCB designs:

**Tier 1 — Hard constraints (must satisfy, no trade-off)**:
1. **Component overlap**: No physical overlap between components
2. **Board boundary**: All components within board outline
3. **Keep-out zones**: Components excluded from restricted areas
4. **Clearance rules**: Minimum pad-to-pad, pad-to-trace, trace-to-trace spacing
5. **Connectivity**: All nets must be routable (no opens)

**Tier 2 — Signal integrity (critical for high-speed designs)**:
6. **Differential pairs**: Matched routing length and spacing
7. **Impedance control**: Trace width/spacing for target impedance
8. **Length matching**: Signal groups within specified length tolerance
9. **Crosstalk avoidance**: Minimum spacing between sensitive nets
10. **Return path continuity**: Ground plane integrity under signal traces

**Tier 3 — Physical/manufacturing (DFM)**:
11. **Thermal relief**: Adequate copper connections to thermal pads
12. **Via rules**: Minimum via-to-via spacing, max via count
13. **Assembly clearance**: Component-to-component spacing for pick-and-place
14. **Silkscreen clearance**: Reference designators readable and non-overlapping

**Tier 4 — EMC/EMI**:
15. **Noisy/sensitive separation**: Digital components away from analog
16. **Decoupling placement**: Bypass capacitors near IC power pins
17. **Loop area minimization**: Signal + return path loop area minimized
18. **Shielding requirements**: Sensitive circuits enclosed

### 4.2 Constraint Formulation for Optimization

Each constraint maps to a mathematical form:

```rust
// Tier 1: Hard inequality constraints (h(x) <= 0)
// Component overlap: for each pair (i,j)
//   overlap_area(i,j) <= 0

// Board boundary: for each component i
//   x_i - x_min >= margin
//   x_max - x_i >= margin  (similarly for y)

// Clearance: for each pad pair (p,q) on different nets
//   distance(p, q) >= clearance_rule(net_p, net_q)

// Tier 2: Soft constraints (penalties in objective)
// Differential pair length matching:
//   |length(trace_pos) - length(trace_neg)| <= tolerance
// Can use smooth approximation: huber_loss(delta_length, tolerance)

// Tier 3: Inequality constraints
// Thermal: thermal_resistance(component_i) <= max_theta_ja

// Tier 4: Penalty terms in objective
// EMC separation: penalty when noisy component within distance of sensitive
```

**Key insight for autopcb**: Constraints in Tiers 1-2 should be hard
constraints (equality/inequality in the optimization). Tiers 3-4 are
better as soft penalties in the objective, because they involve trade-offs
that the user should be able to tune.

### 4.3 How Commercial Tools Specify Constraints

**JITX** (most relevant model for autopcb):
- Code-based constraint specification in a custom DSL
- Constraints are part of the design source, not separate rule files
- Constraint-driven router: given rules, directly make a layout that meets
  them (vs. traditional DRC which only reports violations)
- Optimization for size, cost, power by changing a single line of code
- Continuous checking during design, not post-hoc verification

**OpenROAD (SDC-based)**:
- Timing constraints via Synopsys Design Constraints (SDC) format
- Placement density via `-density` parameter (0.0-1.0)
- Region constraints for macro/cell placement
- Net weighting for timing-critical paths

**Cadence OrCAD X / Siemens Xpedition**:
- Constraint managers with hierarchical rule specification
- Net classes group nets with similar requirements
- Rules propagate from schematic to layout
- Real-time DRC during interactive routing

**atopile** (open-source, code-first):
- Declarative `.ato` files describe circuits
- Constraints embedded in component/module definitions
- Compiles to KiCad-compatible output
- Version-controllable, testable design

### 4.4 Constraint Specification Language for autopcb

Based on the survey, the winning approach combines:
1. **Code-first** (like JITX/atopile): Constraints are part of the design source
2. **Hierarchical** (like commercial tools): Module-level constraints compose
3. **Typed** (like Solverang): Constraints have well-defined mathematical types
4. **Differentiable** (like DREAMPlace): All constraints produce gradients

```
// Hypothetical autopcb constraint syntax (maps to Solverang traits)
board my_board {
    boundary: rect(100mm, 80mm);
    layers: 4;

    module power_supply {
        components: [U1, C1, C2, L1, D1];
        constraints {
            clearance(C1, U1) <= 5mm;      // hard inequality
            clearance(C2, U1) <= 5mm;
            region: rect(0, 0, 30mm, 25mm); // containment
        }
    }

    net_class high_speed {
        nets: [CLK, DATA0..DATA7];
        impedance: 50ohm +/- 10%;
        max_length: 50mm;
        length_match: 1mm;
        spacing: 0.2mm;
    }

    optimize {
        minimize: total_wirelength;
        minimize: board_area;        // weighted
        subject_to: all_constraints;
    }
}
```

This maps directly to Solverang's `Objective` + `InequalityFn` + `Constraint`
traits.

---

## 5. Rust-Only Advantages

### 5.1 Existing Rust EDA Tools

| Project | Scope | Status | License |
|---------|-------|--------|---------|
| **LibrEDA** | ASIC physical design framework | Early, WIP | Libre |
| **Copper** | PCB editor | Early | Open source |
| **Atlantix-EDA** | PCB libraries | Active | Open source |

LibrEDA is the most relevant: a Rust framework for chip physical design with
placement (electron-placer), legalization (tetris-legalizer), routing
(mycelium-router), STA, and LEF/DEF/OASIS I/O. However, it is early-stage
and not production-ready.

**No Rust-native PCB analytical placer exists.** This is the gap autopcb fills.

### 5.2 Performance: Rust vs. Python EDA

DREAMPlace achieves its performance through CUDA kernels, not Python. The
Python layer is orchestration only. The performance stack:

```
Python (DREAMPlace) = Python orchestration + PyTorch + custom CUDA kernels
Rust (autopcb)      = Rust orchestration + Rust compute kernels
```

**Where Rust wins over Python+CUDA**:
1. **No GIL**: True multi-threading for placement + routing + constraint
   checking in parallel
2. **No serialization overhead**: Component data stays in Rust structs,
   no Python-to-C++ marshalling
3. **Predictable latency**: No garbage collection pauses during optimization
4. **Single binary**: No Python environment, no CUDA driver dependency
5. **CPU-only competitive**: For PCB-scale problems (hundreds of components),
   CPU-only Rust is competitive with GPU Python because the problem fits in
   cache and GPU kernel launch overhead dominates at small scale

**Where Python+GPU wins**:
- VLSI-scale problems (millions of cells): GPU parallelism essential
- Rapid prototyping of new algorithms
- Ecosystem of ML tools (for RL-based approaches)

**Performance estimate for PCB placement (500 components, 1000 nets)**:
- DREAMPlace (GPU): ~1-5 seconds (kernel launch overhead dominates)
- DREAMPlace (CPU): ~10-30 seconds
- Rust native (estimated): ~2-10 seconds (cache-friendly, no overhead)
- Python pure (estimated): ~100-300 seconds (interpreter overhead)

For PCB-scale problems, Rust is in the same ballpark as GPU without requiring
GPU hardware. This is the sweet spot.

### 5.3 Compile-Time Symbolic AD Advantages

Solverang's existing `Expr::differentiate()` provides compile-time symbolic
differentiation. For PCB optimization, this means:

**What compile-time symbolic AD can do that runtime AD cannot**:

1. **Zero-cost gradient evaluation**: The gradient code is generated at compile
   time and optimized by LLVM. No tape construction, no dynamic dispatch.
   For the wirelength WA model, the gradient is a fixed sequence of
   arithmetic operations per net.

2. **Guaranteed correctness**: The derivative is symbolically exact. No
   floating-point accumulation errors from tape replay. For placement where
   gradients drive convergence, this matters.

3. **Expression simplification before codegen**: `Expr::simplify()` can cancel
   terms, fold constants, and eliminate dead code before Cranelift JIT
   compiles the result. Runtime AD frameworks (JAX, PyTorch) rely on the
   compiler to do this, with less domain knowledge.

4. **Hessian availability without runtime cost**: For small PCB problems
   (N < 100 components = 300 variables), exact symbolic Hessians are feasible.
   This enables Newton's method for placement, which converges quadratically
   vs. Nesterov's linear convergence.

5. **Static analysis of constraint structure**: At compile time, the macro can
   determine which variables each constraint depends on, enabling optimal
   sparse Jacobian/Hessian assembly without runtime graph tracing.

**What compile-time AD cannot do (requiring runtime fallback)**:
- User-defined objective functions with control flow (if/else, loops)
- Objectives that call external functions (SPICE simulation, EM solvers)
- Very large problems where compile time becomes prohibitive (N > 50)

For these, Solverang's planned dual-number forward-mode AD provides a runtime
fallback. When Rust's `#[autodiff]` stabilizes (Enzyme-based, in progress),
LLVM-level AD becomes a third option.

### 5.4 Unique Solverang Capabilities for PCB

1. **Constraint decomposition**: Solverang's graph-based decomposition can
   identify independent sub-problems (e.g., separate functional blocks on a
   PCB) and solve them in parallel. No other PCB tool does this.

2. **Implicit differentiation**: Given a solved constraint system, Solverang
   can compute `dx/dp` (sensitivity of solution to parameter changes) via one
   back-substitution per parameter, using the already-factored Jacobian. This
   enables "what-if" analysis: how does changing a board dimension affect
   optimal component placement?

3. **Incremental re-solve**: When one component moves, only affected constraints
   need re-evaluation. The existing warm-start infrastructure supports this.

4. **Lagrange multiplier exposure**: After optimization, multipliers tell the
   user which constraints are binding and how much the objective would improve
   if a constraint were relaxed. "Relaxing the board width by 1mm would reduce
   wirelength by 15%" — no other PCB tool provides this insight.

---

## 6. Spec Language + Optimization: How Others Do It

### 6.1 Survey of Constraint Specification Approaches

| Tool | Approach | Constraints | Optimization |
|------|----------|-------------|--------------|
| **JITX** | Custom DSL (Stanza-based) | In code, checked continuously | Size/cost/power via parameter |
| **atopile** | Declarative `.ato` files | In module definitions | None (manual) |
| **tscircuit** | React/TypeScript components | In JSX props | Autorouter only |
| **SKiDL** | Python code | Python assertions | None |
| **KiCad** | GUI + Python scripting | Constraint manager + Python API | Plugin-based |
| **OpenROAD** | TCL + SDC files | SDC timing, TCL physical | Built-in placement/routing |
| **Cadence** | GUI + Skill/TCL | Constraint manager hierarchy | Built-in optimizer |

### 6.2 The Gap autopcb Should Fill

No existing tool combines:
1. A typed constraint language (not string-based rules)
2. Automatic differentiation of constraints
3. Gradient-based optimization of placement + routing
4. Sensitivity analysis (Lagrange multipliers)
5. Decomposition for parallel sub-problem solving

JITX comes closest with its code-first, constraint-driven approach, but it is:
- Closed-source and commercial
- Based on a custom language (Stanza), not a general-purpose language
- Does not expose optimization internals (no multipliers, no gradients)

### 6.3 Recommended autopcb Architecture

```
User spec (Rust DSL or config)
    |
    v
Constraint graph construction
    |
    v
Problem classification (Solverang Classify phase)
    |
    v
Decomposition into independent sub-problems
    |
    v
Per-subproblem optimization:
    - Analytical placement (Nesterov + WA wirelength + density penalty)
    - Differentiable routing estimation
    - Signal integrity constraints as penalties
    |
    v
Legalization (snap to grid, resolve remaining overlaps)
    |
    v
Detailed placement (local refinement)
    |
    v
Output: component positions + routing guides + sensitivity report
```

---

## 7. Concrete Recommendations for Implementation

### 7.1 Phase 1: Analytical PCB Placer (MVP)

Build on Solverang's optimization infrastructure (Phase 1 from `00_synthesis.md`):

1. **Weighted-Average wirelength model**: Implement as an `Objective` trait.
   The WA model is a ratio of exponential sums — fits naturally in `Expr`.
   ```
   WA_x(net) = sum(xi * exp(gamma*xi)) / sum(exp(gamma*xi))
             - sum(xi * exp(-gamma*xi)) / sum(exp(-gamma*xi))
   ```

2. **Bin-based density**: For PCB scale, divide the board into bins and
   penalize bins where component area exceeds capacity. Simpler than ePlace
   FFT, sufficient for hundreds of components.

3. **Non-overlap constraints**: Pairwise `InequalityFn` for component pairs.
   Use separating axis theorem for rotated rectangles. This is a hard
   constraint, not a penalty.

4. **Board boundary constraints**: Containment `InequalityFn` per component.

5. **Nesterov optimizer**: Implement as a new solver alongside BFGS/ALM.
   Nesterov with Lipschitz step size prediction is the standard for placement.

**Test**: Place 10-50 components on a board, minimizing wirelength subject to
non-overlap. Compare with random placement and force-directed placement.

### 7.2 Phase 2: Constraint-Driven Refinement

6. **Clearance constraints**: Per-pad-pair inequality constraints.
7. **Region constraints**: Functional blocks confined to board regions.
8. **Net class constraints**: Differential pair matching, length matching.
9. **Legalization**: Snap components to grid, resolve overlaps via QP.

### 7.3 Phase 3: Differentiable Routing Integration

10. **Routing DAG construction**: For each net, build a DAG of possible
    routing paths (simplified version of DGR).
11. **Gumbel-Softmax relaxation**: Make path selection differentiable.
12. **Congestion penalty**: Differentiable overflow cost on routing grid.
13. **Joint optimization**: Backpropagate routing gradients to placement.

### 7.4 Phase 4: Full Co-Optimization

14. **ADMM-based alternating optimization** (RUPlace-style).
15. **Signal integrity penalties**: Impedance, crosstalk, EMC.
16. **Sensitivity analysis**: Expose Lagrange multipliers to user.
17. **Design space exploration**: Parametric sweeps using `dx/dp`.

### 7.5 What NOT to Build

- **GPU acceleration**: PCB-scale problems (< 10K components) don't need it.
  CPU-only Rust with SIMD is sufficient and dramatically simpler.
- **RL-based placement**: The RL approaches (DeepPR, PCBAgent) require
  training data and infrastructure that doesn't justify the quality gain
  for a solver-based tool.
- **LLM integration**: PCBAgent's LLM agent for interactive optimization is
  interesting but orthogonal to the core solver.
- **Full VLSI flow**: Focus on PCB, not ASIC. Don't build legalization or
  detailed placement infrastructure that only matters at million-cell scale.

---

## 8. Key Papers and References

### Analytical Placement
- **ePlace**: Lu et al., "ePlace: Electrostatics Based Placement Using
  Nesterov's Method," DAC 2014. Foundational electrostatics-based density.
- **DREAMPlace**: Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU
  Acceleration for Modern VLSI Placement," DAC 2019. GPU-accelerated ePlace.
- **DREAMPlace 4.0**: Liang et al., "Timing-driven Global Placement with
  Momentum-based Net Weighting," DATE 2022. Multi-objective placement.
- **Cypress**: Zhang et al., "VLSI-Inspired PCB Placement with GPU
  Acceleration," ISPD 2025 Best Paper. PCB-specific analytical placement.
- **RUPlace**: Chen et al., "Optimizing Routability via Unified Placement and
  Routing Formulation," DAC 2025. ADMM-based joint optimization.

### Differentiable Routing
- **DGR**: NVIDIA, "Differentiable Global Router," DAC 2024. DAG forest +
  Gumbel-Softmax for differentiable routing.
- **INSTA**: DAC 2025 Best Paper. Differentiable statistical STA.
- **LEGO-Size**: ISPD 2025. Differentiable gate sizing.

### PCB-Specific
- **Clearance-Constrained PCB Placement**: Chen et al., DAC 2025.
  Pad-to-pad clearance + wire-area model.
- **PCBAgent**: ASP-DAC 2025. RL + LLM for PCB placement.
- **Net Separation PCB Placement**: Margin maximization, ASP-DAC 2022.

### Wirelength Models
- **WA Model**: Hsu et al., "TSV-Aware Analytical Placement Based on a Novel
  Weighted-Average Wirelength Model," TCAD 2013.
- **Moreau Envelope**: Yu et al., "On a Moreau Envelope Wirelength Model for
  Analytical Global Placement," DAC 2023.

### Constraint Specification
- **JITX**: Software-defined electronics with constraint-driven layout.
  https://www.jitx.com/
- **atopile**: Code-first circuit description. https://github.com/atopile/atopile
- **tscircuit**: React-based PCB design. https://tscircuit.com/

### Rust EDA
- **LibrEDA**: Rust framework for chip physical design. https://libreda.org/
- **"EDA Needs to be Using Rust"**: McCampbell, 2023. Industry argument for
  Rust adoption in EDA.

---

## 9. Summary: What Makes autopcb Unique

| Capability | DREAMPlace/Cypress | JITX | KiCad+Plugins | autopcb (planned) |
|---|---|---|---|---|
| Analytical placement | Yes (GPU) | Proprietary | No | Yes (CPU, Rust) |
| Differentiable routing | No (separate) | No | No | Yes (Phase 3) |
| Joint place+route optimization | No | Unknown | No | Yes (Phase 4) |
| Compile-time symbolic AD | No (runtime) | No | No | Yes (Solverang) |
| Constraint decomposition | No | No | No | Yes (Solverang) |
| Lagrange multiplier exposure | No | No | No | Yes (Solverang) |
| Sensitivity analysis (dx/dp) | No | No | No | Yes (Solverang) |
| Code-first constraints | No | Yes | Partial (Python) | Yes (Rust DSL) |
| Open source | Yes (Apache-2.0) | No | Yes (GPL) | Yes (planned) |
| No GPU required | No | Unknown | Yes | Yes |

The unique value proposition: **a solver-native PCB optimization tool that
treats placement and routing as a single constrained optimization problem,
with exact symbolic derivatives, constraint decomposition, and sensitivity
analysis — all in a single Rust binary with no GPU required.**
