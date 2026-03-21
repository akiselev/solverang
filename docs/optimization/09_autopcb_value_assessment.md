# AutoPCB Value Assessment: Would Solverang Optimization Help?

Concrete analysis of whether adding true optimization (`min f(x) s.t. constraints`)
to Solverang would significantly improve the autopcb PCB design tool.

---

## Background: How AutoPCB Uses Solverang Today

The placer (`autopcb-placement`) already uses Solverang's LM solver heavily:

- **8 constraint types** implemented as `Constraint` trait objects: BoardContainment
  (4 inequality rows via slack variables), ComponentClearance (1 elliptical inequality
  row), EdgePlacement (1 equality row), DirectionalOrdering (1 inequality row),
  NearConstraint (1 inequality row), RegionContainment (4 inequality rows),
  FixedPosition (2-3 equality rows), SmoothHpwlConstraint (2 soft rows).
- **Inequality constraints use the slack-variable trick**: `g(x) - s^2 = 0`,
  converting each inequality into an equality that the LM solver can handle.
- **HPWL is a soft constraint**: `SmoothHpwlConstraint` returns `is_soft() -> true`
  and `weight() -> self.weight` (default 0.01). The solver treats the weighted
  residuals like any other equation -- LM minimizes `||r||^2`, so HPWL is
  implicitly minimized alongside hard constraint satisfaction.
- **Multi-phase pipeline**: LM solve (gamma=2) -> add clearance pairs -> re-solve
  (gamma=10) -> grid snap -> structured legalization -> optional part/pin swap ->
  optional SA refinement.

The router (`autopcb-router`) has **stubs** for two Solverang integrations (DRC
repair, rubber-banding) that currently fall back to geometric-only or no-op.

---

## Use Case Assessments

### A. Placement as Optimization

**Current approach**: HPWL is a soft constraint with weight 0.01. LM minimizes
`sum(w_i * r_i)^2` where HPWL residuals sit alongside containment/clearance
residuals. This means the solver tries to drive HPWL residuals toward zero (minimize
wire length) while simultaneously satisfying hard constraints. The weight balances
the two goals.

**What true optimization would change**: Instead of one big residual vector, the
problem becomes `minimize HPWL(x) subject to containment(x) >= 0, clearance(x) >= 0`.
ALM or SQP would handle this.

**Assessment: MODERATE improvement.**

The current soft-constraint approach actually works reasonably well for placement
because:

1. LM-with-soft-constraints is mathematically close to a penalty method
   (`min f(x) + rho * ||c(x)||^2`), which IS an optimization algorithm -- just a
   primitive one. The solver is already doing weighted least-squares optimization.
2. The weight of 0.01 was tuned to let hard constraints dominate, which is the right
   behavior.
3. The real quality bottleneck is the subsequent SA phase (discrete rotation, swap
   decisions), not the continuous LM solve.

Where true optimization would help:

- **Weight tuning becomes unnecessary.** With ALM, hard constraints are hard and the
  objective is the objective. No balancing act. The current 0.01 weight is fragile --
  too high and components escape the board, too low and HPWL barely improves.
- **Feasibility guarantees.** The current solver can converge to a point where hard
  constraints are partially violated (PartiallySolved). ALM with an inner loop would
  provide stronger feasibility guarantees.
- **Better HPWL at convergence.** The soft-constraint approach under-optimizes HPWL
  because the solver "wastes" effort on reducing already-small constraint residuals.
  True optimization would push HPWL further once constraints are satisfied.

Estimated HPWL improvement: 5-15% on typical boards. Not transformative, but
meaningful -- it compounds through the routing phase (shorter rats = shorter traces).

### B. Multi-Objective Placement

**Current approach**: Only HPWL weight exists. No congestion metric in the
analytical phase (congestion is only a weight in the SA cost function, and even that
defaults to 0.0). No area minimization.

**What true optimization would change**: Weighted-sum or epsilon-constraint approach
to `minimize w1*HPWL + w2*congestion + w3*area`.

**Assessment: LOW-to-MODERATE improvement.**

Multi-objective placement is theoretically appealing but practically limited:

1. **Congestion requires routing estimates.** The CongestionOracle trait exists but
   only integrates with SA, not the analytical solver. Adding it as an objective
   would require either a fast congestion estimator (RUDY-style) or coupling to
   the router, both of which are substantial engineering independent of the solver.
2. **Area minimization conflicts with clearance.** Components are already packed as
   tightly as clearance allows. There is no free variable to trade.
3. **Pareto fronts are overkill for PCB placement.** Designers want one good
   placement, not a menu of trade-offs. Weighted-sum with sensible defaults
   (w_congestion = 0.3, w_hpwl = 1.0) would suffice.
4. **The SA phase already does multi-objective** via its cost function
   (HPWL + overlap + congestion + net crossings). Adding more objectives to the
   analytical LM phase is less impactful because SA re-optimizes afterward.

The main value would be replacing the SA's heuristic cost function with a proper
constrained optimization, but SA's strength is its ability to escape local minima
through random moves -- something gradient-based optimization cannot do. The two
approaches are complementary, not substitutes.

### C. DRC Repair as Optimization

**Current approach**: Completely unimplemented. The `repair_with_solverang` function
is a stub that logs a warning and returns all violations unrepaired.

**What true optimization would enable**: For each clearance violation, set up
`minimize sum(displacement_i^2) subject to clearance(i,j) >= required` and nudge
trace vertices to fix the violation.

**Assessment: HIGH value -- this is the best use case.**

DRC repair is where optimization shines:

1. **The problem is naturally an optimization.** You want minimum-displacement
   corrections subject to clearance constraints. There is no reasonable formulation
   as pure constraint satisfaction.
2. **The problem is small.** Each repair involves 2-10 trace vertices (4-20
   variables) and 1-5 clearance constraints. This is exactly the sweet spot for
   ALM or SQP with exact Hessians.
3. **No alternative exists.** Without optimization, the only options are (a) rip-up
   and re-route (expensive, may cascade), or (b) heuristic nudging (fragile,
   may create new violations). Optimization gives the mathematically minimal fix.
4. **It compounds.** A reliable DRC repair pass means the router can be more
   aggressive, knowing violations will be cleaned up. This can improve overall
   route completion rate.
5. **The infrastructure is ready.** The `repair.rs` file already has the function
   signature, test scaffolding, and feature-flag plumbing. It just needs the
   solver integration.

This is the **strongest argument for adding optimization to Solverang** from the
autopcb perspective. The stub is literally waiting for it.

### D. Rubber-Banding as Optimization

**Current approach**: Pure geometric projection. Each internal vertex is moved
toward the straight line between its neighbors. No clearance checking -- violations
are caught by a subsequent DRC pass.

**What true optimization would enable**: `minimize total_trace_length subject to
clearance(vertex, obstacle) >= min_clearance` for all vertices and nearby obstacles.

**Assessment: MODERATE-to-HIGH improvement.**

The current geometric approach has a fundamental limitation: it cannot find
paths that go "around" obstacles. It only shortens by pulling vertices toward the
straight-line path. A constrained optimization would:

1. **Respect clearance during tightening.** Currently, rubber-banding may introduce
   violations that require a subsequent DRC pass, potentially un-doing the
   improvement. Optimization would produce violation-free results directly.
2. **Handle multi-vertex coordination.** The current greedy per-vertex approach is
   non-optimal for chains of vertices where moving one affects the optimal position
   of others. Optimization would jointly optimize all vertices.
3. **Enable clearance-hugging.** The optimal trace often hugs an obstacle at exactly
   the minimum clearance distance. Geometric projection cannot find this path;
   constrained optimization can.

The problem structure is favorable: 2-20 vertex positions (4-40 variables), trace
length is a smooth objective, clearance constraints are smooth. This is a natural
small-scale NLP.

The `rubber_band_solverang` function stub already exists with the right signature
and falls back to geometric rubber-banding. Implementation is straightforward once
ALM/SQP is available.

### E. Length Matching as Optimization

**Current approach**: Not implemented at all in the current codebase. The
implementation plan lists it as a Phase 8 item: "Length matching / serpentine
insertion for matched net groups."

**What true optimization would enable**: `minimize max(length_i) - min(length_i)
subject to clearance constraints, route topology preserved`.

**Assessment: HIGH value for the feature, but optimization is only part of it.**

Length matching is a critical feature for high-speed PCB design (DDR, USB, HDMI).
The optimization formulation is clean:

1. **Post-route length equalization**: Given routed nets, adjust vertex positions to
   equalize total lengths. This is a well-posed NLP: minimize length variance
   subject to clearance and topology constraints.
2. **Serpentine insertion as optimization**: Place serpentine segments to add length
   to short nets while minimizing board area consumed. This is harder -- it
   involves both continuous (segment positions) and discrete (number of meanders)
   decisions.

However, the bulk of the engineering work is **not** the optimizer:
- Identifying matched net groups from design rules
- Computing trace lengths through vias and layer changes
- Generating initial serpentine patterns
- Maintaining coupling in differential pairs

The optimizer would be the final 20% that makes it work well rather than being a
crude add-then-check loop.

### F. Differential Pair Placement

**Current approach**: Not implemented. Differential pairs are recognized in the
Altium file format (DifferentialPairsRouting rule kind) but no placement-level
handling exists.

**What true optimization would enable**: `minimize HPWL subject to
|position(net_A) - position(net_B)| = target_spacing` for differential pair
components.

**Assessment: LOW incremental value from optimization specifically.**

Differential pair spacing is fundamentally a **constraint**, not an optimization
objective. The current constraint solver can already handle this:

```rust
// This is just a distance-equals-constant constraint
struct DiffPairSpacing {
    // residual = distance(a, b) - target_spacing
}
```

No optimization extension is needed. The existing `Constraint` trait with an
equality residual handles this directly. The missing piece is detecting differential
pairs from the netlist and generating the constraint, not solving it.

Where optimization would matter is **differential pair routing** (maintaining
coupling along the route), but that is primarily a router algorithm question, not
a solver question.

### G. Spec Language Optimization

**Current approach**: The spec grammar defines an `optimize { ratsnest: true,
ratsnest_weight: 1.0 }` block, but it only controls the HPWL weight. No ability
to express custom objectives, inequality constraints, or multi-objective trade-offs.

**What true optimization would enable**:

```
placement {
    minimize { total_wirelength }
    subject_to {
        clearance >= 0.2mm
        board_containment
        U1 left_of U2
    }
}
```

Or more ambitiously:

```
placement {
    minimize { max_trace_length - min_trace_length }
    subject_to {
        all_nets_routable
        clearance >= 0.15mm
    }
}
```

**Assessment: MODERATE value -- nice to have, not transformative.**

The spec language improvement is real but bounded:

1. **Most placement objectives are the same.** Every board wants minimum
   wirelength, minimum area, and no violations. Exotic objectives are rare.
   A well-tuned default (`minimize { wirelength }`) covers 90% of cases.
2. **The LLM generating specs doesn't need this expressiveness.** It can already
   express intent through constraints (near, left_of, edge). Adding `minimize { }`
   blocks makes the spec more principled but doesn't unlock new capabilities the
   constraint-based approach cannot approximate.
3. **The real bottleneck is constraint generation, not objective specification.**
   The LLM's value comes from translating design intent ("keep decoupling caps
   close to the MCU") into constraints, not from specifying mathematical
   objectives.

That said, `subject_to` with explicit inequality syntax would be genuinely useful
for expressing design rules that are currently hard-coded (clearance values,
region bounds). And it would make the system more self-documenting.

---

## Summary Table

| Use Case | Value | Effort | Priority |
|----------|-------|--------|----------|
| **C. DRC repair** | HIGH | LOW (small NLPs, stub ready) | **1st** |
| **D. Rubber-banding** | MOD-HIGH | LOW (small NLPs, stub ready) | **2nd** |
| **E. Length matching** | HIGH | HIGH (mostly non-solver work) | **3rd** |
| **A. Placement as optimization** | MODERATE | MODERATE | 4th |
| **G. Spec language** | MODERATE | MODERATE | 5th |
| **B. Multi-objective** | LOW-MOD | HIGH | 6th |
| **F. Differential pairs** | LOW | LOW | 7th (constraint only) |

---

## Honest Bottom Line

Adding optimization to Solverang would provide **meaningful but not transformative**
improvement to autopcb's placement quality. The current soft-constraint approach is a
decent approximation of penalty-method optimization, and the SA refinement phase
catches much of what the analytical solver misses.

Where optimization is **genuinely needed** is in the **router post-processing**:
DRC repair and rubber-banding. These are problems that have no reasonable
formulation without an objective function, and they are currently unimplemented
(stubs only). The problem sizes are small (5-40 variables), the mathematical
structure is clean, and the infrastructure (function signatures, feature flags,
test scaffolding) is already in place.

The recommended implementation order:

1. **ALM with LM inner loop** (Phase 1 of optimization extension)
2. **DRC repair implementation** using ALM -- fill in `repair_with_solverang()`
3. **Rubber-band optimization** using ALM -- fill in `rubber_band_solverang()`
4. **Placement-as-optimization** -- replace soft-constraint HPWL with proper
   objective, keeping the same pipeline
5. **Spec language `minimize`/`subject_to`** -- expose the optimizer through the
   agent-facing grammar

Steps 2 and 3 would deliver immediate, tangible improvements to route quality with
minimal additional autopcb-side engineering. The Solverang optimization extension is
the bottleneck for these features, not the autopcb integration.

### What Would NOT Improve

- **SA refinement quality**: SA is inherently a stochastic search; replacing its cost
  function with a formal objective would not change its ability to escape local
  minima.
- **Clustering/partitioning**: This is a graph algorithm, not an optimization
  problem in the NLP sense.
- **Pin/part swap**: These are combinatorial decisions. The current greedy swap
  passes are appropriate.
- **Route completion rate**: The PathFinder negotiation algorithm's convergence is
  independent of the solver. Optimization would improve route quality (shorter,
  cleaner traces), not whether routing succeeds.
