# Agent-in-the-Loop PCB Design: Sensitivity-Driven Constraint Negotiation

## Overview

This document describes an architecture where an AI agent (Claude Code) and a
constraint solver (Solverang-based autopcb placer/router) collaborate in a
closed feedback loop. The agent writes a PCB spec in the ops DSL, the solver
places and routes the board, and the solver returns Lagrange multipliers
alongside the solution. The agent reads these multipliers -- which encode the
*marginal cost* of each constraint -- and uses them to decide what to relax,
tighten, or restructure. This cycle repeats until the design is satisfactory.

No existing EDA tool exposes constraint multipliers to an AI agent. This
architecture turns the agent from a one-shot spec generator into a
*constraint negotiator* that converges on a good design through informed
trade-offs.

---

## 1. Sensitivity Analysis: What Lagrange Multipliers Tell the Agent

### 1.1 Mathematical Foundation

After the solver converges on `min HPWL(x) s.t. constraints(x) = 0`, the
Lagrange multipliers are the dual variables of the KKT system:

```
grad HPWL + sum_i lambda_i * grad g_i(x) = 0
g_i(x) = 0   for all i
```

The key interpretation: **lambda_i = -d(HPWL*) / d(b_i)**, where b_i is the
right-hand side of constraint i. That is, lambda_i tells you how much the
optimal HPWL would decrease if you relaxed constraint i by one unit.

For inequality constraints (implemented via the slack-variable trick
`g(x) - s^2 = 0` in autopcb), the multiplier lambda is non-negative when
the constraint is active (slack = 0) and zero when inactive (slack > 0). An
inactive constraint does not affect the objective and can be ignored.

For the current LM solver, the pseudo-multipliers are recovered from:

```
lambda = -(J^T J)^{-1} J^T r
```

where J is the Jacobian at the solution and r is the residual vector. These
approximate true Lagrange multipliers and become exact as the residual
approaches zero (fully converged solution).

### 1.2 Per-Constraint Multiplier Interpretation

#### BoardContainment (4 inequality rows per component)

**Residual structure:**
```
r_0 = (comp_min_x - board_min_x - clearance) - s_0^2   [left edge]
r_1 = (board_max_x - clearance - comp_max_x) - s_1^2   [right edge]
r_2 = (comp_min_y - board_min_y - clearance) - s_2^2   [bottom edge]
r_3 = (board_max_y - clearance - comp_max_y) - s_3^2   [top edge]
```

**Multiplier meaning:**
- lambda_0 for left containment: "Moving the board left edge outward by 1mm
  (or shrinking the component by 1mm) would improve HPWL by lambda_0 mm."
- lambda_1 for right containment: same, for the right edge.

**What the agent sees:**

| Constraint | Component | lambda | Interpretation |
|-----------|-----------|--------|---------------|
| BoardContainment.right | U1 (MCU, 15x15mm) | 12.4 | U1 is pressed against the right edge. Widening the board 1mm rightward saves 12.4mm of wirelength. |
| BoardContainment.left | C3 (0402 cap) | 0.0 | C3 is not touching the left edge. This constraint is inactive. |
| BoardContainment.top | U2 (regulator) | 8.7 | U2 is pressed against the top. Board height is limiting. |

**Agent decision rule:** If max(lambda_containment) > threshold (e.g., > 5.0),
consider enlarging the board in that direction. If the board outline is fixed,
this tells the agent which components are "fighting" the board boundary and
should be moved toward the interior -- possibly at the cost of relaxing other
constraints.

**Concrete example:**
The agent specifies a 50mm x 30mm board. The solver reports lambda_right = 12.4
for U1 and lambda_top = 8.7 for U2. The agent knows: "The board is too narrow
horizontally and too short vertically, but horizontal is the worse bottleneck."
It modifies the spec to 55mm x 33mm and re-runs. HPWL drops by approximately
12.4 * 5 + 8.7 * 3 ~= 88mm (linear approximation, actual improvement may differ).

#### ComponentClearance (1 inequality row per pair)

**Residual structure:**
```
g = (dx/combined_hw)^2 + (dy/combined_hh)^2 - 1.0
r = g - s^2
```

This is an elliptical separation constraint. g >= 0 means the bounding boxes
(plus clearance margin) do not overlap.

**Multiplier meaning:**
- lambda for clearance(U1, U2): "Reducing the clearance requirement between U1
  and U2 by 1mm (or equivalently shrinking one of them by 1mm on each axis)
  would improve HPWL by lambda mm."

**What the agent sees:**

| Pair | Clearance (mm) | lambda | Interpretation |
|------|---------------|--------|---------------|
| U1 -- U2 | 0.5 | 23.1 | These two ICs are fighting each other. Relaxing clearance by 0.1mm saves ~2.3mm wirelength. |
| R1 -- R2 | 0.5 | 0.0 | Passive pair is not tight. Clearance is inactive. |
| U1 -- C1 | 0.5 | 0.8 | Decoupling cap near MCU, slightly tight but not critical. |

**Agent decision rule:** Sort all clearance multipliers descending. The top
entries identify the *bottleneck pairs* -- the component pairs whose proximity
requirements dominate the layout. The agent can:

1. Reduce clearance for that specific pair if manufacturing allows (e.g.,
   `clearance { U1, U2: 0.3mm }`).
2. Move one component to a different region to reduce congestion.
3. Rotate one component to change its bounding box aspect ratio.

**Concrete example:**
The agent initially specifies `clearance { all: 0.5mm }`. The solver reports
that clearance(U1, U2) has lambda = 23.1, far above all other pairs. The agent
modifies the spec to `clearance { all: 0.5mm, pair U1 U2: 0.25mm }` since those
two components are on the same power island with matching ground planes and the
reduced clearance is manufacturable. On re-solve, HPWL drops by ~5.8mm.

#### EdgePlacement (1 equality row)

**Residual structure:**
Pins a component to a specific board edge at a given inset:
```
r = comp_edge_coordinate - (board_edge + inset)   [equality]
```

For example, `edge: Top, inset: 2mm` means the component's top edge
is constrained to sit at `board_max_y - 2mm`.

**Multiplier meaning:**
- lambda for edge_placement(J1): "Allowing J1 to float 1mm from its specified
  edge position would improve HPWL by |lambda| mm. The sign of lambda indicates
  *which direction* the solver wants to move J1."

**What the agent sees:**

| Component | Edge | lambda | Interpretation |
|-----------|------|--------|---------------|
| J1 (USB connector) | Top | -4.2 | Solver wants to push J1 further from the top edge (negative = toward board interior). The inset is too tight. |
| J2 (barrel jack) | Left | -0.3 | Left placement is nearly free. |

**Agent decision rule:** High |lambda| on an edge constraint means the edge
assignment is costly. The agent considers: Is this edge assignment a hard
physical requirement (connector must be accessible)? If not, relax or remove it.
If it is a hard requirement, accept the cost and look elsewhere.

**Concrete example:**
J1 is a USB-C connector that must be on the top edge. lambda = -4.2. The agent
cannot remove this constraint, but it can adjust the inset: changing from
`inset: 2mm` to `inset: 3mm` gives J1 more room. Or the agent can move nearby
components away from J1 to reduce congestion in that region.

#### DirectionalOrdering (1 inequality row)

**Residual structure:**
```
g = edge_separation - gap   [e.g., for LeftOf: b_min_x - a_max_x - gap]
r = g - s^2
```

**Multiplier meaning:**
- lambda for left_of(U1, U2, gap=2mm): "Reducing the gap requirement by 1mm
  would improve HPWL by lambda mm. Alternatively, allowing U1 and U2 to be
  side-by-side instead of strictly left_of would save even more."

**What the agent sees:**

| Constraint | lambda | Interpretation |
|-----------|--------|---------------|
| U1 left_of U2 (gap: 2mm) | 6.3 | This ordering is expensive. The solver wants to move U2 closer to or past U1. |
| C1 above C2 (gap: 0mm) | 0.0 | Vertical ordering between caps is free -- they naturally stack. |

**Agent decision rule:** If lambda > threshold for a directional constraint, ask:
Is this ordering truly necessary? Many directional constraints express "I want the
MCU near the left and the connector on the right" -- they are preferences, not
physics. The agent can reduce the gap, or replace `left_of` with a `near`
constraint if the ordering does not matter physically.

#### NearConstraint (1 inequality row)

**Residual structure:**
```
g = max_dist^2 - (dx^2 + dy^2)
r = g - s^2
```

**Multiplier meaning:**
- lambda for near(C1, U1, max_dist=5mm): "Increasing the allowed distance by
  1mm (from 5mm to 6mm) would improve HPWL by lambda mm. The constraint is
  preventing the solver from spreading components out."

**What the agent sees:**

| Pair | max_dist | lambda | Interpretation |
|------|----------|--------|---------------|
| C1 -- U1 | 5mm | 3.4 | Decoupling cap proximity is somewhat costly. |
| C2 -- U1 | 3mm | 9.1 | This cap is too close. 3mm is very tight for the available space. |
| R1 -- U2 | 10mm | 0.0 | Inactive -- components are already within 10mm naturally. |

**Agent decision rule:** When lambda for a `near` constraint is high, the agent
should check whether the proximity requirement is grounded in electrical
reality. A decoupling capacitor truly needs to be within a few mm of its IC's
power pin. A pull-up resistor might not need to be as close. The agent can:

1. Increase max_distance if the electrical requirement allows.
2. Keep the constraint but accept the wirelength cost.
3. Restructure: place both components on the same side of the board.

#### RegionContainment (4 inequality rows)

**Residual structure:**
Same pattern as BoardContainment but for a user-defined rectangular region:
```
r_0 = (x - region.min_x) - s_0^2
r_1 = (region.max_x - x) - s_1^2
r_2 = (y - region.min_y) - s_2^2
r_3 = (region.max_y - y) - s_3^2
```

**Multiplier meaning:**
- lambda_0 for region left boundary: "Expanding the region leftward by 1mm would
  improve HPWL by lambda_0 mm."

**Agent decision rule:** If multiple region constraints have high multipliers,
the region is too small for its assigned components. The agent can enlarge the
region, move some components to a different region, or remove the region
constraint entirely.

#### FixedPosition (2-3 equality rows)

**Residual structure:**
```
r_0 = x - target_x
r_1 = y - target_y
r_2 = theta - target_theta   [optional]
```

**Multiplier meaning:**
- lambda_0: "Moving the fixed component 1mm in x would improve HPWL by
  |lambda_0| mm."
- lambda_1: same for y.
- lambda_2: sensitivity to rotation (in mm HPWL per radian).

**Agent decision rule:** A fixed component with high multipliers is poorly
placed. The agent should question: Is the exact position mandatory, or is the
component just "placed there for now"? If the agent can loosen to a region
constraint instead, the solver gains freedom. If the position is non-negotiable
(e.g., mounting holes, test points), the multiplier tells the agent how much
that constraint costs.

**Concrete example:**
A mounting hole M1 is fixed at (25, 15). lambda_0 = 0.3, lambda_1 = 0.2.
Low multipliers: this position is cheap, keep it. A test point TP1 fixed at
(10, 5) has lambda_0 = 7.8. High: the solver would much prefer TP1 elsewhere.
Agent converts the fixed constraint to `near { TP1, corner: 8mm }`.

#### SmoothHpwlConstraint (2 soft rows per net)

**Residual structure:**
Smooth log-sum-exp approximation to HPWL:
```
r_x = weight * (gamma * ln(sum exp(x_i/gamma)) + gamma * ln(sum exp(-x_i/gamma)))
r_y = weight * (gamma * ln(sum exp(y_i/gamma)) + gamma * ln(sum exp(-y_i/gamma)))
```

**Multiplier meaning:**
This constraint is soft (`is_soft() -> true`), so its multiplier has a different
interpretation. Because HPWL is the *objective* being minimized (encoded as a
soft constraint), its multiplier tells the agent the relative contribution of
each net to total wirelength pressure.

A net with high HPWL multiplier magnitude is a *long net* that the solver is
spending significant effort trying to shorten. The agent can:

1. Add proximity constraints between pins of that net.
2. Restructure the schematic to use local bypass instead of long traces.
3. Accept the length if it is a power rail (long traces are acceptable for
   power distribution).

### 1.3 Summary Table: Multiplier Quick Reference

| Constraint | Rows | lambda units | "If I relax by 1mm..." |
|-----------|------|-------------|----------------------|
| BoardContainment | 4 | mm HPWL / mm board edge | HPWL drops by lambda mm |
| ComponentClearance | 1 | mm HPWL / mm clearance | HPWL drops by lambda mm |
| EdgePlacement | 1 | mm HPWL / mm inset | HPWL drops by \|lambda\| mm |
| DirectionalOrdering | 1 | mm HPWL / mm gap | HPWL drops by lambda mm |
| NearConstraint | 1 | mm HPWL / mm^2 dist^2 | HPWL drops by lambda per mm^2 of slack |
| RegionContainment | 4 | mm HPWL / mm region edge | HPWL drops by lambda mm |
| FixedPosition | 2-3 | mm HPWL / mm position | HPWL drops by \|lambda\| mm per mm moved |
| HPWL (soft) | 2/net | relative net pressure | Not directly actionable as relaxation |

**Important caveat:** These are *linear approximations* valid near the current
solution. Large relaxations (e.g., doubling the board size) will not see
HPWL improvements proportional to lambda * delta. The agent should make
*incremental* changes and re-solve.

---

## 2. Spec Language Extensions for Optimization

### 2.1 Current Syntax

The existing spec language supports:

```
placement {
    optimize { ratsnest: true, ratsnest_weight: 1.0 }
    clearance { all: 0.5mm, edge: 0.2mm }

    place U1 { at: (25mm, 15mm), fixed: true }
    place C1 { near: U1, max_distance: 5mm }

    U1 left_of U2 { gap: 2mm }

    group "power" { components: [U3, C4, C5, L1] }
}
```

### 2.2 Extended Syntax: Objective and Subject-To

```
placement {
    // Explicit objective declaration (replaces "optimize { ratsnest: true }")
    minimize { wirelength }

    subject_to {
        clearance { all: 0.5mm }
        board_containment
        U1 left_of U2 { gap: 2mm }
    }

    place U1 { at: (25mm, 15mm) }
    place C1 { near: U1, max_distance: 5mm }
}
```

When `minimize` is present, the solver uses the true optimization pipeline
(ALM/SQP) rather than the soft-constraint LM approach. The `subject_to` block
makes the distinction between *objective* and *constraints* explicit.

If `minimize` is absent, the solver falls back to the current LM soft-constraint
behavior for backward compatibility.

#### Multi-objective

```
placement {
    minimize {
        wirelength { weight: 0.7 }
        congestion { weight: 0.3 }
    }

    subject_to {
        clearance { all: 0.3mm }
    }
}
```

The solver computes a weighted-sum objective:
`f(x) = 0.7 * HPWL(x) + 0.3 * congestion(x)`. Multipliers are reported
relative to this combined objective.

#### Available objectives

| Keyword | Meaning | Units |
|---------|---------|-------|
| `wirelength` | Sum of per-net HPWL | mm |
| `congestion` | Routing congestion estimate (RUDY-style) | dimensionless |
| `area` | Bounding box area of all components | mm^2 |
| `max_net_length` | Length of the longest net | mm |
| `length_variance` | Variance across matched net group lengths | mm^2 |

### 2.3 Constraint Relaxation Hints

The agent can annotate constraints with metadata the solver uses for sensitivity
reporting and automatic relaxation:

```
subject_to {
    clearance {
        all: 0.5mm
        relaxable: true        // solver may report relaxation suggestions
        min: 0.15mm            // never relax below this
        priority: low          // relax low-priority first
    }

    near C1, U1 {
        max_distance: 5mm
        relaxable: true
        max: 10mm              // never relax beyond this
        priority: high         // electrical requirement, relax last
    }

    // Non-relaxable: agent commits to this constraint
    U1 left_of U2 { gap: 2mm, relaxable: false }

    // Fixed constraints are never relaxable
    place J1 { edge: top, inset: 2mm }
}
```

The `relaxable` / `priority` / `min` / `max` fields do not change the solver's
behavior. They annotate the sensitivity report so the agent (or a human
reviewing the report) can quickly identify which constraints are candidates for
adjustment.

### 2.4 Sensitivity Output Format

The solver outputs a JSON report alongside the placement result:

```json
{
  "status": "Converged",
  "objective": {
    "name": "wirelength",
    "value_mm": 842.3,
    "unit": "mm"
  },
  "iterations": 187,
  "duration_ms": 2340,

  "sensitivity": {
    "binding_constraints": [
      {
        "type": "ComponentClearance",
        "entities": ["U1", "U2"],
        "clearance_mm": 0.5,
        "multiplier": 23.1,
        "interpretation": "Relaxing clearance(U1,U2) by 0.1mm saves ~2.3mm HPWL",
        "relaxable": true,
        "priority": "low",
        "suggested_value_mm": 0.3
      },
      {
        "type": "BoardContainment",
        "entity": "U1",
        "edge": "right",
        "multiplier": 12.4,
        "interpretation": "Widening the board 1mm rightward saves ~12.4mm HPWL",
        "relaxable": false,
        "priority": null,
        "suggested_value_mm": null
      },
      {
        "type": "NearConstraint",
        "entities": ["C2", "U1"],
        "max_distance_mm": 3.0,
        "multiplier": 9.1,
        "interpretation": "Increasing max_distance(C2,U1) by 1mm saves ~9.1mm HPWL",
        "relaxable": true,
        "priority": "high",
        "suggested_value_mm": 5.0
      }
    ],

    "inactive_constraints": [
      {
        "type": "NearConstraint",
        "entities": ["R1", "U2"],
        "max_distance_mm": 10.0,
        "multiplier": 0.0,
        "note": "Naturally satisfied. Consider removing to reduce solver work."
      }
    ],

    "constraint_summary": {
      "total": 47,
      "binding": 12,
      "inactive": 35,
      "violated": 0,
      "max_multiplier": 23.1,
      "sum_multipliers": 78.4
    }
  },

  "components": [
    {
      "designator": "U1",
      "x_mm": 32.4,
      "y_mm": 18.7,
      "rotation_deg": 0.0,
      "binding_constraints": ["BoardContainment.right", "Clearance(U1,U2)"]
    }
  ],

  "suggestions": [
    {
      "action": "relax_clearance",
      "target": ["U1", "U2"],
      "current_mm": 0.5,
      "suggested_mm": 0.3,
      "estimated_hpwl_improvement_mm": 4.6,
      "priority": "low",
      "rationale": "Highest multiplier. These components share a ground plane; 0.3mm clearance is manufacturable with standard process."
    },
    {
      "action": "enlarge_board",
      "direction": "right",
      "current_mm": 50.0,
      "suggested_mm": 53.0,
      "estimated_hpwl_improvement_mm": 37.2,
      "priority": null,
      "rationale": "U1 is pressed against right edge. 3mm expansion has highest ROI."
    }
  ]
}
```

The `suggestions` array is generated by the solver's post-processing phase. It
ranks relaxation candidates by `|multiplier| * available_slack` where
`available_slack = current_value - min_value` for relaxable constraints, and
estimates the HPWL improvement from the linear approximation
`delta_HPWL ~= lambda * delta_constraint`.

---

## 3. Agent Feedback Protocol

### 3.1 Architecture

```
+-------+        .spec         +----------+       PcbIr        +---------+
| Agent | ------------------> | Compiler | -----------------> | Placer  |
|       |                      +----------+                    |         |
|       |                                                      | Solver  |
|       |    sensitivity.json  +----------+  PlacementResult   |         |
|       | <------------------ | Reporter | <----------------- |         |
+-------+                      +----------+                    +---------+
    |                                                              |
    | decision                                                     |
    | (relax/tighten/                                              |
    |  restructure)                                                |
    v                                                              |
+-------+        .spec (v2)   +----------+                         |
| Agent | ------------------> | Compiler | ----> [next iteration] -+
+-------+                      +----------+
```

### 3.2 Solver Output Schema

The solver produces two outputs:

1. **PlacementResult** (existing): component positions, HPWL, overlap count.
2. **SensitivityReport** (new): the JSON structure from section 2.4.

These are written to files:

```
output/
  placement_result.json     # positions, HPWL, status
  sensitivity_report.json   # multipliers, suggestions
  placement_log.txt         # solver trace (iterations, convergence)
```

### 3.3 Fields the Agent Needs

| Field | Type | Purpose |
|-------|------|---------|
| `status` | enum | Converged / PartiallySolved / Infeasible / MaxIter |
| `objective.value_mm` | f64 | Current HPWL (or combined objective) |
| `sensitivity.binding_constraints` | array | All active constraints with multipliers |
| `sensitivity.binding_constraints[].multiplier` | f64 | The Lagrange multiplier value |
| `sensitivity.binding_constraints[].relaxable` | bool | Whether the agent marked this as relaxable |
| `sensitivity.binding_constraints[].priority` | string | Agent-assigned priority |
| `sensitivity.inactive_constraints` | array | Constraints that are not binding |
| `suggestions` | array | Solver-generated relaxation recommendations |
| `components[].binding_constraints` | array | Which constraints each component participates in |

### 3.4 Agent Decision Algorithm

The agent follows a priority-ordered decision procedure:

**Step 1: Check feasibility.**
If status is `Infeasible`, the constraints are mutually contradictory. Look at
the constraints with the largest multipliers (these are the ones the solver
"wants" to violate the most) and relax or remove them.

**Step 2: Check for high-multiplier binding constraints.**
Sort `binding_constraints` by `|multiplier|` descending. For each:

```python
for constraint in sorted(binding, key=lambda c: abs(c.multiplier), reverse=True):
    if constraint.multiplier < INSIGNIFICANT_THRESHOLD:  # e.g., 0.5
        break  # remaining constraints are cheap, stop looking

    if not constraint.relaxable:
        # Hard constraint. Log it as a known cost. Move on.
        log(f"{constraint} costs {constraint.multiplier} mm/mm but is non-negotiable")
        continue

    if constraint.priority == "high":
        # Electrical requirement. Only relax if multiplier is very large.
        if constraint.multiplier > HIGH_PRIORITY_THRESHOLD:  # e.g., 15.0
            relax(constraint, amount=small_step)
        continue

    if constraint.priority == "low":
        # Manufacturing preference. Relax more aggressively.
        relax(constraint, amount=moderate_step)
        continue
```

**Step 3: Check for inactive constraints.**
Remove or widen inactive constraints. They add solver overhead without affecting
the solution. This makes subsequent solves faster.

**Step 4: Check for improvements.**
Compare current HPWL to previous iteration. If improvement < epsilon (e.g.,
< 1mm), the agent has converged and should accept the design.

**Step 5: Re-submit.**
Write the modified spec and invoke the solver again.

### 3.5 Decision Thresholds

These thresholds are calibrated for typical PCB designs (50-100mm boards,
10-50 components). The agent should adjust based on board size and component
count.

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| INSIGNIFICANT_THRESHOLD | 0.5 mm/mm | Below this, the constraint's contribution to HPWL is negligible |
| HIGH_PRIORITY_THRESHOLD | 15.0 mm/mm | Only relax electrical constraints when the cost is severe |
| CONVERGENCE_EPSILON | 1.0 mm | Stop iterating when HPWL improvement is less than 1mm |
| MAX_ITERATIONS | 8 | Safety limit on agent loop iterations |
| RELAXATION_STEP_SMALL | 0.5mm or 10% | For high-priority constraints |
| RELAXATION_STEP_MODERATE | 1.0mm or 25% | For low-priority constraints |

---

## 4. Concrete Example: Full Agent Loop

### Design task: 5V-to-3.3V power supply module

A 40mm x 25mm board with:
- U1: LM1117 voltage regulator (SOT-223, 6.5mm x 3.5mm)
- C1: 10uF input capacitor (0805, must be within 3mm of U1 input pin)
- C2: 10uF output capacitor (0805, must be within 3mm of U1 output pin)
- C3: 100nF ceramic bypass (0402, must be within 2mm of U1)
- J1: 2-pin input header (must be on left edge)
- J2: 2-pin output header (must be on right edge)
- R1: 10K feedback resistor (0402, near U1)
- D1: protection diode (SOD-123, near J1)
- LED1: power indicator LED (0805)
- R2: LED current-limiting resistor (0402, near LED1)

### Iteration 0: Agent writes initial spec

```
board { width: 40mm, height: 25mm }

placement {
    minimize { wirelength }

    subject_to {
        clearance { all: 0.5mm, relaxable: true, min: 0.15mm, priority: low }

        near C1, U1 { max_distance: 3mm, relaxable: true, max: 8mm, priority: high }
        near C2, U1 { max_distance: 3mm, relaxable: true, max: 8mm, priority: high }
        near C3, U1 { max_distance: 2mm, relaxable: true, max: 5mm, priority: high }
        near R1, U1 { max_distance: 5mm, priority: medium }
        near D1, J1 { max_distance: 5mm, priority: medium }
        near R2, LED1 { max_distance: 3mm, priority: low }
    }

    place J1 { edge: left, inset: 1mm }
    place J2 { edge: right, inset: 1mm }

    J1 left_of U1 { gap: 3mm }
    U1 left_of J2 { gap: 3mm }
}
```

### Iteration 0 result: Infeasible

```json
{
  "status": "PartiallySolved",
  "objective": { "value_mm": 312.4 },
  "iterations": 250,
  "sensitivity": {
    "binding_constraints": [
      {
        "type": "NearConstraint",
        "entities": ["C3", "U1"],
        "max_distance_mm": 2.0,
        "multiplier": 18.7,
        "relaxable": true,
        "priority": "high"
      },
      {
        "type": "DirectionalOrdering",
        "constraint": "U1 left_of J2",
        "gap_mm": 3.0,
        "multiplier": 14.2,
        "relaxable": true,
        "priority": null
      },
      {
        "type": "ComponentClearance",
        "entities": ["U1", "C1"],
        "clearance_mm": 0.5,
        "multiplier": 11.3,
        "relaxable": true,
        "priority": "low"
      },
      {
        "type": "NearConstraint",
        "entities": ["C1", "U1"],
        "max_distance_mm": 3.0,
        "multiplier": 9.8,
        "relaxable": true,
        "priority": "high"
      },
      {
        "type": "BoardContainment",
        "entity": "U1",
        "edge": "right",
        "multiplier": 8.1
      }
    ],
    "constraint_summary": {
      "total": 31,
      "binding": 14,
      "inactive": 15,
      "violated": 2,
      "max_multiplier": 18.7
    }
  }
}
```

### Agent analysis (iteration 0 -> 1)

The agent reads the report and reasons:

1. **C3-U1 near constraint (lambda=18.7):** This is the highest multiplier and
   has priority "high". The 2mm distance is very tight -- C3 is a 0402 body
   (1mm x 0.5mm) and U1 is an SOT-223 (6.5mm x 3.5mm). With 0.5mm clearance,
   the center-to-center distance is already ~4mm. The 2mm max_distance may be
   infeasible given the component sizes. The agent relaxes to 4mm.

2. **U1-left_of-J2 ordering (lambda=14.2):** The 3mm gap between U1 and J2 is
   forcing U1 far from the right side of the board. The agent reduces the gap
   to 1mm -- traces can run between them.

3. **U1-C1 clearance (lambda=11.3):** The 0.5mm global clearance between the
   regulator and its input cap is unnecessarily large. The agent sets a
   pair-specific clearance of 0.2mm.

4. **Two constraints violated:** The solver could not fully satisfy all
   constraints. The relaxations above should make the system feasible.

### Iteration 1: Agent modifies spec

```
placement {
    minimize { wirelength }

    subject_to {
        clearance {
            all: 0.5mm, relaxable: true, min: 0.15mm, priority: low
            pair U1 C1: 0.2mm    // relaxed from 0.5mm (lambda=11.3)
            pair U1 C3: 0.2mm    // also tight pair
        }

        near C1, U1 { max_distance: 3mm, priority: high }
        near C2, U1 { max_distance: 3mm, priority: high }
        near C3, U1 { max_distance: 4mm, priority: high }  // relaxed from 2mm
        near R1, U1 { max_distance: 5mm, priority: medium }
        near D1, J1 { max_distance: 5mm, priority: medium }
        near R2, LED1 { max_distance: 3mm, priority: low }
    }

    place J1 { edge: left, inset: 1mm }
    place J2 { edge: right, inset: 1mm }

    J1 left_of U1 { gap: 2mm }      // reduced from 3mm
    U1 left_of J2 { gap: 1mm }      // reduced from 3mm (lambda=14.2)
}
```

### Iteration 1 result: Converged

```json
{
  "status": "Converged",
  "objective": { "value_mm": 187.6 },
  "iterations": 142,
  "sensitivity": {
    "binding_constraints": [
      {
        "type": "NearConstraint",
        "entities": ["C1", "U1"],
        "max_distance_mm": 3.0,
        "multiplier": 4.2,
        "priority": "high"
      },
      {
        "type": "BoardContainment",
        "entity": "U1",
        "edge": "top",
        "multiplier": 3.1
      },
      {
        "type": "ComponentClearance",
        "entities": ["C2", "R1"],
        "clearance_mm": 0.5,
        "multiplier": 2.8,
        "priority": "low"
      }
    ],
    "constraint_summary": {
      "total": 31,
      "binding": 7,
      "inactive": 24,
      "violated": 0,
      "max_multiplier": 4.2
    }
  }
}
```

### Agent analysis (iteration 1 -> 2)

1. HPWL dropped from 312.4mm to 187.6mm -- a 40% improvement. Good.
2. No violations. The system is feasible.
3. Max multiplier is 4.2 (C1-U1 near), which is below
   HIGH_PRIORITY_THRESHOLD (15.0). This is a high-priority electrical
   constraint. The agent keeps it.
4. All other multipliers are below 5.0. The layout is not severely
   constrained anywhere.
5. Improvement is 124.8mm, well above CONVERGENCE_EPSILON (1mm), so the agent
   could try one more iteration.

The agent decides to relax the C2-R1 clearance since it is the only low-priority
binding constraint:

### Iteration 2: Minor refinement

```
clearance {
    all: 0.5mm, relaxable: true, min: 0.15mm, priority: low
    pair U1 C1: 0.2mm
    pair U1 C3: 0.2mm
    pair C2 R1: 0.25mm    // new: relaxed from 0.5mm
}
```

### Iteration 2 result: Converged

```json
{
  "status": "Converged",
  "objective": { "value_mm": 183.1 },
  "sensitivity": {
    "constraint_summary": {
      "binding": 5,
      "violated": 0,
      "max_multiplier": 3.8
    }
  }
}
```

### Agent accepts the design

HPWL improved from 187.6 to 183.1 (delta = 4.5mm). Below a useful threshold for
another iteration. Max multiplier is 3.8. No violations. The agent runs DRC:

```
DRC: 0 violations. All clearances met. All nets routable.
```

The agent writes the final spec and commits: "Power supply placement converged
in 3 iterations. Final HPWL: 183.1mm, 0 DRC violations. Relaxed C3-U1 proximity
from 2mm to 4mm (infeasible at 2mm given component sizes) and reduced inter-stage
gap from 3mm to 1mm. All electrical proximity requirements preserved."

---

## 5. What This Enables That Nothing Else Can

### 5.1 The Current State of EDA + AI

Today, when an AI agent generates a PCB layout spec, the workflow is:

1. Agent writes constraints based on design rules and engineering judgment.
2. Solver runs and either succeeds or fails.
3. If it fails, the agent gets a binary signal: "infeasible" or "converged with
   violations." No information about *why* it failed or *which* constraint to
   relax.
4. The agent guesses. It might randomly relax constraints, or give up, or
   reduce all clearances uniformly.

This is "write constraints and hope." The agent has no quantitative signal to
guide its decisions. Every iteration is a shot in the dark.

Commercial EDA tools (Cadence Allegro, Altium Designer, KiCad) do not expose
Lagrange multipliers or sensitivity data at all. Their solvers are black boxes
that report pass/fail. Even advanced autorouters (SPECCTRA, Topological Router)
report only DRC violations and completion rates, not the *cost structure* of the
constraint system.

### 5.2 What Multipliers Change

With multiplier feedback, the agent becomes a *constraint negotiator*:

1. **Informed relaxation.** The agent knows *exactly* which constraint is the
   bottleneck and *exactly* how much HPWL it costs. It does not guess.

2. **Minimum-regret changes.** The agent relaxes the cheapest constraint first
   (highest lambda, lowest priority). This produces the maximum HPWL improvement
   per unit of constraint relaxation.

3. **Convergence in few iterations.** The power supply example converged in 3
   iterations. Without multipliers, the agent would need 10-20 iterations of
   trial-and-error, if it converged at all.

4. **Explanation.** The agent can explain *why* it made each change: "Relaxed
   C3-U1 proximity from 2mm to 4mm because the multiplier was 18.7, indicating
   this constraint was responsible for 60% of the total wirelength pressure."
   This is auditable. A human reviewer can verify the reasoning.

5. **Discovery of infeasibility root causes.** When the solver reports
   "Infeasible," the high-multiplier constraints identify the contradictory
   subset. The agent can report: "The combination of 2mm C3-U1 proximity, 0.5mm
   global clearance, and 3mm U1-J2 gap is geometrically impossible on a 40mm
   board." No existing tool provides this diagnostic.

### 5.3 Designs That Become Possible

**Dense mixed-signal boards.** These have tight analog proximity requirements
(reference voltage routing, guard rings) alongside digital clearance rules.
Without sensitivity data, an agent must choose conservative constraints that
waste board space. With multipliers, the agent can push each constraint to its
actual limit, achieving higher density without violating any electrical rule.

**Iterative board outline negotiation.** Often the board shape is not fixed at
design start -- it must fit in an enclosure, but there is some flexibility in
dimensions. Multipliers on BoardContainment tell the agent exactly how much
wirelength is saved per mm of board expansion in each direction, enabling
cost-benefit analysis against mechanical constraints.

**Automatic design rule trade-offs.** A high-multiplier clearance constraint
between two components might prompt the agent to change the footprint to a
smaller package (0402 instead of 0805), reducing the clearance pressure. This
cross-domain reasoning (layout <-> BOM) is only possible when the agent
understands the quantitative cost of each constraint.

**Multi-board partitioning.** When a design does not fit on one board, the
agent can use multiplier data to decide where to split: components with high
clearance multipliers between them are candidates for splitting onto separate
boards, because their proximity is the most costly constraint.

### 5.4 The Agent's New Role

Without multipliers: the agent is a **spec writer**. It translates requirements
into constraints and submits them to an oracle. Success depends on getting the
constraints right on the first try.

With multipliers: the agent is a **design negotiator**. It engages in a
structured dialogue with the physics of the layout. Each iteration, the solver
says "here is what your constraints cost" and the agent says "here is what I am
willing to trade." The agent converges on a Pareto-optimal design that balances
all requirements -- not because it was lucky, but because it followed the
gradient of the dual variables to the best achievable trade-off.

This is the same role a senior PCB designer plays when hand-placing components:
they try a layout, see where it is tight, loosen the tight spots, and iterate
until the board is good. The difference is that the multipliers make this
process *quantitative* and *automated*, enabling the agent to handle boards with
hundreds of components and thousands of constraints -- far beyond what a human
can reason about simultaneously.

---

## Appendix A: Implementation Roadmap

### Phase 1: Multiplier Extraction (Solverang)

1. Implement `MultiplierStore` and `MultiplierId` (designed in 03_pipeline_design.md).
2. After LM solve, compute pseudo-multipliers via `lambda = -(J^T J)^{-1} J^T r`.
3. Populate `MultiplierStore` from the solved system.
4. Expose `ConstraintSystem::multiplier(constraint_id) -> Option<&[f64]>`.

### Phase 2: Sensitivity Reporter (autopcb-placement)

1. After `solve_placement()`, extract multipliers for each `UserConstraint`.
2. Map `ConstraintId` back to `UserConstraint` identity (designators, type).
3. Generate `SensitivityReport` JSON.
4. Extend `PlacementResult` to include the report path.

### Phase 3: Spec Language Extensions (altium-format-spec)

1. Add `minimize { ... }` and `subject_to { ... }` to the placement grammar.
2. Add `relaxable`, `priority`, `min`, `max` fields to constraint declarations.
3. Compile these to the existing `PlacementConfig` + `UserConstraint` types,
   passing relaxation metadata through to the sensitivity reporter.

### Phase 4: Agent Loop Harness

1. Build a CLI command: `autopcb solve-loop --spec input.spec --max-iters 8`.
2. The harness invokes compile -> place -> report in a loop.
3. Between iterations, the harness calls Claude Code with the sensitivity report
   and the current spec, asking it to produce a modified spec.
4. The harness validates the modified spec (parses, compiles) before re-solving.

### Phase 5: True Optimization (Solverang ALM/SQP)

1. Implement ALM solver with LM inner loop (Phase 1 of optimization extension).
2. Replace soft-constraint HPWL with a proper `Objective` in the placer.
3. Multipliers become true Lagrange multipliers rather than pseudo-multipliers.
4. Sensitivity data becomes exact rather than approximate.

---

## Appendix B: Multiplier Recovery from LM Solver

The current autopcb placer uses Solverang's LM (Levenberg-Marquardt) solver,
which minimizes `||F(x)||^2` where `F(x)` is the stacked residual vector from
all constraints (hard and soft).

At the converged solution x*, the LM solver has computed the Jacobian J = F'(x*)
and the residual r = F(x*). The pseudo-multipliers are:

```
lambda = -(J^T J + mu*I)^{-1} J^T r
```

where mu is the LM damping parameter. As mu -> 0 (undamped, at convergence),
this approaches the true least-squares multiplier `-(J^T J)^{-1} J^T r`.

For inequality constraints using the slack-variable trick `g(x) - s^2 = 0`, the
multiplier on the original inequality is:

```
lambda_ineq = lambda_equality / (2 * s)
```

where s is the slack variable value at the solution. When s = 0 (active
constraint), the multiplier is formally infinite but the equality multiplier
itself is the relevant quantity. When s > 0 (inactive), lambda_ineq approaches
zero, consistent with complementarity.

In practice, the solver reports lambda_equality directly, and the sensitivity
reporter classifies constraints as binding (|s| < epsilon) or inactive
(|s| >= epsilon). For binding constraints, lambda_equality is the relevant
sensitivity measure. For inactive constraints, the multiplier is reported as
zero regardless of the computed value.
