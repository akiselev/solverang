# Review: 00_synthesis.md

Reviewed against sub-documents 01–07 and the actual codebase.
All findings have been applied as direct edits to `00_synthesis.md` via `> **REVIEW NOTE:**` blockquotes.
This file summarizes what was changed and why, for audit purposes.

---

## Cross-Document Contradictions

- **[00] vs [03]: Pipeline stage order** — 00 had `Decompose → Classify → ...`; 03 specifies `Classify → Decompose → ...`. 03 is architecturally correct (Classify must precede Decompose to enable objective-aware clustering). Fixed in 00.

- **[00] vs [04]: TDD stage content** — 00's 12-stage table diverged from doc 04 at every stage from 4 onward. Specific conflicts: Stage 4 (Rosenbrock vs simple quadratic), Stage 5 (L-BFGS vs Newton), Stage 6 (ALM vs penalty method), Stage 8 (L-BFGS-B vs variable bounds). Doc 04 is the authoritative source. Fixed in 00.

- **[00/02] vs [05]: Macro-generated Hessians** — Phase 3 of the implementation plan proposes generating Hessians via the macro. Doc 05's single most important recommendation is "Do not extend the macro to generate Hessians." This is a direct conflict over the Phase 3 strategy. Flagged as REVIEW NOTE in 00; Phase 3 items made conditional pending empirical validation.

- **[00] vs [07]: Phase 1 algorithm** — 00 says ALM first; 07's phased rollout puts BFGS first (KKT approach) and defers ALM to Phase 4. Both are defensible depending on priority. Flagged as REVIEW NOTE in executive summary.

- **[00] vs [03]: `ConstraintSystem` API** — 00 omitted `clear_objective()`, `set_opt_config()`, `multiplier()`, and `multipliers()` methods that doc 03 defines. Fixed in 00.

---

## Naming Inconsistencies

- **System-level inequality trait**: named `SystemInequalityConstraint` in 00, `InequalityFn` in 01. Both differ from the existing `InequalityConstraint` in `constraints/inequality.rs`. Fixed to use `InequalityFn` throughout 00 (consistent with 01, avoids collision with existing name).

- **`ConstraintHessian` method**: named `hessian_entries(&self, store, eq_index)` in 00; named `residual_hessian(&self, equation_row, store)` in 01 (different name, different parameter order). Fixed to use 01's signature in 00.

- **`ObjectiveId` / `MultiplierId` field types**: 00 used bare `u32` for `generation`; 01 used the project-wide `Generation` type alias. Fixed to use `Generation` in 00.

- **`OptimizationProblem`**: defined as a TRAIT in 01, as a STRUCT in 04. The synthesis does not use this name, so no collision in 00, but implementers will encounter this when reading 01 vs 04. Not edited (requires a design decision outside the synthesis's scope), but the REVIEW NOTE on trait divergence in Section 1 alerts readers.

---

## Missing from Synthesis

- **05's two most critical recommendations** were absent: (1) do not macro-generate Hessians; (2) replace slack variables with log-barrier as default for optimization. Both are now present — (1) as REVIEW NOTE in Hessian Strategy, (2) already in the Inequality Handling section (was present).

- **Three missing risks from 05**: saddle point convergence (HIGH), unbounded problems (MEDIUM), and degenerate constraint qualification / LICQ violation (HIGH). All three added to the Key Risks table in Section 7.

- **Runtime AD fallback**: 05 states this is "not optional" for objectives with control flow. The synthesis only mentioned a `ManualObjective` escape hatch, which is weaker. Updated mitigation in Section 7.

- **07 use case tiering**: doc 07 explicitly ranks use cases as Tier 1 (parametric optimization, fit, soft constraints), Tier 2 (mechanism synthesis, assembly clearance, tolerance), Tier 3 (shape, topology, path planning). Not added to synthesis body (would require significant restructuring), but the parametric optimization reference in Section 8 now cites doc 07's "HIGHEST" tier designation.

- **Rust ecosystem candidates**: Clarabel.rs and OSQP are mentioned in Phase 3 but not explained. Section 8 now includes brief descriptions from doc 06. argmin-rs reference added as potential line-search/trust-region source.

- **Enzyme / `#[autodiff]`**: doc 06 covers the active Rust `#[autodiff]` upstreaming effort (2024). Added to Section 8 as a future dual-AD pathway.

- **`LeastSquaresObjective` adapter used wrong signatures**: the code in Section 5 used `ParamStore`-based signatures (`store: &ParamStore`) but the `Problem` trait uses `&[f64]`. Fixed to use array-based API with correct `gradient()` implementation (`J^T * r`).

---

## Unsupported Claims in Synthesis

- **"Constraint decomposition (rare in optimization; usually manual)"** — doc 06 and doc 07 confirm Solverang's decomposition is a differentiator relative to CAD tools, but neither characterizes it as "rare in optimization" in general. Multi-block methods like ALADIN and Ceres's residual-block decomposition show decomposition is not universally absent. Reworded to make a more defensible, specific claim.

- **Property test code used `store: &ParamStore`** in `check_gradient_fd` even though the `Objective` in the `OptimizationBuilder` path uses `x: &[f64]`. The code example was internally inconsistent with the trait it demonstrated. Fixed to use the array-based API with relative-error tolerance (matching doc 04's pattern).

---

## Codebase Inconsistencies

All of these were already correctly described in 00; they are noted here for completeness:

- **`InequalityConstraint` in `constraints/inequality.rs`**: uses `g(x) >= 0` convention with `evaluate(&self, x: &[f64]) -> f64`. Synthesis correctly identifies this as the existing Problem-level trait that the new system-level trait must not shadow.

- **`Constraint` trait in `constraint/mod.rs`**: exactly as described. Has `id()`, `name()`, `entity_ids()`, `param_ids()`, `equation_count()`, `residuals()`, `jacobian()`, `weight()`, `is_soft()`. No changes needed.

- **`Generation = u32` type alias in `id.rs`**: all existing ID types use `pub(crate) generation: Generation`. The new `ObjectiveId` and `MultiplierId` structs must follow the same pattern.

- **`ParamId::raw_index()`** is `pub(crate)` in the actual code. The `LeastSquaresObjective` adapter in 00 referenced `pid.raw_index()` in error handling — this is crate-internal. Removed from the adapter code.

---

## Overall Coherence Assessment

The synthesis accurately captures the high-level architecture and the rationale for the two-level design. Its major weakness is treating the sub-documents as a consistent whole when they contain genuine design forks: the `Objective` and inequality trait APIs differ materially between the mathematical architecture document (01), the user API document (04), and the synthesis itself. An implementer reading only the synthesis would build something incompatible with the detailed specs.

The second major weakness is that the synthesis's implementation plan conflicts with the strongest recommendations from the risks analysis document (05). The Phase 3 macro-Hessian plan is presented with the same confidence as the uncontroversial Phase 1-2 plans, despite 05's explicit warning that this approach hits a hard wall around N=30-50 variables.

After the edits applied to `00_synthesis.md`, the synthesis now: (1) names the trait conflicts explicitly with REVIEW NOTEs, (2) presents the Hessian strategy conflict as a decision requiring empirical validation before committing, (3) corrects the pipeline stage order to match doc 03, (4) corrects the TDD table to match doc 04, (5) adds the three missing risk categories from doc 05, and (6) adds the Rust ecosystem context from doc 06 that was entirely absent.
