Differential Dataflow Decision Critique: Final Verdict

  VERDICT: REVISE

  The core claim that "differential dataflow is used for incremental constraint computation" is false.
  The DD code exists but is unused, and the actual implementation uses batch recomputation.

  ---
  VERIFICATION SUMMARY

  FAILED (6)
  ID: C1
  Item: DD is used for incremental computation
  Finding: SolverDataflow is only used in its own unit tests, never in production
  ────────────────────────────────────────
  ID: C2
  Item: Incremental matches batch
  Finding: No comparison tests exist
  ────────────────────────────────────────
  ID: C6
  Item: Dataflow runs in separate thread
  Finding: SolverWorker is synchronous; DD thread code is never called
  ────────────────────────────────────────
  ID: A1
  Item: DD provides incremental computation
  Finding: System uses batch BFS recomputation on every change
  ────────────────────────────────────────
  ID: A2
  Item: Component algorithm converges
  Finding: DD only propagates 1 hop - broken for graph diameter > 2
  ────────────────────────────────────────
  ID: A3
  Item: Remove operations work
  Finding: RemoveVariable/RemoveConstraint are documented no-ops
  VERIFIED (4)
  ┌─────┬─────────────────────────────────┬───────────────────────────────────────────────────────┐
  │ ID  │              Item               │                        Finding                        │
  ├─────┼─────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ C3  │ SolverWorker handles add/remove │ Works correctly via batch BFS                         │
  ├─────┼─────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ C4  │ Component merge/split works     │ Tests verify correct behavior in SolverWorker         │
  ├─────┼─────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ C5  │ DOF calculation correct         │ Tests verify var_dof - constraint_equations formula   │
  ├─────┼─────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ J1  │ BFS vs DD is intentional        │ Comments say "For Phase 2: use a simplified approach" │
  └─────┴─────────────────────────────────┴───────────────────────────────────────────────────────┘
  UNCERTAIN (0)

  All verifiable items were conclusively determined.

  ---
  CHALLENGE ASSESSMENT

  Strongest Challenge: The "differential dataflow implementation" is a misnomer - the system uses O(n)
  batch recomputation on every edit. The DD code that exists has critical bugs (1-hop propagation, no-op
  deletions) and is completely dead code.

  Response: This is factually accurate. The decision to call this a "differential dataflow
  implementation" is misleading. The actual implementation (SolverWorker) is correct and likely adequate
  for typical use cases, but the module-level documentation creates false expectations.

  ---
  RECOMMENDATIONS

  Option A: Remove DD Code (Recommended for Phase 2)

  1. Delete dataflow.rs - It's ~390 lines of unused, broken code
  2. Update module docs in lib.rs - Remove claims about DD, describe actual BFS implementation
  3. Rename if needed - The module name ecad_solver is fine, but internal docs should be accurate
  4. Add performance tests - Verify BFS is fast enough for target drawing sizes (100, 1000, 10000
  entities)

  Option B: Fix and Enable DD Code (If Incremental Is Needed)

  If you determine batch BFS is too slow for large drawings:

  1. Fix component algorithm - Add iteration until convergence:
  // Instead of single-hop, use iterate::Variable for label propagation
  let components = self_labels.iterate(|inner| {
      // Propagate minimum labels until fixpoint
  });
  2. Fix RemoveVariable/RemoveConstraint - Track state internally:
  // Maintain HashMap<VariableId, Variable> to enable retractions
  let old_var = state.remove(&vid);
  var_input.update((vid, old_var), -1); // Retract
  3. Wire SolverDataflow into production - Replace or complement SolverWorker
  4. Add equivalence tests - Verify DD output matches BFS output for same inputs
  5. Benchmark - Prove DD is faster than batch for realistic workloads

  Documentation Fix (Either Option)

  Update crates/ecad_solver/src/lib.rs lines 7-14:
  //! # Architecture
  //!
  //! The solver uses batch BFS recomputation for:
  //! - Connected component detection (independent constraint subsystems)
  //! - DOF (degrees of freedom) status per component
  //!
  //! Note: Differential-dataflow based incremental computation is planned for
  //! a future phase but not currently implemented.

  ---
  Summary
  ┌─────────────────────┬──────────────────────────┬────────────────────────┐
  │       Aspect        │      Current State       │       Assessment       │
  ├─────────────────────┼──────────────────────────┼────────────────────────┤
  │ SolverWorker (BFS)  │ Used, tested, correct    │ ✅ Production-ready    │
  ├─────────────────────┼──────────────────────────┼────────────────────────┤
  │ SolverDataflow (DD) │ Unused, untested, broken │ ❌ Dead code with bugs │
  ├─────────────────────┼──────────────────────────┼────────────────────────┤
  │ Documentation       │ Claims DD is used        │ ❌ Misleading          │
  ├─────────────────────┼──────────────────────────┼────────────────────────┤
  │ Performance         │ Batch O(n) on every edit │ ⚠️ Unknown if adequate │
  └─────────────────────┴──────────────────────────┴────────────────────────┘
  The SolverWorker implementation is actually solid - it just isn't what the documentation claims. The
  path forward is either removing the DD code and fixing the docs, or properly implementing DD if
  incremental computation is actually needed.
