//! Differential-oracle lane: `solverang` vs the Siemens D-Cubed 2D DCM.
//!
//! # What this is
//!
//! A *solve-and-diff* oracle lane. Each test poses a canonical 2D constraint
//! scenario in `solverang`, computes its **semantic classification**
//! (consistency, degrees-of-freedom class, redundancy/conflict), and asserts it
//! matches the classification the licensed **D-Cubed 2D DCM** (`dcu2d78.dll`)
//! reports for the *same* scenario. The D-Cubed verdicts were obtained by
//! running the reference solver out-of-tree under wine and transcribing only the
//! discrete semantic observables — never coordinates.
//!
//! # Clean-room boundary (read before editing)
//!
//! * `solverang` links, loads, and embeds **nothing** from D-Cubed. This file
//!   contains no proprietary code, no lifted coordinates, and no copied data.
//! * The only thing crossing the boundary is a **classification** — well- /
//!   under- / over-defined, a DOF count, a satisfied/consistent verdict. Those
//!   are facts about correct constraint solving, independently re-derivable from
//!   the geometry by hand (each scenario documents the derivation). Storing an
//!   *expected classification* as a test assertion is fine; storing a solved
//!   coordinate from the DLL would not be, and none appear here.
//! * The reference verdicts live in the `DCUBED_*` comments next to each test,
//!   marked "cross-checked against the D-Cubed reference via differential
//!   testing".
//!
//! # How the reference verdicts were generated
//!
//! In the private RE repo `solverang-re` (which is the *only* place the DLL is
//! ever executed):
//!
//! ```text
//! make -C harness diff_2d.exe            # cross-compiled PE probe, -O0
//! bash harness/run-diff.sh               # runs each scenario in its own wine process
//! ```
//!
//! `harness/diff_2d.c` builds each scenario through the public `DCM_*` C API,
//! grounds the reference geometry with `DCM_fix` (so per-geometry DOF is
//! world-relative and comparable to `solverang`'s grounded `analyze_dof`), then
//! reads the semantic observables:
//!
//! | D-Cubed observable            | meaning                                   | solverang analogue                    |
//! |-------------------------------|-------------------------------------------|---------------------------------------|
//! | `DCM_degree_of_singularity`   | dos_status: 0 SUCCEEDED / 1 NOT_SATISFIED | `solve().status` Solved vs failure    |
//! | `DCM_underdefined_dof`        | per-geometry residual DOF count           | `analyze_dof().entities[..].dof`      |
//! | `DCM_underdefined_status`     | 0 well-defined / 0x10 position-underdef   | sign of `analyze_dof().total_dof`     |
//! | `dof_result` = UNKNOWN(1)     | geometry is over-constrained              | `analyze_redundancy()` rank deficiency|
//!
//! # Grounding convention
//!
//! Every scenario grounds a reference (the first point / an edge) in BOTH
//! engines, so DOF is measured modulo rigid-body motion — the "grounded CAD
//! sketch" model. In `solverang` grounding is `add_fixed_point`; in D-Cubed it
//! is `DCM_fix`. Point-only scenarios (S1–S5) yield integer DOF parity;
//! line/circle scenarios (S6–S8) compare the **consistency** verdict only,
//! because the two engines model a line differently (D-Cubed: infinite
//! point+direction line; solverang: a finite segment sharing endpoint params),
//! so raw DOF counts are representation-dependent and not a like-for-like
//! comparison. This is noted per test.

use solverang::sketch2d::Sketch2DBuilder;
use solverang::system::SystemStatus;

/// A scenario's `solverang` semantic verdict, reduced to comparables.
#[derive(Debug)]
struct Verdict {
    /// Raw `solve().status == Solved`. NOTE: this alone is *not* a satisfaction
    /// certificate — an over-determined cluster reports `Solved` when the
    /// least-squares solver converges, even to a non-zero minimum (see the S5
    /// divergence test and `TODO.md`). Consistency is derived by
    /// [`Verdict::consistent`] from conflict detection + the residual instead.
    solve_solved: bool,
    /// Largest `|residual|` over all constraints after `solve()`.
    max_residual: f64,
    /// Rank-based total DOF: `free_params - rank(Jacobian)` (`analyze_dof`).
    total_dof: i32,
    /// DOF of the tracked free entity, if one was requested.
    entity_dof: Option<usize>,
    /// Redundant (over-constrained, consistent) constraint count.
    redundant: usize,
    /// Conflicting (over-constrained, inconsistent) constraint count.
    conflicts: usize,
    /// Rank deficiency `equation_count - jacobian_rank` (>0 ⇒ over-constrained).
    rank_deficiency: usize,
}

impl Verdict {
    /// The consistency **classification** solverang reports for the lane: the
    /// system is inconsistent iff a conflict group is detected OR the solved
    /// state still violates a constraint. This is the honest analogue of
    /// D-Cubed's `dos_status` (SUCCEEDED vs NOT_SATISFIED) and matches it on
    /// every scenario. (It deliberately does not rely on `solve().status`,
    /// which over-reports `Solved`; that gap is pinned by
    /// `s5_known_divergence_*` and recorded in `TODO.md`.)
    fn consistent(&self) -> bool {
        self.conflicts == 0 && self.max_residual <= 1e-6
    }
    fn well_defined(&self) -> bool {
        self.total_dof == 0
    }
    fn under_defined(&self) -> bool {
        self.total_dof > 0
    }
    fn over_constrained(&self) -> bool {
        self.rank_deficiency > 0
    }
}

/// Build the scenario's verdict: analyze the freshly-posed (satisfying) system,
/// then solve it. `track` is an optional free entity whose per-entity DOF we
/// compare against D-Cubed's per-geometry `underdefined_dof`.
fn verdict(mut system: solverang::ConstraintSystem, track: Option<solverang::EntityId>) -> Verdict {
    let dof = system.analyze_dof();
    let redundancy = system.analyze_redundancy();

    let entity_dof = track.and_then(|eid| {
        dof.entities
            .iter()
            .find(|e| e.entity_id == eid)
            .map(|e| e.dof)
    });

    let total_dof = dof.total_dof;
    let redundant = redundancy.redundant.len();
    let conflicts = redundancy.conflicts.len();
    let rank_deficiency = redundancy.rank_deficiency();

    let result = system.solve();
    let solve_solved = matches!(result.status, SystemStatus::Solved);
    let max_residual = system
        .compute_residuals()
        .iter()
        .fold(0.0_f64, |m, r| m.max(r.abs()));

    let v = Verdict {
        solve_solved,
        max_residual,
        total_dof,
        entity_dof,
        redundant,
        conflicts,
        rank_deficiency,
    };
    eprintln!(
        "  solverang verdict: {v:?} (solve status {:?})",
        result.status
    );
    if v.solve_solved && !v.consistent() {
        // The S5 divergence surfacing in the lane: solve() accepted a system the
        // classification (conflict + residual) marks inconsistent.
        eprintln!("  NOTE: solve()==Solved but the classification is INCONSISTENT (see s5_known_divergence_*)");
    }
    v
}

// ---------------------------------------------------------------------------
// S1 — two points + coincident, one grounded.
// Geometry: p0 fixed at (0,0); p1 free, posed AT (0,0). Coincident(p0,p1).
// By hand: p1 has 2 free params; coincident removes both ⇒ 0 residual DOF,
// consistent, exactly constrained.
//
// DCUBED_S1 (cross-checked via differential testing, harness/diff_2d.c S1):
//   dos=0 SUCCEEDED · underdef_status=0x0 (well-defined) · dof_result=SUCCESS dof_count=0
//   ⇒ well-defined, 0 DOF, consistent.
// ---------------------------------------------------------------------------
#[test]
fn s1_coincident_grounded_is_well_defined() {
    eprintln!("S1 two points + coincident (p0 grounded)");
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(0.0, 0.0);
    b.constrain_coincident(p0, p1).unwrap();

    let v = verdict(b.build(), Some(p1.entity_id()));

    assert!(v.consistent(), "S1 must be consistent (D-Cubed: SUCCEEDED)");
    assert!(
        v.well_defined(),
        "S1 must be well-defined, 0 DOF (D-Cubed: dof_count=0); got total_dof={}",
        v.total_dof
    );
    assert_eq!(
        v.entity_dof,
        Some(0),
        "S1 p1 per-entity DOF (D-Cubed underdefined_dof=0)"
    );
    assert!(!v.over_constrained(), "S1 has no redundancy");
    assert_eq!(v.conflicts, 0);
}

// ---------------------------------------------------------------------------
// S2 — two points + distance, one grounded.
// Geometry: p0 fixed (0,0); p1 free at (10,0). distance(p0,p1)=10.
// By hand: p1 (2 free) minus 1 distance equation ⇒ 1 residual DOF — p1 may
// rotate about p0 on the radius-10 circle. Consistent, under-defined.
//
// DCUBED_S2 (cross-checked, diff_2d.c S2):
//   dos=0 SUCCEEDED · underdef_status=0x10 POSITION_UNDERDEFINED · dof_result=SUCCESS dof_count=1
//   ⇒ under-defined by 1 DOF (rotation), consistent.
// ---------------------------------------------------------------------------
#[test]
fn s2_distance_grounded_is_under_defined_by_rotation() {
    eprintln!("S2 two points + distance (p0 grounded)");
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    b.constrain_distance(p0, p1, 10.0).unwrap();

    let v = verdict(b.build(), Some(p1.entity_id()));

    assert!(v.consistent(), "S2 must be consistent (D-Cubed: SUCCEEDED)");
    assert!(
        v.under_defined(),
        "S2 must be under-defined (D-Cubed: POSITION_UNDERDEFINED); got total_dof={}",
        v.total_dof
    );
    assert_eq!(
        v.total_dof, 1,
        "S2 residual DOF = 1 rotation (D-Cubed dof_count=1)"
    );
    assert_eq!(
        v.entity_dof,
        Some(1),
        "S2 p1 per-entity DOF (D-Cubed underdefined_dof=1)"
    );
    assert!(!v.over_constrained(), "S2 has no redundancy");
}

// ---------------------------------------------------------------------------
// S3 — rigid triangle, grounded edge.
// Geometry: p0 fixed (0,0), p1 fixed (10,0); p2 free at (5,6). Two distances
// d(p0,p2)=d(p1,p2)=sqrt(61). By hand: p2 (2 free) minus 2 independent distance
// equations ⇒ 0 residual DOF. The triangle is internally rigid and, with the
// base edge grounded, fully defined. Consistent.
//
// DCUBED_S3 (cross-checked, diff_2d.c S3):
//   dos=0 SUCCEEDED · underdef_status=0x0 (well-defined) · dof_result=SUCCESS dof_count=0
//   ⇒ well-defined (rigid), 0 DOF, consistent.
// ---------------------------------------------------------------------------
#[test]
fn s3_rigid_triangle_grounded_edge_is_well_defined() {
    eprintln!("S3 rigid triangle (edge p0-p1 grounded)");
    let side = 61.0_f64.sqrt();
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(10.0, 0.0);
    let p2 = b.add_point(5.0, 6.0);
    b.constrain_distance(p0, p2, side).unwrap();
    b.constrain_distance(p1, p2, side).unwrap();

    let v = verdict(b.build(), Some(p2.entity_id()));

    assert!(v.consistent(), "S3 must be consistent (D-Cubed: SUCCEEDED)");
    assert!(
        v.well_defined(),
        "S3 must be well-defined, 0 DOF (D-Cubed: dof_count=0); got total_dof={}",
        v.total_dof
    );
    assert_eq!(
        v.entity_dof,
        Some(0),
        "S3 p2 per-entity DOF (D-Cubed underdefined_dof=0)"
    );
    assert!(!v.over_constrained(), "S3 has no redundancy");
}

// ---------------------------------------------------------------------------
// S4 — redundant distance (over-constrained, consistent).
// Geometry: p0 fixed (0,0); p1 free at (10,0). TWO identical distance(p0,p1)=10.
// By hand: the second distance duplicates the first — a redundant equation. The
// system is consistent (both satisfiable simultaneously) but carries one
// rank-deficient direction.
//
// DCUBED_S4 (cross-checked, diff_2d.c S4):
//   dos=0 SUCCEEDED (solver absorbs the redundancy) · dof_result=UNKNOWN(1)
//   ⇒ over-constrained but consistent (redundant).
// solverang additionally names the redundant constraint (a strength: it reports
// rank deficiency where D-Cubed only signals UNKNOWN).
// ---------------------------------------------------------------------------
#[test]
fn s4_redundant_distance_is_over_constrained_but_consistent() {
    eprintln!("S4 redundant distance (p0 grounded)");
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    b.constrain_distance(p0, p1, 10.0).unwrap();
    b.constrain_distance(p0, p1, 10.0).unwrap();

    let v = verdict(b.build(), Some(p1.entity_id()));

    assert!(
        v.consistent(),
        "S4 must stay consistent — redundancy is not conflict (D-Cubed: SUCCEEDED)"
    );
    assert!(
        v.over_constrained(),
        "S4 must show rank deficiency (D-Cubed: dof_result=UNKNOWN); rank_deficiency={}",
        v.rank_deficiency
    );
    assert!(
        v.redundant >= 1,
        "S4 must flag >=1 redundant constraint; got {}",
        v.redundant
    );
    assert_eq!(v.conflicts, 0, "S4 is redundant, not conflicting");
}

// ---------------------------------------------------------------------------
// S5 — conflicting distance (over-constrained, inconsistent).
// Geometry: p0 fixed (0,0); p1 free at (10,0). distance(p0,p1)=10 AND =7.
// By hand: no position of p1 satisfies both radii ⇒ unsatisfiable.
//
// DCUBED_S5 (cross-checked, diff_2d.c S5):
//   dos=1 NOT_SATISFIED · dof_result=UNKNOWN(1)
//   ⇒ over-constrained and inconsistent (conflict).
// ---------------------------------------------------------------------------
#[test]
fn s5_conflict_distance_is_inconsistent() {
    eprintln!("S5 conflicting distance (p0 grounded)");
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    b.constrain_distance(p0, p1, 10.0).unwrap();
    b.constrain_distance(p0, p1, 7.0).unwrap();

    let v = verdict(b.build(), Some(p1.entity_id()));

    assert!(
        !v.consistent(),
        "S5 must be inconsistent (D-Cubed: NOT_SATISFIED)"
    );
    assert!(
        v.conflicts >= 1,
        "S5 must flag a conflicting constraint group; got {}",
        v.conflicts
    );
    assert!(
        v.over_constrained(),
        "S5 must show rank deficiency (D-Cubed: dof_result=UNKNOWN)"
    );
    // The residual left after solving is large (~1.5): the two radii cannot both
    // be met. This is what makes `consistent()` false.
    assert!(
        v.max_residual > 1.0,
        "S5 solved state must still violate a constraint; max_residual={}",
        v.max_residual
    );
}

// ---------------------------------------------------------------------------
// S5-divergence — a REAL FINDING from the differential lane, pinned here.
//
// D-Cubed reports the conflicting system as NOT_SATISFIED. `solverang` *detects*
// the conflict correctly via `analyze_redundancy().conflicts`, and the honest
// consistency classification (see `Verdict::consistent`) agrees with D-Cubed.
// HOWEVER, `solve().status` alone diverges: it returns `Solved` because the
// over-determined cluster's least-squares solver converges (to a NON-zero
// minimum). `certify_final_residuals` deliberately exempts over-determined
// clusters that report a matching least-squares residual, so an unsatisfiable
// system is not surfaced by `solve().status`.
//
// This test documents (does not "fix") the divergence, so it is `#[ignore]`d and
// run explicitly with `--ignored`. The gap is recorded in `solverang/TODO.md`:
// `solve()` should mark a cluster that converged to a non-zero least-squares
// residual as PartiallySolved / DiagnosticFailure, not Solved.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "known divergence: solve().status is Solved for an unsatisfiable over-determined cluster where D-Cubed reports NOT_SATISFIED; solverang surfaces the conflict via analyze_redundancy() instead. See TODO.md."]
fn s5_known_divergence_solve_status_reports_solved_on_conflict() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    b.constrain_distance(p0, p1, 10.0).unwrap();
    b.constrain_distance(p0, p1, 7.0).unwrap();
    let mut system = b.build();

    let result = system.solve();
    let max_residual = system
        .compute_residuals()
        .iter()
        .fold(0.0_f64, |m, r| m.max(r.abs()));

    // The divergence, asserted so it fails loudly if solverang's behavior changes:
    assert!(
        matches!(result.status, SystemStatus::Solved),
        "divergence no longer holds: solve() status is now {:?} (D-Cubed: NOT_SATISFIED) — update TODO.md",
        result.status
    );
    assert!(
        max_residual > 1.0,
        "solve() reported Solved yet the residual is {max_residual} — the unsatisfiable system was accepted"
    );
}

// ---------------------------------------------------------------------------
// S6 — line tangent to circle (satisfiability verdict only).
// Geometry: line segment on y=0 (endpoints grounded); circle centre (0,2) r=2,
// tangent to the line. Satisfying configuration. DOF is representation-specific
// (see module docs) so only the consistency verdict is asserted.
//
// DCUBED_S6 (cross-checked, diff_2d.c S6): dos=0 SUCCEEDED ⇒ consistent.
// ---------------------------------------------------------------------------
#[test]
fn s6_tangent_line_circle_is_consistent() {
    eprintln!("S6 line tangent to circle");
    let mut b = Sketch2DBuilder::new();
    let a = b.add_fixed_point(0.0, 0.0);
    let c = b.add_fixed_point(4.0, 0.0);
    let line = b.add_line_segment(a, c).unwrap();
    let circle = b.add_circle(0.0, 2.0, 2.0);
    b.constrain_tangent_line_circle(line, circle).unwrap();

    let v = verdict(b.build(), None);
    assert!(v.consistent(), "S6 must be consistent (D-Cubed: SUCCEEDED)");
    assert_eq!(v.conflicts, 0, "S6 has no conflict");
}

// ---------------------------------------------------------------------------
// S7 — two parallel lines (satisfiability verdict only).
// Geometry: line L0 grounded on y=0; line L1 on y=5, same direction. Parallel.
//
// DCUBED_S7 (cross-checked, diff_2d.c S7): dos=0 SUCCEEDED ⇒ consistent.
// ---------------------------------------------------------------------------
#[test]
fn s7_parallel_lines_are_consistent() {
    eprintln!("S7 two parallel lines");
    let mut b = Sketch2DBuilder::new();
    let a0 = b.add_fixed_point(0.0, 0.0);
    let a1 = b.add_fixed_point(10.0, 0.0);
    let l0 = b.add_line_segment(a0, a1).unwrap();
    let b0 = b.add_point(0.0, 5.0);
    let b1 = b.add_point(10.0, 5.0);
    let l1 = b.add_line_segment(b0, b1).unwrap();
    b.constrain_parallel(l0, l1).unwrap();

    let v = verdict(b.build(), None);
    assert!(v.consistent(), "S7 must be consistent (D-Cubed: SUCCEEDED)");
    assert_eq!(v.conflicts, 0, "S7 has no conflict");
}

// ---------------------------------------------------------------------------
// S8 — two perpendicular lines (satisfiability verdict only).
// Geometry: horizontal line L0 grounded; vertical line L1. Perpendicular.
//
// DCUBED_S8 (cross-checked, diff_2d.c S8): dos=0 SUCCEEDED ⇒ consistent.
// ---------------------------------------------------------------------------
#[test]
fn s8_perpendicular_lines_are_consistent() {
    eprintln!("S8 two perpendicular lines");
    let mut b = Sketch2DBuilder::new();
    let a0 = b.add_fixed_point(0.0, 0.0);
    let a1 = b.add_fixed_point(10.0, 0.0);
    let l0 = b.add_line_segment(a0, a1).unwrap();
    let b0 = b.add_point(0.0, 0.0);
    let b1 = b.add_point(0.0, 10.0);
    let l1 = b.add_line_segment(b0, b1).unwrap();
    b.constrain_perpendicular(l0, l1).unwrap();

    let v = verdict(b.build(), None);
    assert!(v.consistent(), "S8 must be consistent (D-Cubed: SUCCEEDED)");
    assert_eq!(v.conflicts, 0, "S8 has no conflict");
}
