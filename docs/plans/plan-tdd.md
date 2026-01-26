# Phase 0: TDD Test Infrastructure for Agentic Development

## Overview

This plan implements the test infrastructure foundation for red-green test-driven development of the MCAD geometric kernel. The infrastructure enables AI agents to have hard pass/fail targets: run tests, see failures, implement fixes, verify passes.

**Key deliverables:**
1. OCC reference oracle via `opencascade-rs` FFI
2. Test corpus management with checksums
3. Property test patterns for geometric invariants
4. CI integration with fail-fast behavior
5. Test-runner CLI for selective suite execution
6. Golden file management for regression testing

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|----------|-----------------|
| **FFI via opencascade-rs** | User requires direct OCC integration -> FFI is faster than subprocess -> accepts C++ toolchain requirement in CI -> enables compile-time type safety for OCC types |
| **New crate `atomic_test_infra`** | Test infrastructure spans all kernel phases (1-6) -> embedding in atomic_solver mixes concerns -> dedicated crate enables reuse across atomic_geometry, atomic_brep, atomic_modeling -> clean dependency graph |
| **Download script + checksums** | Test corpora are large (5-10GB for ABC) -> Git LFS has bandwidth costs -> download script is free -> checksums ensure reproducibility -> can add S3 mirror later if links rot |
| **Human-readable test output** | AI agents parse cargo test output well -> JSON adds complexity without benefit -> human-readable enables developer debugging too -> no custom output format needed |
| **Feature-gated OCC tests** | OCC requires C++ toolchain -> not all developers have it -> feature gate `with-occ` enables opt-in -> CI always enables feature -> local dev works without OCC |
| **Lazy corpus loading** | 10k ABC models is large -> loading all on startup is slow -> lazy load with caching -> first test run downloads, subsequent runs use cache |
| **Content-hash golden files** | Semantic versioning requires manual updates -> content hash (SHA256 of result) is automatic -> enables bisecting regressions -> no manual version bumping |
| **Proptest with fixed seeds** | Random seeds cause flaky tests -> fixed seed in CI ensures reproducibility -> captured regressions file records failures -> agents get deterministic results |
| **Fail-fast CI** | Silent test skips hide bugs -> missing OCC must fail CI not skip -> explicit dependency checks before test run -> prevents shipping broken code |
| **Tolerance-based OCC comparison** | Floating-point exact comparison fails -> geometric tolerance (1e-10) for point comparison -> topology comparison via Euler characteristic -> avoids false positives from precision |
| **Retry strategy (3 retries, exp backoff 1s/2s/4s)** | Network flakiness is common -> 3 retries is standard practice -> exponential backoff prevents thundering herd -> 7s max total delay is acceptable for CI -> mirrors reqwest-retry defaults |
| **Geometry bounds ±1000.0** | MCAD domain operates in millimeter/meter scale -> ±1000 covers building-scale objects -> matches SolidWorks/Fusion default workspace -> micron-scale handled separately via scaled tests |
| **Platform support: Linux tier-1, macOS best-effort** | Linux is primary development/CI platform -> macOS support enables developer convenience -> Windows deferred (no C++ OCC Windows support in CI) -> tier-1 means "must pass", best-effort means "investigate failures but don't block" |
| **Cache: XDG on Linux, ~/Library on macOS** | XDG (`~/.cache/`) is Linux standard -> macOS uses `~/Library/Caches/` by convention -> `dirs` crate handles this automatically -> Windows deferred with platform support |
| **OCC setup instructions per-platform** | Ubuntu: `apt install libocct-dev` -> macOS: `brew install opencascade` -> specific package names documented in error message -> enables copy-paste fix |
| **Golden diff: JSON serialization + line diff** | Geometric data is structured -> JSON enables readable diffs -> line-by-line comparison shows exact field changes -> text diff tools work naturally |
| **Proptest regression: `.proptest-regressions` file** | Proptest standard location -> file per test module (`*.proptest-regressions`) -> automatic persistence of shrunk failures -> crate convention, no custom format |
| **NURBS degree limits: 1-7** | Degree 1-3 covers typical MCAD curves -> degree 4-7 covers high-continuity surfaces -> degree >7 is rare in practice -> matches OCC/typical CAD limits |
| **Golden file selection: by filename mtime** | Multiple versions can exist during transition -> most recent (mtime) is active version -> allows gradual migration -> Git tracks version history |
| **Progress callback: `Fn(u64, u64)` bytes/total** | Simple signature covers download progress -> sync callback fine for CLI output -> test infrastructure can ignore callback -> no async complexity |
| **Test output: passthrough cargo stdout/stderr** | AI agents parse cargo output well -> no filtering needed for TDD -> passthrough preserves colors/formatting -> simplest implementation |
| **Cache cleanup: rely on GitHub Actions 7-day TTL** | GitHub auto-cleans caches older than 7 days -> no explicit cleanup needed -> cache key rotation on checksum change -> accept minor disk usage |
| **OCC wrappers: owned via opencascade-rs Handle** | opencascade-rs uses `Handle<T>` (reference-counted) -> Rust wrappers hold Handle -> automatic cleanup via Drop -> inter-object refs handled by OCC internal refcounting |
| **Golden update: CLI flag, stored in env var internally** | User-facing is `--update-golden` flag -> test-runner sets `UPDATE_GOLDEN=1` env var -> golden module checks env var -> consistent with cargo test pattern |
| **NURBS knot vector validation** | arb_nurbs_curve() generates valid knot vectors by: (1) non-decreasing sequence, (2) clamped endpoints (first/last knot repeated degree+1 times), (3) multiplicity ≤ degree at internal knots -> ensures property tests fail only on kernel bugs, not generator bugs |
| **Retry exhaustion: fail CI** | After 3 failed download retries -> fail CI immediately with error message including URL and failure reason -> never use stale cache (could mask bugs) -> never skip tests (defeats purpose) -> fail-fast principle applies |
| **CI corpus subset: random 1k with seed** | ABC 10k is too large for CI -> select 1k models using `hashlib.sha256(filename).hexdigest()[:8] < "00029000"` (~10%) -> deterministic selection based on filename -> ensures reproducibility -> subset file list committed to repo |
| **OCC test isolation: serial execution** | OCC is not thread-safe -> enforce serial execution via `cargo test -p atomic_test_infra --features with-occ -- --test-threads=1` -> documented in CLAUDE.md -> test-runner enforces this when OCC enabled |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| **Subprocess OCC wrapper** | User chose FFI for speed. Subprocess has spawn overhead per test. |
| **Git LFS for corpora** | Bandwidth costs for large datasets. Download script is free. |
| **JSON test output** | User chose human-readable. Agents handle cargo output fine. |
| **Extend atomic_solver tests** | Mixes geometric kernel tests with solver. Dedicated crate is cleaner. |
| **TAP output format** | Less readable than cargo default. Adds dependency without benefit. |
| **Docker OCC container** | Slower than FFI. Adds container infrastructure complexity. |

### Constraints & Assumptions

**Technical:**
- `opencascade-rs` crate requires CMake and C++ compiler (clang/gcc)
- OpenCASCADE 7.6+ required (version in opencascade-rs)
- Proptest 1.4+ for property testing
- Criterion 0.5+ for benchmarks
- Rust 1.75+ (async in traits)

**Test Corpora:**
- NIST Manufacturing: ~100MB, STEP AP203 format
- ABC Dataset subset: ~5-10GB, OBJ format (10k models)
- CAx-IF STEP: ~200MB, STEP AP203/AP242

**CI Requirements:**
- GitHub Actions with CMake/C++ toolchain
- OCC installation or system packages
- Fail on missing dependencies (not skip)

### Known Risks

| Risk | Mitigation | Anchor |
|------|------------|--------|
| **OCC version mismatch** | Pin OCC version in CI. Document required version in README. | N/A (new code) |
| **Corpus download failures** | Retry with exponential backoff. Cache downloaded files. Add mirror URLs. | N/A (new code) |
| **Property test flakiness** | Fixed seed in CI. Regression file captures failures. | N/A (new code) |
| **CMake not found in CI** | Explicit `apt-get install cmake` step. Fail-fast check before tests. | N/A (new code) |
| **ABC dataset too large** | Use curated 1k subset for CI, full 10k for nightly. Feature gate. | N/A (new code) |
| **Golden file drift** | Require review for golden updates. Git diff shows changes. | N/A (new code) |

## Invisible Knowledge

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        atomic_test_infra crate                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │   occ_oracle    │   │     corpus      │   │   generators    │          │
│   │                 │   │                 │   │                 │          │
│   │ - OccPoint      │   │ - download()    │   │ - arb_point3d() │          │
│   │ - OccCurve      │   │ - verify()      │   │ - arb_nurbs()   │          │
│   │ - OccSurface    │   │ - cache_path()  │   │ - arb_solid()   │          │
│   │ - compare_*()   │   │ - Corpus enum   │   │                 │          │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
│            │                     │                     │                    │
│            └──────────────┬──────┴─────────────────────┘                    │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                          golden                                      │   │
│   │  - load_golden() / save_golden()                                    │   │
│   │  - compare_with_golden()                                            │   │
│   │  - GoldenFile { hash, content, updated }                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         runner                                       │   │
│   │  - TestSuite enum { Solver, Geometry, Topology, Modeling, ... }     │   │
│   │  - run_suite() / run_all()                                          │   │
│   │  - filter by name pattern                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │         bin/test-runner           │
                    │   cargo run -p atomic_test_infra  │
                    │       --bin test-runner           │
                    └───────────────────────────────────┘
```

### Data Flow

```
Test Execution Flow:

  AI Agent
      │
      ▼
  test-runner --suite geometry --filter nurbs
      │
      ├──► Check OCC available (fail-fast if not)
      │
      ├──► Check corpus downloaded (download if missing)
      │
      ├──► Run matching tests via cargo test
      │        │
      │        ├──► Property tests (proptest strategies)
      │        │        │
      │        │        └──► Generate random geometry
      │        │             Assert invariants
      │        │
      │        ├──► OCC comparison tests
      │        │        │
      │        │        ├──► Create geometry (our impl)
      │        │        ├──► Create geometry (OCC)
      │        │        └──► Compare with tolerance
      │        │
      │        └──► Golden tests
      │                 │
      │                 ├──► Load golden file
      │                 ├──► Execute operation
      │                 └──► Compare hash
      │
      └──► Output: PASSED / FAILED with details
```

### Why This Structure

**Dedicated crate (`atomic_test_infra`):**
- Reused by all kernel crates (atomic_geometry, atomic_brep, atomic_modeling)
- Avoids circular dependencies (solver doesn't depend on geometry tests)
- Single place to manage OCC bindings and corpus

**Module separation:**
- `occ_oracle`: Isolated FFI complexity. If OCC API changes, only this module updates.
- `corpus`: Unified download/cache logic. Adding new corpora is one function.
- `generators`: Proptest strategies shared across all property tests.
- `golden`: Regression infrastructure independent of specific tests.
- `runner`: CLI orchestration without test implementation details.

**Feature gates:**
- `with-occ`: Compile OCC FFI only when needed. Reduces build time for non-OCC work.
- `full-corpus`: Download all 10k ABC models. CI uses subset by default.

**OCC memory ownership:**
- opencascade-rs uses `Handle<T>` (OCC's reference-counted smart pointer)
- Rust wrapper structs hold `Handle<T>` directly: `OccPoint(Handle<gp_Pnt>)`
- Automatic cleanup: Drop trait releases Handle, OCC refcount decrements
- Inter-object references (Edge→Curve) handled by OCC's internal refcounting
- No raw pointers exposed; all access through Handle methods
- Thread safety: OCC objects are NOT thread-safe; each test gets own instances

### Invariants

**Corpus integrity:**
- Downloaded files must match SHA256 checksums
- Checksum file is authoritative; mismatch fails tests
- Cache is invalidated if checksum file changes

**OCC availability:**
- OCC tests MUST fail (not skip) if OCC unavailable
- Smoke test runs before any OCC comparison tests
- Fail message includes setup instructions

**Golden file consistency:**
- Golden files are content-addressed (hash in filename)
- Updating golden requires explicit flag (`--update-golden`)
- Git diff shows exactly what changed

**Proptest reproducibility:**
- CI uses fixed seed (configurable via env var)
- Regression file captures failing cases
- Same seed produces same random sequence

### Tradeoffs

| Choice | Benefit | Cost |
|--------|---------|------|
| FFI over subprocess | Fast test execution | Requires C++ toolchain |
| Download script over LFS | No bandwidth costs | First run is slow |
| Content-hash golden files | Automatic versioning | Filename changes on update |
| Feature-gated OCC | Optional dependency | Conditional compilation complexity |
| Lazy corpus loading | Fast startup | First test run downloads |

---

## Milestones

### Milestone 1: Crate Skeleton and OCC Oracle

**Files:**
- `crates/atomic_test_infra/Cargo.toml`
- `crates/atomic_test_infra/src/lib.rs`
- `crates/atomic_test_infra/src/occ_oracle/mod.rs`
- `crates/atomic_test_infra/src/occ_oracle/types.rs`
- `crates/atomic_test_infra/src/occ_oracle/compare.rs`
- `Cargo.toml` (workspace members update)

**Flags:** `conformance`, `error-handling`

**Requirements:**
- Create `atomic_test_infra` crate with `occ_oracle` module
- Wrap opencascade-rs types: `OccPoint`, `OccEdge`, `OccFace`, `OccSolid`
- Implement comparison functions with geometric tolerance
- Feature-gate OCC compilation under `with-occ`
- Fail with clear error if OCC unavailable when feature enabled

**Acceptance Criteria:**
- `cargo build -p atomic_test_infra` succeeds without OCC
- `cargo build -p atomic_test_infra --features with-occ` succeeds with OCC
- `compare_points(p1, p2, 1e-10)` returns true for points within tolerance
- Missing OCC produces error message with setup instructions

**Tests:**
- **Test files:** `crates/atomic_test_infra/src/occ_oracle/tests.rs`
- **Test type:** unit + integration
- **Backing:** default-derived
- **Scenarios:**
  - Normal: Compare identical OCC and native points (distance 0.0)
  - Edge: Compare points at tolerance boundary - pass at `1e-10 - 5e-11` (inside), fail at `1e-10 + 5e-11` (outside)
  - Error: OCC unavailable produces `OccError::Unavailable` with platform-specific setup instructions

**Code Intent:**
- New crate `atomic_test_infra` in workspace
- `occ_oracle` module with submodules `types`, `compare`
- `OccPoint`, `OccEdge`, `OccFace`, `OccSolid` wrapper structs
- `compare_points()`, `compare_edges()`, `compare_topology()` functions
- `OccError` enum:
  - `Unavailable { setup_instructions: &'static str }` - OCC not installed
  - `FfiFailed { source: opencascade_rs::Error }` - FFI call failed
  - `ComparisonFailed { expected: String, actual: String, tolerance: f64 }` - values differ
  - `InternalError { message: String }` - OCC exception caught
- Feature `with-occ` gates `opencascade-rs` dependency
- `check_occ_available()` function returning `Result<(), OccError::Unavailable>`
- Setup instructions per platform:
  - Ubuntu: `sudo apt install libocct-dev cmake`
  - macOS: `brew install opencascade cmake`
- OCC test isolation: tests MUST run with `--test-threads=1` when `with-occ` enabled (OCC is not thread-safe)
  - test-runner enforces this automatically
  - Document in CLAUDE.md for manual test runs

**Code Changes:**

```diff
--- /dev/null
+++ b/Cargo.toml
@@ -13,6 +13,7 @@ members = [
     "crates/ecad_solver",
     "crates/atomic_solver",
+    "crates/atomic_test_infra",
     "crates/atomic_scripting",
     "crates/atomic_scripting_macros",
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/Cargo.toml
@@ -0,0 +1,37 @@
+[package]
+name = "atomic_test_infra"
+version = "0.1.0"
+edition = "2021"
+description = "Test infrastructure for TDD-driven MCAD geometric kernel development"
+license = "MIT OR Apache-2.0"
+
+[lib]
+path = "src/lib.rs"
+
+[features]
+default = []
+with-occ = ["opencascade-rs"]
+full-corpus = []
+
+[dependencies]
+anyhow = "1.0"
+thiserror = "2.0"
+
+# OCC FFI (feature-gated)
+opencascade-rs = { version = "0.4", optional = true }
+
+# Corpus download
+reqwest = { workspace = true, features = ["stream"] }
+tokio = { workspace = true, features = ["fs", "io-util"] }
+sha2 = { workspace = true }
+dirs = { workspace = true }
+
+# Property testing
+proptest = "1.4"
+
+# Test runner
+clap = { workspace = true, features = ["derive"] }
+regex = { workspace = true }
+
+# Golden files
+serde_json = { workspace = true }
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/lib.rs
@@ -0,0 +1,9 @@
+//! Test infrastructure for TDD-driven MCAD geometric kernel development.
+
+#[cfg(feature = "with-occ")]
+pub mod occ_oracle;
+
+pub mod corpus;
+pub mod generators;
+pub mod golden;
+pub mod runner;
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/occ_oracle/mod.rs
@@ -0,0 +1,52 @@
+//! OpenCASCADE reference oracle for geometric operation validation.
+
+mod types;
+mod compare;
+
+#[cfg(test)]
+mod tests;
+
+pub use types::{OccPoint, OccEdge, OccFace, OccSolid};
+pub use compare::{compare_points, compare_edges, compare_topology};
+
+use thiserror::Error;
+
+#[derive(Debug, Error)]
+pub enum OccError {
+    #[error("OpenCASCADE not available. Setup instructions:\n{setup_instructions}")]
+    Unavailable { setup_instructions: &'static str },
+
+    #[error("OCC FFI call failed: {source}")]
+    FfiFailed {
+        #[from]
+        source: opencascade_rs::Error,
+    },
+
+    #[error("Comparison failed: expected {expected}, actual {actual}, tolerance {tolerance}")]
+    ComparisonFailed { expected: String, actual: String, tolerance: f64 },
+
+    #[error("OCC internal error: {message}")]
+    InternalError { message: String },
+}
+
+pub fn check_occ_available() -> Result<(), OccError> {
+    #[cfg(target_os = "linux")]
+    const SETUP: &str = "sudo apt install libocct-dev cmake";
+
+    #[cfg(target_os = "macos")]
+    const SETUP: &str = "brew install opencascade cmake";
+
+    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
+    const SETUP: &str = "OpenCASCADE installation not documented for this platform";
+
+    Err(OccError::Unavailable {
+        setup_instructions: SETUP
+    })
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/occ_oracle/types.rs
@@ -0,0 +1,29 @@
+//! OCC type wrappers with owned Handle<T> for automatic memory management.
+
+use opencascade_rs::primitives::{gp_Pnt, Handle};
+
+pub struct OccPoint(Handle<gp_Pnt>);
+
+impl OccPoint {
+    pub fn new(x: f64, y: f64, z: f64) -> Self {
+        OccPoint(Handle::new(gp_Pnt::new(x, y, z)))
+    }
+
+    pub fn handle(&self) -> &Handle<gp_Pnt> {
+        &self.0
+    }
+}
+
+pub struct OccEdge;
+pub struct OccFace;
+pub struct OccSolid;
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/occ_oracle/compare.rs
@@ -0,0 +1,39 @@
+//! Comparison functions with geometric tolerance.
+
+use super::{OccPoint, OccEdge, OccError};
+
+pub fn compare_points(p1: &OccPoint, p2: &OccPoint, tolerance: f64) -> Result<bool, OccError> {
+    let h1 = p1.handle();
+    let h2 = p2.handle();
+
+    let dx = h1.x() - h2.x();
+    let dy = h1.y() - h2.y();
+    let dz = h1.z() - h2.z();
+    let distance = (dx*dx + dy*dy + dz*dz).sqrt();
+
+    if distance <= tolerance {
+        Ok(true)
+    } else {
+        Err(OccError::ComparisonFailed {
+            expected: format!("({}, {}, {})", h1.x(), h1.y(), h1.z()),
+            actual: format!("({}, {}, {})", h2.x(), h2.y(), h2.z()),
+            tolerance,
+        })
+    }
+}
+
+pub fn compare_edges(_e1: &OccEdge, _e2: &OccEdge, _tolerance: f64) -> Result<bool, OccError> {
+    todo!("Edge comparison not yet implemented")
+}
+
+pub fn compare_topology(expected_euler: i32, actual_euler: i32) -> Result<bool, OccError> {
+    if expected_euler == actual_euler {
+        Ok(true)
+    } else {
+        Err(OccError::ComparisonFailed {
+            expected: format!("Euler={}", expected_euler),
+            actual: format!("Euler={}", actual_euler),
+            tolerance: 0.0,
+        })
+    }
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/occ_oracle/tests.rs
@@ -0,0 +1,52 @@
+//! Unit tests for OCC oracle.
+
+use super::*;
+
+#[test]
+fn test_compare_identical_points() {
+    let p1 = OccPoint::new(1.0, 2.0, 3.0);
+    let p2 = OccPoint::new(1.0, 2.0, 3.0);
+
+    assert!(compare_points(&p1, &p2, 1e-10).is_ok());
+}
+
+#[test]
+fn test_compare_points_within_tolerance() {
+    let p1 = OccPoint::new(0.0, 0.0, 0.0);
+    let p2 = OccPoint::new(0.0, 0.0, 5e-11);
+
+    assert!(compare_points(&p1, &p2, 1e-10).is_ok());
+}
+
+#[test]
+fn test_compare_points_outside_tolerance() {
+    let p1 = OccPoint::new(0.0, 0.0, 0.0);
+    let p2 = OccPoint::new(0.0, 0.0, 1.5e-10);
+
+    let result = compare_points(&p1, &p2, 1e-10);
+    assert!(result.is_err());
+
+    if let Err(OccError::ComparisonFailed { tolerance, .. }) = result {
+        assert_eq!(tolerance, 1e-10);
+    }
+}
+
+#[test]
+fn test_occ_unavailable_error() {
+    let result = check_occ_available();
+    assert!(result.is_err());
+
+    if let Err(OccError::Unavailable { setup_instructions }) = result {
+        #[cfg(target_os = "linux")]
+        assert!(setup_instructions.contains("apt install libocct-dev cmake"));
+
+        #[cfg(target_os = "macos")]
+        assert!(setup_instructions.contains("brew install opencascade cmake"));
+    }
+}
+
+#[test]
+fn test_topology_comparison() {
+    assert!(compare_topology(2, 2).is_ok());
+    assert!(compare_topology(2, 3).is_err());
+}
```

---

### Milestone 2: Test Corpus Management

**Files:**
- `crates/atomic_test_infra/src/corpus/mod.rs`
- `crates/atomic_test_infra/src/corpus/download.rs`
- `crates/atomic_test_infra/src/corpus/verify.rs`
- `crates/atomic_test_infra/src/corpus/cache.rs`
- `tests/fixtures/checksums.txt`
- `tests/fixtures/download.sh`
- `tests/fixtures/README.md`

**Flags:** `error-handling`

**Requirements:**
- `Corpus` enum with variants: `NistManufacturing`, `AbcDataset`, `CaxIfStep`
- `download(corpus)` function with retry and progress reporting
- `verify(corpus)` function checking SHA256 checksums
- `cache_path(corpus)` returning `PathBuf` to cached files
- Lazy loading: download on first access, cache thereafter
- Shell script `download.sh` for manual/CI download

**Acceptance Criteria:**
- `download(Corpus::NistManufacturing)` downloads to cache directory
- `verify(Corpus::NistManufacturing)` passes with correct checksums
- `verify()` fails with mismatch details if file corrupted
- Repeated `download()` uses cache (no re-download)
- `./tests/fixtures/download.sh` downloads all corpora

**Tests:**
- **Test files:** `crates/atomic_test_infra/src/corpus/tests.rs`
- **Test type:** integration
- **Backing:** default-derived
- **Scenarios:**
  - Normal: Download small test file (~1KB fixture), verify SHA256 checksum
  - Edge: Corrupted file (modified after download) detected by checksum mismatch
  - Error: HTTP 503 simulated via `wiremock` - verify 3 retries with 1s/2s/4s backoff, then `CorpusError::DownloadFailed`
  - Error: Timeout simulated via delayed wiremock response (30s delay) - verify `CorpusError::DownloadFailed` with timeout message
  - Error: Connection refused simulated via wiremock server stop - verify `CorpusError::DownloadFailed` with connection error message

**Code Intent:**
- `Corpus` enum with `NistManufacturing`, `AbcDataset`, `CaxIfStep`
- `CorpusConfig` struct with `url`, `checksum`, `cache_dir`
- `download()` async fn with reqwest, retry logic (3 retries, 1s/2s/4s backoff)
- `ProgressCallback` type: `Fn(bytes_downloaded: u64, total_bytes: u64)` for CLI progress display
- `verify()` fn computing SHA256 and comparing to checksums.txt
- `cache_path()` returning platform-specific cache dir via `dirs` crate (`~/.cache/` on Linux, `~/Library/Caches/` on macOS)
- `checksums.txt` with format: `<sha256> <filename>`
- `download.sh` shell script calling curl with checksum verification
- `CorpusError` enum: `DownloadFailed`, `ChecksumMismatch`, `NotFound`

**Code Changes:**

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/corpus/mod.rs
@@ -0,0 +1,7 @@
+//! Test corpus management with download, caching, and verification.
+
+mod download;
+mod verify;
+mod cache;
+
+pub use download::{Corpus, download, ProgressCallback};
+pub use verify::verify;
+pub use cache::cache_path;
+
+use thiserror::Error;
+
+#[derive(Debug, Error)]
+pub enum CorpusError {
+    #[error("Download failed: {0}")]
+    DownloadFailed(String),
+
+    #[error("Checksum mismatch for {filename}: expected {expected}, got {actual}")]
+    ChecksumMismatch { filename: String, expected: String, actual: String },
+
+    #[error("Corpus not found: {0}")]
+    NotFound(String),
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/corpus/download.rs
@@ -0,0 +1,69 @@
+//! Corpus download with retry and progress reporting.
+
+use super::CorpusError;
+use std::time::Duration;
+use tokio::fs::File;
+use tokio::io::AsyncWriteExt;
+
+pub type ProgressCallback = dyn Fn(u64, u64);
+
+#[derive(Debug, Clone, Copy)]
+pub enum Corpus {
+    NistManufacturing,
+    AbcDataset,
+    CaxIfStep,
+}
+
+impl Corpus {
+    pub fn url(&self) -> &'static str {
+        match self {
+            Corpus::NistManufacturing => "https://example.com/nist-manufacturing.tar.gz",
+            Corpus::AbcDataset => "https://example.com/abc-dataset.tar.gz",
+            Corpus::CaxIfStep => "https://example.com/caxif-step.tar.gz",
+        }
+    }
+
+    pub fn name(&self) -> &'static str {
+        match self {
+            Corpus::NistManufacturing => "nist-manufacturing",
+            Corpus::AbcDataset => "abc-dataset",
+            Corpus::CaxIfStep => "caxif-step",
+        }
+    }
+}
+
+pub async fn download(
+    corpus: Corpus,
+    progress: Option<&ProgressCallback>,
+) -> Result<(), CorpusError> {
+    let mut retries = 0;
+    let max_retries = 3;
+    let backoff_delays = [1000, 2000, 4000];
+
+    while retries < max_retries {
+        match download_attempt(corpus, progress).await {
+            Ok(()) => return Ok(()),
+            Err(e) if retries < max_retries - 1 => {
+                let delay = Duration::from_millis(backoff_delays[retries]);
+                tokio::time::sleep(delay).await;
+                retries += 1;
+            }
+            Err(e) => {
+                return Err(CorpusError::DownloadFailed(format!(
+                    "Failed after {} retries: {}",
+                    max_retries, e
+                )));
+            }
+        }
+    }
+
+    unreachable!()
+}
+
+async fn download_attempt(
+    corpus: Corpus,
+    _progress: Option<&ProgressCallback>,
+) -> Result<(), Box<dyn std::error::Error>> {
+    todo!("Download implementation")
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/corpus/verify.rs
@@ -0,0 +1,24 @@
+//! Checksum verification.
+
+use super::{Corpus, CorpusError};
+use sha2::{Sha256, Digest};
+use std::path::Path;
+
+pub async fn verify(corpus: Corpus, path: &Path) -> Result<(), CorpusError> {
+    let expected_checksum = get_expected_checksum(corpus);
+    let actual_checksum = compute_checksum(path).await?;
+
+    if expected_checksum != actual_checksum {
+        return Err(CorpusError::ChecksumMismatch {
+            filename: corpus.name().to_string(),
+            expected: expected_checksum,
+            actual: actual_checksum,
+        });
+    }
+
+    Ok(())
+}
+
+fn get_expected_checksum(corpus: Corpus) -> String {
+    todo!("Load from checksums.txt")
+}
+
+async fn compute_checksum(path: &Path) -> Result<String, CorpusError> {
+    todo!("Compute SHA256")
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/corpus/cache.rs
@@ -0,0 +1,21 @@
+//! Platform-specific cache path management.
+
+use super::Corpus;
+use std::path::PathBuf;
+
+pub fn cache_path(corpus: Corpus) -> PathBuf {
+    let cache_dir = dirs::cache_dir()
+        .unwrap_or_else(|| PathBuf::from(".cache"));
+
+    cache_dir
+        .join("atomic_test_infra")
+        .join(corpus.name())
+}
```

```diff
--- /dev/null
+++ b/tests/fixtures/checksums.txt
@@ -0,0 +1,3 @@
+e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  nist-manufacturing.tar.gz
+e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  abc-dataset.tar.gz
+e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  caxif-step.tar.gz
```

```diff
--- /dev/null
+++ b/tests/fixtures/download.sh
@@ -0,0 +1,22 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/atomic_test_infra"
+mkdir -p "$CACHE_DIR"
+
+download_corpus() {
+    local name=$1
+    local url=$2
+    local output="$CACHE_DIR/$name.tar.gz"
+
+    if [ ! -f "$output" ]; then
+        echo "Downloading $name..."
+        curl -L -o "$output" "$url"
+    fi
+}
+
+download_corpus "nist-manufacturing" "https://example.com/nist-manufacturing.tar.gz"
+download_corpus "abc-dataset" "https://example.com/abc-dataset.tar.gz"
+download_corpus "caxif-step" "https://example.com/caxif-step.tar.gz"
+
+sha256sum -c "$(dirname "$0")/checksums.txt"
```

```diff
--- /dev/null
+++ b/tests/fixtures/README.md
@@ -0,0 +1,21 @@
+# Test Fixtures
+
+Test corpora for geometric kernel validation.
+
+## Download
+
+```bash
+./download.sh
+```
+
+## Verify
+
+```bash
+sha256sum -c checksums.txt
+```
+
+## Corpora
+
+- **NIST Manufacturing**: ~100MB, STEP AP203 format
+- **ABC Dataset**: ~5-10GB, OBJ format (10k models)
+- **CAx-IF STEP**: ~200MB, STEP AP203/AP242
```

---

### Milestone 3: Property Test Generators

**Files:**
- `crates/atomic_test_infra/src/generators/mod.rs`
- `crates/atomic_test_infra/src/generators/primitives.rs`
- `crates/atomic_test_infra/src/generators/curves.rs`
- `crates/atomic_test_infra/src/generators/surfaces.rs`
- `crates/atomic_test_infra/src/generators/topology.rs`

**Flags:** `complex-algorithm`

**Requirements:**
- Proptest strategies for geometric primitives: `arb_point2d()`, `arb_point3d()`, `arb_vector3d()`
- Strategies for curves: `arb_line()`, `arb_arc()`, `arb_nurbs_curve()`
- Strategies for surfaces: `arb_plane()`, `arb_nurbs_surface()`
- Strategies for topology: `arb_edge()`, `arb_face()`, `arb_solid()` (simple primitives)
- Configurable bounds (default ±1000.0)
- Fixed seed support via `PROPTEST_SEED` env var

**Acceptance Criteria:**
- `proptest! { fn test(p in arb_point3d()) { ... } }` compiles and runs
- Generated points are within configured bounds
- Same seed produces same sequence
- Strategies compose: `arb_line()` uses `arb_point3d()` internally

**Tests:**
- **Test files:** `crates/atomic_test_infra/src/generators/tests.rs`
- **Test type:** property-based
- **Backing:** default-derived
- **Scenarios:**
  - Normal: Generated points within bounds (default ±1000.0)
  - Normal: Generated NURBS curves have valid knot vectors (non-decreasing, clamped endpoints, multiplicity ≤ degree)
  - Normal: Generated NURBS curves satisfy minimum control points constraint (count ≥ degree + 1)
  - Edge: Extreme bounds (±1e10) don't cause overflow

**Code Intent:**
- `primitives.rs`: `arb_point2d()`, `arb_point3d()`, `arb_vector2d()`, `arb_vector3d()`
- `curves.rs`: `arb_line()`, `arb_arc()`, `arb_circle()`, `arb_nurbs_curve(degree, control_points)`
- `surfaces.rs`: `arb_plane()`, `arb_cylinder()`, `arb_nurbs_surface()`
- `topology.rs`: `arb_vertex()`, `arb_edge()`, `arb_wire()`, `arb_face()`, `arb_shell()`, `arb_solid()`
- Each strategy uses `prop_compose!` macro
- `GeneratorConfig` struct:
  - `bounds: f64` (default ±1000.0 for MCAD millimeter/meter scale)
  - `nurbs_degree_min: u32` (default 1)
  - `nurbs_degree_max: u32` (default 7 - covers typical MCAD complexity)
- Proptest regression files: standard `.proptest-regressions` per module (e.g., `primitives.proptest-regressions`)
- Re-export all strategies in `generators/mod.rs`

**Code Changes:**

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/generators/mod.rs
@@ -0,0 +1,30 @@
+//! Proptest strategies for geometric primitives.
+
+mod primitives;
+mod curves;
+mod surfaces;
+mod topology;
+
+pub use primitives::{arb_point2d, arb_point3d, arb_vector2d, arb_vector3d};
+pub use curves::{arb_line, arb_arc, arb_circle, arb_nurbs_curve};
+pub use surfaces::{arb_plane, arb_cylinder, arb_nurbs_surface};
+pub use topology::{arb_vertex, arb_edge, arb_wire, arb_face, arb_shell, arb_solid};
+
+#[derive(Debug, Clone)]
+pub struct GeneratorConfig {
+    pub bounds: f64,
+    pub nurbs_degree_min: u32,
+    pub nurbs_degree_max: u32,
+}
+
+impl Default for GeneratorConfig {
+    fn default() -> Self {
+        Self {
+            bounds: 1000.0,
+            nurbs_degree_min: 1,
+            nurbs_degree_max: 7,
+        }
+    }
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/generators/primitives.rs
@@ -0,0 +1,48 @@
+//! Proptest strategies for primitive geometric types.
+
+use proptest::prelude::*;
+use super::GeneratorConfig;
+
+pub fn arb_point2d() -> impl Strategy<Value = (f64, f64)> {
+    arb_point2d_with_config(GeneratorConfig::default())
+}
+
+pub fn arb_point2d_with_config(config: GeneratorConfig) -> impl Strategy<Value = (f64, f64)> {
+    let b = config.bounds;
+    (-b..=b, -b..=b)
+}
+
+pub fn arb_point3d() -> impl Strategy<Value = (f64, f64, f64)> {
+    arb_point3d_with_config(GeneratorConfig::default())
+}
+
+pub fn arb_point3d_with_config(config: GeneratorConfig) -> impl Strategy<Value = (f64, f64, f64)> {
+    let b = config.bounds;
+    (-b..=b, -b..=b, -b..=b)
+}
+
+pub fn arb_vector2d() -> impl Strategy<Value = (f64, f64)> {
+    arb_vector2d_with_config(GeneratorConfig::default())
+}
+
+pub fn arb_vector2d_with_config(config: GeneratorConfig) -> impl Strategy<Value = (f64, f64)> {
+    let b = config.bounds;
+    (-b..=b, -b..=b)
+}
+
+pub fn arb_vector3d() -> impl Strategy<Value = (f64, f64, f64)> {
+    arb_vector3d_with_config(GeneratorConfig::default())
+}
+
+pub fn arb_vector3d_with_config(config: GeneratorConfig) -> impl Strategy<Value = (f64, f64, f64)> {
+    let b = config.bounds;
+    (-b..=b, -b..=b, -b..=b)
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/generators/curves.rs
@@ -0,0 +1,45 @@
+//! Proptest strategies for curve types.
+
+use proptest::prelude::*;
+use super::GeneratorConfig;
+
+pub struct Line {
+    pub start: (f64, f64, f64),
+    pub end: (f64, f64, f64),
+}
+
+pub fn arb_line() -> impl Strategy<Value = Line> {
+    (super::primitives::arb_point3d(), super::primitives::arb_point3d())
+        .prop_map(|(start, end)| Line { start, end })
+}
+
+pub struct Arc {
+    pub center: (f64, f64, f64),
+    pub radius: f64,
+    pub start_angle: f64,
+    pub end_angle: f64,
+}
+
+pub fn arb_arc() -> impl Strategy<Value = Arc> {
+    (
+        super::primitives::arb_point3d(),
+        1.0..1000.0,
+        0.0..std::f64::consts::TAU,
+        0.0..std::f64::consts::TAU,
+    ).prop_map(|(center, radius, start_angle, end_angle)| Arc {
+        center,
+        radius,
+        start_angle,
+        end_angle,
+    })
+}
+
+pub fn arb_circle() -> impl Strategy<Value = (f64, f64, f64, f64)> {
+    (super::primitives::arb_point3d(), 1.0..1000.0)
+        .prop_map(|(center, radius)| (center.0, center.1, center.2, radius))
+}
+
+pub fn arb_nurbs_curve() -> impl Strategy<Value = Vec<(f64, f64, f64)>> {
+    prop::collection::vec(super::primitives::arb_point3d(), 2..10)
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/generators/surfaces.rs
@@ -0,0 +1,31 @@
+//! Proptest strategies for surface types.
+
+use proptest::prelude::*;
+
+pub struct Plane {
+    pub origin: (f64, f64, f64),
+    pub normal: (f64, f64, f64),
+}
+
+pub fn arb_plane() -> impl Strategy<Value = Plane> {
+    (super::primitives::arb_point3d(), super::primitives::arb_vector3d())
+        .prop_map(|(origin, normal)| Plane { origin, normal })
+}
+
+pub struct Cylinder {
+    pub axis_origin: (f64, f64, f64),
+    pub axis_direction: (f64, f64, f64),
+    pub radius: f64,
+}
+
+pub fn arb_cylinder() -> impl Strategy<Value = Cylinder> {
+    (super::primitives::arb_point3d(), super::primitives::arb_vector3d(), 1.0..1000.0)
+        .prop_map(|(axis_origin, axis_direction, radius)| Cylinder {
+            axis_origin,
+            axis_direction,
+            radius,
+        })
+}
+
+pub fn arb_nurbs_surface() -> impl Strategy<Value = Vec<Vec<(f64, f64, f64)>>> {
+    prop::collection::vec(prop::collection::vec(super::primitives::arb_point3d(), 2..5), 2..5)
+}
```

```diff
--- /dev/null
+++ b/crates/atomic_test_infra/src/generators/topology.rs
@@ -0,0 +1,36 @@
+//! Proptest strategies for topological types.
+
+use proptest::prelude::*;
+
+pub type Vertex = (f64, f64, f64);
+
+pub fn arb_vertex() -> impl Strategy<Value = Vertex> {
+    super::primitives::arb_point3d()
+}
+
+pub struct Edge {
+    pub start: Vertex,
+    pub end: Vertex,
+}
+
+pub fn arb_edge() -> impl Strategy<Value = Edge> {
+    (arb_vertex(), arb_vertex())
+        .prop_map(|(start, end)| Edge { start, end })
+}
+
+pub fn arb_wire() -> impl Strategy<Value = Vec<Edge>> {
+    prop::collection::vec(arb_edge(), 1..10)
+}
+
+pub fn arb_face() -> impl Strategy<Value = Vec<Vertex>> {
+    prop::collection::vec(arb_vertex(), 3..10)
+}
+
+pub fn arb_shell() -> impl Strategy<Value = Vec<Vec<Vertex>>> {
+    prop::collection::vec(arb_face(), 1..10)
+}
+
+pub fn arb_solid() -> impl Strategy<Value = Vec<Vec<Vertex>>> {
+    arb_shell()
+}
```

---

### Milestone 4: Golden File Management

**Files:**
- `crates/atomic_test_infra/src/golden/mod.rs`
- `crates/atomic_test_infra/src/golden/file.rs`
- `crates/atomic_test_infra/src/golden/compare.rs`
- `tests/fixtures/golden/.gitkeep`
- `tests/fixtures/golden/README.md`

**Flags:** `needs-rationale`

**Requirements:**
- `GoldenFile` struct with `name`, `content_hash`, `content`
- `load_golden(test_name)` loading from `tests/fixtures/golden/`
- `save_golden(test_name, content)` with content-hash filename
- `compare_with_golden(test_name, actual)` returning diff on mismatch
- `--update-golden` flag support for test runner
- Human-readable diff output on mismatch

**Acceptance Criteria:**
- `save_golden("test_a", data)` creates `tests/fixtures/golden/test_a.<hash>.golden`
- `load_golden("test_a")` finds latest golden file by name prefix
- Mismatch produces diff showing expected vs actual
- Update mode overwrites golden with new hash

**Tests:**
- **Test files:** `crates/atomic_test_infra/src/golden/tests.rs`
- **Test type:** unit
- **Backing:** default-derived
- **Scenarios:**
  - Normal: Save and load golden file
  - Normal: Detect mismatch with diff
  - Edge: Multiple golden files with same prefix (use latest)
  - Error: Missing golden file with helpful message

**Code Intent:**
- `GoldenFile` struct: `name: String`, `hash: String`, `content: Vec<u8>`
- `load_golden(name)` scanning `tests/fixtures/golden/` for matching prefix, selecting by filesystem mtime (most recent wins)
- `save_golden(name, content)` computing SHA256, writing `<name>.<hash>.golden`
- `compare_with_golden(name, actual)` returning `GoldenResult::Match | Mismatch { diff: String }`
  - Diff format: JSON serialize both values, then line-by-line text diff showing `- expected` / `+ actual`
- `UPDATE_GOLDEN` env var checked for update mode (set by test-runner's `--update-golden` flag)
- `GoldenError` enum: `NotFound { name: String }`, `IoError(std::io::Error)`, `ParseError { reason: String }`
- README.md explaining golden file workflow (save/load/update/diff)

**Code Changes:** (Developer fills after plan approval)

---

### Milestone 5: Test Runner CLI

**Files:**
- `crates/atomic_test_infra/src/runner/mod.rs`
- `crates/atomic_test_infra/src/runner/suites.rs`
- `crates/atomic_test_infra/src/runner/filter.rs`
- `crates/atomic_test_infra/src/bin/test-runner.rs`

**Flags:** `conformance`

**Requirements:**
- `TestSuite` enum: `Solver`, `Geometry`, `Topology`, `Modeling`, `History`, `Io`, `All`
- `run_suite(suite, filter)` executing cargo test with appropriate flags
- Filter by test name pattern (regex support)
- `--with-occ` flag enabling OCC comparison tests
- `--update-golden` flag for golden file updates
- Exit code reflects test pass/fail

**Acceptance Criteria:**
- `cargo run -p atomic_test_infra --bin test-runner -- --suite solver` runs solver tests
- `cargo run -p atomic_test_infra --bin test-runner -- --suite geometry --filter nurbs` runs filtered tests
- `--with-occ` adds `--features with-occ` to cargo test command
- Exit code 0 on pass, 1 on fail

**Tests:**
- **Test files:** `crates/atomic_test_infra/src/runner/tests.rs`
- **Test type:** integration
- **Backing:** default-derived
- **Scenarios:**
  - Normal: Spawn test-runner process for solver suite, assert `process.exit_status().code() == Some(0)` on pass
  - Normal: Filter tests by name pattern `--filter arc`, verify only arc tests run
  - Edge: Empty filter (`--filter zzz_nonexistent`) exits with code 0 (no tests to fail)

**Code Intent:**
- `TestSuite` enum with `target_crate()` and `test_prefix()` methods
- `RunConfig` struct: `suite`, `filter`, `with_occ`, `update_golden`
- `run_suite(config)` building cargo command, spawning, capturing exit code
- `filter.rs` with regex-based test name filtering
- `bin/test-runner.rs` using clap for CLI parsing
- Output handling: passthrough cargo stdout/stderr directly (no filtering/buffering) - preserves colors and formatting

**Code Changes:** (Developer fills after plan approval)

---

### Milestone 6: CI Integration

**Files:**
- `.github/workflows/geometry-tests.yml`
- `scripts/ci/check-occ.sh`
- `scripts/ci/setup-occ.sh`

**Flags:** `error-handling`

**Requirements:**
- GitHub Actions workflow for geometric kernel tests
- Fail-fast: OCC check before tests (not skip)
- Corpus download with checksum verification
- Feature-gated test runs (`with-occ`, `full-corpus`)
- Cache OCC installation and corpus downloads
- Matrix: Ubuntu + macOS

**Acceptance Criteria:**
- Push to main triggers test workflow
- Missing OCC fails workflow with setup instructions
- Corpus checksum mismatch fails workflow
- Cached runs skip download/install
- All test suites run in sequence

**Tests:**
- Skip: CI workflow is tested by running it

**Code Intent:**
- `geometry-tests.yml` workflow with steps:
  - Install Rust toolchain
  - Cache OCC installation
  - Run `check-occ.sh` (fail-fast)
  - Cache corpus downloads
  - Run `download.sh` if cache miss
  - Run `sha256sum -c checksums.txt`
  - Run test-runner for each suite
- `check-occ.sh` verifying `occt-config` or fallback detection
- `setup-occ.sh` installing OCC via apt/brew
- Matrix: `ubuntu-latest`, `macos-latest`
- Cache keys: `occ-${{ runner.os }}-v1`, `corpus-${{ hashFiles('checksums.txt') }}`

**Code Changes:** (Developer fills after plan approval)

---

### Milestone 7: Documentation

**Delegated to:** @agent-technical-writer (mode: post-implementation)

**Source:** `## Invisible Knowledge` section of this plan

**Files:**
- `crates/atomic_test_infra/CLAUDE.md`
- `crates/atomic_test_infra/README.md`
- `tests/fixtures/README.md`

**Requirements:**
Delegate to Technical Writer. Key deliverables:
- CLAUDE.md: Navigation index for test infrastructure
- README.md: Setup instructions, usage examples, architecture explanation
- tests/fixtures/README.md: Corpus download and verification instructions

**Acceptance Criteria:**
- CLAUDE.md is tabular index only
- README.md explains OCC setup on Ubuntu/macOS
- README.md includes test-runner usage examples
- tests/fixtures/README.md documents checksum verification

---

## Milestone Dependencies

```
M1 (OCC Oracle) ────┬────► M5 (Test Runner) ────► M6 (CI)
                    │
M2 (Corpus) ────────┤
                    │
M3 (Generators) ────┤
                    │
M4 (Golden) ────────┘
                                                    │
                                                    ▼
                                              M7 (Docs)
```

**Wave 1 (Parallel):** M1, M2, M3, M4 - independent foundation modules
**Wave 2 (Sequential):** M5 - depends on all Wave 1
**Wave 3 (Sequential):** M6 - depends on M5
**Wave 4 (Sequential):** M7 - depends on all

---

## Test Targets by Layer (For AI Agents)

This section defines specific pass/fail targets for each kernel layer.

### Layer 1: Geometry
| Test | Target | Pass Criteria |
|------|--------|---------------|
| NURBS curve evaluation | OCC comparison | Points match within 1e-10 |
| Surface normal | OCC comparison | Normals match within 1e-10 |
| Curve continuity | Property test | G0/G1/G2 at join points |
| Curve intersection | OCC comparison | Intersection points match |

### Layer 2: Topology
| Test | Target | Pass Criteria |
|------|--------|---------------|
| Euler characteristic | Property test | V - E + F = 2 for closed solids |
| Half-edge twin | Property test | twin(twin(e)) == e for all edges |
| Face orientation | Property test | All faces consistently oriented |
| Manifold property | Property test | Each edge has exactly 2 faces |

### Layer 3: Modeling
| Test | Target | Pass Criteria |
|------|--------|---------------|
| Boolean union | OCC comparison | Topology matches OCC result |
| Boolean difference | OCC comparison | Topology matches OCC result |
| Fillet | Property test | G1 continuity with adjacent faces |
| Extrude | Property test | Result is valid closed solid |

### Layer 4: Constraints
| Test | Target | Pass Criteria |
|------|--------|---------------|
| NIST benchmarks | Existing suite | Pass 32/32 problems |
| Arc constraints | Property test | Convergence in <50 iterations |
| Assembly DOF | Property test | DOF count matches expected |

### Layer 5: History
| Test | Target | Pass Criteria |
|------|--------|---------------|
| Dependency tracking | Unit test | Topological sort correct |
| Incremental regen | Property test | Same result as full regen |
| Undo/redo | Integration test | State correctly restored |

### Layer 6: I/O
| Test | Target | Pass Criteria |
|------|--------|---------------|
| STEP export | CAx-IF tests | Pass recommended practices |
| STEP round-trip | Golden files | Import(Export(model)) == model |
| Tessellation | Property test | Mesh is watertight |
