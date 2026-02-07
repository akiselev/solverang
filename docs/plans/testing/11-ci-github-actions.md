# Plan 11: CI GitHub Actions -- Cross-Platform Testing

## Goal

Create comprehensive CI/CD pipelines using GitHub Actions to:
- Test across **Linux, macOS, and Windows**
- Test all feature combinations (including V3 modules that are always compiled)
- Run V3 integration test suites (`pipeline/incremental_tests.rs`,
  `pipeline/integration_tests.rs`, `pipeline/minpack_bridge_tests.rs`)
- Run V3 end-to-end tests via `Sketch2DBuilder`
- Run linting, formatting, and doc tests (including V3 doc examples)
- Integrate nightly testing (mutation, fuzzing, sanitizers, coverage)
- Track benchmark performance (legacy + V3 pipeline benchmarks)
- Support snapshot testing (plan 04) with `INSTA_UPDATE=no`
- Automate releases

Currently, **no CI/CD exists** -- no `.github/workflows/` directory.

## Architecture Considerations

### V3 Modules Are Always Compiled

The V3 modules (`id`, `param`, `entity`, `constraint`, `graph`, `solve`, `reduce`,
`dataflow`, `system`, `sketch2d`, `sketch3d`, `assembly`, `pipeline`) are declared
unconditionally in `lib.rs` -- they have no feature gate. This means:

- Every `cargo test` invocation runs V3 unit tests automatically.
- Feature matrix testing is about **optional dependencies** (`parallel`, `sparse`,
  `jit`, `macros`, `nist`, `geometry`), not V3 modules.
- The V3 test suites (333 tests across the new modules) run on every CI build.

### Test Suite Sizes

| Module | In-module Tests | Notes |
|--------|----------------|-------|
| `system.rs` | 16 tests | ConstraintSystem coordinator |
| `id.rs` | 4 tests | Generational IDs |
| `param/store.rs` | 8 tests | ParamStore with free-list |
| `dataflow/tracker.rs` | 13 tests | ChangeTracker |
| `dataflow/cache.rs` | 10 tests | SolutionCache |
| `graph/` | 46 tests | Bipartite graph, clusters, DOF, redundancy, patterns |
| `pipeline/mod.rs` | 8 tests | Pipeline construction + end-to-end |
| `pipeline/integration_tests.rs` | ~30 tests | Cross-module integration |
| `pipeline/incremental_tests.rs` | ~25 tests | Incremental solving |
| `pipeline/minpack_bridge_tests.rs` | ~29 tests | Legacy Problem bridge |
| `reduce/` | 18 tests | Symbolic reduction |
| `solve/` | 32 tests | ReducedSubProblem, closed-form, branch, drag |
| `sketch2d/` | 62 tests | 16 constraints, 5 entities, builder |
| `sketch3d/` | 21 tests | 8 constraints, 4 entities |
| `assembly/` | 17 tests | RigidBody, Mate, Gear, Insert, Coaxial |

## Workflow Files Overview

| Workflow | Trigger | Purpose | Est. Minutes |
|----------|---------|---------|-------------|
| `ci.yml` | Push/PR to main | Primary testing, lint, format, doc tests | 15-20 |
| `nightly.yml` | Daily schedule | Mutation, fuzzing, sanitizers, coverage | 45-60 |
| `benchmarks.yml` | PR to main | Performance comparison (legacy + V3) | 10-15 |
| `release.yml` | Tag push | Release validation + publish | 10-15 |

## Workflow A: `ci.yml` -- Primary CI

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

permissions:
  contents: read

jobs:
  test:
    name: Test (${{ matrix.os }}, ${{ matrix.rust }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust ${{ matrix.rust }}
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}

      - name: Cache cargo registry and build
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ matrix.rust }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: |
            ${{ runner.os }}-cargo-${{ matrix.rust }}-

      - name: Check (all features)
        run: cargo check --workspace --all-features

      - name: Test (all features)
        run: cargo test --workspace --all-features

      - name: Test (no default features -- V3 modules only)
        run: cargo test -p solverang --no-default-features --features std

      - name: Test (only geometry)
        run: cargo test -p solverang --no-default-features --features std,geometry

      - name: Test (only sparse)
        run: cargo test -p solverang --no-default-features --features std,sparse

      - name: Test (only parallel)
        run: cargo test -p solverang --no-default-features --features std,parallel

      - name: Test (only jit)
        run: cargo test -p solverang --no-default-features --features std,jit

      - name: Test (only macros)
        run: cargo test -p solverang --no-default-features --features std,macros

  # V3 end-to-end tests: build a ConstraintSystem via Sketch2DBuilder, solve, verify
  v3-e2e:
    name: V3 End-to-End Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-v3e2e-${{ hashFiles('**/Cargo.lock') }}

      - name: Run V3 pipeline integration tests
        run: |
          cargo test -p solverang --no-default-features --features std \
            -- pipeline::integration_tests pipeline::incremental_tests pipeline::minpack_bridge_tests

      - name: Run V3 sketch2d tests
        run: |
          cargo test -p solverang --no-default-features --features std \
            -- sketch2d::

      - name: Run V3 sketch3d tests
        run: |
          cargo test -p solverang --no-default-features --features std \
            -- sketch3d::

      - name: Run V3 assembly tests
        run: |
          cargo test -p solverang --no-default-features --features std \
            -- assembly::

      - name: Run V3 graph tests
        run: |
          cargo test -p solverang --no-default-features --features std \
            -- graph::

      - name: Run V3 system tests
        run: |
          cargo test -p solverang --no-default-features --features std \
            -- system::tests

  # Feature combination testing
  feature-combos:
    name: Feature combos (${{ matrix.features }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        features:
          - "std"
          - "std,geometry"
          - "std,sparse"
          - "std,parallel"
          - "std,jit"
          - "std,macros"
          - "std,nist"
          - "std,geometry,sparse"
          - "std,geometry,parallel"
          - "std,geometry,jit,macros"
          - "std,geometry,parallel,sparse,jit,macros,nist"  # all features
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test -p solverang --no-default-features --features ${{ matrix.features }}

  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy

      - name: Cache cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-lint-${{ hashFiles('**/Cargo.lock') }}

      - name: Clippy (all features)
        run: cargo clippy --workspace --all-features -- -D warnings

      - name: Clippy (no default features -- V3 only)
        run: cargo clippy -p solverang --no-default-features --features std -- -D warnings

  format:
    name: Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt

      - name: Check formatting
        run: cargo fmt --all -- --check

  doc:
    name: Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Build docs (check for broken links and warnings)
        env:
          RUSTDOCFLAGS: "-D warnings"
        run: cargo doc --workspace --all-features --no-deps

      - name: Run doc tests (includes V3 doc examples)
        run: cargo test --doc --workspace --all-features

  # Snapshot testing (plan 04) -- fail if snapshots need updating
  snapshot:
    name: Snapshot Tests
    runs-on: ubuntu-latest
    if: false  # Enable when plan 04 is implemented
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run snapshot tests (no auto-update)
        env:
          INSTA_UPDATE: "no"
        run: cargo test -p solverang --all-features -- snapshot

  # Macros crate standalone
  macros:
    name: Macros crate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Test macros crate
        run: cargo test -p solverang_macros
```

## Workflow B: `nightly.yml` -- Nightly Testing

```yaml
name: Nightly

on:
  schedule:
    - cron: '0 4 * * *'  # 4 AM UTC daily
  workflow_dispatch:  # Manual trigger

env:
  CARGO_TERM_COLOR: always

jobs:
  mutation-testing:
    name: Mutation Testing
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-mutants
        run: cargo install cargo-mutants

      - name: Run mutation testing (legacy core modules)
        run: |
          cargo mutants -p solverang \
            --features geometry,parallel,sparse \
            -f crates/solverang/src/solver/newton_raphson.rs \
            -f crates/solverang/src/solver/lm_adapter.rs \
            -f crates/solverang/src/jacobian/numeric.rs \
            -f crates/solverang/src/decomposition.rs \
            --timeout 120 \
            --output mutants-report-legacy

      - name: Run mutation testing (V3 core modules)
        run: |
          cargo mutants -p solverang \
            --no-default-features --features std \
            -f crates/solverang/src/system.rs \
            -f crates/solverang/src/param/store.rs \
            -f crates/solverang/src/dataflow/tracker.rs \
            -f crates/solverang/src/dataflow/cache.rs \
            -f crates/solverang/src/reduce/substitute.rs \
            -f crates/solverang/src/reduce/merge.rs \
            -f crates/solverang/src/reduce/eliminate.rs \
            -f crates/solverang/src/solve/closed_form.rs \
            --timeout 120 \
            --output mutants-report-v3

      - name: Upload mutation reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mutation-reports
          path: |
            mutants-report-legacy/
            mutants-report-v3/

  fuzz-testing:
    name: Fuzz (${{ matrix.target }})
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        target:
          - fuzz_solver_inputs
          - fuzz_jit_compiler
          - fuzz_constraint_system
          - fuzz_v3_pipeline
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly

      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz

      - name: Run fuzz target
        working-directory: crates/solverang
        run: |
          cargo fuzz run ${{ matrix.target }} -- \
            -max_total_time=900 \
            -jobs=2 \
            -workers=2

      - name: Upload crash artifacts
        uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: fuzz-crashes-${{ matrix.target }}
          path: crates/solverang/fuzz/artifacts/

  sanitizers:
    name: Sanitizers
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: rust-src

      - name: ThreadSanitizer
        env:
          RUSTFLAGS: "-Z sanitizer=thread"
        run: |
          cargo +nightly test -p solverang \
            --features parallel,sparse \
            --target x86_64-unknown-linux-gnu \
            -- --test-threads=1 \
            stress_ concurrent_ parallel_

  miri:
    name: Miri
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri

      - name: Run Miri on V3 unit tests (no JIT, no parallel)
        run: |
          cargo +nightly miri test -p solverang \
            --no-default-features --features std \
            -- --test-threads=1 \
            test_empty_system test_add_entity test_solve_single \
            test_alloc_and_get test_param_id_equality \
            new_tracker_has_no_changes new_cache_is_empty \
            test_solver_mapping test_snapshot

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Generate coverage (all features, includes V3 modules)
        run: |
          cargo tarpaulin \
            --workspace \
            --all-features \
            --out xml html \
            --output-dir coverage/

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage/

      - name: Upload to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: coverage/cobertura.xml
          fail_ci_if_error: false

  nightly-rust:
    name: Nightly Rust
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly

      - name: Test with nightly (all features)
        run: cargo test --workspace --all-features
        continue-on-error: true  # Nightly breakage is informational
```

## Workflow C: `benchmarks.yml` -- Performance Tracking

```yaml
name: Benchmarks

on:
  pull_request:
    branches: [main]
    paths:
      - 'crates/solverang/src/**'
      - 'crates/solverang/benches/**'
      - 'crates/solverang/Cargo.toml'

env:
  CARGO_TERM_COLOR: always

jobs:
  benchmark:
    name: Performance Comparison
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-bench-${{ hashFiles('**/Cargo.lock') }}

      - name: Install critcmp
        run: cargo install critcmp

      - name: Benchmark base branch
        run: |
          git checkout ${{ github.event.pull_request.base.sha }}
          cargo bench --features parallel,sparse,jit,geometry -- --save-baseline base

      - name: Benchmark PR
        run: |
          git checkout ${{ github.event.pull_request.head.sha }}
          cargo bench --features parallel,sparse,jit,geometry -- --save-baseline pr

      - name: Compare benchmarks
        id: compare
        run: |
          critcmp base pr > benchmark_diff.txt 2>&1
          cat benchmark_diff.txt
          # Check for significant regressions (>15%)
          if critcmp base pr --threshold 15 2>&1 | grep -q "regressed"; then
            echo "regression=true" >> $GITHUB_OUTPUT
          else
            echo "regression=false" >> $GITHUB_OUTPUT
          fi

      - name: Comment on PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const diff = fs.readFileSync('benchmark_diff.txt', 'utf8');
            const regression = '${{ steps.compare.outputs.regression }}' === 'true';
            const body = `## Benchmark Results\n\n\`\`\`\n${diff}\n\`\`\`\n\n${
              regression
                ? 'WARNING: Performance regression detected. Please investigate.'
                : 'No significant performance regressions.'
            }`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

      - name: Upload benchmark artifacts
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

## Workflow D: `release.yml` -- Release Validation

```yaml
name: Release

on:
  push:
    tags: ['v*']

env:
  CARGO_TERM_COLOR: always

jobs:
  validate:
    name: Validate Release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Full test suite (all features, V3 + legacy)
        run: cargo test --workspace --all-features

      - name: Clippy (strict)
        run: cargo clippy --workspace --all-features -- -D warnings

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Build docs
        env:
          RUSTDOCFLAGS: "-D warnings"
        run: cargo doc --workspace --all-features --no-deps

      - name: Doc tests (includes V3 doc examples)
        run: cargo test --doc --workspace --all-features

      - name: Dry-run publish (macros)
        run: cargo publish --dry-run -p solverang_macros

      - name: Dry-run publish (solverang)
        run: cargo publish --dry-run -p solverang

  cross-platform:
    name: Cross-platform (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Test
        run: cargo test --workspace --all-features

  release:
    name: Create Release
    needs: [validate, cross-platform]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
```

## Cross-Platform Considerations

### Cranelift JIT

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Full support | Primary development platform |
| macOS x86_64 | Full support | Intel Macs |
| macOS aarch64 | Full support | Apple Silicon |
| Windows x86_64 | Full support | MSVC and GNU toolchains |

JIT tests are feature-gated (`#[cfg(feature = "jit")]`) and run on all OS targets
when the feature is enabled.

### V3 Modules -- No Platform-Specific Issues

The V3 modules use only standard Rust (`Vec`, `HashMap`, `HashSet`, `f64`, nalgebra).
They have no platform-specific code, no FFI, no file I/O, and no unsafe code.
Cross-platform testing is primarily needed for:
- nalgebra SVD numerical behavior differences (FMA availability)
- Floating-point precision across different compilers/CPUs

### Platform-Specific Issues

| Issue | Affected OS | Mitigation |
|-------|-------------|------------|
| FP precision differences | All (FMA availability varies) | Round snapshot values to 10 digits |
| Path separators | Windows | Use `std::path` consistently |
| Line endings | Windows | `.gitattributes` with `* text=auto` |
| Signal handling | Windows | Timeout tests may need platform-specific handling |
| Rayon thread count | All (different core counts) | Don't hardcode thread expectations |

### `.gitattributes` (add to repo root)

```
* text=auto
*.rs text eol=lf
*.toml text eol=lf
*.yml text eol=lf
*.md text eol=lf
```

## Feature Matrix Testing

V3 modules are always compiled (no feature gate). Feature matrix testing covers
the optional dependencies:

| Feature | Dependencies | What It Enables |
|---------|-------------|-----------------|
| `std` | (standard library) | Required |
| `parallel` | `rayon` | Legacy parallel decomposition solver |
| `sparse` | `faer` | Legacy sparse solver |
| `jit` | `cranelift`, `cranelift-jit`, `cranelift-module`, `cranelift-native` | JIT compilation |
| `macros` | `solverang_macros` | `#[auto_jacobian]`, `residual!` proc macros |
| `geometry` | (none, cfg-gated module) | Legacy 2D/3D geometry module |
| `nist` | (none, cfg-gated data) | NIST StRD test problems |

Key interaction to test: `--no-default-features --features std` compiles and tests
all V3 modules without any optional dependencies. This is the minimal V3 configuration.

## Coverage Notes

### V3 Module Coverage Targets

The V3 modules have 333 tests across 21k LOC. Coverage should be tracked per module:

| Module | LOC | Tests | Target Coverage |
|--------|-----|-------|----------------|
| `system.rs` | 978 | 16 | >80% |
| `id.rs` | 148 | 4 | >90% |
| `param/store.rs` | 378 | 8 | >85% |
| `dataflow/` | 700 | 25 | >85% |
| `graph/` | 2878 | 46 | >75% |
| `pipeline/` | 5534+2607 | 84 | >70% |
| `reduce/` | 1058 | 18 | >80% |
| `solve/` | 1974 | 32 | >75% |
| `sketch2d/` | 3402 | 62 | >80% |
| `sketch3d/` | 1583 | 21 | >75% |
| `assembly/` | 1422 | 17 | >75% |

### Tarpaulin Configuration

```bash
cargo tarpaulin \
  --workspace \
  --all-features \
  --out xml html \
  --output-dir coverage/ \
  --ignore-tests \
  --skip-clean
```

The `--all-features` flag ensures V3 modules (always compiled) and all optional
modules are covered. Tarpaulin instruments all code paths including the V3 pipeline
phases, closed-form solvers, and sketch builders.

## Caching Strategy

### Cache Key Hierarchy

```yaml
# Level 1: Exact match (same lock file)
key: ${{ runner.os }}-cargo-${{ matrix.rust }}-${{ hashFiles('**/Cargo.lock') }}

# Level 2: Same toolchain
restore-keys: |
  ${{ runner.os }}-cargo-${{ matrix.rust }}-

# Level 3: Same OS
  ${{ runner.os }}-cargo-
```

### What to Cache

```yaml
path: |
  ~/.cargo/registry/index/
  ~/.cargo/registry/cache/
  ~/.cargo/git/db/
  target/
```

### Cache Size Management

The `target/` directory can grow large. Consider caching only `target/release/` for
benchmarks and `target/debug/` for tests. Use `cargo cache --autoclean` periodically.

## Secrets and Permissions

| Workflow | Permissions Needed | Secrets |
|----------|-------------------|---------|
| ci.yml | `contents: read` | None |
| nightly.yml | `contents: read` | Codecov token (optional) |
| benchmarks.yml | `pull-requests: write` (for comments) | None |
| release.yml | `contents: write` | `CARGO_REGISTRY_TOKEN` (for publish) |

## Branch Protection Rules (Recommended)

```
Settings -> Branches -> main:
  [x] Require a pull request before merging
  [x] Require status checks to pass:
    - Test (ubuntu-latest, stable)
    - Test (macos-latest, stable)
    - Test (windows-latest, stable)
    - V3 End-to-End Tests
    - Lint
    - Format
    - Documentation
  [x] Require branches to be up to date
  [x] Do not allow bypassing the above settings
```

## README Badge Integration

```markdown
[![CI](https://github.com/akiselev/solverang/actions/workflows/ci.yml/badge.svg)](https://github.com/akiselev/solverang/actions/workflows/ci.yml)
[![Nightly](https://github.com/akiselev/solverang/actions/workflows/nightly.yml/badge.svg)](https://github.com/akiselev/solverang/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/akiselev/solverang/branch/main/graph/badge.svg)](https://codecov.io/gh/akiselev/solverang)
```

## Estimated CI Minutes

| Workflow | Frequency | Jobs | Est. Minutes/Run | Monthly Minutes |
|----------|-----------|------|-------------------|----------------|
| ci.yml | ~20 PRs/month | 11 (3 OS x 2 Rust + v3-e2e + feature-combos + lint + fmt + doc + macros) | 20 each | ~4,400 |
| nightly.yml | 30/month | 7 (mutation x2 + 4 fuzz + sanitizer + miri + coverage + nightly) | 50 each | ~1,500 |
| benchmarks.yml | ~20 PRs/month | 1 | 15 each | ~300 |
| release.yml | ~2/month | 4 | 15 each | ~30 |
| **Total** | | | | **~6,230** |

GitHub Free tier includes 2,000 minutes/month for private repos (unlimited for public).
For a private repo, consider reducing nightly frequency or feature-combo parallelism.

## Directory Structure

```
.github/
    workflows/
        ci.yml
        nightly.yml
        benchmarks.yml
        release.yml
.gitattributes
```

## Estimated Effort

| Task | Time |
|------|------|
| Write ci.yml (with V3 e2e job, feature combos, doc tests) | 3-4 hours |
| Write nightly.yml (with V3 mutation targets, Miri V3 tests) | 2-3 hours |
| Write benchmarks.yml (with V3 pipeline benchmarks) | 2 hours |
| Write release.yml | 1 hour |
| Test and debug workflows (multiple iterations) | 4-5 hours |
| Add .gitattributes | 15 min |
| Configure branch protection (add V3 e2e as required check) | 30 min |
| Add README badges | 15 min |
| **Total** | **~14-18 hours** |
