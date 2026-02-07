# Plan 11: CI GitHub Actions — Cross-Platform Testing

## Goal

Create comprehensive CI/CD pipelines using GitHub Actions to:
- Test across **Linux, macOS, and Windows**
- Test all feature combinations
- Run linting and formatting checks
- Integrate nightly testing (mutation, fuzzing, sanitizers)
- Track benchmark performance
- Automate releases

Currently, **no CI/CD exists** — no `.github/workflows/` directory.

## Workflow Files Overview

| Workflow | Trigger | Purpose | Est. Minutes |
|----------|---------|---------|-------------|
| `ci.yml` | Push/PR to main | Primary testing, lint, format | 15-20 |
| `nightly.yml` | Daily schedule | Mutation, fuzzing, sanitizers, coverage | 45-60 |
| `benchmarks.yml` | PR to main | Performance comparison | 10-15 |
| `release.yml` | Tag push | Release validation + publish | 10-15 |

## Workflow A: `ci.yml` — Primary CI

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

      - name: Test (no default features)
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

      - name: Clippy (no default features)
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

      - name: Build docs
        env:
          RUSTDOCFLAGS: "-D warnings"
        run: cargo doc --workspace --all-features --no-deps

  # Verify that the macros crate compiles independently
  macros:
    name: Macros crate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Test macros crate
        run: cargo test -p solverang_macros
```

## Workflow B: `nightly.yml` — Nightly Testing

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

      - name: Run mutation testing (core modules)
        run: |
          cargo mutants -p solverang \
            --features geometry,parallel,sparse \
            -f crates/solverang/src/solver/newton_raphson.rs \
            -f crates/solverang/src/solver/lm_adapter.rs \
            -f crates/solverang/src/jacobian/numeric.rs \
            -f crates/solverang/src/decomposition.rs \
            --timeout 120 \
            --output mutants-report

      - name: Upload mutation report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mutation-report
          path: mutants-report/

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

      - name: Run Miri (no JIT, no parallel)
        run: |
          cargo +nightly miri test -p solverang \
            --no-default-features --features std,geometry \
            -- --test-threads=1

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Generate coverage
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

      - name: Test with nightly
        run: cargo test --workspace --all-features
        continue-on-error: true  # Nightly breakage is informational
```

## Workflow C: `benchmarks.yml` — Performance Tracking

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
            const body = `## Benchmark Results\n\n\`\`\`\n${diff}\n\`\`\`\n\n${
              '${{ steps.compare.outputs.regression }}' === 'true'
                ? '⚠️ **Performance regression detected!** Please investigate.'
                : '✅ No significant performance regressions.'
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

## Workflow D: `release.yml` — Release Validation

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

      - name: Full test suite
        run: cargo test --workspace --all-features

      - name: Clippy (strict)
        run: cargo clippy --workspace --all-features -- -D warnings

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Build docs
        env:
          RUSTDOCFLAGS: "-D warnings"
        run: cargo doc --workspace --all-features --no-deps

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

Cranelift supports all major platforms, but JIT tests should be feature-gated and
run on all three OS targets.

### Platform-specific issues

| Issue | Affected OS | Mitigation |
|-------|-------------|------------|
| FP precision differences | All (FMA availability varies) | Round snapshot values to 10 digits |
| Path separators | Windows | Use `std::path` consistently, not string literals |
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

Test these specific combinations to catch feature interaction bugs:

```yaml
# In ci.yml, add a feature-combination job
feature-combos:
  name: Feature combinations
  runs-on: ubuntu-latest
  strategy:
    matrix:
      features:
        - "std"
        - "std,geometry"
        - "std,sparse"
        - "std,parallel"
        - "std,jit"
        - "std,macros"
        - "std,geometry,sparse"
        - "std,geometry,parallel"
        - "std,geometry,jit,macros"
        - "std,geometry,parallel,sparse,jit,macros,nist"  # all features
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo test -p solverang --no-default-features --features ${{ matrix.features }}
```

## Caching Strategy

### Cache key hierarchy

```yaml
# Level 1: Exact match (same lock file)
key: ${{ runner.os }}-cargo-${{ matrix.rust }}-${{ hashFiles('**/Cargo.lock') }}

# Level 2: Same toolchain
restore-keys: |
  ${{ runner.os }}-cargo-${{ matrix.rust }}-

# Level 3: Same OS
  ${{ runner.os }}-cargo-
```

### What to cache

```yaml
path: |
  ~/.cargo/registry/index/
  ~/.cargo/registry/cache/
  ~/.cargo/git/db/
  target/
```

### Cache size management

- The `target/` directory can grow large. Consider caching only `target/release/` for
  benchmarks and `target/debug/` for tests.
- Use `cargo cache --autoclean` periodically (via `cargo-cache` tool).

## Secrets and Permissions

| Workflow | Permissions Needed | Secrets |
|----------|-------------------|---------|
| ci.yml | `contents: read` | None |
| nightly.yml | `contents: read` | Codecov token (optional) |
| benchmarks.yml | `pull-requests: write` (for comments) | None |
| release.yml | `contents: write` | `CARGO_REGISTRY_TOKEN` (for publish) |

```yaml
# Minimal permissions (add to each workflow)
permissions:
  contents: read
```

## Branch Protection Rules (Recommended)

```
Settings → Branches → main:
  ✅ Require a pull request before merging
  ✅ Require status checks to pass:
    - Test (ubuntu-latest, stable)
    - Test (macos-latest, stable)
    - Test (windows-latest, stable)
    - Lint
    - Format
  ✅ Require branches to be up to date
  ✅ Do not allow bypassing the above settings
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
| ci.yml | ~20 PRs/month | 8 (3 OS × 2 Rust + lint + fmt) | 15 each | ~2,400 |
| nightly.yml | 30/month | 5 (mutation + 3 fuzz + sanitizer) | 45 each | ~1,350 |
| benchmarks.yml | ~20 PRs/month | 1 | 15 each | ~300 |
| release.yml | ~2/month | 4 | 15 each | ~30 |
| **Total** | | | | **~4,080** |

GitHub Free tier includes 2,000 minutes/month for private repos (unlimited for public).
This may require a paid plan for private repos, or reducing nightly frequency.

## Directory Structure

```
.github/
└── workflows/
    ├── ci.yml
    ├── nightly.yml
    ├── benchmarks.yml
    └── release.yml
.gitattributes
```

## Estimated Effort

| Task | Time |
|------|------|
| Write ci.yml | 2-3 hours |
| Write nightly.yml | 2-3 hours |
| Write benchmarks.yml | 2 hours |
| Write release.yml | 1 hour |
| Test and debug workflows | 3-4 hours |
| Add .gitattributes | 15 min |
| Configure branch protection | 30 min |
| Add README badges | 15 min |
| **Total** | **~12-15 hours** |
