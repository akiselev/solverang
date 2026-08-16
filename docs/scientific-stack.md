# Scientific-stack boundary

Solverang is the numerical-algorithm product. `solverang-contracts` now owns the small portable
operator/DAE seam previously imported from an absolute Sinbad checkout. This removes the
packaging cycle that prevented clean CI while preserving Solverang's high-level geometric,
nonlinear, optimization and integration APIs.

Resolvent may lower a mathematical `OperatorProgram` to an implementation of these contracts.
Sinbad chooses solve/integration policy and supplies the compiled operator. Solverang never
imports a physics assembler.

The `jit` feature retains the historical `crate::jit` facade as a compatibility path, but it
is optional and fetched from a fixed Sinbad revision instead of `/home/dev/sinbad`. New
solver code should consume `ExecutableResidual`; the long-term Anvil packaging target is a
standalone execution-compiler package, not a reverse dependency from Solverang into Sinbad.
