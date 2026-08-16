# Solverang in the Resolvent/Sinbad science stack

Solverang remains an independent, batteries-included geometric/general constraint solver and
numerical solver package. Resolvent does **not** replace its public problem, sketch, assembly,
diagnosis, optimization, continuation, or solve APIs.

## Two roles are intentional

```text
                     Resolvent Operator
                            |
                            v
                        Solverang
                 numerical solve algorithms
                            |
                            v
                          Sinbad

and independently:

  Sketch2D / Sketch3D / Assembly / ConstraintSystem
                            |
                  Solverang product API
                            |
             +--------------+--------------+
             |                             |
       numerical solve              optional exact/symbolic
                                          analysis
                                             |
                                          Resolvent
```

The first role makes Solverang one execution engine for compiled scientific operators. The
second is the existing product: users construct geometric and general constraints directly
and receive solving plus diagnosis without knowing anything about Sinbad or PDEs.

## Symbolic export policy

`constraint::symbolic` adds a companion capability rather than modifying `Constraint`'s
runtime contract. Concrete constraints can expose exact residual semantics through a generic
expression builder. A Resolvent adapter implements that builder using Resolvent expression
handles.

Properties:

- no mandatory Resolvent dependency;
- no foreign AST stored inside every constraint;
- no name-string probing or numerical reconstruction of semantics;
- `f64` constants must be lifted exactly as IEEE-754 dyadic values or declined;
- unsupported transcendental/piecewise constraints remain on the numeric path;
- symbolic analysis is allowed to be partial without changing interactive solve behavior.

The initial consumers are diagnosis rather than exact whole-sketch solving:

1. generic Jacobian rank over a finite field;
2. dependency certificates for redundant constraints (`implied_by`);
3. exact polynomial metadata for decomposition/rigidity work;
4. later ideal-membership/inconsistency certificates after proper small-cluster decomposition.

Do **not** schedule Gröbner solving of whole sketches. The tractability boundary is the
constraint decomposition boundary; whole connected components can contain hundreds of
variables even though each individual residual touches only a small local parameter set.

## Ownership

Solverang continues to own:

- `Constraint` and `ConstraintSystem` semantics;
- Sketch2D/Sketch3D and rigid-body assembly convenience APIs;
- parameter/entity IDs and solve lifecycle;
- DOF/redundancy/conflict user-facing diagnosis;
- Newton, LM, trust-region/globalization and sparse orchestration;
- optimization, continuation, event/time integration and warm starts;
- branch behavior and interactive ergonomics.

Resolvent may own/reuse:

- exact scalar/polynomial representations;
- finite-field row reduction and dependency certificates;
- symbolic expression calculus;
- algebraic elimination/ideal machinery;
- formal refinement/certificate vocabulary shared with the broader science stack.

Anvil remains the owner of JIT/code generation and finite-precision executable graph
optimization.

## Migration gate

The existing numeric residual and Jacobian remain authoritative while symbolic coverage grows.
For every symbolic implementation, add differential tests evaluating the symbolic residual and
its derivative at randomized finite points against `Constraint::residuals` and
`Constraint::jacobian`. A symbolic path may not silently change the solved equation merely to
make algebra easier.
