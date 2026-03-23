# CLAUDE.md

## Overview

Procedural macro implementation for `#[auto_jacobian]`, `#[auto_diff]`, and `#[hessian]`: symbolic expression parsing, differentiation, simplification, and code generation.

## Index

| File | Contents (WHAT) | Read When (WHEN) |
| --- | --- | --- |
| `lib.rs` | Macro entry points: `auto_jacobian`, `auto_diff`, `residual`, `objective`, `hessian`; orchestration logic for generating `jacobian_entries`, `gradient_entries`, `hessian_entries` | Understanding the macro API, adding a new marker attribute, changing when Hessian generation triggers |
| `expr.rs` | `Expr` AST: `Var`, `Const`, `RuntimeConst`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`, `Sqrt`, `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Sinh`, `Cosh`, `Tanh`, `Ln`, `Exp`, `Abs`, `Atan2`; `differentiate()`, `simplify()`, `to_tokens()`, `collect_variables_into()` | Adding a new math function, fixing a derivative rule, changing simplification |
| `parse.rs` | Converts `syn::Expr` → `Expr` AST; handles method calls (`.sin()`, `.asin()`, `.sinh()`, etc.) and function calls (`f64::sin()`, etc.) | Supporting a new syntax form, debugging parse errors |
| `codegen.rs` | `generate_jacobian_entries`, `generate_jacobian_method`, `generate_hessian_entries`, `generate_hessian_method` — produces `TokenStream` for the generated methods; Hessian entries are lower-triangle only (i >= j) | Changing generated method signatures, fixing Hessian entry ordering, debugging codegen output |
| `codegen_opcodes.rs` | JIT opcode emission: `generate_residual_opcode_method`, `generate_jacobian_opcode_method` | Working with JIT compilation path |
