# Macro Extensions for Optimization Support

## 1. Current Architecture Summary

The `#[auto_jacobian]` proc macro in `crates/macros/src/` currently:

1. **Parses** a `#[residual]`-annotated method's body into an `Expr` AST (`parse.rs`)
2. **Differentiates** the `Expr` once with respect to each variable via `Expr::differentiate(var_id)` (`expr.rs`)
3. **Simplifies** each derivative via `Expr::simplify()` (algebraic identity elimination)
4. **Generates** interpreted Rust code for `jacobian_entries()` (`codegen.rs`)
5. **Generates** JIT opcode-emitting code for `lower_residual_ops` and `lower_jacobian_ops` (`codegen_opcodes.rs`)

The `Expr` enum has 17 variants: `Var`, `Const`, `RuntimeConst`, `Neg`, `Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `Sin`, `Cos`, `Tan`, `Atan2`, `Pow`, `Abs`, `Ln`, `Exp`. The `differentiate()` method implements standard calculus rules (product rule, quotient rule, chain rule) for each variant, returning a new `Expr` tree.

Note: the `#[auto_jacobian]` doc comment in `lib.rs` does not list `ln` and `exp` as supported
operations, but they are fully supported by both `parse.rs` (method syntax `x.ln()`, `x.exp()`)
and `expr.rs` (differentiation rules and `simplify()` constant folding).

Key observation: **second-order differentiation requires no new infrastructure in `expr.rs`**. Calling `differentiate()` on an already-differentiated expression works because the output is the same `Expr` type. The challenge is managing expression tree size.

> **REVIEW NOTE (existing limitation):** The current `#[auto_jacobian]` macro supports only
> **single-residual** functions. `codegen.rs` defines `JacobianInfo { residual_row, entries }`
> and `generate_jacobian_method` accepts `&[JacobianInfo]` (multi-residual scaffolding exists),
> but `lib.rs` `generate_jacobian_impl()` always hardcodes `residual_row: 0`. A user needing
> multiple residuals from one impl block currently must use separate impl blocks or implement
> `jacobian()` manually. This design carries forward into `#[auto_diff]` with `#[residual]`.
> The multi-residual case is deferred but should be documented as a known limitation.

---

## 2. Second-Order Symbolic Differentiation

### 2.1 How It Works

The Hessian of a scalar function `f(x)` is the matrix `H[i,j] = d^2 f / (dx_i dx_j)`. To compute it symbolically:

```
gradient[i] = f.differentiate(var_i).simplify()
hessian[i][j] = gradient[i].differentiate(var_j).simplify()
```

Since `differentiate()` returns an `Expr` and operates on an `Expr`, chained differentiation is already supported. No changes to `differentiate()` itself are needed.

### 2.2 Symmetry Exploitation

The Hessian is symmetric: `H[i,j] = H[j,i]`. We only need to compute the upper triangle where `i <= j`. For `N` variables this reduces from `N^2` to `N*(N+1)/2` entries.

### 2.3 Required Changes to `expr.rs`

No changes to `differentiate()` are needed. However, we should add a helper method:

```rust
impl Expr {
    /// Compute the second derivative d^2(self) / (d var_i d var_j).
    ///
    /// This is equivalent to self.differentiate(var_i).simplify().differentiate(var_j).simplify()
    /// but applies simplification between passes to control expression growth.
    pub fn differentiate2(&self, var_i: usize, var_j: usize) -> Expr {
        let first = self.differentiate(var_i).simplify();
        // Early exit: if the first derivative is constant, second derivative is zero
        if matches!(&first, Expr::Const(_) | Expr::RuntimeConst(_)) {
            return Expr::Const(0.0);
        }
        first.differentiate(var_j).simplify()
    }
}
```

The critical insight is the **intermediate simplification**: simplifying after the first differentiation before applying the second dramatically reduces tree size. Without it, the product/quotient rule expansions compound quadratically.

---

## 3. New Attribute Designs

### 3.1 Attribute Naming: `#[auto_diff]` as Generalization

The current `#[auto_jacobian]` name is constraint-specific. For optimization we propose keeping `#[auto_jacobian]` as-is for backward compatibility and adding a new `#[auto_diff]` attribute that is a superset:

```rust
// Existing (unchanged, backward compatible):
#[auto_jacobian(array_param = "x")]

// New general-purpose attribute:
#[auto_diff(array_param = "x")]
```

`#[auto_diff]` recognizes these inner method attributes:
- `#[residual]` -- generates `jacobian_entries()` (same as today)
- `#[objective]` -- generates `gradient()` only (BFGS default; Hessian opt-in via `#[objective(hessian = "exact")]`)
- `#[inequality]` -- generates residual/Jacobian plus bound metadata

The `#[auto_jacobian]` attribute remains as an alias for `#[auto_diff]` that only recognizes `#[residual]`.

### 3.2 `#[objective]` Attribute

The `#[objective]` attribute marks a method returning `f64` as a scalar objective function to minimize. By default the macro generates only a `gradient()` method; BFGS is the default optimizer and requires only first derivatives. To also generate `hessian_entries()`, use `#[objective(hessian = "exact")]` (recommended only for N≤30).

**Input:**

```rust
#[auto_diff(array_param = "x")]
impl Rosenbrock {
    #[objective]
    fn cost(&self, x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }
}
```

**Generated output for `#[objective]` (gradient only, conceptual):**

```rust
impl Rosenbrock {
    fn cost(&self, x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    /// Gradient of the objective as a dense vector of length N.
    ///
    /// Auto-generated by symbolic differentiation of the `cost` expression.
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        // df/dx[0] = 2*(1.0 - x[0])*(-1.0) + 100.0 * 2*(x[1] - x[0]*x[0])*(-2.0*x[0])
        //          = -2.0*(1.0 - x[0]) - 400.0*x[0]*(x[1] - x[0]*x[0])
        let g0 = ((-2.0) * (1.0 - x[0])) + ((-400.0) * x[0] * (x[1] - x[0] * x[0]));

        // df/dx[1] = 100.0 * 2*(x[1] - x[0]*x[0])
        //          = 200.0*(x[1] - x[0]*x[0])
        let g1 = 200.0 * (x[1] - x[0] * x[0]);

        vec![g0, g1]
    }

    /// Hessian of the objective as sparse upper-triangle triplets (i, j, value) where i <= j.
    ///
    /// Auto-generated by symbolic second-order differentiation.
    /// Only generated when #[objective(hessian = "exact")] is used.
    fn hessian_entries(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(3);

        // d^2f/dx[0]^2 = 2.0 + 800.0*x[0]*x[0] - 400.0*(x[1] - x[0]*x[0])
        //              = 2.0 - 400.0*(x[1] - 3.0*x[0]*x[0])
        entries.push((0, 0, 2.0 + 800.0 * x[0] * x[0] - 400.0 * (x[1] - x[0] * x[0])));

        // d^2f/(dx[0] dx[1]) = -400.0 * x[0]
        entries.push((0, 1, -400.0 * x[0]));

        // d^2f/dx[1]^2 = 200.0
        entries.push((1, 1, 200.0));

        entries
    }
}
```

### 3.3 `#[inequality]` Attribute

Inequality constraints of the form `h(x) <= 0` or `lb <= h(x) <= ub` are common in optimization. The `#[inequality]` attribute generates the constraint value, its Jacobian, and bound metadata. A bare `#[inequality]` with no arguments defaults to `upper = 0.0` (i.e., `h(x) <= 0`), the standard mathematical convention matching the Lagrangian sign convention used throughout this document.

**Input:**

```rust
#[auto_diff(array_param = "x")]
impl CircleConstraint {
    /// x[0]^2 + x[1]^2 <= 1  (must stay inside unit circle)
    #[inequality(upper = 1.0)]
    fn inside_circle(&self, x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    /// 0.5 <= x[0] + x[1] <= 2.0
    #[inequality(lower = 0.5, upper = 2.0)]
    fn sum_bounds(&self, x: &[f64]) -> f64 {
        x[0] + x[1]
    }
}
```

**Generated output:**

```rust
impl CircleConstraint {
    fn inside_circle(&self, x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    fn sum_bounds(&self, x: &[f64]) -> f64 {
        x[0] + x[1]
    }

    /// Constraint bounds: (lower, upper) for each inequality.
    fn constraint_bounds(&self) -> Vec<(f64, f64)> {
        vec![
            (f64::NEG_INFINITY, 1.0),  // inside_circle: g(x) <= 1.0
            (0.5, 2.0),                // sum_bounds: 0.5 <= g(x) <= 2.0
        ]
    }

    /// Evaluate all inequality constraint values.
    fn constraint_values(&self, x: &[f64]) -> Vec<f64> {
        vec![
            self.inside_circle(x),
            self.sum_bounds(x),
        ]
    }

    /// Jacobian of inequality constraints as sparse triplets (row, col, value).
    fn constraint_jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(4);
        // d(inside_circle)/dx[0] = 2*x[0]
        entries.push((0, 0, 2.0 * x[0]));
        // d(inside_circle)/dx[1] = 2*x[1]
        entries.push((0, 1, 2.0 * x[1]));
        // d(sum_bounds)/dx[0] = 1.0
        entries.push((1, 0, 1.0));
        // d(sum_bounds)/dx[1] = 1.0
        entries.push((1, 1, 1.0));
        entries
    }
}
```

> **REVIEW NOTE:** The constraint row index assigned to each `#[inequality]` method is determined
> by declaration order in the impl block. This is an API stability concern: reordering methods,
> or inserting a new `#[inequality]` method before an existing one, silently changes which row
> index each constraint occupies. The generated `constraint_bounds()`, `constraint_values()`, and
> `constraint_jacobian()` must all use the same ordering. Callers who hard-code row indices will
> break silently. Consider documenting method order as a stable API contract, or adding a
> `#[inequality(id = N, ...)]` numeric argument to pin the row index.

<!-- Decision: Bare #[inequality] with neither lower nor upper defaults to upper = 0.0 (i.e., h(x) ≤ 0), the standard mathematical convention. This matches the Lagrangian sign convention used throughout this document suite. -->

### 3.4 Combined Objective + Inequality Example

A realistic optimization problem combines an objective with constraints:

```rust
#[auto_diff(array_param = "x")]
impl ConvexProblem {
    #[objective]
    fn cost(&self, x: &[f64]) -> f64 {
        (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.5) * (x[1] - 2.5)
    }

    #[inequality(upper = 0.0)]
    fn c1(&self, x: &[f64]) -> f64 {
        x[0] - 2.0 * x[1] + 2.0
    }

    #[inequality(upper = 0.0)]
    fn c2(&self, x: &[f64]) -> f64 {
        -x[0] - 2.0 * x[1] + 6.0
    }

    #[inequality(upper = 0.0)]
    fn c3(&self, x: &[f64]) -> f64 {
        -x[0] + 2.0 * x[1] + 2.0
    }
}
```

This generates `gradient()`, `constraint_bounds()`, `constraint_values()`, and `constraint_jacobian()`. To also generate `hessian_entries()`, use `#[objective(hessian = "exact")]`. This provides everything an SQP or interior-point solver needs.

---

## 4. Implementation in `lib.rs`

### 4.1 New Argument Parsing

The `#[auto_diff]` attribute uses the same `JacobianArgs` structure (since `array_param` is the only parameter for now). The new proc macro entry point:

```rust
#[proc_macro_attribute]
pub fn auto_diff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as DiffArgs);
    let impl_block = parse_macro_input!(item as ItemImpl);

    match generate_diff_impl(args, impl_block) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn objective(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item // marker attribute, processed by auto_diff
}

#[proc_macro_attribute]
pub fn inequality(attr: TokenStream, item: TokenStream) -> TokenStream {
    item // marker attribute with bounds, processed by auto_diff
}
```

### 4.2 Method Discovery

The `generate_diff_impl` function scans the impl block for three categories of annotated methods:

```rust
fn generate_diff_impl(args: DiffArgs, mut impl_block: ItemImpl) -> syn::Result<TokenStream2> {
    let mut residual_methods = Vec::new();
    let mut objective_methods = Vec::new();
    let mut inequality_methods = Vec::new();

    for item in &impl_block.items {
        if let ImplItem::Fn(method) = item {
            for attr in &method.attrs {
                if attr.path().is_ident("residual") {
                    residual_methods.push((method.sig.clone(), method.block.clone()));
                } else if attr.path().is_ident("objective") {
                    objective_methods.push((method.sig.clone(), method.block.clone()));
                } else if attr.path().is_ident("inequality") {
                    let bounds = parse_inequality_bounds(attr)?;
                    inequality_methods.push((method.sig.clone(), method.block.clone(), bounds));
                }
            }
        }
    }

    // Generate Jacobian for residuals (existing behavior)
    // Generate gradient + Hessian for objectives
    // Generate constraint value/Jacobian/bounds for inequalities
    // ...
}
```

> **REVIEW NOTE:** Three implementation details require explicit decisions:
>
> 1. **`DiffArgs` vs `JacobianArgs`**: The proposed code uses `parse_macro_input!(attr as DiffArgs)`
>    but `DiffArgs` is never defined. Either rename the existing `JacobianArgs` to `DiffArgs` or
>    create a type alias. Since the only parameter so far is `array_param`, a type alias is sufficient.
>
> 2. **Coexistence of `#[objective]` and `#[residual]`**: If both are present in the same impl block
>    under `#[auto_diff]`, `generate_diff_impl` will generate both `jacobian_entries()` (from the
>    residual) and `gradient()` / `hessian_entries()` (from the objective). This is a legitimate
>    use case for SQP formulations. The implementation must verify there are no method name
>    conflicts. Define this behavior explicitly.
>
> 3. **Attribute stripping**: Like `#[residual]` (which is stripped from the output impl block
>    by `lib.rs` lines 412-418), `#[objective]` and `#[inequality]` must also be stripped from
>    the output. Otherwise `rustc` will error on unknown attributes. This is not shown in the
>    code above and must be implemented.

### 4.3 Inequality Bound Parsing

```rust
struct InequalityBounds {
    lower: Option<f64>,  // None means -infinity
    upper: Option<f64>,  // None means +infinity
}

fn parse_inequality_bounds(attr: &syn::Attribute) -> syn::Result<InequalityBounds> {
    // Parse #[inequality(lower = 0.5, upper = 2.0)]
    // or    #[inequality(upper = 0.0)]
    // ...
}
```

---

## 5. Codegen for Gradient and Hessian

### 5.1 `codegen.rs` Extensions

New functions in `codegen.rs`:

```rust
/// Generate gradient entries for a scalar objective.
///
/// Returns a list of (variable_index_tokens, derivative_tokens) pairs, using
/// the same String-keyed format as `generate_jacobian_entries` for consistency.
/// The index token string is the raw index expression (e.g., "0", "1").
pub fn generate_gradient_entries(
    objective_expr: &Expr,
    variables: &[VarRef],
) -> Vec<(String, TokenStream)> {
    let mut entries = Vec::new();

    for var in variables {
        let derivative = objective_expr.differentiate(var.id).simplify();
        if derivative.is_zero() {
            continue;
        }
        let derivative_tokens = derivative.to_tokens();
        entries.push((var.index_tokens.clone(), derivative_tokens));
    }

    entries
}

/// Generate the gradient method body.
///
/// Produces a dense gradient vector. Zero entries are left as 0.0.
/// `entries` uses the same `(String, TokenStream)` format as `generate_jacobian_entries`.
/// The `String` is the raw index token (e.g., `"0"`, `"1"`) which is parsed to `usize`
/// here for array indexing in the generated code.
pub fn generate_gradient_method(
    entries: &[(String, TokenStream)],
    n_vars: usize,
) -> TokenStream {
    let mut assignments = Vec::new();

    for (col_tokens, deriv_tokens) in entries {
        let col: TokenStream = col_tokens.parse().expect("valid column expression");
        assignments.push(quote! {
            grad[#col] = #deriv_tokens;
        });
    }

    quote! {
        let mut grad = vec![0.0; #n_vars];
        #(#assignments)*
        grad
    }
}

/// Generate Hessian entries as sparse upper-triangle triplets.
///
/// Only computes H[i,j] where i <= j (exploiting symmetry).
/// Returns `(row_tokens, col_tokens, derivative_tokens)` where the token strings
/// are the raw index expressions (consistent with the Jacobian codegen convention).
pub fn generate_hessian_entries(
    objective_expr: &Expr,
    variables: &[VarRef],
) -> Vec<(String, String, TokenStream)> {
    let mut entries = Vec::new();

    for (idx_i, var_i) in variables.iter().enumerate() {
        // First derivative w.r.t. var_i (compute once, reuse for all j >= i)
        let first_deriv = objective_expr.differentiate(var_i.id).simplify();

        // Early exit: if first derivative is constant, all H[i, *] = 0
        if matches!(&first_deriv, Expr::Const(_) | Expr::RuntimeConst(_)) {
            continue;
        }

        for var_j in variables.iter().skip(idx_i) {
            let second_deriv = first_deriv.differentiate(var_j.id).simplify();

            if second_deriv.is_zero() {
                continue;
            }

            let deriv_tokens = second_deriv.to_tokens();
            entries.push((
                var_i.index_tokens.clone(),
                var_j.index_tokens.clone(),
                deriv_tokens,
            ));
        }
    }

    entries
}

/// Generate the hessian_entries method body.
pub fn generate_hessian_method(
    entries: &[(String, String, TokenStream)],
) -> TokenStream {
    let capacity = entries.len();
    let mut pushes = Vec::new();

    for (i_tokens, j_tokens, deriv_tokens) in entries {
        let i: TokenStream = i_tokens.parse().expect("valid row expression");
        let j: TokenStream = j_tokens.parse().expect("valid col expression");
        pushes.push(quote! {
            entries.push((#i, #j, #deriv_tokens));
        });
    }

    quote! {
        let mut entries = Vec::with_capacity(#capacity);
        #(#pushes)*
        entries
    }
}
```

### 5.2 Gradient Code Generation Flow

```
objective method body
  -> collect_let_bindings() + extract_return_expr()
  -> expand_bindings() -- inline all let-bound variables
  -> parse::parse_residual() -- parse into Expr AST
  -> for each variable: expr.differentiate(var_id).simplify()
  -> generate Rust tokens via Expr::to_tokens()
  -> emit gradient() method
```

### 5.3 Hessian Code Generation Flow

```
objective method body
  -> (same parsing as above, producing Expr + variables)
  -> for each variable i:
       first_deriv = expr.differentiate(var_i).simplify()
       if first_deriv is constant: skip row
       for each variable j >= i:
         second_deriv = first_deriv.differentiate(var_j).simplify()
         if second_deriv is zero: skip entry
         emit (i, j, second_deriv.to_tokens())
  -> emit hessian_entries() method
```

---

## 6. Expression Tree Size Analysis

### 6.1 Growth Under Differentiation

Let `|e|` denote the number of nodes in expression `e`. The differentiation rules have the following size growth characteristics:

| Rule | Input Size | Output Size (before simplify) |
|------|-----------|-------------------------------|
| `d(a + b)` | `|a| + |b|` | `|da| + |db|` -- same order |
| `d(a * b)` | `|a| + |b|` | `2*(|a| + |b|) + |da| + |db|` -- product rule duplicates both |
| `d(a / b)` | `|a| + |b|` | `3*(|a| + |b|) + |da| + |db|` -- quotient rule is worse |
| `d(f(g))` | `1 + |g|` | `O(|g|) + |dg|` -- chain rule adds `g` copy |

The product rule is the main amplifier. For an expression that is a product of `k` terms, each differentiation roughly doubles the tree size. After two differentiations:

- **First derivative**: `O(k * |expr|)` -- one term per product-rule expansion
- **Second derivative**: `O(k^2 * |expr|)` -- product rule applied to product-rule output

For an objective function with `N` variables and `T` nodes in the original expression:

| Quantity | Node Count (unsimplified) | After simplification |
|----------|--------------------------|---------------------|
| Original `f` | `T` | `T` |
| Gradient `df/dx_i` | `O(T * P)` where `P` = product depth | `O(T)` typically |
| Full gradient (all `N`) | `O(N * T * P)` | `O(N * T)` |
| Single Hessian entry `H[i,j]` | `O(T * P^2)` | `O(T)` typically |
| Full upper-triangle Hessian | `O(N^2/2 * T * P^2)` | `O(N^2/2 * T)` |

The key observation: **simplification after the first differentiation is essential**. Without it, the product-rule expansions compound and the second differentiation operates on a tree that is already bloated.

### 6.2 Concrete Example: Rosenbrock

For the Rosenbrock function `f(x,y) = (1-x)^2 + 100*(y - x^2)^2`:

- Original expression: ~15 AST nodes
- After expansion (inlining `a`, `b`): ~20 nodes
- `df/dx` (unsimplified): ~80 nodes
- `df/dx` (simplified): ~25 nodes
- `d^2f/dx^2` (from simplified `df/dx`): ~60 nodes
- `d^2f/dx^2` (simplified): ~20 nodes
- `d^2f/dxdy` (from simplified `df/dx`): ~30 nodes
- `d^2f/dxdy` (simplified): ~8 nodes

Without intermediate simplification, `d^2f/dx^2` would be ~400 nodes before simplification.

> **REVIEW NOTE:** These node counts are estimates, not verified measurements. The actual count
> depends on how many times each let-bound variable (`a`, `b`) appears after `expand_bindings`
> inlines them, and on which simplification rules fire. Treat as order-of-magnitude guidance.
> A unit test that calls `node_count()` on the differentiated Rosenbrock expression would
> verify these figures once `node_count()` is implemented.

### 6.3 Worst Cases

Functions involving `sqrt`, `atan2`, and division produce the largest derivative trees because:

- `d(sqrt(e))` introduces a division
- `d(a/b)` (quotient rule) produces `(da*b - a*db) / b^2` which is 3x larger
- Second derivative of quotient involves differentiating a quotient, producing roughly 9x expansion

For a distance constraint like `sqrt(dx^2 + dy^2)`:
- `f` = 7 nodes
- `df/dx` simplified = ~12 nodes
- `d^2f/dx^2` simplified = ~25 nodes
- `d^2f/dx^2` unsimplified (without intermediate simplify) = ~200 nodes

---

## 7. Common Subexpression Elimination (CSE)

### 7.1 Why CSE Matters

The generated Hessian code recomputes shared subexpressions. For example, `d^2f/dx^2` and `d^2f/dxdy` both contain terms involving `df/dx`'s intermediate values. Without CSE, the generated code evaluates `x[1] - x[0]*x[0]` in every Hessian entry.

### 7.2 Compile-Time CSE Strategy

CSE should be performed on the **generated Rust tokens**, not on the `Expr` AST. The simplest approach:

**Phase 1 (immediate, low effort)**: Let the Rust compiler's optimizer handle it. `rustc` with `-O` already performs GVN (Global Value Numbering) and will deduplicate most shared subexpressions. This is the recommended starting approach.

**Phase 2 (if compile times become problematic)**: Implement CSE at the `Expr` level before `to_tokens()`:

```rust
/// A CSE-aware code generator that hashes expression subtrees and emits
/// let-bindings for shared subexpressions.
pub struct CseCodegen {
    /// Map from expression hash to (let-binding identifier, original expr, generated tokens).
    ///
    /// IMPORTANT: The cache stores the original `Expr` alongside the identifier so that
    /// hash collisions between distinct expressions can be detected. On a cache hit,
    /// call `structurally_equal()` to verify identity before reusing. A hash collision
    /// that goes unchecked would silently produce wrong numeric answers in the solver.
    cache: HashMap<u64, (Ident, Expr, TokenStream)>,
    /// Let-binding statements to prepend.
    bindings: Vec<TokenStream>,
    /// Counter for generating unique identifiers.
    counter: usize,
}

impl CseCodegen {
    /// Convert an expression to tokens, introducing let-bindings for
    /// subexpressions that appear more than once.
    pub fn expr_to_tokens(&mut self, expr: &Expr) -> TokenStream {
        let hash = expr.structural_hash();

        if let Some((ident, cached_expr, _)) = self.cache.get(&hash) {
            // Verify structural identity before reusing (guard against hash collisions).
            if cached_expr.structurally_equal(expr) {
                return quote! { #ident };
            }
            // Hash collision: fall through and generate fresh tokens without caching.
        }

        let tokens = self.inner_to_tokens(expr);

        // Only cache non-trivial expressions (not Var or Const)
        if expr.node_count() > 3 {
            let ident = format_ident!("__cse_{}", self.counter);
            self.counter += 1;
            self.bindings.push(quote! { let #ident = #tokens; });
            self.cache.insert(hash, (ident.clone(), expr.clone(), tokens));
            return quote! { #ident };
        }

        tokens
    }
}
```

This requires adding two new methods to `Expr`:

```rust
impl Expr {
    /// Count the number of nodes in this expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) | Expr::RuntimeConst(_) => 1,
            Expr::Neg(e) | Expr::Sqrt(e) | Expr::Sin(e) | Expr::Cos(e)
            | Expr::Tan(e) | Expr::Abs(e) | Expr::Ln(e) | Expr::Exp(e) => {
                1 + e.node_count()
            }
            Expr::Pow(base, _) => 1 + base.node_count(),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Atan2(a, b) => {
                1 + a.node_count() + b.node_count()
            }
        }
    }

    /// Compute a structural hash of this expression for CSE.
    pub fn structural_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.hash_recursive(&mut hasher);
        hasher.finish()
    }

    fn hash_recursive(&self, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        std::mem::discriminant(self).hash(hasher);
        match self {
            Expr::Var(v) => v.id.hash(hasher),
            Expr::Const(v) => v.to_bits().hash(hasher),
            Expr::RuntimeConst(s) => s.hash(hasher),
            Expr::Neg(e) | Expr::Sqrt(e) | Expr::Sin(e) | Expr::Cos(e)
            | Expr::Tan(e) | Expr::Abs(e) | Expr::Ln(e) | Expr::Exp(e) => {
                e.hash_recursive(hasher);
            }
            Expr::Pow(base, exp) => {
                base.hash_recursive(hasher);
                exp.to_bits().hash(hasher);
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Atan2(a, b) => {
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
        }
    }
}
```

### 7.3 Cross-Entry CSE for Hessians

The most impactful CSE opportunity is across Hessian entries. All entries in row `i` of the Hessian share the first derivative `df/dx_i` as a starting point. The code generator should:

1. Compute `gradient[i] = f.differentiate(var_i).simplify()` once
2. For each `j >= i`, compute `gradient[i].differentiate(var_j).simplify()`
3. In the generated code, emit common terms from `gradient[i]` as shared let-bindings

This is naturally achieved by the structure of `generate_hessian_entries()` (Section 5.1), which computes `first_deriv` once per row and reuses it.

### 7.4 Recommended Phasing

1. **Phase 1 (now)**: Rely on `rustc -O` for CSE. Generate straightforward per-entry code.
2. **Phase 2 (when N > 10)**: Implement `CseCodegen` for Hessian entries within the same row.
3. **Phase 3 (when N > 50)**: Implement cross-row CSE and consider flattened SSA-style code generation.

---

## 8. Enhanced Simplification for Second Derivatives

> **REVIEW NOTE:** When implementing `#[objective]`, the macro should emit a compile-time warning
> if the objective expression contains `Abs` (i.e., calls `.abs()` or `f64::abs()`). The derivative
> of `|e|` is `e / |e| * de`, which is `0/0 = NaN` at `e = 0`. This produces a non-finite
> Jacobian/gradient and causes the solver to stop with a generic "non-finite" error rather than
> a diagnostic message about non-differentiability. `00_synthesis.md` Section 2 explicitly calls
> for this warning. Smooth alternatives (Huber loss, `sqrt(e^2 + eps)`) should be documented.

### 8.1 Current Simplification Rules

The existing `simplify()` handles:
- `0 + x = x`, `x + 0 = x`
- `0 * x = 0`, `1 * x = x`, `x * 1 = x`
- `0 / x = 0`, `x / 1 = x`
- `x^0 = 1`, `x^1 = x`
- `-(-x) = x`
- Constant folding for all operations

### 8.2 Additional Rules Needed for Hessians

Second differentiation produces patterns that the current simplifier misses. These rules should be added:

```rust
// In simplify():

// Rule: x - x = 0 (appears from quotient rule second derivatives)
Expr::Sub(a, b) if a.structurally_equal(b) => Expr::Const(0.0),

// Rule: x / x = 1 (appears from chain rule cancellation)
Expr::Div(a, b) if a.structurally_equal(b) => Expr::Const(1.0),

// Rule: x * x = x^2 (normalizes for further simplification)
Expr::Mul(a, b) if a.structurally_equal(b) => Expr::Pow(a, 2.0),

// Rule: x^a * x^b = x^(a+b) (combines power terms from product rule)
Expr::Mul(box Expr::Pow(base1, e1), box Expr::Pow(base2, e2))
    if base1.structurally_equal(base2) => Expr::Pow(base1, e1 + e2),

// Rule: (a * b) + (a * c) = a * (b + c)  [factoring, aggressive]
// This one is more complex and may not be worth the compile-time cost.

// Rule: sin^2(x) + cos^2(x) = 1  [trigonometric identity]
// Only relevant for problems with angular variables.
```

The `structurally_equal` method compares two `Expr` trees for structural identity:

```rust
impl Expr {
    pub fn structurally_equal(&self, other: &Expr) -> bool {
        match (self, other) {
            (Expr::Var(a), Expr::Var(b)) => a.id == b.id,
            (Expr::Const(a), Expr::Const(b)) => a.to_bits() == b.to_bits(),
            (Expr::RuntimeConst(a), Expr::RuntimeConst(b)) => a == b,
            (Expr::Neg(a), Expr::Neg(b)) => a.structurally_equal(b),
            (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
                a1.structurally_equal(b1) && a2.structurally_equal(b2)
            }
            // ... etc for all variants
            _ => false,
        }
    }
}
```

### 8.3 Multi-Pass Simplification

A single simplification pass may not reach a fixed point because one rule's output can enable another. For Hessian expressions, we should apply simplification iteratively:

```rust
impl Expr {
    /// Simplify to a fixed point (up to a maximum number of iterations).
    ///
    /// Requires `node_count()` (see Section 7.2) to be implemented first.
    ///
    /// Note: the termination criterion `node_count() >= prev_count` is
    /// conservative. Some rules (e.g., `x * x -> x^2`) do not reduce node
    /// count, so they will never trigger early exit. The 5-iteration cap is
    /// always sufficient for the patterns produced by first- and second-order
    /// differentiation. If a new simplification rule produces cycles
    /// (extremely unlikely), the cap prevents infinite loops.
    pub fn simplify_full(self) -> Expr {
        let mut expr = self;
        for _ in 0..5 {
            let simplified = expr.clone().simplify();
            if simplified.node_count() >= expr.node_count() {
                // No improvement in node count; stop.
                // Note: equal-size transformations (e.g., x*x -> x^2) do not
                // trigger this break but are still applied in earlier iterations.
                break;
            }
            expr = simplified;
        }
        expr
    }
}
```

For second-order derivatives, the macro should call `simplify_full()` instead of `simplify()`.

> **REVIEW NOTE:** `simplify_full` depends on `node_count()` which does not yet exist in `expr.rs`.
> Both must be added together (Phase 4 in the roadmap). Do not implement `simplify_full` before
> `node_count()` is available.

---

## 9. JIT Opcode Extensions

### 9.1 New `ConstraintOp` Variants

The existing opcode set has `StoreResidual` and `StoreJacobian`/`StoreJacobianIndexed`. For optimization, we add:

```rust
pub enum ConstraintOp {
    // ... existing variants ...

    /// Store a gradient entry.
    ///
    /// Stores the value in register `src` to the gradient array at `var_idx`.
    StoreGradient {
        /// Variable index (column in the gradient vector).
        var_idx: u32,
        /// Source register containing the gradient value.
        src: Reg,
    },

    /// Store a Hessian entry (upper-triangle, sparse).
    ///
    /// Stores a Hessian value at the given output index. The sparsity pattern
    /// (row, col pairs with row <= col) is stored separately.
    StoreHessianIndexed {
        /// Index in the Hessian values array.
        output_idx: u32,
        /// Source register containing the Hessian value.
        src: Reg,
    },

    /// Store an objective function value.
    StoreObjective {
        /// Source register containing the objective value.
        src: Reg,
    },

    /// Store an inequality constraint value.
    StoreConstraint {
        /// Constraint index.
        constraint_idx: u32,
        /// Source register containing the constraint value.
        src: Reg,
    },
}
```

### 9.2 New `OpcodeEmitter` Methods

```rust
impl OpcodeEmitter {
    /// Store a gradient entry.
    pub fn store_gradient(&mut self, var_idx: u32, src: Reg) {
        self.ops.push(ConstraintOp::StoreGradient { var_idx, src });
    }

    /// Store a Hessian entry at a specific output index.
    pub fn store_hessian(&mut self, row: u32, col: u32, src: Reg) {
        let output_idx = self.hessian_entries.len() as u32;
        self.hessian_entries.push(HessianEntry { row, col });
        self.ops.push(ConstraintOp::StoreHessianIndexed { output_idx, src });
    }

    /// Store the objective value.
    pub fn store_objective(&mut self, src: Reg) {
        self.ops.push(ConstraintOp::StoreObjective { src });
    }

    /// Store an inequality constraint value.
    pub fn store_constraint(&mut self, constraint_idx: u32, src: Reg) {
        self.ops.push(ConstraintOp::StoreConstraint { constraint_idx, src });
    }
}
```

### 9.3 Extended `CompiledConstraints`

The `CompiledConstraints` struct (or a new `CompiledOptimization` struct) gains:

```rust
/// Compiled optimization problem ready for JIT.
#[derive(Clone, Debug)]
pub struct CompiledOptimization {
    /// Number of input variables.
    pub n_vars: usize,

    /// Opcode stream for objective evaluation.
    pub objective_ops: Vec<ConstraintOp>,

    /// Opcode stream for gradient evaluation.
    pub gradient_ops: Vec<ConstraintOp>,

    /// Opcode stream for Hessian evaluation.
    pub hessian_ops: Vec<ConstraintOp>,

    /// Number of non-zero Hessian entries.
    pub hessian_nnz: usize,

    /// Hessian sparsity pattern: (row, col) with row <= col.
    pub hessian_pattern: Vec<HessianEntry>,

    /// Number of inequality constraints.
    pub n_constraints: usize,

    /// Opcode stream for inequality constraint evaluation.
    pub constraint_ops: Vec<ConstraintOp>,

    /// Opcode stream for constraint Jacobian evaluation.
    pub constraint_jacobian_ops: Vec<ConstraintOp>,

    /// Constraint Jacobian sparsity pattern.
    pub constraint_jacobian_pattern: Vec<JacobianEntry>,

    /// Constraint bounds: (lower, upper) for each inequality.
    pub constraint_bounds: Vec<(f64, f64)>,

    /// Maximum register index used across all opcode streams.
    pub max_register: u16,
}

/// Hessian entry in COO (coordinate) format, upper triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HessianEntry {
    /// Row index (i <= j).
    pub row: u32,
    /// Column index.
    pub col: u32,
}
```

### 9.4 `codegen_opcodes.rs` Extensions

New functions parallel to the existing `generate_jacobian_opcode_method`.

> **REVIEW NOTE (performance):** The existing `generate_jacobian_opcode_method` in
> `codegen_opcodes.rs` and `generate_jacobian_entries` in `codegen.rs` both independently call
> `differentiate(var.id).simplify()` for each variable. The same symbolic differentiation is
> done twice at macro expansion time when both interpreted and JIT paths are generated. For
> Hessians (N*(N+1)/2 second-order derivatives), this doubles macro execution time. Consider
> computing the derivatives once and passing the results to both code generators, or caching
> the intermediate `Expr` trees.

```rust
/// Generate the `lower_gradient_ops` method body.
pub fn generate_gradient_opcode_method(
    objective_expr: &Expr,
    variables: &[VarRef],
    emitter_ident: &proc_macro2::Ident,
) -> TokenStream {
    let mut stmts = Vec::new();

    for var in variables {
        let derivative = objective_expr.differentiate(var.id).simplify();
        if derivative.is_zero() {
            continue;
        }

        let deriv_tokens = derivative.to_opcode_tokens(emitter_ident);
        let col: TokenStream = var.index_tokens.parse().expect("valid tokens");

        stmts.push(quote! {
            {
                let __deriv = #deriv_tokens;
                #emitter_ident.store_gradient(#col as u32, __deriv);
            }
        });
    }

    quote! { #(#stmts)* }
}

/// Generate the `lower_hessian_ops` method body.
///
/// Only generates upper-triangle entries (row <= col).
pub fn generate_hessian_opcode_method(
    objective_expr: &Expr,
    variables: &[VarRef],
    emitter_ident: &proc_macro2::Ident,
) -> TokenStream {
    let mut stmts = Vec::new();

    for (idx_i, var_i) in variables.iter().enumerate() {
        let first_deriv = objective_expr.differentiate(var_i.id).simplify();
        if matches!(&first_deriv, Expr::Const(_) | Expr::RuntimeConst(_)) {
            continue;
        }

        for var_j in variables.iter().skip(idx_i) {
            let second_deriv = first_deriv.differentiate(var_j.id).simplify();
            if second_deriv.is_zero() {
                continue;
            }

            let deriv_tokens = second_deriv.to_opcode_tokens(emitter_ident);
            let row: TokenStream = var_i.index_tokens.parse().expect("valid tokens");
            let col: TokenStream = var_j.index_tokens.parse().expect("valid tokens");

            stmts.push(quote! {
                {
                    let __deriv = #deriv_tokens;
                    #emitter_ident.store_hessian(#row as u32, #col as u32, __deriv);
                }
            });
        }
    }

    quote! { #(#stmts)* }
}
```

---

## 10. Compile-Time Complexity Analysis

### 10.1 Macro Execution Time

The proc macro's execution time is dominated by symbolic differentiation and simplification. Let `T` = expression tree nodes, `N` = number of variables:

| Operation | Differentiations | Total Nodes Processed |
|-----------|------------------|-----------------------|
| Jacobian (existing) | `N` | `O(N * T)` |
| Gradient | `N` | `O(N * T)` |
| Hessian (upper triangle) | `N + N*(N+1)/2` | `O(N^2 * T)` |

For typical constraint problems:
- `N = 4` (two 2D points): Hessian = 10 second-order differentiations. Negligible.
- `N = 12` (four 3D points): Hessian = 78 second-order differentiations. Still fast.
- `N = 50` (large optimization): Hessian = 1,275 second-order differentiations. May take a few seconds at compile time.
- `N = 200` (very large): Hessian = 20,100 second-order differentiations. Could be slow.

### 10.2 Generated Code Size

Each non-zero Hessian entry generates a code block. The total generated code is:

| Problem Size | Gradient Code | Hessian Code (dense) | Hessian Code (sparse typical) |
|-------------|---------------|----------------------|-------------------------------|
| N = 4 | 4 blocks | 10 blocks | 6-8 blocks |
| N = 12 | 12 blocks | 78 blocks | 30-50 blocks |
| N = 50 | 50 blocks | 1,275 blocks | 200-500 blocks |

Each "block" is 1-5 lines of arithmetic code after simplification. For `N = 50`, the Hessian method body could be 1,000-2,500 lines of generated code. This is within Rust's comfortable range, but the macro itself may take 1-2 seconds to execute.

### 10.3 Mitigation Strategies for Large N

For problems where compile-time symbolic Hessians become impractical:

1. **Finite-difference Hessians**: Provide a `#[objective(hessian = "finite_diff")]` option that generates only the gradient symbolically and computes the Hessian by finite-differencing the gradient at runtime. This is `O(N)` differentiations at compile time instead of `O(N^2)`.

2. **BFGS approximation**: Many optimizers (L-BFGS, SR1) only need the gradient and build an approximate Hessian at runtime. The `#[objective]` attribute could accept `hessian = "none"` to skip Hessian generation entirely.

3. **Lazy Hessian-vector products**: For large-scale optimization, generate a method that computes `H * v` (Hessian-vector product) for a given direction `v`, which requires only `O(N)` second-order differentiations per call.

```rust
#[auto_diff(array_param = "x")]
impl LargeObjective {
    #[objective(hessian = "hvp")]  // Hessian-vector product only
    fn cost(&self, x: &[f64]) -> f64 { ... }
}

// Generates:
impl LargeObjective {
    fn gradient(&self, x: &[f64]) -> Vec<f64> { ... }

    /// Compute H * v (Hessian-vector product) without forming the full Hessian.
    fn hessian_vec_product(&self, x: &[f64], v: &[f64]) -> Vec<f64> {
        // For each variable i, compute:
        //   (H*v)[i] = sum_j H[i,j] * v[j]
        //            = d/dt [ gradient(x + t*v) ] at t=0  (directional derivative)
        // Implemented as: sum over j of (d^2f/dx_i dx_j) * v[j]
        // where each d^2f/dx_i dx_j is symbolically computed
        ...
    }
}
```

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Minimal Viable Optimization Support)

1. Add `#[auto_diff]` and `#[objective]` proc macro attributes to `lib.rs`
2. Add `differentiate2()` and `simplify_full()` to `Expr`
3. Add `generate_gradient_entries()` and `generate_gradient_method()` to `codegen.rs`
4. Add `generate_hessian_entries()` and `generate_hessian_method()` to `codegen.rs`
5. Generate `gradient()` and `hessian_entries()` methods for `#[objective]`

### Phase 2: Inequality Constraints

6. Add `#[inequality]` attribute parsing with bound extraction
7. Generate `constraint_bounds()`, `constraint_values()`, `constraint_jacobian()`
8. Support multiple `#[inequality]` methods in the same impl block

### Phase 3: JIT Integration

9. Add `StoreGradient`, `StoreHessianIndexed`, `StoreObjective`, `StoreConstraint` to `ConstraintOp`
10. Add corresponding `OpcodeEmitter` methods
11. Add `generate_gradient_opcode_method()` and `generate_hessian_opcode_method()` to `codegen_opcodes.rs`
12. Define `CompiledOptimization` struct

### Phase 4: Scalability

13. Add `structurally_equal()` and enhanced simplification rules to `Expr`
14. Implement `node_count()` and `structural_hash()` for CSE readiness
15. Add `CseCodegen` for large Hessians
16. Implement `hessian = "hvp"` mode for large-scale problems
17. Implement `hessian = "finite_diff"` fallback

---

## 12. Summary of File Changes

| File | Changes |
|------|---------|
| `crates/macros/src/lib.rs` | Add `#[auto_diff]`, `#[objective]`, `#[inequality]` proc macro entry points. Add `generate_diff_impl()`. Parse inequality bounds. Generate gradient/hessian/constraint methods. |
| `crates/macros/src/expr.rs` | Add `differentiate2()`, `simplify_full()`, `structurally_equal()`, `node_count()`, `structural_hash()`. Extend `simplify()` with `x-x=0`, `x/x=1` rules. |
| `crates/macros/src/codegen.rs` | Add `generate_gradient_entries()`, `generate_gradient_method()`, `generate_hessian_entries()`, `generate_hessian_method()`. |
| `crates/macros/src/codegen_opcodes.rs` | Add `generate_gradient_opcode_method()`, `generate_hessian_opcode_method()`. |
| `crates/macros/src/parse.rs` | No changes needed. The existing parser handles objectives the same as residuals. |
| `crates/solverang/src/jit/opcodes.rs` | Add `StoreGradient`, `StoreHessianIndexed`, `StoreObjective`, `StoreConstraint` variants. Add `HessianEntry`. Define `CompiledOptimization`. |
| `crates/solverang/src/jit/lower.rs` | Add `store_gradient()`, `store_hessian()`, `store_objective()`, `store_constraint()` to `OpcodeEmitter`. |

The existing `#[auto_jacobian]` attribute and all current constraint code remain unchanged and fully backward compatible.
