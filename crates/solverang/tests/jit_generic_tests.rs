//! Test suite for generic JIT lowering of arbitrary Problem implementations.
//!
//! This test suite drives the TDD implementation of the hybrid JIT approach:
//! extending `#[auto_jacobian]` to also generate opcode-lowering code so that
//! ANY annotated Problem can be JIT-compiled — not just geometric sketch constraints.
//!
//! ## Test Layers
//!
//! 1. **New Opcodes**: Exp, Ln, Pow, Tan exist in ConstraintOp and OpcodeEmitter
//! 2. **Cranelift Codegen**: New opcodes compile to native code via libm calls
//! 3. **Expr AST**: New Ln/Exp variants + differentiation rules in the macro crate
//! 4. **Round-trip**: JIT residuals/Jacobian match interpreted for all test problems
//! 5. **RuntimeConst**: self.field values baked into JIT code correctly
//! 6. **End-to-end**: PCB-style equations through the full pipeline
//!
//! Run with: `cargo test -p solverang --test jit_generic_tests`

#![cfg(feature = "jit")]
#![cfg(feature = "macros")]

use solverang::{
    auto_jacobian, verify_jacobian, CompiledConstraints, ConstraintOp, JITCompiler, OpcodeEmitter,
    Problem, Reg,
};

// ============================================================================
// LAYER 1: New opcodes exist in ConstraintOp and OpcodeEmitter
// ============================================================================

mod layer1_opcodes {
    use super::*;

    #[test]
    fn emitter_has_exp() {
        let mut e = OpcodeEmitter::new();
        let x = e.load_var(0);
        let r = e.exp(x);
        let ops = e.ops();
        assert!(
            matches!(ops.last(), Some(ConstraintOp::Exp { dst, src }) if *dst == r && *src == x),
            "OpcodeEmitter::exp should emit ConstraintOp::Exp"
        );
    }

    #[test]
    fn emitter_has_ln() {
        let mut e = OpcodeEmitter::new();
        let x = e.load_var(0);
        let r = e.ln(x);
        let ops = e.ops();
        assert!(
            matches!(ops.last(), Some(ConstraintOp::Ln { dst, src }) if *dst == r && *src == x),
            "OpcodeEmitter::ln should emit ConstraintOp::Ln"
        );
    }

    #[test]
    fn emitter_has_pow() {
        let mut e = OpcodeEmitter::new();
        let base = e.load_var(0);
        let exp = e.load_var(1);
        let r = e.pow(base, exp);
        let ops = e.ops();
        assert!(
            matches!(
                ops.last(),
                Some(ConstraintOp::Pow { dst, base: b, exp: ex })
                    if *dst == r && *b == base && *ex == exp
            ),
            "OpcodeEmitter::pow should emit ConstraintOp::Pow"
        );
    }

    #[test]
    fn emitter_has_tan() {
        let mut e = OpcodeEmitter::new();
        let x = e.load_var(0);
        let r = e.tan(x);
        let ops = e.ops();
        assert!(
            matches!(ops.last(), Some(ConstraintOp::Tan { dst, src }) if *dst == r && *src == x),
            "OpcodeEmitter::tan should emit ConstraintOp::Tan"
        );
    }

    #[test]
    fn new_opcodes_register_tracking() {
        // Verify uses_register and defines_register work for new opcodes
        let r0 = Reg::new(0);
        let r1 = Reg::new(1);
        let r2 = Reg::new(2);

        let exp_op = ConstraintOp::Exp {
            dst: r1,
            src: r0,
        };
        assert!(exp_op.uses_register(r0));
        assert!(!exp_op.uses_register(r1));
        assert!(exp_op.defines_register(r1));

        let ln_op = ConstraintOp::Ln {
            dst: r1,
            src: r0,
        };
        assert!(ln_op.uses_register(r0));
        assert!(ln_op.defines_register(r1));

        let pow_op = ConstraintOp::Pow {
            dst: r2,
            base: r0,
            exp: r1,
        };
        assert!(pow_op.uses_register(r0));
        assert!(pow_op.uses_register(r1));
        assert!(pow_op.defines_register(r2));

        let tan_op = ConstraintOp::Tan {
            dst: r1,
            src: r0,
        };
        assert!(tan_op.uses_register(r0));
        assert!(tan_op.defines_register(r1));
    }
}

// ============================================================================
// LAYER 2: Cranelift codegen compiles new opcodes to native code
// ============================================================================

mod layer2_cranelift {
    use super::*;

    /// Helper: build opcodes manually, compile, evaluate, check result.
    fn jit_eval_residual(ops: Vec<ConstraintOp>, n_vars: usize, vars: &[f64]) -> f64 {
        let max_reg = ops
            .iter()
            .filter_map(|op| match op {
                ConstraintOp::LoadVar { dst, .. }
                | ConstraintOp::LoadConst { dst, .. }
                | ConstraintOp::Add { dst, .. }
                | ConstraintOp::Sub { dst, .. }
                | ConstraintOp::Mul { dst, .. }
                | ConstraintOp::Div { dst, .. }
                | ConstraintOp::Neg { dst, .. }
                | ConstraintOp::Sqrt { dst, .. }
                | ConstraintOp::Sin { dst, .. }
                | ConstraintOp::Cos { dst, .. }
                | ConstraintOp::Atan2 { dst, .. }
                | ConstraintOp::Abs { dst, .. }
                | ConstraintOp::Max { dst, .. }
                | ConstraintOp::Min { dst, .. }
                | ConstraintOp::Exp { dst, .. }
                | ConstraintOp::Ln { dst, .. }
                | ConstraintOp::Pow { dst, .. }
                | ConstraintOp::Tan { dst, .. } => Some(dst.index()),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        let mut cc = CompiledConstraints::new(n_vars, 1);
        cc.residual_ops = ops;
        cc.max_register = max_reg;

        let mut compiler = JITCompiler::new().expect("compiler creation failed");
        let jit_fn = compiler.compile(&cc).expect("compilation failed");

        let mut residuals = [0.0];
        jit_fn.evaluate_residuals(vars, &mut residuals);
        residuals[0]
    }

    #[test]
    fn jit_exp() {
        // e^2.0 ≈ 7.389056
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Exp {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[2.0]);
        assert!(
            (result - 2.0_f64.exp()).abs() < 1e-10,
            "exp(2.0) = {}, JIT got {}",
            2.0_f64.exp(),
            result
        );
    }

    #[test]
    fn jit_exp_zero() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Exp {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[0.0]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "exp(0) should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn jit_ln() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Ln {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[std::f64::consts::E]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "ln(e) should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn jit_ln_identity() {
        // ln(exp(3.5)) == 3.5
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Exp {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::Ln {
                dst: Reg::new(2),
                src: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[3.5]);
        assert!(
            (result - 3.5).abs() < 1e-10,
            "ln(exp(3.5)) should be 3.5, got {}",
            result
        );
    }

    #[test]
    fn jit_pow() {
        // 2.0^3.0 = 8.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadConst {
                dst: Reg::new(1),
                value: 3.0,
            },
            ConstraintOp::Pow {
                dst: Reg::new(2),
                base: Reg::new(0),
                exp: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[2.0]);
        assert!(
            (result - 8.0).abs() < 1e-10,
            "2^3 should be 8.0, got {}",
            result
        );
    }

    #[test]
    fn jit_pow_fractional() {
        // 9.0^0.5 = 3.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadConst {
                dst: Reg::new(1),
                value: 0.5,
            },
            ConstraintOp::Pow {
                dst: Reg::new(2),
                base: Reg::new(0),
                exp: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[9.0]);
        assert!(
            (result - 3.0).abs() < 1e-10,
            "9^0.5 should be 3.0, got {}",
            result
        );
    }

    #[test]
    fn jit_tan() {
        // tan(pi/4) = 1.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Tan {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let pi_over_4 = std::f64::consts::FRAC_PI_4;
        let result = jit_eval_residual(ops, 1, &[pi_over_4]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "tan(pi/4) should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn jit_sin_full_range() {
        // This tests that sin uses libm, not Taylor (Taylor fails at large angles)
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Sin {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        // sin(5.0) ≈ -0.9589... — well outside Taylor approximation range
        let result = jit_eval_residual(ops, 1, &[5.0]);
        assert!(
            (result - 5.0_f64.sin()).abs() < 1e-10,
            "sin(5.0) should be {}, got {} (Taylor approx gives wrong answer here)",
            5.0_f64.sin(),
            result
        );
    }

    #[test]
    fn jit_cos_full_range() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Cos {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[5.0]);
        assert!(
            (result - 5.0_f64.cos()).abs() < 1e-10,
            "cos(5.0) should be {}, got {}",
            5.0_f64.cos(),
            result
        );
    }

    #[test]
    fn jit_atan2_full_range() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadVar {
                dst: Reg::new(1),
                var_idx: 1,
            },
            ConstraintOp::Atan2 {
                dst: Reg::new(2),
                y: Reg::new(0),
                x: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        // atan2(-1, -1) = -3*pi/4 ≈ -2.356 — in third quadrant, Taylor approx fails
        let result = jit_eval_residual(ops, 2, &[-1.0, -1.0]);
        let expected = (-1.0_f64).atan2(-1.0);
        assert!(
            (result - expected).abs() < 1e-10,
            "atan2(-1, -1) should be {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn jit_compound_exp_ln_expression() {
        // Compute: ln(x[0]) * exp(x[1]) - should exercise both new opcodes together
        // ln(e) * exp(0) = 1.0 * 1.0 = 1.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadVar {
                dst: Reg::new(1),
                var_idx: 1,
            },
            ConstraintOp::Ln {
                dst: Reg::new(2),
                src: Reg::new(0),
            },
            ConstraintOp::Exp {
                dst: Reg::new(3),
                src: Reg::new(1),
            },
            ConstraintOp::Mul {
                dst: Reg::new(4),
                a: Reg::new(2),
                b: Reg::new(3),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(4),
            },
        ];
        let result = jit_eval_residual(ops, 2, &[std::f64::consts::E, 0.0]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "ln(e) * exp(0) should be 1.0, got {}",
            result
        );
    }
}

// ============================================================================
// LAYER 3: Expr AST has Ln/Exp + correct differentiation
// (tested via macro usage — the macro parses .ln()/.exp() and differentiates)
// ============================================================================

mod layer3_expr_parsing {
    use super::*;

    /// Constraint using natural log: ln(x) - target = 0
    struct LnConstraint {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl LnConstraint {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            x[0].ln() - self.target
        }
    }

    impl Problem for LnConstraint {
        fn name(&self) -> &str {
            "Ln"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor.max(0.1)]
        }
    }

    #[test]
    fn ln_residual_correct() {
        let c = LnConstraint {
            target: 1.0_f64.ln(),
        };
        // ln(1.0) = 0, so residual at x=1 is 0 - 0 = 0
        assert!((c.residual(&[1.0]) - 0.0).abs() < 1e-10);
        // ln(e) = 1.0, target = 0, residual = 1.0
        assert!((c.residual(&[std::f64::consts::E]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ln_jacobian_correct() {
        let c = LnConstraint { target: 0.0 };
        // d/dx(ln(x)) = 1/x
        let x = &[3.0];
        let jac = c.jacobian(x);
        assert_eq!(jac.len(), 1);
        assert!(
            (jac[0].2 - 1.0 / 3.0).abs() < 1e-10,
            "d/dx ln(x) at x=3 should be 1/3, got {}",
            jac[0].2
        );
    }

    #[test]
    fn ln_jacobian_finite_difference() {
        let c = LnConstraint { target: 1.0 };
        let result = verify_jacobian(&c, &[2.5], 1e-7, 1e-5);
        assert!(
            result.passed,
            "Ln Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }

    /// Constraint using exp: exp(x) - target = 0
    struct ExpConstraint {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl ExpConstraint {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            x[0].exp() - self.target
        }
    }

    impl Problem for ExpConstraint {
        fn name(&self) -> &str {
            "Exp"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor]
        }
    }

    #[test]
    fn exp_residual_correct() {
        let c = ExpConstraint { target: 1.0 };
        // exp(0) = 1, residual = 0
        assert!((c.residual(&[0.0])).abs() < 1e-10);
    }

    #[test]
    fn exp_jacobian_correct() {
        let c = ExpConstraint { target: 0.0 };
        // d/dx(exp(x)) = exp(x)
        let x = &[2.0];
        let jac = c.jacobian(x);
        assert_eq!(jac.len(), 1);
        assert!(
            (jac[0].2 - 2.0_f64.exp()).abs() < 1e-10,
            "d/dx exp(x) at x=2 should be exp(2), got {}",
            jac[0].2
        );
    }

    #[test]
    fn exp_jacobian_finite_difference() {
        let c = ExpConstraint { target: 1.0 };
        let result = verify_jacobian(&c, &[1.5], 1e-7, 1e-5);
        assert!(
            result.passed,
            "Exp Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }

    /// Compound: ln(x) * exp(y) - target = 0
    /// Tests chain rule across ln and exp in a multi-variable expression.
    struct LnExpCompound {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl LnExpCompound {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            x[0].ln() * x[1].exp() - self.target
        }
    }

    impl Problem for LnExpCompound {
        fn name(&self) -> &str {
            "LnExpCompound"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, _: f64) -> Vec<f64> {
            vec![1.0, 0.0]
        }
    }

    #[test]
    fn ln_exp_compound_jacobian_finite_difference() {
        let c = LnExpCompound { target: 0.0 };
        // At x=[2.0, 1.0]:
        //   f = ln(2)*exp(1) = 0.6931 * 2.718 = 1.884
        //   df/dx0 = (1/x0) * exp(x1) = 0.5 * e = 1.359
        //   df/dx1 = ln(x0) * exp(x1) = ln(2) * e = 1.884
        let result = verify_jacobian(&c, &[2.0, 1.0], 1e-7, 1e-5);
        assert!(
            result.passed,
            "LnExp compound Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }
}

// ============================================================================
// LAYER 4: JIT round-trip — JIT output matches interpreted for annotated problems
//
// This is the core test: for each problem annotated with #[jit_problem], compile
// it to JIT and verify residuals + Jacobian match the Rust implementation.
// ============================================================================

mod layer4_jit_roundtrip {
    use super::*;

    /// Helper trait that Problems annotated with #[jit_problem] will implement.
    /// This generates the opcode-lowering code alongside the Jacobian.
    ///
    /// For now we test by manually lowering, since the macro extension doesn't
    /// exist yet. These tests will be updated to use the macro once it's built.

    /// Quadratic: x^2 - target = 0
    /// Tests: LoadVar, Mul, LoadConst (RuntimeConst), Sub, StoreResidual
    struct JitQuadratic {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl JitQuadratic {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            x[0] * x[0] - self.target
        }
    }

    impl Problem for JitQuadratic {
        fn name(&self) -> &str {
            "JitQuadratic"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, f: f64) -> Vec<f64> {
            vec![f]
        }
    }

    // These tests manually build opcodes to verify the JIT round-trip concept.
    // Layer 7 tests below use the macro-generated lower_to_compiled_constraints().

    #[test]
    fn quadratic_jit_residual_matches_interpreted() {
        let problem = JitQuadratic { target: 4.0 };

        // Manually emit what the macro SHOULD generate:
        // residual = x[0] * x[0] - self.target
        let mut e = OpcodeEmitter::new();
        let x0 = e.load_var(0);
        let x0_sq = e.mul(x0, x0);
        let target = e.const_f64(problem.target); // RuntimeConst → const_f64(self.target)
        let res = e.sub(x0_sq, target);
        e.store_residual(0, res);

        let mut cc = CompiledConstraints::new(1, 1);
        cc.residual_ops = e.into_ops();
        cc.max_register = 3;

        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        // Test at multiple points
        for &val in &[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 10.0] {
            let vars = [val];
            let interpreted = problem.residuals(&vars);

            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(&vars, &mut jit_result);

            assert!(
                (jit_result[0] - interpreted[0]).abs() < 1e-10,
                "Quadratic residual mismatch at x={}: interpreted={}, jit={}",
                val,
                interpreted[0],
                jit_result[0]
            );
        }
    }

    #[test]
    fn quadratic_jit_jacobian_matches_interpreted() {
        let problem = JitQuadratic { target: 4.0 };

        // Manually emit Jacobian: d/dx(x^2 - target) = 2*x
        let mut e = OpcodeEmitter::new();
        let x0 = e.load_var(0);
        let two = e.const_f64(2.0);
        let deriv = e.mul(two, x0); // simplified: x + x → 2*x
        e.store_jacobian(0, 0, deriv);

        let mut cc = CompiledConstraints::new(1, 1);
        // Need residual ops too (even if empty for this test)
        cc.jacobian_ops = e.ops().to_vec();
        cc.jacobian_pattern = e.jacobian_entries().to_vec();
        cc.jacobian_nnz = cc.jacobian_pattern.len();
        cc.max_register = 2;

        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        for &val in &[-3.0, 0.0, 1.0, 2.0, 5.0] {
            let vars = [val];
            let interpreted = problem.jacobian(&vars);

            let mut jit_values = [0.0];
            jit_fn.evaluate_jacobian(&vars, &mut jit_values);

            let jit_coo = jit_fn.jacobian_to_coo(&jit_values);

            assert_eq!(
                interpreted.len(),
                jit_coo.len(),
                "Jacobian nnz mismatch at x={}",
                val
            );
            assert!(
                (jit_coo[0].2 - interpreted[0].2).abs() < 1e-10,
                "Quadratic Jacobian mismatch at x={}: interpreted={}, jit={}",
                val,
                interpreted[0].2,
                jit_coo[0].2
            );
        }
    }

    /// Distance: sqrt((x2-x0)^2 + (y2-y0)^2) - target = 0
    /// Tests: Sub, Mul, Add, Sqrt, LoadConst, Sub, StoreResidual
    struct JitDistance {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl JitDistance {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            let dx = x[2] - x[0];
            let dy = x[3] - x[1];
            (dx * dx + dy * dy).sqrt() - self.target
        }
    }

    impl Problem for JitDistance {
        fn name(&self) -> &str {
            "JitDistance"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            4
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, _: f64) -> Vec<f64> {
            vec![0.0, 0.0, 1.0, 0.0]
        }
    }

    #[test]
    fn distance_jit_residual_matches_interpreted() {
        let problem = JitDistance { target: 5.0 };

        let mut e = OpcodeEmitter::new();
        let x0 = e.load_var(0);
        let y0 = e.load_var(1);
        let x1 = e.load_var(2);
        let y1 = e.load_var(3);
        let dx = e.sub(x1, x0);
        let dy = e.sub(y1, y0);
        let dx2 = e.mul(dx, dx);
        let dy2 = e.mul(dy, dy);
        let sum = e.add(dx2, dy2);
        let dist = e.sqrt(sum);
        let target = e.const_f64(problem.target);
        let res = e.sub(dist, target);
        e.store_residual(0, res);

        let mut cc = CompiledConstraints::new(4, 1);
        cc.max_register = e.max_register();
        cc.residual_ops = e.into_ops();

        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        let test_points = [
            [0.0, 0.0, 3.0, 4.0],
            [1.0, 2.0, 4.0, 6.0],
            [-1.0, -1.0, 2.0, 3.0],
            [0.0, 0.0, 6.0, 8.0],
        ];

        for vars in &test_points {
            let interpreted = problem.residuals(vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(vars, &mut jit_result);

            assert!(
                (jit_result[0] - interpreted[0]).abs() < 1e-10,
                "Distance residual mismatch at {:?}: interpreted={}, jit={}",
                vars,
                interpreted[0],
                jit_result[0]
            );
        }
    }
}

// ============================================================================
// LAYER 5: RuntimeConst — self.field values correctly baked into JIT
// ============================================================================

mod layer5_runtime_const {
    use super::*;

    /// Two instances with different targets should produce different JIT results.
    struct ScaledConstraint {
        scale: f64,
        offset: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl ScaledConstraint {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            self.scale * x[0] + self.offset
        }
    }

    impl Problem for ScaledConstraint {
        fn name(&self) -> &str {
            "Scaled"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, f: f64) -> Vec<f64> {
            vec![f]
        }
    }

    #[test]
    fn different_instances_different_jit_results() {
        let p1 = ScaledConstraint {
            scale: 2.0,
            offset: 3.0,
        };
        let p2 = ScaledConstraint {
            scale: 10.0,
            offset: -5.0,
        };

        // Build JIT for p1: 2.0 * x + 3.0
        let mut e1 = OpcodeEmitter::new();
        let scale1 = e1.const_f64(p1.scale); // RuntimeConst → LoadConst
        let x1 = e1.load_var(0);
        let prod1 = e1.mul(scale1, x1);
        let offset1 = e1.const_f64(p1.offset);
        let res1 = e1.add(prod1, offset1);
        e1.store_residual(0, res1);

        let mut cc1 = CompiledConstraints::new(1, 1);
        cc1.max_register = e1.max_register();
        cc1.residual_ops = e1.into_ops();

        // Build JIT for p2: 10.0 * x + (-5.0)
        let mut e2 = OpcodeEmitter::new();
        let scale2 = e2.const_f64(p2.scale);
        let x2 = e2.load_var(0);
        let prod2 = e2.mul(scale2, x2);
        let offset2 = e2.const_f64(p2.offset);
        let res2 = e2.add(prod2, offset2);
        e2.store_residual(0, res2);

        let mut cc2 = CompiledConstraints::new(1, 1);
        cc2.max_register = e2.max_register();
        cc2.residual_ops = e2.into_ops();

        let mut compiler1 = JITCompiler::new().unwrap();
        let jit1 = compiler1.compile(&cc1).unwrap();
        let mut compiler2 = JITCompiler::new().unwrap();
        let jit2 = compiler2.compile(&cc2).unwrap();

        let vars = [7.0];

        let mut r1 = [0.0];
        let mut r2 = [0.0];
        jit1.evaluate_residuals(&vars, &mut r1);
        jit2.evaluate_residuals(&vars, &mut r2);

        // p1: 2*7 + 3 = 17
        assert!(
            (r1[0] - 17.0).abs() < 1e-10,
            "p1 JIT should give 17.0, got {}",
            r1[0]
        );
        // p2: 10*7 - 5 = 65
        assert!(
            (r2[0] - 65.0).abs() < 1e-10,
            "p2 JIT should give 65.0, got {}",
            r2[0]
        );

        // Cross-check with interpreted
        assert!((r1[0] - p1.residuals(&vars)[0]).abs() < 1e-10);
        assert!((r2[0] - p2.residuals(&vars)[0]).abs() < 1e-10);
    }
}

// ============================================================================
// LAYER 6: End-to-end PCB-style equations
//
// These represent the actual use case: physics equations from pcb-toolkit
// compiled to JIT and called repeatedly with perturbed inputs.
// ============================================================================

mod layer6_pcb_equations {
    use super::*;

    /// Simplified microstrip impedance: target matching problem.
    ///
    /// Z0 ≈ (87 / sqrt(Er + 1.41)) * ln(5.98 * H / (0.8 * W + T))
    ///
    /// Variables: x[0] = W (trace width)
    /// Constants: H (height), T (thickness), Er, target_zo
    ///
    /// Residual: Z0(W) - target_zo = 0
    struct MicrostripImpedance {
        height: f64,
        thickness: f64,
        er: f64,
        target_zo: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl MicrostripImpedance {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            // Z0 ≈ (87 / sqrt(Er + 1.41)) * ln(5.98 * H / (0.8 * W + T))
            let w = x[0];
            let denom = (self.er + 1.41).sqrt();
            let inner = 5.98 * self.height / (0.8 * w + self.thickness);
            87.0 / denom * inner.ln() - self.target_zo
        }
    }

    impl Problem for MicrostripImpedance {
        fn name(&self) -> &str {
            "MicrostripImpedance"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, _: f64) -> Vec<f64> {
            vec![8.0] // 8 mil trace width initial guess
        }
    }

    #[test]
    fn microstrip_jacobian_finite_difference() {
        let problem = MicrostripImpedance {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zo: 50.0,
        };

        let result = verify_jacobian(&problem, &[8.0], 1e-7, 1e-5);
        assert!(
            result.passed,
            "Microstrip Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }

    #[test]
    fn microstrip_jit_residual_matches_interpreted() {
        let problem = MicrostripImpedance {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zo: 50.0,
        };

        // Manually build the opcode sequence for this residual.
        // This is what the macro SHOULD generate.
        let mut e = OpcodeEmitter::new();

        let w = e.load_var(0);

        // denom = sqrt(er + 1.41)
        let er = e.const_f64(problem.er);
        let c1_41 = e.const_f64(1.41);
        let er_plus = e.add(er, c1_41);
        let denom = e.sqrt(er_plus);

        // inner = 5.98 * height / (0.8 * w + thickness)
        let c5_98 = e.const_f64(5.98);
        let h = e.const_f64(problem.height);
        let numer = e.mul(c5_98, h);
        let c0_8 = e.const_f64(0.8);
        let w_scaled = e.mul(c0_8, w);
        let t = e.const_f64(problem.thickness);
        let inner_denom = e.add(w_scaled, t);
        let inner = e.div(numer, inner_denom);

        // ln(inner)
        let ln_inner = e.ln(inner);

        // 87.0 / denom * ln(inner) - target
        let c87 = e.const_f64(87.0);
        let ratio = e.div(c87, denom);
        let zo = e.mul(ratio, ln_inner);
        let target = e.const_f64(problem.target_zo);
        let res = e.sub(zo, target);
        e.store_residual(0, res);

        let mut cc = CompiledConstraints::new(1, 1);
        cc.max_register = e.max_register();
        cc.residual_ops = e.into_ops();

        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        // Test at many trace widths — this is the autorouter use case
        for w_mil in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0] {
            let vars = [w_mil];
            let interpreted = problem.residuals(&vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(&vars, &mut jit_result);

            assert!(
                (jit_result[0] - interpreted[0]).abs() < 1e-10,
                "Microstrip residual mismatch at W={}: interpreted={}, jit={}",
                w_mil,
                interpreted[0],
                jit_result[0]
            );
        }
    }

    #[test]
    fn microstrip_jit_called_repeatedly() {
        // Simulate autorouter: compile once, call 10000 times
        let problem = MicrostripImpedance {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zo: 50.0,
        };

        let mut e = OpcodeEmitter::new();
        let w = e.load_var(0);
        let er = e.const_f64(problem.er);
        let c1_41 = e.const_f64(1.41);
        let er_plus = e.add(er, c1_41);
        let denom = e.sqrt(er_plus);
        let c5_98 = e.const_f64(5.98);
        let h = e.const_f64(problem.height);
        let numer = e.mul(c5_98, h);
        let c0_8 = e.const_f64(0.8);
        let w_scaled = e.mul(c0_8, w);
        let t = e.const_f64(problem.thickness);
        let inner_denom = e.add(w_scaled, t);
        let inner = e.div(numer, inner_denom);
        let ln_inner = e.ln(inner);
        let c87 = e.const_f64(87.0);
        let ratio = e.div(c87, denom);
        let zo = e.mul(ratio, ln_inner);
        let target = e.const_f64(problem.target_zo);
        let res = e.sub(zo, target);
        e.store_residual(0, res);

        let mut cc = CompiledConstraints::new(1, 1);
        cc.max_register = e.max_register();
        cc.residual_ops = e.into_ops();

        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        // Call 10000 times with varying widths — all should match interpreted
        let mut max_error = 0.0_f64;
        for i in 0..10000 {
            let w = 3.0 + (i as f64) * 0.002; // 3 mil to 23 mil sweep
            let vars = [w];
            let interpreted = problem.residuals(&vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(&vars, &mut jit_result);
            max_error = max_error.max((jit_result[0] - interpreted[0]).abs());
        }

        assert!(
            max_error < 1e-10,
            "Max error over 10000 evaluations: {} (should be < 1e-10)",
            max_error
        );
    }

    /// Skin depth: delta = sqrt(rho / (pi * f * mu_0))
    /// Variables: x[0] = f (frequency)
    /// Tests: Div, Mul, Sqrt with physics constants
    struct SkinDepth {
        resistivity: f64,      // Ohm·m (copper: 1.68e-8)
        target_depth: f64,     // meters
    }

    #[auto_jacobian(array_param = "x")]
    impl SkinDepth {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            let mu_0 = 4.0e-7 * std::f64::consts::PI;
            let delta = (self.resistivity / (std::f64::consts::PI * x[0] * mu_0)).sqrt();
            delta - self.target_depth
        }
    }

    impl Problem for SkinDepth {
        fn name(&self) -> &str {
            "SkinDepth"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, _: f64) -> Vec<f64> {
            vec![1e9]
        }
    }

    #[test]
    fn skin_depth_jacobian_finite_difference() {
        let problem = SkinDepth {
            resistivity: 1.68e-8,
            target_depth: 2.0e-6,
        };
        let result = verify_jacobian(&problem, &[1e9], 1e-2, 1e-4);
        assert!(
            result.passed,
            "Skin depth Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }

    /// Via impedance: Z_via = sqrt(L/C)
    /// L = 5.08 * h * (ln(4*h/d) + 1)  nH
    /// C = 1.41 * Er * h * Dp / (Da - Dp)  pF
    /// Tests ln in a real physics context.
    struct ViaImpedance {
        hole_diameter: f64,
        pad_diameter: f64,
        height: f64,
        er: f64,
        target_z: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl ViaImpedance {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            // x[0] = antipad_diameter (the variable we're solving for)
            let l_nh = 5.08 * self.height * ((4.0 * self.height / self.hole_diameter).ln() + 1.0);
            let c_pf = 1.41 * self.er * self.height * self.pad_diameter
                / (x[0] - self.pad_diameter);
            // Z = sqrt(L_nH / C_pF) * sqrt(1000) to convert nH/pF to Ohms
            let z = (l_nh / c_pf).sqrt() * 1000.0_f64.sqrt();
            z - self.target_z
        }
    }

    impl Problem for ViaImpedance {
        fn name(&self) -> &str {
            "ViaImpedance"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, _: f64) -> Vec<f64> {
            vec![30.0]
        }
    }

    #[test]
    fn via_impedance_jacobian_finite_difference() {
        let problem = ViaImpedance {
            hole_diameter: 10.0,
            pad_diameter: 20.0,
            height: 62.0,
            er: 4.2,
            target_z: 50.0,
        };
        let result = verify_jacobian(&problem, &[30.0], 1e-7, 1e-5);
        assert!(
            result.passed,
            "Via impedance Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }

    /// Multi-variable: differential impedance
    /// Zdiff = 2 * Z0 * (1 - 0.48 * exp(-0.96 * S / H))
    /// Variables: x[0] = W (width), x[1] = S (spacing)
    /// Tests exp in a multi-variable physics context.
    struct DiffImpedance {
        height: f64,
        thickness: f64,
        er: f64,
        target_zdiff: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl DiffImpedance {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            let w = x[0];
            let s = x[1];
            // Simplified Z0 for single-ended
            let z0 = 87.0 / (self.er + 1.41).sqrt()
                * (5.98 * self.height / (0.8 * w + self.thickness)).ln();
            // Differential coupling correction
            let zdiff = 2.0 * z0 * (1.0 - 0.48 * (-0.96 * s / self.height).exp());
            zdiff - self.target_zdiff
        }
    }

    impl Problem for DiffImpedance {
        fn name(&self) -> &str {
            "DiffImpedance"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![self.residual(x)]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.jacobian_entries(x)
        }
        fn initial_point(&self, _: f64) -> Vec<f64> {
            vec![5.0, 5.0]
        }
    }

    #[test]
    fn diff_impedance_jacobian_finite_difference() {
        let problem = DiffImpedance {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zdiff: 100.0,
        };
        let result = verify_jacobian(&problem, &[5.0, 5.0], 1e-7, 1e-4);
        assert!(
            result.passed,
            "Differential impedance Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }
}

// ============================================================================
// LAYER 7: Auto-generated JIT lowering via #[auto_jacobian] macro
//
// The macro now generates lower_residual_ops, lower_jacobian_ops, and
// lower_to_compiled_constraints methods behind #[cfg(feature = "jit")].
// ============================================================================

mod layer7_auto_lowerable {
    use super::*;

    /// Quadratic: x^2 - target. Simplest possible test.
    struct AutoQuadratic {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl AutoQuadratic {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            x[0] * x[0] - self.target
        }
    }

    impl Problem for AutoQuadratic {
        fn name(&self) -> &str { "AutoQuadratic" }
        fn residual_count(&self) -> usize { 1 }
        fn variable_count(&self) -> usize { 1 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
        fn initial_point(&self, f: f64) -> Vec<f64> { vec![f] }
    }

    #[test]
    fn auto_quadratic_jit_residual_matches_interpreted() {
        let problem = AutoQuadratic { target: 4.0 };
        let cc = problem.lower_to_compiled_constraints();
        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        for &val in &[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 10.0] {
            let vars = [val];
            let interpreted = problem.residuals(&vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(&vars, &mut jit_result);
            assert!(
                (jit_result[0] - interpreted[0]).abs() < 1e-10,
                "Auto quadratic mismatch at x={}: interpreted={}, jit={}",
                val, interpreted[0], jit_result[0]
            );
        }
    }

    #[test]
    fn auto_quadratic_jit_jacobian_matches_interpreted() {
        let problem = AutoQuadratic { target: 4.0 };
        let cc = problem.lower_to_compiled_constraints();
        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        for &val in &[-3.0, 0.0, 1.0, 2.0, 5.0] {
            let vars = [val];
            let interpreted = problem.jacobian(&vars);
            let mut jit_values = vec![0.0; jit_fn.jacobian_nnz()];
            jit_fn.evaluate_jacobian(&vars, &mut jit_values);
            let jit_coo = jit_fn.jacobian_to_coo(&jit_values);

            assert_eq!(interpreted.len(), jit_coo.len(),
                "Jacobian nnz mismatch at x={}", val);
            for (interp, jit) in interpreted.iter().zip(jit_coo.iter()) {
                assert_eq!(interp.0, jit.0, "row mismatch");
                assert_eq!(interp.1, jit.1, "col mismatch");
                assert!(
                    (interp.2 - jit.2).abs() < 1e-10,
                    "Jacobian value mismatch at x={}: interpreted={}, jit={}",
                    val, interp.2, jit.2
                );
            }
        }
    }

    /// Distance: sqrt((x2-x0)^2 + (y2-y0)^2) - target. Multi-variable.
    struct AutoDistance {
        target: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl AutoDistance {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            let dx = x[2] - x[0];
            let dy = x[3] - x[1];
            (dx * dx + dy * dy).sqrt() - self.target
        }
    }

    impl Problem for AutoDistance {
        fn name(&self) -> &str { "AutoDistance" }
        fn residual_count(&self) -> usize { 1 }
        fn variable_count(&self) -> usize { 4 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0, 1.0, 0.0] }
    }

    #[test]
    fn auto_distance_jit_matches_interpreted() {
        let problem = AutoDistance { target: 5.0 };
        let cc = problem.lower_to_compiled_constraints();
        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        let test_points = [
            [0.0, 0.0, 3.0, 4.0],
            [1.0, 2.0, 4.0, 6.0],
            [-1.0, -1.0, 2.0, 3.0],
        ];

        for vars in &test_points {
            let interpreted = problem.residuals(vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(vars, &mut jit_result);
            assert!(
                (jit_result[0] - interpreted[0]).abs() < 1e-10,
                "Auto distance mismatch at {:?}: interpreted={}, jit={}",
                vars, interpreted[0], jit_result[0]
            );
        }
    }

    /// Microstrip impedance with ln — the PCB use case, fully auto-generated.
    struct AutoMicrostrip {
        height: f64,
        thickness: f64,
        er: f64,
        target_zo: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl AutoMicrostrip {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            let w = x[0];
            let denom = (self.er + 1.41).sqrt();
            let inner = 5.98 * self.height / (0.8 * w + self.thickness);
            87.0 / denom * inner.ln() - self.target_zo
        }
    }

    impl Problem for AutoMicrostrip {
        fn name(&self) -> &str { "AutoMicrostrip" }
        fn residual_count(&self) -> usize { 1 }
        fn variable_count(&self) -> usize { 1 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![8.0] }
    }

    #[test]
    fn auto_microstrip_jit_matches_interpreted() {
        let problem = AutoMicrostrip {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zo: 50.0,
        };
        let cc = problem.lower_to_compiled_constraints();
        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        // Sweep trace widths — the autorouter use case
        let mut max_error = 0.0_f64;
        for i in 0..1000 {
            let w = 3.0 + (i as f64) * 0.02;
            let vars = [w];
            let interpreted = problem.residuals(&vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(&vars, &mut jit_result);
            max_error = max_error.max((jit_result[0] - interpreted[0]).abs());
        }
        assert!(
            max_error < 1e-10,
            "Auto microstrip max error over 1000 evaluations: {}",
            max_error
        );
    }

    #[test]
    fn auto_microstrip_jit_jacobian_matches_interpreted() {
        let problem = AutoMicrostrip {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zo: 50.0,
        };
        let cc = problem.lower_to_compiled_constraints();
        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        for w in [3.0, 5.0, 8.0, 12.0, 20.0] {
            let vars = [w];
            let interpreted = problem.jacobian(&vars);
            let mut jit_values = vec![0.0; jit_fn.jacobian_nnz()];
            jit_fn.evaluate_jacobian(&vars, &mut jit_values);
            let jit_coo = jit_fn.jacobian_to_coo(&jit_values);

            assert_eq!(interpreted.len(), jit_coo.len());
            for (interp, jit) in interpreted.iter().zip(jit_coo.iter()) {
                assert!(
                    (interp.2 - jit.2).abs() < 1e-8,
                    "Microstrip Jacobian mismatch at W={}: interpreted={}, jit={}",
                    w, interp.2, jit.2
                );
            }
        }
    }

    /// Differential impedance with exp — multi-variable PCB equation.
    struct AutoDiffImpedance {
        height: f64,
        thickness: f64,
        er: f64,
        target_zdiff: f64,
    }

    #[auto_jacobian(array_param = "x")]
    impl AutoDiffImpedance {
        #[residual]
        fn residual(&self, x: &[f64]) -> f64 {
            let w = x[0];
            let s = x[1];
            let z0 = 87.0 / (self.er + 1.41).sqrt()
                * (5.98 * self.height / (0.8 * w + self.thickness)).ln();
            let zdiff = 2.0 * z0 * (1.0 - 0.48 * (-0.96 * s / self.height).exp());
            zdiff - self.target_zdiff
        }
    }

    impl Problem for AutoDiffImpedance {
        fn name(&self) -> &str { "AutoDiffImpedance" }
        fn residual_count(&self) -> usize { 1 }
        fn variable_count(&self) -> usize { 2 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![self.residual(x)] }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { self.jacobian_entries(x) }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![5.0, 5.0] }
    }

    #[test]
    fn auto_diff_impedance_jit_matches_interpreted() {
        let problem = AutoDiffImpedance {
            height: 4.0,
            thickness: 1.4,
            er: 4.2,
            target_zdiff: 100.0,
        };
        let cc = problem.lower_to_compiled_constraints();
        let mut compiler = JITCompiler::new().unwrap();
        let jit_fn = compiler.compile(&cc).unwrap();

        let test_points = [
            [4.0, 4.0], [5.0, 5.0], [6.0, 8.0], [8.0, 3.0], [10.0, 10.0],
        ];

        for vars in &test_points {
            let interpreted = problem.residuals(vars);
            let mut jit_result = [0.0];
            jit_fn.evaluate_residuals(vars, &mut jit_result);
            assert!(
                (jit_result[0] - interpreted[0]).abs() < 1e-10,
                "Auto diff impedance mismatch at {:?}: interpreted={}, jit={}",
                vars, interpreted[0], jit_result[0]
            );
        }
    }

    /// Two instances with different constants produce different JIT results.
    #[test]
    fn different_instances_produce_different_jit() {
        let p1 = AutoQuadratic { target: 4.0 };
        let p2 = AutoQuadratic { target: 100.0 };

        let cc1 = p1.lower_to_compiled_constraints();
        let cc2 = p2.lower_to_compiled_constraints();

        let mut c1 = JITCompiler::new().unwrap();
        let mut c2 = JITCompiler::new().unwrap();
        let jit1 = c1.compile(&cc1).unwrap();
        let jit2 = c2.compile(&cc2).unwrap();

        let vars = [5.0];
        let mut r1 = [0.0];
        let mut r2 = [0.0];
        jit1.evaluate_residuals(&vars, &mut r1);
        jit2.evaluate_residuals(&vars, &mut r2);

        // 5^2 - 4 = 21, 5^2 - 100 = -75
        assert!((r1[0] - 21.0).abs() < 1e-10, "p1: expected 21, got {}", r1[0]);
        assert!((r2[0] - (-75.0)).abs() < 1e-10, "p2: expected -75, got {}", r2[0]);
    }
}
