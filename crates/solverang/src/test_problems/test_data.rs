//! MINPACK test data and test case definitions.
//!
//! Contains the test case specifications from MINPACK test suites.

use crate::Problem;
use super::*;

/// Expected INFO value from original MINPACK tests.
/// INFO = 1 means normal convergence.
pub const INFO_ORIGINAL: i32 = 1;

/// Identifier for a test case
#[derive(Clone, Debug, PartialEq)]
pub struct TestCaseId {
    /// Problem number (1-based)
    pub problem_number: usize,
    /// Number of variables
    pub n: usize,
    /// Number of equations
    pub m: usize,
    /// Starting point factor
    pub factor: f64,
}

/// A least-squares test case specification (from test_lmder.f90)
#[derive(Clone, Debug)]
pub struct LeastSquaresTestCase {
    /// Test case ID
    pub id: TestCaseId,
    /// Problem name
    pub name: &'static str,
    /// Expected final residual sum of squares (FNORM2)
    pub expected_fnorm2: f64,
    /// Expected INFO value from original MINPACK
    pub expected_info: i32,
}

/// A nonlinear equation test case specification (from test_hybrj.f90)
#[derive(Clone, Debug)]
pub struct NonlinearTestCase {
    /// Test case ID
    pub id: TestCaseId,
    /// Problem name
    pub name: &'static str,
    /// Expected final residual norm (FNORM)
    pub expected_fnorm: f64,
    /// Expected INFO value from original MINPACK
    pub expected_info: i32,
}

/// Get all 53 least-squares test cases from MINPACK test_lmder.f90
pub fn all_least_squares_test_cases() -> Vec<LeastSquaresTestCase> {
    vec![
        // Linear Full Rank (Problem 1)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 1, n: 5, m: 10, factor: 1.0 }, name: "Linear Full Rank", expected_fnorm2: 0.5000000000E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 1, n: 5, m: 50, factor: 1.0 }, name: "Linear Full Rank", expected_fnorm2: 0.5000000000E+01, expected_info: 1 },
        // Linear Rank 1 (Problem 2)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 2, n: 5, m: 10, factor: 1.0 }, name: "Linear Rank 1", expected_fnorm2: 0.2142857143E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 2, n: 5, m: 50, factor: 1.0 }, name: "Linear Rank 1", expected_fnorm2: 0.2420995001E+01, expected_info: 1 },
        // Linear Rank 1 Zero Columns (Problem 3)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 3, n: 5, m: 10, factor: 1.0 }, name: "Linear Rank 1 Zero Columns", expected_fnorm2: 0.2000000000E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 3, n: 5, m: 50, factor: 1.0 }, name: "Linear Rank 1 Zero Columns", expected_fnorm2: 0.2000000000E+01, expected_info: 1 },
        // Rosenbrock (Problem 4)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 4, n: 2, m: 2, factor: 1.0 }, name: "Rosenbrock", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 4, n: 2, m: 2, factor: 10.0 }, name: "Rosenbrock", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 4, n: 2, m: 2, factor: 100.0 }, name: "Rosenbrock", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        // Helical Valley (Problem 5)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 5, n: 3, m: 3, factor: 1.0 }, name: "Helical Valley", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 5, n: 3, m: 3, factor: 10.0 }, name: "Helical Valley", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 5, n: 3, m: 3, factor: 100.0 }, name: "Helical Valley", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        // Powell Singular (Problem 6)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 6, n: 4, m: 4, factor: 1.0 }, name: "Powell Singular", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 6, n: 4, m: 4, factor: 10.0 }, name: "Powell Singular", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 6, n: 4, m: 4, factor: 100.0 }, name: "Powell Singular", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        // Freudenstein-Roth (Problem 7)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 7, n: 2, m: 2, factor: 1.0 }, name: "Freudenstein-Roth", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 7, n: 2, m: 2, factor: 10.0 }, name: "Freudenstein-Roth", expected_fnorm2: 0.4898979486E+02, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 7, n: 2, m: 2, factor: 100.0 }, name: "Freudenstein-Roth", expected_fnorm2: 0.4898979486E+02, expected_info: 1 },
        // Bard (Problem 8)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 8, n: 3, m: 15, factor: 1.0 }, name: "Bard", expected_fnorm2: 0.8214877306E-02, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 8, n: 3, m: 15, factor: 10.0 }, name: "Bard", expected_fnorm2: 0.8214877306E-02, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 8, n: 3, m: 15, factor: 100.0 }, name: "Bard", expected_fnorm2: 0.8214877306E-02, expected_info: 1 },
        // Kowalik-Osborne (Problem 9)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 9, n: 4, m: 11, factor: 1.0 }, name: "Kowalik-Osborne", expected_fnorm2: 0.3075056038E-03, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 9, n: 4, m: 11, factor: 10.0 }, name: "Kowalik-Osborne", expected_fnorm2: 0.3075056038E-03, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 9, n: 4, m: 11, factor: 100.0 }, name: "Kowalik-Osborne", expected_fnorm2: 0.3075056038E-03, expected_info: 1 },
        // Meyer (Problem 10)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 10, n: 3, m: 16, factor: 1.0 }, name: "Meyer", expected_fnorm2: 0.8794585517E+02, expected_info: 1 },
        // Watson (Problem 11)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 11, n: 6, m: 31, factor: 1.0 }, name: "Watson", expected_fnorm2: 0.2287676763E-02, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 11, n: 9, m: 31, factor: 1.0 }, name: "Watson", expected_fnorm2: 0.1175695682E-05, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 11, n: 12, m: 31, factor: 1.0 }, name: "Watson", expected_fnorm2: 0.2170000409E-07, expected_info: 1 },
        // Box3D (Problem 12)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 12, n: 3, m: 10, factor: 1.0 }, name: "Box 3D", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 12, n: 3, m: 10, factor: 10.0 }, name: "Box 3D", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 12, n: 3, m: 10, factor: 100.0 }, name: "Box 3D", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        // Jennrich-Sampson (Problem 13)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 13, n: 2, m: 10, factor: 1.0 }, name: "Jennrich-Sampson", expected_fnorm2: 0.1243621823E+02, expected_info: 1 },
        // Brown-Dennis (Problem 14)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 14, n: 4, m: 20, factor: 1.0 }, name: "Brown-Dennis", expected_fnorm2: 0.8582220163E+05, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 14, n: 4, m: 20, factor: 10.0 }, name: "Brown-Dennis", expected_fnorm2: 0.8582220163E+05, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 14, n: 4, m: 20, factor: 100.0 }, name: "Brown-Dennis", expected_fnorm2: 0.8582220163E+05, expected_info: 1 },
        // Chebyquad (Problem 15)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 15, n: 1, m: 8, factor: 1.0 }, name: "Chebyquad", expected_fnorm2: 0.1883300373E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 15, n: 1, m: 8, factor: 10.0 }, name: "Chebyquad", expected_fnorm2: 0.1883300373E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 15, n: 1, m: 8, factor: 100.0 }, name: "Chebyquad", expected_fnorm2: 0.1883300373E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 15, n: 10, m: 10, factor: 1.0 }, name: "Chebyquad", expected_fnorm2: 0.6506280001E-02, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 15, n: 10, m: 10, factor: 10.0 }, name: "Chebyquad", expected_fnorm2: 0.6506280001E-02, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 15, n: 10, m: 10, factor: 100.0 }, name: "Chebyquad", expected_fnorm2: 0.6506280001E-02, expected_info: 1 },
        // Brown Almost-Linear (Problem 16)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 16, n: 10, m: 10, factor: 1.0 }, name: "Brown Almost-Linear", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 16, n: 10, m: 10, factor: 10.0 }, name: "Brown Almost-Linear", expected_fnorm2: 0.1000000000E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 16, n: 10, m: 10, factor: 100.0 }, name: "Brown Almost-Linear", expected_fnorm2: 0.1000000000E+01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 16, n: 30, m: 30, factor: 1.0 }, name: "Brown Almost-Linear", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 16, n: 40, m: 40, factor: 1.0 }, name: "Brown Almost-Linear", expected_fnorm2: 0.0000000000E+00, expected_info: 1 },
        // Osborne 1 (Problem 17)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 17, n: 5, m: 33, factor: 1.0 }, name: "Osborne 1", expected_fnorm2: 0.5464894697E-04, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 17, n: 5, m: 33, factor: 10.0 }, name: "Osborne 1", expected_fnorm2: 0.5464894697E-04, expected_info: 1 },
        // Osborne 2 (Problem 18)
        LeastSquaresTestCase { id: TestCaseId { problem_number: 18, n: 11, m: 65, factor: 1.0 }, name: "Osborne 2", expected_fnorm2: 0.4013774100E-01, expected_info: 1 },
        LeastSquaresTestCase { id: TestCaseId { problem_number: 18, n: 11, m: 65, factor: 10.0 }, name: "Osborne 2", expected_fnorm2: 0.4013774100E-01, expected_info: 1 },
    ]
}

/// Get all nonlinear equation test cases from MINPACK test_hybrj.f90
pub fn all_nonlinear_test_cases() -> Vec<NonlinearTestCase> {
    vec![
        // Rosenbrock (Problem 1)
        NonlinearTestCase { id: TestCaseId { problem_number: 1, n: 2, m: 2, factor: 1.0 }, name: "Rosenbrock", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 1, n: 2, m: 2, factor: 10.0 }, name: "Rosenbrock", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 1, n: 2, m: 2, factor: 100.0 }, name: "Rosenbrock", expected_fnorm: 0.0, expected_info: 1 },
        // Powell Singular (Problem 2)
        NonlinearTestCase { id: TestCaseId { problem_number: 2, n: 4, m: 4, factor: 1.0 }, name: "Powell Singular", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 2, n: 4, m: 4, factor: 10.0 }, name: "Powell Singular", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 2, n: 4, m: 4, factor: 100.0 }, name: "Powell Singular", expected_fnorm: 0.0, expected_info: 1 },
        // Powell Badly Scaled (Problem 3)
        NonlinearTestCase { id: TestCaseId { problem_number: 3, n: 2, m: 2, factor: 1.0 }, name: "Powell Badly Scaled", expected_fnorm: 0.0, expected_info: 1 },
        // Wood (Problem 4)
        NonlinearTestCase { id: TestCaseId { problem_number: 4, n: 4, m: 4, factor: 1.0 }, name: "Wood", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 4, n: 4, m: 4, factor: 10.0 }, name: "Wood", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 4, n: 4, m: 4, factor: 100.0 }, name: "Wood", expected_fnorm: 0.0, expected_info: 1 },
        // Helical Valley (Problem 5)
        NonlinearTestCase { id: TestCaseId { problem_number: 5, n: 3, m: 3, factor: 1.0 }, name: "Helical Valley", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 5, n: 3, m: 3, factor: 10.0 }, name: "Helical Valley", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 5, n: 3, m: 3, factor: 100.0 }, name: "Helical Valley", expected_fnorm: 0.0, expected_info: 1 },
        // Watson (Problem 6)
        NonlinearTestCase { id: TestCaseId { problem_number: 6, n: 6, m: 6, factor: 1.0 }, name: "Watson", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 6, n: 9, m: 9, factor: 1.0 }, name: "Watson", expected_fnorm: 0.0, expected_info: 1 },
        // Chebyquad (Problem 7)
        NonlinearTestCase { id: TestCaseId { problem_number: 7, n: 5, m: 5, factor: 1.0 }, name: "Chebyquad", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 7, n: 6, m: 6, factor: 1.0 }, name: "Chebyquad", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 7, n: 7, m: 7, factor: 1.0 }, name: "Chebyquad", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 7, n: 9, m: 9, factor: 1.0 }, name: "Chebyquad", expected_fnorm: 0.0, expected_info: 1 },
        // Brown Almost-Linear (Problem 8)
        NonlinearTestCase { id: TestCaseId { problem_number: 8, n: 10, m: 10, factor: 1.0 }, name: "Brown Almost-Linear", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 8, n: 30, m: 30, factor: 1.0 }, name: "Brown Almost-Linear", expected_fnorm: 0.0, expected_info: 1 },
        NonlinearTestCase { id: TestCaseId { problem_number: 8, n: 40, m: 40, factor: 1.0 }, name: "Brown Almost-Linear", expected_fnorm: 0.0, expected_info: 1 },
        // Discrete Boundary Value (Problem 9)
        NonlinearTestCase { id: TestCaseId { problem_number: 9, n: 10, m: 10, factor: 1.0 }, name: "Discrete Boundary Value", expected_fnorm: 0.0, expected_info: 1 },
        // Discrete Integral Equation (Problem 10)
        NonlinearTestCase { id: TestCaseId { problem_number: 10, n: 10, m: 10, factor: 1.0 }, name: "Discrete Integral Equation", expected_fnorm: 0.0, expected_info: 1 },
        // Trigonometric (Problem 11)
        NonlinearTestCase { id: TestCaseId { problem_number: 11, n: 10, m: 10, factor: 1.0 }, name: "Trigonometric", expected_fnorm: 0.0, expected_info: 1 },
        // Variably Dimensioned (Problem 12)
        NonlinearTestCase { id: TestCaseId { problem_number: 12, n: 10, m: 10, factor: 1.0 }, name: "Variably Dimensioned", expected_fnorm: 0.0, expected_info: 1 },
        // Broyden Tridiagonal (Problem 13)
        NonlinearTestCase { id: TestCaseId { problem_number: 13, n: 10, m: 10, factor: 1.0 }, name: "Broyden Tridiagonal", expected_fnorm: 0.0, expected_info: 1 },
        // Broyden Banded (Problem 14)
        NonlinearTestCase { id: TestCaseId { problem_number: 14, n: 10, m: 10, factor: 1.0 }, name: "Broyden Banded", expected_fnorm: 0.0, expected_info: 1 },
    ]
}

/// Create a problem for a given test case specification
pub fn create_problem_for_case(test_case: &LeastSquaresTestCase) -> Option<Box<dyn Problem>> {
    let id = &test_case.id;
    match id.problem_number {
        1 => Some(Box::new(LinearFullRank::new(id.n, id.m))),
        2 => Some(Box::new(LinearRank1::new(id.n, id.m))),
        3 => Some(Box::new(LinearRank1ZeroColumns::new(id.n, id.m))),
        4 => Some(Box::new(Rosenbrock)),
        5 => Some(Box::new(HelicalValley)),
        6 => Some(Box::new(PowellSingular)),
        7 => Some(Box::new(FreudensteinRoth)),
        8 => Some(Box::new(Bard)),
        9 => Some(Box::new(KowalikOsborne)),
        10 => Some(Box::new(Meyer)),
        11 => Some(Box::new(Watson::new(id.n))),
        12 => Some(Box::new(Box3D::new(id.m))),
        13 => Some(Box::new(JennrichSampson::new(id.m))),
        14 => Some(Box::new(BrownDennis::new(id.m))),
        15 => Some(Box::new(Chebyquad::with_m(id.n, id.m))),
        16 => Some(Box::new(BrownAlmostLinear::new(id.n))),
        17 => Some(Box::new(Osborne1)),
        18 => Some(Box::new(Osborne2)),
        _ => None,
    }
}
