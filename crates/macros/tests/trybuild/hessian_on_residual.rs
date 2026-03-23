use solverang_macros::{auto_diff, hessian, residual};

struct Bad;

#[auto_diff(array_param = "x")]
impl Bad {
    #[residual]
    #[hessian]
    fn r(&self, x: &[f64]) -> f64 {
        x[0] * x[0]
    }
}

fn main() {}
