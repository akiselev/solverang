use solverang_macros::{auto_diff, objective};

struct Bad;

#[auto_diff(array_param = "x")]
impl Bad {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].foobar()
    }
}

fn main() {}
