//! Interactive geometric constraint solver example.
//!
//! Run with: cargo run -p solverang --features geometry --example geometric_constraints

use solverang::geometry::{ConstraintSystemBuilder, Point2D};
use solverang::{LMConfig, LMSolver};

fn main() {
    println!("=== Geometric Constraint Solver Demo ===\n");

    // Example 1: Equilateral Triangle
    println!("1. EQUILATERAL TRIANGLE");
    println!("   Goal: 3 points forming a triangle with all sides = 10 units");
    println!("   Constraints: fix p0 at origin, horizontal(p0,p1), distance constraints\n");

    let system = ConstraintSystemBuilder::<2>::new()
        .name("Equilateral Triangle")
        .point(Point2D::new(0.0, 0.0)) // p0 - will be fixed
        .point(Point2D::new(5.0, 1.0)) // p1 - initial guess (wrong position)
        .point(Point2D::new(3.0, 4.0)) // p2 - initial guess (wrong position)
        .fix(0) // Fix first point at origin
        .horizontal(0, 1) // p0-p1 is horizontal (constrains p1.y = p0.y)
        .distance(0, 1, 10.0) // p0 to p1 = 10
        .distance(1, 2, 10.0) // p1 to p2 = 10
        .distance(2, 0, 10.0) // p2 to p0 = 10
        .build();

    println!("   Initial points:");
    for (i, p) in system.points().iter().enumerate() {
        println!("     p{}: ({:.4}, {:.4})", i, p[0], p[1]);
    }
    println!("   DOF: {} (should be 0 for well-constrained)", system.degrees_of_freedom());

    let x0 = system.current_values();
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&system, &x0);

    println!("\n   Converged: {}", result.is_converged());
    println!("   Iterations: {:?}", result.iterations());
    println!("   Residual norm: {:?}", result.residual_norm());

    if let Some(solution) = result.solution() {
        println!("\n   Solved points:");
        // Point 0 is fixed, so it's not in the solution vector
        println!("     p0: (0.0000, 0.0000) [fixed]");
        // Solution contains only free points: p1 and p2
        println!("     p1: ({:.4}, {:.4})", solution[0], solution[1]);
        println!("     p2: ({:.4}, {:.4})", solution[2], solution[3]);

        // Verify distances using actual positions
        let p0 = (0.0, 0.0);
        let p1 = (solution[0], solution[1]);
        let p2 = (solution[2], solution[3]);

        let d01 = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
        let d12 = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
        let d20 = ((p0.0 - p2.0).powi(2) + (p0.1 - p2.1).powi(2)).sqrt();

        println!("\n   Verification:");
        println!("     |p0-p1| = {:.6} (target: 10)", d01);
        println!("     |p1-p2| = {:.6} (target: 10)", d12);
        println!("     |p2-p0| = {:.6} (target: 10)", d20);
        println!("     p1.y = {:.6} (should equal p0.y = 0)", p1.1);
    }

    // Example 2: Rectangle
    println!("\n\n2. RECTANGLE");
    println!("   Goal: 4 points forming a 10x5 rectangle");
    println!("   Constraints: horizontal/vertical edges, distance 10 and 5\n");

    let system = ConstraintSystemBuilder::<2>::new()
        .name("Rectangle")
        .point(Point2D::new(0.0, 0.0)) // p0 - bottom-left (fixed)
        .point(Point2D::new(8.0, 1.0)) // p1 - bottom-right (initial guess)
        .point(Point2D::new(9.0, 6.0)) // p2 - top-right (initial guess)
        .point(Point2D::new(1.0, 5.0)) // p3 - top-left (initial guess)
        .fix(0) // Fix bottom-left
        .horizontal(0, 1) // Bottom edge horizontal
        .horizontal(3, 2) // Top edge horizontal
        .vertical(0, 3) // Left edge vertical
        .vertical(1, 2) // Right edge vertical
        .distance(0, 1, 10.0) // Bottom = 10
        .distance(1, 2, 5.0) // Right = 5
        .build();

    println!("   Initial points:");
    for (i, p) in system.points().iter().enumerate() {
        println!("     p{}: ({:.4}, {:.4})", i, p[0], p[1]);
    }
    println!("   DOF: {}", system.degrees_of_freedom());

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    println!("\n   Converged: {}", result.is_converged());

    if let Some(solution) = result.solution() {
        println!("\n   Solved points:");
        println!("     p0: (0.0000, 0.0000) [fixed]");
        println!("     p1: ({:.4}, {:.4})", solution[0], solution[1]);
        println!("     p2: ({:.4}, {:.4})", solution[2], solution[3]);
        println!("     p3: ({:.4}, {:.4})", solution[4], solution[5]);
    }

    // Example 3: Point on Circle
    println!("\n\n3. POINT ON CIRCLE");
    println!("   Goal: Constrain a point to lie on a circle of radius 5\n");

    let system = ConstraintSystemBuilder::<2>::new()
        .name("Point on Circle")
        .point(Point2D::new(0.0, 0.0)) // p0 - center (fixed)
        .point(Point2D::new(7.0, 3.0)) // p1 - point (initial guess, not on circle)
        .fix(0)
        .point_on_circle(1, 0, 5.0) // p1 on circle centered at p0 with radius 5
        .build();

    let initial_dist = (49.0_f64 + 9.0).sqrt();
    println!("   Initial: p1 = ({:.4}, {:.4})", system.points()[1][0], system.points()[1][1]);
    println!("   Distance from origin: {:.4}", initial_dist);

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    if let Some(solution) = result.solution() {
        let x = solution[0];
        let y = solution[1];
        let dist = (x * x + y * y).sqrt();
        println!("\n   Solved: p1 = ({:.4}, {:.4})", x, y);
        println!("   Distance from origin: {:.6} (target: 5)", dist);
    }

    // Example 4: Midpoint constraint
    println!("\n\n4. MIDPOINT CONSTRAINT");
    println!("   Goal: Place a point at the midpoint of a line segment\n");

    let system = ConstraintSystemBuilder::<2>::new()
        .name("Midpoint")
        .point(Point2D::new(0.0, 0.0)) // p0 - start (fixed)
        .point(Point2D::new(10.0, 6.0)) // p1 - end (fixed)
        .point(Point2D::new(3.0, 1.0)) // p2 - midpoint (initial guess, wrong)
        .fix(0)
        .fix(1)
        .midpoint(2, 0, 1) // p2 is midpoint of p0-p1
        .build();

    println!("   Fixed: p0 = (0, 0), p1 = (10, 6)");
    println!("   Initial midpoint guess: p2 = (3, 1)");

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    if let Some(solution) = result.solution() {
        let x = solution[0];
        let y = solution[1];
        println!("\n   Solved midpoint: p2 = ({:.4}, {:.4})", x, y);
        println!("   Expected: (5.0, 3.0)");
    }

    // Example 5: Perpendicular Lines
    println!("\n\n5. PERPENDICULAR LINES");
    println!("   Goal: Two line segments meeting at right angles\n");

    let system = ConstraintSystemBuilder::<2>::new()
        .name("Perpendicular")
        .point(Point2D::new(0.0, 0.0)) // p0 - line1 start (fixed)
        .point(Point2D::new(5.0, 0.5)) // p1 - shared point (initial)
        .point(Point2D::new(6.0, 4.0)) // p2 - line2 end (initial)
        .fix(0)
        .horizontal(0, 1) // First line horizontal
        .distance(0, 1, 5.0) // First line length = 5
        .perpendicular(0, 1, 1, 2) // Lines are perpendicular
        .distance(1, 2, 4.0) // Second line length = 4
        .build();

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    println!("   Converged: {}", result.is_converged());

    if let Some(solution) = result.solution() {
        let p0 = (0.0, 0.0);
        let p1 = (solution[0], solution[1]);
        let p2 = (solution[2], solution[3]);

        // Calculate vectors
        let v1 = (p1.0 - p0.0, p1.1 - p0.1);
        let v2 = (p2.0 - p1.0, p2.1 - p1.1);

        // Dot product should be 0 for perpendicular
        let dot = v1.0 * v2.0 + v1.1 * v2.1;

        println!("\n   Solved points:");
        println!("     p0: (0.0000, 0.0000) [fixed]");
        println!("     p1: ({:.4}, {:.4})", p1.0, p1.1);
        println!("     p2: ({:.4}, {:.4})", p2.0, p2.1);
        println!("\n   Verification:");
        println!("     Dot product (should be 0): {:.6}", dot);
        println!("     |p0-p1| = {:.4} (target: 5)", (v1.0*v1.0 + v1.1*v1.1).sqrt());
        println!("     |p1-p2| = {:.4} (target: 4)", (v2.0*v2.0 + v2.1*v2.1).sqrt());
    }

    println!("\n=== Demo Complete ===");
}
