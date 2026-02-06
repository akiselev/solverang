//! Interactive geometric constraint solver example.
//!
//! Run with: cargo run -p solverang --features geometry --example geometric_constraints

use solverang::geometry::ConstraintSystemBuilder;
use solverang::{LMConfig, LMSolver};

fn main() {
    println!("=== Geometric Constraint Solver Demo ===\n");

    // Example 1: Equilateral Triangle
    println!("1. EQUILATERAL TRIANGLE");
    println!("   Goal: 3 points forming a triangle with all sides = 10 units");
    println!("   Constraints: fix p0 at origin, horizontal(p0,p1), distance constraints\n");

    let mut system = ConstraintSystemBuilder::new()
        .name("Equilateral Triangle")
        .point_2d_fixed(0.0, 0.0)  // entity 0 (p0) - fixed at origin
        .point_2d(5.0, 1.0)        // entity 1 (p1) - initial guess (wrong position)
        .point_2d(3.0, 4.0)        // entity 2 (p2) - initial guess (wrong position)
        .horizontal(0, 1)           // p0-p1 is horizontal (constrains p1.y = p0.y)
        .distance(0, 1, 10.0)       // p0 to p1 = 10
        .distance(1, 2, 10.0)       // p1 to p2 = 10
        .distance(2, 0, 10.0)       // p2 to p0 = 10
        .build();

    let handles = system.handles();
    println!("   Initial points:");
    for (i, h) in handles.iter().enumerate() {
        let vals = system.params().get_entity_values(h);
        println!("     p{}: ({:.4}, {:.4})", i, vals[0], vals[1]);
    }
    println!("   DOF: {} (should be 0 for well-constrained)", system.degrees_of_freedom());

    let x0 = system.current_values();
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&system, &x0);

    println!("\n   Converged: {}", result.is_converged());
    println!("   Iterations: {:?}", result.iterations());
    println!("   Residual norm: {:?}", result.residual_norm());

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let handles = system.handles();

        println!("\n   Solved points:");
        for (i, h) in handles.iter().enumerate() {
            let vals = system.params().get_entity_values(h);
            let fixed = system.is_param_fixed(h.params.start);
            let suffix = if fixed { " [fixed]" } else { "" };
            println!("     p{}: ({:.4}, {:.4}){}", i, vals[0], vals[1], suffix);
        }

        // Verify distances using the solved entity values
        let p0 = system.params().get_entity_values(&handles[0]);
        let p1 = system.params().get_entity_values(&handles[1]);
        let p2 = system.params().get_entity_values(&handles[2]);

        let d01 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        let d12 = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let d20 = ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2)).sqrt();

        println!("\n   Verification:");
        println!("     |p0-p1| = {:.6} (target: 10)", d01);
        println!("     |p1-p2| = {:.6} (target: 10)", d12);
        println!("     |p2-p0| = {:.6} (target: 10)", d20);
        println!("     p1.y = {:.6} (should equal p0.y = 0)", p1[1]);
    }

    // Example 2: Rectangle
    println!("\n\n2. RECTANGLE");
    println!("   Goal: 4 points forming a 10x5 rectangle");
    println!("   Constraints: horizontal/vertical edges, distance 10 and 5\n");

    let mut system = ConstraintSystemBuilder::new()
        .name("Rectangle")
        .point_2d_fixed(0.0, 0.0)  // entity 0 (p0) - bottom-left (fixed)
        .point_2d(8.0, 1.0)        // entity 1 (p1) - bottom-right (initial guess)
        .point_2d(9.0, 6.0)        // entity 2 (p2) - top-right (initial guess)
        .point_2d(1.0, 5.0)        // entity 3 (p3) - top-left (initial guess)
        .horizontal(0, 1)           // Bottom edge horizontal
        .horizontal(3, 2)           // Top edge horizontal
        .vertical(0, 3)             // Left edge vertical
        .vertical(1, 2)             // Right edge vertical
        .distance(0, 1, 10.0)       // Bottom = 10
        .distance(1, 2, 5.0)        // Right = 5
        .build();

    let handles = system.handles();
    println!("   Initial points:");
    for (i, h) in handles.iter().enumerate() {
        let vals = system.params().get_entity_values(h);
        println!("     p{}: ({:.4}, {:.4})", i, vals[0], vals[1]);
    }
    println!("   DOF: {}", system.degrees_of_freedom());

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    println!("\n   Converged: {}", result.is_converged());

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let handles = system.handles();

        println!("\n   Solved points:");
        for (i, h) in handles.iter().enumerate() {
            let vals = system.params().get_entity_values(h);
            let fixed = system.is_param_fixed(h.params.start);
            let suffix = if fixed { " [fixed]" } else { "" };
            println!("     p{}: ({:.4}, {:.4}){}", i, vals[0], vals[1], suffix);
        }
    }

    // Example 3: Point on Circle
    println!("\n\n3. POINT ON CIRCLE");
    println!("   Goal: Constrain a point to lie on a circle of radius 5\n");

    let mut system = ConstraintSystemBuilder::new()
        .name("Point on Circle")
        .circle_2d(0.0, 0.0, 5.0)  // entity 0 - circle at origin with radius 5
        .point_2d(7.0, 3.0)        // entity 1 - point (initial guess, not on circle)
        .fix(0)                      // Fix the circle (center and radius are immutable)
        .point_on_circle(1, 0)       // point (entity 1) must lie on circle (entity 0)
        .build();

    let handles = system.handles();
    let pt_vals = system.params().get_entity_values(&handles[1]);
    let initial_dist = (pt_vals[0] * pt_vals[0] + pt_vals[1] * pt_vals[1]).sqrt();
    println!("   Circle: center (0, 0), radius 5 [fixed]");
    println!("   Initial: p1 = ({:.4}, {:.4})", pt_vals[0], pt_vals[1]);
    println!("   Distance from origin: {:.4}", initial_dist);

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let handles = system.handles();
        let pt = system.params().get_entity_values(&handles[1]);
        let dist = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
        println!("\n   Solved: p1 = ({:.4}, {:.4})", pt[0], pt[1]);
        println!("   Distance from origin: {:.6} (target: 5)", dist);
    }

    // Example 4: Midpoint constraint
    println!("\n\n4. MIDPOINT CONSTRAINT");
    println!("   Goal: Place a point at the midpoint of a line segment\n");

    let mut system = ConstraintSystemBuilder::new()
        .name("Midpoint")
        .point_2d_fixed(0.0, 0.0)   // entity 0 (p0) - start (fixed)
        .point_2d_fixed(10.0, 6.0)  // entity 1 (p1) - end (fixed)
        .point_2d(3.0, 1.0)         // entity 2 (p2) - midpoint (initial guess, wrong)
        .midpoint(2, 0, 1)           // p2 is midpoint of p0-p1
        .build();

    println!("   Fixed: p0 = (0, 0), p1 = (10, 6)");
    println!("   Initial midpoint guess: p2 = (3, 1)");

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let handles = system.handles();
        let mid = system.params().get_entity_values(&handles[2]);
        println!("\n   Solved midpoint: p2 = ({:.4}, {:.4})", mid[0], mid[1]);
        println!("   Expected: (5.0, 3.0)");
    }

    // Example 5: Perpendicular Lines
    println!("\n\n5. PERPENDICULAR LINES");
    println!("   Goal: Two line segments meeting at right angles\n");

    let mut system = ConstraintSystemBuilder::new()
        .name("Perpendicular")
        .line_2d(0.0, 0.0, 5.0, 0.0)  // entity 0 - line1: horizontal from origin (fixed)
        .line_2d(5.0, 0.0, 5.5, 4.0)  // entity 1 - line2: starts at line1's end (initial)
        .fix(0)                         // Fix line1 entirely
        .fix_param_at(1, 0)             // Fix line2 start x = 5 (shared endpoint)
        .fix_param_at(1, 1)             // Fix line2 start y = 0
        .perpendicular(0, 1)            // Lines must be perpendicular
        .build();

    let x0 = system.current_values();
    let result = solver.solve(&system, &x0);

    println!("   Converged: {}", result.is_converged());

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let handles = system.handles();

        // Line1 params: [x1, y1, x2, y2]
        let l1 = system.params().get_entity_values(&handles[0]);
        // Line2 params: [x1, y1, x2, y2]
        let l2 = system.params().get_entity_values(&handles[1]);

        // Direction vectors
        let v1 = (l1[2] - l1[0], l1[3] - l1[1]);
        let v2 = (l2[2] - l2[0], l2[3] - l2[1]);

        // Dot product should be 0 for perpendicular
        let dot = v1.0 * v2.0 + v1.1 * v2.1;

        let len1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let len2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();

        println!("\n   Solved lines:");
        println!("     line1: ({:.4}, {:.4}) -> ({:.4}, {:.4}) [fixed]", l1[0], l1[1], l1[2], l1[3]);
        println!("     line2: ({:.4}, {:.4}) -> ({:.4}, {:.4})", l2[0], l2[1], l2[2], l2[3]);
        println!("\n   Verification:");
        println!("     Dot product (should be 0): {:.6}", dot);
        println!("     |line1| = {:.4}", len1);
        println!("     |line2| = {:.4}", len2);
    }

    println!("\n=== Demo Complete ===");
}
