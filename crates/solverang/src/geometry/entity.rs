/// All supported geometric entity types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityKind {
    // === 2D Primitives ===
    Point2D,          // [x, y] — 2 params
    Line2D,           // [x1, y1, x2, y2] — 4 params
    Circle2D,         // [cx, cy, r] — 3 params
    Arc2D,            // [cx, cy, r, start_angle, end_angle] — 5 params
    Ellipse2D,        // [cx, cy, rx, ry, rotation] — 5 params
    EllipticArc2D,    // [cx, cy, rx, ry, rot, t0, t1] — 7 params
    CubicBezier2D,    // [x0,y0, x1,y1, x2,y2, x3,y3] — 8 params
    QuadBezier2D,     // [x0,y0, x1,y1, x2,y2] — 6 params

    // === 3D Primitives ===
    Point3D,          // [x, y, z] — 3 params
    Line3D,           // [x1,y1,z1, x2,y2,z2] — 6 params
    Circle3D,         // [cx,cy,cz, nx,ny,nz, r] — 7 params
    Arc3D,            // [cx,cy,cz, nx,ny,nz, r, t0, t1] — 9 params
    Sphere,           // [cx,cy,cz, r] — 4 params
    Cylinder,         // [px,py,pz, dx,dy,dz, r] — 7 params
    Cone,             // [px,py,pz, dx,dy,dz, half_angle] — 7 params
    Torus,            // [cx,cy,cz, nx,ny,nz, R, r] — 8 params
    Plane,            // [px,py,pz, nx,ny,nz] — 6 params
    Ellipse3D,        // [cx,cy,cz, major_x,y,z, minor_x,y,z] — 9 params
    CubicBezier3D,    // [x0..z0, x1..z1, x2..z2, x3..z3] — 12 params

    // === Auxiliary ===
    Scalar,           // [value] — 1 param
}

impl EntityKind {
    /// Returns the number of parameters for this entity kind.
    pub fn param_count(&self) -> usize {
        match self {
            // 2D
            EntityKind::Point2D => 2,
            EntityKind::Line2D => 4,
            EntityKind::Circle2D => 3,
            EntityKind::Arc2D => 5,
            EntityKind::Ellipse2D => 5,
            EntityKind::EllipticArc2D => 7,
            EntityKind::CubicBezier2D => 8,
            EntityKind::QuadBezier2D => 6,

            // 3D
            EntityKind::Point3D => 3,
            EntityKind::Line3D => 6,
            EntityKind::Circle3D => 7,
            EntityKind::Arc3D => 9,
            EntityKind::Sphere => 4,
            EntityKind::Cylinder => 7,
            EntityKind::Cone => 7,
            EntityKind::Torus => 8,
            EntityKind::Plane => 6,
            EntityKind::Ellipse3D => 9,
            EntityKind::CubicBezier3D => 12,

            // Auxiliary
            EntityKind::Scalar => 1,
        }
    }

    /// Returns a short string name for this entity kind.
    pub fn name(&self) -> &'static str {
        match self {
            // 2D
            EntityKind::Point2D => "Point2D",
            EntityKind::Line2D => "Line2D",
            EntityKind::Circle2D => "Circle2D",
            EntityKind::Arc2D => "Arc2D",
            EntityKind::Ellipse2D => "Ellipse2D",
            EntityKind::EllipticArc2D => "EllipticArc2D",
            EntityKind::CubicBezier2D => "CubicBezier2D",
            EntityKind::QuadBezier2D => "QuadBezier2D",

            // 3D
            EntityKind::Point3D => "Point3D",
            EntityKind::Line3D => "Line3D",
            EntityKind::Circle3D => "Circle3D",
            EntityKind::Arc3D => "Arc3D",
            EntityKind::Sphere => "Sphere",
            EntityKind::Cylinder => "Cylinder",
            EntityKind::Cone => "Cone",
            EntityKind::Torus => "Torus",
            EntityKind::Plane => "Plane",
            EntityKind::Ellipse3D => "Ellipse3D",
            EntityKind::CubicBezier3D => "CubicBezier3D",

            // Auxiliary
            EntityKind::Scalar => "Scalar",
        }
    }

    /// Returns the geometric dimension of this entity.
    /// Returns Some(2) for 2D entities, Some(3) for 3D entities, None for Scalar.
    pub fn dimension(&self) -> Option<usize> {
        match self {
            // 2D
            EntityKind::Point2D
            | EntityKind::Line2D
            | EntityKind::Circle2D
            | EntityKind::Arc2D
            | EntityKind::Ellipse2D
            | EntityKind::EllipticArc2D
            | EntityKind::CubicBezier2D
            | EntityKind::QuadBezier2D => Some(2),

            // 3D
            EntityKind::Point3D
            | EntityKind::Line3D
            | EntityKind::Circle3D
            | EntityKind::Arc3D
            | EntityKind::Sphere
            | EntityKind::Cylinder
            | EntityKind::Cone
            | EntityKind::Torus
            | EntityKind::Plane
            | EntityKind::Ellipse3D
            | EntityKind::CubicBezier3D => Some(3),

            // Auxiliary
            EntityKind::Scalar => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_count() {
        assert_eq!(EntityKind::Point2D.param_count(), 2);
        assert_eq!(EntityKind::Point3D.param_count(), 3);
        assert_eq!(EntityKind::Line2D.param_count(), 4);
        assert_eq!(EntityKind::Line3D.param_count(), 6);
        assert_eq!(EntityKind::Circle2D.param_count(), 3);
        assert_eq!(EntityKind::Circle3D.param_count(), 7);
        assert_eq!(EntityKind::Arc2D.param_count(), 5);
        assert_eq!(EntityKind::Arc3D.param_count(), 9);
        assert_eq!(EntityKind::Ellipse2D.param_count(), 5);
        assert_eq!(EntityKind::Ellipse3D.param_count(), 9);
        assert_eq!(EntityKind::EllipticArc2D.param_count(), 7);
        assert_eq!(EntityKind::CubicBezier2D.param_count(), 8);
        assert_eq!(EntityKind::CubicBezier3D.param_count(), 12);
        assert_eq!(EntityKind::QuadBezier2D.param_count(), 6);
        assert_eq!(EntityKind::Sphere.param_count(), 4);
        assert_eq!(EntityKind::Cylinder.param_count(), 7);
        assert_eq!(EntityKind::Cone.param_count(), 7);
        assert_eq!(EntityKind::Torus.param_count(), 8);
        assert_eq!(EntityKind::Plane.param_count(), 6);
        assert_eq!(EntityKind::Scalar.param_count(), 1);
    }

    #[test]
    fn test_name() {
        assert_eq!(EntityKind::Point2D.name(), "Point2D");
        assert_eq!(EntityKind::Circle2D.name(), "Circle2D");
        assert_eq!(EntityKind::Sphere.name(), "Sphere");
        assert_eq!(EntityKind::Scalar.name(), "Scalar");
    }

    #[test]
    fn test_dimension() {
        // 2D entities
        assert_eq!(EntityKind::Point2D.dimension(), Some(2));
        assert_eq!(EntityKind::Line2D.dimension(), Some(2));
        assert_eq!(EntityKind::Circle2D.dimension(), Some(2));
        assert_eq!(EntityKind::Arc2D.dimension(), Some(2));
        assert_eq!(EntityKind::Ellipse2D.dimension(), Some(2));
        assert_eq!(EntityKind::EllipticArc2D.dimension(), Some(2));
        assert_eq!(EntityKind::CubicBezier2D.dimension(), Some(2));
        assert_eq!(EntityKind::QuadBezier2D.dimension(), Some(2));

        // 3D entities
        assert_eq!(EntityKind::Point3D.dimension(), Some(3));
        assert_eq!(EntityKind::Line3D.dimension(), Some(3));
        assert_eq!(EntityKind::Circle3D.dimension(), Some(3));
        assert_eq!(EntityKind::Arc3D.dimension(), Some(3));
        assert_eq!(EntityKind::Sphere.dimension(), Some(3));
        assert_eq!(EntityKind::Cylinder.dimension(), Some(3));
        assert_eq!(EntityKind::Cone.dimension(), Some(3));
        assert_eq!(EntityKind::Torus.dimension(), Some(3));
        assert_eq!(EntityKind::Plane.dimension(), Some(3));
        assert_eq!(EntityKind::Ellipse3D.dimension(), Some(3));
        assert_eq!(EntityKind::CubicBezier3D.dimension(), Some(3));

        // Auxiliary
        assert_eq!(EntityKind::Scalar.dimension(), None);
    }

    #[test]
    fn test_entity_kind_traits() {
        // Test that EntityKind implements required traits
        let kind = EntityKind::Circle2D;
        let kind2 = kind; // Copy
        assert_eq!(kind, kind2); // PartialEq

        // Test Hash by putting in a HashSet
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EntityKind::Point2D);
        set.insert(EntityKind::Point2D); // duplicate
        assert_eq!(set.len(), 1);
        set.insert(EntityKind::Circle2D);
        assert_eq!(set.len(), 2);
    }
}
