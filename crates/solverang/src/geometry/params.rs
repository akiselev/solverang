use std::collections::HashMap;
use super::entity::EntityKind;

/// Unique entity identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(pub usize);

/// Unique constraint identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintId(pub usize);

/// A contiguous range of parameters belonging to one entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamRange {
    pub start: usize,
    pub count: usize,
}

impl ParamRange {
    /// Iterate over all parameter indices in this range.
    pub fn iter(&self) -> impl Iterator<Item = usize> {
        self.start..self.start + self.count
    }

    /// Get the parameter index at a specific offset within this range.
    /// Panics if offset >= count.
    pub fn get(&self, offset: usize) -> usize {
        assert!(offset < self.count, "offset {} out of range (count: {})", offset, self.count);
        self.start + offset
    }
}

/// Typed entity handle — zero-cost wrapper providing named access.
#[derive(Clone, Copy, Debug)]
pub struct EntityHandle {
    pub id: EntityId,
    pub kind: EntityKind,
    pub params: ParamRange,
}

impl EntityHandle {
    /// Get the parameter index for a specific offset within this entity.
    /// Convenience method that delegates to params.get().
    pub fn param(&self, offset: usize) -> usize {
        self.params.get(offset)
    }
}

/// The central parameter storage. All solver variables live in a flat Vec<f64>.
pub struct ParameterStore {
    /// Flat vector of all solver parameters.
    values: Vec<f64>,
    /// Per-parameter: is this value fixed (driven) or free (solved)?
    fixed: Vec<bool>,
    /// Per-parameter: human-readable label for diagnostics.
    labels: Vec<String>,
    /// Entity registry: maps EntityId to its handle.
    entities: HashMap<EntityId, EntityHandle>,
    /// Next entity ID to allocate.
    next_entity_id: usize,
}

impl ParameterStore {
    /// Create a new empty parameter store.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            fixed: Vec::new(),
            labels: Vec::new(),
            entities: HashMap::new(),
            next_entity_id: 0,
        }
    }

    /// Add a new entity with initial parameter values.
    /// Returns a typed handle for this entity.
    pub fn add_entity(&mut self, kind: EntityKind, initial_values: &[f64]) -> EntityHandle {
        let expected_count = kind.param_count();
        assert_eq!(
            initial_values.len(),
            expected_count,
            "Expected {} parameters for {:?}, got {}",
            expected_count,
            kind,
            initial_values.len()
        );

        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;

        let start = self.values.len();
        let count = initial_values.len();

        // Allocate parameters
        self.values.extend_from_slice(initial_values);
        self.fixed.extend(std::iter::repeat(false).take(count));

        // Generate default labels
        for i in 0..count {
            self.labels.push(format!("{}_{}.p{}", kind.name(), id.0, i));
        }

        let handle = EntityHandle {
            id,
            kind,
            params: ParamRange { start, count },
        };

        self.entities.insert(id, handle);
        handle
    }

    /// Add a new entity with initial parameter values and custom labels.
    pub fn add_entity_with_labels(
        &mut self,
        kind: EntityKind,
        initial_values: &[f64],
        labels: &[&str],
    ) -> EntityHandle {
        let expected_count = kind.param_count();
        assert_eq!(
            initial_values.len(),
            expected_count,
            "Expected {} parameters for {:?}, got {}",
            expected_count,
            kind,
            initial_values.len()
        );
        assert_eq!(
            labels.len(),
            expected_count,
            "Expected {} labels for {:?}, got {}",
            expected_count,
            kind,
            labels.len()
        );

        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;

        let start = self.values.len();
        let count = initial_values.len();

        // Allocate parameters
        self.values.extend_from_slice(initial_values);
        self.fixed.extend(std::iter::repeat(false).take(count));

        // Use custom labels
        for label in labels {
            self.labels.push(label.to_string());
        }

        let handle = EntityHandle {
            id,
            kind,
            params: ParamRange { start, count },
        };

        self.entities.insert(id, handle);
        handle
    }

    /// Convenience: add a 2D point.
    pub fn add_point_2d(&mut self, x: f64, y: f64) -> EntityHandle {
        self.add_entity(EntityKind::Point2D, &[x, y])
    }

    /// Convenience: add a 3D point.
    pub fn add_point_3d(&mut self, x: f64, y: f64, z: f64) -> EntityHandle {
        self.add_entity(EntityKind::Point3D, &[x, y, z])
    }

    /// Convenience: add a 2D circle.
    pub fn add_circle_2d(&mut self, cx: f64, cy: f64, r: f64) -> EntityHandle {
        self.add_entity(EntityKind::Circle2D, &[cx, cy, r])
    }

    /// Convenience: add a 2D line.
    pub fn add_line_2d(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) -> EntityHandle {
        self.add_entity(EntityKind::Line2D, &[x1, y1, x2, y2])
    }

    /// Convenience: add a 2D arc.
    pub fn add_arc_2d(&mut self, cx: f64, cy: f64, r: f64, start_angle: f64, end_angle: f64) -> EntityHandle {
        self.add_entity(EntityKind::Arc2D, &[cx, cy, r, start_angle, end_angle])
    }

    /// Convenience: add a 2D cubic Bezier curve.
    pub fn add_cubic_bezier_2d(&mut self, points: [[f64; 2]; 4]) -> EntityHandle {
        let params = [
            points[0][0], points[0][1],
            points[1][0], points[1][1],
            points[2][0], points[2][1],
            points[3][0], points[3][1],
        ];
        self.add_entity(EntityKind::CubicBezier2D, &params)
    }

    /// Convenience: add a 2D ellipse.
    pub fn add_ellipse_2d(&mut self, cx: f64, cy: f64, rx: f64, ry: f64, rotation: f64) -> EntityHandle {
        self.add_entity(EntityKind::Ellipse2D, &[cx, cy, rx, ry, rotation])
    }

    /// Convenience: add a scalar parameter.
    pub fn add_scalar(&mut self, value: f64) -> EntityHandle {
        self.add_entity(EntityKind::Scalar, &[value])
    }

    /// Get the entity handle for a given ID.
    pub fn get_entity(&self, id: EntityId) -> Option<&EntityHandle> {
        self.entities.get(&id)
    }

    /// Get a reference to all parameter values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get a mutable reference to all parameter values.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Get the value of a single parameter.
    pub fn get_value(&self, idx: usize) -> f64 {
        self.values[idx]
    }

    /// Set the value of a single parameter.
    pub fn set_value(&mut self, idx: usize, val: f64) {
        self.values[idx] = val;
    }

    /// Get the parameter values for a specific entity.
    pub fn get_entity_values(&self, handle: &EntityHandle) -> &[f64] {
        &self.values[handle.params.start..handle.params.start + handle.params.count]
    }

    /// Fix a parameter (make it non-variable).
    pub fn fix_param(&mut self, idx: usize) {
        self.fixed[idx] = true;
    }

    /// Free a parameter (make it variable).
    pub fn free_param(&mut self, idx: usize) {
        self.fixed[idx] = false;
    }

    /// Check if a parameter is fixed.
    pub fn is_fixed(&self, idx: usize) -> bool {
        self.fixed[idx]
    }

    /// Fix all parameters of an entity.
    pub fn fix_entity(&mut self, handle: &EntityHandle) {
        for idx in handle.params.iter() {
            self.fixed[idx] = true;
        }
    }

    /// Free all parameters of an entity.
    pub fn free_entity(&mut self, handle: &EntityHandle) {
        for idx in handle.params.iter() {
            self.fixed[idx] = false;
        }
    }

    /// Total number of parameters allocated.
    pub fn param_count(&self) -> usize {
        self.values.len()
    }

    /// Number of free (non-fixed) parameters.
    pub fn free_param_count(&self) -> usize {
        self.fixed.iter().filter(|&&f| !f).count()
    }

    /// Get the indices of all free parameters.
    pub fn free_indices(&self) -> Vec<usize> {
        self.fixed
            .iter()
            .enumerate()
            .filter_map(|(i, &fixed)| if !fixed { Some(i) } else { None })
            .collect()
    }

    /// Total number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get all entity handles sorted by EntityId.
    pub fn entity_handles(&self) -> Vec<EntityHandle> {
        let mut handles: Vec<EntityHandle> = self.entities.values().copied().collect();
        handles.sort_by_key(|h| h.id.0);
        handles
    }

    /// Get the current values of all free parameters (in order of free_indices).
    pub fn current_free_values(&self) -> Vec<f64> {
        self.free_indices()
            .iter()
            .map(|&i| self.values[i])
            .collect()
    }

    /// Write solution values back to free parameter slots.
    /// The length of `values` must equal free_param_count().
    pub fn set_free_values(&mut self, values: &[f64]) {
        let free_indices = self.free_indices();
        assert_eq!(
            values.len(),
            free_indices.len(),
            "Expected {} free values, got {}",
            free_indices.len(),
            values.len()
        );

        for (&idx, &value) in free_indices.iter().zip(values.iter()) {
            self.values[idx] = value;
        }
    }

    /// Get the label for a parameter.
    pub fn label(&self, idx: usize) -> &str {
        &self.labels[idx]
    }
}

impl Default for ParameterStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_range_iter() {
        let range = ParamRange { start: 5, count: 3 };
        let indices: Vec<usize> = range.iter().collect();
        assert_eq!(indices, vec![5, 6, 7]);
    }

    #[test]
    fn test_param_range_get() {
        let range = ParamRange { start: 10, count: 4 };
        assert_eq!(range.get(0), 10);
        assert_eq!(range.get(1), 11);
        assert_eq!(range.get(3), 13);
    }

    #[test]
    #[should_panic(expected = "offset 4 out of range")]
    fn test_param_range_get_out_of_bounds() {
        let range = ParamRange { start: 10, count: 4 };
        range.get(4);
    }

    #[test]
    fn test_entity_handle_param() {
        let handle = EntityHandle {
            id: EntityId(0),
            kind: EntityKind::Circle2D,
            params: ParamRange { start: 5, count: 3 },
        };
        assert_eq!(handle.param(0), 5);
        assert_eq!(handle.param(1), 6);
        assert_eq!(handle.param(2), 7);
    }

    #[test]
    fn test_parameter_store_new() {
        let store = ParameterStore::new();
        assert_eq!(store.param_count(), 0);
        assert_eq!(store.entity_count(), 0);
        assert_eq!(store.free_param_count(), 0);
    }

    #[test]
    fn test_add_point_2d() {
        let mut store = ParameterStore::new();
        let p = store.add_point_2d(1.0, 2.0);

        assert_eq!(p.kind, EntityKind::Point2D);
        assert_eq!(p.params.start, 0);
        assert_eq!(p.params.count, 2);
        assert_eq!(store.param_count(), 2);
        assert_eq!(store.get_value(0), 1.0);
        assert_eq!(store.get_value(1), 2.0);
    }

    #[test]
    fn test_add_circle_2d() {
        let mut store = ParameterStore::new();
        let c = store.add_circle_2d(3.0, 4.0, 5.0);

        assert_eq!(c.kind, EntityKind::Circle2D);
        assert_eq!(c.params.count, 3);
        assert_eq!(store.param_count(), 3);
        assert_eq!(store.get_value(0), 3.0);
        assert_eq!(store.get_value(1), 4.0);
        assert_eq!(store.get_value(2), 5.0);
    }

    #[test]
    fn test_add_multiple_entities() {
        let mut store = ParameterStore::new();
        let p1 = store.add_point_2d(0.0, 0.0);
        let p2 = store.add_point_2d(1.0, 1.0);
        let circle = store.add_circle_2d(0.5, 0.5, 0.7);

        assert_eq!(p1.params.start, 0);
        assert_eq!(p2.params.start, 2);
        assert_eq!(circle.params.start, 4);
        assert_eq!(store.param_count(), 7);
        assert_eq!(store.entity_count(), 3);
    }

    #[test]
    fn test_get_entity_values() {
        let mut store = ParameterStore::new();
        let circle = store.add_circle_2d(10.0, 20.0, 30.0);
        let values = store.get_entity_values(&circle);
        assert_eq!(values, &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_fix_and_free_param() {
        let mut store = ParameterStore::new();
        store.add_point_2d(1.0, 2.0);

        assert!(!store.is_fixed(0));
        assert!(!store.is_fixed(1));

        store.fix_param(0);
        assert!(store.is_fixed(0));
        assert!(!store.is_fixed(1));

        store.free_param(0);
        assert!(!store.is_fixed(0));
    }

    #[test]
    fn test_fix_and_free_entity() {
        let mut store = ParameterStore::new();
        let circle = store.add_circle_2d(1.0, 2.0, 3.0);

        assert_eq!(store.free_param_count(), 3);

        store.fix_entity(&circle);
        assert_eq!(store.free_param_count(), 0);
        assert!(store.is_fixed(0));
        assert!(store.is_fixed(1));
        assert!(store.is_fixed(2));

        store.free_entity(&circle);
        assert_eq!(store.free_param_count(), 3);
    }

    #[test]
    fn test_free_indices() {
        let mut store = ParameterStore::new();
        store.add_point_2d(0.0, 0.0); // params 0, 1
        store.add_point_2d(1.0, 1.0); // params 2, 3

        store.fix_param(1);
        store.fix_param(3);

        let free = store.free_indices();
        assert_eq!(free, vec![0, 2]);
    }

    #[test]
    fn test_current_free_values() {
        let mut store = ParameterStore::new();
        store.add_point_2d(10.0, 20.0);
        store.add_point_2d(30.0, 40.0);

        store.fix_param(1);

        let free_vals = store.current_free_values();
        assert_eq!(free_vals, vec![10.0, 30.0, 40.0]);
    }

    #[test]
    fn test_set_free_values() {
        let mut store = ParameterStore::new();
        store.add_point_2d(0.0, 0.0);
        store.add_point_2d(0.0, 0.0);

        store.fix_param(1);

        let new_vals = vec![100.0, 200.0, 300.0];
        store.set_free_values(&new_vals);

        assert_eq!(store.get_value(0), 100.0);
        assert_eq!(store.get_value(1), 0.0); // Still fixed, unchanged
        assert_eq!(store.get_value(2), 200.0);
        assert_eq!(store.get_value(3), 300.0);
    }

    #[test]
    fn test_labels() {
        let mut store = ParameterStore::new();
        let p = store.add_point_2d(1.0, 2.0);

        assert_eq!(store.label(p.param(0)), "Point2D_0.p0");
        assert_eq!(store.label(p.param(1)), "Point2D_0.p1");
    }

    #[test]
    fn test_labels_custom() {
        let mut store = ParameterStore::new();
        let p = store.add_entity_with_labels(
            EntityKind::Point2D,
            &[5.0, 10.0],
            &["my_point.x", "my_point.y"],
        );

        assert_eq!(store.label(p.param(0)), "my_point.x");
        assert_eq!(store.label(p.param(1)), "my_point.y");
    }

    #[test]
    fn test_get_entity() {
        let mut store = ParameterStore::new();
        let p = store.add_point_2d(1.0, 2.0);

        let retrieved = store.get_entity(p.id).unwrap();
        assert_eq!(retrieved.id, p.id);
        assert_eq!(retrieved.kind, EntityKind::Point2D);
    }

    #[test]
    fn test_add_entity_with_wrong_param_count() {
        let mut store = ParameterStore::new();
        // Circle2D expects 3 params, but we provide 2
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.add_entity(EntityKind::Circle2D, &[1.0, 2.0])
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_methods() {
        let mut store = ParameterStore::new();

        let p2d = store.add_point_2d(1.0, 2.0);
        assert_eq!(p2d.kind, EntityKind::Point2D);

        let p3d = store.add_point_3d(1.0, 2.0, 3.0);
        assert_eq!(p3d.kind, EntityKind::Point3D);
        assert_eq!(p3d.params.count, 3);

        let line = store.add_line_2d(0.0, 0.0, 10.0, 10.0);
        assert_eq!(line.kind, EntityKind::Line2D);
        assert_eq!(line.params.count, 4);

        let arc = store.add_arc_2d(5.0, 5.0, 3.0, 0.0, 1.57);
        assert_eq!(arc.kind, EntityKind::Arc2D);
        assert_eq!(arc.params.count, 5);

        let scalar = store.add_scalar(42.0);
        assert_eq!(scalar.kind, EntityKind::Scalar);
        assert_eq!(scalar.params.count, 1);
        assert_eq!(store.get_value(scalar.param(0)), 42.0);
    }

    #[test]
    fn test_values_mut() {
        let mut store = ParameterStore::new();
        store.add_point_2d(1.0, 2.0);

        {
            let vals = store.values_mut();
            vals[0] = 100.0;
        }

        assert_eq!(store.get_value(0), 100.0);
    }
}
