use std::collections::HashMap;

/// Component identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComponentId(pub usize);

/// Incremental union-find that supports both union and split (via rebuild).
///
/// This data structure maintains a forest of disjoint sets with path compression
/// and union by rank. It supports incremental additions via `add_edge()` and
/// removals via `remove_edge()`, where removals trigger a rebuild of affected
/// components.
///
/// Key insight: removals are rare compared to additions in an interactive editor,
/// so we optimize for fast additions (O(α(n)) amortized) and accept O(component_size)
/// for removals by rebuilding from the edge list.
pub struct IncrementalUnionFind {
    /// Parent pointers for union-find. parent[i] = i means i is a root.
    parent: Vec<usize>,
    /// Rank (approximate height) for union by rank heuristic.
    rank: Vec<usize>,
    /// All edges currently in the graph (for rebuild on removal).
    edges: Vec<(usize, usize)>,
    /// Total number of elements.
    size: usize,
}

impl IncrementalUnionFind {
    /// Create a new union-find with `size` initially disjoint elements.
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
            edges: Vec::new(),
            size,
        }
    }

    /// Find the representative (root) of the set containing `x` with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        if x >= self.size {
            panic!("Element {} out of bounds (size: {})", x, self.size);
        }

        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    /// Union the sets containing `a` and `b` using union by rank.
    /// This is the internal implementation that doesn't record the edge.
    fn union_internal(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            return; // Already in the same set
        }

        // Union by rank
        match self.rank[root_a].cmp(&self.rank[root_b]) {
            std::cmp::Ordering::Less => {
                self.parent[root_a] = root_b;
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_b] = root_a;
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_b] = root_a;
                self.rank[root_a] += 1;
            }
        }
    }

    /// Union the sets containing `a` and `b`. This is the public API.
    pub fn union(&mut self, a: usize, b: usize) {
        self.union_internal(a, b);
    }

    /// Check if `a` and `b` are in the same connected component.
    pub fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    /// Add an edge between `a` and `b`, recording it and performing union.
    pub fn add_edge(&mut self, a: usize, b: usize) {
        if a >= self.size || b >= self.size {
            panic!("Edge ({}, {}) out of bounds (size: {})", a, b, self.size);
        }

        // Avoid duplicate edges
        if !self.edges.contains(&(a, b)) && !self.edges.contains(&(b, a)) {
            self.edges.push((a, b));
        }

        self.union_internal(a, b);
    }

    /// Remove an edge between `a` and `b`, then rebuild from the remaining edges.
    ///
    /// This is O(n + e) where n is the number of elements and e is the number
    /// of edges, but in practice only affects the component that contained the edge.
    pub fn remove_edge(&mut self, a: usize, b: usize) {
        // Remove the edge from our edge list
        self.edges.retain(|&(x, y)| !((x == a && y == b) || (x == b && y == a)));

        // Full rebuild: reset all parents and ranks, then replay all edges
        self.rebuild();
    }

    /// Rebuild the entire union-find structure from the edge list.
    /// This is called after edge removal to handle potential component splits.
    pub fn rebuild(&mut self) {
        // Reset all parents to self (disjoint sets)
        for i in 0..self.size {
            self.parent[i] = i;
            self.rank[i] = 0;
        }

        // Replay all edges (clone to avoid borrow checker issues)
        let edges = self.edges.clone();
        for (a, b) in edges {
            self.union_internal(a, b);
        }
    }

    /// Get all connected components as a map from root → members.
    /// Note: This mutates self due to path compression in find().
    pub fn components(&mut self) -> HashMap<usize, Vec<usize>> {
        let mut result: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..self.size {
            let root = self.find(i);
            result.entry(root).or_default().push(i);
        }

        result
    }

    /// Count the number of distinct connected components.
    pub fn component_count(&mut self) -> usize {
        let mut roots = std::collections::HashSet::new();
        for i in 0..self.size {
            roots.insert(self.find(i));
        }
        roots.len()
    }

    /// Resize the union-find to accommodate `new_size` elements.
    /// New elements are initialized as singleton components.
    pub fn resize(&mut self, new_size: usize) {
        if new_size > self.size {
            let old_size = self.size;
            self.parent.extend(old_size..new_size);
            self.rank.extend(std::iter::repeat(0).take(new_size - old_size));
            self.size = new_size;
        } else if new_size < self.size {
            // Shrinking: remove edges involving elements >= new_size
            self.edges.retain(|&(a, b)| a < new_size && b < new_size);
            self.parent.truncate(new_size);
            self.rank.truncate(new_size);
            self.size = new_size;
            self.rebuild(); // Rebuild to reflect removed edges
        }
    }

    /// Get the current size (number of elements).
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the union-find is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the number of edges currently stored.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let uf = IncrementalUnionFind::new(5);
        assert_eq!(uf.len(), 5);
        assert!(!uf.is_empty());
    }

    #[test]
    fn test_find_initially_disjoint() {
        let mut uf = IncrementalUnionFind::new(5);
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }
    }

    #[test]
    fn test_union_basic() {
        let mut uf = IncrementalUnionFind::new(5);
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
        assert!(uf.connected(0, 1));
    }

    #[test]
    fn test_union_transitive() {
        let mut uf = IncrementalUnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);

        assert!(uf.connected(0, 1));
        assert!(uf.connected(0, 2));
        assert!(uf.connected(1, 2));
        assert!(uf.connected(3, 4));
        assert!(!uf.connected(0, 3));
        assert!(!uf.connected(2, 4));
    }

    #[test]
    fn test_add_edge() {
        let mut uf = IncrementalUnionFind::new(4);
        uf.add_edge(0, 1);
        uf.add_edge(2, 3);

        assert!(uf.connected(0, 1));
        assert!(uf.connected(2, 3));
        assert!(!uf.connected(0, 2));

        assert_eq!(uf.component_count(), 2);
    }

    #[test]
    fn test_remove_edge_split() {
        let mut uf = IncrementalUnionFind::new(4);
        uf.add_edge(0, 1);
        uf.add_edge(1, 2);
        uf.add_edge(2, 3);

        // All connected
        assert_eq!(uf.component_count(), 1);
        assert!(uf.connected(0, 3));

        // Remove edge (1, 2) - this should split into two components
        uf.remove_edge(1, 2);

        assert_eq!(uf.component_count(), 2);
        assert!(uf.connected(0, 1));
        assert!(uf.connected(2, 3));
        assert!(!uf.connected(0, 2));
        assert!(!uf.connected(1, 3));
    }

    #[test]
    fn test_remove_edge_no_split() {
        let mut uf = IncrementalUnionFind::new(4);
        // Create a cycle: 0-1-2-0
        uf.add_edge(0, 1);
        uf.add_edge(1, 2);
        uf.add_edge(2, 0);

        assert_eq!(uf.component_count(), 2); // {0,1,2} and {3}

        // Remove one edge from the cycle - should remain connected
        uf.remove_edge(0, 1);

        assert_eq!(uf.component_count(), 2); // Still {0,1,2} and {3}
        assert!(uf.connected(0, 1)); // Still connected via 2
    }

    #[test]
    fn test_components() {
        let mut uf = IncrementalUnionFind::new(6);
        uf.add_edge(0, 1);
        uf.add_edge(1, 2);
        uf.add_edge(3, 4);

        let comps = uf.components();
        assert_eq!(comps.len(), 3); // {0,1,2}, {3,4}, {5}

        // Find which components exist
        let mut comp_sizes: Vec<usize> = comps.values().map(|v| v.len()).collect();
        comp_sizes.sort_unstable();
        assert_eq!(comp_sizes, vec![1, 2, 3]);
    }

    #[test]
    fn test_component_count() {
        let mut uf = IncrementalUnionFind::new(5);
        assert_eq!(uf.component_count(), 5);

        uf.union(0, 1);
        assert_eq!(uf.component_count(), 4);

        uf.union(2, 3);
        assert_eq!(uf.component_count(), 3);

        uf.union(0, 2);
        assert_eq!(uf.component_count(), 2);
    }

    #[test]
    fn test_resize_grow() {
        let mut uf = IncrementalUnionFind::new(3);
        uf.add_edge(0, 1);

        uf.resize(5);
        assert_eq!(uf.len(), 5);
        assert!(uf.connected(0, 1));
        assert!(!uf.connected(0, 3));
        assert_eq!(uf.component_count(), 4); // {0,1}, {2}, {3}, {4}
    }

    #[test]
    fn test_resize_shrink() {
        let mut uf = IncrementalUnionFind::new(5);
        uf.add_edge(0, 1);
        uf.add_edge(2, 3);
        uf.add_edge(3, 4);

        uf.resize(3); // Remove elements 3 and 4
        assert_eq!(uf.len(), 3);
        assert!(uf.connected(0, 1));
        assert_eq!(uf.component_count(), 2); // {0,1}, {2}
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_find_out_of_bounds() {
        let mut uf = IncrementalUnionFind::new(5);
        uf.find(10);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_add_edge_out_of_bounds() {
        let mut uf = IncrementalUnionFind::new(5);
        uf.add_edge(0, 10);
    }

    #[test]
    fn test_duplicate_edge_idempotent() {
        let mut uf = IncrementalUnionFind::new(3);
        uf.add_edge(0, 1);
        uf.add_edge(0, 1); // Duplicate
        uf.add_edge(1, 0); // Reverse duplicate

        assert_eq!(uf.edge_count(), 1); // Should only have one edge
        assert!(uf.connected(0, 1));
    }

    #[test]
    fn test_rebuild() {
        let mut uf = IncrementalUnionFind::new(4);
        uf.add_edge(0, 1);
        uf.add_edge(2, 3);

        // Manually call rebuild
        uf.rebuild();

        // Structure should be preserved
        assert!(uf.connected(0, 1));
        assert!(uf.connected(2, 3));
        assert!(!uf.connected(0, 2));
        assert_eq!(uf.component_count(), 2);
    }

    #[test]
    fn test_path_compression() {
        let mut uf = IncrementalUnionFind::new(5);
        // Create a long chain: 0-1-2-3-4
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);

        // All should have the same root
        let root = uf.find(0);
        for i in 1..5 {
            assert_eq!(uf.find(i), root);
        }
    }

    #[test]
    fn test_empty_union_find() {
        let uf = IncrementalUnionFind::new(0);
        assert!(uf.is_empty());
        assert_eq!(uf.len(), 0);
    }

    #[test]
    fn test_complex_add_remove_sequence() {
        let mut uf = IncrementalUnionFind::new(6);

        // Build a connected graph
        uf.add_edge(0, 1);
        uf.add_edge(1, 2);
        uf.add_edge(3, 4);
        uf.add_edge(4, 5);
        assert_eq!(uf.component_count(), 2);

        // Connect the two components
        uf.add_edge(2, 3);
        assert_eq!(uf.component_count(), 1);

        // Remove the bridge - should split back to 2
        uf.remove_edge(2, 3);
        assert_eq!(uf.component_count(), 2);

        // Remove another edge
        uf.remove_edge(1, 2);
        assert_eq!(uf.component_count(), 3); // {0,1}, {2}, {3,4,5}
    }
}
