use std::collections::VecDeque;

/// Result of Dulmage-Mendelsohn decomposition.
#[derive(Clone, Debug)]
pub struct DMDecomposition {
    /// Under-determined part: more variables than constraints.
    pub under: DMBlock,
    /// Well-determined part: square blocks in topological order.
    pub well: Vec<DMBlock>,
    /// Over-determined part: more constraints than variables.
    pub over: DMBlock,
    /// Constraint indices that are unmatched in the maximum matching.
    /// These are the structurally redundant constraints.
    pub unmatched_constraints: Vec<usize>,
}

/// A block in the DM decomposition.
#[derive(Clone, Debug, Default)]
pub struct DMBlock {
    pub constraint_indices: Vec<usize>,
    pub variable_indices: Vec<usize>,
}

impl DMBlock {
    pub fn is_empty(&self) -> bool {
        self.constraint_indices.is_empty() && self.variable_indices.is_empty()
    }

    pub fn equation_count(&self) -> usize {
        self.constraint_indices.len()
    }

    pub fn variable_count(&self) -> usize {
        self.variable_indices.len()
    }
}

/// Perform Dulmage-Mendelsohn decomposition on a bipartite graph.
///
/// # Arguments
/// * `n_constraints` - number of constraint rows
/// * `n_variables` - number of variable columns
/// * `edges` - (constraint_idx, variable_idx) pairs representing the Jacobian sparsity pattern
///
/// # Returns
/// A DMDecomposition partitioning the graph into under/well/over-determined parts.
pub fn dm_decompose(
    n_constraints: usize,
    n_variables: usize,
    edges: &[(usize, usize)],
) -> DMDecomposition {
    // Handle empty graph
    if n_constraints == 0 && n_variables == 0 {
        return DMDecomposition {
            under: DMBlock::default(),
            well: vec![],
            over: DMBlock::default(),
            unmatched_constraints: vec![],
        };
    }

    // Build adjacency list for constraints → variables
    let mut adj_c_to_v: Vec<Vec<usize>> = vec![Vec::new(); n_constraints];
    let mut adj_v_to_c: Vec<Vec<usize>> = vec![Vec::new(); n_variables];

    for &(c, v) in edges {
        if c < n_constraints && v < n_variables {
            adj_c_to_v[c].push(v);
            adj_v_to_c[v].push(c);
        }
    }

    // Find maximum matching using Hopcroft-Karp
    let (match_c, match_v) = hopcroft_karp(n_constraints, n_variables, &adj_c_to_v, &adj_v_to_c);

    // Partition into under/well/over using alternating path analysis
    partition_dm(
        n_constraints,
        n_variables,
        &adj_c_to_v,
        &adj_v_to_c,
        &match_c,
        &match_v,
    )
}

/// Maximum bipartite matching using Hopcroft-Karp algorithm.
///
/// Returns (match_c, match_v) where:
/// - match_c[i] = Some(j) means constraint i is matched to variable j
/// - match_v[j] = Some(i) means variable j is matched to constraint i
fn hopcroft_karp(
    n_constraints: usize,
    n_variables: usize,
    adj_c_to_v: &[Vec<usize>],
    adj_v_to_c: &[Vec<usize>],
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut match_c = vec![None; n_constraints];
    let mut match_v = vec![None; n_variables];
    let mut dist = vec![0; n_constraints + 1];

    loop {
        // BFS to find blocking flow
        if !bfs_hopcroft(
            n_constraints,
            &match_c,
            &match_v,
            adj_c_to_v,
            &mut dist,
        ) {
            break; // No more augmenting paths
        }

        // DFS to find augmenting paths along BFS layers
        for c in 0..n_constraints {
            if match_c[c].is_none() {
                dfs_hopcroft(
                    c,
                    adj_c_to_v,
                    adj_v_to_c,
                    &mut match_c,
                    &mut match_v,
                    &mut dist,
                );
            }
        }
    }

    (match_c, match_v)
}

/// BFS phase of Hopcroft-Karp: build layer graph from unmatched constraints.
fn bfs_hopcroft(
    n_constraints: usize,
    match_c: &[Option<usize>],
    match_v: &[Option<usize>],
    adj_c_to_v: &[Vec<usize>],
    dist: &mut [usize],
) -> bool {
    let mut queue = VecDeque::new();

    // Initialize: all unmatched constraints at distance 0
    for c in 0..n_constraints {
        if match_c[c].is_none() {
            dist[c] = 0;
            queue.push_back(c);
        } else {
            dist[c] = usize::MAX;
        }
    }

    dist[n_constraints] = usize::MAX; // Sentinel for unmatched state

    while let Some(c) = queue.pop_front() {
        if dist[c] < dist[n_constraints] {
            // Explore neighbors
            for &v in &adj_c_to_v[c] {
                if let Some(next_c) = match_v[v] {
                    if dist[next_c] == usize::MAX {
                        dist[next_c] = dist[c] + 1;
                        queue.push_back(next_c);
                    }
                } else {
                    // Found an augmenting path to an unmatched variable
                    dist[n_constraints] = dist[c] + 1;
                }
            }
        }
    }

    dist[n_constraints] != usize::MAX
}

/// DFS phase of Hopcroft-Karp: find augmenting path along BFS layers.
fn dfs_hopcroft(
    c: usize,
    adj_c_to_v: &[Vec<usize>],
    adj_v_to_c: &[Vec<usize>],
    match_c: &mut [Option<usize>],
    match_v: &mut [Option<usize>],
    dist: &mut [usize],
) -> bool {
    if dist[c] == usize::MAX {
        return false;
    }

    for &v in &adj_c_to_v[c] {
        let next_c = match_v[v];

        if next_c.is_none() || (dist[next_c.unwrap()] == dist[c] + 1
            && dfs_hopcroft(next_c.unwrap(), adj_c_to_v, adj_v_to_c, match_c, match_v, dist))
        {
            // Found augmenting path - flip the matching
            match_v[v] = Some(c);
            match_c[c] = Some(v);
            return true;
        }
    }

    dist[c] = usize::MAX; // Mark as visited
    false
}

/// Partition the bipartite graph into under/well/over-determined parts.
fn partition_dm(
    n_constraints: usize,
    n_variables: usize,
    adj_c_to_v: &[Vec<usize>],
    adj_v_to_c: &[Vec<usize>],
    match_c: &[Option<usize>],
    match_v: &[Option<usize>],
) -> DMDecomposition {
    // Use alternating path analysis to partition
    // 1. Start from unmatched constraints, do alternating BFS to find over-determined part
    // 2. Start from unmatched variables, do alternating BFS to find under-determined part
    // 3. Remaining nodes form well-determined part

    let mut c_reachable = vec![false; n_constraints];
    let mut v_reachable = vec![false; n_variables];

    // BFS from unmatched constraints (over-determined part)
    let mut queue: VecDeque<(bool, usize)> = VecDeque::new();

    for c in 0..n_constraints {
        if match_c[c].is_none() {
            c_reachable[c] = true;
            queue.push_back((true, c)); // (is_constraint, index)
        }
    }

    while let Some((is_constraint, idx)) = queue.pop_front() {
        if is_constraint {
            // From constraint, follow edges to variables
            for &v in &adj_c_to_v[idx] {
                if !v_reachable[v] {
                    v_reachable[v] = true;
                    // Follow matched edge back to constraint
                    if let Some(next_c) = match_v[v] {
                        if !c_reachable[next_c] {
                            c_reachable[next_c] = true;
                            queue.push_back((true, next_c));
                        }
                    }
                }
            }
        }
    }

    // BFS from unmatched variables (under-determined part)
    let mut c_reachable_under = vec![false; n_constraints];
    let mut v_reachable_under = vec![false; n_variables];

    for v in 0..n_variables {
        if match_v[v].is_none() {
            v_reachable_under[v] = true;
            queue.push_back((false, v)); // (is_constraint, index)
        }
    }

    while let Some((is_constraint, idx)) = queue.pop_front() {
        if !is_constraint {
            // From variable, follow reverse edges to constraints
            for &c in &adj_v_to_c[idx] {
                if !c_reachable_under[c] {
                    c_reachable_under[c] = true;
                    // Follow matched edge back to variable
                    if let Some(next_v) = match_c[c] {
                        if !v_reachable_under[next_v] {
                            v_reachable_under[next_v] = true;
                            queue.push_back((false, next_v));
                        }
                    }
                }
            }
        }
    }

    // Partition nodes
    let mut over_constraints = Vec::new();
    let mut over_variables = Vec::new();
    let mut under_constraints = Vec::new();
    let mut under_variables = Vec::new();
    let mut well_constraints = Vec::new();
    let mut well_variables = Vec::new();

    for c in 0..n_constraints {
        if c_reachable[c] {
            over_constraints.push(c);
        } else if c_reachable_under[c] {
            under_constraints.push(c);
        } else {
            well_constraints.push(c);
        }
    }

    for v in 0..n_variables {
        if v_reachable[v] {
            over_variables.push(v);
        } else if v_reachable_under[v] {
            under_variables.push(v);
        } else {
            well_variables.push(v);
        }
    }

    // For the well-determined part, we should ideally find strongly connected
    // components to get a finer decomposition. For now, we return it as a single block.
    let well_blocks = if !well_constraints.is_empty() {
        vec![DMBlock {
            constraint_indices: well_constraints,
            variable_indices: well_variables,
        }]
    } else {
        vec![]
    };

    // Collect unmatched constraints (those with no matching variable)
    let unmatched_constraints: Vec<usize> = (0..n_constraints)
        .filter(|&c| match_c[c].is_none())
        .collect();

    DMDecomposition {
        under: DMBlock {
            constraint_indices: under_constraints,
            variable_indices: under_variables,
        },
        well: well_blocks,
        over: DMBlock {
            constraint_indices: over_constraints,
            variable_indices: over_variables,
        },
        unmatched_constraints,
    }
}

/// Find strongly connected components using Tarjan's algorithm.
/// This can be used to further decompose the well-determined block.
#[allow(dead_code)]
fn tarjan_scc(n: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut index = 0;
    let mut stack = Vec::new();
    let mut indices = vec![None; n];
    let mut lowlinks = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut sccs = Vec::new();

    fn strongconnect(
        v: usize,
        index: &mut usize,
        stack: &mut Vec<usize>,
        indices: &mut [Option<usize>],
        lowlinks: &mut [usize],
        on_stack: &mut [bool],
        sccs: &mut Vec<Vec<usize>>,
        adj: &[Vec<usize>],
    ) {
        indices[v] = Some(*index);
        lowlinks[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &adj[v] {
            if indices[w].is_none() {
                strongconnect(w, index, stack, indices, lowlinks, on_stack, sccs, adj);
                lowlinks[v] = lowlinks[v].min(lowlinks[w]);
            } else if on_stack[w] {
                lowlinks[v] = lowlinks[v].min(indices[w].unwrap());
            }
        }

        if lowlinks[v] == indices[v].unwrap() {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for v in 0..n {
        if indices[v].is_none() {
            strongconnect(
                v,
                &mut index,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut sccs,
                adj,
            );
        }
    }

    sccs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let dm = dm_decompose(0, 0, &[]);
        assert!(dm.under.is_empty());
        assert!(dm.over.is_empty());
        assert!(dm.well.is_empty());
    }

    #[test]
    fn test_perfect_matching_2x2() {
        // 2 constraints, 2 variables, perfect matching
        // c0 -> v0
        // c1 -> v1
        let edges = vec![(0, 0), (1, 1)];
        let dm = dm_decompose(2, 2, &edges);

        assert!(dm.under.is_empty());
        assert!(dm.over.is_empty());
        assert_eq!(dm.well.len(), 1);
        assert_eq!(dm.well[0].equation_count(), 2);
        assert_eq!(dm.well[0].variable_count(), 2);
    }

    #[test]
    fn test_over_determined() {
        // 3 constraints, 2 variables (more constraints than variables)
        // c0, c1, c2 all connect to v0, v1
        let edges = vec![
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
        ];
        let dm = dm_decompose(3, 2, &edges);

        // Should have non-empty over-determined part
        assert!(!dm.over.is_empty());
        assert!(dm.over.equation_count() > 0);
    }

    #[test]
    fn test_under_determined() {
        // 2 constraints, 3 variables (more variables than constraints)
        // c0 -> v0, v1
        // c1 -> v1, v2
        let edges = vec![
            (0, 0), (0, 1),
            (1, 1), (1, 2),
        ];
        let dm = dm_decompose(2, 3, &edges);

        // Should have non-empty under-determined part
        assert!(!dm.under.is_empty());
        assert!(dm.under.variable_count() > 0);
    }

    #[test]
    fn test_hopcroft_karp_simple() {
        // Simple 3x3 graph with perfect matching possible
        let adj_c_to_v = vec![
            vec![0],       // c0 -> v0
            vec![1],       // c1 -> v1
            vec![2],       // c2 -> v2
        ];
        let adj_v_to_c = vec![
            vec![0],       // v0 -> c0
            vec![1],       // v1 -> c1
            vec![2],       // v2 -> c2
        ];

        let (match_c, match_v) = hopcroft_karp(3, 3, &adj_c_to_v, &adj_v_to_c);

        // Check perfect matching
        assert_eq!(match_c[0], Some(0));
        assert_eq!(match_c[1], Some(1));
        assert_eq!(match_c[2], Some(2));

        assert_eq!(match_v[0], Some(0));
        assert_eq!(match_v[1], Some(1));
        assert_eq!(match_v[2], Some(2));
    }

    #[test]
    fn test_hopcroft_karp_augmenting_path() {
        // Graph where augmenting path is needed:
        // c0 -> v0, v1
        // c1 -> v0
        // c2 -> v1
        let adj_c_to_v = vec![
            vec![0, 1],    // c0 -> v0, v1
            vec![0],       // c1 -> v0
            vec![1],       // c2 -> v1
        ];
        let adj_v_to_c = vec![
            vec![0, 1],    // v0 -> c0, c1
            vec![0, 2],    // v1 -> c0, c2
        ];

        let (match_c, _match_v) = hopcroft_karp(3, 2, &adj_c_to_v, &adj_v_to_c);

        // Should match 2 out of 3 constraints
        let matched_count = match_c.iter().filter(|m| m.is_some()).count();
        assert_eq!(matched_count, 2);
    }

    #[test]
    fn test_block_triangular_structure() {
        // Create a block-triangular structure:
        // Block 1: c0, c1 <-> v0, v1
        // Block 2: c2, c3 <-> v2, v3 (depends on v1 from block 1)
        let edges = vec![
            // Block 1
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            // Block 2
            (2, 1), (2, 2),  // c2 depends on v1 (from block 1) and v2
            (3, 2), (3, 3),
        ];

        let dm = dm_decompose(4, 4, &edges);

        // Should be well-determined
        assert!(dm.under.is_empty());
        assert!(dm.over.is_empty());
        assert!(!dm.well.is_empty());
    }

    #[test]
    fn test_disconnected_components() {
        // Two separate perfect matchings
        let edges = vec![
            (0, 0),  // Component 1
            (1, 1),  // Component 2
        ];

        let dm = dm_decompose(2, 2, &edges);

        assert!(dm.under.is_empty());
        assert!(dm.over.is_empty());
        assert_eq!(dm.well.len(), 1);
        assert_eq!(dm.well[0].equation_count(), 2);
    }

    #[test]
    fn test_no_edges() {
        // Constraints and variables but no edges
        let dm = dm_decompose(2, 2, &[]);

        // All constraints are unmatched (over-determined)
        // All variables are unmatched (under-determined)
        // Actually, with no edges, both should be in their respective parts
        assert_eq!(dm.over.equation_count(), 2);
        assert_eq!(dm.under.variable_count(), 2);
    }

    #[test]
    fn test_dm_block_methods() {
        let block = DMBlock {
            constraint_indices: vec![0, 1, 2],
            variable_indices: vec![0, 1],
        };

        assert!(!block.is_empty());
        assert_eq!(block.equation_count(), 3);
        assert_eq!(block.variable_count(), 2);

        let empty = DMBlock::default();
        assert!(empty.is_empty());
        assert_eq!(empty.equation_count(), 0);
        assert_eq!(empty.variable_count(), 0);
    }

    #[test]
    fn test_tarjan_scc() {
        // Test the SCC algorithm on a simple directed graph
        // Graph: 0 -> 1 -> 2 -> 0 (cycle), 3 -> 4
        let adj = vec![
            vec![1],       // 0 -> 1
            vec![2],       // 1 -> 2
            vec![0],       // 2 -> 0
            vec![4],       // 3 -> 4
            vec![],        // 4 -> nothing
        ];

        let sccs = tarjan_scc(5, &adj);

        // Should find 3 SCCs: {0,1,2}, {3}, {4}
        assert_eq!(sccs.len(), 3);

        // Find the cycle SCC
        let cycle_scc = sccs.iter().find(|scc| scc.len() == 3);
        assert!(cycle_scc.is_some());
    }

    #[test]
    fn test_out_of_bounds_edges_ignored() {
        // Edge with out-of-bounds indices should be ignored
        let edges = vec![
            (0, 0),
            (1, 1),
            (10, 10), // Out of bounds, should be ignored
        ];

        let dm = dm_decompose(2, 2, &edges);

        // Should work as if only the first two edges exist
        assert!(!dm.well.is_empty());
    }

    #[test]
    fn test_larger_over_determined() {
        // 5 constraints, 3 variables
        let edges = vec![
            (0, 0), (0, 1),
            (1, 0), (1, 2),
            (2, 1), (2, 2),
            (3, 0),
            (4, 1),
        ];

        let dm = dm_decompose(5, 3, &edges);

        // More constraints than variables, so over-determined
        assert!(dm.over.equation_count() >= 2);
    }
}
