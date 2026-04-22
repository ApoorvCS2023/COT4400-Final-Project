"""
algorithms.py — Graph Algorithm Implementations
================================================
Contains the GraphAlgorithms class with three canonical graph algorithms
applied to the Florida Supply Chain & Logistics Network:

    1. BFS   — Breadth-First Search for reachability traversal
    2. Dijkstra — Single-source shortest paths (non-negative weights)
    3. Kruskal   — Minimum Spanning Tree via greedy edge selection

Each method is self-contained, heavily documented, and returns structured
results so that experiment.py can measure both runtime and memory cost
without modifying the core logic.

Course  : COT 4400 — Analysis of Algorithms
Project : Graph Algorithms in Network Optimization
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple

# Local import — only the graph data structure is needed here
from graph import LogisticsGraph


# ---------------------------------------------------------------------------
# Helper: Disjoint-Set Union (Union-Find) — used exclusively by Kruskal
# ---------------------------------------------------------------------------
class _UnionFind:
    """
    Path-compressed, union-by-rank Disjoint-Set Union structure.

    Supports two operations in near-O(1) amortised time (O(α(n)) via the
    inverse-Ackermann function):
        • find(x)    — return the representative (root) of x's component
        • union(x,y) — merge the components containing x and y

    Space: O(n) for the parent and rank arrays.
    """

    def __init__(self, n: int) -> None:
        """
        Initialise n singleton sets.

        Parameters
        ----------
        n : int   Number of elements (vertices), labelled 0 … n-1.
        """
        # parent[i] points to i's parent; initially each node is its own root
        self.parent: List[int] = list(range(n))
        # rank[i] is an upper bound on the height of i's subtree
        self.rank: List[int] = [0] * n

    def find(self, x: int) -> int:
        """
        Return the root of the set containing x.

        Path compression flattens the tree so future queries on the same
        node are answered in O(1).

        Parameters
        ----------
        x : int   Element whose set representative is requested.
        """
        if self.parent[x] != x:
            # Recursively find root AND compress path in one pass
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing x and y.

        Union-by-rank keeps the tree shallow: the root of the smaller-rank
        tree is attached under the root of the larger-rank tree.

        Parameters
        ----------
        x, y : int   Elements whose sets should be merged.

        Returns
        -------
        bool
            True  if x and y were in different sets (merge occurred).
            False if they were already in the same set (cycle detected).
        """
        root_x = self.find(x)
        root_y = self.find(y)

        # Already connected — adding this edge would form a cycle
        if root_x == root_y:
            return False

        # Attach smaller tree under larger tree to keep height bounded
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # Equal rank: arbitrarily make root_y the new root and bump rank
            self.parent[root_x] = root_y
            self.rank[root_y] += 1

        return True


# ===========================================================================
class GraphAlgorithms:
    """
    Collection of graph algorithms applied to a LogisticsGraph.

    All methods are static so the class acts as a clean namespace; no
    internal state is mutated between calls.

    Usage
    -----
    >>> from graph import DatasetGenerator
    >>> from algorithms import GraphAlgorithms
    >>> g = DatasetGenerator.build_florida_graph()
    >>> bfs_result   = GraphAlgorithms.bfs(g, source=0)
    >>> dist, prev   = GraphAlgorithms.dijkstra(g, source=0)
    >>> mst_edges, w = GraphAlgorithms.kruskal(g)
    """

    # ======================================================================
    # 1. BFS — Breadth-First Search
    # ======================================================================
    @staticmethod
    def bfs(
        graph: LogisticsGraph,
        source: int,
    ) -> Dict[int, int]:
        """
        Perform a Breadth-First Search from *source* and return the
        discovery order and parent map for every reachable vertex.

        BFS explores vertices level by level (closest nodes first), which
        makes it ideal for:
            • Finding all cities reachable from a distribution hub
            • Computing the *unweighted* shortest hop-count path
            • Detecting connected components

        Algorithm outline
        -----------------
        1. Enqueue the source; mark it visited.
        2. While the queue is non-empty:
           a. Dequeue the front vertex u.
           b. For every neighbour v of u:
              – If v is unvisited, mark it visited, record its parent,
                and enqueue it.
        3. Return the parent map (BFS tree).

        Time Complexity  : O(V + E)  — each vertex and edge visited once.
        Space Complexity : O(V)      — visited set + queue + parent map.

        Parameters
        ----------
        graph  : LogisticsGraph   The logistics network to traverse.
        source : int              Starting vertex (distribution hub index).

        Returns
        -------
        dict[int, int]
            parent[v] = u means u was the vertex that first discovered v.
            parent[source] = -1 (sentinel for "no parent").
            Only reachable vertices appear as keys.
        """
        # Visited set: O(V) space, O(1) average membership test
        visited: set[int] = set()
        # Queue for FIFO traversal; deque gives O(1) popleft
        queue: deque[int] = deque()
        # parent[v] = the vertex that discovered v; used to reconstruct paths
        parent: Dict[int, int] = {}

        # --- Initialisation ---
        visited.add(source)
        queue.append(source)
        parent[source] = -1  # Sentinel: source has no parent in the BFS tree

        # --- Main BFS loop ---
        while queue:
            # Dequeue the next vertex to process (FIFO order)
            u = queue.popleft()

            # Inspect every neighbour of u
            for v, _weight in graph.get_neighbors(u):
                # Weight is ignored in BFS (we care about hop count, not cost)
                if v not in visited:
                    visited.add(v)
                    parent[v] = u        # u is the BFS-tree parent of v
                    queue.append(v)      # Schedule v for exploration

        return parent

    @staticmethod
    def bfs_reachable(
        graph: LogisticsGraph,
        source: int,
    ) -> List[int]:
        """
        Convenience wrapper: return a list of all vertices reachable from
        *source* in BFS discovery order (source is always first).

        Parameters
        ----------
        graph  : LogisticsGraph
        source : int

        Returns
        -------
        list[int]   Reachable vertex indices in discovery order.
        """
        parent = GraphAlgorithms.bfs(graph, source)
        # Sort by insertion order isn't guaranteed in older Pythons,
        # but dict preserves insertion order in CPython 3.7+.
        return list(parent.keys())

    # ======================================================================
    # 2. Dijkstra — Single-Source Shortest Paths
    # ======================================================================
    @staticmethod
    def dijkstra(
        graph: LogisticsGraph,
        source: int,
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        Compute the minimum-cost delivery route from *source* to every other
        city in the network using Dijkstra's algorithm.

        Dijkstra's algorithm is a greedy method that always relaxes the
        cheapest unprocessed vertex next.  It is correct for graphs with
        non-negative edge weights (all distances here are ≥ 0).

        Algorithm outline
        -----------------
        1. Initialise dist[source] = 0, dist[v] = ∞ for all other v.
        2. Push (0, source) onto a min-heap.
        3. While the heap is non-empty:
           a. Pop (d, u) — the vertex with the current known minimum cost.
           b. If d > dist[u], skip (stale entry from a previous relaxation).
           c. For each neighbour v of u with edge weight w:
              – If dist[u] + w < dist[v], update dist[v] and push to heap.
        4. Return distance map and predecessor map.

        Implementation note
        -------------------
        Python's heapq is a min-heap with O(log n) push/pop.  We use the
        "lazy deletion" pattern: stale (dist, node) pairs are left in the
        heap and discarded when popped with the guard at step 3b.

        Time Complexity  : O((V + E) log V) with a binary min-heap.
        Space Complexity : O(V + E)  — dist/prev arrays + heap entries.

        Parameters
        ----------
        graph  : LogisticsGraph   Undirected weighted logistics network.
        source : int              Index of the starting distribution hub.

        Returns
        -------
        dist : dict[int, float]
            Shortest-path cost from source to every reachable vertex.
            Unreachable vertices map to float('inf').
        prev : dict[int, int]
            prev[v] = u means the shortest path to v goes through u.
            prev[source] = -1; unreachable vertices map to -1.
        """
        INF = float("inf")

        # Initialise all distances to infinity; source is zero cost
        dist: Dict[int, float] = {v: INF for v in graph.adj}
        dist[source] = 0.0

        # Predecessor map for path reconstruction
        prev: Dict[int, int] = {v: -1 for v in graph.adj}

        # Min-heap entries: (tentative_distance, vertex)
        # Using a list as a heap via the heapq module
        heap: List[Tuple[float, int]] = [(0.0, source)]

        # --- Main Dijkstra loop ---
        while heap:
            # Extract the vertex with the smallest known tentative distance
            d, u = heapq.heappop(heap)

            # Lazy-deletion guard: if we already found a better path to u,
            # this heap entry is outdated — discard it
            if d > dist[u]:
                continue

            # Relax all edges leaving u
            for v, weight in graph.get_neighbors(u):
                candidate = dist[u] + weight  # Cost of path through u

                if candidate < dist[v]:
                    # Found a cheaper route to v; update and schedule
                    dist[v] = candidate
                    prev[v] = u
                    heapq.heappush(heap, (candidate, v))

        return dist, prev

    @staticmethod
    def reconstruct_path(
        prev: Dict[int, int],
        source: int,
        target: int,
    ) -> Optional[List[int]]:
        """
        Walk the predecessor map backwards to recover the shortest path
        from *source* to *target*.

        Parameters
        ----------
        prev   : dict[int, int]   Predecessor map returned by dijkstra().
        source : int              Start vertex.
        target : int              Destination vertex.

        Returns
        -------
        list[int] | None
            Ordered list of vertex indices from source to target,
            or None if target is unreachable from source.
        """
        if prev.get(target, -1) == -1 and target != source:
            return None  # target was never reached

        path: List[int] = []
        current = target
        while current != -1:
            path.append(current)
            current = prev[current]

        path.reverse()  # We built it backwards; flip to source → target

        # Sanity check: first element must be the source
        if path[0] != source:
            return None

        return path

    # ======================================================================
    # 3. Kruskal — Minimum Spanning Tree
    # ======================================================================
    @staticmethod
    def kruskal(
        graph: LogisticsGraph,
    ) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        Find the Minimum Spanning Tree (MST) of the logistics network using
        Kruskal's greedy algorithm.

        The MST is the cheapest subset of roads that keeps all distribution
        centres connected — minimising total infrastructure cost.

        Algorithm outline
        -----------------
        1. Collect all edges and sort them by weight ascending.
        2. Initialise a Union-Find structure with one set per vertex.
        3. Iterate over the sorted edges:
           a. For edge (u, v, w): call union(u, v).
           b. If u and v were in different components (no cycle), add the
              edge to the MST.
           c. Stop early when |MST edges| = V - 1.
        4. Return the MST edge list and total weight.

        Correctness relies on the Cut Property: the minimum-weight edge
        crossing any cut of the graph is always part of some MST.

        Time Complexity  : O(E log E) dominated by the initial edge sort.
                           Union-Find operations are O(E · α(V)) ≈ O(E).
        Space Complexity : O(V + E) — edge list + Union-Find arrays.

        Parameters
        ----------
        graph : LogisticsGraph   Undirected weighted logistics network.

        Returns
        -------
        mst_edges : list[tuple[int, int, float]]
            The V-1 edges chosen for the MST as (u, v, weight) triples.
        total_weight : float
            Sum of all MST edge weights (minimum total road cost).
        """
        # Gather every undirected edge exactly once
        all_edges: List[Tuple[int, int, float]] = graph.get_all_edges()

        # Step 1 — Sort edges by weight (ascending); ties broken arbitrarily
        # Python's sort is Timsort: O(E log E) worst case
        all_edges.sort(key=lambda edge: edge[2])

        n = graph.num_nodes
        # Step 2 — Each city starts in its own isolated component
        uf = _UnionFind(n)

        mst_edges: List[Tuple[int, int, float]] = []
        total_weight: float = 0.0

        # Step 3 — Greedily add the cheapest edge that doesn't form a cycle
        for u, v, weight in all_edges:
            # union() returns True only if u and v are in different components
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight

                # An MST of a connected graph has exactly V-1 edges;
                # we can stop as soon as that count is reached
                if len(mst_edges) == n - 1:
                    break

        return mst_edges, round(total_weight, 2)
