"""
graph.py — Florida Supply Chain & Logistics Network
=====================================================
Defines the LogisticsGraph class, which models an undirected, weighted graph
using an adjacency list.  Also contains a synthetic dataset generator that
seeds the graph with the 20 real Florida cities used throughout this project
and can scale up to arbitrary sizes for the experimental analysis section.

Course  : COT 4400 — Analysis of Algorithms
Project : Graph Algorithms in Network Optimization
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------
# Adjacency list: node → list of (neighbour, weight) pairs
AdjList = Dict[int, List[Tuple[int, float]]]


# ---------------------------------------------------------------------------
# Fixed dataset — 20 Florida cities used as the "real-world" scenario
# ---------------------------------------------------------------------------
FLORIDA_CITIES: List[str] = [
    "Jacksonville",   # 0
    "Miami",          # 1
    "Tampa",          # 2
    "Orlando",        # 3
    "St. Petersburg", # 4
    "Hialeah",        # 5
    "Tallahassee",    # 6
    "Port St. Lucie", # 7
    "Cape Coral",     # 8
    "Fort Lauderdale",# 9
    "Pembroke Pines", # 10
    "Hollywood",      # 11
    "Gainesville",    # 12
    "Miramar",        # 13
    "Coral Springs",  # 14
    "Clearwater",     # 15
    "Palm Bay",       # 16
    "Pompano Beach",  # 17
    "West Palm Beach",# 18
    "Lakeland",       # 19
]

# Approximate straight-line distances (miles) between select city pairs that
# form a realistic Florida road network.  Every pair listed here becomes an
# undirected edge in the 20-city base graph.
FLORIDA_EDGES: List[Tuple[int, int, float]] = [
    # Jacksonville corridor (north Florida)
    (0,  6,  163.0),  # Jacksonville ↔ Tallahassee
    (0, 12,   71.0),  # Jacksonville ↔ Gainesville
    (0,  3,  140.0),  # Jacksonville ↔ Orlando
    # Gainesville hub
    (12,  6,   97.0), # Gainesville ↔ Tallahassee
    (12,  2,  123.0), # Gainesville ↔ Tampa
    (12,  3,  111.0), # Gainesville ↔ Orlando
    # Central Florida
    (3,  2,   84.0),  # Orlando ↔ Tampa
    (3, 19,   55.0),  # Orlando ↔ Lakeland
    (3, 16,   72.0),  # Orlando ↔ Palm Bay
    (3,  7,  114.0),  # Orlando ↔ Port St. Lucie
    # Tampa Bay area
    (2,  4,   20.0),  # Tampa ↔ St. Petersburg
    (2, 15,   22.0),  # Tampa ↔ Clearwater
    (2, 19,   35.0),  # Tampa ↔ Lakeland
    (4, 15,    5.0),  # St. Petersburg ↔ Clearwater
    # Atlantic coast corridor
    (16,  7,   72.0), # Palm Bay ↔ Port St. Lucie
    (7, 18,   47.0),  # Port St. Lucie ↔ West Palm Beach
    (18,  9,   45.0), # West Palm Beach ↔ Fort Lauderdale
    (18, 17,   56.0), # West Palm Beach ↔ Pompano Beach
    # South Florida metro cluster
    (9, 17,   11.0),  # Fort Lauderdale ↔ Pompano Beach
    (9, 14,   13.0),  # Fort Lauderdale ↔ Coral Springs
    (9, 11,    7.0),  # Fort Lauderdale ↔ Hollywood
    (9, 10,   10.0),  # Fort Lauderdale ↔ Pembroke Pines
    (9, 13,   12.0),  # Fort Lauderdale ↔ Miramar
    (11, 13,   9.0),  # Hollywood ↔ Miramar
    (10, 13,   5.0),  # Pembroke Pines ↔ Miramar
    (13,  5,   6.0),  # Miramar ↔ Hialeah
    (5,  1,    8.0),  # Hialeah ↔ Miami
    (1,  9,   30.0),  # Miami ↔ Fort Lauderdale
    # Gulf coast / west
    (8,  4,   87.0),  # Cape Coral ↔ St. Petersburg
    (8,  2,   98.0),  # Cape Coral ↔ Tampa
    # Cross-state links
    (19,  7,  133.0), # Lakeland ↔ Port St. Lucie
    (16,  1,  161.0), # Palm Bay ↔ Miami
    (6,  2,  254.0),  # Tallahassee ↔ Tampa (I-10/I-75)
]


# ===========================================================================
class LogisticsGraph:
    """
    Undirected, weighted graph representing a supply-chain logistics network.

    Internally uses an adjacency list (dict of lists) which gives:
        • O(V + E) space complexity
        • O(degree(v)) edge lookup per vertex
        • O(1) amortised vertex insertion

    Parameters
    ----------
    num_nodes : int
        Total number of vertices to pre-allocate.
    city_names : list[str] | None
        Optional label list; falls back to "City_<i>" if omitted or shorter
        than num_nodes.
    """

    def __init__(
        self,
        num_nodes: int = 0,
        city_names: Optional[List[str]] = None,
    ) -> None:
        self._num_nodes: int = num_nodes
        self._num_edges: int = 0

        # Primary storage: node id → [(neighbour_id, weight), ...]
        self._adj: AdjList = {i: [] for i in range(num_nodes)}

        # Human-readable labels for each vertex index
        if city_names is not None:
            self._labels: Dict[int, str] = {
                i: (city_names[i] if i < len(city_names) else f"City_{i}")
                for i in range(num_nodes)
            }
        else:
            self._labels = {i: f"City_{i}" for i in range(num_nodes)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        """Return the current number of vertices."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Return the current number of undirected edges."""
        return self._num_edges

    @property
    def adj(self) -> AdjList:
        """Read-only view of the adjacency list."""
        return self._adj

    @property
    def labels(self) -> Dict[int, str]:
        """Mapping from vertex index to city name."""
        return self._labels

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_node(self, node_id: int, label: Optional[str] = None) -> None:
        """
        Insert a new vertex.  No-op if the vertex already exists.

        Parameters
        ----------
        node_id : int   Unique integer identifier for the vertex.
        label   : str   Optional human-readable city name.
        """
        if node_id not in self._adj:
            self._adj[node_id] = []
            self._labels[node_id] = label if label else f"City_{node_id}"
            self._num_nodes += 1

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """
        Add an undirected weighted edge (u, v) with the given weight.

        Both vertices must already exist.  Parallel edges are allowed
        (the generator avoids them, but the data structure does not
        enforce uniqueness for generality).

        Parameters
        ----------
        u, v   : int    Vertex identifiers for both endpoints.
        weight : float  Non-negative road distance / cost in miles.
        """
        if u not in self._adj or v not in self._adj:
            raise ValueError(
                f"Cannot add edge ({u}, {v}): one or both vertices are missing."
            )
        self._adj[u].append((v, weight))
        self._adj[v].append((u, weight))
        self._num_edges += 1

    def get_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """
        Return the neighbour list of *node* as (neighbour_id, weight) pairs.

        Parameters
        ----------
        node : int   Vertex whose neighbours are requested.
        """
        return self._adj.get(node, [])

    def get_all_edges(self) -> List[Tuple[int, int, float]]:
        """
        Return every undirected edge exactly once as (u, v, weight) triples.
        Useful for Kruskal's algorithm which needs a flat edge list.
        """
        seen: set[frozenset] = set()
        edges: List[Tuple[int, int, float]] = []
        for u in self._adj:
            for v, w in self._adj[u]:
                key = frozenset((u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v, w))
        return edges

    def label(self, node: int) -> str:
        """Return the human-readable name for *node*."""
        return self._labels.get(node, f"Node_{node}")

    def __repr__(self) -> str:
        return (
            f"LogisticsGraph(vertices={self._num_nodes}, "
            f"edges={self._num_edges})"
        )


# ===========================================================================
class DatasetGenerator:
    """
    Factory that builds LogisticsGraph instances for experiments.

    Two construction modes are supported:

    1. ``build_florida_graph()``
       Returns the canonical 20-city Florida logistics network with
       realistic road distances hard-coded in FLORIDA_EDGES.

    2. ``build_random_graph(n, density, weight_range, seed)``
       Returns an *n*-vertex random graph where each possible edge is
       included with probability *density*.  Used by experiment.py to
       stress-test algorithms at different sizes and sparsity levels.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def build_florida_graph() -> LogisticsGraph:
        """
        Construct the canonical 20-city Florida supply-chain graph.

        Returns
        -------
        LogisticsGraph
            Populated with FLORIDA_CITIES vertices and FLORIDA_EDGES.
        """
        g = LogisticsGraph(
            num_nodes=len(FLORIDA_CITIES),
            city_names=FLORIDA_CITIES,
        )
        for u, v, w in FLORIDA_EDGES:
            g.add_edge(u, v, w)
        return g

    # ------------------------------------------------------------------
    @staticmethod
    def build_random_graph(
        n: int,
        density: float = 0.3,
        weight_range: Tuple[float, float] = (10.0, 500.0),
        seed: Optional[int] = 42,
    ) -> LogisticsGraph:
        """
        Generate a random undirected weighted graph for benchmarking.

        The graph is guaranteed to be connected: a random spanning tree is
        built first (ensuring connectivity), then additional edges are added
        according to the density parameter.

        Parameters
        ----------
        n            : int    Number of vertices.
        density      : float  Probability [0, 1] that any non-tree edge exists.
        weight_range : tuple  (min_weight, max_weight) for edge costs.
        seed         : int    RNG seed for reproducibility (None → random).

        Returns
        -------
        LogisticsGraph
            Randomly connected graph with ~density × n×(n-1)/2 edges.
        """
        if not (0.0 <= density <= 1.0):
            raise ValueError("density must be in [0.0, 1.0].")

        rng = random.Random(seed)
        lo, hi = weight_range

        # Assign generic city labels for large synthetic graphs
        labels = [f"City_{i}" for i in range(n)]
        g = LogisticsGraph(num_nodes=n, city_names=labels)

        # Step 1 — Random spanning tree guarantees full connectivity.
        # Shuffle vertices, then connect each new vertex to a random
        # already-included vertex (Prüfer / random-walk approach).
        nodes = list(range(n))
        rng.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i]
            v = nodes[rng.randint(0, i - 1)]
            w = round(rng.uniform(lo, hi), 2)
            g.add_edge(u, v, w)

        # Step 2 — Probabilistically add remaining edges to reach target density.
        # Track existing edges to avoid duplicates.
        existing: set[frozenset] = {
            frozenset((u, v)) for u, v, _ in g.get_all_edges()
        }
        for u in range(n):
            for v in range(u + 1, n):
                if frozenset((u, v)) not in existing:
                    if rng.random() < density:
                        w = round(rng.uniform(lo, hi), 2)
                        g.add_edge(u, v, w)
                        existing.add(frozenset((u, v)))

        return g
