# Complexity Analysis — Graph Algorithms in Network Optimization
**Course:** COT 4400 — Analysis of Algorithms  
**Project:** Florida Supply Chain & Logistics Network  

---

## Table of Contents
1. [Problem Overview](#1-problem-overview)
2. [BFS — Breadth-First Search](#2-bfs--breadth-first-search)
3. [Dijkstra's Algorithm](#3-dijkstras-algorithm)
4. [Kruskal's Algorithm](#4-kruskals-algorithm)
5. [Comparative Summary](#5-comparative-summary)

---

## 1. Problem Overview

The Florida Supply Chain network is modelled as an **undirected, weighted graph**:

| Symbol | Meaning |
|--------|---------|
| `V`    | Number of vertices (distribution centres / cities) |
| `E`    | Number of edges (delivery routes) |
| `w(u,v)` | Non-negative weight of edge (u, v) — road distance in miles |

For a **dense** graph: `E = O(V²)` (up to V(V−1)/2 edges)  
For a **sparse** graph: `E = O(V)`

---

## 2. BFS — Breadth-First Search

### Purpose
Find **all cities reachable** from a starting distribution hub and compute the minimum **hop-count** (number of roads travelled, ignoring distance).

---

### Pseudocode

```
BFS(Graph G, source s):
  visited  ← empty set
  queue    ← empty queue
  parent   ← empty map

  ADD s TO visited
  ENQUEUE s INTO queue
  SET parent[s] ← -1          // sentinel: s has no parent

  WHILE queue IS NOT EMPTY:
    u ← DEQUEUE from queue

    FOR EACH (v, weight) IN G.neighbors(u):
      IF v NOT IN visited:
        ADD v TO visited
        SET parent[v] ← u
        ENQUEUE v INTO queue

  RETURN parent
```

---

### Time Complexity

| Case | Complexity | Reason |
|------|-----------|--------|
| Best | `O(1)` | Source has no neighbours (isolated vertex) |
| Average / Worst | **`O(V + E)`** | Every vertex enqueued/dequeued once (`O(V)`); every edge scanned exactly once across all adjacency lists (`O(E)`) |

**Derivation:**  
- The outer `WHILE` loop runs at most `V` times (each vertex is enqueued at most once due to the `visited` guard).  
- The inner `FOR EACH` loop totals `Σ deg(u)` iterations across all vertices `u`.  
- By the Handshaking Lemma: `Σ deg(u) = 2E` for undirected graphs.  
- Therefore total work = `O(V) + O(2E)` = **`O(V + E)`**.

---

### Space Complexity

| Structure | Space |
|-----------|-------|
| `visited` set | `O(V)` |
| `queue` (at most V elements) | `O(V)` |
| `parent` map | `O(V)` |
| **Total** | **`O(V)`** |

The adjacency list itself is `O(V + E)` but is shared and not duplicated by BFS.

---

### Best / Worst Case Discussion

**Best case:** The source vertex `s` is isolated (degree 0). BFS terminates after the first dequeue with no edges examined → `O(1)`.

**Worst case:** A complete graph where every vertex is reachable. All `V` vertices are enqueued and all `E = V(V−1)/2` edges are traversed → `O(V + V²)` = `O(V²)` for dense graphs, but the canonical expression stays `O(V + E)`.

**Unweighted shortest path:** BFS visits vertices in non-decreasing hop-count order, guaranteeing the first time a vertex `v` is dequeued, the path `s → v` via `parent` is the minimum-hop route.

---

## 3. Dijkstra's Algorithm

### Purpose
Find the **minimum-cost delivery route** (minimum total road miles) from a single hub `s` to every other distribution centre in the network.

---

### Pseudocode

```
DIJKSTRA(Graph G, source s):
  FOR EACH vertex v IN G:
    dist[v]  ← ∞
    prev[v]  ← -1

  dist[s] ← 0
  heap    ← MIN-HEAP
  PUSH (0, s) INTO heap

  WHILE heap IS NOT EMPTY:
    (d, u) ← POP-MIN from heap

    IF d > dist[u]:           // stale entry — skip (lazy deletion)
      CONTINUE

    FOR EACH (v, w) IN G.neighbors(u):
      candidate ← dist[u] + w

      IF candidate < dist[v]:
        dist[v] ← candidate
        prev[v] ← u
        PUSH (candidate, v) INTO heap

  RETURN dist, prev
```

---

### Time Complexity

The complexity depends on the priority-queue (min-heap) implementation.

#### Binary Min-Heap (used in this project — Python `heapq`)

| Operation | Per-call Cost | # of Calls | Total |
|-----------|--------------|------------|-------|
| `PUSH` to heap | `O(log V)` | `O(E)` | `O(E log V)` |
| `POP-MIN` | `O(log V)` | `O(V + E)` *(lazy)* | `O((V+E) log V)` |

**Overall: `O((V + E) log V)`**

With the lazy-deletion pattern, each edge can produce at most one heap insertion, so total heap operations are bounded by `E`.  The heap size is also bounded by `E`, hence each `log` factor is `log E ≤ log V²` = `2 log V` = `O(log V)`.

| Graph Type | E = ? | Dijkstra Cost |
|------------|-------|---------------|
| Sparse `E = O(V)` | `O(V)` | **`O(V log V)`** |
| Dense `E = O(V²)` | `O(V²)` | **`O(V² log V)`** |

> **Note:** A Fibonacci Heap would reduce this to `O(E + V log V)` (amortised), but is rarely used in practice due to high constant factors.

---

### Space Complexity

| Structure | Space |
|-----------|-------|
| `dist[]` and `prev[]` arrays | `O(V)` each |
| Min-heap (up to `E` entries with lazy deletion) | `O(E)` |
| **Total** | **`O(V + E)`** |

---

### Best / Worst Case Discussion

**Best case:** The source is connected only to one vertex and the target is that immediate neighbour. Only one edge is relaxed → effectively `O(log V)` for the single heap operation. In practice we still initialise `O(V)` distances.

**Worst case:** Dense graph (`E = O(V²)`) where every edge must be relaxed, and every relaxation produces a heap push → `O(V² log V)`.

**Correctness invariant (Greedy proof):**  
At each iteration, the vertex `u` extracted from the heap has `dist[u]` equal to its true shortest-path distance. This follows from the fact that all edge weights are non-negative, so no future path through unprocessed vertices can improve `dist[u]` once it is extracted. This is exactly where Dijkstra **fails** on negative weights (Bellman-Ford must be used instead).

---

## 4. Kruskal's Algorithm

### Purpose
Find the **Minimum Spanning Tree (MST)** — the cheapest set of roads that keeps all `V` distribution centres connected, minimising total infrastructure cost.

---

### Pseudocode

```
KRUSKAL(Graph G):
  edges ← GET_ALL_EDGES(G)       // collect undirected edges as (u, v, w)
  SORT edges BY weight ASCENDING  // O(E log E)

  uf ← UNION-FIND(G.num_nodes)   // initialise V singleton sets
  mst_edges   ← empty list
  total_weight ← 0

  FOR EACH (u, v, w) IN edges:
    IF UNION(uf, u, v) == TRUE:   // returns True if u, v in different sets
      APPEND (u, v, w) TO mst_edges
      total_weight ← total_weight + w

      IF |mst_edges| == V - 1:    // MST complete — early exit
        BREAK

  RETURN mst_edges, total_weight
```

#### Union-Find sub-procedures

```
FIND(uf, x):                       // Path compression
  IF uf.parent[x] ≠ x:
    uf.parent[x] ← FIND(uf, uf.parent[x])
  RETURN uf.parent[x]

UNION(uf, x, y):                   // Union by rank
  rx ← FIND(uf, x)
  ry ← FIND(uf, y)
  IF rx == ry: RETURN FALSE        // Same component — would form cycle

  IF rank[rx] < rank[ry]: parent[rx] ← ry
  ELSE IF rank[rx] > rank[ry]: parent[ry] ← rx
  ELSE: parent[rx] ← ry ; rank[ry] ← rank[ry] + 1

  RETURN TRUE
```

---

### Time Complexity

| Step | Cost |
|------|------|
| Collect all edges | `O(E)` |
| Sort edges by weight | **`O(E log E)`** |
| `V` Union-Find initialisations | `O(V)` |
| `E` UNION / FIND calls (path compression + union by rank) | `O(E · α(V))` ≈ `O(E)` |
| **Total** | **`O(E log E)`** |

Since `E ≤ V²`, we have `log E ≤ 2 log V`, so `O(E log E) = O(E log V)`.

`α(V)` is the inverse-Ackermann function which grows so slowly it is effectively a constant (≤ 4) for all practical input sizes.

| Graph Type | Dominant term |
|------------|--------------|
| Sparse `E = O(V)` | **`O(V log V)`** |
| Dense `E = O(V²)` | **`O(V² log V)`** |

---

### Space Complexity

| Structure | Space |
|-----------|-------|
| Edge list copy for sorting | `O(E)` |
| Union-Find `parent[]` and `rank[]` | `O(V)` |
| MST output list (V-1 edges) | `O(V)` |
| **Total** | **`O(V + E)`** |

---

### Best / Worst Case Discussion

**Best case:** The graph is already a spanning tree (`E = V − 1`). Sorting `V-1` edges takes `O(V log V)` and every edge is accepted (no cycle possible) → `O(V log V)`.

**Worst case:** Complete graph (`E = V(V−1)/2`). All `O(V²)` edges are sorted and Union-Find must process each one → `O(V² log V)`.

**Correctness (Cut Property):**  
Kruskal's correctness is proven via the **Cut Property**: for any partition of `V` into two non-empty sets `S` and `V − S`, the minimum-weight edge crossing the cut belongs to some MST. Since Kruskal always picks the globally lightest edge that doesn't form a cycle, it is always picking an edge that safely crosses some cut, guaranteeing optimality.

---

## 5. Comparative Summary

### Time Complexity at a Glance

| Algorithm | Best Case | Average Case | Worst Case | Notes |
|-----------|-----------|-------------|-----------|-------|
| **BFS** | `O(1)` | `O(V + E)` | `O(V + E)` | Ignores weights |
| **Dijkstra** | `O(V log V)` | `O((V+E) log V)` | `O(V² log V)` | Non-negative weights only |
| **Kruskal** | `O(V log V)` | `O(E log E)` | `O(V² log V)` | Requires full edge sort |

### Space Complexity

| Algorithm | Space |
|-----------|-------|
| **BFS** | `O(V)` |
| **Dijkstra** | `O(V + E)` |
| **Kruskal** | `O(V + E)` |

### When to Use Each

| Goal | Algorithm |
|------|-----------|
| "Which cities can we reach from Tampa?" | **BFS** |
| "What is the fastest/cheapest route from Miami to Jacksonville?" | **Dijkstra** |
| "What is the minimum road network to connect all 20 cities?" | **Kruskal** |

### Density Impact

- **Sparse graphs (`E ≈ V`):** All three algorithms perform near their best; Dijkstra and Kruskal both approach `O(V log V)`.
- **Dense graphs (`E ≈ V²`):** Dijkstra and Kruskal both degrade to `O(V² log V)`. BFS is unaffected by weights and remains `O(V + E)` = `O(V²)`.

### Practical Observations (from experiment.py results)

1. **Small graphs (n=20):** Both Dijkstra and Kruskal complete in microseconds; differences are negligible.
2. **Medium graphs (n=100):** Dijkstra is slightly faster in sparse graphs because heap operations are fewer; Kruskal's sort cost starts to show in dense graphs.
3. **Large graphs (n=500, dense):** The quadratic edge count (`~100,000` edges at 80% density) makes the sort in Kruskal and the heap operations in Dijkstra both significant. Memory usage grows visibly as `O(E)` structures are populated.

---

*End of Complexity Analysis*
