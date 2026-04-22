"""
experiment.py — Experimental Analysis & Performance Comparison
==============================================================
Benchmarks Dijkstra and Kruskal across two dimensions:

    Dimension 1 — Graph Size  : n ∈ {20, 100, 500} vertices
    Dimension 2 — Density     : Sparse (10%) vs Dense (80%)

For each (size, density) combination we measure:
    • Wall-clock execution time  via time.perf_counter()
    • Peak heap memory usage     via tracemalloc

Results are printed to the console in a formatted table and saved as a
four-panel PNG figure (results.png) in the project directory.

Run this file directly:
    python experiment.py

Dependencies (install once):
    pip install matplotlib

Course  : COT 4400 — Analysis of Algorithms
Project : Graph Algorithms in Network Optimization
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")           # Non-interactive backend; no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from graph import DatasetGenerator, LogisticsGraph
from algorithms import GraphAlgorithms


# ===========================================================================
# Benchmark configuration
# ===========================================================================

# Graph sizes to evaluate (number of vertices)
SIZES: List[int] = [20, 100, 500]

# Edge-density levels: (label, probability)
DENSITIES: List[Tuple[str, float]] = [
    ("Sparse (10%)", 0.10),
    ("Dense  (80%)", 0.80),
]

# Number of repeated runs to average (reduces noise on small graphs)
REPEAT: int = 5

# Random seed for reproducibility across runs
SEED: int = 42

# Output file name for the chart
OUTPUT_FILE: str = "results.png"


# ===========================================================================
# Timing & memory measurement utility
# ===========================================================================

def measure(func: Callable[[], Any], repeat: int = REPEAT) -> Tuple[float, float]:
    """
    Execute *func* *repeat* times and return the average wall-clock time
    (seconds) and the peak memory allocation (kilobytes) across all runs.

    We use ``tracemalloc`` rather than raw RSS because it measures only the
    memory allocated *inside* the Python process for the function call,
    which is a fairer comparison between two algorithms sharing the same
    interpreter.

    Parameters
    ----------
    func   : callable   Zero-argument callable wrapping the algorithm call.
    repeat : int        Number of trial runs.

    Returns
    -------
    avg_time_sec : float   Mean execution time in seconds.
    avg_mem_kb   : float   Mean peak memory in kilobytes.
    """
    times: List[float] = []
    memories: List[float] = []

    for _ in range(repeat):
        # --- Memory measurement ---
        tracemalloc.start()                         # Begin tracking allocations
        t_start = time.perf_counter()               # High-resolution wall clock

        func()                                      # Run the target algorithm

        t_end = time.perf_counter()
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()                          # Release the tracer

        times.append(t_end - t_start)
        memories.append(peak_bytes / 1024)          # Convert bytes → kilobytes

    avg_time = sum(times) / repeat
    avg_mem  = sum(memories) / repeat
    return avg_time, avg_mem


# ===========================================================================
# Core experiment runner
# ===========================================================================

def run_experiment() -> Dict[str, Any]:
    """
    Build graphs for every (size, density) combination, run Dijkstra and
    Kruskal, collect measurements, and return all results in a structured
    dictionary.

    Returns
    -------
    dict with keys:
        "sizes"         : list[int]    — graph sizes tested
        "density_labels": list[str]    — human-readable density names
        "dijkstra_time" : list[list]   — [density_idx][size_idx] → seconds
        "dijkstra_mem"  : list[list]   — [density_idx][size_idx] → KB
        "kruskal_time"  : list[list]
        "kruskal_mem"   : list[list]
    """
    # Pre-allocate result containers indexed as [density_idx][size_idx]
    num_d = len(DENSITIES)
    num_s = len(SIZES)

    dijkstra_time = [[0.0] * num_s for _ in range(num_d)]
    dijkstra_mem  = [[0.0] * num_s for _ in range(num_d)]
    kruskal_time  = [[0.0] * num_s for _ in range(num_d)]
    kruskal_mem   = [[0.0] * num_s for _ in range(num_d)]

    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  Florida Supply Chain & Logistics Network — Algorithm Benchmark")
    print("=" * 70)
    print(f"  Sizes    : {SIZES}")
    print(f"  Densities: {[d[0] for d in DENSITIES]}")
    print(f"  Repeats  : {REPEAT} runs per configuration")
    print("=" * 70)
    header = f"{'Size':>6}  {'Density':<14}  {'Dijk Time(s)':>13}  {'Dijk Mem(KB)':>13}"
    header += f"  {'Krus Time(s)':>13}  {'Krus Mem(KB)':>13}"
    print(header)
    print("-" * 70)

    for di, (d_label, d_prob) in enumerate(DENSITIES):
        for si, n in enumerate(SIZES):

            # Build the graph once; share it between both algorithm runs
            # so we compare apples-to-apples on identical input data
            if n == 20:
                # Use the canonical Florida 20-city network for the base case
                g: LogisticsGraph = DatasetGenerator.build_florida_graph()
            else:
                g = DatasetGenerator.build_random_graph(
                    n=n,
                    density=d_prob,
                    seed=SEED,
                )

            source = 0  # Always start from vertex 0 (Jacksonville / City_0)

            # ----- Dijkstra measurement -----
            def run_dijkstra() -> None:
                GraphAlgorithms.dijkstra(g, source)

            d_time, d_mem = measure(run_dijkstra)
            dijkstra_time[di][si] = d_time
            dijkstra_mem[di][si]  = d_mem

            # ----- Kruskal measurement -----
            def run_kruskal() -> None:
                GraphAlgorithms.kruskal(g)

            k_time, k_mem = measure(run_kruskal)
            kruskal_time[di][si] = k_time
            kruskal_mem[di][si]  = k_mem

            # Print table row
            print(
                f"{n:>6}  {d_label:<14}  "
                f"{d_time:>13.6f}  {d_mem:>13.2f}  "
                f"{k_time:>13.6f}  {k_mem:>13.2f}"
            )

    print("=" * 70)

    return {
        "sizes":          SIZES,
        "density_labels": [d[0] for d in DENSITIES],
        "dijkstra_time":  dijkstra_time,
        "dijkstra_mem":   dijkstra_mem,
        "kruskal_time":   kruskal_time,
        "kruskal_mem":    kruskal_mem,
    }


# ===========================================================================
# Chart generation
# ===========================================================================

def plot_results(results: Dict[str, Any]) -> None:
    """
    Generate a 2×2 subplot figure comparing Dijkstra and Kruskal on:
        • Execution time  (top row)
        • Memory usage    (bottom row)
    across Sparse and Dense graph densities.

    The figure is saved to OUTPUT_FILE (results.png) in the working directory.

    Parameters
    ----------
    results : dict   Output from run_experiment().
    """
    sizes          = results["sizes"]
    d_labels       = results["density_labels"]
    dijkstra_time  = results["dijkstra_time"]
    dijkstra_mem   = results["dijkstra_mem"]
    kruskal_time   = results["kruskal_time"]
    kruskal_mem    = results["kruskal_mem"]

    # -----------------------------------------------------------------------
    # Visual style
    # -----------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")

    # Colour palette — consistent across all panels
    DIJK_COLORS = ["#2196F3", "#1565C0"]   # blue shades  (sparse, dense)
    KRUS_COLORS = ["#FF5722", "#B71C1C"]   # red  shades  (sparse, dense)

    # Marker styles differentiate the two algorithms on shared axes
    DIJK_MARKER = "o"
    KRUS_MARKER = "s"

    LINE_WIDTH  = 2.2
    MARKER_SIZE = 8

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    fig.suptitle(
        "Florida Supply Chain — Dijkstra vs Kruskal Performance\n"
        "COT 4400 · Graph Algorithms in Network Optimization",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # -----------------------------------------------------------------------
    # Helper: draw one panel
    # -----------------------------------------------------------------------
    def draw_panel(
        ax: plt.Axes,
        y_dijk: List[float],
        y_krus: List[float],
        d_color: str,
        k_color: str,
        title: str,
        ylabel: str,
    ) -> None:
        """
        Plot Dijkstra and Kruskal lines on *ax* for a single density level.
        """
        x = list(range(len(sizes)))         # Categorical x-positions
        x_labels = [str(s) for s in sizes]

        ax.plot(
            x, y_dijk,
            color=d_color,
            marker=DIJK_MARKER,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Dijkstra",
        )
        ax.plot(
            x, y_krus,
            color=k_color,
            marker=KRUS_MARKER,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            linestyle="--",
            label="Kruskal",
        )

        # Annotate data points with exact values
        for xi, (yd, yk) in enumerate(zip(y_dijk, y_krus)):
            ax.annotate(
                f"{yd:.4f}" if "Time" in ylabel else f"{yd:.1f}",
                (xi, yd),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color=d_color,
            )
            ax.annotate(
                f"{yk:.4f}" if "Time" in ylabel else f"{yk:.1f}",
                (xi, yk),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7.5,
                color=k_color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Graph Size (# Vertices)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter("%.4f") if "Time" in ylabel
            else mticker.FormatStrFormatter("%.1f")
        )

    # -----------------------------------------------------------------------
    # Top row — Execution Time
    # -----------------------------------------------------------------------
    for di in range(len(DENSITIES)):
        draw_panel(
            ax=axes[0][di],
            y_dijk=dijkstra_time[di],
            y_krus=kruskal_time[di],
            d_color=DIJK_COLORS[di],
            k_color=KRUS_COLORS[di],
            title=f"Execution Time — {d_labels[di]}",
            ylabel="Time (seconds)",
        )

    # -----------------------------------------------------------------------
    # Bottom row — Memory Usage
    # -----------------------------------------------------------------------
    for di in range(len(DENSITIES)):
        draw_panel(
            ax=axes[1][di],
            y_dijk=dijkstra_mem[di],
            y_krus=kruskal_mem[di],
            d_color=DIJK_COLORS[di],
            k_color=KRUS_COLORS[di],
            title=f"Peak Memory — {d_labels[di]}",
            ylabel="Memory (KB)",
        )

    # -----------------------------------------------------------------------
    # Save figure
    # -----------------------------------------------------------------------
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {OUTPUT_FILE}")


# ===========================================================================
# Demo: display the 20-city Florida network results
# ===========================================================================

def demo_florida_graph() -> None:
    """
    Run all three algorithms on the canonical 20-city Florida graph and
    print human-readable results to the console.
    """
    g = DatasetGenerator.build_florida_graph()

    print("\n" + "=" * 70)
    print("  DEMO — 20-City Florida Logistics Network")
    print("=" * 70)
    print(f"  Vertices : {g.num_nodes}  |  Edges : {g.num_edges}")
    print()

    # --- BFS from Jacksonville (index 0) ---
    print("[ BFS ] Reachable cities from Jacksonville (hub 0):")
    reachable = GraphAlgorithms.bfs_reachable(g, source=0)
    for idx in reachable:
        print(f"    {idx:>2} — {g.label(idx)}")

    # --- Dijkstra from Jacksonville (index 0) ---
    print("\n[ Dijkstra ] Shortest delivery routes from Jacksonville:")
    dist, prev = GraphAlgorithms.dijkstra(g, source=0)
    for node in sorted(dist):
        cost = dist[node]
        path = GraphAlgorithms.reconstruct_path(prev, 0, node)
        path_names = " -> ".join(g.label(p) for p in path) if path else "unreachable"
        print(f"    To {g.label(node):<20}: {cost:>7.1f} mi  |  {path_names}")

    # --- Kruskal MST ---
    print("\n[ Kruskal ] Minimum Spanning Tree (cheapest road network):")
    mst_edges, total_w = GraphAlgorithms.kruskal(g)
    for u, v, w in mst_edges:
        print(f"    {g.label(u):<22} <-> {g.label(v):<22}  {w:>7.1f} mi")
    print(f"\n    Total MST weight: {total_w:.1f} miles")
    print("=" * 70)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # 1. Show the Florida network demo first
    demo_florida_graph()

    # 2. Run the full benchmark and collect measurements
    results = run_experiment()

    # 3. Generate and save the comparison chart
    plot_results(results)

    print("\n  Experiment complete.  Open results.png to view the charts.\n")
