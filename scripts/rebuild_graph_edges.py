#!/usr/bin/env python3
"""
P0-2 + P1-3: Rebuild similar edges with new threshold + Leiden community detection.

- Removes existing similar edges below SIMILAR_EDGE_THRESHOLD (0.82)
- Runs Leiden community detection
- Saves updated graph.json + communities.json

Usage: python3 scripts/rebuild_graph_edges.py
"""

import json
import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

# Paths
GRAPH_PATH = os.path.expanduser("~/.hermes/graphs/hermes-memory/graph.json")
COMMUNITIES_PATH = os.path.expanduser("~/.hermes/graphs/hermes-memory/communities.json")
CAT_INDEX_PATH = os.path.expanduser("~/.hermes/graphs/hermes-memory/category_index.json")

SIMILAR_THRESHOLD = 0.82
SEMANTIC_DEDUP_THRESHOLD = 0.88  # Matches graph_updater.py's SEMANTIC_DEDUP_THRESHOLD
EMBEDDING_DIM = 1024


def load_graph():
    import networkx as nx
    with open(GRAPH_PATH) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def save_graph(graph):
    import networkx as nx
    graph_data = nx.node_link_data(graph)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, separators=(",", ":"))


def rebuild_similar_edges(graph):
    """Remove existing similar edges below threshold, recalculate from embeddings."""
    print("  Loading embeddings...")
    fact_nodes = []
    fact_embs = []
    for nid, ndata in graph.nodes(data=True):
        if ndata.get("type") != "fact":
            continue
        emb = ndata.get("embedding", [])
        if len(emb) == EMBEDDING_DIM:
            fact_nodes.append(nid)
            fact_embs.append(emb)

    if not fact_embs:
        print("  No embeddings found, skipping edge rebuild.")
        return 0

    # Normalize embeddings
    emb_matrix = np.array(fact_embs, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.maximum(norms, 1e-10)

    # Remove existing similar edges
    similar_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "similar"]
    graph.remove_edges_from(similar_edges)
    print(f"  Removed {len(similar_edges)} existing similar edges")

    # Compute similarity matrix (batch to avoid memory issues)
    BATCH = 200
    new_edges = 0
    n = len(fact_nodes)

    for i_start in range(0, n, BATCH):
        i_end = min(i_start + BATCH, n)
        batch_sim = emb_matrix[i_start:i_end] @ emb_matrix.T

        for i in range(i_start, i_end):
            for j in range(i + 1, n):
                sim = float(batch_sim[i - i_start, j])
                if SIMILAR_THRESHOLD <= sim < SEMANTIC_DEDUP_THRESHOLD:  # below dedup threshold
                    graph.add_edge(fact_nodes[i], fact_nodes[j],
                                   weight=round(sim, 4), type="similar")
                    new_edges += 1

        if (i_start // BATCH) % 5 == 0:
            print(f"    Processing batch {i_start // BATCH + 1}/{(n + BATCH - 1) // BATCH}...")

    return new_edges


def run_leiden_communities(graph):
    """Run Leiden community detection on the graph."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        print("  Warning: igraph or leidenalg not installed, skipping community detection.")
        return {}

    # Create subgraph with only similar edges for community detection
    similar_edges = [(u, v, d.get("weight", 1.0))
                     for u, v, d in graph.edges(data=True)
                     if d.get("type") == "similar"]

    if not similar_edges:
        print("  No similar edges found, skipping community detection.")
        return {}

    # Map node IDs to integers for igraph
    fact_nodes = set()
    for u, v, _ in similar_edges:
        fact_nodes.add(u)
        fact_nodes.add(v)
    node_list = list(fact_nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}

    # Build igraph
    g = ig.Graph()
    g.add_vertices(len(node_list))
    edges_with_weights = [(node_idx[u], node_idx[v], w) for u, v, w in similar_edges]
    g.add_edges([(e[0], e[1]) for e in edges_with_weights])
    g.es["weight"] = [e[2] for e in edges_with_weights]

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=42
    )

    communities = {}
    for i, community in enumerate(partition):
        comm_id = f"community_{i}"
        communities[comm_id] = [node_list[idx] for idx in community]

    print(f"  Leiden detected {len(communities)} communities")
    for cid, members in sorted(communities.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {cid}: {len(members)} nodes")

    # Save communities
    with open(COMMUNITIES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "communities": communities,
            "num_communities": len(communities),
            "algorithm": "leiden",
            "threshold": SIMILAR_THRESHOLD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, ensure_ascii=False)

    return communities


if __name__ == "__main__":
    print("=" * 60)
    print("Hermes Memory — Rebuild Edges + Leiden Communities")
    print("=" * 60)

    print("\n[1/3] Loading graph...")
    graph = load_graph()
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("\n[2/3] Rebuilding similar edges (threshold >= 0.82)...")
    new_similar = rebuild_similar_edges(graph)
    print(f"  Added {new_similar} new similar edges")

    print("\n[3/3] Running Leiden community detection...")
    communities = run_leiden_communities(graph)

    print("\n  Saving graph...")
    backup = GRAPH_PATH + f".pre_rebuild_{int(time.time())}"
    shutil.copy2(GRAPH_PATH, backup)
    save_graph(graph)

    # Update category index
    categories = {}
    for node, data in graph.nodes(data=True):
        cat = data.get("category", "general")
        categories.setdefault(cat, []).append(node)
    with open(CAT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "categories": categories,
            "num_categories": len(categories),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  {len(communities)} Leiden communities")
