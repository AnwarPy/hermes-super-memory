#!/usr/bin/env python3
"""
P0-1 Cleanup: Remove API key / secret facts from existing JSONL files.
Also removes them from graph.json and rebuilds the graph.

Usage: python3 scripts/cleanup_api_keys.py [--dry-run]
"""

import json
import os
import sys
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from quality_gates import is_sensitive_fact

FACTS_DIR = os.path.expanduser("~/.hermes/memory/facts_auto")
GRAPH_PATH = os.path.expanduser("~/.hermes/graphs/hermes-memory/graph.json")
TRACKER_FILE = os.path.expanduser("~/.hermes/memory/.graph_tracker.json")

DRY_RUN = "--dry-run" in sys.argv

def cleanup_jsonl_files():
    """Remove sensitive facts from JSONL files. Returns count of removed facts."""
    removed = 0
    for filename in sorted(os.listdir(FACTS_DIR)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(FACTS_DIR, filename)
        kept_lines = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact = json.loads(line)
                    key = fact.get("key", "")
                    if is_sensitive_fact(key):
                        removed += 1
                        if DRY_RUN:
                            print(f"  REMOVE [{filename}]: {key[:80]}")
                        continue
                except json.JSONDecodeError:
                    continue
                kept_lines.append(line)

        if not DRY_RUN and len(kept_lines) < sum(1 for _ in open(filepath, encoding="utf-8")):
            # Write back only if something was removed
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(kept_lines) + "\n")

    return removed


def rebuild_graph():
    """Remove sensitive fact nodes from graph.json. Returns count of removed nodes."""
    if not os.path.exists(GRAPH_PATH):
        print("  Graph not found, skipping rebuild.")
        return 0, 0

    with open(GRAPH_PATH) as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Find sensitive fact nodes
    sensitive_ids = set()
    kept_nodes = []
    for node in nodes:
        if node.get("type") == "fact":
            content = node.get("content", "")
            if is_sensitive_fact(content):
                sensitive_ids.add(node["id"])
                if DRY_RUN:
                    print(f"  REMOVE NODE: {node['id']} — {content[:80]}")
                continue
        kept_nodes.append(node)

    # Remove edges connected to sensitive nodes
    kept_edges = []
    removed_edges = 0
    for edge in edges:
        src = edge.get("source", edge.get("from"))
        tgt = edge.get("target", edge.get("to"))
        if src in sensitive_ids or tgt in sensitive_ids:
            removed_edges += 1
            continue
        kept_edges.append(edge)

    if not DRY_RUN:
        # Backup before overwrite
        backup = GRAPH_PATH + f".pre_cleanup_{int(datetime.now().timestamp())}"
        shutil.copy2(GRAPH_PATH, backup)
        print(f"  Backed up graph to: {backup}")

        graph_data["nodes"] = kept_nodes
        graph_data["edges"] = kept_edges
        with open(GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False)

    return len(sensitive_ids), removed_edges


def cleanup_tracker(removed_hashes):
    """Remove deleted fact hashes from the graph tracker."""
    if not os.path.exists(TRACKER_FILE) or not removed_hashes:
        return
    with open(TRACKER_FILE) as f:
        tracker = json.load(f)
    current = set(tracker.get("indexed_fact_hashes", []))
    current -= removed_hashes
    tracker["indexed_fact_hashes"] = list(current)
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print(f"Hermes Memory — API Key Cleanup ({'DRY RUN' if DRY_RUN else 'EXECUTING'})")
    print("=" * 60)

    print("\n[1/3] Cleaning JSONL files...")
    jsonl_removed = cleanup_jsonl_files()
    print(f"  Removed {jsonl_removed} sensitive facts from JSONL files")

    print("\n[2/3] Rebuilding graph.json...")
    nodes_removed, edges_removed = rebuild_graph()
    print(f"  Removed {nodes_removed} nodes, {edges_removed} edges from graph")

    print("\n✓ Cleanup complete.")
