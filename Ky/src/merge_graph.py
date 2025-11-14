"""Merge pipeline outputs into the final knowledge graph."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_cluster_entries(raw_clusters: Any) -> List[Dict[str, Any]]:
    """Return a flat list of cluster dicts regardless of the input layout."""
    if isinstance(raw_clusters, list):
        return [entry for entry in raw_clusters if isinstance(entry, dict)]

    if isinstance(raw_clusters, dict):
        nested = raw_clusters.get("clusters")
        if isinstance(nested, list):
            return [entry for entry in nested if isinstance(entry, dict)]

        coerced: List[Dict[str, Any]] = []
        for cluster_id, details in raw_clusters.items():
            entry = {"id": cluster_id}
            if isinstance(details, dict):
                entry.update(details)
            coerced.append(entry)
        return coerced

    return []


def convert_to_frontend_format(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal graph structure into the frontend rendering format.
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    metadata = graph_data.get("metadata", {})

    frontend_nodes = [
        {
            "id": node.get("id"),
            "orig_id": node.get("orig_id"),
            "cluster_id": node.get("cluster_id"),
            "cluster_name": node.get("cluster_name"),
            "timestamp": node.get("timestamp"),
            "num_messages": node.get("num_messages")
        }
        for node in nodes
    ]

    type_map = {
        "high": "hard",
        "llm_verified": "insight",
        "medium": "insight"
    }
    frontend_edges = [
        {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "weight": edge.get("weight"),
            "type": type_map.get(edge.get("confidence"), "insight"),
            "intraCluster": edge.get("is_intra_cluster", False)
        }
        for edge in edges
    ]

    cluster_entries = _normalize_cluster_entries(metadata.get("clusters"))

    frontend_clusters = [
        {
            "id": cluster.get("id"),
            "name": cluster.get("name"),
            "description": cluster.get("description"),
            "size": cluster.get("size", 0),
            "themes": cluster.get("key_themes") or cluster.get("themes") or []
        }
        for cluster in cluster_entries
    ]

    stats = {
        "nodes": metadata.get("total_nodes", len(nodes)),
        "edges": metadata.get("total_edges", len(edges)),
        "clusters": metadata.get("total_clusters", len(frontend_clusters))
    }

    return {
        "nodes": frontend_nodes,
        "edges": frontend_edges,
        "clusters": frontend_clusters,
        "stats": stats
    }


def save_graph(
    graph_data: Dict[str, Any],
    output_path: Path,
    frontend_output_path: Optional[Path] = None
) -> None:
    """
    Persist graph data to disk and optionally write the frontend format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    if frontend_output_path:
        frontend_output_path.parent.mkdir(parents=True, exist_ok=True)
        frontend_data = convert_to_frontend_format(graph_data)
        with open(frontend_output_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def merge_graph_data(
    features_path: Path,
    cluster_path: Path,
    edges_path: Path,
    output_path: Path,
    frontend_output_path: Optional[Path] = None,
    verbose: bool = True
) -> None:
    """
    Merge pipeline outputs into final graph.json.

    This function combines:
    - features.json: conversation features with keywords and embeddings
    - clusters.json: LLM-generated cluster assignments
    - edges.json: similarity-based edges with metadata

    Into a unified graph structure with nodes, edges, and comprehensive metadata.

    Args:
        features_path: Path to features.json (from extract_features.py)
        cluster_path: Path to clusters.json (from cluster_with_llm.py)
        edges_path: Path to edges.json (from build_edges.py)
        output_path: Path to output graph.json
        verbose: Print progress messages
    """
    if verbose:
        print("🔗 Merging graph data...")

    # Load all input files
    if verbose:
        print("  📂 Loading features.json...")
    features_data = load_json(features_path)
    conversations = features_data.get("conversations", [])
    features_metadata = features_data.get("metadata", {})

    if verbose:
        print("  📂 Loading clusters.json...")
    cluster_data = load_json(cluster_path)
    assignments = cluster_data.get("assignments", [])
    clusters = cluster_data.get("clusters", [])
    cluster_metadata = cluster_data.get("metadata", {})

    if verbose:
        print("  📂 Loading edges.json...")
    edges_data = load_json(edges_path)
    edges = edges_data.get("edges", [])
    edge_metadata = edges_data.get("metadata", {})

    # Create lookup maps
    if verbose:
        print("  🔗 Building node mappings...")

    # Map: conversation_id -> cluster assignment info
    assignment_map = {
        assign["conversation_id"]: {
            "cluster_id": assign["cluster_id"],
            "confidence": assign["confidence"],
            "top_keywords": assign.get("top_keywords", [])
        }
        for assign in assignments
    }

    # Map: cluster_id -> cluster details
    cluster_map = {
        cluster["id"]: {
            "name": cluster["name"],
            "description": cluster["description"],
            "key_themes": cluster.get("key_themes", []),
            "size": cluster.get("size", 0)
        }
        for cluster in clusters
    }

    # Build enriched nodes
    if verbose:
        print("  🔨 Building enriched nodes...")

    nodes = []
    for conv in conversations:
        conv_id = conv["id"]
        assignment = assignment_map.get(conv_id, {})
        cluster_id = assignment.get("cluster_id", "unknown")
        cluster_info = cluster_map.get(cluster_id, {})

        node = {
            "id": conv_id,
            "orig_id": conv["orig_id"],
            "cluster_id": cluster_id,
            "cluster_name": cluster_info.get("name", "Unknown"),
            "cluster_confidence": assignment.get("confidence", 0.0),
            "keywords": conv["keywords"],
            "top_keywords": assignment.get("top_keywords", []),
            "timestamp": conv.get("timestamp"),
            "num_messages": conv.get("num_messages", 0)
        }
        nodes.append(node)

    # Generate comprehensive metadata
    if verbose:
        print("  📊 Generating metadata...")

    # Cluster statistics with details
    cluster_stats = {}
    for cluster in clusters:
        cluster_stats[cluster["id"]] = {
            "name": cluster["name"],
            "description": cluster["description"],
            "size": cluster.get("size", 0),
            "key_themes": cluster.get("key_themes", [])
        }

    # Edge statistics
    edge_stats = {
        "total_edges": edge_metadata.get("total_edges", len(edges)),
        "intra_cluster_edges": edge_metadata.get("intra_cluster_edges", 0),
        "inter_cluster_edges": edge_metadata.get("inter_cluster_edges", 0),
        "high_confidence_edges": edge_metadata.get("high_confidence_edges", 0),
        "llm_verified_edges": edge_metadata.get("llm_verified_edges", 0),
        "thresholds": edge_metadata.get("thresholds", {})
    }

    # Calculate edge density
    num_nodes = len(nodes)
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2 if num_nodes > 1 else 0
    edge_density = len(edges) / max_possible_edges if max_possible_edges > 0 else 0.0
    edge_stats["edge_density"] = round(edge_density, 4)

    # Timing information from all steps
    timing_info = {
        "feature_extraction": features_metadata.get("timing", {}),
        "clustering": {
            "total_seconds": cluster_metadata.get("clustering_time_seconds", 0)
        },
        "edge_generation": {
            "total_seconds": edge_metadata.get("edge_generation_time_seconds", 0)
        }
    }

    # Calculate total pipeline time
    total_time = (
        timing_info["feature_extraction"].get("total_seconds", 0) +
        timing_info["clustering"]["total_seconds"] +
        timing_info["edge_generation"]["total_seconds"]
    )
    timing_info["total_pipeline_seconds"] = round(total_time, 2)

    # Final metadata structure
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_clusters": len(clusters),
        "clusters": cluster_stats,
        "edge_statistics": edge_stats,
        "timing": timing_info,
        "pipeline_params": {
            "embedding_model": features_metadata.get("embedding_model", "unknown"),
            "clustering_model": cluster_metadata.get("clustering_model", "unknown"),
            "keyword_params": features_metadata.get("keyword_params", {}),
            "preprocess_params": features_metadata.get("preprocess_params", {})
        },
        "source_files": {
            "features": str(features_path),
            "clusters": str(cluster_path),
            "edges": str(edges_path)
        }
    }

    # Create final graph structure
    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata
    }

    # Save to file(s)
    if verbose:
        print(f"  💾 Writing to {output_path}...")
        if frontend_output_path:
            print(f"  💾 Writing frontend graph to {frontend_output_path}...")

    save_graph(graph, output_path, frontend_output_path)

    # Print summary
    if verbose:
        print(f"\n✅ Graph merged successfully!")
        print(f"\n📊 Final Graph Statistics:")
        print(f"  Nodes:                  {len(nodes)}")
        print(f"  Edges:                  {len(edges)}")
        print(f"  Clusters:               {len(clusters)}")
        print(f"  Intra-cluster edges:    {edge_stats['intra_cluster_edges']} ({edge_stats['intra_cluster_edges']/len(edges)*100:.1f}%)" if edges else "  Intra-cluster edges:    0")
        print(f"  Inter-cluster edges:    {edge_stats['inter_cluster_edges']} ({edge_stats['inter_cluster_edges']/len(edges)*100:.1f}%)" if edges else "  Inter-cluster edges:    0")
        print(f"  Edge density:           {edge_stats['edge_density']:.4f}")
        print(f"\n💾 Saved to: {output_path.resolve()}")


def validate_input_files(
    features_path: Path,
    cluster_path: Path,
    edges_path: Path
) -> None:
    """
    Validate that all required input files exist.

    Raises:
        FileNotFoundError: If any required file is missing
    """
    files = [
        (features_path, "features.json"),
        (cluster_path, "clusters.json"),
        (edges_path, "edges.json")
    ]

    missing = []
    for path, name in files:
        if not path.exists():
            missing.append(f"{name} at {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing required files:\n" + "\n".join(f"  - {f}" for f in missing)
        )


def main():
    """CLI entry point for merge_graph.py"""
    parser = argparse.ArgumentParser(
        description="Merge pipeline outputs into final graph.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python merge_graph.py \\
    --features output/features.json \\
    --clusters output/clusters.json \\
    --edges output/edges.json \\
    --output output/graph.json

  # Quiet mode
  python merge_graph.py \\
    --features output/features.json \\
    --clusters output/clusters.json \\
    --edges output/edges.json \\
    --output output/graph.json \\
    --quiet
        """
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features.json (from extract_features.py)"
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to clusters.json (from cluster_with_llm.py)"
    )
    parser.add_argument(
        "--edges",
        type=Path,
        required=True,
        help="Path to edges.json (from build_edges.py)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output graph.json"
    )
    parser.add_argument(
        "--frontend-output",
        type=Path,
        default=None,
        help="Path to save frontend-formatted JSON (optional)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    try:
        # Validate input files
        validate_input_files(args.features, args.clusters, args.edges)

        # Merge graph data
        merge_graph_data(
            features_path=args.features,
            cluster_path=args.clusters,
            edges_path=args.edges,
            output_path=args.output,
            frontend_output_path=args.frontend_output,
            verbose=not args.quiet
        )

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
