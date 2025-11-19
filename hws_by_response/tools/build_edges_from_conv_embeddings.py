"""
Conversation-level edges from conversation_embeddings.pkl
- Input: conversation_embeddings.pkl (conversation_id -> embedding)
- Output: graph.json (edges between conversations)
"""
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 실행
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_conversation_embeddings(path: str) -> Dict[int, Dict[str, Any]]:
    """Load conversation embeddings from PKL"""
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_similarity_matrix(conv_embeddings: Dict[int, Dict[str, Any]], metric: str = "cosine") -> tuple:
    """
    Compute pairwise similarity matrix

    Args:
        conv_embeddings: Conversation embeddings
        metric: Distance metric ("cosine", "l2", "l1")

    Returns:
        (similarity_matrix, conv_ids)
    """
    conv_ids = sorted(conv_embeddings.keys())
    embeddings = np.array([conv_embeddings[cid]['embedding'] for cid in conv_ids])

    if metric == "cosine":
        sim_matrix = cosine_similarity(embeddings)
    elif metric == "l2":
        # L2 distance -> similarity: 1 / (1 + dist)
        dist_matrix = euclidean_distances(embeddings)
        sim_matrix = 1.0 / (1.0 + dist_matrix)
    elif metric == "l1":
        # L1 distance -> similarity: 1 / (1 + dist)
        dist_matrix = manhattan_distances(embeddings)
        sim_matrix = 1.0 / (1.0 + dist_matrix)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from: cosine, l2, l1")

    return sim_matrix, conv_ids


def load_categories(path: str) -> Dict[int, str]:
    """Load categories from JSON. Returns {conversation_id: category}"""
    if not path:
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    categories = {}
    
    # Handle different JSON formats
    if isinstance(data, dict):
        for key, value in data.items():
            try:
                conv_id = int(key)
                if isinstance(value, dict):
                    categories[conv_id] = value.get("category", "Uncategorized")
                else:
                    categories[conv_id] = str(value)
            except (ValueError, TypeError):
                continue
    
    return categories


def build_edges(
    sim_matrix: np.ndarray,
    conv_ids: List[int],
    hard_threshold: float = 0.7,
    pending_low: float = 0.5,
    pending_high: float = 0.7,
    categories: Dict[int, str] = None
) -> tuple:
    """
    Build edges from similarity matrix
    
    Returns:
        (edges, stats, similarities)
    """
    n = len(conv_ids)
    edges = []
    similarities = []
    
    # Extract upper triangle (avoid duplicates)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            similarities.append(sim)
            
            if sim >= hard_threshold:
                edge_type = "hard"
            elif pending_low <= sim < pending_high:
                edge_type = "pending"
            else:
                continue
            
            edges.append({
                "source": conv_ids[i],
                "target": conv_ids[j],
                "weight": float(sim),
                "type": edge_type
            })
    
    # Statistics
    similarities = np.array(similarities)
    stats = {
        "total_pairs": len(similarities),
        "total_edges": len(edges),
        "hard_edges": sum(1 for e in edges if e["type"] == "hard"),
        "pending_edges": sum(1 for e in edges if e["type"] == "pending"),
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "std_similarity": float(np.std(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "within_category_edges": 0,
        "cross_category_edges": 0,
        "hard_within_category_edges": 0,
        "hard_cross_category_edges": 0,
        "pending_within_category_edges": 0,
        "pending_cross_category_edges": 0,
    }
    
    # Category-based statistics
    if categories:
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            src_cat = categories.get(src)
            tgt_cat = categories.get(tgt)
            
            if src_cat and tgt_cat:
                if src_cat == tgt_cat:
                    # Within category (inter-cluster)
                    stats["within_category_edges"] += 1
                    if edge["type"] == "hard":
                        stats["hard_within_category_edges"] += 1
                    elif edge["type"] == "pending":
                        stats["pending_within_category_edges"] += 1
                else:
                    # Cross category (cross-cluster)
                    stats["cross_category_edges"] += 1
                    if edge["type"] == "hard":
                        stats["hard_cross_category_edges"] += 1
                    elif edge["type"] == "pending":
                        stats["pending_cross_category_edges"] += 1
    
    return edges, stats, similarities


def load_keywords(path: str, top_n: int = 3) -> Dict[int, List[Dict]]:
    """Load keywords from JSON. Returns {conversation_id: [top keywords]}"""
    if not path or not Path(path).exists():
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    keywords_map = {}
    for item in data:
        conv_id = item.get("conversation_id")
        keywords = item.get("keywords", [])
        if conv_id is not None and keywords:
            # Convert format: {"keyword": "...", "score": ...} -> {"term": "...", "score": ...}
            formatted_keywords = []
            for kw in keywords[:top_n]:
                formatted_keywords.append({
                    "term": kw.get("keyword", ""),
                    "score": kw.get("score", 0.0)
                })
            keywords_map[int(conv_id)] = formatted_keywords
    
    return keywords_map


def save_graph(
    edges: List[Dict],
    output_path: Path,
    conv_embeddings: Dict[int, Dict[str, Any]],
    categories: Dict[int, str] = None,
    keywords_map: Dict[int, List[Dict]] = None,
    stats: Dict = None
):
    """Save graph to JSON with full metadata"""
    
    # Build category clusters
    category_to_id = {}
    cluster_info = {}
    if categories:
        unique_cats = sorted(set(categories.values()))
        for i, cat in enumerate(unique_cats):
            cluster_id = f"cluster_{i+1}"
            category_to_id[cat] = cluster_id
            cluster_info[cluster_id] = {
                "id": cluster_id,
                "name": cat,
                "description": f"Conversations categorized as {cat}",
                "key_themes": [],
                "size": sum(1 for c in categories.values() if c == cat)
            }
    
    # Build nodes
    nodes = []
    for conv_id, data in sorted(conv_embeddings.items()):
        category = categories.get(conv_id, "Uncategorized") if categories else "Uncategorized"
        cluster_id = category_to_id.get(category, "cluster_0")
        
        keywords = keywords_map.get(conv_id, []) if keywords_map else []
        top_keywords = [kw["term"] for kw in keywords] if keywords else []
        
        node = {
            "id": conv_id,
            "orig_id": str(conv_id),
            "cluster_id": cluster_id,
            "cluster_name": category,
            "cluster_confidence": 0.85,  # Default confidence
            "keywords": keywords,
            "top_keywords": top_keywords,
            "timestamp": None,
            "num_messages": data.get("qa_count", 0)
        }
        nodes.append(node)
    
    # Update edges with cluster info
    # Only hard and pending edges are included (filtered in build_edges)
    formatted_edges = []
    for edge in edges:
        src_cat = categories.get(edge["source"]) if categories else None
        tgt_cat = categories.get(edge["target"]) if categories else None
        is_intra = (src_cat == tgt_cat) if (src_cat and tgt_cat) else False
        
        # Map edge type to confidence
        edge_type = edge.get("type", "pending")
        if edge_type == "hard":
            confidence = "high"
        elif edge_type == "pending":
            confidence = "medium"
        else:
            confidence = "low"
        
        formatted_edge = {
            "source": edge["source"],
            "target": edge["target"],
            "weight": edge["weight"],
            "type": "semantic",
            "is_intra_cluster": is_intra,
            "confidence": confidence
        }
        formatted_edges.append(formatted_edge)
    
    # Build metadata
    metadata = {
        "total_nodes": len(nodes),
        "total_edges": len(formatted_edges),
        "clusters": {
            "total_clusters": len(cluster_info),
            "clusters": list(cluster_info.values())
        },
        "edge_stats": stats if stats else {}
    }
    
    # Final graph structure
    graph = {
        "nodes": nodes,
        "edges": formatted_edges,
        "metadata": metadata
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def plot_similarity_distribution(similarities: np.ndarray, output_path: Path, stats: Dict):
    """Plot similarity distribution and save as PNG"""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return

    metric = stats.get('metric', 'cosine')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Similarity Distribution ({metric.upper()})', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(stats['mean_similarity'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_similarity']:.4f}")
    ax.axvline(stats['median_similarity'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median_similarity']:.4f}")
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Similarity Distribution (Histogram)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax = axes[0, 1]
    ax.boxplot(similarities, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_title('Similarity Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distribution
    ax = axes[1, 0]
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax.plot(sorted_sims, cumulative, linewidth=2, color='purple')
    ax.axhline(0.5, color='green', linestyle='--', linewidth=1, label='50th percentile')
    ax.axhline(0.75, color='orange', linestyle='--', linewidth=1, label='75th percentile')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, label='95th percentile')
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Statistics Summary
    ==================
    Total Pairs: {stats['total_pairs']:,}
    
    Mean:        {stats['mean_similarity']:.4f}
    Median:      {stats['median_similarity']:.4f}
    Std Dev:     {stats['std_similarity']:.4f}
    
    Min:         {stats['min_similarity']:.4f}
    Max:         {stats['max_similarity']:.4f}
    
    Edges
    -----
    Hard:        {stats['hard_edges']:,}
    Pending:     {stats['pending_edges']:,}
    
    Percentiles
    -----------
    25th:        {np.percentile(similarities, 25):.4f}
    50th:        {np.percentile(similarities, 50):.4f}
    75th:        {np.percentile(similarities, 75):.4f}
    95th:        {np.percentile(similarities, 95):.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build edges from qa_embeddings.pkl")
    parser.add_argument("--embeddings", type=str, default="output/qa_embeddings.pkl")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--hard-threshold", type=float, default=None, help="Hard edge threshold (default: 90th percentile)")
    parser.add_argument("--pending-low", type=float, default=None, help="Pending edge lower bound (default: 80th percentile)")
    parser.add_argument("--pending-high", type=float, default=None, help="Pending edge upper bound (default: same as hard_threshold)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "l1"], help="Distance metric (cosine, l2, l1)")
    parser.add_argument("--plot", action="store_true", help="Save similarity distribution plot as PNG")
    parser.add_argument("--categories", type=str, default=None, help="Optional: category JSON file (conversation_id -> category)")
    parser.add_argument("--keywords", type=str, default=None, help="Optional: keywords JSON file for node metadata")
    parser.add_argument("--target-hard-cross-edges", type=int, default=300, help="Target number of hard cross-category edges for dynamic threshold (default: 300, use -1 to disable and use P90)")
    args = parser.parse_args()
    
    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("로딩 중...")
    conv_embeddings = load_conversation_embeddings(args.embeddings)
    print(f"Loaded {len(conv_embeddings)} conversation embeddings")
    
    # Load categories if provided
    categories = None
    if args.categories:
        categories = load_categories(args.categories)
        print(f"Loaded categories for {len(categories)} conversations")
    
    # Load keywords if provided
    keywords_map = None
    if args.keywords:
        keywords_map = load_keywords(args.keywords)
        print(f"Loaded keywords for {len(keywords_map)} conversations")
    
    print("유사도 계산 중...")
    sim_matrix, conv_ids = compute_similarity_matrix(conv_embeddings, metric=args.metric)
    print(f"Similarity matrix: {sim_matrix.shape}")
    
    print("엣지 생성 중...")

    # First pass: compute similarities to determine auto thresholds
    n = len(conv_ids)
    all_similarities = []
    cross_category_similarities = []  # For dynamic threshold calculation

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            all_similarities.append(sim)

            # Track cross-category similarities if categories are provided
            if categories:
                src_cat = categories.get(conv_ids[i])
                tgt_cat = categories.get(conv_ids[j])
                if src_cat and tgt_cat and src_cat != tgt_cat:
                    cross_category_similarities.append(sim)

    all_similarities = np.array(all_similarities)

    # Auto-calculate thresholds if not provided
    hard_threshold = args.hard_threshold
    pending_low = args.pending_low
    pending_high = args.pending_high

    if hard_threshold is None or pending_low is None or pending_high is None:
        median = float(np.median(all_similarities))
        p80 = float(np.percentile(all_similarities, 80))
        p90 = float(np.percentile(all_similarities, 90))

        # Dynamic threshold: target ~N hard cross-category edges
        target_hard_cross_edges = args.target_hard_cross_edges

        if hard_threshold is None and target_hard_cross_edges > 0 and categories and cross_category_similarities:
            # Dynamic threshold based on target number of cross-category edges
            cross_cat_sims = np.array(cross_category_similarities)
            cross_cat_sims_sorted = np.sort(cross_cat_sims)[::-1]  # Descending order

            if len(cross_cat_sims_sorted) > target_hard_cross_edges:
                # Take the similarity of the Nth edge as threshold
                hard_threshold = float(cross_cat_sims_sorted[target_hard_cross_edges - 1])
                print(f"Dynamic hard_threshold for ~{target_hard_cross_edges} cross-category edges: {hard_threshold:.4f}")
                print(f"  (Total cross-category pairs: {len(cross_cat_sims_sorted)})")
            else:
                # Not enough cross-category pairs, use P90
                hard_threshold = p90
                print(f"Not enough cross-category pairs ({len(cross_cat_sims_sorted)}), using P90: {hard_threshold:.4f}")
        elif hard_threshold is None:
            # Use P90 (target_hard_cross_edges == -1 or no categories)
            hard_threshold = p90
            if target_hard_cross_edges == -1:
                print(f"Using automatic P90 threshold (target_hard_cross_edges=-1): {hard_threshold:.4f}")

        if pending_high is None:
            pending_high = hard_threshold  # Same as hard_threshold
        if pending_low is None:
            pending_low = p80

        print(f"Threshold 계산: Median={median:.4f}, P80={p80:.4f}, P90={p90:.4f}")
        print(f"  hard_threshold={hard_threshold:.4f}, pending_low={pending_low:.4f}, pending_high={pending_high:.4f}\n")
    
    edges, stats, similarities = build_edges(
        sim_matrix,
        conv_ids,
        hard_threshold=hard_threshold,
        pending_low=pending_low,
        pending_high=pending_high,
        categories=categories
    )
    
    print("\n=== Statistics ===")
    print(f"Total conversation pairs: {stats['total_pairs']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Hard edges: {stats['hard_edges']}")
    print(f"Pending edges: {stats['pending_edges']}")
    print(f"Mean similarity: {stats['mean_similarity']:.4f}")
    print(f"Median similarity: {stats['median_similarity']:.4f}")
    print(f"Max similarity: {stats['max_similarity']:.4f}")
    print(f"Min similarity: {stats['min_similarity']:.4f}")
    
    if categories:
        print("\n=== Category-based Statistics ===")
        print(f"Within-category edges (inter-cluster): {stats['within_category_edges']}")
        print(f"  - Hard: {stats['hard_within_category_edges']}")
        print(f"  - Pending: {stats['pending_within_category_edges']}")
        print(f"Cross-category edges (cross-cluster): {stats['cross_category_edges']}")
        print(f"  - Hard: {stats['hard_cross_category_edges']}")
        print(f"  - Pending: {stats['pending_cross_category_edges']}")
    
    # Save graph
    metric_suffix = f"_{args.metric}" if args.metric != "cosine" else ""
    graph_path = out_dir / f"graph{metric_suffix}.json"
    save_graph(edges, graph_path, conv_embeddings, categories, keywords_map, stats)
    print(f"\nGraph saved: {graph_path}")

    # Save stats
    stats_path = out_dir / f"edge_stats{metric_suffix}.json"
    stats["metric"] = args.metric  # Add metric to stats
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved: {stats_path}")

    # Plot similarity distribution
    if args.plot:
        plot_path = out_dir / f"similarity_distribution{metric_suffix}.png"
        plot_similarity_distribution(similarities, plot_path, stats)
    
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
