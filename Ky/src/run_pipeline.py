"""End-to-end pipeline orchestration for conversation graph building."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from merge_graph import merge_graph_data

SRC_DIR = Path(__file__).resolve().parent


def _script_path(filename: str) -> str:
    """Return an absolute path to a sibling step script."""
    return str(SRC_DIR / filename)


def run_step(cmd: List[str], step_name: str, verbose: bool = True) -> bool:
    """
    Execute a subprocess command for a pipeline step.

    Args:
        cmd: Command to execute as list of strings
        step_name: Descriptive name for the step
        verbose: Whether to print progress messages

    Returns:
        True if command succeeded (returncode == 0), False otherwise
    """
    if verbose:
        print(f"\n🚀 Running {step_name}...")
        print(f"   Command: {' '.join(cmd)}")

    try:
        # Run command and capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            if verbose:
                print(f"✅ {step_name} completed successfully")
            return True
        else:
            print(f"❌ {step_name} failed with return code {return_code}")
            sys.exit(1)

    except Exception as exc:
        print(f"❌ {step_name} failed with exception: {exc}")
        sys.exit(1)


def validate_file_exists(path: Path, description: str) -> None:
    """
    Validate that a file exists.

    Args:
        path: Path to check
        description: Human-readable description of the file

    Raises:
        SystemExit if file does not exist
    """
    if not path.exists():
        print(f"❌ {description} not found: {path}")
        sys.exit(1)


def main() -> None:
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="End-to-end conversation graph building pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input chat history JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        help="Fixed number of clusters (optional)",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
        help="Min clusters (if num-clusters not set, default: 3)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Max clusters (if num-clusters not set, default: 8)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.8,
        help="High confidence edge threshold (default: 0.8)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.6,
        help="Medium confidence edge threshold (default: 0.6)",
    )
    parser.add_argument(
        "--no-llm-edges",
        action="store_true",
        help="Skip LLM verification for edges",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider for clustering (default: openai)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for clustering (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate input files
    input_path = args.input.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()

    validate_file_exists(input_path, "Input chat history JSON")
    validate_file_exists(config_path, "Config YAML file")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("=" * 60)
        print("🚀 Starting Conversation Graph Building Pipeline")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Config: {config_path}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

    pipeline_start = time.perf_counter()

    # Define output paths
    features_path = output_dir / "features.json"
    cluster_path = output_dir / "clusters.json"
    edges_path = output_dir / "edges.json"
    final_graph_path = output_dir / "graph.json"

    # Step 1: Extract keywords and embeddings
    run_step(
        [
            sys.executable,
            _script_path("extract_features.py"),
            "--in",
            str(input_path),
            "--out",
            str(features_path),
            "--cfg",
            str(config_path),
        ],
        "Step 1: Keyword & Embedding Extraction",
        verbose=args.verbose,
    )

    # Validate feature extraction results
    validate_file_exists(features_path, "Feature data JSON")

    with open(features_path, "r", encoding="utf-8") as f:
        features_data = json.load(f)
    timing = features_data.get("metadata", {}).get("timing", {})
    step1_total = float(timing.get("total_seconds", 0.0) or 0.0)
    step1_embedding = float(timing.get("embedding_seconds", 0.0) or 0.0)
    step1_keyword = float(timing.get("keyword_seconds", 0.0) or 0.0)

    print(f"✓ Step 1 completed in {step1_total:.1f}s")
    print(
        f"  └─ Embedding: {step1_embedding:.1f}s, Keyword: {step1_keyword:.1f}s\n"
    )

    # Step 2: LLM-based clustering
    cluster_cmd = [
        sys.executable,
        _script_path("cluster_with_llm.py"),
        "--input",
        str(features_path),
        "--output",
        str(cluster_path),
        "--provider",
        args.provider,
        "--model",
        args.model,
    ]

    if args.num_clusters:
        cluster_cmd.extend(["--num-clusters", str(args.num_clusters)])
    else:
        cluster_cmd.extend(
            [
                "--min-clusters",
                str(args.min_clusters),
                "--max-clusters",
                str(args.max_clusters),
            ]
        )

    if args.verbose:
        cluster_cmd.append("--verbose")

    step2_start = time.perf_counter()
    run_step(
        cluster_cmd,
        "Step 2: LLM-based Clustering",
        verbose=args.verbose,
    )
    step2_time = time.perf_counter() - step2_start

    # Validate cluster results
    validate_file_exists(cluster_path, "Cluster assignments JSON")

    # Step 3: Build edges
    edge_cmd = [
        sys.executable,
        _script_path("build_edges.py"),
        "--intermediate",
        str(features_path),
        "--clusters",
        str(cluster_path),
        "--output",
        str(edges_path),
        "--high-threshold",
        str(args.high_threshold),
        "--medium-threshold",
        str(args.medium_threshold),
    ]

    if args.no_llm_edges:
        edge_cmd.append("--no-llm")

    if args.verbose:
        edge_cmd.append("--verbose")

    step3_start = time.perf_counter()
    run_step(
        edge_cmd,
        "Step 3: Edge Generation",
        verbose=args.verbose,
    )
    step3_time = time.perf_counter() - step3_start

    # Validate edge results
    validate_file_exists(edges_path, "Edges JSON")

    # Step 4: Merge results into final graph
    print(f"\n{'='*60}")
    print(f"🔗 Step 4: Merging final graph...")
    print(f"{'='*60}\n")

    step4_start = time.perf_counter()
    merge_graph_data(
        features_path=features_path,
        cluster_path=cluster_path,
        edges_path=edges_path,
        output_path=final_graph_path,
        verbose=args.verbose,
    )
    step4_time = time.perf_counter() - step4_start

    print(f"\n✓ Step 4 completed in {step4_time:.1f}s\n")

    total_pipeline_time = time.perf_counter() - pipeline_start

    print(f"\n{'='*60}")
    print("✅ Pipeline Complete!")
    print(f"{'='*60}")
    print(f"\n📊 Timing Summary:")
    print(f"  Step 1 (Feature Extraction):  {step1_total:.1f}s")
    print(f"    ├─ Embedding generation:    {step1_embedding:.1f}s")
    print(f"    └─ Keyword extraction:      {step1_keyword:.1f}s")
    print(f"  Step 2 (LLM Clustering):      {step2_time:.1f}s")
    print(f"  Step 3 (Edge Generation):     {step3_time:.1f}s")
    print(f"  Step 4 (Graph Merging):       {step4_time:.1f}s")
    print(f"  {'─'*40}")
    print(f"  Total Pipeline Time:          {total_pipeline_time:.1f}s")

    print(f"\n📊 Final Graph Statistics:")

    # Load and display graph statistics
    with open(final_graph_path) as f:
        graph_data = json.load(f)
        metadata = graph_data.get("metadata", {})
        edge_stats = metadata.get("edge_statistics", {})

        print(f"  Nodes:                  {metadata.get('total_nodes', 0)}")
        print(f"  Edges:                  {metadata.get('total_edges', 0)}")
        print(f"  Clusters:               {metadata.get('total_clusters', 0)}")
        print(f"  Intra-cluster edges:    {edge_stats.get('intra_cluster_edges', 0)}")
        print(f"  Inter-cluster edges:    {edge_stats.get('inter_cluster_edges', 0)}")
        print(f"  Edge density:           {edge_stats.get('edge_density', 0):.4f}")

    print(f"\n💾 Final graph saved to: {final_graph_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
