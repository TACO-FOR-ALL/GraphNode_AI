"""Edge generation module for conversation graph using similarity-based and LLM verification."""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None
    find_dotenv = None

if load_dotenv:
    # Use python-dotenv's search to pick up a .env in parent directories (e.g., repo root)
    env_path = find_dotenv(usecwd=True) if find_dotenv else None
    load_dotenv(env_path or None, override=False)


@dataclass
class EdgeCandidate:
    """Represents a potential edge between two nodes."""

    source_id: int
    target_id: int
    similarity: float
    source_cluster: Optional[str]
    target_cluster: Optional[str]


@dataclass
class Edge:
    """Final edge with metadata."""

    source: int
    target: int
    weight: float
    type: str = "semantic"
    is_intra_cluster: bool = False  # True if both nodes in same cluster
    confidence: str = "high"  # "high" or "llm_verified"


def _flatten_similarity_values(similarity_matrix: np.ndarray) -> np.ndarray:
    """Return finite off-diagonal similarity values."""
    if similarity_matrix.size == 0:
        return np.array([], dtype=np.float32)
    n = similarity_matrix.shape[0]
    if n < 2:
        return np.array([], dtype=np.float32)
    triu_idx = np.triu_indices(n, k=1)
    values = similarity_matrix[triu_idx]
    finite = values[np.isfinite(values)]
    return finite


def compute_similarity_stats(similarity_matrix: np.ndarray) -> Dict[str, Optional[float]]:
    """Summarize the similarity distribution for adaptive thresholding."""
    values = _flatten_similarity_values(similarity_matrix)
    if values.size == 0:
        return {
            "count": 0,
            "max": None,
            "min": None,
            "mean": None,
            "p90": None,
            "p75": None,
        }
    stats = {
        "count": int(values.size),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p75": float(np.percentile(values, 75)),
    }
    return stats


def adjust_similarity_thresholds(
    stats: Dict[str, Optional[float]],
    high_threshold: float,
    medium_threshold: float,
    verbose: bool = False,
) -> Tuple[float, float, bool]:
    """
    Lower thresholds when no edge can meet the requested values.

    Returns (effective_high, effective_medium, adjusted_flag).
    """
    max_sim = stats.get("max")
    if max_sim is None or not np.isfinite(max_sim) or max_sim <= 0:
        return high_threshold, medium_threshold, False

    effective_high = high_threshold
    effective_medium = medium_threshold
    adjusted = False

    p90 = stats.get("p90")
    fallback_medium = min(medium_threshold, max_sim * 0.85)
    if p90 is not None and np.isfinite(p90):
        fallback_medium = max(fallback_medium, float(p90))
    fallback_medium = min(fallback_medium, max_sim - 1e-4)
    fallback_medium = max(fallback_medium, 0.0)

    fallback_high = min(high_threshold, max_sim * 0.95)
    if p90 is not None and np.isfinite(p90):
        fallback_high = max(fallback_high, float(p90))
    fallback_high = min(fallback_high, max_sim - 5e-4)
    fallback_high = max(fallback_high, fallback_medium + 5e-3)

    if max_sim < medium_threshold:
        effective_medium = fallback_medium
        adjusted = True
    if max_sim < high_threshold:
        effective_high = fallback_high
        adjusted = True

    if effective_high <= effective_medium:
        effective_high = min(max_sim - 1e-4, max(effective_medium + 0.01, fallback_high))
        effective_medium = min(effective_medium, effective_high - 0.01)
        effective_medium = max(effective_medium, 0.0)

    if verbose and adjusted:
        print(
            "  ⚠️  Similarity scores peak at "
            f"{max_sim:.3f}, below requested thresholds "
            f"(high={high_threshold}, medium={medium_threshold}). "
            f"Using adaptive thresholds high={effective_high:.3f}, "
            f"medium={effective_medium:.3f}."
        )

    return effective_high, effective_medium, adjusted


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix from embeddings.

    Args:
        embeddings: Normalized embedding matrix of shape (n_conversations, embedding_dim)

    Returns:
        Similarity matrix of shape (n_conversations, n_conversations) with diagonal set to -inf
    """
    # Compute cosine similarity (embeddings should already be normalized)
    similarity = embeddings @ embeddings.T

    # Set diagonal to -inf to ignore self-similarity
    np.fill_diagonal(similarity, -np.inf)

    return similarity


def generate_edge_candidates(
    similarity_matrix: np.ndarray,
    cluster_assignments: Dict[int, str],
    high_threshold: float = 0.8,
    medium_threshold: float = 0.6,
) -> Tuple[List[Edge], List[EdgeCandidate]]:
    """
    Generate edge candidates based on similarity thresholds.

    Args:
        similarity_matrix: Cosine similarity matrix
        cluster_assignments: Dict mapping conversation_id to cluster_id
        high_threshold: Threshold for high-confidence edges (default: 0.8)
        medium_threshold: Threshold for medium-confidence edges requiring LLM check (default: 0.6)

    Returns:
        Tuple of (confirmed_edges, candidates_for_llm_check)
    """
    n_nodes = similarity_matrix.shape[0]
    confirmed_edges: List[Edge] = []
    candidates_for_llm: List[EdgeCandidate] = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only upper triangle (i < j)
            similarity = float(similarity_matrix[i, j])

            # Skip if below medium threshold
            if similarity < medium_threshold:
                continue

            source_cluster = cluster_assignments.get(i)
            target_cluster = cluster_assignments.get(j)
            is_intra_cluster = (
                source_cluster is not None
                and target_cluster is not None
                and source_cluster == target_cluster
            )

            if similarity >= high_threshold:
                # High confidence edge
                edge = Edge(
                    source=i,
                    target=j,
                    weight=similarity,
                    type="semantic",
                    is_intra_cluster=is_intra_cluster,
                    confidence="high",
                )
                confirmed_edges.append(edge)
            else:
                # Medium confidence - needs LLM verification
                candidate = EdgeCandidate(
                    source_id=i,
                    target_id=j,
                    similarity=similarity,
                    source_cluster=source_cluster,
                    target_cluster=target_cluster,
                )
                candidates_for_llm.append(candidate)

    return confirmed_edges, candidates_for_llm


def llm_verify_edge(
    source_keywords: List[str],
    target_keywords: List[str],
    similarity_score: float,
    llm_client: OpenAI,
) -> bool:
    """
    Verify edge using LLM judgment (single-candidate version).

    NOTE: This function is kept for backward compatibility.
    For better performance, use batch_verify_edges() instead.

    Args:
        source_keywords: Keywords from source conversation
        target_keywords: Keywords from target conversation
        similarity_score: Cosine similarity score
        llm_client: OpenAI client instance

    Returns:
        True if LLM confirms the edge should exist, False otherwise
    """
    # Format keywords for prompt
    source_kw_str = ", ".join(source_keywords) if source_keywords else "(no keywords)"
    target_kw_str = ", ".join(target_keywords) if target_keywords else "(no keywords)"

    prompt = f"""Given two conversations with the following keywords:
Conversation A: {source_kw_str}
Conversation B: {target_kw_str}
Cosine similarity: {similarity_score:.3f}

Should these conversations be connected in a knowledge graph?
Consider: topical overlap, semantic relationship, meaningful connection.
Respond with only "YES" or "NO"."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )

        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer

    except Exception as exc:
        # If LLM call fails, default to rejecting the edge
        print(f"Warning: LLM verification failed: {exc}")
        return False


@dataclass
class BatchVerificationResult:
    """Result from batch LLM verification."""

    pair_id: int
    should_connect: bool
    reason: str


def batch_verify_edges(
    candidates: List[Tuple[int, List[str], List[str], float]],
    llm_client: OpenAI,
) -> List[BatchVerificationResult]:
    """
    Verify multiple edge candidates in a single LLM API call using batch processing.

    This function processes multiple conversation pairs simultaneously, significantly
    reducing API calls and improving performance compared to single-pair verification.

    Args:
        candidates: List of tuples containing (pair_id, source_keywords, target_keywords, similarity)
        llm_client: OpenAI client instance

    Returns:
        List of BatchVerificationResult objects with verification decisions
    """
    if not candidates:
        return []

    # Build the batch prompt
    pairs_text = []
    for pair_id, source_kw, target_kw, similarity in candidates:
        source_kw_str = ", ".join(source_kw) if source_kw else "(no keywords)"
        target_kw_str = ", ".join(target_kw) if target_kw else "(no keywords)"
        pairs_text.append(
            f"[{pair_id}] Conv A: {source_kw_str} | Conv B: {target_kw_str} | Similarity: {similarity:.3f}"
        )

    pairs_str = "\n".join(pairs_text)

    prompt = f"""Evaluate these conversation pairs for connection in a knowledge graph. Respond with a JSON array containing objects with "pair_id", "should_connect" (true/false), and "reason" (brief explanation).

Consider: topical overlap, semantic relationship, and meaningful connection.

Pairs to evaluate:
{pairs_str}

Respond ONLY with a valid JSON array in this exact format:
[
  {{"pair_id": 0, "should_connect": true, "reason": "brief explanation"}},
  {{"pair_id": 1, "should_connect": false, "reason": "brief explanation"}}
]"""

    try:
        # Enforce structured JSON response (object with array field)
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "pair_verifications",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "pair_id": {"type": "integer"},
                                        "should_connect": {"type": "boolean"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["pair_id", "should_connect", "reason"],
                                },
                            },
                        },
                        "required": ["results"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content.strip()

        # Try to parse the JSON response
        try:
            # Handle case where response might be wrapped in additional JSON structure
            parsed = json.loads(content)

            # If the response is a dict with an array field, extract it
            if isinstance(parsed, dict):
                # Look for common array field names
                for key in ["results", "pairs", "evaluations", "data"]:
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                # If still a dict, check if it has numbered keys
                if isinstance(parsed, dict) and all(k.isdigit() for k in parsed.keys()):
                    parsed = [parsed[k] for k in sorted(parsed.keys(), key=int)]

            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed)}")

            results = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue

                pair_id = item.get("pair_id")
                should_connect = item.get("should_connect", False)
                reason = item.get("reason", "")

                if pair_id is not None:
                    results.append(
                        BatchVerificationResult(
                            pair_id=int(pair_id),
                            should_connect=bool(should_connect),
                            reason=str(reason),
                        )
                    )

            return results

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to parse LLM batch response: {e}")
            print(f"Response content: {content[:500]}...")
            # Return empty results on parse failure
            return []

    except Exception as exc:
        print(f"Warning: LLM batch verification failed: {exc}")
        return []


def verify_candidates_with_llm(
    candidates: List[EdgeCandidate],
    conversations: List[Dict[str, Any]],
    llm_client: OpenAI,
    batch_size: int = 40,
    stratum_width: float = 0.05,
    min_approval_rate: float = 0.10,
    verbose: bool = True,
) -> List[Edge]:
    """
    Verify edge candidates using LLM with stratification and batch processing.

    Candidates are stratified into similarity ranges (strata) and processed from
    highest to lowest similarity. Processing stops early if approval rate drops
    below the minimum threshold. Within each stratum, candidates are processed
    in batches to reduce API calls.

    Args:
        candidates: List of edge candidates to verify
        conversations: List of conversation dictionaries with keywords
        llm_client: OpenAI client instance
        batch_size: Number of candidates to process per LLM API call (default: 40)
        stratum_width: Width of each similarity stratum (default: 0.05)
        min_approval_rate: Skip remaining strata if approval drops below this (default: 0.10)
        verbose: Whether to print progress

    Returns:
        List of verified edges
    """
    if not candidates:
        return []

    overall_start = time.perf_counter()

    verified_edges: List[Edge] = []
    total = len(candidates)

    # Sort candidates by similarity (highest first)
    sorted_candidates = sorted(candidates, key=lambda c: c.similarity, reverse=True)

    # Stratify candidates into similarity ranges
    if len(sorted_candidates) > 0:
        max_sim = sorted_candidates[0].similarity
        min_sim = sorted_candidates[-1].similarity

        # Create strata boundaries
        strata: Dict[Tuple[float, float], List[EdgeCandidate]] = {}
        for candidate in sorted_candidates:
            # Determine which stratum this candidate belongs to
            stratum_lower = int(candidate.similarity / stratum_width) * stratum_width
            stratum_upper = stratum_lower + stratum_width
            stratum_key = (stratum_lower, stratum_upper)

            if stratum_key not in strata:
                strata[stratum_key] = []
            strata[stratum_key].append(candidate)

        # Sort strata by upper bound (highest first)
        sorted_strata = sorted(strata.items(), key=lambda x: x[0][1], reverse=True)

        if verbose:
            print(
                f"\nStratified {total} candidates into {len(sorted_strata)} "
                f"strata (width={stratum_width:.2f})"
            )
            print(
                f"LLM verification settings: batch_size={batch_size}, "
                f"min_approval_rate={min_approval_rate:.2f}"
            )

        total_processed = 0
        total_approved = 0
        total_batches = 0

        # Process each stratum
        for stratum_bounds, stratum_candidates in sorted_strata:
            stratum_lower, stratum_upper = stratum_bounds
            stratum_size = len(stratum_candidates)
            num_batches = (stratum_size + batch_size - 1) // batch_size  # Ceiling division
            stratum_start = time.perf_counter()

            if verbose:
                print(
                    f"\nProcessing stratum {stratum_lower:.2f}-{stratum_upper:.2f} "
                    f"({stratum_size} candidates, {num_batches} batch{'es' if num_batches != 1 else ''})..."
                )

            stratum_approved = 0

            # Process stratum in batches
            for batch_idx in range(num_batches):
                batch_start_time = time.perf_counter()
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, stratum_size)
                batch_candidates = stratum_candidates[batch_start:batch_end]

                # Prepare batch data for LLM
                batch_data = []
                for local_idx, candidate in enumerate(batch_candidates):
                    source_conv = conversations[candidate.source_id]
                    target_conv = conversations[candidate.target_id]

                    source_keywords = [kw["term"] for kw in source_conv.get("keywords", [])]
                    target_keywords = [kw["term"] for kw in target_conv.get("keywords", [])]

                    batch_data.append(
                        (local_idx, source_keywords, target_keywords, candidate.similarity)
                    )

                # Call LLM for batch verification
                batch_results = batch_verify_edges(batch_data, llm_client)
                batch_approved = 0

                # Create a map of pair_id to result
                result_map = {r.pair_id: r for r in batch_results}

                # Process results and create edges
                for local_idx, candidate in enumerate(batch_candidates):
                    result = result_map.get(local_idx)

                    if result and result.should_connect:
                        is_intra_cluster = (
                            candidate.source_cluster == candidate.target_cluster
                            and candidate.source_cluster is not None
                        )
                        edge = Edge(
                            source=candidate.source_id,
                            target=candidate.target_id,
                            weight=candidate.similarity,
                            type="semantic",
                            is_intra_cluster=is_intra_cluster,
                            confidence="llm_verified",
                        )
                        verified_edges.append(edge)
                        stratum_approved += 1
                        batch_approved += 1

                total_batches += 1
                batch_duration = time.perf_counter() - batch_start_time
                if verbose:
                    print(
                        f"  Batch {batch_idx + 1}/{num_batches} "
                        f"({len(batch_candidates)} candidates) "
                        f"approved {batch_approved}; took {batch_duration:.2f}s"
                    )

            total_processed += stratum_size
            total_approved += stratum_approved
            stratum_duration = time.perf_counter() - stratum_start

            # Calculate approval rate for this stratum
            approval_rate = stratum_approved / stratum_size if stratum_size > 0 else 0

            if verbose:
                print(
                    f"Stratum {stratum_lower:.2f}-{stratum_upper:.2f}: "
                    f"{stratum_approved}/{stratum_size} approved ({approval_rate*100:.1f}%) "
                    f"in {stratum_duration:.2f}s"
                )

            # Early stopping if approval rate is too low
            if approval_rate < min_approval_rate and total_processed < total:
                remaining = total - total_processed
                if verbose:
                    print(
                        f"\nStopping early: approval rate ({approval_rate*100:.1f}%) "
                        f"below minimum ({min_approval_rate*100:.1f}%). "
                        f"Skipping {remaining} remaining candidates in lower strata."
                    )
                break

        if verbose:
            overall_approval = total_approved / total_processed if total_processed > 0 else 0
            total_duration = time.perf_counter() - overall_start
            print(
                f"\n=== LLM Verification Summary ==="
                f"\nProcessed: {total_processed}/{total} candidates"
                f"\nApproved: {total_approved} edges ({overall_approval*100:.1f}%)"
                f"\nBatches: {total_batches}"
                f"\nTotal LLM verification time: {total_duration:.2f}s"
            )

    return verified_edges


def build_edges(
    intermediate_path: Path,
    cluster_path: Path,
    output_path: Path,
    high_threshold: float = 0.8,
    medium_threshold: float = 0.6,
    use_llm_verification: bool = True,
    batch_size: int = 40,
    stratum_width: float = 0.05,
    min_approval_rate: float = 0.10,
    verbose: bool = True,
) -> None:
    """
    Build edges from intermediate results and cluster assignments.

    Uses a two-tier approach:
    1. High-confidence edges (similarity >= high_threshold) are accepted automatically
    2. Medium-confidence edges (medium_threshold <= similarity < high_threshold) are
       verified using LLM with stratification and batch processing for efficiency

    Args:
        intermediate_path: Path to intermediate JSON (embeddings + conversations)
        cluster_path: Path to cluster assignments JSON
        output_path: Path to output edges JSON
        high_threshold: High confidence threshold (default: 0.8)
        medium_threshold: Medium confidence threshold (default: 0.6)
        use_llm_verification: Whether to use LLM verification for medium-confidence edges
        batch_size: Number of candidates to process per LLM API call (default: 40)
        stratum_width: Width of each similarity stratum for LLM verification (default: 0.05)
        min_approval_rate: Stop processing lower strata if approval drops below this (default: 0.10)
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Loading intermediate results from {intermediate_path}")

    # Load intermediate results
    with open(intermediate_path, "r", encoding="utf-8") as f:
        intermediate_data = json.load(f)

    conversations = intermediate_data["conversations"]
    embeddings = np.array(intermediate_data["embeddings"], dtype=np.float32)

    if verbose:
        print(f"Loaded {len(conversations)} conversations with embeddings")

    # Load cluster assignments
    if verbose:
        print(f"Loading cluster assignments from {cluster_path}")

    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    # Create cluster_assignments dict: {conversation_id: cluster_id}
    cluster_assignments: Dict[int, str] = {}

    assignments = cluster_data.get("assignments")
    if isinstance(assignments, list) and assignments:
        for entry in assignments:
            conv_id = entry.get("conversation_id")
            cluster_id = entry.get("cluster_id") or entry.get("cluster")
            if conv_id is None or cluster_id is None:
                continue
            cluster_assignments[int(conv_id)] = str(cluster_id)
    else:
        for cluster in cluster_data.get("clusters", []):
            cluster_id = cluster.get("cluster_id") or cluster.get("id")
            conv_ids = cluster.get("conversation_ids") or cluster.get("members") or []
            if cluster_id is None:
                continue
            for conv_id in conv_ids:
                cluster_assignments[int(conv_id)] = str(cluster_id)

    if not cluster_assignments and verbose:
        print(
            "Warning: No cluster assignments were found in the cluster file; treating all nodes as unclustered."
        )

    if verbose:
        print(f"Loaded cluster assignments for {len(cluster_assignments)} conversations")

    # Compute similarity matrix
    if verbose:
        print("Computing cosine similarity matrix...")

    similarity_matrix = compute_cosine_similarity(embeddings)
    similarity_stats = compute_similarity_stats(similarity_matrix)
    (
        effective_high_threshold,
        effective_medium_threshold,
        thresholds_adjusted,
    ) = adjust_similarity_thresholds(
        similarity_stats, high_threshold, medium_threshold, verbose=verbose
    )

    if verbose:
        print(
            "Generating edge candidates "
            f"(high_threshold={effective_high_threshold:.3f}, "
            f"medium_threshold={effective_medium_threshold:.3f})..."
        )

    # Generate edge candidates
    confirmed_edges, candidates_for_llm = generate_edge_candidates(
        similarity_matrix,
        cluster_assignments,
        effective_high_threshold,
        effective_medium_threshold,
    )

    if verbose:
        print(f"Generated {len(confirmed_edges)} high-confidence edges")
        print(f"Generated {len(candidates_for_llm)} medium-confidence candidates")
        if candidates_for_llm:
            medium_nodes = {c.source_id for c in candidates_for_llm} | {
                c.target_id for c in candidates_for_llm
            }
            print(
                f"Medium-confidence candidates involve {len(medium_nodes)} unique nodes"
            )

    # Verify medium-confidence candidates with LLM
    verified_edges: List[Edge] = []
    if use_llm_verification and candidates_for_llm:
        if verbose:
            print("Verifying medium-confidence candidates with LLM...")

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Warning: OPENAI_API_KEY not found in environment. Skipping LLM verification."
            )
        else:
            llm_client = OpenAI(api_key=api_key)
            verify_start = time.perf_counter()
            verified_edges = verify_candidates_with_llm(
                candidates_for_llm,
                conversations,
                llm_client,
                batch_size=batch_size,
                stratum_width=stratum_width,
                min_approval_rate=min_approval_rate,
                verbose=verbose,
            )
            if verbose:
                verify_duration = time.perf_counter() - verify_start
                print(
                    f"LLM verification completed: {len(verified_edges)} edges in "
                    f"{verify_duration:.2f}s"
                )
    elif not use_llm_verification and verbose:
        print("LLM verification disabled, skipping medium-confidence candidates")

    # Combine all edges
    all_edges = confirmed_edges + verified_edges

    # Compute statistics
    total_edges = len(all_edges)
    intra_count = sum(1 for edge in all_edges if edge.is_intra_cluster)
    inter_count = total_edges - intra_count
    high_count = sum(1 for edge in all_edges if edge.confidence == "high")
    llm_count = sum(1 for edge in all_edges if edge.confidence == "llm_verified")

    if verbose:
        print("\n=== Edge Statistics ===")
        print(f"Total edges: {total_edges}")
        print(f"Intra-cluster edges: {intra_count} ({100*intra_count/total_edges:.1f}%)" if total_edges > 0 else "Intra-cluster edges: 0")
        print(f"Inter-cluster edges: {inter_count} ({100*inter_count/total_edges:.1f}%)" if total_edges > 0 else "Inter-cluster edges: 0")
        print(f"High-confidence edges: {high_count}")
        print(f"LLM-verified edges: {llm_count}")

    # Save to output JSON
    output_data = {
        "edges": [asdict(edge) for edge in all_edges],
        "metadata": {
            "total_edges": total_edges,
            "intra_cluster_edges": intra_count,
            "inter_cluster_edges": inter_count,
            "high_confidence_edges": high_count,
            "llm_verified_edges": llm_count,
            "thresholds": {
                "requested": {
                    "high": high_threshold,
                    "medium": medium_threshold,
                },
                "effective": {
                    "high": effective_high_threshold,
                    "medium": effective_medium_threshold,
                },
                "adjusted": thresholds_adjusted,
            },
            "similarity_stats": similarity_stats,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nEdges saved to {output_path.resolve()}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for edge generation."""
    parser = argparse.ArgumentParser(
        description="Generate edges from embeddings and cluster assignments with optional LLM verification."
    )
    parser.add_argument(
        "--intermediate",
        "--features",
        type=Path,
        required=True,
        help="Path to intermediate/features JSON (embeddings + conversations)",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to cluster assignments JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output edges JSON",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.8,
        help="High confidence threshold (default: 0.8)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.6,
        help="Medium confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM verification for medium-confidence edges",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of edge candidates to process per LLM API call (default: 40)",
    )
    parser.add_argument(
        "--stratum-width",
        type=float,
        default=0.05,
        help="Width of each similarity stratum for stratified processing (default: 0.05)",
    )
    parser.add_argument(
        "--min-approval-rate",
        type=float,
        default=0.10,
        help="Stop processing lower strata if approval rate drops below this threshold (default: 0.10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    build_edges(
        intermediate_path=args.intermediate,
        cluster_path=args.clusters,
        output_path=args.output,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
        use_llm_verification=not args.no_llm,
        batch_size=args.batch_size,
        stratum_width=args.stratum_width,
        min_approval_rate=args.min_approval_rate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
