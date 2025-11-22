import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
from hdbscan import HDBSCAN

# Ensure local imports work from tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.extract_qa_keywords import extract_keywords_for_conv  # type: ignore


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cluster_embeddings(qa_ids: List[str], X: np.ndarray, min_cluster_size: int, min_samples: int | None, metric: str) -> Dict[str, Any]:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int(np.sum(labels == -1))

    clusters: Dict[int, Dict[str, Any]] = {}
    for qa_id, label in zip(qa_ids, labels):
        lab = int(label)
        info = clusters.setdefault(
            lab,
            {
                "cluster_id": lab,
                "size": 0,
                "qa_ids": [],
                "qa_indices": [],
            },
        )
        info["size"] += 1
        info["qa_ids"].append(qa_id)

    return {
        "algo": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples if min_samples is not None else min_cluster_size,
        "metric": metric,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "clusters": sorted(clusters.values(), key=lambda c: c["cluster_id"]),
    }


def length_weighted_pool(
    qa_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    qa_lengths: Dict[str, float],
) -> Tuple[np.ndarray, float]:
    weights = np.array([max(qa_lengths.get(qid, 0.0), 0.0) for qid in qa_ids], dtype=np.float64)
    # Fallback to uniform if all zero
    if not np.any(weights):
        weights = np.ones(len(qa_ids), dtype=np.float64)
    W = np.sum(weights)
    mat = np.stack([embeddings[qid] for qid in qa_ids], axis=0)
    pooled = (weights[:, None] * mat).sum(axis=0) / W
    return pooled, float(W)


def build_cluster_and_conversation_embeddings(
    conv_id: int,
    qa_pairs: List[Dict[str, Any]],
    emb_records: List[Dict[str, Any]],
    cluster_json: Dict[str, Any],
    out_dir_embeddings: Path,
) -> Tuple[Path, Path]:
    # Build lookup maps
    qa_len: Dict[str, float] = {}
    for p in qa_pairs:
        if int(p.get("conversation_id", -1)) != int(conv_id):
            continue
        qa_id = str(p.get("qa_id"))
        q = p.get("question", "") or ""
        a = p.get("answer", "") or ""
        # simple char-length; replace with tokens if desired
        qa_len[qa_id] = float(len(q) + len(a))

    qa_emb: Dict[str, np.ndarray] = {}
    for rec in emb_records:
        qa_id = str(rec.get("qa_id"))
        v = np.asarray(rec.get("qa_embedding"), dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        qa_emb[qa_id] = v

    # Special case: only one QA in this conversation -> use its embedding directly
    if len(emb_records) == 1:
        only = emb_records[0]
        only_id = str(only.get("qa_id"))
        only_vec = np.asarray(only.get("qa_embedding"), dtype=np.float32)
        if only_vec.ndim == 2 and only_vec.shape[0] == 1:
            only_vec = only_vec[0]
        # weight from precomputed lengths
        only_weight = float(qa_len.get(only_id, 1.0))

        out_dir_embeddings.mkdir(parents=True, exist_ok=True)
        per_cluster_path = out_dir_embeddings / f"conv_{conv_id}_cluster_embeddings.pkl"
        with per_cluster_path.open("wb") as f:
            pickle.dump(
                {
                    0: {
                        "cluster_id": 0,
                        "size": 1,
                        "weight_sum": only_weight,
                        "embedding": only_vec.astype(np.float32).tolist(),
                        "qa_ids": [only_id],
                    }
                },
                f,
            )

        conv_path = out_dir_embeddings / f"conversation_embedding_{conv_id}.pkl"
        with conv_path.open("wb") as f:
            pickle.dump(
                {
                    "conversation_id": conv_id,
                    "embedding": only_vec.astype(np.float32).tolist(),
                    "weight_sum": only_weight,
                    "n_qas": 1,
                },
                f,
            )

        return per_cluster_path, conv_path

    # Per-cluster length-weighted pooling
    cluster_embs: Dict[int, Dict[str, Any]] = {}
    for c in cluster_json.get("clusters", []):
        cid = int(c.get("cluster_id", -1))
        # exclude noise cluster (-1)
        if cid < 0:
            continue
        ids = [str(x) for x in c.get("qa_ids", [])]
        valid_ids = [qid for qid in ids if qid in qa_emb]
        if not valid_ids:
            continue
        pooled, weight_sum = length_weighted_pool(valid_ids, qa_emb, qa_len)
        cluster_embs[cid] = {
            "cluster_id": cid,
            "size": int(c.get("size", len(valid_ids))),
            "weight_sum": weight_sum,
            "embedding": pooled.astype(np.float32).tolist(),
            "qa_ids": valid_ids,
        }

    # Conversation-level pooling from all QAs (equivalently, weight over clusters by weight_sum)
    all_ids = list(qa_emb.keys())
    if all_ids:
        conv_vec, conv_w = length_weighted_pool(all_ids, qa_emb, qa_len)
    else:
        conv_vec = np.zeros_like(next(iter(qa_emb.values()))) if qa_emb else np.zeros((384,), dtype=np.float32)
        conv_w = 0.0

    out_dir_embeddings.mkdir(parents=True, exist_ok=True)
    per_cluster_path = out_dir_embeddings / f"conv_{conv_id}_cluster_embeddings.pkl"
    with per_cluster_path.open("wb") as f:
        pickle.dump(cluster_embs, f)

    conv_path = out_dir_embeddings / f"conversation_embedding_{conv_id}.pkl"
    with conv_path.open("wb") as f:
        pickle.dump({
            "conversation_id": conv_id,
            "embedding": conv_vec.astype(np.float32).tolist(),
            "weight_sum": conv_w,
            "n_qas": len(all_ids),
        }, f)

    return per_cluster_path, conv_path


def run_pipeline(
    conversation_id: int,
    qa_pairs_path: Path,
    model_name: str,
    ngram_max: int,
    max_candidates: int,
    top_n: int,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
) -> None:
    t0 = time.time()

    out_keywords = PROJECT_ROOT / "output" / "keywords" / f"qa_keywords_custom_{conversation_id}_e5.json"
    out_keywords.parent.mkdir(parents=True, exist_ok=True)

    # 1) Extract QA keywords AND per-QA embeddings (saved as output/embeddings/qa_keyword_embeddings_{conv}.pkl)
    t_kw_start = time.time()
    extract_keywords_for_conv(
        qa_pairs_path=qa_pairs_path,
        conv_id_target=conversation_id,
        model_name=model_name,
        ngram_max=ngram_max,
        max_candidates=max_candidates,
        top_n=top_n,
        output_path=out_keywords,
    )
    kw_elapsed = time.time() - t_kw_start

    # Load the saved per-QA embeddings produced by extract_qa_keywords.py
    emb_pkl = PROJECT_ROOT / "output" / "embeddings" / f"qa_keyword_embeddings_{conversation_id}.pkl"
    if not emb_pkl.exists():
        raise FileNotFoundError(f"Expected embeddings not found: {emb_pkl}")
    with emb_pkl.open("rb") as f:
        emb_records: List[Dict[str, Any]] = pickle.load(f)

    qa_ids: List[str] = []
    X_list: List[np.ndarray] = []
    for rec in emb_records:
        qa_id = str(rec.get("qa_id"))
        qa_ids.append(qa_id)
        v = np.asarray(rec.get("qa_embedding"), dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        X_list.append(v)

    if not X_list:
        print(f"No embeddings to cluster for conversation_id={conversation_id}")
        return

    X = np.stack(X_list, axis=0)

    # 2) HDBSCAN clustering
    t_clu_start = time.time()
    # If only one QA, emit a singleton cluster (id=0)
    if X.shape[0] == 1:
        cluster_json = {
            "algo": "singleton",
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples if min_samples is not None else min_cluster_size,
            "metric": metric,
            "n_clusters": 1,
            "n_noise": 0,
            "clusters": [
                {
                    "cluster_id": 0,
                    "size": 1,
                    "qa_ids": qa_ids,
                    "qa_indices": [0],
                }
            ],
        }
    else:
        # HDBSCAN requires k=min_samples neighbors; effectively needs at least (min_samples+1) points
        min_required = max(2, (min_samples if min_samples is not None else min_cluster_size) + 1, min_cluster_size)
        if X.shape[0] < min_required:
            # Too few points to run HDBSCAN safely; mark all as noise
            cluster_json = {
                "algo": "hdbscan",
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples if min_samples is not None else min_cluster_size,
                "metric": metric,
                "n_clusters": 0,
                "n_noise": int(X.shape[0]),
                "clusters": [
                    {
                        "cluster_id": -1,
                        "size": int(X.shape[0]),
                        "qa_ids": qa_ids,
                        "qa_indices": list(range(len(qa_ids))),
                    }
                ],
            }
        else:
            cluster_json = cluster_embeddings(
                qa_ids=qa_ids,
                X=X,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
            )
    cluster_json.update({
        "conversation_id": conversation_id,
    })

    out_clusters = PROJECT_ROOT / "output" / "cluster_results" / f"qa_clusters_{conversation_id}_hdbscan_{metric}.json"
    out_clusters.parent.mkdir(parents=True, exist_ok=True)
    out_clusters.write_text(json.dumps(cluster_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"클러스터 결과 저장: {out_clusters}")
    clu_elapsed = time.time() - t_clu_start

    # 3) Length-weighted pooling: per-cluster + per-conversation
    t_pool_start = time.time()
    qa_pairs = load_qa_pairs(qa_pairs_path)
    cluster_embs_path, conv_emb_path = build_cluster_and_conversation_embeddings(
        conv_id=conversation_id,
        qa_pairs=qa_pairs,
        emb_records=emb_records,
        cluster_json=cluster_json,
        out_dir_embeddings=PROJECT_ROOT / "output" / "embeddings",
    )
    print(f"클러스터 대표 임베딩 저장: {cluster_embs_path}")
    print(f"대화 임베딩 저장(길이 가중 풀링): {conv_emb_path}")
    pool_elapsed = time.time() - t_pool_start

    total_elapsed = time.time() - t0

    print("=" * 80)
    print("Pipeline timings (conversation_id={})".format(conversation_id))
    print("- Keywords + per-QA embeddings: {:.2f}s".format(kw_elapsed))
    print("- Clustering (HDBSCAN): {:.2f}s".format(clu_elapsed))
    print("- Pooling (clusters + conversation): {:.2f}s".format(pool_elapsed))
    print("- Total: {:.2f}s".format(total_elapsed))
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description="Run per-conversation pipeline: keywords -> HDBSCAN clustering -> length-weighted pooled embeddings")
    p.add_argument("--conversation-id", type=int, required=True)
    p.add_argument("--qa-pairs", type=str, default=str(PROJECT_ROOT / "output" / "qa_pairs.json"))
    p.add_argument("--model", type=str, default="intfloat/multilingual-e5-base")
    p.add_argument("--ngram-max", type=int, default=1)
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--min-samples", type=int, default=1)
    p.add_argument("--metric", type=str, default="euclidean")

    args = p.parse_args()

    # Resolve qa_pairs path: allow either absolute or path relative to project root
    qa_pairs_path = Path(args.qa_pairs)
    if not qa_pairs_path.is_absolute():
        qa_pairs_path = PROJECT_ROOT / qa_pairs_path
    if not qa_pairs_path.exists():
        raise FileNotFoundError(f"qa_pairs not found at: {qa_pairs_path}. Build it with: python build_qa_pairs.py --data-path data/conversations.json --output output/qa_pairs.json")

    run_pipeline(
        conversation_id=args.conversation_id,
        qa_pairs_path=qa_pairs_path,
        model_name=args.model,
        ngram_max=args.ngram_max,
        max_candidates=args.max_candidates,
        top_n=args.top_n,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
