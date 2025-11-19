import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans


def load_qa_embeddings(path: Path) -> Dict[str, Dict[str, Any]]:
    """qa_embeddings.pkl 로드.

    예상 구조:
    { qa_id(str): {"conversation_id": int, "qa_index": int, "embedding": list[float], ... }, ... }
    """
    import pickle

    with path.open("rb") as f:
        data = pickle.load(f)
    return data
def cluster_qa_single_conv(
    qa_emb_path: Path,
    conversation_id: int,
    algo: str,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    n_clusters_kmeans: int | None,
    output_path: Path,
) -> None:
    qa_index = load_qa_embeddings(qa_emb_path)
    if not qa_index:
        print(f"No QA embeddings in {qa_emb_path}")
        return

    qa_ids: List[str] = []
    X_list: List[np.ndarray] = []

    for qa_id, rec in qa_index.items():
        conv_id = rec.get("conversation_id")
        if int(conv_id) != int(conversation_id):
            continue
        emb = rec.get("embedding")
        if emb is None:
            continue
        v = np.asarray(emb, dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        qa_ids.append(str(qa_id))
        X_list.append(v)

    if not X_list:
        print(f"No valid embeddings for conversation_id={conversation_id} to cluster.")
        return

    X = np.stack(X_list, axis=0)
    print(
        f"Loaded {X.shape[0]} QA embeddings for conversation_id={conversation_id}, dim={X.shape[1]}"
    )

    if algo == "hdbscan":
        print(
            f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric})..."
        )
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
    else:
        # KMeans: noise 개념 없이 0..k-1 클러스터
        if n_clusters_kmeans is None:
            # 대충 sqrt(N) 정도를 기본 클러스터 수로 설정
            n_clusters_kmeans = max(2, int(len(X_list) ** 0.5))
        print(f"Running KMeans (n_clusters={n_clusters_kmeans})...")
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
        labels = kmeans.fit_predict(X)
        n_clusters = int(len(set(labels)))
        n_noise = 0

    print(f"Found {n_clusters} clusters, noise points: {n_noise}")

    clusters: Dict[int, Dict[str, Any]] = {}
    for qa_id, label in zip(qa_ids, labels):
        label_int = int(label)
        rec = qa_index.get(qa_id, {})
        conv_id = rec.get("conversation_id")
        qa_index_val = rec.get("qa_index")

        info = clusters.setdefault(
            label_int,
            {
                "cluster_id": label_int,
                "size": 0,
                "qa_ids": [],
                "conversation_id": conversation_id,
                "qa_indices": [],
            },
        )
        info["size"] += 1
        info["qa_ids"].append(qa_id)
        info["qa_indices"].append(qa_index_val)

    out = {
        "qa_embeddings_path": str(qa_emb_path),
        "conversation_id": conversation_id,
        "algo": algo,
        "min_cluster_size": min_cluster_size if algo == "hdbscan" else None,
        "min_samples": min_samples if algo == "hdbscan" else None,
        "metric": metric if algo == "hdbscan" else None,
        "n_clusters_kmeans": n_clusters_kmeans if algo == "kmeans" else None,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "clusters": sorted(clusters.values(), key=lambda c: c["cluster_id"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n클러스터 결과 저장: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster QA embeddings for a single conversation (HDBSCAN or KMeans)"
    )
    parser.add_argument(
        "--qa-embeddings",
        type=str,
        default="output/qa_embeddings.pkl",
        help="qa_embeddings.pkl 경로 (기본: output/qa_embeddings.pkl)",
    )
    parser.add_argument(
        "--conversation-id",
        type=int,
        required=True,
        help="단일 conversation_id (예: 283)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="hdbscan",
        choices=["hdbscan", "kmeans"],
        help="클러스터링 알고리즘 선택: hdbscan 또는 kmeans (기본: hdbscan)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="HDBSCAN min_cluster_size (algo=hdbscan일 때만 사용, 기본: 10)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (기본: None = min_cluster_size와 동일)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="거리 metric (기본: euclidean, 임베딩이 정규화된 cosine이면 euclidean≈cosine)",
    )
    parser.add_argument(
        "--n-clusters-kmeans",
        type=int,
        default=None,
        help="KMeans 클러스터 개수 (기본: None = sqrt(N)로 자동 추정)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="클러스터 결과 JSON 경로 (기본: output/qa_clusters_{conv}_{algo}.json)",
    )

    args = parser.parse_args()

    qa_emb_path = Path(args.qa_embeddings)

    if args.output is None:
        default_name = f"qa_clusters_{args.conversation_id}_{args.algo}.json"
        output_path = Path("output") / default_name
    else:
        output_path = Path(args.output)

    min_samples = (
        args.min_samples if args.min_samples is not None else args.min_cluster_size
    )

    cluster_qa_single_conv(
        qa_emb_path=qa_emb_path,
        conversation_id=args.conversation_id,
        algo=args.algo,
        min_cluster_size=args.min_cluster_size,
        min_samples=min_samples,
        metric=args.metric,
        n_clusters_kmeans=args.n_clusters_kmeans,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
