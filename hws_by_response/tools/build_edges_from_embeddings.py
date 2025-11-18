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


def load_responses(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_doc_index(responses: List[Dict[str, Any]]) -> Dict[Any, int]:
    return {r["response_id"]: i for i, r in enumerate(responses)}


def align_embeddings(responses: List[Dict[str, Any]], payload: Dict[str, Any]) -> np.ndarray:
    idx = build_doc_index(responses)

    items: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
        items = payload["items"]
    elif isinstance(payload, dict) and payload:
        for rid, val in payload.items():
            if rid == "items":
                continue
            if isinstance(val, dict) and "embedding" in val:
                items.append({"response_id": rid, "embedding": val["embedding"]})
            else:
                items.append({"response_id": rid, "embedding": val})
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    emb_dim = None
    for it in items:
        vec = it.get("embedding")
        if isinstance(vec, (list, tuple, np.ndarray)) and len(vec) > 0:
            emb_dim = len(vec)
            break
    if emb_dim is None:
        emb_dim = 0

    embs = np.zeros((len(responses), emb_dim), dtype=np.float32) if emb_dim > 0 else np.zeros((len(responses), 0), dtype=np.float32)

    count = 0
    for it in items:
        rid = it.get("response_id")
        vec = it.get("embedding")
        if rid in idx and isinstance(vec, (list, tuple, np.ndarray)):
            embs[idx[rid]] = np.asarray(vec, dtype=np.float32)
            count += 1

    try:
        zero_rows = int((embs.shape[1] == 0) or (np.sum(np.linalg.norm(embs, axis=1) == 0) if embs.size > 0 else len(responses)))
    except Exception:
        zero_rows = 0
    print(f"Embeddings matched: {count}/{len(responses)} | dim={emb_dim} | zero_rows={zero_rows}")
    return embs


def pool_by_conversation_length(
    responses: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Dict[Any, Dict[str, Any]]:
    """길이 가중 평균으로 대화별 벡터 풀링.

    각 응답의 길이(문자 수)를 weight로 사용해서 conversation별 가중 평균 벡터를 만든다.
    """
    if embeddings.size == 0:
        return {}

    conv_vectors: Dict[Any, Dict[str, Any]] = {}

    # conversation_id별로 모으기
    tmp: Dict[Any, List[int]] = {}
    for i, r in enumerate(responses):
        cid = r.get("conversation_id")
        if cid is None:
            continue
        tmp.setdefault(cid, []).append(i)

    for cid, idxs in tmp.items():
        if not idxs:
            continue
        vecs = embeddings[idxs]
        # 응답 길이 (content 기준) -> weight
        lengths = []
        for i in idxs:
            txt = responses[i].get("content", "") or ""
            lengths.append(len(txt))
        w = np.asarray(lengths, dtype=np.float32)
        if np.all(w == 0):
            w = np.ones_like(w)
        w = w / w.sum()
        # 길이 가중 평균
        pooled = np.average(vecs, axis=0, weights=w)

        conv_vectors[cid] = {
            "vector": pooled,
            "response_ids": [responses[i]["response_id"] for i in idxs],
        }

    try:
        sizes = [len(v["response_ids"]) for v in conv_vectors.values()]
        if sizes:
            import numpy as _np

            print(
                f"Conv pooled vectors: docs={len(sizes)} | avg_responses={float(_np.mean(sizes)):.2f} | "
                f"min={int(min(sizes))} | max={int(max(sizes))}"
            )
        else:
            print("Conv pooled vectors: none")
    except Exception:
        pass

    return conv_vectors


def _l2_normalize_rows(M: np.ndarray) -> np.ndarray:
    if M.size == 0:
        return M
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def compute_similarity(A: np.ndarray, B: np.ndarray, metric: str = "cosine") -> float:
    """A, B는 각각 (1, dim) 형태의 벡터.
    
    metric:
      - 'cosine': 코사인 유사도 (1 = 같음, -1 = 반대, 0 = 직교)
      - 'l2': L2 거리 (작을수록 유사) -> 유사도로 변환: 1 / (1 + dist)
      - 'l1': L1 거리 (작을수록 유사) -> 유사도로 변환: 1 / (1 + dist)
    """
    if metric == "cosine":
        return float(cosine_similarity(A, B)[0, 0])
    elif metric == "l2":
        dist = float(euclidean_distances(A, B)[0, 0])
        # 거리를 유사도로 변환 (0~1 범위)
        return 1.0 / (1.0 + dist)
    elif metric == "l1":
        dist = float(manhattan_distances(A, B)[0, 0])
        # 거리를 유사도로 변환 (0~1 범위)
        return 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_conv_edges_from_pooled(
    conv_vectors: Dict[Any, Dict[str, Any]],
    hard_threshold: float = 0.7,
    pending_threshold_low: float = 0.5,
    pending_threshold_high: float = 0.7,
    normalize: bool = True,
    metric: str = "cosine",
    categories: Dict[Any, str] = None,
) -> tuple:
    """길이 가중 평균 벡터로 대화 간 유사도 계산.

    - hard: similarity >= hard_threshold (기본 0.7)
    - pending: pending_threshold_low <= similarity < pending_threshold_high (기본 0.5~0.7)
    
    Returns:
        (edges, stats_dict)
    """
    cids = list(conv_vectors.keys())
    edges: List[Dict[str, Any]] = []
    sims_all: List[float] = []
    top_k: List[Any] = []

    # 행렬로 한번에 모아두면 조금 더 빠를 수 있지만, 여기서는 pairwise 루프 유지
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a = cids[i]
            b = cids[j]
            va = conv_vectors.get(a)
            vb = conv_vectors.get(b)
            if not va or not vb:
                continue
            A = np.asarray(va["vector"], dtype=np.float32).reshape(1, -1)
            B = np.asarray(vb["vector"], dtype=np.float32).reshape(1, -1)
            if A.size == 0 or B.size == 0:
                continue
            if normalize and metric == "cosine":
                A = _l2_normalize_rows(A)
                B = _l2_normalize_rows(B)
            sim = compute_similarity(A, B, metric=metric)
            sims_all.append(sim)

            status = None
            if sim >= hard_threshold:
                status = "hard"
            elif pending_threshold_low <= sim < pending_threshold_high:
                status = "pending"

            if status:
                edges.append({"source": a, "target": b, "similarity": sim, "status": status})

            try:
                if len(top_k) < 10:
                    top_k.append((sim, a, b))
                    top_k.sort(reverse=True)
                elif sim > top_k[-1][0]:
                    top_k[-1] = (sim, a, b)
                    top_k.sort(reverse=True)
            except Exception:
                pass

    # 통계 계산
    stats = {
        "total_edges": len(edges),
        "within_category_edges": 0,
        "cross_category_edges": 0,
        "hard_within_category_edges": 0,
        "hard_cross_category_edges": 0,
        "pending_within_category_edges": 0,
        "pending_cross_category_edges": 0,
    }

    if categories:
        cat_map = categories
        for e in edges:
            a = e["source"]
            b = e["target"]
            if cat_map.get(a) and cat_map.get(b):
                if cat_map[a] == cat_map[b]:
                    stats["within_category_edges"] += 1
                    if e.get("status") == "hard":
                        stats["hard_within_category_edges"] += 1
                    elif e.get("status") == "pending":
                        stats["pending_within_category_edges"] += 1
                else:
                    stats["cross_category_edges"] += 1
                    if e.get("status") == "hard":
                        stats["hard_cross_category_edges"] += 1
                    elif e.get("status") == "pending":
                        stats["pending_cross_category_edges"] += 1

    try:
        if sims_all:
            import numpy as _np

            arr = _np.asarray(sims_all, dtype=float)
            print(
                f"Similarity dist: n={arr.size} | mean={float(arr.mean()):.4f} | "
                f"median={float(_np.median(arr)):.4f} | std={float(arr.std()):.4f} | "
                f"max={float(arr.max()):.4f} | min={float(arr.min()):.4f}"
            )
            stats["similarity_dist"] = {
                "n": int(arr.size),
                "mean": round(float(arr.mean()), 4),
                "median": round(float(_np.median(arr)), 4),
                "std": round(float(arr.std()), 4),
                "max": round(float(arr.max()), 4),
                "min": round(float(arr.min()), 4),
            }
            if top_k:
                print("Top edges (score, source, target):")
                for s, a, b in top_k:
                    print(f"  {s:.4f}\t{a}\t{b}")
        else:
            print("Similarity dist: no comparable pairs")
    except Exception:
        pass

    return edges, stats, sims_all


def export_edges(edges: List[Dict[str, Any]], output_dir: Path, metric: str = "cosine") -> None:
    hard = [e for e in edges if e["status"] == "hard"]
    pending = [e for e in edges if e["status"] == "pending"]
    suffix = f"_{metric}" if metric != "cosine" else ""
    with open(output_dir / f"s5_edges_hard_lenpool{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(hard, f, ensure_ascii=False, indent=2)
    with open(output_dir / f"s5_edges_pending_lenpool{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


def export_stats(stats: Dict[str, Any], output_dir: Path, metric: str = "cosine") -> None:
    suffix = f"_{metric}" if metric != "cosine" else ""
    with open(output_dir / f"s7_stats_lenpool{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def plot_similarity_distribution(
    sims_all: List[float],
    output_dir: Path,
    metric: str = "cosine",
    hard_threshold: float = 0.7,
    pending_low: float = 0.5,
    pending_high: float = 0.7,
) -> None:
    """Similarity 분포를 히스토그램과 통계로 시각화해서 PNG로 저장."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return

    if not sims_all:
        print("분포 시각화: 비교할 쌍이 없습니다.")
        return

    arr = np.asarray(sims_all, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Similarity Distribution ({metric.upper()})", fontsize=16, fontweight="bold")

    # 1. 히스토그램 (상단 좌)
    ax = axes[0, 0]
    ax.hist(arr, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(hard_threshold, color="red", linestyle="--", linewidth=2, label=f"Hard threshold ({hard_threshold})")
    ax.axvline(pending_low, color="orange", linestyle="--", linewidth=2, label=f"Pending low ({pending_low})")
    ax.axvline(pending_high, color="orange", linestyle=":", linewidth=2, label=f"Pending high ({pending_high})")
    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Histogram", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 누적 분포 (상단 우)
    ax = axes[0, 1]
    sorted_arr = np.sort(arr)
    cumsum = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax.plot(sorted_arr, cumsum, linewidth=2, color="darkblue")
    ax.axvline(hard_threshold, color="red", linestyle="--", linewidth=2, label=f"Hard ({hard_threshold})")
    ax.axvline(pending_low, color="orange", linestyle="--", linewidth=2, label=f"Pending low ({pending_low})")
    ax.axvline(pending_high, color="orange", linestyle=":", linewidth=2, label=f"Pending high ({pending_high})")
    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 박스플롯 (하단 좌)
    ax = axes[1, 0]
    bp = ax.boxplot(arr, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax.axhline(hard_threshold, color="red", linestyle="--", linewidth=2, label=f"Hard ({hard_threshold})")
    ax.axhline(pending_low, color="orange", linestyle="--", linewidth=2, label=f"Pending low ({pending_low})")
    ax.axhline(pending_high, color="orange", linestyle=":", linewidth=2, label=f"Pending high ({pending_high})")
    ax.set_ylabel("Similarity Score", fontsize=11)
    ax.set_title("Box Plot", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. 통계 정보 (하단 우)
    ax = axes[1, 1]
    ax.axis("off")

    # 카테고리별 개수 (threshold 기반)
    hard_count = np.sum(arr >= hard_threshold)
    pending_count = np.sum((arr >= pending_low) & (arr < pending_high))
    other_count = len(arr) - hard_count - pending_count

    stats_text = f"""
Similarity Statistics ({metric.upper()})

Total pairs: {len(arr):,}

Hard edges (≥ {hard_threshold}): {hard_count:,} ({100*hard_count/len(arr):.2f}%)
Pending edges ({pending_low} ≤ x < {pending_high}): {pending_count:,} ({100*pending_count/len(arr):.2f}%)
Other edges (< {pending_low}): {other_count:,} ({100*other_count/len(arr):.2f}%)

Mean: {float(arr.mean()):.4f}
Median: {float(np.median(arr)):.4f}
Std Dev: {float(arr.std()):.4f}
Min: {float(arr.min()):.4f}
Max: {float(arr.max()):.4f}
Q1: {float(np.percentile(arr, 25)):.4f}
Q3: {float(np.percentile(arr, 75)):.4f}
    """

    ax.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    suffix = f"_{metric}" if metric != "cosine" else ""
    output_path = output_dir / f"similarity_dist{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"분포 시각화 저장: {output_path}")
    plt.close()


def load_categories(path: str) -> Dict[Any, Dict[str, Any]]:
    """카테고리 정보 로드. 반환: {conversation_id: {category, conversation_title, ...}}"""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Case 1: full result file with 'assignments' object
    if isinstance(data, dict) and "assignments" in data and isinstance(data["assignments"], dict):
        assign = data["assignments"]
        out: Dict[Any, Dict[str, Any]] = {}
        for k, v in assign.items():
            cid = int(k) if isinstance(k, str) and k.isdigit() else k
            if isinstance(v, dict):
                out[cid] = v
            else:
                out[cid] = {"category": v}
        return out
    
    # Case 2: direct mapping {conversation_id: {...}}
    if isinstance(data, dict):
        out: Dict[Any, Dict[str, Any]] = {}
        for k, v in data.items():
            cid = int(k) if isinstance(k, str) and k.isdigit() else k
            if isinstance(v, dict):
                out[cid] = v
            else:
                out[cid] = {"category": v}
        return out
    
    # Fallback: unsupported format
    return {}


def build_graph_json(
    responses: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    categories: Dict[Any, Dict[str, Any]],
    output_dir: Path,
    metric: str = "cosine",
) -> None:
    """graph.json 생성: nodes, edges, clusters, stats"""
    
    # 1. 카테고리별 cluster_id 매핑 생성
    unique_categories = sorted(set(
        categories.get(cid, {}).get("category", "Uncategorized") 
        for cid in set(r.get("conversation_id") for r in responses)
    ))
    category_to_cluster_id = {cat: f"cluster_{i}" for i, cat in enumerate(unique_categories)}
    
    # 2. Nodes 생성
    nodes = []
    conv_id_to_node_id = {}  # conversation_id -> node_id 매핑
    
    # responses에서 conversation_id별 title 추출
    conv_id_to_title = {}
    for r in responses:
        conv_id = r.get("conversation_id")
        if conv_id and conv_id not in conv_id_to_title:
            conv_id_to_title[conv_id] = r.get("conversation_title", f"Conversation {conv_id}")
    
    for idx, conv_id in enumerate(sorted(set(r.get("conversation_id") for r in responses))):
        cat_info = categories.get(conv_id, {})
        category = cat_info.get("category", "Uncategorized")
        # responses에서 title 가져오기, 없으면 categories에서, 그것도 없으면 기본값
        title = conv_id_to_title.get(conv_id, cat_info.get("conversation_title", f"Conversation {conv_id}"))
        
        # 해당 대화의 응답 개수
        num_messages = len([r for r in responses if r.get("conversation_id") == conv_id])
        
        node = {
            "id": idx,
            "orig_id": str(conv_id),
            "cluster_id": category_to_cluster_id.get(category, "cluster_unknown"),
            "cluster_name": category,
            "title": title,
            "timestamp": None,  # 필요시 responses에서 추출
            "num_messages": num_messages,
        }
        nodes.append(node)
        conv_id_to_node_id[conv_id] = idx
    
    # 2. Edges 변환 (conversation_id -> node_id) - hard와 insight만
    graph_edges = []
    for edge in edges:
        status = edge.get("status", "pending")
        # hard와 insight 엣지만 포함
        if status not in ["hard", "insight"]:
            continue
        
        source_conv = edge.get("source")
        target_conv = edge.get("target")
        
        if source_conv not in conv_id_to_node_id or target_conv not in conv_id_to_node_id:
            continue
        
        source_id = conv_id_to_node_id[source_conv]
        target_id = conv_id_to_node_id[target_conv]
        
        # 같은 클러스터인지 확인
        source_cluster = nodes[source_id]["cluster_name"]
        target_cluster = nodes[target_id]["cluster_name"]
        intra_cluster = source_cluster == target_cluster
        
        graph_edge = {
            "source": source_id,
            "target": target_id,
            "weight": round(edge.get("similarity", 0.0), 3),
            "type": status,  # "hard" or "insight"
            "intraCluster": intra_cluster,
        }
        graph_edges.append(graph_edge)
    
    # 3. Clusters 생성 (카테고리 기반)
    cluster_map = {}  # cluster_id -> {name, size, ...}
    for node in nodes:
        cluster_id = node["cluster_id"]
        cluster_name = node["cluster_name"]
        
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = {
                "id": cluster_id,
                "name": cluster_name,
                "description": f"Conversations related to {cluster_name}.",
                "size": 0,
                "themes": [],
            }
        cluster_map[cluster_id]["size"] += 1
    
    clusters = list(cluster_map.values())
    
    # 4. Stats
    stats = {
        "nodes": len(nodes),
        "edges": len(graph_edges),
        "clusters": len(clusters),
    }
    
    # 5. 최종 graph 객체
    graph = {
        "nodes": nodes,
        "edges": graph_edges,
        "clusters": clusters,
        "stats": stats,
    }
    
    # 6. 저장
    suffix = f"_{metric}" if metric != "cosine" else ""
    output_path = output_dir / f"graph{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Graph JSON 저장: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Conversation-level edges from response_embeddings.pkl (length-weighted pooling)")
    parser.add_argument("--responses", type=str, default="test/output/s1_ai_responses.json", help="responses.json 경로")
    parser.add_argument("--embeddings", type=str, default="test/output/response_embeddings.pkl", help="response_embeddings.pkl 경로")
    parser.add_argument("--output-dir", type=str, default="test/output", help="출력 디렉토리")
    parser.add_argument("--hard-threshold", type=float, default=None, help="hard 엣지 임계값 (기본: Q3)")
    parser.add_argument("--pending-low", type=float, default=None, help="pending 하한 (기본: Q3 - 1.5*IQR)")
    parser.add_argument("--pending-high", type=float, default=None, help="pending 상한 (기본: hard-threshold와 동일)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "l1"], help="거리 메트릭 (cosine, l2, l1)")
    parser.add_argument("--no-normalize", action="store_true", help="cosine 계산 전에 L2 정규화 비활성화")
    parser.add_argument("--categories", type=str, default=None, help="선택: 대분류 카테고리 JSON (conversation_id -> category)")
    parser.add_argument("--plot", action="store_true", help="similarity 분포를 PNG로 시각화 저장")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("로딩 중...")
    t_load0 = time.time()
    responses = load_responses(args.responses)
    payload = load_embeddings_pkl(args.embeddings)
    X = align_embeddings(responses, payload)
    t_load = time.time() - t_load0

    print("대화별 길이 가중 풀링...")
    t_pool0 = time.time()
    conv_vectors = pool_by_conversation_length(responses, X)
    t_pool = time.time() - t_pool0

    print("대화간 유사도 및 엣지 생성...")
    t_edges0 = time.time()
    cats_full = load_categories(args.categories) if args.categories else None
    # compute_conv_edges_from_pooled를 위해 conversation_id -> category 매핑으로 변환
    cats_for_edges = {cid: info.get("category") for cid, info in cats_full.items()} if cats_full else None
    
    # threshold 기본값 계산 (None인 경우만)
    hard_threshold = args.hard_threshold
    pending_low = args.pending_low
    pending_high = args.pending_high
    
    edges, stats, sims_all = compute_conv_edges_from_pooled(
        conv_vectors,
        hard_threshold=hard_threshold if hard_threshold is not None else 0.7,  # 임시값
        pending_threshold_low=pending_low if pending_low is not None else 0.5,  # 임시값
        pending_threshold_high=pending_high if pending_high is not None else 0.7,  # 임시값
        normalize=not args.no_normalize,
        metric=args.metric,
        categories=cats_for_edges,
    )
    
    # sims_all에서 Median과 Q3 계산
    if sims_all:
        arr = np.asarray(sims_all, dtype=float)
        median = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
        
        # threshold 기본값 설정
        if hard_threshold is None:
            hard_threshold = q3
        if pending_low is None:
            pending_low = (median + q3) / 2.0
        if pending_high is None:
            pending_high = hard_threshold
        
        print(f"Threshold 자동 계산: Median={median:.4f}, Q3={q3:.4f}")
        print(f"  hard_threshold={hard_threshold:.4f}, pending_low={pending_low:.4f}, pending_high={pending_high:.4f}")
        
        # threshold가 변경되었으면 다시 계산
        if args.hard_threshold is None or args.pending_low is None or args.pending_high is None:
            edges, stats, sims_all = compute_conv_edges_from_pooled(
                conv_vectors,
                hard_threshold=hard_threshold,
                pending_threshold_low=pending_low,
                pending_threshold_high=pending_high,
                normalize=not args.no_normalize,
                metric=args.metric,
                categories=cats_for_edges,
            )
    
    t_edges = time.time() - t_edges0

    export_edges(edges, out_dir, metric=args.metric)
    
    # 통계 정보 추가
    t_total = time.time() - t_start
    stats.update({
        "load_seconds": round(t_load, 2),
        "pool_seconds": round(t_pool, 2),
        "edges_seconds": round(t_edges, 2),
        "total_seconds": round(t_total, 2),
        "metric": args.metric,
    })
    export_stats(stats, out_dir, metric=args.metric)

    # 분포 시각화 (최종 threshold 값 사용)
    if args.plot:
        plot_similarity_distribution(
            sims_all,
            out_dir,
            metric=args.metric,
            hard_threshold=hard_threshold,
            pending_low=pending_low,
            pending_high=pending_high,
        )

    # Graph JSON 생성 (카테고리 정보 포함)
    build_graph_json(responses, edges, cats_full if cats_full else {}, out_dir, metric=args.metric)

    print(f"\n완료: edges={len(edges)} | metric={args.metric} | output_dir={out_dir}")


if __name__ == "__main__":
    main()
