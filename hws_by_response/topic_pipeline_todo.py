import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def load_responses(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_embeddings_pkl(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_doc_index(responses: List[Dict[str, Any]]) -> Dict[Any, int]:
    return {r["response_id"]: i for i, r in enumerate(responses)}


def align_embeddings(responses: List[Dict[str, Any]], payload: Dict[str, Any]) -> np.ndarray:
    idx = build_doc_index(responses)

    # Normalize payload into a list of {response_id, embedding}
    items: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
        items = payload["items"]
    elif isinstance(payload, dict) and payload:  # possibly {rid: {embedding: [...]}} or {rid: [...]}
        for rid, val in payload.items():
            if rid == "items":
                continue
            if isinstance(val, dict) and "embedding" in val:
                items.append({"response_id": rid, "embedding": val["embedding"]})
            else:
                # assume val itself is the vector
                items.append({"response_id": rid, "embedding": val})
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    # Determine embedding dimension safely
    emb_dim = None
    for it in items:
        vec = it.get("embedding")
        if isinstance(vec, (list, tuple, np.ndarray)) and len(vec) > 0:
            emb_dim = len(vec)
            break
    if emb_dim is None:
        # fallback: zero matrix with 0 dim is invalid; instead, infer from first response content length 0
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


def run_bertopic(
    documents: List[str],
    embeddings: np.ndarray,
    min_topic_size: int = 15,
    top_n_words: int = 10,
    ngram_min: int = 1,
    ngram_max: int = 2,
    stop_words: str = 'english',
) -> Tuple[BERTopic, List[int]]:
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(ngram_min, ngram_max))
    model = BERTopic(
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
        vectorizer_model=vectorizer,
    )
    topics, _ = model.fit_transform(documents, embeddings)
    try:
        outliers = int(sum(1 for t in topics if t == -1))
        ratio = (outliers / max(1, len(topics))) if topics is not None else 0.0
        print(f"Outliers: {outliers}/{len(topics)} ({ratio:.2%})")
    except Exception:
        pass
    return model, topics


def pool_by_conversation(responses: List[Dict[str, Any]], topics: List[int], embeddings: np.ndarray) -> Dict[Any, List[Dict[str, Any]]]:
    conv_map: Dict[Any, Dict[int, List[int]]] = {}
    for i, r in enumerate(responses):
        cid = r.get("conversation_id")
        tid = topics[i]
        if tid == -1:
            continue
        conv_map.setdefault(cid, {}).setdefault(tid, []).append(i)
    conv_topic_vectors: Dict[Any, List[Dict[str, Any]]] = {}
    for cid, topic_idx in conv_map.items():
        items = []
        for tid, idxs in topic_idx.items():
            vec = embeddings[idxs].mean(axis=0)
            items.append({"topic_id": int(tid), "vector": vec, "response_ids": [responses[j]["response_id"] for j in idxs]})
        conv_topic_vectors[cid] = items
    try:
        sizes = [len(v) for v in conv_topic_vectors.values()]
        if sizes:
            import numpy as _np
            print(f"Conv topic vectors: docs={len(sizes)} | avg={float(_np.mean(sizes)):.2f} | min={int(min(sizes))} | max={int(max(sizes))}")
        else:
            print("Conv topic vectors: none")
    except Exception:
        pass
    return conv_topic_vectors


def _l2_normalize_rows(M: np.ndarray) -> np.ndarray:
    if M.size == 0:
        return M
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def compute_conversation_similarity(
    conv_vectors: Dict[Any, List[Dict[str, Any]]],
    hard_threshold: float = 0.8,
    pending_threshold: float = 0.5,
    agg: str = "bestmean",
    categories: Optional[Dict[Any, Any]] = None,
    cross_multiplier: float = 1.0,
    normalize_before: bool = False,
) -> List[Dict[str, Any]]:
    cids = list(conv_vectors.keys())
    edges = []
    sims_all: List[float] = []
    top_k: List[Tuple[float, Any, Any]] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a = cids[i]
            b = cids[j]
            va = conv_vectors.get(a, [])
            vb = conv_vectors.get(b, [])
            if not va or not vb:
                continue
            A = np.stack([x["vector"] for x in va])
            B = np.stack([x["vector"] for x in vb])
            if normalize_before:
                A = _l2_normalize_rows(A)
                B = _l2_normalize_rows(B)
            sims = cosine_similarity(A, B)
            if agg == "mean":
                score = float(sims.mean())
            else:  # bestmean (symmetric best-match mean)
                row_best = sims.max(axis=1).mean()
                col_best = sims.max(axis=0).mean()
                score = float((row_best + col_best) / 2.0)

            # apply cross-category multiplier if categories differ
            if categories is not None:
                ca = categories.get(a)
                cb = categories.get(b)
                if ca is not None and cb is not None and ca != cb:
                    score *= float(cross_multiplier)
            sims_all.append(score)
            status = "hard" if score >= hard_threshold else ("pending" if score >= pending_threshold else None)
            if status:
                edges.append({"source": a, "target": b, "similarity": score, "status": status})
            try:
                if len(top_k) < 10:
                    top_k.append((score, a, b))
                    top_k.sort(reverse=True)
                elif score > top_k[-1][0]:
                    top_k[-1] = (score, a, b)
                    top_k.sort(reverse=True)
            except Exception:
                pass
    try:
        if sims_all:
            import numpy as _np
            arr = _np.asarray(sims_all, dtype=float)
            print(f"Similarity dist: n={arr.size} | mean={float(arr.mean()):.4f} | median={float(_np.median(arr)):.4f} | std={float(arr.std()):.4f} | max={float(arr.max()):.4f} | min={float(arr.min()):.4f}")
            if top_k:
                print("Top edges (score, source, target):")
                for s, a, b in top_k:
                    print(f"  {s:.4f}\t{a}\t{b}")
        else:
            print("Similarity dist: no comparable pairs")
    except Exception:
        pass
    return edges


def export_topic_info(model: BERTopic, output_dir: Path):
    topics_info = model.get_topic_info().to_dict(orient="records")
    topic_words = {}
    for t in topics_info:
        tid = int(t["Topic"]) if isinstance(t["Topic"], (int, np.integer)) else t["Topic"]
        if tid == -1:
            continue
        words = model.get_topic(tid)
        topic_words[tid] = [{"word": w, "score": float(s)} for w, s in words]
    with open(output_dir / "s3_topics.json", "w", encoding="utf-8") as f:
        json.dump({"topics": topic_words, "info": topics_info}, f, ensure_ascii=False, indent=2)


def export_doc_topics(responses: List[Dict[str, Any]], topics: List[int], output_dir: Path):
    items = []
    for r, t in zip(responses, topics):
        items.append({"response_id": r["response_id"], "conversation_id": r.get("conversation_id"), "topic_id": int(t)})
    with open(output_dir / "s3_doc_topics.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def export_conv_vectors(conv_vectors: Dict[Any, List[Dict[str, Any]]], output_dir: Path):
    out = {}
    for cid, items in conv_vectors.items():
        out[cid] = [{"topic_id": x["topic_id"], "vector": np.asarray(x["vector"]).tolist(), "response_ids": x["response_ids"]} for x in items]
    with open(output_dir / "s4_conversation_topic_vectors.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def export_edges(edges: List[Dict[str, Any]], output_dir: Path):
    hard = [e for e in edges if e["status"] == "hard"]
    pending = [e for e in edges if e["status"] == "pending"]
    with open(output_dir / "s5_edges_hard.json", "w", encoding="utf-8") as f:
        json.dump(hard, f, ensure_ascii=False, indent=2)
    with open(output_dir / "s5_edges_pending.json", "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


def export_graph(responses: List[Dict[str, Any]], edges: List[Dict[str, Any]], categories: Dict[Any, str], output_dir: Path):
    nodes = []
    for r in responses:
        nodes.append({
            "id": r["conversation_id"],
            "title": r.get("conversation_title"),
            "category": categories.get(r["conversation_id"]) if categories else None,
        })
    with open(output_dir / "s6_graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)


def export_insight_edges(edges: List[Dict[str, Any]], output_dir: Path):
    with open(output_dir / "s5_edges_insight.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)


def load_doc_topics(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_excluded_topics_from_info(topics_info_path: Path, top_k: int) -> set:
    if top_k <= 0 or not topics_info_path.exists():
        return set()
    with open(topics_info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    info = data.get("info", [])
    items = []
    for rec in info:
        tid = rec.get("Topic")
        if isinstance(tid, (int, np.integer)):
            tid = int(tid)
        if tid == -1:
            continue
        cnt = rec.get("Count")
        if isinstance(cnt, (int, np.integer)):
            cnt = int(cnt)
        else:
            try:
                cnt = int(cnt)
            except Exception:
                cnt = 0
        items.append((tid, cnt))
    items.sort(key=lambda x: x[1], reverse=True)
    return set([tid for tid, _ in items[:top_k]])


def build_conv_topic_buckets(doc_topics: List[Dict[str, Any]]):
    by_conv_topic_counts: Dict[Any, Dict[int, int]] = {}
    by_conv_topic_ids: Dict[Any, Dict[int, List[Any]]] = {}
    totals: Dict[Any, int] = {}
    for it in doc_topics:
        cid = it.get("conversation_id")
        tid = it.get("topic_id")
        rid = it.get("response_id")
        if tid == -1:
            continue
        by_conv_topic_counts.setdefault(cid, {}).setdefault(int(tid), 0)
        by_conv_topic_counts[cid][int(tid)] += 1
        by_conv_topic_ids.setdefault(cid, {}).setdefault(int(tid), []).append(rid)
        totals[cid] = totals.get(cid, 0) + 1
    return by_conv_topic_counts, by_conv_topic_ids, totals


def refine_pending_to_insight(
    edges: List[Dict[str, Any]],
    by_counts: Dict[Any, Dict[int, int]],
    by_ids: Dict[Any, Dict[int, List[Any]]],
    totals: Dict[Any, int],
    exclude_topics: set,
    min_overlap: float,
    drop_overlap: float,
    min_concentration: float,
    min_max_shared: int,
    examples_per_side: int = 3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in edges:
        if e.get("status") != "pending":
            continue
        a = e.get("source")
        b = e.get("target")
        ta = by_counts.get(a, {})
        tb = by_counts.get(b, {})
        if not ta or not tb:
            continue
        shared = [t for t in set(ta.keys()) & set(tb.keys()) if t not in exclude_topics]
        if not shared:
            continue
        matches = [min(ta[t], tb[t]) for t in shared]
        sum_match = sum(matches)
        min_total = max(1, min(totals.get(a, 0), totals.get(b, 0)))
        overlap = sum_match / min_total
        concentration = (max(matches) / sum_match) if sum_match > 0 else 0.0
        max_shared = max(matches) if matches else 0
        if overlap >= min_overlap and (concentration >= min_concentration or max_shared >= min_max_shared):
            max_t = shared[matches.index(max_shared)] if shared else None
            ex_a = by_ids.get(a, {}).get(max_t, [])[:examples_per_side] if max_t is not None else []
            ex_b = by_ids.get(b, {}).get(max_t, [])[:examples_per_side] if max_t is not None else []
            out.append({
                "source": a,
                "target": b,
                "similarity": e.get("similarity"),
                "status": "insight",
                "overlap": round(float(overlap), 6),
                "concentration": round(float(concentration), 6),
                "max_shared_count": int(max_shared),
                "shared_topics": [int(t) for t in shared],
                "max_shared_topic": int(max_t) if max_t is not None else None,
                "examples": {"a": ex_a, "b": ex_b},
            })
        elif overlap < drop_overlap:
            pass
        else:
            continue
    return out


def load_categories(path: str) -> Dict[Any, str]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Case 1: full result file with 'assignments' object
    if isinstance(data, dict) and 'assignments' in data and isinstance(data['assignments'], dict):
        assign = data['assignments']
        out: Dict[Any, str] = {}
        for k, v in assign.items():
            cid = int(k) if isinstance(k, str) and k.isdigit() else k
            if isinstance(v, dict):
                cat = v.get('category')
            else:
                cat = v
            out[cid] = cat
        return out
    # Case 2: direct mapping {conversation_id: category}
    if isinstance(data, dict):
        return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.items()}
    # Fallback: unsupported format
    return {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Topic pipeline: BERTopic clustering, pooling, edges, optional categories, and graph export")
    parser.add_argument("--responses", type=str, required=True, help="responses.json 경로")
    parser.add_argument("--embeddings", type=str, required=True, help="임베딩 PKL 경로")
    parser.add_argument("--output-dir", type=str, default="test/output", help="출력 디렉토리")
    parser.add_argument("--min-topic-size", type=int, default=15)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--ngram-min", type=int, default=1, help="CountVectorizer ngram 최소")
    parser.add_argument("--ngram-max", type=int, default=2, help="CountVectorizer ngram 최대")
    parser.add_argument("--stop-words", type=str, default="english", help="불용어 설정 (None, english 등)")
    parser.add_argument("--categories", type=str, default=None, help="선택: 대분류 카테고리 JSON (conversation_id -> category)")
    parser.add_argument("--hard-threshold", type=float, default=0.8, help="하드 엣지 임계값 (코사인 유사도)")
    parser.add_argument("--pending-threshold", type=float, default=0.5, help="펜딩 엣지 임계값 (코사인 유사도)")
    parser.add_argument("--sim-agg", type=str, default="bestmean", choices=["mean", "bestmean"], help="대화 유사도 집계 방식")
    parser.add_argument("--normalize-pooling", action="store_true", help="풀링 전후 임베딩 L2 정규화 사용")
    parser.add_argument("--cross-multiplier", type=float, default=1.0, help="서로 다른 카테고리 간 유사도 곱셈 계수(<1로 두면 교차 엣지 감소)")
    parser.add_argument("--insight-min-overlap", type=float, default=0.4, help="insight 엣지 승격용 최소 overlap")
    parser.add_argument("--insight-drop-overlap", type=float, default=0.2, help="insight 판단에서 overlap이 이 값보다 낮으면 제외")
    parser.add_argument("--insight-min-concentration", type=float, default=0.6, help="insight 엣지 승격용 최소 concentration")
    parser.add_argument("--insight-min-max-shared", type=int, default=5, help="insight 엣지 승격용 최소 max_shared_count")
    parser.add_argument("--insight-exclude-top-k-topics", type=int, default=0, help="전역 상위 빈도 토픽 K개 제외(0이면 비활성)")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("로딩 중...")
    t_load0 = time.time()
    responses = load_responses(args.responses)
    emb_payload = load_embeddings_pkl(args.embeddings)
    X = align_embeddings(responses, emb_payload)
    docs = [r.get("content", "") for r in responses]
    t_load = time.time() - t_load0

    print("BERTopic 클러스터링...")
    t0 = time.time()
    model, topics = run_bertopic(
        docs,
        X,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        stop_words=(None if args.stop_words.lower() == 'none' else args.stop_words),
    )
    t_bertopic = time.time() - t0
    print(f"BERTopic 완료: {t_bertopic:.2f}초")

    print("토픽 정보/할당 저장...")
    t_topics_export0 = time.time()
    export_topic_info(model, out_dir)
    export_doc_topics(responses, topics, out_dir)
    t_topics_export = time.time() - t_topics_export0

    print("대화별 토픽 벡터 풀링...")
    t_pool0 = time.time()
    conv_vectors = pool_by_conversation(responses, topics, X)
    export_conv_vectors(conv_vectors, out_dir)
    t_pool = time.time() - t_pool0

    print("대화간 유사도 및 엣지 생성...")
    t_edges0 = time.time()
    # Load categories early for category-aware scoring
    cats = load_categories(args.categories)

    edges = compute_conversation_similarity(
        conv_vectors,
        hard_threshold=args.hard_threshold,
        pending_threshold=args.pending_threshold,
        agg=args.sim_agg,
        categories=cats if cats else None,
        cross_multiplier=args.cross_multiplier,
        normalize_before=bool(args.normalize_pooling),
    )
    export_edges(edges, out_dir)
    t_edges = time.time() - t_edges0

    print("그래프 저장...")
    t_graph0 = time.time()
    export_graph(responses, edges, cats, out_dir)
    t_graph = time.time() - t_graph0

    print("insight 엣지 생성...")
    t_insight0 = time.time()
    doc_topics_path = out_dir / "s3_doc_topics.json"
    topics_info_path = out_dir / "s3_topics.json"
    try:
        doc_topics = load_doc_topics(doc_topics_path)
        by_counts, by_ids, totals = build_conv_topic_buckets(doc_topics)
        excluded = compute_excluded_topics_from_info(topics_info_path, int(args.insight_exclude_top_k_topics))
        insight_edges = refine_pending_to_insight(
            edges,
            by_counts,
            by_ids,
            totals,
            excluded,
            float(args.insight_min_overlap),
            float(args.insight_drop_overlap),
            float(args.insight_min_concentration),
            int(args.insight_min_max_shared),
        )
        export_insight_edges(insight_edges, out_dir)
        t_insight = time.time() - t_insight0
    except Exception:
        insight_edges = []
        t_insight = time.time() - t_insight0

    total_edges = len(edges)
    within = 0
    cross = 0
    hard_within = 0
    hard_cross = 0
    pending_within = 0
    pending_cross = 0
    if cats:
        cat_map = cats
        for e in edges:
            a = e["source"]
            b = e["target"]
            if cat_map.get(a) and cat_map.get(b):
                if cat_map[a] == cat_map[b]:
                    within += 1
                    if e.get("status") == "hard":
                        hard_within += 1
                    elif e.get("status") == "pending":
                        pending_within += 1
                else:
                    cross += 1
                    if e.get("status") == "hard":
                        hard_cross += 1
                    elif e.get("status") == "pending":
                        pending_cross += 1

    t_total = time.time() - t_start
    stats = {
        "bertopic_seconds": round(t_bertopic, 2),
        "load_seconds": round(t_load, 2),
        "topics_export_seconds": round(t_topics_export, 2),
        "pool_seconds": round(t_pool, 2),
        "edges_seconds": round(t_edges, 2),
        "graph_seconds": round(t_graph, 2),
        "insight_seconds": round(t_insight, 2),
        "total_seconds": round(t_total, 2),
        "total_edges": total_edges,
        "within_category_edges": within,
        "cross_category_edges": cross,
        "hard_within_category_edges": hard_within,
        "hard_cross_category_edges": hard_cross,
        "pending_within_category_edges": pending_within,
        "pending_cross_category_edges": pending_cross,
        "insight_edges": len(insight_edges) if 'insight_edges' in locals() else 0,
    }
    with open(out_dir / "s6_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("완료")


if __name__ == "__main__":
    main()
