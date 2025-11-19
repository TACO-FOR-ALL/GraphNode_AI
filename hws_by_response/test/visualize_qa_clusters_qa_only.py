import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Set

from pyvis.network import Network


def load_qa_clusters(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_qa_keywords(path: Path) -> Dict[str, Dict[str, Any]]:
    """qa_keywords_custom_*.json 을 qa_id 기준 index로 변환."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[str, Dict[str, Any]] = {}
    for rec in data:
        qa_id = str(rec.get("qa_id"))
        index[qa_id] = rec
    return index


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def build_network(
    qa_clusters_path: Path,
    qa_keywords_path: Path,
    output_html: Path,
    max_keywords_per_qa: int = 5,
    min_jaccard: float = 0.1,
) -> None:
    clusters = load_qa_clusters(qa_clusters_path)
    qa_kw_index = load_qa_keywords(qa_keywords_path)

    net = Network(height="800px", width="100%", notebook=False, directed=False)
    # 고정 레이아웃을 쓰기 위해 physics 끄기 (좌표를 직접 지정)
    net.toggle_physics(False)

    # 색상 팔레트 (클러스터별 구분용)
    cluster_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    clusters_list: List[Dict[str, Any]] = clusters.get("clusters", [])

    # QA 노드 정보 수집: qa_id -> {cluster_id, keywords(set), tooltip}
    qa_info: Dict[str, Dict[str, Any]] = {}
    cluster_to_qas: Dict[int, List[str]] = {}

    for c in clusters_list:
        cluster_id = int(c.get("cluster_id", -1))
        qa_ids = [str(q) for q in c.get("qa_ids", [])]
        color = cluster_colors[cluster_id % len(cluster_colors)] if cluster_id >= 0 else "#cccccc"

        cluster_to_qas.setdefault(cluster_id, [])

        for qa_id in qa_ids:
            rec = qa_kw_index.get(qa_id)
            keywords_list: List[str] = []
            if rec is not None:
                kws = rec.get("keywords", [])[:max_keywords_per_qa]
                keywords_list = [str(k.get("keyword")) for k in kws]

            kw_set: Set[str] = set(keywords_list)
            # HTML 툴팁: 줄바꿈에 <br> 사용
            tooltip = "QA {}<br>keywords: {}".format(qa_id, ", ".join(keywords_list))

            qa_info[qa_id] = {
                "cluster_id": cluster_id,
                "color": color,
                "keywords": kw_set,
                "tooltip": tooltip,
            }
            cluster_to_qas[cluster_id].append(qa_id)

    # 클러스터별로 큰 원 위에 배치, 각 클러스터 안 QA들은 작은 원 위에 배치
    import math

    unique_clusters = sorted(cluster_to_qas.keys())
    num_clusters = len(unique_clusters)
    big_radius = 500  # 전체 클러스터 배치 반경

    for idx, cluster_id in enumerate(unique_clusters):
        qa_ids = cluster_to_qas[cluster_id]
        n_qas = len(qa_ids)
        if n_qas == 0:
            continue

        angle = 2 * math.pi * idx / max(1, num_clusters)
        cx = big_radius * math.cos(angle)
        cy = big_radius * math.sin(angle)

        small_radius = 100 + 10 * n_qas  # 클러스터 내 반경 (QA 개수에 따라 조금 조정)

        for j, qa_id in enumerate(qa_ids):
            theta = 2 * math.pi * j / n_qas if n_qas > 1 else 0
            x = cx + small_radius * math.cos(theta)
            y = cy + small_radius * math.sin(theta)

            info = qa_info[qa_id]
            # 라벨: QA ID + 상위 키워드 max_keywords_per_qa개까지 한 줄씩 표시
            top_kws = list(info["keywords"])[:max_keywords_per_qa]
            if top_kws:
                label = qa_id + "\n" + "\n".join(top_kws)
            else:
                label = qa_id
            net.add_node(
                qa_id,
                label=label,
                color=info["color"],
                title=info["tooltip"],
                shape="dot",
                x=x,
                y=y,
                physics=False,
            )

    # 엣지 추가: QA 간 Jaccard 유사도 기반 (같은/다른 클러스터 상관없이 키워드 겹치면 연결)
    qa_ids_list = list(qa_info.keys())
    n = len(qa_ids_list)
    for i in range(n):
        qa_i = qa_ids_list[i]
        info_i = qa_info[qa_i]
        kws_i = info_i["keywords"]
        for j in range(i + 1, n):
            qa_j = qa_ids_list[j]
            info_j = qa_info[qa_j]
            # 같은 클러스터일 때 좀 더 강하게 붙을 가능성이 높음
            sim = jaccard(kws_i, info_j["keywords"])
            if sim < min_jaccard:
                continue
            # 엣지 두께/투명도에 유사도 반영
            width = 1 + 4 * sim  # 1~5 정도
            net.add_edge(
                qa_i,
                qa_j,
                value=sim,
                width=width,
                title=f"Jaccard={sim:.2f}",
                color="#999999",
            )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_html), notebook=False, open_browser=False)
    print(f"QA-only 시각화 HTML 저장: {output_html}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize QA clusters as QA-only graph (nodes=QAs, edges=keyword similarity)",
    )
    parser.add_argument(
        "--qa-clusters-json",
        type=str,
        required=True,
        help="qa_clusters_{conv}_{algo}.json 경로 (예: output/qa_clusters_283_kmeans.json)",
    )
    parser.add_argument(
        "--qa-keywords-json",
        type=str,
        required=True,
        help="qa_keywords_custom_{conv}.json 경로 (예: test/qa_keywords_custom_283.json)",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="output/qa_clusters_qa_only_graph.html",
        help="pyvis HTML 출력 경로 (기본: output/qa_clusters_qa_only_graph.html)",
    )
    parser.add_argument(
        "--max-keywords-per-qa",
        type=int,
        default=5,
        help="각 QA당 유사도 계산/툴팁에 사용할 키워드 개수 (기본: 5)",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.1,
        help="엣지 생성 최소 Jaccard 유사도 (기본: 0.1)",
    )

    args = parser.parse_args()

    build_network(
        qa_clusters_path=Path(args.qa_clusters_json),
        qa_keywords_path=Path(args.qa_keywords_json),
        output_html=Path(args.output_html),
        max_keywords_per_qa=args.max_keywords_per_qa,
        min_jaccard=args.min_jaccard,
    )


if __name__ == "__main__":
    main()
