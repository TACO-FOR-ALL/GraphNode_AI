# Graph Visualization Viewer

## 사용법

1. **그래프 생성**:
```bash
cd hws_by_response
python tools/build_edges_from_conv_embeddings.py \
  --embeddings output/conversation_embeddings.pkl \
  --output-dir output \
  --categories output/s6_categories_assignments.json \
  --keywords output/s2_keywords_pipeline_embedreuse_tfidf.json \
  --plot
```

2. **Viewer 열기**:
- `viewer/visualize_cluster_bubbles.html`을 브라우저에서 열기
- 자동으로 `../output/graph.json` 로드됨
- 또는 "Choose JSON File" 버튼으로 다른 파일 선택 가능

## 기능

- **클러스터 버블 시각화**: 각 클러스터를 버블로 표시
- **노드 표시**: 각 대화를 클러스터 내 작은 원으로 표시
- **엣지 필터링**:
  - All: 모든 엣지
  - Hard: High confidence 엣지만
  - Intra: 클러스터 내부 엣지만
  - Inter: 클러스터 간 엣지만
- **줌/팬**: 마우스 휠로 줌, 드래그로 이동
- **툴팁**: 노드에 마우스 오버시 키워드 표시

## 지원 형식

`graph.json` 형식:
```json
{
  "nodes": [
    {
      "id": 1,
      "cluster_id": "cluster_1",
      "cluster_name": "...",
      "cluster_confidence": 0.85,
      "keywords": [{"term": "...", "score": 0.8}],
      "top_keywords": ["...", "..."],
      "num_messages": 10
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "weight": 0.9,
      "type": "semantic",
      "is_intra_cluster": true,
      "confidence": "high"
    }
  ],
  "metadata": {
    "total_nodes": 374,
    "total_edges": 323,
    "clusters": {
      "total_clusters": 5,
      "clusters": [
        {
          "id": "cluster_1",
          "name": "...",
          "description": "...",
          "size": 100
        }
      ]
    }
  }
}
```
