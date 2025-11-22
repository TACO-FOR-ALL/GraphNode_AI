
# 전처리 및 QA쌍 분리
```bash
python build_qa_pairs.py --data-path data/conversations.json --output output/qa_pairs.json --sample-size 400
```

# 키워드 추출
```bash
python tools/extract_qa_keywords.py --conversation-id 49 --model intfloat/multilingual-e5-base --ngram-max 1 --max-candidates 0 --top-n 10 --output output/keywords/qa_keywords_custom_19_e5.json
```


# QA클러스터링

## QA 임베딩 기반

1. hdbscan
```bash
python tools/cluster_qa.py --conversation-id 49 --algo hdbscan --min-cluster-size 2 --min-samples 1 
```

2. kmeans
```bash
python tools/cluster_qa.py --qa-embeddings output/embeddings/qa_embeddings.pkl --conversation-id 49 --algo kmeans --n-clusters-kmeans 4
```

# 위 과정 한번에
```bash
python tools/run_conv_pipeline.py --qa-pairs output/qa_pairs.json --model intfloat/multilingual-e5-base --ngram-max 1 --max-candidates 0 --top-n 10 --min-cluster-size 2 --min-samples 1 --metric cosine --conversation-id 283

```

배치(전체 대화)
```bash
python tools/extract_qa_embedding_keywords.py --qa-pairs output/qa_pairs.json --model intfloat/multilingual-e5-base --ngram-max 1 --max-candidates 0 --top-n 10 --min-cluster-size 2 --min-samples 1 --metric euclidean
```
또는

```bash
python extract_qa_embedding_keywords.py --qa-pairs output/qa_pairs.json --model intfloat/multilingual-e5-base --ngram-max 1 --max-candidates 0 --top-n 10 --min-cluster-size 2 --min-samples 1 --metric euclidean
```

일부 대화만

```bash
python tools/extract_qa_embedding_keywords.py --qa-pairs output/qa_pairs.json --include 3,49
```
---

## 단일 대화 전처리
```bash
python tools/preview_preprocess_conversation.py --data-path data/conversations.json --conversation-id 49 --output test/preprocess_49.json
```

# 시각화
```bash
python viewer/visualize_qa_clusters.py --conversation-id 283 --min-jaccard 0.1 --metric cosine
```


---


# 대분류

```bash
python llm_categorize_batch.py --keywords-input output/s2_keywords_pipeline_embedreuse_tfidf.json
```

# 엣지생성

## 일반 유사도 기반 생성
```bash
# 카테고리 + 키워드 포함 (추천)
python tools/build_edges_from_conv_embeddings.py --embeddings output/qa_embeddings.pkl --output-dir output --categories output/s6_categories_assignments.json --keywords output/s2_keywords_pipeline_embedreuse_tfidf.json --plot --target-hard-cross-edges 50 --metric cosine

# 카테고리만
python tools/build_edges_from_conv_embeddings.py --embeddings output/conversation_embeddings.pkl --output-dir output --categories output/s6_categories_assignments.json --plot
```
