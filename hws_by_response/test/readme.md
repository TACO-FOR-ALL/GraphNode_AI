
# 키워드 추출
```bash
python test/extract_qa_keywords.py --conversation-id 375 --model intfloat/multilingual-e5-base --ngram-max 1 --max-candidates 0 --top-n 10 --output test/keywords/qa_keywords_custom_283_e5.json
```

# QA클러스터링

## 키워드 기반 
```bash
python test/cluster_keywords_hdbscan.py --conversation-id 283 --min-cluster-size 2 --min-samples 1 --metric euclidean
```

## QA 임베딩 기반 (추천)

1. hdbscan
```bash
python test/cluster_qa.py --conversation-id 283 --algo hdbscan --min-cluster-size 2 --min-samples 1
```

2. kmeans
```bash
python test/cluster_qa.py --qa-embeddings output/qa_embeddings.pkl --conversation-id 283 --algo kmeans --n-clusters-kmeans 4
```


# 시각화
```bash
python test/visualize_qa_clusters_qa_only.py --qa-clusters-json output/qa_clusters_1_hdbscan.json --qa-keywords-json test/keywords/qa_keywords_custom_1_e5.json --output-html output/qa_clusters_1_hdbscan_e5.html --min-jaccard 0.1
```