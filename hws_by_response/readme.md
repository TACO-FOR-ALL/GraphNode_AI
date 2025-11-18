archive파일은 코드작성하고 갈아엎으면서 생성해왔던 파일들입니다.

models_cache는 모델을 저장하는 폴더입니다.


중간 생성물은 전부 output/에 생성될 예정이며, analyze package를 기반으, tools/*.py들과 메인 directory의 *.py들로 workflow가 진행됩니다.


# 응답만 분리 (기존 방식)

```bash
python build_responses.py --data-path data/conversations.json --output output/s1_ai_responses.json --sample-size 400
```

# Q-A 쌍 추출 (새 방식 - 추천) 

```bash
python build_qa_pairs.py --data-path data/conversations.json --output output/qa_pairs.json --sample-size 400
```

# Q-A 쌍 embedding 추출

**방법 1: 기본 방식 (Q 전체 + A 앞부분, 512 토큰 맞춤)**
```bash
python tools/extract_embeddings_qa_pairs.py --input output/qa_pairs.json --out-pkl output/qa_embeddings.pkl --model intfloat/multilingual-e5-base --normalize --save-every 500
```

**방법 2: Instruction Prefix + Weighted Embedding (Question 비중 강화, 추천)** 
```bash
# Question 70% 비중 (기본)
python tools/extract_embeddings_qa_pairs.py --input output/qa_pairs.json --out-pkl output/qa_embeddings.pkl --model intfloat/multilingual-e5-base --normalize --save-every 500 --use-instruction-prefix --question-weight 0.7

# Question 80% 비중
python tools/extract_embeddings_qa_pairs.py --input output/qa_pairs.json --out-pkl output/qa_embeddings.pkl --model intfloat/multilingual-e5-base --normalize --save-every 500 --use-instruction-prefix --question-weight 0.8
```

# Conversation-level pooling

```bash
python tools/pool_qa_to_conversation.py --input output/qa_embeddings.pkl --output output/conversation_embeddings.pkl --pooling length_weighted
```

# extract embeddings (기존 방식)

```bash
python tools/extract_embeddings_responses.py --input output/s1_ai_responses.json --out-pkl output/response_embeddings.pkl --normalize --long-strategy chunk-mean --batch-size 64 --model intfloat/multilingual-e5-base --save-every 500
```

## question 키워드만 추출

  # 1. Question만 embedding 추출
```bash  
python tools/extract_embeddings_qa_pairs.py --input output/qa_pairs.json --out-pkl output/qa_embeddings.pkl    --extract-mode question
```
  # → 출력: output/qa_embeddings_question.pkl

  # 2. Conversation-level로 pooling
  python pool_qa_to_conversation.py \
    --input output/qa_embeddings_question.pkl \
    --output output/conversation_embeddings_question.pkl      

  # 3. 키워드 추출
  python extract_keywords_conv_tfidf.py \
    --input output/s1_ai_responses.json \
    --emb-pkl output/conversation_embeddings_question.pkl \   
    --emb-source conversation \
    --output output/s2_keywords_tfidf.json
  # → 출력: output/s2_keywords_tfidf_qa.json

# extract keywords

- 기존 방식 (상위 500개)

```bash
python tools/extract_keywords_conv_embedreuse.py ^
  --input output/s1_ai_responses.json ^
  --output output/s2_keywords_pipeline_embedreuse.json ^
  --emb-pkl output/response_embeddings.pkl ^
  --model intfloat/multilingual-mpnet-base-v2 ^
  --cache-dir models_cache ^
  --ngram-max 3 ^
  --max-candidates 500 ^
  --top-n 5
```

- tf-idf

```bash
python tools/extract_keywords_conv_tfidf.py --input output/s1_ai_responses.json --output output/s2_keywords_pipeline_embedreuse_tfidf.json --emb-pkl output/conversation_embeddings.pkl --model paraphrase-multilingual-mpnet-base-v2 --cache-dir models_cache --ngram-max 3 --max-candidates 100 --top-n 5
```

# 대분류 시작

```bash
python llm_categorize_batch.py --keywords-input output/s2_keywords_pipeline_embedreuse.json
```

# 엣지생성

## 그냥 유사도 기반 생성
```bash
# 카테고리 + 키워드 포함 (추천)
python tools/build_edges_from_conv_embeddings.py --embeddings output/conversation_embeddings.pkl --output-dir output --categories output/s6_categories_assignments.json --keywords output/s2_keywords_pipeline_embedreuse_tfidf.json --plot

# 카테고리만
python tools/build_edges_from_conv_embeddings.py --embeddings output/conversation_embeddings.pkl --output-dir output --categories output/s6_categories_assignments.json --plot
```

## BertTopic기반

```bash
python topic_pipeline.py ^
  --responses "output\s1_ai_responses.json" ^
  --embeddings "output\response_embeddings.pkl" ^
  --output-dir "output" ^
  --min-topic-size 2 ^
  --top-n-words 5 ^
  --ngram-min 1 ^
  --ngram-max 1 ^
  --stop-words english ^
  --hard-threshold 0.5 ^
  --pending-threshold 0.3 ^
  --sim-agg bestmean ^
  --normalize-pooling ^
  --cross-multiplier 0.7 ^
  --categories "output\s6_categories_assignments.json" ^
  --insight-min-overlap 0.4 ^
  --insight-drop-overlap 0.2 ^
  --insight-min-concentration 0.6 ^
  --insight-min-max-shared 5 ^
  --insight-exclude-top-k-topics 0
```