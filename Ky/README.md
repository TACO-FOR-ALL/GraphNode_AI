# Ky - Chat Conversation Graph Builder

대화 히스토리를 분석하여 토픽 기반 지식 그래프를 구축하는 파이프라인입니다. 임베딩, 키워드 추출, LLM 기반 클러스터링, 유사도 기반 엣지 생성을 통해 대화의 구조와 관계를 시각화합니다.

## 📋 목차

- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [파이프라인 구조](#-파이프라인-구조)
- [개별 모듈 사용법](#-개별-모듈-사용법)
- [설정](#-설정)
- [출력 형식](#-출력-형식)
- [LLM 프로바이더 설정](#-llm-프로바이더-설정)

## 🚀 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## ⚡ 빠른 시작

### 전체 파이프라인 한 번에 실행

```bash
python src/run_pipeline.py \
  --input input_data/mock_data.json \
  --config config.yaml \
  --output-dir output \
  --provider openai \
  --model gpt-4o-mini
```

### 단계별 실행 (디버깅용)

```bash
# 1. 특징 추출 (임베딩 + 키워드)
python src/extract_features.py \
  --in input_data/mock_data.json \
  --out output/features.json \
  --cfg config.yaml

# 2. LLM 기반 클러스터링
python src/cluster_with_llm.py \
  --input output/features.json \
  --output output/clusters.json \
  --provider openai \
  --model gpt-4o-mini

# 3. 엣지 생성
python src/build_edges.py \
  --intermediate output/features.json \
  --clusters output/clusters.json \
  --output output/edges.json

# 4. 최종 그래프 병합
python src/merge_graph.py \
  --features output/features.json \
  --clusters output/clusters.json \
  --edges output/edges.json \
  --output output/graph.json
```

## 📊 파이프라인 구조

```
입력 (대화 히스토리)
    ↓
┌─────────────────────────────────────┐
│ 1. extract_features.py              │
│   - 텍스트 전처리 및 정규화          │
│   - 임베딩 생성 (Sentence-Transformers)│
│   - 키워드 추출 (KeyBERT)            │
│   → features.json                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. cluster_with_llm.py              │
│   - LLM 기반 토픽 클러스터 생성      │
│   - 대화-클러스터 할당               │
│   → clusters.json                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. build_edges.py                   │
│   - 코사인 유사도 계산               │
│   - 임계값 기반 엣지 생성            │
│   - LLM 검증 (선택)                  │
│   → edges.json                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. merge_graph.py                   │
│   - 모든 결과 병합                   │
│   - 메타데이터 생성                  │
│   → graph.json                       │
└─────────────────────────────────────┘
```

## 🔧 개별 모듈 사용법

### 1. extract_features.py

대화 텍스트를 전처리하고 임베딩과 키워드를 추출합니다.

```bash
python src/extract_features.py \
  --input <대화_히스토리.json> \
  --output <특징_출력.json> \
  --cfg <설정파일.yaml>
```

**주요 기능:**

- 다국어 전처리 (URL, 코드 블록 제거)
- Sentence-Transformers 기반 임베딩 생성
- KeyBERT 키워드 추출 및 중복 제거
- 청크 기반 긴 텍스트 처리

**출력:** `features.json`

```json
{
  "conversations": [
    {
      "id": 0,
      "orig_id": "conv_123",
      "keywords": [{"term": "python", "score": 0.85}],
      "timestamp": "2024-01-01T00:00:00",
      "num_messages": 5
    }
  ],
  "embeddings": [[0.1, 0.2, ...]],
  "metadata": {
    "total_conversations": 100,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "timing": {...}
  }
}
```

### 2. cluster_with_llm.py

LLM을 사용하여 대화를 의미적 토픽 클러스터로 그룹화합니다.

```bash
python src/cluster_with_llm.py \
  --input features.json \
  --output clusters.json \
  --provider openai \
  --model gpt-4o-mini \
  --num-clusters 5
```

**옵션:**

- `--num-clusters N`: 고정된 클러스터 개수 지정
- `--min-clusters N` / `--max-clusters N`: LLM이 선택할 범위 지정
- `--provider`: `openai`, `qwen`, `groq`, `gemini`
- `--batch-size`: Phase 2 배치 크기 (기본: 50)

**출력:** `clusters.json`

```json
{
  "clusters": [
    {
      "id": "cluster_1",
      "name": "Python Programming",
      "description": "Conversations about Python coding",
      "key_themes": ["python", "coding", "debugging"],
      "size": 25
    }
  ],
  "assignments": [
    {
      "conversation_id": 0,
      "cluster_id": "cluster_1",
      "confidence": 0.92,
      "top_keywords": ["python", "flask"]
    }
  ]
}
```

### 3. build_edges.py

임베딩 간 코사인 유사도를 계산하여 그래프 엣지를 생성합니다.

```bash
python src/build_edges.py \
  --intermediate features.json \
  --clusters clusters.json \
  --output edges.json \
  --high-threshold 0.8 \
  --medium-threshold 0.6 \
  --no-llm  # LLM 검증 비활성화
```

**엣지 생성 전략:**

- **High confidence** (≥ 0.8): 자동 승인
- **Medium confidence** (0.6-0.8): LLM 검증 (선택적)
- **Low** (< 0.6): 제외

**출력:** `edges.json`

```json
{
  "edges": [
    {
      "source": 0,
      "target": 5,
      "weight": 0.85,
      "type": "semantic",
      "is_intra_cluster": true,
      "confidence": "high"
    }
  ],
  "metadata": {
    "total_edges": 120,
    "intra_cluster_edges": 80,
    "inter_cluster_edges": 40,
    "edge_density": 0.0234
  }
}
```

### 4. merge_graph.py

모든 파이프라인 출력을 최종 그래프로 병합합니다.

```bash
python src/merge_graph.py \
  --features features.json \
  --clusters clusters.json \
  --edges edges.json \
  --output graph.json \
  --frontend-output frontend_graph.json  # 선택적
```

**출력:** `graph.json` (통합 그래프 데이터)

## ⚙️ 설정

`config.yaml` 파일로 모든 파라미터를 제어합니다:

```yaml
# 임베딩 모델
embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 키워드 추출
keyword:
  top_n: 5 # 대화당 키워드 개수
  max_ngram: 3 # 최대 n-gram 길이
  dedup_thresh: 0.8 # 중복 제거 임계값 (Jaccard)

# 그래프 구성
graph:
  sim_top_k: 5 # 노드당 상위 k개 엣지
  sim_threshold: null # 또는 고정 임계값 (예: 0.7)

# 클러스터링 (HDBSCAN, 미사용)
cluster:
  min_cluster_size: 5
  min_samples: 5
  metric: euclidean

# 전처리
preprocess:
  lower: true # 소문자 변환
  strip_urls: true # URL 제거
  strip_code: true # 코드 블록 제거
  strip_punct: false # 구두점 제거 여부
  stopwords_langs: [en, zh, ko] # 불용어 언어
```

## 📤 출력 형식

### features.json

```json
{
  "conversations": [...],
  "embeddings": [[...]],
  "metadata": {
    "timing": {
      "embedding_seconds": 12.5,
      "keyword_seconds": 3.2
    }
  }
}
```

### clusters.json

```json
{
  "clusters": [
    {
      "id": "cluster_1",
      "name": "Python Development",
      "description": "...",
      "key_themes": ["python", "flask"],
      "size": 25
    }
  ],
  "assignments": [
    {
      "conversation_id": 0,
      "cluster_id": "cluster_1",
      "confidence": 0.92
    }
  ]
}
```

### edges.json

```json
{
  "edges": [{
    "source": 0,
    "target": 5,
    "weight": 0.85,
    "confidence": "high",
    "is_intra_cluster": true
  }],
  "metadata": {
    "total_edges": 120,
    "similarity_stats": {...}
  }
}
```

### graph.json (최종 출력)

```json
{
  "nodes": [{
    "id": 0,
    "orig_id": "conv_123",
    "cluster_id": "cluster_1",
    "cluster_name": "Python Development",
    "keywords": [...],
    "timestamp": "2024-01-01"
  }],
  "edges": [...],
  "metadata": {
    "total_nodes": 100,
    "total_edges": 120,
    "total_clusters": 5,
    "edge_statistics": {...},
    "timing": {...}
  }
}
```

## 🤖 LLM 프로바이더 설정

`.env` 파일에 API 키를 설정하세요:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # 기본값

# Qwen (DashScope)
DASHSCOPE_API_KEY=your_key
QWEN_MODEL=qwen3-max

# Groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

# Gemini
GEMINI_API_KEY=your_key
```

### Gemini 사용 시

```bash
pip install google-generativeai

python src/cluster_with_llm.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --input features.json
```

## 📁 프로젝트 구조

```
Ky/
├── src/
│   ├── extract_features.py      # 1단계: 특징 추출
│   ├── cluster_with_llm.py      # 2단계: LLM 클러스터링
│   ├── build_edges.py           # 3단계: 엣지 생성
│   ├── merge_graph.py           # 4단계: 그래프 병합
│   ├── run_pipeline.py          # 전체 파이프라인 실행
│   └── util/
│       ├── io_schemas.py        # Pydantic 데이터 모델
│       └── llm_clients.py       # LLM 클라이언트 (OpenAI/Qwen/Groq/Gemini)
├── config.yaml                   # 파이프라인 설정
├── requirements.txt              # Python 의존성
└── README.md
```

## 🔍 주요 특징

- **다국어 지원**: 한국어, 중국어, 영어 등 다국어 텍스트 처리
- **유연한 LLM 통합**: OpenAI, Qwen, Groq, Gemini 등 다양한 프로바이더
- **적응형 임계값**: 데이터에 따라 엣지 생성 임계값 자동 조정
- **청크 기반 처리**: 긴 대화도 효율적으로 임베딩
- **상세한 메타데이터**: 타이밍, 통계, 파라미터 추적

## 🐛 문제 해결

### 임베딩 모델 로드 실패

```bash
# 수동으로 모델 다운로드
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### LLM API 에러

- `.env` 파일에 올바른 API 키가 있는지 확인
- 네트워크 연결 확인
- `--verbose` 플래그로 상세 로그 확인

### 메모리 부족

- `--batch-size` 줄이기 (기본: 50 → 20)
- 더 작은 임베딩 모델 사용
- 입력 데이터 분할 처리

## 📊 성능 최적화

- **병렬 처리**: 임베딩 생성 시 배치 처리
- **캐싱**: 청크 분할 결과 재사용
- **적응형 임계값**: 데이터 분포에 따라 동적 조정

## 📝 라이선스

이 프로젝트는 실험용 파이프라인입니다.
