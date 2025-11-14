# Chat Conversation Graph Builder

Build a similarity graph from ChatGPT-style chat histories with multilingual preprocessing, keyword extraction, clustering, and graph metadata.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All experiment scripts now live under `src/` and mirror the pipeline order:

```
src/
  extract_features.py
  cluster_with_llm.py
  build_edges.py
  merge_graph.py
  run_pipeline.py        # orchestration entry point
  util/
    io_schemas.py
    llm_clients.py
```

Run individual steps from the repo root (or `graph_part/experiments/src`) using standard Python:

```bash
python graph_part/experiments/src/extract_features.py \
  --in sample_history.json \
  --out graph.json \
  --cfg graph_part/experiments/config.yaml
```

To process a different conversation, point `--in` at your own history (for example, `input_data/mock_data.json`) and reuse or customize the config.

For a one-shot run, execute the orchestrator:

```bash
python graph_part/experiments/src/run_pipeline.py \
  --input input_data/mock_data.json \
  --config graph_part/experiments/config.yaml \
  --output-dir graph_part/experiments/output
```

Each step prints a concise summary and writes its artifacts (`features.json`, `clusters.json`, `edges.json`, and final `graph.json`) into the chosen output directory.

## Configuration

`config.yaml` captures all tunable parameters:

- `embedding_model`: Sentence-Transformers model name. Defaults to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `keyword`: KeyBERT extraction knobs (`top_n`, `max_ngram`, `dedup_thresh`).
- `graph`: Similarity graph strategy (`sim_top_k` or `sim_threshold`).
- `cluster`: HDBSCAN parameters (`min_cluster_size`, `min_samples`, `metric`).
- `preprocess`: Text cleaning controls, URL/code stripping, casing, punctuation, and stopword languages.

Set values in a copy of the YAML file and pass it to `--cfg`.

## Architecture

1. **Input validation** – `src/util/io_schemas.py` provides Pydantic models to enforce the chat schema and final output contract.
2. **Preprocessing** – configurable cleaning removes URLs and fenced code, lowercases text, builds multilingual stoplists, and chunks long messages (~512 chars).
3. **Embeddings & keywords** – Sentence-Transformers (with deterministic fallback) create normalized embeddings; KeyBERT extracts deduplicated keywords per message..
4. **Clustering & summaries** – HDBSCAN identifies topical clusters; a TF-IDF pass over cluster text surfaces top descriptive terms.
5. **Graph construction** – cosine similarities define edges (top-k or threshold), producing `graph.json` with nodes, edges, and metadata (counts, params, clusters).

## LLM-Based Clustering

`src/cluster_with_llm.py` provides LLM-based conversation clustering using various providers (Qwen, Groq, Gemini).

### Setup

First, configure your API keys in `.env`:

```bash
# For Qwen (DashScope)
DASHSCOPE_API_KEY=your_dashscope_api_key
# Optional overrides:
# DASHSCOPE_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# QWEN_MODEL=qwen3-max
# ALLOW_INSECURE_CONNECTIONS=false

# For Groq
GROQ_API_KEY=your_groq_api_key
GROQ_API_URL=https://api.groq.com/openai/v1

# For Gemini (Google)
GEMINI_API_KEY=your_gemini_api_key
```

For Gemini support, install the Google Generative AI package:

```bash
pip install google-generativeai
```

### Usage

Cluster conversations using Groq:
```bash
python graph_part/experiments/src/cluster_with_llm.py \
  --provider groq \
  --input test_output.json \
  --output clustered_output.json \
  --num-clusters 4
```

Cluster conversations using Gemini:
```bash
python graph_part/experiments/src/cluster_with_llm.py \
  --provider gemini \
  --input test_output.json \
  --output clustered_output.json \
  --num-clusters 4
```

Cluster conversations using Qwen (default):
```bash
python graph_part/experiments/src/cluster_with_llm.py \
  --provider qwen \
  --input test_output.json \
  --output clustered_output.json \
  --num-clusters 4
```

### Default Models

- **Qwen**: `qwen3-max`
- **Groq**: `llama-3.3-70b-versatile`
- **Gemini**: `gemini-2.5-flash`

You can override the model with `--model`:
```bash
python graph_part/experiments/src/cluster_with_llm.py \
  --provider groq \
  --model llama3-70b-8192 \
  --input test_output.json
```

## Tests

After installation, run:

```bash
pytest
```

Tests execute the pipeline on `sample_history.json`, assert schema compliance, and validate graph density under alternate settings. They remain lightweight (<10s).
