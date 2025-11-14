# Experiments `src/` Guide

This folder contains the individual pipeline steps plus shared utilities that build the conversation graph.

| File | Purpose | Typical Command |
| ---- | ------- | ---------------- |
| `extract_features.py` | Cleans chat history, generates embeddings + keywords, and writes `features.json`. | `python extract_features.py --in ../../input_data/mock_data.json --out ../output/features.json --cfg ../config.yaml` |
| `cluster_with_llm.py` | Uses the configured LLM provider to group conversations and produce `clusters.json`. | `python cluster_with_llm.py --input ../output/features.json --output ../output/clusters.json --provider openai --model gpt-4o-mini` |
| `build_edges.py` | Computes cosine similarities, applies thresholds, optionally verifies with LLM, and saves `edges.json`. | `python build_edges.py --intermediate ../output/features.json --clusters ../output/clusters.json --output ../output/edges.json --high-threshold 0.8 --medium-threshold 0.6` |
| `merge_graph.py` | Merges features, clusters, and edges into the final `graph.json` (and optional frontend JSON). | `python merge_graph.py --features ../output/features.json --clusters ../output/clusters.json --edges ../output/edges.json --output ../output/graph.json` |
| `run_pipeline.py` | Orchestrates the four steps above in sequence and reports timing statistics. | `python run_pipeline.py --input ../../input_data/mock_data.json --config ../config.yaml --output-dir ../output` |
| `util/` | Shared helpers used by multiple steps (`io_schemas.py`, `llm_clients.py`). | Imported directly (no CLI). |

## Minimal Pipeline

From inside `graph_part/experiments/src`:

```bash
# 1) Feature extraction
python extract_features.py --in ../../input_data/mock_data.json --out ../output/features.json --cfg ../config.yaml

# 2) LLM clustering
python cluster_with_llm.py --input ../output/features.json --output ../output/clusters.json --provider openai --model gpt-4o-mini

# 3) Edge construction
python build_edges.py --intermediate ../output/features.json --clusters ../output/clusters.json --output ../output/edges.json

# 4) Merge everything
python merge_graph.py --features ../output/features.json --clusters ../output/clusters.json --edges ../output/edges.json --output ../output/graph.json
```

Or simply run:

```bash
python run_pipeline.py \
  --input ../../input_data/mock_data.json \
  --config ../config.yaml \
  --output-dir ../output \
  --provider openai \
  --model gpt-4o-mini
```

Adjust the input/config/output paths and provider/model flags as needed for your environment.
