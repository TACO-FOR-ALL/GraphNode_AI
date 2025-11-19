"""
Fast conversation-level keyword extraction reusing response embeddings (Step 3)
- For each conversation:
  1) Build conversation vector by averaging response-level embeddings from PKL
  2) Build candidate phrases from combined text via multilingual CountVectorizer
  3) Encode candidates once with the same SBERT model
  4) Rank by cosine similarity to the conversation vector
  5) Output top-N keywords per conversation

Notes:
- Assumes embeddings PKL from tools/extract_embeddings_responses.py
- No jieba here; preprocessing already applied at Step 1
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from keybert import KeyBERT  # not used directly, but kept if future extension
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tools.keyword_tokenizer import multi_lang_tokenize


def clean_minimal(text: str) -> str:
    import re
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def group_by_conversation(rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r['conversation_id']].append(r)
    return grouped


def load_response_embeddings(pkl_path: Path) -> Dict[str, Dict[str, Any]]:
    import pickle
    with pkl_path.open('rb') as f:
        data = pickle.load(f)
    return data  # dict[response_id] = {..., 'embedding': np.ndarray}


def conversation_vector_for(conv_rows: List[Dict[str, Any]], emb_index: Dict[str, Dict[str, Any]]) -> np.ndarray:
    embs: List[np.ndarray] = []
    for r in conv_rows:
        rid = r.get('response_id')
        if rid is None:
            continue
        rec = emb_index.get(str(rid)) or emb_index.get(rid)
        if rec is None:
            continue
        e = rec.get('embedding')
        if e is None:
            continue
        e = np.asarray(e)
        if e.ndim == 1:
            embs.append(e)
        elif e.ndim == 2 and e.shape[0] == 1:
            embs.append(e[0])
    if not embs:
        return None
    return np.mean(np.stack(embs, axis=0), axis=0)


def build_candidates_from_text(full_text: str, ngram_max: int, max_candidates: int) -> Tuple[List[str], np.ndarray]:
    token_pattern = r"(?u)(?:[\u4E00-\u9FFF]{1,}|[\u3131-\u318E\uAC00-\uD7A3]{2,}|[A-Za-z0-9_]{2,})"
    vectorizer = CountVectorizer(analyzer="word", tokenizer=multi_lang_tokenize, ngram_range=(1, ngram_max))
    X = vectorizer.fit_transform([full_text])  # 1 x V
    vocab = np.array(vectorizer.get_feature_names_out())
    counts = X.toarray()[0]
    # Rank by frequency and cap to max_candidates
    order = np.argsort(-counts)
    if max_candidates > 0:
        order = order[:max_candidates]
    return vocab[order].tolist(), counts[order]


def extract_for_conv(
    conv_id: Any,
    rows: List[Dict[str, Any]],
    st_model: SentenceTransformer,
    emb_index: Dict[str, Dict[str, Any]],
    ngram_max: int,
    max_candidates: int,
    top_n: int,
) -> Dict[str, Any]:
    # 1) conversation vector via mean of response embeddings
    conv_vec = conversation_vector_for(rows, emb_index)

    # 2) combined text -> candidates
    parts: List[str] = []
    for r in rows:
        t = clean_minimal(r.get('content', ''))
        if t:
            parts.append(t)
    full_text = ' '.join(parts).strip()

    if not full_text:
        result = {
            'conversation_id': conv_id,
            'conversation_title': rows[0].get('conversation_title', '') if rows else '',
            'response_count': len(rows),
            'keywords': []
        }
        return result

    candidates, _ = build_candidates_from_text(full_text, ngram_max=ngram_max, max_candidates=max_candidates)
    if not candidates:
        return {
            'conversation_id': conv_id,
            'conversation_title': rows[0].get('conversation_title', '') if rows else '',
            'response_count': len(rows),
            'keywords': []
        }

    # 3) embed candidates once
    cand_vecs = st_model.encode(candidates, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    # If no conv_vec available (missing embeddings), fall back to encoding full_text once
    if conv_vec is None:
        conv_vec = st_model.encode([full_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    else:
        # normalize for cosine
        norm = np.linalg.norm(conv_vec)
        if norm > 0:
            conv_vec = conv_vec / norm

    # 4) cosine similarity and rank
    sims = cosine_similarity(conv_vec.reshape(1, -1), cand_vecs).flatten()
    order = np.argsort(-sims)[:top_n]
    final = [{'keyword': candidates[i], 'score': float(sims[i])} for i in order]

    return {
        'conversation_id': conv_id,
        'conversation_title': rows[0].get('conversation_title', '') if rows else '',
        'response_count': len(rows),
        'keywords': final
    }


def run(
    input_path: Path,
    output_path: Path,
    emb_pkl_path: Path,
    model_name: str,
    cache_dir: str,
    ngram_max: int,
    max_candidates: int,
    top_n: int,
):
    with input_path.open('r', encoding='utf-8') as f:
        rows = json.load(f)
    if not rows:
        print(f"No rows in {input_path}")
        return

    grouped = group_by_conversation(rows)

    # Load response embeddings
    emb_index = load_response_embeddings(emb_pkl_path)

    # SentenceTransformer model
    st_model = SentenceTransformer(model_name, cache_folder=cache_dir) if cache_dir else SentenceTransformer(model_name)

    results: List[Dict[str, Any]] = []
    items = list(grouped.items())
    for conv_id, conv_rows in tqdm(items, desc="키워드 추출 (conversations, embed-reuse)"):
        results.append(
            extract_for_conv(
                conv_id,
                conv_rows,
                st_model,
                emb_index,
                ngram_max,
                max_candidates,
                top_n,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Extract keywords per conversation reusing response embeddings (fast)')
    p.add_argument('--input', type=str, default='test/output_full/s1_ai_responses.json')
    p.add_argument('--output', type=str, default='test/output_full/s2_keywords_pipeline_embedreuse.json')
    p.add_argument('--emb-pkl', type=str, default='test/output_full/response_embeddings.pkl')
    p.add_argument('--model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--ngram-max', type=int, default=2)
    p.add_argument('--max-candidates', type=int, default=2000, help='limit number of candidate phrases to encode')
    p.add_argument('--top-n', type=int, default=5)
    args = p.parse_args()

    run(
        Path(args.input),
        Path(args.output),
        Path(args.emb_pkl),
        args.model,
        args.cache_dir,
        args.ngram_max,
        args.max_candidates,
        args.top_n,
    )


if __name__ == '__main__':
    main()
