"""
TF-IDF based conversation-level keyword extraction with embedding reuse (Step 3)
- For each conversation:
  1) Build conversation vector by averaging response-level embeddings from PKL
  2) Build candidate phrases using TF-IDF across all conversations
  3) Encode candidates once with the same SBERT model
  4) Rank by cosine similarity to the conversation vector
  5) Output top-N keywords per conversation

Key differences from embedreuse version:
- Uses TF-IDF instead of raw frequency for candidate selection
- Reduces 1-gram bias by considering document frequency
- Better at surfacing meaningful 2-gram and 3-gram phrases

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
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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


def conversation_vector_for(
    conv_rows: List[Dict[str, Any]],
    emb_index: Dict[str, Dict[str, Any]],
    emb_source: str = "response",
    conv_id: Any = None
) -> np.ndarray:
    """
    Get conversation vector from embeddings

    Args:
        conv_rows: List of response rows for this conversation
        emb_index: Embedding cache (response or conversation level)
        emb_source: "response" or "conversation"
        conv_id: Conversation ID (used when emb_source="conversation")

    Returns:
        Conversation vector (np.ndarray) or None
    """
    if emb_source == "conversation":
        # Use pre-pooled conversation embedding
        rec = emb_index.get(str(conv_id)) or emb_index.get(conv_id)
        if rec is None:
            return None
        e = rec.get('embedding')
        if e is None:
            return None
        e = np.asarray(e)
        if e.ndim == 1:
            return e
        elif e.ndim == 2 and e.shape[0] == 1:
            return e[0]
        return None

    # emb_source == "response": original behavior
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


def extract_for_conv(
    conv_idx: int,
    conv_id: Any,
    rows: List[Dict[str, Any]],
    st_model: SentenceTransformer,
    emb_index: Dict[str, Dict[str, Any]],
    tfidf_matrix: np.ndarray,
    vocab: np.ndarray,
    full_text: str,
    max_candidates: int,
    top_n: int,
    emb_source: str = "response",
) -> Dict[str, Any]:
    # 1) conversation vector via mean of response embeddings
    conv_vec = conversation_vector_for(rows, emb_index, emb_source=emb_source, conv_id=conv_id)

    if not full_text:
        return {
            'conversation_id': conv_id,
            'conversation_title': rows[0].get('conversation_title', '') if rows else '',
            'response_count': len(rows),
            'keywords': []
        }

    # 2) Get TF-IDF scores for this conversation
    tfidf_scores = tfidf_matrix[conv_idx].toarray()[0]

    # 3) Select top candidates by TF-IDF score with n-gram ratio control
    # Separate candidates by n-gram type
    ngram_candidates = {1: [], 2: [], 3: []}
    ngram_scores = {1: [], 2: [], 3: []}
    
    for idx in np.argsort(-tfidf_scores):
        phrase = vocab[idx]
        score = tfidf_scores[idx]
        if score == 0:
            break
        
        # Count spaces to determine n-gram type
        n = phrase.count(' ') + 1
        if n <= 3:
            ngram_candidates[n].append(phrase)
            ngram_scores[n].append(score)
    
    # Apply ratio: 1-gram 40%, 2-gram 40%, 3-gram 20%
    n1_count = int(max_candidates * 0.4)
    n2_count = int(max_candidates * 0.4)
    n3_count = max_candidates - n1_count - n2_count  # Remaining for 3-gram
    
    candidates = []
    candidates.extend(ngram_candidates[1][:n1_count])
    candidates.extend(ngram_candidates[2][:n2_count])
    candidates.extend(ngram_candidates[3][:n3_count])

    if not candidates:
        return {
            'conversation_id': conv_id,
            'conversation_title': rows[0].get('conversation_title', '') if rows else '',
            'response_count': len(rows),
            'keywords': []
        }

    # 4) embed candidates once
    cand_vecs = st_model.encode(candidates, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    # If no conv_vec available (missing embeddings), fall back to encoding full_text once
    if conv_vec is None:
        conv_vec = st_model.encode([full_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    else:
        # normalize for cosine
        norm = np.linalg.norm(conv_vec)
        if norm > 0:
            conv_vec = conv_vec / norm

    # 5) cosine similarity and rank
    sims = cosine_similarity(conv_vec.reshape(1, -1), cand_vecs).flatten()
    final_order = np.argsort(-sims)[:top_n]
    final = [{'keyword': candidates[i], 'score': float(sims[i])} for i in final_order]

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
    max_df: float,
    min_df: int,
    emb_source: str = "response",
):
    with input_path.open('r', encoding='utf-8') as f:
        rows = json.load(f)
    if not rows:
        print(f"No rows in {input_path}")
        return

    grouped = group_by_conversation(rows)
    print(f"총 {len(grouped)}개 대화 발견")

    # Load embeddings (response or conversation level)
    print(f"Loading embeddings from {emb_pkl_path} (source: {emb_source})")
    emb_index = load_response_embeddings(emb_pkl_path)

    # SentenceTransformer model
    st_model = SentenceTransformer(model_name, cache_folder=cache_dir) if cache_dir else SentenceTransformer(model_name)

    # STEP 1: Collect all conversation texts for TF-IDF
    print("전체 대화 텍스트 수집 중...")
    all_conv_texts = []
    conv_ids = []
    conv_data = []  # Store (conv_id, conv_rows) pairs in order

    for conv_id, conv_rows in grouped.items():
        parts = [clean_minimal(r.get('content', '')) for r in conv_rows if r.get('content')]
        full_text = ' '.join(parts).strip()
        all_conv_texts.append(full_text)
        conv_ids.append(conv_id)
        conv_data.append((conv_id, conv_rows, full_text))

    # STEP 2: Build TF-IDF matrix
    print(f"TF-IDF 학습 중 (ngram_max={ngram_max}, max_df={max_df}, min_df={min_df})...")
    token_pattern = r"(?u)(?:[\u4E00-\u9FFF]{1,}|[\u3131-\u318E\uAC00-\uD7A3]{2,}|[A-Za-z0-9_]{2,})"

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=multi_lang_tokenize,
        ngram_range=(1, ngram_max),
        max_df=max_df,  # Ignore terms that appear in more than max_df of documents
        min_df=min_df,  # Ignore terms that appear in fewer than min_df documents
        sublinear_tf=True,  # Use log(1 + tf) instead of raw tf
    )

    tfidf_matrix = vectorizer.fit_transform(all_conv_texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    print(f"TF-IDF 완료: {len(vocab)}개 유니크 n-gram 생성")

    # STEP 3: Extract keywords for each conversation
    results: List[Dict[str, Any]] = []
    for idx, (conv_id, conv_rows, full_text) in enumerate(tqdm(conv_data, desc="키워드 추출 (TF-IDF)")):
        results.append(
            extract_for_conv(
                idx,
                conv_id,
                conv_rows,
                st_model,
                emb_index,
                tfidf_matrix,
                vocab,
                full_text,
                max_candidates,
                top_n,
                emb_source=emb_source,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path}")


def main():
    p = argparse.ArgumentParser(description='Extract keywords per conversation using TF-IDF and embeddings')
    p.add_argument('--input', type=str, default='test/output_full/s1_ai_responses.json')
    p.add_argument('--output', type=str, default='test/output_full/s2_keywords_tfidf.json')
    p.add_argument('--emb-pkl', type=str, default='test/output_full/response_embeddings.pkl')
    p.add_argument('--model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--ngram-max', type=int, default=2)
    p.add_argument('--max-candidates', type=int, default=100, help='limit number of candidate phrases to encode')
    p.add_argument('--top-n', type=int, default=5, help='final number of keywords per conversation')
    p.add_argument('--max-df', type=float, default=0.7, help='ignore terms appearing in more than this fraction of documents')
    p.add_argument('--min-df', type=int, default=1, help='ignore terms appearing in fewer than this number of documents')
    p.add_argument('--emb-source', type=str, default='response', choices=['response', 'conversation'],
                   help='Embedding source: response (response_embeddings.pkl), conversation (conversation_embeddings.pkl)')
    args = p.parse_args()

    # Auto-adjust output filename based on emb_source
    output_path = Path(args.output)
    if args.emb_source == 'conversation':
        # Add suffix before .json extension
        output_path = output_path.parent / f"{output_path.stem}_qa{output_path.suffix}"

    run(
        Path(args.input),
        output_path,
        Path(args.emb_pkl),
        args.model,
        args.cache_dir,
        args.ngram_max,
        args.max_candidates,
        args.top_n,
        args.max_df,
        args.min_df,
        emb_source=args.emb_source,
    )


if __name__ == '__main__':
    main()
