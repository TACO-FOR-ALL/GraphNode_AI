"""
Q-A 쌍 embedding 추출 (512 토큰 제한)
- Input: qa_pairs.json
- Output: qa_embeddings.pkl
- Strategy: Q 전체 + A 앞부분 (512 토큰 맞춤)
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


def load_qa_pairs(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_cache(path: Path):
    if not path.exists():
        return {}
    with path.open('rb') as f:
        return pickle.load(f)


def save_cache(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f)


def truncate_qa_to_512(
    question: str,
    answer: str,
    tokenizer,
    max_tokens: int = 512
) -> str:
    """
    Q 전체 + A 앞부분을 512 토큰에 맞춤
    
    Args:
        question: 질문 텍스트
        answer: 답변 텍스트
        tokenizer: 토크나이저
        max_tokens: 최대 토큰 수
        
    Returns:
        512 토큰에 맞춰진 Q-A 텍스트
    """
    # Q 토큰화
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    
    # 남은 토큰 수 계산 (special tokens 고려)
    special_tokens_count = 3  # [CLS], [SEP], [SEP]
    remaining_tokens = max_tokens - len(q_tokens) - special_tokens_count
    
    if remaining_tokens <= 0:
        # Q가 너무 길면 Q만 truncate
        q_tokens_truncated = q_tokens[:max_tokens - special_tokens_count]
        return tokenizer.decode(q_tokens_truncated, skip_special_tokens=True)
    
    # A 토큰화 및 truncate
    a_tokens = tokenizer.encode(answer, add_special_tokens=False)
    a_tokens_truncated = a_tokens[:remaining_tokens]
    a_truncated = tokenizer.decode(a_tokens_truncated, skip_special_tokens=True)
    
    # Q + A 합치기
    qa_text = f"Q: {question} A: {a_truncated}"
    
    return qa_text


def run(
    input_path: Path,
    out_pkl: Path,
    model_name: str,
    cache_dir: str,
    batch_size: int = 32,
    device: str = "auto",
    fp16: bool = False,
    normalize_embeddings: bool = True,
    save_every: int = 500,
    max_seq_length: int = 512,
    use_instruction_prefix: bool = True,
    question_weight: float = 0.7,
    extract_mode: str = "qa",
):
    qa_pairs = load_qa_pairs(input_path)
    
    if not qa_pairs:
        print(f"No Q-A pairs in {input_path}")
        return
    
    cache: Dict[str, Dict[str, Any]] = load_cache(out_pkl)
    
    # 새로 encode할 항목 찾기
    to_encode = [pair for pair in qa_pairs if pair['qa_id'] not in cache]
    
    print(f"Total Q-A pairs: {len(qa_pairs)} | New to encode: {len(to_encode)} | Cached: {len(cache)}")
    
    if to_encode:
        # 모델 로드
        if cache_dir:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        else:
            model = SentenceTransformer(model_name)
        
        model.max_seq_length = max_seq_length
        print(f"Set model.max_seq_length to {max_seq_length}")
        
        # 디바이스 설정
        if device == "auto":
            use_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            use_device = device
        
        try:
            model.to(use_device)
        except Exception as e:
            print(f"Warning: failed to move model to {use_device}: {e}")
            use_device = "cpu"
        
        if fp16 and use_device.startswith("cuda"):
            try:
                model.half()
            except Exception as e:
                print(f"Warning: failed to switch model to fp16: {e}")
        
        print(f"Model device: {use_device}")
        
        # 토크나이저 가져오기
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None:
            print("Warning: tokenizer not found. Using simple truncation.")
        
        # Encoding
        print(f"Extract mode: {extract_mode}")
        print(f"Encoding strategy: {'Instruction Prefix (Q weight: ' + str(question_weight) + ')' if use_instruction_prefix else 'Standard concatenation'}")
        print(f"Using batch_size={batch_size}")

        # Prepare encode kwargs
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": True,
        }

        pbar = tqdm(total=len(to_encode), desc=f"Encoding ({extract_mode})", unit="pair")
        encoded_count = 0

        # Process in batches
        def iter_batches(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i + n]

        for batch_pairs in iter_batches(to_encode, batch_size):
            try:
                # Prepare texts based on extract mode
                if extract_mode == "question":
                    # Question만 사용
                    texts = []
                    for pair in batch_pairs:
                        question = pair.get('question', '')
                        if use_instruction_prefix:
                            texts.append(f"query: {question}")
                        else:
                            texts.append(question)

                    # Batch encode
                    embeddings = model.encode(texts, **encode_kwargs)

                elif extract_mode == "answer":
                    # Answer만 사용 (512 토큰 제한)
                    texts = []
                    for pair in batch_pairs:
                        answer = pair.get('answer', '')
                        if tokenizer:
                            a_tokens = tokenizer.encode(answer, add_special_tokens=False)
                            max_a_tokens = max_seq_length - 10
                            a_tokens_truncated = a_tokens[:max_a_tokens]
                            a_truncated = tokenizer.decode(a_tokens_truncated, skip_special_tokens=True)
                        else:
                            a_truncated = answer

                        if use_instruction_prefix:
                            texts.append(f"passage: {a_truncated}")
                        else:
                            texts.append(a_truncated)

                    # Batch encode
                    embeddings = model.encode(texts, **encode_kwargs)

                else:  # extract_mode == "qa" (기본값)
                    if use_instruction_prefix:
                        # Instruction Prefix + Weighted Embedding
                        q_texts = []
                        a_texts = []

                        for pair in batch_pairs:
                            question = pair.get('question', '')
                            answer = pair.get('answer', '')

                            q_texts.append(f"query: {question}")

                            # A는 512 토큰 제한
                            if tokenizer:
                                a_tokens = tokenizer.encode(answer, add_special_tokens=False)
                                max_a_tokens = max_seq_length - 10
                                a_tokens_truncated = a_tokens[:max_a_tokens]
                                a_truncated = tokenizer.decode(a_tokens_truncated, skip_special_tokens=True)
                            else:
                                a_truncated = answer

                            a_texts.append(f"passage: {a_truncated}")

                        # Batch encode Q and A separately
                        q_embeddings = model.encode(q_texts, **encode_kwargs)
                        a_embeddings = model.encode(a_texts, **encode_kwargs)

                        # Weighted combination
                        embeddings = question_weight * q_embeddings + (1 - question_weight) * a_embeddings

                        # Normalize
                        if normalize_embeddings:
                            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                            norms[norms == 0] = 1.0
                            embeddings = embeddings / norms

                    else:
                        # Standard concatenation: Q 전체 + A 앞부분 (512 토큰 맞춤)
                        texts = []
                        for pair in batch_pairs:
                            question = pair.get('question', '')
                            answer = pair.get('answer', '')

                            # Q와 A 전체를 그대로 사용
                            qa_text = f"Q: {question} A: {answer}"
                            texts.append(qa_text)

                        # 토크나이저와 max_seq_length가 없으면 단순 인코딩 (내부 truncate)
                        if tokenizer is None or not max_seq_length:
                            embeddings = model.encode(texts, **encode_kwargs)
                        else:
                            max_len = max_seq_length

                            # 1단계: 모든 텍스트를 토크나이즈하고 chunk로 분할
                            all_chunk_texts = []
                            chunk_metadata = []  # (text_idx, chunk_lengths_list)

                            for text_idx, text in enumerate(texts):
                                ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)

                                if len(ids) <= max_len:
                                    # 짧은 텍스트: chunking 불필요
                                    all_chunk_texts.append(text)
                                    chunk_metadata.append((text_idx, [len(ids)]))
                                else:
                                    # 긴 텍스트: max_len 단위로 나누기
                                    chunk_lengths = []
                                    for i in range(0, len(ids), max_len):
                                        chunk_ids = ids[i:i + max_len]
                                        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                                        all_chunk_texts.append(chunk_text)
                                        chunk_lengths.append(len(chunk_ids))
                                    chunk_metadata.append((text_idx, chunk_lengths))

                            # 2단계: 모든 chunk를 한 번에 배치 인코딩
                            all_chunk_embeddings = model.encode(all_chunk_texts, **encode_kwargs)
                            all_chunk_embeddings = np.asarray(all_chunk_embeddings, dtype=np.float32)

                            # 3단계: 원래 텍스트 단위로 임베딩 재구성 (토큰 길이 비율 가중 평균)
                            embeddings = []
                            chunk_idx = 0

                            for text_idx, chunk_lengths in chunk_metadata:
                                num_chunks = len(chunk_lengths)

                                if num_chunks == 1:
                                    # 단일 chunk (짧은 텍스트)
                                    embeddings.append(all_chunk_embeddings[chunk_idx])
                                    chunk_idx += 1
                                else:
                                    # 여러 chunk: 토큰 길이 비율 가중 평균
                                    chunk_vecs = all_chunk_embeddings[chunk_idx:chunk_idx + num_chunks]
                                    chunk_idx += num_chunks

                                    chunk_lengths_arr = np.asarray(chunk_lengths, dtype=np.float32)
                                    weights = chunk_lengths_arr / chunk_lengths_arr.sum()
                                    mean_vec = np.average(chunk_vecs, axis=0, weights=weights)

                                    # 필요 시 재정규화
                                    if normalize_embeddings:
                                        norm = np.linalg.norm(mean_vec)
                                        if norm > 0:
                                            mean_vec = mean_vec / norm

                                    embeddings.append(mean_vec.astype(np.float32))

                            embeddings = np.array(embeddings)

                # Store embeddings
                embeddings = np.asarray(embeddings, dtype=np.float32)
                for pair, vec in zip(batch_pairs, embeddings):
                    cache[pair['qa_id']] = {
                        'qa_id': pair['qa_id'],
                        'conversation_id': pair.get('conversation_id'),
                        'conversation_title': pair.get('conversation_title', ''),
                        'qa_index': pair.get('qa_index'),
                        'timestamp': pair.get('timestamp'),
                        'embedding': vec,
                    }

                pbar.update(len(batch_pairs))
                encoded_count += len(batch_pairs)

                if save_every and (encoded_count % save_every == 0):
                    save_cache(out_pkl, cache)

            except Exception as e:
                print(f"\nError encoding batch: {e}")
                raise

        pbar.close()
        save_cache(out_pkl, cache)
        print(f"Encoded and cached: {len(to_encode)} Q-A embeddings")
    else:
        print("No new Q-A pairs to encode. Cache up-to-date.")
    
    print(f"Q-A embeddings cache size: {len(cache)}")


def main():
    p = argparse.ArgumentParser(description='Extract Q-A pair embeddings (512 token limit)')
    p.add_argument('--input', type=str, default='output/qa_pairs.json')
    p.add_argument('--out-pkl', type=str, default='output/qa_embeddings.pkl')
    p.add_argument('--model', type=str, default='intfloat/multilingual-e5-base')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', type=str, default='auto', help="cpu, cuda, or auto")
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--normalize', action='store_true', default=True)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length (default: 512)')
    p.add_argument('--use-instruction-prefix', action='store_true', help='Use E5 instruction prefix (query/passage) with weighted embedding')
    p.add_argument('--question-weight', type=float, default=0.7, help='Weight for question embedding when using instruction prefix (0.0-1.0, default: 0.7)')
    p.add_argument('--extract-mode', type=str, default='qa', choices=['qa', 'question', 'answer'],
                   help='Extract mode: qa (both), question (only), answer (only). Default: qa')
    args = p.parse_args()

    # Auto-adjust output filename based on extract_mode
    out_pkl = Path(args.out_pkl)
    if args.extract_mode != 'qa':
        # Add suffix before .pkl extension
        out_pkl = out_pkl.parent / f"{out_pkl.stem}_{args.extract_mode}{out_pkl.suffix}"

    run(
        Path(args.input),
        out_pkl,
        args.model,
        args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16,
        normalize_embeddings=args.normalize,
        save_every=args.save_every,
        max_seq_length=args.max_seq_length,
        use_instruction_prefix=args.use_instruction_prefix,
        question_weight=args.question_weight,
        extract_mode=args.extract_mode,
    )


if __name__ == '__main__':
    main()
