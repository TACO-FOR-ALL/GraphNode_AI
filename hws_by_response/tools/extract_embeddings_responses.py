"""
Step 3: Extract SBERT embeddings per response (incremental cache)
- Input: s1 responses json (default: test/output_full/s1_ai_responses.json)
- Output cache PKL: test/output_full/response_embeddings.pkl
- Model: paraphrase-multilingual-mpnet-base-v2 by default (override via --model)
- Cache dir: models_cache by default (override via --cache-dir)
Cache format: dict[response_id] = {
  'conversation_id': ..., 'response_id': ..., 'embedding': np.ndarray,
  'conversation_title': ..., 'timestamp': ...
}
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
from tqdm import tqdm


def load_rows(path: Path):
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


def run(
    input_path: Path,
    out_pkl: Path,
    model_name: str,
    cache_dir: str,
    batch_size: int = 32,
    device: str = "auto",
    fp16: bool = False,
    normalize_embeddings: bool = True,
    chunk_size: int = 4096,
    save_every: int = 2000,
    num_workers: int = 0,
    long_strategy: str = "truncate",
    max_seq_length: int = 512,
):
    rows = load_rows(input_path)
    # basic schema check
    if not rows:
        print(f"No rows in {input_path}")
        return
    required = {"response_id", "conversation_id", "content"}
    missing = required - set(rows[0].keys())
    if missing:
        print(f"Warning: input schema missing keys {missing}. Proceeding with best-effort.")
    cache: Dict[str, Dict[str, Any]] = load_cache(out_pkl)

    # build list of new items to encode
    to_encode = [r for r in rows if r['response_id'] not in cache]

    print(f"Total rows: {len(rows)} | New to encode: {len(to_encode)} | Cached: {len(cache)}")
    if to_encode:
        if cache_dir:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        else:
            model = SentenceTransformer(model_name)

        # Set max_seq_length to handle longer texts
        model.max_seq_length = max_seq_length
        print(f"Set model.max_seq_length to {max_seq_length}")

        if device == "auto":
            use_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            use_device = device
        try:
            model.to(use_device)
        except Exception as e:
            print(f"Warning: failed to move model to {use_device}: {e}. Using default device.")
            use_device = str(model._target_device) if hasattr(model, "_target_device") else "cpu"

        if fp16 and use_device.startswith("cuda"):
            try:
                model.half()
            except Exception as e:
                print(f"Warning: failed to switch model to fp16: {e}")

        try:
            max_len = getattr(model, 'max_seq_length', None)
            print(f"Model max_seq_length: {max_len}")
        except Exception:
            max_len = None
        # crude estimate using tokenizer if available
        over_count = 0
        try:
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is not None:
                sample_texts = [r.get('content', '') for r in to_encode[:2000]]
                for t in sample_texts:
                    if not t:
                        continue
                    ids = tokenizer.encode(t, add_special_tokens=True, truncation=False)
                    if max_len and len(ids) > max_len:
                        over_count += 1
            print(f"Estimated long responses > max_seq_length among first {min(2000, len(to_encode))}: {over_count}")
        except Exception as e:
            print(f"Note: could not estimate long responses: {e}")

        def iter_chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i + n]

        encoded_count = 0

        # Prepare encode kwargs
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": True,
        }
        if num_workers > 0:
            encode_kwargs["num_workers"] = num_workers

        print(f"Using batch_size={batch_size}, num_workers={num_workers}, device={use_device}")

        pbar = tqdm(total=len(to_encode), desc="Encoding responses", unit="response")

        # Process in chunks for memory management and periodic saves
        for chunk in iter_chunks(to_encode, chunk_size):
            # Extract texts from chunk
            texts = [r.get('content', '') for r in chunk]

            try:
                # Batch encode all texts in chunk at once
                if long_strategy == "truncate":
                    # Simple batch encoding with truncation
                    embeddings = model.encode(texts, **encode_kwargs)

                else:
                    # chunk-mean strategy: batch process all chunks together
                    if tokenizer is None or not max_len:
                        # Fallback to simple batch encoding
                        embeddings = model.encode(texts, **encode_kwargs)
                    else:
                        # Step 1: Tokenize all texts and prepare chunks
                        all_chunk_texts = []
                        chunk_metadata = []  # (text_idx, chunk_lengths_list)

                        for text_idx, text in enumerate(texts):
                            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)

                            if len(ids) <= max_len:
                                # Short text: no chunking needed
                                all_chunk_texts.append(text)
                                chunk_metadata.append((text_idx, [len(ids)]))
                            else:
                                # Long text: split into chunks
                                chunk_lengths = []
                                for i in range(0, len(ids), max_len):
                                    chunk_ids = ids[i:i+max_len]
                                    chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                                    all_chunk_texts.append(chunk_text)
                                    chunk_lengths.append(len(chunk_ids))
                                chunk_metadata.append((text_idx, chunk_lengths))

                        # Step 2: Batch encode ALL chunks at once (parallel processing)
                        all_chunk_embeddings = model.encode(all_chunk_texts, **encode_kwargs)
                        all_chunk_embeddings = np.asarray(all_chunk_embeddings, dtype=np.float32)

                        # Step 3: Reconstruct embeddings per original text
                        embeddings = []
                        chunk_idx = 0

                        for text_idx, chunk_lengths in chunk_metadata:
                            num_chunks = len(chunk_lengths)

                            if num_chunks == 1:
                                # Single chunk (short text)
                                embeddings.append(all_chunk_embeddings[chunk_idx])
                                chunk_idx += 1
                            else:
                                # Multiple chunks: length-weighted average
                                chunk_vecs = all_chunk_embeddings[chunk_idx:chunk_idx + num_chunks]
                                chunk_idx += num_chunks

                                # Length-weighted average
                                chunk_lengths_arr = np.asarray(chunk_lengths, dtype=np.float32)
                                weights = chunk_lengths_arr / chunk_lengths_arr.sum()
                                mean_vec = np.average(chunk_vecs, axis=0, weights=weights)

                                # Re-normalize if needed
                                if normalize_embeddings:
                                    norm = np.linalg.norm(mean_vec)
                                    if norm > 0:
                                        mean_vec = mean_vec / norm

                                embeddings.append(mean_vec.astype(np.float32))

                        embeddings = np.array(embeddings)

                # Store embeddings in cache
                embeddings = np.asarray(embeddings, dtype=np.float32)
                for r, vec in zip(chunk, embeddings):
                    cache[r['response_id']] = {
                        'conversation_id': r.get('conversation_id'),
                        'response_id': r.get('response_id'),
                        'conversation_title': r.get('conversation_title', ''),
                        'timestamp': r.get('timestamp'),
                        'embedding': vec,
                    }

                pbar.update(len(chunk))
                encoded_count += len(chunk)

                # Periodic save
                if save_every and (encoded_count % save_every == 0):
                    save_cache(out_pkl, cache)

            except Exception as e:
                print(f"\nError encoding batch: {e}")
                raise

        pbar.close()

        save_cache(out_pkl, cache)
        print(f"Encoded and cached: {len(to_encode)} embeddings")
    else:
        print("No new responses to encode. Cache up-to-date.")

    print(f"Embeddings cache size: {len(cache)}")


def main():
    p = argparse.ArgumentParser(description='Extract SBERT embeddings (incremental)')
    p.add_argument('--input', type=str, default='output/s1_ai_responses.json')
    p.add_argument('--out-pkl', type=str, default='output/response_embeddings.pkl')
    p.add_argument('--model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--batch-size', type=int, default=128, help='Batch size for encoding (increase for faster processing)')
    p.add_argument('--device', type=str, default='auto', help="cpu, cuda, or auto")
    p.add_argument('--fp16', action='store_true', help='Use FP16 precision (faster on GPU)')
    p.add_argument('--normalize', action='store_true', help='Normalize embeddings to unit length')
    p.add_argument('--chunk-size', type=int, default=4096, help='Number of items to process before saving')
    p.add_argument('--save-every', type=int, default=2000, help='Save cache every N items')
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader workers for tokenization (0=single-threaded, 4=recommended)')
    p.add_argument('--long-strategy', type=str, default='truncate', choices=['truncate', 'chunk-mean'], help='Handle long responses by truncation or chunk-averaged embeddings')
    p.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length for model (default: 512)')
    args = p.parse_args()

    run(
        Path(args.input),
        Path(args.out_pkl),
        args.model,
        args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16,
        normalize_embeddings=args.normalize,
        chunk_size=args.chunk_size,
        save_every=args.save_every,
        num_workers=args.num_workers,
        long_strategy=args.long_strategy,
        max_seq_length=args.max_seq_length,
    )


if __name__ == '__main__':
    main()
