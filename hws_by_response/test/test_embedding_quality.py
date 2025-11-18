#!/usr/bin/env python3
"""Embedding 품질 확인 스크립트"""
import pickle
import json
import numpy as np
from collections import defaultdict

# Load embeddings
with open("test/output/response_embeddings.pkl", "rb") as f:
    emb_cache = pickle.load(f)

# Load responses
with open("test/output/s1_ai_responses.json") as f:
    responses = json.load(f)

# Load categories
with open("test/output/s6_categories_assignments.json") as f:
    categories = json.load(f)

# Build conversation vectors (length-weighted pooling)
conv_vectors = defaultdict(lambda: {"vecs": [], "lengths": []})
for r in responses:
    cid = r.get("conversation_id")
    rid = r.get("response_id")
    if rid in emb_cache:
        emb = emb_cache[rid]["embedding"]
        length = len(r.get("content", ""))
        conv_vectors[cid]["vecs"].append(emb)
        conv_vectors[cid]["lengths"].append(length)

# Pool by length
pooled = {}
for cid, data in conv_vectors.items():
    vecs = np.array(data["vecs"])
    lengths = np.array(data["lengths"], dtype=np.float32)
    if np.all(lengths == 0):
        lengths = np.ones_like(lengths)
    weights = lengths / lengths.sum()
    pooled_vec = np.average(vecs, axis=0, weights=weights)
    pooled[cid] = pooled_vec

print(f"Total conversations: {len(pooled)}")
print(f"Embedding dimension: {pooled[list(pooled.keys())[0]].shape}")

# Check embedding norms
norms = [np.linalg.norm(v) for v in pooled.values()]
print(f"\nEmbedding norms:")
print(f"  Mean: {np.mean(norms):.4f}")
print(f"  Std: {np.std(norms):.4f}")
print(f"  Min: {np.min(norms):.4f}")
print(f"  Max: {np.max(norms):.4f}")

# Check within-category similarities
within_sims = []
cross_sims = []

conv_ids = sorted(pooled.keys())
for i, cid1 in enumerate(conv_ids):
    cat1 = categories.get(str(cid1), {}).get("category")
    for cid2 in conv_ids[i+1:]:
        cat2 = categories.get(str(cid2), {}).get("category")
        
        v1 = pooled[cid1]
        v2 = pooled[cid2]
        
        # L2 distance
        dist = np.linalg.norm(v1 - v2)
        sim = 1.0 / (1.0 + dist)
        
        if cat1 == cat2:
            within_sims.append(sim)
        else:
            cross_sims.append(sim)

print(f"\nWithin-category similarities (n={len(within_sims)}):")
print(f"  Mean: {np.mean(within_sims):.4f}")
print(f"  Median: {np.median(within_sims):.4f}")
print(f"  Std: {np.std(within_sims):.4f}")
print(f"  Min: {np.min(within_sims):.4f}")
print(f"  Max: {np.max(within_sims):.4f}")

print(f"\nCross-category similarities (n={len(cross_sims)}):")
print(f"  Mean: {np.mean(cross_sims):.4f}")
print(f"  Median: {np.median(cross_sims):.4f}")
print(f"  Std: {np.std(cross_sims):.4f}")
print(f"  Min: {np.min(cross_sims):.4f}")
print(f"  Max: {np.max(cross_sims):.4f}")

print(f"\nDifference (within - cross):")
print(f"  Mean diff: {np.mean(within_sims) - np.mean(cross_sims):.4f}")
print(f"  Median diff: {np.median(within_sims) - np.median(cross_sims):.4f}")
