"""
Q-A 쌍 embedding을 Conversation-level로 pooling
- Input: qa_embeddings.pkl
- Output: conversation_embeddings.pkl
- Strategy: 길이 가중 평균 (긴 Q-A 쌍에 더 큰 가중치)
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import numpy as np


def load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open('rb') as f:
        return pickle.load(f)


def save_cache(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f)


def pool_qa_embeddings_to_conversation(
    qa_embeddings: Dict[str, Dict[str, Any]],
    pooling_strategy: str = "length_weighted"
) -> Dict[int, Dict[str, Any]]:
    """
    Q-A 쌍 embedding을 Conversation-level로 pooling
    
    Args:
        qa_embeddings: Q-A 쌍 embedding cache
        pooling_strategy: pooling 전략 ("mean", "length_weighted", "recency_weighted")
        
    Returns:
        Conversation-level embedding cache
    """
    # Conversation별로 Q-A 쌍 그룹화
    conv_qa_groups = defaultdict(list)
    
    for qa_id, qa_data in qa_embeddings.items():
        conv_id = qa_data['conversation_id']
        conv_qa_groups[conv_id].append(qa_data)
    
    # Conversation-level embedding 생성
    conversation_embeddings = {}
    
    for conv_id, qa_list in conv_qa_groups.items():
        # qa_index로 정렬 (시간순)
        qa_list_sorted = sorted(qa_list, key=lambda x: x.get('qa_index', 0))
        
        # Embedding 추출
        embeddings = np.array([qa['embedding'] for qa in qa_list_sorted])
        
        # Pooling 전략에 따라 가중치 계산
        if pooling_strategy == "mean":
            # 균등 평균
            weights = np.ones(len(embeddings))
        
        elif pooling_strategy == "length_weighted":
            # 길이 가중 평균 (embedding norm 기반)
            weights = np.linalg.norm(embeddings, axis=1)
        
        elif pooling_strategy == "recency_weighted":
            # 최신 Q-A에 더 큰 가중치
            weights = np.arange(1, len(embeddings) + 1, dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # 가중 평균
        weights = weights / weights.sum()
        conv_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # 정규화
        norm = np.linalg.norm(conv_embedding)
        if norm > 0:
            conv_embedding = conv_embedding / norm
        
        # 저장
        conversation_embeddings[conv_id] = {
            'conversation_id': conv_id,
            'conversation_title': qa_list_sorted[0].get('conversation_title', ''),
            'qa_count': len(qa_list_sorted),
            'embedding': conv_embedding.astype(np.float32),
        }
    
    return conversation_embeddings


def run(
    input_pkl: Path,
    output_pkl: Path,
    pooling_strategy: str = "length_weighted"
):
    print(f"Loading Q-A embeddings from: {input_pkl}")
    qa_embeddings = load_cache(input_pkl)
    print(f"Loaded {len(qa_embeddings)} Q-A embeddings")
    
    print(f"Pooling strategy: {pooling_strategy}")
    conversation_embeddings = pool_qa_embeddings_to_conversation(
        qa_embeddings,
        pooling_strategy=pooling_strategy
    )
    
    print(f"Generated {len(conversation_embeddings)} conversation embeddings")
    
    save_cache(output_pkl, conversation_embeddings)
    print(f"Saved to: {output_pkl}")
    
    # 통계 출력
    qa_counts = [conv['qa_count'] for conv in conversation_embeddings.values()]
    print(f"\nStatistics:")
    print(f"  Total conversations: {len(conversation_embeddings)}")
    print(f"  Avg Q-A pairs per conversation: {np.mean(qa_counts):.2f}")
    print(f"  Min Q-A pairs: {np.min(qa_counts)}")
    print(f"  Max Q-A pairs: {np.max(qa_counts)}")


def main():
    p = argparse.ArgumentParser(description='Pool Q-A embeddings to conversation-level')
    p.add_argument('--input', type=str, default='output/qa_embeddings.pkl', help='Input Q-A embeddings pkl')
    p.add_argument('--output', type=str, default='output/conversation_embeddings.pkl', help='Output conversation embeddings pkl')
    p.add_argument('--pooling', type=str, default='length_weighted', 
                   choices=['mean', 'length_weighted', 'recency_weighted'],
                   help='Pooling strategy')
    args = p.parse_args()

    run(
        Path(args.input),
        Path(args.output),
        pooling_strategy=args.pooling
    )


if __name__ == '__main__':
    main()
