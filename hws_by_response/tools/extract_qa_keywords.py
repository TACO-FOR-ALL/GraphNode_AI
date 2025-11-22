import json
import argparse
from pathlib import Path
from typing import Any, Dict, List
import sys
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure we can import keyword_tokenizer from the sibling tools/ directory
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from keyword_tokenizer import multi_lang_tokenize


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_candidates_for_text(text: str, ngram_max: int, max_candidates: int) -> List[str]:
    """텍스트 하나에서 1~ngram_max 후보 n-gram 생성 (빈도 기준).

    multi_lang_tokenize를 토크나이저로 사용해서 이미 형태소/불용어 처리가 된 토큰 기준으로 n-gram을 만든다.
    """
    if not text:
        return []

    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=multi_lang_tokenize,
        token_pattern=None,
        ngram_range=(1, ngram_max),
        min_df=1,
    )
    X = vectorizer.fit_transform([text])  # 1 x V
    vocab = np.array(vectorizer.get_feature_names_out())
    counts = X.toarray()[0]

    order = np.argsort(-counts)
    if max_candidates > 0:
        order = order[:max_candidates]

    return vocab[order].tolist()


def extract_keywords_for_conv(
    qa_pairs_path: Path,
    conv_id_target: int,
    model_name: str,
    ngram_max: int,
    max_candidates: int,
    top_n: int,
    output_path: Path,
) -> None:
    qa_pairs = load_qa_pairs(qa_pairs_path)
    if not qa_pairs:
        print(f"No QA pairs in {qa_pairs_path}")
        return

    # 대상 conversation_id 의 QA만 필터링
    qa_in_conv = [p for p in qa_pairs if int(p.get("conversation_id", -1)) == int(conv_id_target)]
    if not qa_in_conv:
        print(f"conversation_id {conv_id_target}에 해당하는 QA 쌍이 없습니다.")
        return

    print(f"conversation_id {conv_id_target} QA 쌍 수: {len(qa_in_conv)}")

    # SBERT 모델 로드
    print(f"Loading model: {model_name}")
    # 기존 파이프라인과 동일하게 models_cache를 캐시 폴더로 사용
    model = SentenceTransformer(model_name, cache_folder="models_cache")

    results: List[Dict[str, Any]] = []
    emb_records: List[Dict[str, Any]] = []

    for pair in qa_in_conv:
        qa_id = pair.get("qa_id")
        question = pair.get("question", "")
        answer = pair.get("answer", "")

        # QA 텍스트 합치기 (질문+답변)
        qa_text = f"Q: {question} A: {answer}"

        # QA 전체 임베딩
        qa_vec = model.encode(
            [qa_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        # 후보 n-gram 생성
        candidates = build_candidates_for_text(qa_text, ngram_max=ngram_max, max_candidates=max_candidates)
        if not candidates:
            results.append(
                {
                    "qa_id": qa_id,
                    "conversation_id": pair.get("conversation_id"),
                    "qa_index": pair.get("qa_index"),
                    "keywords": [],
                }
            )
            continue

        # 후보 임베딩 + 코사인 유사도
        cand_vecs = model.encode(
            candidates,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sims = cosine_similarity(qa_vec.reshape(1, -1), cand_vecs).flatten()

        order = np.argsort(-sims)
        top_indices = order[:top_n]

        top_keywords = [
            {"keyword": candidates[i], "similarity": float(sims[i])}
            for i in top_indices
        ]

        results.append(
            {
                "qa_id": qa_id,
                "conversation_id": pair.get("conversation_id"),
                "qa_index": pair.get("qa_index"),
                "keywords": top_keywords,
            }
        )

        emb_records.append(
            {
                "qa_id": qa_id,
                "conversation_id": pair.get("conversation_id"),
                "qa_index": pair.get("qa_index"),
                "qa_embedding": qa_vec.tolist(),
                "candidates": [
                    {
                        "text": candidates[i],
                        "embedding": cand_vecs[i].tolist(),
                        "similarity": float(sims[i]),
                    }
                    for i in range(len(candidates))
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {output_path}")

    # 항상 QA 및 후보 임베딩을 conv별 pickle로 저장
    emb_output_path = Path("output") / "embeddings" / f"qa_keyword_embeddings_{conv_id_target}.pkl"
    emb_output_path.parent.mkdir(parents=True, exist_ok=True)
    with emb_output_path.open("wb") as f:
        pickle.dump(emb_records, f)
    print(f"임베딩 저장 완료: {emb_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract QA-level keywords for a given conversation id")
    parser.add_argument("--conversation-id", type=int, required=True, help="대상 conversation_id")
    parser.add_argument(
        "--input",
        type=str,
        default="output/qa_pairs.json",
        help="Q-A 쌍 JSON 경로 (기본: output/qa_pairs.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models--intfloat--multilingual-e5-base",
        help="SentenceTransformer 모델 이름 또는 로컬 경로",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="후보 n-gram 최대 길이 (기본: 2)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="각 QA별 후보 n-gram 최대 개수 (0이면 제한 없음)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="각 QA별 최종 키워드 개수",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/keywords/qa_keywords_conv.json",
        help="출력 JSON 경로 (기본: output/keywords/qa_keywords_conv.json)",
    )

    args = parser.parse_args()

    conv_id = args.conversation_id
    input_path = Path(args.input)
    output_path = Path(args.output)

    # conv id에 맞춰 기본 출력 파일명 자동 조정
    if output_path.name == "qa_keywords_conv.json":
        output_path = output_path.parent / f"qa_keywords_conv_{conv_id}.json"

    extract_keywords_for_conv(
        qa_pairs_path=input_path,
        conv_id_target=conv_id,
        model_name=args.model,
        ngram_max=args.ngram_max,
        max_candidates=args.max_candidates,
        top_n=args.top_n,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
