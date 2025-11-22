import sys
import json
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
try:
    from tools.preprocess import preprocess_content
except Exception:
    preprocess_content = None


def build_qa_pairs(sample_size: int = 50, data_path: str = None, output_path: str = None) -> str:
    """
    Q-A 쌍 추출 및 전처리
    
    Args:
        sample_size: 샘플 대화 개수
        data_path: 원본 데이터 경로
        output_path: 생성할 qa_pairs.json 경로
        
    Returns:
        생성된 파일 경로
    """
    start = time.time()

    loader = ConversationLoader(data_path=data_path) if data_path else ConversationLoader()
    load_t0 = time.time()
    conversations = loader.load_sample(n=sample_size)
    load_elapsed = time.time() - load_t0

    parser = NoteParser(min_content_length=20)
    parse_t0 = time.time()
    qa_pairs = parser.parse_qa_pairs(conversations)
    parse_elapsed = time.time() - parse_t0

    if output_path is None:
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        qa_pairs_file = output_dir / "qa_pairs.json"
    else:
        qa_pairs_file = Path(output_path)
        qa_pairs_file.parent.mkdir(parents=True, exist_ok=True)

    # 전처리 적용
    processed_pairs = []
    for pair in qa_pairs:
        # Question과 Answer 각각 전처리
        cleaned_question = preprocess_content(pair['question']) if preprocess_content else pair['question']
        cleaned_answer = preprocess_content(pair['answer']) if preprocess_content else pair['answer']
        
        # 전처리 후에도 최소 길이 체크
        if len(cleaned_question) < 5 or len(cleaned_answer) < 10:
            continue
        
        processed_pairs.append({
            "qa_id": pair['qa_id'],
            "conversation_id": pair['conversation_id'],
            "conversation_title": pair['conversation_title'],
            "question": cleaned_question,
            "answer": cleaned_answer,
            "qa_index": pair['qa_index'],
            "timestamp": pair.get('timestamp')
        })

    with open(qa_pairs_file, "w", encoding="utf-8") as f:
        json.dump(processed_pairs, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start

    print("=" * 80)
    print("Q-A Pairs JSON Builder")
    print("=" * 80)
    print(f"대화 로딩: {load_elapsed:.2f}초")
    print(f"Q-A 쌍 파싱: {parse_elapsed:.2f}초")
    print(f"총 소요 시간: {total_elapsed:.2f}초")
    print(f"총 Q-A 쌍 수: {len(processed_pairs)}")
    print(f"저장: {qa_pairs_file}")

    return str(qa_pairs_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build Q-A pairs JSON for embedding extraction")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    parser.add_argument("--data-path", type=str, default=None, help="원본 데이터 경로 (ex. data/conversations.json)")
    parser.add_argument("--output", type=str, default=None, help="생성할 qa_pairs.json 경로")
    args = parser.parse_args()

    build_qa_pairs(
        sample_size=args.sample_size,
        data_path=args.data_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
