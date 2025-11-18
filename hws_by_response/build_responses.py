import sys
import json
import time
import re
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
try:
    from preprocess import preprocess_content
except Exception:
    preprocess_content = None


def clean_text(text: str) -> str:
    """
    텍스트 전처리: 개행문자, 이모티콘, 코드 블록, 특수문자 등 제거

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    if preprocess_content is not None:
        return preprocess_content(text or "")
    # fallback: minimal cleanup if preprocess.py not available
    if not text:
        return ""
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_responses(sample_size: int = 50, data_path: str = None, output_path: str = None, compat_s1: bool = True) -> str:
    start = time.time()

    loader = ConversationLoader(data_path=data_path) if data_path else ConversationLoader()
    load_t0 = time.time()
    conversations = loader.load_sample(n=sample_size)
    load_elapsed = time.time() - load_t0

    parser = NoteParser(min_content_length=20)
    parse_t0 = time.time()
    ai_responses = parser.parse_ai_responses(conversations)
    parse_elapsed = time.time() - parse_t0

    if output_path is None:
        output_dir = project_root / "test" / "output"
        output_dir.mkdir(exist_ok=True)
        responses_file = output_dir / "responses.json"
    else:
        responses_file = Path(output_path)
        responses_file.parent.mkdir(parents=True, exist_ok=True)

    responses_dict = []
    for resp in ai_responses:
        # content 전처리 적용
        cleaned_content = clean_text(resp.content)

        d = {
            "response_id": resp.response_id,
            "conversation_id": resp.conversation_id,
            "conversation_title": resp.conversation_title,
            "message_index": resp.message_index,
            "content": cleaned_content,
        }
        if resp.timestamp is not None:
            d["timestamp"] = resp.timestamp
        responses_dict.append(d)

    with open(responses_file, "w", encoding="utf-8") as f:
        json.dump(responses_dict, f, ensure_ascii=False, indent=2)

    # keep backward-compat with existing scripts (optional)
    s1_file = None
    if compat_s1:
        s1_file = responses_file.parent / "s1_ai_responses.json"
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(responses_dict, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start

    print("=" * 80)
    print("Responses JSON Builder")
    print("=" * 80)
    print(f"대화 로딩: {load_elapsed:.2f}초")
    print(f"AI 답변 파싱: {parse_elapsed:.2f}초")
    print(f"총 소요 시간: {total_elapsed:.2f}초")
    print(f"총 답변 수: {len(responses_dict)}")
    print(f"저장: {responses_file}")
    if s1_file is not None:
        print(f"호환 파일: {s1_file}")

    return str(responses_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build responses JSON for downstream steps")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    parser.add_argument("--data-path", type=str, default=None, help="원본 데이터 경로 (ex. data/mock_data.json)")
    parser.add_argument("--output", type=str, default=None, help="생성할 responses.json 경로")
    parser.add_argument("--no-compat-s1", action="store_true", help="s1_ai_responses.json 호환 파일 저장 비활성화")
    args = parser.parse_args()

    build_responses(
        sample_size=args.sample_size,
        data_path=args.data_path,
        output_path=args.output,
        compat_s1=not args.no_compat_s1,
    )


if __name__ == "__main__":
    main()
 