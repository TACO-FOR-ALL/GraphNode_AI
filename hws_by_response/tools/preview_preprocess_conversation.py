import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path so we can import analyze/* from tools/*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from analyze.parser import NoteParser
try:
    from tools.preprocess import preprocess_content
except Exception:
    preprocess_content = None


def load_conversations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Preview math/markup preprocessing for a single conversation")
    p.add_argument("--data-path", type=str, default="input_data/conversations.json", help="원본 conversations.json 경로")
    p.add_argument("--conversation-id", type=int, required=True, help="대상 conversation_id (1-based)")
    p.add_argument("--output", type=str, default=None, help="결과 저장 경로 (기본: output/tests/preprocess_{id}.json)")
    args = p.parse_args()

    data_path = Path(args.data_path)
    conversations = load_conversations(data_path)

    # Only parse the requested single conversation (1-based index)
    conv_id = int(args.conversation_id)
    if conv_id <= 0 or conv_id > len(conversations):
        print(f"대상 conversation_id 범위 오류: {conv_id} (총 {len(conversations)})")
        return

    single_conversations = [conversations[conv_id - 1]]

    parser = NoteParser(min_content_length=0)
    qa_pairs = parser.parse_qa_pairs(single_conversations)
    # We parsed only one conversation; use all pairs returned
    target_pairs = qa_pairs

    previews: List[Dict[str, Any]] = []
    for pair in target_pairs:
        q_raw = pair.get("question", "")
        a_raw = pair.get("answer", "")
        if preprocess_content is not None:
            q_clean = preprocess_content(q_raw or "")
            a_clean = preprocess_content(a_raw or "")
        else:
            q_clean = q_raw or ""
            a_clean = a_raw or ""
        previews.append(
            {
                "qa_id": pair.get("qa_id"),
                # Stamp the original requested conversation id for clarity
                "conversation_id": conv_id,
                "qa_index": pair.get("qa_index"),
                "question_raw": q_raw,
                "question_clean": q_clean,
                "answer_raw": a_raw,
                "answer_clean": a_clean,
            }
        )

    out_path = (
        Path(args.output)
        if args.output is not None
        else Path("output") / "tests" / f"preprocess_{conv_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"원본/전처리 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
