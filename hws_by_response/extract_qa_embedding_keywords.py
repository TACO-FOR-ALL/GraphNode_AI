import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Set

# hws_by_response is the project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure local imports work when running from hws_by_response/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "tools"))

from tools.run_conv_pipeline import run_pipeline  # noqa: E402


def resolve_qa_pairs(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


essential_outputs = {
    "keywords": PROJECT_ROOT / "output" / "keywords",
    "embeddings": PROJECT_ROOT / "output" / "embeddings",
    "clusters": PROJECT_ROOT / "output" / "cluster_results",
}


def load_all_conversation_ids(qa_pairs_path: Path) -> List[int]:
    data = json.loads(qa_pairs_path.read_text(encoding="utf-8"))
    ids: Set[int] = set()
    for rec in data:
        cid = rec.get("conversation_id")
        if isinstance(cid, int):
            ids.add(cid)
        else:
            try:
                ids.add(int(cid))
            except Exception:
                continue
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the QA keyword+clustering+pooling pipeline for all conversations in qa_pairs.json (tools/)"
    )
    parser.add_argument("--qa-pairs", type=str, default=str(PROJECT_ROOT / "output" / "qa_pairs.json"))
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated conversation IDs to include. If empty, run all.",
    )
    parser.add_argument("--start", type=int, default=None, help="Start conversation_id (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End conversation_id (inclusive)")

    args = parser.parse_args()

    # Ensure output folders exist
    for p in essential_outputs.values():
        p.mkdir(parents=True, exist_ok=True)

    qa_pairs_path = resolve_qa_pairs(args.qa_pairs)
    if not qa_pairs_path.exists():
        raise FileNotFoundError(f"qa_pairs.json not found: {qa_pairs_path}")

    if args.include:
        ids: List[int] = []
        for x in args.include.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                ids.append(int(x))
            except ValueError:
                continue
        conv_ids = sorted(set(ids))
    else:
        conv_ids = load_all_conversation_ids(qa_pairs_path)
        # Apply start/end range if provided
        if args.start is not None or args.end is not None:
            s = args.start if args.start is not None else min(conv_ids) if conv_ids else 0
            e = args.end if args.end is not None else max(conv_ids) if conv_ids else -1
            conv_ids = [cid for cid in conv_ids if s <= cid <= e]

    print(f"Found {len(conv_ids)} conversations to process: {conv_ids}")

    t_total = time.time()
    for i, conv_id in enumerate(conv_ids, 1):
        print("=" * 80)
        print(f"[{i}/{len(conv_ids)}] Running pipeline for conversation_id={conv_id}")
        t0 = time.time()
        run_pipeline(
            conversation_id=conv_id,
            qa_pairs_path=qa_pairs_path,
            model_name=args.model,
            ngram_max=args.ngram_max,
            max_candidates=args.max_candidates,
            top_n=args.top_n,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric=args.metric,
        )
        print(f"conversation_id={conv_id} done in {time.time() - t0:.2f}s")

    print("=" * 80)
    print(f"All done in {time.time() - t_total:.2f}s for {len(conv_ids)} conversations")


if __name__ == "__main__":
    main()
