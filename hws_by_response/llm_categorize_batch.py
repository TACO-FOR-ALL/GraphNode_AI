import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse
import math

from dotenv import load_dotenv

# OpenAI client (use the official openai package)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def load_keywords(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_keywords_per_conversation(items: List[Dict[str, Any]], top_n_per_response: int = 5) -> tuple[Dict[Any, List[str]], Dict[Any, str]]:
    conv_kw: Dict[Any, List[str]] = {}
    conv_titles: Dict[Any, str] = {}

    for it in items:
        cid = it.get('conversation_id')
        kws = it.get('keywords', [])

        # Store conversation title (first occurrence wins)
        if cid not in conv_titles and 'conversation_title' in it:
            conv_titles[cid] = it['conversation_title']

        # take top-N per response to avoid overwhelming the prompt
        picked = [k.get('keyword') for k in kws[:top_n_per_response] if isinstance(k, dict) and 'keyword' in k]
        if not picked:
            continue
        conv_kw.setdefault(cid, []).extend(picked)

    # de-duplicate and keep order per conversation
    for cid, arr in conv_kw.items():
        seen = set()
        deduped = []
        for k in arr:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        conv_kw[cid] = deduped

    return conv_kw, conv_titles


def load_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def call_llm(system_prompt: str, user_prompt: str, model: str, provider: str = 'openai', json_mode: bool = True) -> str:
    if provider != 'openai':
        raise ValueError(f"Unsupported provider: {provider}")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install with: pip install openai python-dotenv")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Create a .env with OPENAI_API_KEY=... or export it.")

    client = OpenAI(api_key=api_key)

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 1.0,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def define_categories(conv_keywords: Dict[Any, List[str]], prompt_template: str, model: str, provider: str = 'openai') -> tuple[List[str], str]:
    """
    STEP 1: Define 3-5 categories based on ALL keywords from ALL conversations
    """
    # Create a representation of ALL keywords from ALL conversations
    all_keywords_data = {}
    for cid, kws in conv_keywords.items():
        all_keywords_data[str(cid)] = kws[:15]  # Top 15 keywords per conversation

    # Use the loaded prompt template as system prompt
    system_prompt = prompt_template

    # User prompt contains the actual data
    user_prompt = f"""[입력 데이터: 대화별 키워드]
{json.dumps(all_keywords_data, ensure_ascii=False, indent=2)}"""

    print(f"[STEP 1] 전체 {len(conv_keywords)}개 대화의 키워드로 카테고리 정의 중...")
    content = call_llm(system_prompt, user_prompt, model=model, provider=provider, json_mode=True)

    try:
        data = json.loads(content)
        categories = data.get('categories', [])
        if not (3 <= len(categories) <= 8):
            raise ValueError(f"Expected 3-8 categories, got {len(categories)}")
        print(f"[STEP 1] 정의된 카테고리: {categories}")
        print(f"[STEP 1] 이유: {data.get('reason', 'N/A')}")
        return categories, data.get('reason', '')
    except Exception as e:
        raise RuntimeError(f"Failed to parse category definition response: {content}\nError: {e}")


def assign_conversations_in_batches(
    conv_keywords: Dict[Any, List[str]],
    conv_titles: Dict[Any, str],
    categories: List[str],
    batch_size: int,
    model: str,
    provider: str = 'openai'
) -> Dict[str, Dict[str, Any]]:
    """
    STEP 2: Assign conversations to categories in batches
    """
    categories_block = json.dumps(categories, ensure_ascii=False, indent=2)
    system_prompt = (
        f"You are a highly skilled data scientist. You have defined {len(categories)} categories.\n\n"
        f"The categories are:\n{categories_block}\n\n"
        "Your task:\n"
        "1. Review the keywords for each conversation in the batch\n"
        "2. Assign EACH conversation to exactly ONE category that best fits\n"
        "3. Provide a brief reason for each assignment\n\n"
        "Return JSON only in this format:\n"
        "{{\n"
        "  \"assignments\": {{\n"
        "    \"conversation_id\": {{\n"
        "      \"category\": \"category name\",\n"
        "      \"reason\": \"brief explanation\"\n"
        "    }},\n"
        "    ...\n"
        "  }}\n"
        "}}"
    )

    conv_ids = list(conv_keywords.keys())
    total_conversations = len(conv_ids)
    num_batches = math.ceil(total_conversations / batch_size)

    all_assignments = {}

    print(f"\n[STEP 2] 총 {total_conversations}개 대화를 {num_batches}개 배치로 분류 중...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_conversations)
        batch_conv_ids = conv_ids[start_idx:end_idx]

        # Prepare batch data
        batch_data = {}
        for cid in batch_conv_ids:
            title = conv_titles.get(cid, f"Conversation {cid}")
            keywords = conv_keywords[cid][:20]  # Top 20 keywords per conversation
            batch_data[str(cid)] = {
                "title": title,
                "keywords": keywords
            }

        user_prompt = f"""Assign these conversations to the most suitable categories:

{json.dumps(batch_data, ensure_ascii=False, indent=2)}

Available categories: {', '.join(categories)}

For each conversation, choose exactly ONE category and explain why."""

        print(f"  배치 {batch_idx + 1}/{num_batches} 처리 중 (대화 {start_idx + 1}-{end_idx})... ", end='', flush=True)

        try:
            content = call_llm(system_prompt, user_prompt, model=model, provider=provider, json_mode=True)
            data = json.loads(content)
            batch_assignments = data.get('assignments', {})

            # Validate and add assignments
            for conv_id, assignment in batch_assignments.items():
                category = assignment.get('category')
                reason = assignment.get('reason', '')

                # Validate category
                if category not in categories:
                    print(f"\n    경고: 대화 {conv_id}의 카테고리 '{category}'가 잘못됨, 첫 번째 카테고리로 대체")
                    category = categories[0]

                all_assignments[conv_id] = {
                    "category": category,
                    "reason": reason
                }

            print(f"완료 ({len(batch_assignments)}개)")

        except Exception as e:
            print(f"실패: {e}")
            # Fallback: assign all in batch to first category
            for cid in batch_conv_ids:
                all_assignments[str(cid)] = {
                    "category": categories[0],
                    "reason": f"Batch assignment failed: {e}"
                }

    print(f"\n[STEP 2] 완료: {len(all_assignments)}개 대화 분류됨")
    return all_assignments


def main():
    project_root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="LLM-based batch categorization using extracted keywords")
    parser.add_argument("--keywords-input", type=str, default=str(project_root / 'output' / 's2_ai_responses_with_keywords.json'), help="키워드 JSON 경로")
    parser.add_argument("--prompt-file", type=str, default=str(project_root / 'llm_clustering_prompt.txt'), help="프롬프트 템플릿 파일 경로")
    parser.add_argument("--output", type=str, default=str(project_root / 'output' / 's6_categories.json'), help="대분류 결과 JSON 출력 경로")
    parser.add_argument("--assignments-output", type=str, default=str(project_root / 'output' / 's6_categories_assignments.json'), help="assignments 전용 매핑 JSON 출력 경로")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM 모델명 (예: gpt-4o)")
    parser.add_argument("--provider", type=str, default="openai", help="LLM 제공자 (현재 openai만 지원)")
    parser.add_argument("--top-n-per-response", type=int, default=5, help="응답당 사용할 상위 키워드 수")
    parser.add_argument("--batch-size", type=int, default=50, help="한 번에 처리할 대화 수")
    parser.add_argument("--raw-output", type=str, default=str(project_root / 'output' / 's6_categories_raw_batch.txt'), help="요약 정보 저장 경로")
    args = parser.parse_args()

    # .env 로드
    load_dotenv()

    # 프롬프트 템플릿 로드
    print("프롬프트 템플릿 로드 중...")
    prompt_template = load_prompt(args.prompt_file)

    # 데이터 로드 및 집계
    print("데이터 로드 중...")
    kw_items = load_keywords(args.keywords_input)
    conv_keywords, conv_titles = aggregate_keywords_per_conversation(kw_items, top_n_per_response=args.top_n_per_response)
    print(f"총 {len(conv_keywords)}개 대화 발견")

    # STEP 1: Define categories
    categories, reason = define_categories(conv_keywords, prompt_template, model=args.model, provider=args.provider)

    # STEP 2: Assign conversations in batches
    assignments = assign_conversations_in_batches(
        conv_keywords,
        conv_titles,
        categories,
        batch_size=args.batch_size,
        model=args.model,
        provider=args.provider
    )

    # Combine results
    result = {
        "categories": categories,
        "assignments": assignments,
        "reason": reason
    }

    # 저장 (JSON 결과)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # assignments-only 저장 (conversation_title 포함)
    enriched_assignments = {}
    for conv_id, assignment in assignments.items():
        enriched_assignment = assignment.copy() if isinstance(assignment, dict) else {"category": assignment}
        enriched_assignment['conversation_title'] = conv_titles.get(int(conv_id) if conv_id.isdigit() else conv_id, f"Conversation {conv_id}")
        enriched_assignments[conv_id] = enriched_assignment

    assign_path = Path(args.assignments_output)
    assign_path.parent.mkdir(parents=True, exist_ok=True)
    with open(assign_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_assignments, f, ensure_ascii=False, indent=2)

    # 결과 요약 저장
    summary = {
        "total_conversations": len(conv_keywords),
        "categories": categories,
        "reason": reason,
        "category_counts": {}
    }
    for cat in categories:
        count = sum(1 for a in assignments.values() if a.get('category') == cat)
        summary["category_counts"][cat] = count

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[완료] 대분류 결과 저장: {out_path}")
    print(f"[완료] 카테고리 매핑 저장: {assign_path}")
    print(f"[완료] 요약 정보 저장: {raw_path}")
    print(f"\n카테고리별 대화 수:")
    for cat, count in summary["category_counts"].items():
        print(f"  - {cat}: {count}개")


if __name__ == "__main__":
    main()