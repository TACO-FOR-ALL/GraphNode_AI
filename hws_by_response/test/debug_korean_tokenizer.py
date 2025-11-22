import sys
from pathlib import Path

# hws_by_response/test 기준 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

# tools 디렉토리를 import 경로에 추가
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import keyword_tokenizer as kt  # type: ignore


TEST_SENTENCES = [
    "조건입니다.",
    "의미입니다.",
    "값입니다.",
    "출력합니다.",
    "해석해보겠습니다.",
    # 혼합 한국어+영어 키워드 디버그용
    "지지집합support",
    "disjoint서로",
    "사라지는지vanishing",
    "비포화nonsaturating",
    "gradient는",
    "판별자discriminator",
    "판별자critic",
    "clipping보다",
]


def debug_sentence(text: str) -> None:
    print("=" * 80)
    print(f"원문: {text}")

    # 형태소 분석기/불용어 로더 준비
    kt._load_okt()
    kt._load_stopwords()

    # 1) 형태소 분석 결과 (Okt)
    if kt._okt is None:
        print("[Okt] 사용 불가 (konlpy 미설치 또는 로딩 실패) -> fallback 토크나이저 사용 중")
    else:
        morphs = kt._okt.pos(text, stem=True)
        print("[Okt.pos 결과]")
        print(morphs)

    # 2) multi_lang_tokenize 적용 결과 (불용어/접사 제거 후)
    tokens = kt.multi_lang_tokenize(text)
    print("[multi_lang_tokenize 결과]")
    print(tokens)


def main() -> None:
    print("Korean tokenizer / stopword debug (test/)\n")
    kt._load_okt()
    kt._load_stopwords()
    print(f"Okt available: {kt._okt is not None}")
    print(f"불용어 개수: {len(kt._ko_stopwords or [])}")
    print()

    for s in TEST_SENTENCES:
        debug_sentence(s)
        print()


if __name__ == "__main__":
    main()
