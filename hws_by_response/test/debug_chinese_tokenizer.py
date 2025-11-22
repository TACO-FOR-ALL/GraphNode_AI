import sys
from pathlib import Path

# hws_by_response/test 기준 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import jieba  # type: ignore
import jieba.posseg as pseg  # type: ignore
import keyword_tokenizer as kt  # type: ignore
from tools.preprocess import preprocess_content  # type: ignore

TEST_STRINGS = [
    "的\u201c成长\u201d主题",        # 的“成长”主题
    "角色；李双双",               # 角色；李双双
    "英雄主义；李双双",           # 英雄主义；李双双
    "的\u201c传奇性\u201d特征",   # 的“传奇性”特征
    "其\u201c传奇性\u201d特征",   # 其“传奇性”特征
    "중국어是24式；太极拳",
    "是24式；太极拳",
    "是24式太极拳",
    "mark24",
    "mark是24式太极拳"
]


def debug_string(text: str) -> None:
    print("=" * 80)
    print(f"원문: {text}")

    cleaned = preprocess_content(text)
    print(f"[preprocess_content 결과]: {cleaned}")

    # 1) jieba.cut
    cut_tokens = list(jieba.cut(cleaned))
    print("[jieba.cut 결과]:", cut_tokens)

    # 2) jieba.posseg.cut
    pos_tokens = [(w, f) for w, f in pseg.cut(cleaned)]
    print("[jieba.posseg.cut 결과]:", pos_tokens)

    # 3) _tokenize_chinese만 직접 호출
    zh_tokens = kt._tokenize_chinese(cleaned)
    print("[_tokenize_chinese 결과]:", zh_tokens)

    # 4) multi_lang_tokenize (중국어 + 기타 언어 혼합 케이스 확인)
    ml_tokens = kt.multi_lang_tokenize(cleaned)
    print("[multi_lang_tokenize 결과]:", ml_tokens)


def main() -> None:
    print("Chinese tokenizer / POS debug (test/)\n")
    kt._load_jieba()
    kt._load_jieba_pos()
    kt._load_stopwords()

    for s in TEST_STRINGS:
        debug_string(s)
        print()


if __name__ == "__main__":
    main()
