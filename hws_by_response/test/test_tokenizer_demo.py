import sys
from pathlib import Path

# tools/ 경로 추가
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from keyword_tokenizer import multi_lang_tokenize, _tokenize_korean  # type: ignore


def main() -> None:
    samples = [
        "레비아탄의 레비아탄은 레비아탄에서 레비아탄에서는",
        "엔지니어링에서는 엔지니어링에서 엔지니어링에서의 엔지니어링에서도",
        "backtesting은 backtesting인지 backtesting인지,",
        "파이썬에서는 backtesting은 금융에서 자주 쓰인다.",
    ]

    print("=== multi_lang_tokenize 테스트 ===")
    for s in samples:
        print(f"\n입력: {s}")
        print("토큰:", multi_lang_tokenize(s))

    print("\n=== _tokenize_korean fallback 직접 테스트 ===")
    # _okt 가 없으면 fallback 경로로 들어간다.
    ko_samples = [
        "레비아탄의",
        "레비아탄에서는",
        "엔지니어링에서는",
        "엔지니어링에서의",
        "엔지니어링에서도",
        "파이썬에서 파이썬의 조건문은"
    ]
    for s in ko_samples:
        print(f"\n입력: {s}")
        print("토큰:", _tokenize_korean(s))


if __name__ == "__main__":
    main()
