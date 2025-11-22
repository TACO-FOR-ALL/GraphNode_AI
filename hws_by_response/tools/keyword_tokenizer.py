"""Multilingual tokenizer with Korean morphological analysis and stopword filtering.

- Korean: KoNLPy Okt for POS tagging + korean_stopwords.txt
- Chinese: jieba + cn_stopwords.txt
- English: NLTK stopwords.words("english")

Used by keyword extraction scripts to normalize tokens before Count/TF-IDF.
"""

from pathlib import Path
from typing import List
import re

# Lazy-loaded globals
_okt = None
_jieba = None
_jieba_pos = None
_en_stopwords = None
_ko_stopwords = None
_zh_stopwords = None
_ko_suffixes = None


def _load_stopwords() -> None:
    global _en_stopwords, _ko_stopwords, _zh_stopwords, _ko_suffixes
    if _en_stopwords is None:
        try:
            from nltk.corpus import stopwords

            _en_stopwords = set(stopwords.words("english"))
        except Exception:
            _en_stopwords = set()

    base = Path(__file__).parent.parent / "stop_words"

    if _ko_stopwords is None:
        ko_path = base / "korean_stopwords.txt"
        if ko_path.exists():
            _ko_stopwords = set(
                line.strip() for line in ko_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        else:
            _ko_stopwords = set()

    # Korean suffixes for fallback stripping
    if _ko_suffixes is None:
        suf_path = base / "korean_suffixes.txt"
        if suf_path.exists():
            _ko_suffixes = [
                line.strip()
                for line in suf_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            _ko_suffixes = []

    if _zh_stopwords is None:
        cn_path = base / "cn_stopwords.txt"
        if cn_path.exists():
            _zh_stopwords = set(
                line.strip() for line in cn_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        else:
            _zh_stopwords = set()


def _load_okt():
    global _okt
    if _okt is None:
        try:
            from konlpy.tag import Okt

            _okt = Okt()
        except Exception:
            _okt = None


def _load_jieba():
    global _jieba
    if _jieba is None:
        try:
            import jieba  # type: ignore

            _jieba = jieba
        except Exception:
            _jieba = None


def _load_jieba_pos():
    global _jieba_pos
    if _jieba_pos is None:
        try:
            import jieba.posseg as pseg  # type: ignore

            _jieba_pos = pseg
        except Exception:
            _jieba_pos = None


_KO_RE = re.compile(r"^[가-힣]+$")
_ZH_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_EN_RE = re.compile(r"^[A-Za-z]+$")


def _is_korean_token(token: str) -> bool:
    return bool(_KO_RE.match(token))


def _is_chinese_token(token: str) -> bool:
    return bool(_ZH_RE.match(token))


def _is_english_token(token: str) -> bool:
    return bool(_EN_RE.match(token))


def _tokenize_korean(text: str) -> List[str]:
    _load_okt()
    _load_stopwords()
    if not text:
        return []
    if _okt is None:
        # Fallback: simple split + 보수적인 조사/어미 스트립 + stopword 필터
        tokens: List[str] = []
        # 대표적인 조사/어미 suffix들: stop_words/korean_suffixes.txt 에서 로드
        common_suffixes = _ko_suffixes or []

        for tok in text.split():
            # 공백 + 기본 구두점 제거 후 suffix 처리 (ASCII/전각 느낌표 모두 제거)
            t = tok.strip(" ,.;:!?！")
            if not t:
                continue
            # 정확히 stopword에 있으면 버림
            if t in _ko_stopwords:
                continue

            base = t
            # 순수 한글 토큰에 대해서만 suffix 스트립 시도
            if _is_korean_token(t):
                for suf in sorted(common_suffixes, key=len, reverse=True):
                    if base.endswith(suf) and len(base) > len(suf):
                        cand = base[: -len(suf)]
                        # 뿌리가 비어있지 않고, stopword가 아니면 채택
                        if cand and cand not in _ko_stopwords:
                            base = cand
                        break

            if base:
                tokens.append(base)
        return tokens

    # 형태소 분석 + 품사 기반 필터
    keep_pos_prefix = ("N", "V", "SL", "SN")  # 명사/동사/수사/외래어 계열 위주
    tokens: List[str] = []
    for morph, tag in _okt.pos(text, stem=True):
        if morph in _ko_stopwords:
            continue
        if not any(tag.startswith(p) for p in keep_pos_prefix):
            continue
        morph = morph.strip()
        if not morph:
            continue
        tokens.append(morph)
    return tokens


def _tokenize_chinese(text: str) -> List[str]:
    _load_jieba()
    _load_jieba_pos()
    _load_stopwords()
    if not text:
        return []
    tokens: List[str] = []
    if _jieba is None and _jieba_pos is None:
        # Fallback: treat full string as one token if not stopword
        t = text.strip()
        if t and t not in _zh_stopwords:
            tokens.append(t)
        return tokens

    # 우선 posseg 기반 품사 태깅 사용 (가능한 경우)
    if _jieba_pos is not None:
        # 중국어 품사 태그 예: n, nr, ns, nt, nz, v, vn, a, an, m(수사), k(접미) 등
        keep_pos_prefix = ("n", "v", "a")
        seg: List[tuple[str, str]] = []
        for word, flag in _jieba_pos.cut(text):
            w = str(word).strip()
            f = str(flag)
            if not w:
                continue
            seg.append((w, f))

        i = 0
        n = len(seg)
        while i < n:
            w, f = seg[i]
            if not w or w in _zh_stopwords:
                i += 1
                continue

            # 패턴: 숫자(m) + 분류사/접미사(예: "式")를 하나의 토큰으로 병합 (예: "24"+"式" -> "24式")
            if f.startswith("m") and i + 1 < n:
                w2, f2 = seg[i + 1]
                if w2 and w2 not in _zh_stopwords and (f2.startswith("n") or f2 in ("k", "q")):
                    tokens.append(w + w2)
                    i += 2
                    continue

            # 기본: 명사/동사/형용사 계열만 유지
            if any(f.startswith(p) for p in keep_pos_prefix):
                tokens.append(w)

            i += 1

        return tokens

    # posseg 를 쓸 수 없으면 예전처럼 jieba.cut 기반으로만 동작
    for word in _jieba.cut(text):
        w = str(word).strip()
        if not w:
            continue
        if w in _zh_stopwords:
            continue
        tokens.append(w)
    return tokens


def _tokenize_english(text: str) -> List[str]:
    _load_stopwords()
    if not text:
        return []
    tokens: List[str] = []
    for raw in text.split():
        t = raw.strip()
        if not t:
            continue
        low = t.lower()
        if low in _en_stopwords:
            continue
        # 너무 짧은 한 글자 토큰은 버림 ("a", "b" 등)
        if len(low) == 1:
            continue
        tokens.append(low)
    return tokens


def multi_lang_tokenize(text: str) -> List[str]:
    """Tokenize mixed-language text into meaningful tokens with stopword filtering.

    - 한국어: 형태소 분석 + 품사/불용어 필터
    - 중국어: jieba + 불용어 필터
    - 영어: 공백 토큰화 + NLTK 불용어 필터
    - 그 외: 공백 기반 토큰 몇 개만 남김
    """
    if not text:
        return []

    _load_stopwords()

    tokens: List[str] = []

    # 1차: 공백 기준 rough split 후, 토큰별로 언어에 맞게 재토크나이즈
    rough = text.split()
    for chunk in rough:
        c = chunk.strip()
        if not c:
            continue

        # 순수 한글만으로 이뤄진 토큰이면 한국어 처리
        if _is_korean_token(c):
            tokens.extend(_tokenize_korean(c))
        # 순수 한자면 중국어 처리
        elif _is_chinese_token(c):
            tokens.extend(_tokenize_chinese(c))
        # 순수 알파벳이면 영어 처리
        elif _is_english_token(c):
            tokens.extend(_tokenize_english(c))
        else:
            # 혼합 토큰: 예) "backtesting은", "backtesting인지" 등
            # 1) 공백/구두점 제거 (ASCII/전각 느낌표 모두 제거)
            stripped = c.strip(' ,.;:!?！')
            if not stripped:
                continue

            # 구두점 제거 후 순수 한글/한자/알파벳이 되었다면, 해당 언어 토크나이저로 재위임
            if _is_korean_token(stripped):
                tokens.extend(_tokenize_korean(stripped))
                continue
            if _is_chinese_token(stripped):
                tokens.extend(_tokenize_chinese(stripped))
                continue
            if _is_english_token(stripped):
                tokens.extend(_tokenize_english(stripped))
                continue

            # 2) "임의의 root + 한글 tail" 패턴이면 tail을 ko/zh stopwords 기준으로 처리
            #    tail 이 ko/cn stopwords에 해당하면 tail은 버리고 root만 사용
            base = None
            # 앞에 뭐가 오든 마지막 연속 한글 덩어리를 tail로 본다
            m = re.match(r"^(.+?)([가-힣]+)$", stripped)
            if m:
                root, tail = m.group(1), m.group(2)
                _load_stopwords()
                # tail 이 ko/cn stopwords 라면 tail 제거
                if tail in (_ko_stopwords or set()) or tail in (_zh_stopwords or set()):
                    low_root = root.strip().lower()
                    if low_root and low_root not in (_en_stopwords or set()):
                        base = low_root
                else:
                    # tail 이 stopword 가 아니면 전체를 한 토큰으로 보되, 영어 stopword만 필터링
                    low_all = stripped.lower()
                    if low_all and low_all not in (_en_stopwords or set()):
                        base = low_all
            else:
                # 그 외 혼합 토큰은 그대로 쓰되, 영어 stopword만 필터링
                low_all = stripped.lower()
                if low_all and low_all not in (_en_stopwords or set()):
                    base = low_all

            if base:
                tokens.append(base)

    return tokens
