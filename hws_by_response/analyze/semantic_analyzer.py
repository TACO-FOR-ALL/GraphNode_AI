"""
의미 분석 모듈 (S2)
SBERT를 사용한 의미 벡터화 및 핵심 개념 추출
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm
from .parser import Note, AIResponse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("경고: requests가 설치되어 있지 않습니다. DBpedia Spotlight 기능을 사용할 수 없습니다.")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("경고: KeyBERT가 설치되어 있지 않습니다. KeyBERT 기반 키워드 추출을 사용할 수 없습니다.")


class SemanticAnalyzer:
    """의미 분석 클래스"""

    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask",
        use_keybert: bool = True,
        use_dbpedia: bool = False,
        dbpedia_url: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Args:
            model_name: SBERT 모델 이름
            use_keybert: KeyBERT를 사용한 키워드 추출 여부
            use_dbpedia: DBpedia Spotlight 사용 여부
            dbpedia_url: DBpedia Spotlight API URL
            cache_folder: 모델 캐시 폴더 (None이면 프로젝트 내 models/ 사용)
        """
        # 캐시 폴더 설정
        if cache_folder is None:
            from pathlib import Path
            cache_folder = str(Path(__file__).parent.parent / "models")
            Path(cache_folder).mkdir(parents=True, exist_ok=True)

        # 모델이 이미 캐시에 있는지 확인
        from pathlib import Path
        # HuggingFace 캐싱 형식: models--{org}--{name}
        model_path = Path(cache_folder) / f"models--{model_name.replace('/', '--')}"
        if model_path.exists():
            print(f"✓ 캐시된 모델 사용: {model_name}")
            print(f"  경로: {model_path}")
        else:
            print(f"SBERT 모델 다운로드 중: {model_name}")
            print(f"  저장 경로: {cache_folder}")
            print("  (처음 실행 시 시간이 걸립니다. 다음부터는 캐시를 사용합니다.)")

        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        print("✓ SBERT 모델 로딩 완료")

        self.use_keybert = use_keybert and KEYBERT_AVAILABLE
        self.use_dbpedia = use_dbpedia and REQUESTS_AVAILABLE
        self.dbpedia_url = dbpedia_url
        self.stopwords = set()

        # 한국어 불용어 로드
        self._load_korean_stopwords()

        # KeyBERT 초기화 (기존 SBERT 모델 재사용)
        if self.use_keybert:
            try:
                self.keybert = KeyBERT(model=self.model)
                print("✓ KeyBERT 초기화 완료 (SBERT 모델 재사용)")
            except Exception as e:
                print(f"경고: KeyBERT 초기화 실패 - {e}")
                self.use_keybert = False

    def analyze_notes(self, notes: List[Note]) -> Dict[int, Dict[str, Any]]:
        """
        노트 리스트를 분석하여 의미 벡터와 핵심 개념 추출

        Args:
            notes: Note 객체 리스트

        Returns:
            {
                note_id: {
                    'embedding': np.ndarray,
                    'concepts': List[str]
                }
            }
        """
        print(f"\n총 {len(notes)}개의 노트 분석 중...")

        results = {}

        for note in notes:
            try:
                # 1. 의미 벡터 생성
                embedding = self._create_embedding(note.content)

                # 2. 핵심 개념 추출
                concepts = self._extract_concepts(note.content)

                results[note.note_id] = {
                    'embedding': embedding,
                    'concepts': concepts,
                    'title': note.title
                }

            except Exception as e:
                print(f"경고: 노트 {note.note_id} 분석 실패 - {e}")
                continue

        print(f"총 {len(results)}개의 노트 분석 완료")
        return results

    def _create_embedding(self, text: str) -> np.ndarray:
        """
        텍스트를 의미 벡터로 변환

        Args:
            text: 입력 텍스트

        Returns:
            의미 벡터 (numpy array)
        """
        # SBERT는 긴 텍스트를 처리할 수 있지만, 너무 길면 잘라냄
        max_length = 512 * 5  # 약 2500 토큰
        if len(text) > max_length:
            text = text[:max_length]

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def _extract_concepts(self, text: str) -> List[str]:
        """
        텍스트에서 핵심 개념 추출

        Args:
            text: 입력 텍스트

        Returns:
            핵심 개념 리스트
        """
        concepts = set()

        # 1. KeyBERT 사용 (최우선 - 가장 정확)
        if self.use_keybert:
            keybert_concepts = self._extract_keybert_concepts(text)
            concepts.update(keybert_concepts)

        # 2. 기본 키워드 추출 (폴백)
        # basic_concepts = self._extract_basic_keywords(text)
        # concepts.update(basic_concepts)

        # 3. DBpedia Spotlight 사용 (선택적)
        # if self.use_dbpedia and self.dbpedia_url:
        #     dbpedia_concepts = self._extract_dbpedia_concepts(text)
        #     concepts.update(dbpedia_concepts)

        return list(concepts)[:50]  # 최대 50개로 제한

    # def _extract_basic_keywords(self, text: str) -> List[str]:
    #     """
    #     기본 키워드 추출 (정규식 기반)

    #     Args:
    #         text: 입력 텍스트

    #     Returns:
    #         키워드 리스트
    #     """
    #     # 역할 표시 제거
    #     text = re.sub(r'\[(사용자|어시스턴트)\]\s*', '', text)

    #     # 영어 단어 추출 (대문자 시작, 3글자 이상)
    #     english_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)

    #     # 한글 단어 추출 (2글자 이상)
    #     korean_terms = re.findall(r'[가-힣]{2,}', text)

    #     # 빈도수 계산하여 상위 키워드 선택
    #     from collections import Counter
    #     all_terms = english_terms + korean_terms
    #     counter = Counter(all_terms)

    #     # 불용어 사전 사용하여 필터링
    #     keywords = [
    #         word for word, count in counter.most_common(30)
    #         if word not in self.stopwords and count >= 2
    #     ]

    #     return keywords

    def _extract_dbpedia_concepts(self, text: str) -> List[str]:
        """
        DBpedia Spotlight를 사용한 개념 추출

        Args:
            text: 입력 텍스트

        Returns:
            개념 리스트
        """
        if not self.use_dbpedia or not REQUESTS_AVAILABLE:
            return []

        try:
            # 텍스트가 너무 길면 앞부분만 사용
            max_length = 1000
            if len(text) > max_length:
                text = text[:max_length]

            response = requests.post(
                self.dbpedia_url,
                data={'text': text, 'confidence': 0.3},
                headers={'Accept': 'application/json'},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                resources = data.get('Resources', [])
                concepts = [r.get('@surfaceForm') for r in resources if '@surfaceForm' in r]
                return concepts[:20]

        except Exception as e:
            print(f"경고: DBpedia Spotlight 요청 실패 - {e}")

        return []

    def compute_similarity_matrix(self, embeddings_dict: Dict[int, np.ndarray]) -> Dict[tuple, float]:
        """
        모든 노트 간 코사인 유사도 계산

        Args:
            embeddings_dict: {note_id: embedding} 형태의 dict

        Returns:
            {(note_id_1, note_id_2): similarity_score} 형태의 dict
        """
        print("\n유사도 매트릭스 계산 중...")

        note_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[nid] for nid in note_ids])

        # 코사인 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)

        # dict로 변환 (중복 제거, 자기 자신 제외)
        similarities = {}
        n = len(note_ids)

        for i in range(n):
            for j in range(i + 1, n):
                note_id_1 = note_ids[i]
                note_id_2 = note_ids[j]
                similarity = float(similarity_matrix[i, j])
                similarities[(note_id_1, note_id_2)] = similarity

        print(f"총 {len(similarities)}개의 유사도 쌍 계산 완료")
        return similarities

    def _extract_keybert_concepts(self, text: str) -> List[str]:
        """
        KeyBERT를 사용한 키워드 추출

        Args:
            text: 입력 텍스트

        Returns:
            키워드 리스트
        """
        if not self.use_keybert:
            return []

        try:
            # 텍스트가 너무 길면 일부만 처리
            max_length = 5000
            if len(text) > max_length:
                text = text[:max_length]

            # KeyBERT로 키워드 추출 (최적화된 설정)
            # keyphrase_ngram_range: (1, 2) = 1~2 단어 조합
            # stop_words: None (우리가 이미 필터링함)
            # top_n: 상위 10개
            # nr_candidates를 줄여서 속도 향상
            keywords = self.keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words=None,
                top_n=10,
                use_mmr=True,  # MMR로 다양성 확보 (maxsum보다 빠름)
                diversity=0.5,
                nr_candidates=20  # 50 -> 20으로 줄임 (2.5배 빨라짐)
            )

            # (keyword, score) 튜플에서 keyword만 추출
            # 불용어 제외
            concepts = [
                kw for kw, score in keywords
                if kw not in self.stopwords and len(kw) >= 2
            ]

            return concepts[:10]  # 최대 10개

        except Exception as e:
            print(f"경고: KeyBERT 처리 실패 - {e}")
            return []

    def _load_korean_stopwords(self):
        """한국어 불용어 사전 로드"""
        from pathlib import Path
        stopwords_file = Path(__file__).parent / "korean_stopwords.txt"

        if stopwords_file.exists():
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 주석과 빈 줄 제외
                    if line and not line.startswith('#'):
                        self.stopwords.add(line)
            print(f"✓ 한국어 불용어 {len(self.stopwords)}개 로드")
        else:
            print(f"경고: 한국어 불용어 파일을 찾을 수 없습니다: {stopwords_file}")

    def analyze_ai_responses(self, ai_responses: List[AIResponse]) -> Dict[str, Dict[str, Any]]:
        """
        AI 답변 리스트를 분석하여 각각의 임베딩 벡터와 개념 추출

        각 AI 답변마다 개별 임베딩 벡터를 추출합니다.
        이는 CLAUDE.md의 워크플로우에 따라, 노이즈가 많은 사용자 질문 대신
        잘 정제된 AI 답변만을 사용하기 위함입니다.

        Args:
            ai_responses: AIResponse 객체 리스트

        Returns:
            {
                response_id: {
                    'embedding': np.ndarray,
                    'concepts': List[str],
                    'conversation_id': int,
                    'conversation_title': str,
                    'message_index': int
                }
            }
        """
        print(f"\n총 {len(ai_responses)}개의 AI 답변 분석 중...")

        results = {}

        # Step 1: 배치로 임베딩 생성 (훨씬 빠름)
        print("1. 임베딩 배치 생성 중...")
        texts = [resp.content for resp in ai_responses]
        embeddings = self.create_embeddings_batch(texts)

        # Step 2: 개념 추출 (필요한 경우에만)
        if self.use_keybert:
            print("2. 개념 추출 중...")
            progress_bar = tqdm(enumerate(ai_responses), total=len(ai_responses), desc="개념 추출")
        else:
            progress_bar = enumerate(ai_responses)

        for idx, response in progress_bar:
            try:
                # 이미 생성된 임베딩 사용
                embedding = embeddings[idx]

                # 핵심 개념 추출 (KeyBERT가 활성화된 경우에만)
                concepts = self._extract_concepts(response.content) if self.use_keybert else []

                results[response.response_id] = {
                    'embedding': embedding,
                    'concepts': concepts,
                    'conversation_id': response.conversation_id,
                    'conversation_title': response.conversation_title,
                    'message_index': response.message_index
                }

            except Exception as e:
                print(f"\n경고: AI 답변 {response.response_id} 분석 실패 - {e}")
                continue

        print(f"\n✓ 총 {len(results)}개의 AI 답변 분석 완료")
        return results

    def create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        여러 텍스트를 한 번에 임베딩 (배치 처리로 효율성 향상)

        Args:
            texts: 텍스트 리스트

        Returns:
            임베딩 행렬 (shape: [len(texts), embedding_dim])
        """
        print(f"{len(texts)}개의 텍스트를 배치 임베딩 중...")

        # 긴 텍스트 자르기
        max_length = 512 * 5
        processed_texts = [text[:max_length] if len(text) > max_length else text for text in texts]

        # 배치로 인코딩 (더 빠름)
        embeddings = self.model.encode(
            processed_texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True
        )

        print(f"✓ 배치 임베딩 완료")
        return embeddings
