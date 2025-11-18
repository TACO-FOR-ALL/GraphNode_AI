"""
임베딩 처리 모듈
BERTopic을 사용한 AI 답변 클러스터링 및 문서별 풀링 기능 제공
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import re

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("경고: BERTopic이 설치되어 있지 않습니다. pip install bertopic")



@dataclass
class ClusteredResponse:
    """클러스터링된 AI 답변"""
    response_id: str
    conversation_id: int
    embedding: np.ndarray
    topic_id: int  # BERTopic이 할당한 토픽 ID (-1은 outlier)
    topic_keywords: List[str]  # 해당 토픽의 대표 키워드


@dataclass
class DocumentEmbedding:
    """문서(대화)의 주제별 임베딩"""
    conversation_id: int
    conversation_title: str
    topic_embeddings: Dict[int, np.ndarray]  # {topic_id: pooled_embedding}
    topic_keywords: Dict[int, List[str]]  # {topic_id: keywords}
    response_count: int  # 해당 대화의 AI 답변 개수


class EmbeddingProcessor:
    """
    임베딩 처리 클래스 - BERTopic을 사용한 클러스터링 및 풀링

    CLAUDE.md 워크플로우:
    1. SentenceTransformer로 모든 AI 답변 임베딩 생성
    2. BERTopic에 임베딩 전달하여 클러스터링
    3. BERTopic이 클러스터 ID(토픽 ID)와 대표 키워드 생성
    4. 대화별로 토픽 ID별 풀링
    5. 유사도 계산
    """

    def __init__(
        self,
        n_clusters: int = 10,
        min_topic_size: int = 5,
        nr_topics: Optional[int] = None,
        language: str = "multilingual",
        verbose: bool = True,
        use_kmeans: bool = True,
        multilingual_mixed: bool = True
    ):
        """
        Args:
            n_clusters: K-Means 클러스터 개수 (use_kmeans=True일 때)
            min_topic_size: 최소 토픽 크기 (HDBSCAN 사용 시)
            nr_topics: 토픽 개수 (None이면 자동 결정)
            language: 언어 설정 ("multilingual", "korean" 등)
            verbose: 진행상황 출력 여부
            use_kmeans: K-Means 사용 여부 (True면 HDBSCAN 대신 K-Means 사용)
            multilingual_mixed: True면 한국어/중국어/영어 혼합 응답 처리 (문자 기반 n-gram)
        """
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic을 설치해주세요: pip install bertopic")

        self.n_clusters = n_clusters
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.language = language
        self.verbose = verbose
        self.use_kmeans = use_kmeans
        self.multilingual_mixed = multilingual_mixed
        self.topic_model = None
        self.topic_info = None

    def _tokenize_multilingual(self, text: str) -> List[str]:
        """
        한글, 중국어, 영문 혼합 텍스트 토크나이징
        Java 없이 정규식 기반으로 작동
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토크나이징된 단어 리스트
        """
        if not text:
            return []
        
        tokens = []
        
        # 1. 한글 토크나이징 (정규식 기반 - Java 불필요)
        # 한글 단어 추출 (2글자 이상)
        korean_words = re.findall(r'[\uac00-\ud7af]{2,}', text)
        tokens.extend([w.lower() for w in korean_words])
        
        # 2. 중국어 토크나이징 (정규식 기반)
        # 중국어 문자 추출 (2글자 이상)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        tokens.extend([c.lower() for c in chinese_chars])
        
        # 3. 영문 토크나이징 (정규식 기반)
        # 3글자 이상 영문 단어만 추출
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        tokens.extend([w.lower() for w in english_words])
        
        # 4. 숫자 제거 및 중복 제거
        tokens = [t for t in tokens if not t.isdigit()]
        tokens = list(dict.fromkeys(tokens))  # 순서 유지하며 중복 제거
        
        return tokens

    def _merge_similar_keywords(self, keywords: List[str]) -> List[str]:
        """
        유사한 키워드 병합 (예: pip, pip37, pip pip -> pip)
        
        Args:
            keywords: 원본 키워드 리스트
            
        Returns:
            병합된 키워드 리스트
        """
        if not keywords:
            return keywords
        
        # 1. 중복 제거 (정확히 같은 것)
        seen = set()
        unique_kw = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_kw.append(kw)
        
        # 2. 부분 문자열 제거 (예: "pip"가 있으면 "pip37", "pip pip" 제거)
        merged = []
        for kw in unique_kw:
            # 다른 키워드에 포함되지 않으면 추가
            is_substring = False
            for other_kw in unique_kw:
                if kw != other_kw and kw in other_kw:
                    is_substring = True
                    break
            if not is_substring:
                merged.append(kw)
        
        # 3. 공백 제거 (예: "pip pip" -> "pip")
        final = []
        for kw in merged:
            # 공백으로 분리된 단어들이 모두 같으면 하나로 통일
            words = kw.split()
            if len(set(words)) == 1:  # 모든 단어가 같음
                final.append(words[0])
            else:
                final.append(kw)
        
        # 4. 최종 중복 제거
        final = list(dict.fromkeys(final))  # 순서 유지하며 중복 제거
        
        return final

    def cluster_with_bertopic(
        self,
        response_embeddings: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, ClusteredResponse], Dict[int, List[str]]]:
        """
        BERTopic을 사용하여 모든 AI 답변 임베딩을 클러스터링

        CLAUDE.md 워크플로우 step 2:
        "[출력 1] SentenceTransformer로 모든 AI 답변의 임베딩 벡터 생성
        [출력 2] BERTopic이 각 클러스터 ID의 대표 키워드(토픽) 추출
        [출력 3] BERTopic이 각 문서에 클러스터 ID(토픽 ID) 할당"

        Args:
            response_embeddings: {
                response_id: {
                    'embedding': np.ndarray,
                    'conversation_id': int,
                    'content': str,
                    ...
                }
            }

        Returns:
            (
                {response_id: ClusteredResponse},
                {topic_id: [keywords]}
            )
        """
        print(f"\n{'='*80}")
        print(f"BERTopic을 사용한 AI 답변 클러스터링")
        print(f"{'='*80}")
        print(f"총 {len(response_embeddings)}개의 AI 답변 처리 중...")

        # 데이터 준비
        response_ids = list(response_embeddings.keys())
        embeddings = np.array([
            response_embeddings[rid]['embedding']
            for rid in response_ids
        ])
        documents = [
            response_embeddings[rid].get('content', '')
            for rid in response_ids
        ]

        print(f"임베딩 shape: {embeddings.shape}")

        # BERTopic 모델 초기화 및 학습
        print("\nBERTopic 모델 학습 중...")

        # 다국어 혼합 응답 처리: 한글/중국어/영문 토크나이저 + 중복 제거
        vectorizer_model = None
        if self.multilingual_mixed:
            from sklearn.feature_extraction.text import CountVectorizer
            print("다국어 혼합 응답 처리 활성화 (한글/중국어/영문 토크나이저)")
            
            # 커스텀 토크나이저 함수
            def multilingual_tokenizer(text):
                return self._tokenize_multilingual(text)
            
            # 단어 기반 분석: 한글, 중국어, 영문 모두 처리 가능
            vectorizer_model = CountVectorizer(
                analyzer='word',
                tokenizer=multilingual_tokenizer,  # 커스텀 토크나이저 사용
                lowercase=True,
                stop_words=None,
                max_features=1000,
                min_df=2,
                max_df=0.8,
            )

        # K-Means 사용 (HDBSCAN 대신)
        if self.use_kmeans:
            from sklearn.cluster import KMeans
            from bertopic.cluster import BaseCluster

            # K-Means 래퍼 클래스
            class KMeansCluster(BaseCluster):
                def __init__(self, n_clusters):
                    self.n_clusters = n_clusters
                    self.model = KMeans(n_clusters=n_clusters, random_state=42)

                def fit(self, X):
                    self.model.fit(X)
                    return self

                def predict(self, X):
                    return self.model.predict(X)

            print(f"K-Means 클러스터링 사용 (k={self.n_clusters})")
            cluster_model = KMeansCluster(n_clusters=self.n_clusters)

            self.topic_model = BERTopic(
                hdbscan_model=cluster_model,
                vectorizer_model=vectorizer_model,
                nr_topics=self.nr_topics,
                language=self.language,
                verbose=self.verbose,
                calculate_probabilities=False
            )
        else:
            # HDBSCAN 사용 (기본값)
            self.topic_model = BERTopic(
                min_topic_size=self.min_topic_size,
                vectorizer_model=vectorizer_model,
                nr_topics=self.nr_topics,
                language=self.language,
                verbose=self.verbose,
                calculate_probabilities=False
            )

        # 사전 계산된 임베딩으로 학습
        # BERTopic의 fit_transform 메서드에 embeddings 전달
        topics, probabilities = self.topic_model.fit_transform(documents, embeddings)

        # 토픽 정보 추출
        self.topic_info = self.topic_model.get_topic_info()
        print(f"\n✓ 클러스터링 완료!")
        print(f"  생성된 토픽 수: {len(self.topic_info) - 1}")  # -1은 outlier 토픽 제외
        print(f"  Outlier 문서: {sum(1 for t in topics if t == -1)}개")

        # 토픽별 키워드 추출 + 중복 제거
        topic_keywords_map = {}
        for topic_id in set(topics):
            if topic_id == -1:
                topic_keywords_map[-1] = ["outlier"]
            else:
                # BERTopic에서 상위 10개 키워드 추출
                topic_words = self.topic_model.get_topic(topic_id)
                if topic_words:
                    # (word, score) 튜플에서 word만 추출
                    keywords = [word for word, score in topic_words[:10]]
                    # 중복/유사 키워드 병합
                    keywords = self._merge_similar_keywords(keywords)
                    topic_keywords_map[topic_id] = keywords
                else:
                    topic_keywords_map[topic_id] = []

        # ClusteredResponse 객체 생성
        clustered_responses = {}
        for i, response_id in enumerate(response_ids):
            topic_id = int(topics[i])
            keywords = topic_keywords_map.get(topic_id, [])

            clustered_responses[response_id] = ClusteredResponse(
                response_id=response_id,
                conversation_id=response_embeddings[response_id]['conversation_id'],
                embedding=response_embeddings[response_id]['embedding'],
                topic_id=topic_id,
                topic_keywords=keywords
            )

        # 토픽별 분포 출력
        print(f"\n{'='*80}")
        print("토픽별 분포:")
        print(f"{'='*80}")

        from collections import Counter
        topic_counts = Counter(topics)

        # 가장 큰 토픽 10개만 출력
        for topic_id, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            keywords = topic_keywords_map.get(topic_id, [])
            keywords_str = ", ".join(keywords[:5])  # 상위 5개만
            print(f"  토픽 {topic_id:3d} ({count:4d}개): {keywords_str}")

        if len(topic_counts) > 10:
            print(f"  ... 외 {len(topic_counts) - 10}개 토픽")
        print(f"{'='*80}\n")

        return clustered_responses, topic_keywords_map

    def pool_by_conversation(
        self,
        clustered_responses: Dict[str, ClusteredResponse],
        response_embeddings: Dict[str, Dict[str, Any]]
    ) -> Dict[int, DocumentEmbedding]:
        """
        대화별로 토픽 ID에 따라 그룹핑하고 풀링

        CLAUDE.md 워크플로우 step 4:
        "이제 다시 개별 채팅 세션으로 돌아와, 그 안에 속한 답변 벡터들을
        할당된 클러스터 ID별로 그룹핑하고, 같은 그룹 내 벡터들을 풀링(평균)한다.
        결과: 하나의 채팅 세션은 1개가 아닌, 여러 개의 '주제 벡터'를 갖게 됩니다."

        Args:
            clustered_responses: {response_id: ClusteredResponse}
            response_embeddings: {response_id: {...}}

        Returns:
            {conversation_id: DocumentEmbedding}
        """
        print(f"\n{'='*80}")
        print("대화별 토픽 풀링 수행 중...")
        print(f"{'='*80}")

        # 대화별로 그룹핑
        conversation_groups = defaultdict(lambda: defaultdict(list))

        for response_id, clustered_resp in clustered_responses.items():
            conv_id = clustered_resp.conversation_id
            topic_id = clustered_resp.topic_id
            embedding = clustered_resp.embedding

            # Outlier(-1)도 포함
            conversation_groups[conv_id][topic_id].append(embedding)

        # 각 대화의 주제 벡터 생성
        document_embeddings = {}

        for conv_id, topic_embeddings_dict in conversation_groups.items():
            # 각 토픽별로 평균 풀링
            pooled_embeddings = {}
            topic_keywords_dict = {}

            for topic_id, embeddings_list in topic_embeddings_dict.items():
                # 평균 풀링
                pooled = np.mean(embeddings_list, axis=0)
                pooled_embeddings[topic_id] = pooled

                # 키워드 가져오기 (첫 번째 응답에서)
                for response_id, clustered_resp in clustered_responses.items():
                    if (clustered_resp.conversation_id == conv_id and
                        clustered_resp.topic_id == topic_id):
                        topic_keywords_dict[topic_id] = clustered_resp.topic_keywords
                        break

            # 대화 제목 가져오기
            first_response_id = [
                rid for rid, cr in clustered_responses.items()
                if cr.conversation_id == conv_id
            ][0]
            conv_title = response_embeddings[first_response_id]['conversation_title']

            # 응답 개수
            response_count = sum(len(embs) for embs in topic_embeddings_dict.values())

            document_embeddings[conv_id] = DocumentEmbedding(
                conversation_id=conv_id,
                conversation_title=conv_title,
                topic_embeddings=pooled_embeddings,
                topic_keywords=topic_keywords_dict,
                response_count=response_count
            )

        print(f"✓ {len(document_embeddings)}개의 대화에 대한 주제 벡터 생성 완료")

        # 통계 출력
        topic_counts_per_doc = [len(doc.topic_embeddings) for doc in document_embeddings.values()]
        print(f"\n통계:")
        print(f"  평균 주제 수/대화: {np.mean(topic_counts_per_doc):.2f}")
        print(f"  최대 주제 수: {max(topic_counts_per_doc)}")
        print(f"  최소 주제 수: {min(topic_counts_per_doc)}")
        print(f"{'='*80}\n")

        return document_embeddings

    def compute_document_similarity(
        self,
        doc_emb_1: DocumentEmbedding,
        doc_emb_2: DocumentEmbedding
    ) -> float:
        """
        두 문서(대화) 간의 유사도 계산

        CLAUDE.md 워크플로우 step 5:
        "두 채팅 세션(A, B)을 비교할 때, 각 세션이 가진
        주제 벡터 집합 간의 평균 거리를 계산하여 최종 유사도로 삼는다."

        Args:
            doc_emb_1: 문서 1의 임베딩
            doc_emb_2: 문서 2의 임베딩

        Returns:
            유사도 (0~1, 높을수록 유사)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # 각 문서의 모든 주제 벡터 추출
        vectors_1 = list(doc_emb_1.topic_embeddings.values())
        vectors_2 = list(doc_emb_2.topic_embeddings.values())

        if not vectors_1 or not vectors_2:
            return 0.0

        # 모든 쌍 간의 코사인 유사도 계산
        similarities = []
        for v1 in vectors_1:
            for v2 in vectors_2:
                sim = cosine_similarity([v1], [v2])[0, 0]
                similarities.append(sim)

        # 평균 유사도
        avg_similarity = float(np.mean(similarities))
        return avg_similarity

    def compute_all_document_similarities(
        self,
        document_embeddings: Dict[int, DocumentEmbedding]
    ) -> Dict[Tuple[int, int], float]:
        """
        모든 문서 쌍 간의 유사도 계산

        Args:
            document_embeddings: {conversation_id: DocumentEmbedding}

        Returns:
            {(conv_id_1, conv_id_2): similarity}
        """
        print(f"\n{'='*80}")
        print("모든 문서 쌍 간의 유사도 계산 중...")
        print(f"{'='*80}")

        conv_ids = sorted(document_embeddings.keys())
        similarities = {}

        total_pairs = len(conv_ids) * (len(conv_ids) - 1) // 2
        print(f"총 {total_pairs}개의 쌍 계산 예정")

        count = 0
        for i, conv_id_1 in enumerate(conv_ids):
            for conv_id_2 in conv_ids[i + 1:]:
                doc_emb_1 = document_embeddings[conv_id_1]
                doc_emb_2 = document_embeddings[conv_id_2]

                similarity = self.compute_document_similarity(doc_emb_1, doc_emb_2)
                similarities[(conv_id_1, conv_id_2)] = similarity

                count += 1
                if count % 1000 == 0:
                    print(f"  진행: {count}/{total_pairs} ({100 * count / total_pairs:.1f}%)")

        print(f"\n✓ 유사도 계산 완료: {len(similarities)}개의 쌍")

        # 유사도 분포 출력
        sim_values = list(similarities.values())
        print(f"\n유사도 통계:")
        print(f"  평균: {np.mean(sim_values):.4f}")
        print(f"  중앙값: {np.median(sim_values):.4f}")
        print(f"  표준편차: {np.std(sim_values):.4f}")
        print(f"  최대: {np.max(sim_values):.4f}")
        print(f"  최소: {np.min(sim_values):.4f}")
        print(f"{'='*80}\n")

        return similarities

    def get_topic_info(self) -> Any:
        """BERTopic 모델의 토픽 정보 반환"""
        if self.topic_model is None:
            return None
        return self.topic_model.get_topic_info()

    def visualize_topics(self, save_path: Optional[str] = None):
        """
        토픽 시각화 (선택적)

        Args:
            save_path: 저장 경로 (None이면 표시만)
        """
        if self.topic_model is None:
            print("경고: 먼저 클러스터링을 수행해야 합니다.")
            return

        try:
            fig = self.topic_model.visualize_topics()
            if save_path:
                fig.write_html(save_path)
                print(f"✓ 토픽 시각화 저장: {save_path}")
            else:
                fig.show()
        except Exception as e:
            print(f"경고: 시각화 실패 - {e}")
