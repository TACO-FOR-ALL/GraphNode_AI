"""
관계 추론 모듈 (S3)
노트 간의 관계 강도와 논리적 연결 이유를 추론
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import requests
from itertools import combinations

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("경고: requests가 설치되어 있지 않습니다. ConceptNet 기능을 사용할 수 없습니다.")


@dataclass
class Edge:
    """엣지(관계) 데이터 클래스"""
    source_id: int
    target_id: int
    weight: float
    label: Optional[str] = None
    reason: Optional[List[str]] = None

    def __str__(self):
        return f"Edge({self.source_id} -> {self.target_id}, weight={self.weight:.3f}, label={self.label})"


class RelationshipBuilder:
    """노트 간 관계를 추론하는 클래스"""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        use_conceptnet: bool = False,
        conceptnet_url: str = "http://api.conceptnet.io",
        conceptnet_language: str = "ko"
    ):
        """
        Args:
            similarity_threshold: 관계로 인정할 최소 유사도 (0~1)
            use_conceptnet: ConceptNet API 사용 여부
            conceptnet_url: ConceptNet API URL
            conceptnet_language: 언어 코드 (ko, en 등)
        """
        self.similarity_threshold = similarity_threshold
        self.use_conceptnet = use_conceptnet and REQUESTS_AVAILABLE
        self.conceptnet_url = conceptnet_url
        self.conceptnet_language = conceptnet_language

        # ConceptNet 관계 유형 매핑 (한국어 설명)
        self.relation_labels = {
            '/r/RelatedTo': '관련됨',
            '/r/IsA': '~이다',
            '/r/PartOf': '~의 일부',
            '/r/HasA': '~을 가짐',
            '/r/UsedFor': '~에 사용됨',
            '/r/CapableOf': '~할 수 있음',
            '/r/Causes': '~를 야기함',
            '/r/HasProperty': '~의 속성을 가짐',
            '/r/MadeOf': '~로 만들어짐',
            '/r/HasContext': '~의 맥락',
            '/r/SimilarTo': '~와 유사함',
            '/r/Synonym': '동의어',
            '/r/Antonym': '반의어',
        }

    def build_relationships(
        self,
        similarities: Dict[Tuple[int, int], float],
        concepts_dict: Dict[int, List[str]]
    ) -> List[Edge]:
        """
        유사도와 개념 정보를 기반으로 관계 구축

        Args:
            similarities: {(note_id_1, note_id_2): similarity_score}
            concepts_dict: {note_id: [concepts]}

        Returns:
            Edge 객체 리스트
        """
        print(f"\n관계 추론 중 (임계값: {self.similarity_threshold})...")

        edges = []

        for (note_id_1, note_id_2), similarity in similarities.items():
            # 임계값 이상인 경우만 처리
            if similarity < self.similarity_threshold:
                continue

            # 기본 엣지 생성
            edge = Edge(
                source_id=note_id_1,
                target_id=note_id_2,
                weight=similarity
            )

            # ConceptNet으로 논리적 관계 탐색
            if self.use_conceptnet:
                concepts_1 = concepts_dict.get(note_id_1, [])
                concepts_2 = concepts_dict.get(note_id_2, [])

                label, reason = self._find_conceptnet_relation(concepts_1, concepts_2)
                edge.label = label
                edge.reason = reason

            edges.append(edge)

        print(f"총 {len(edges)}개의 관계 생성 완료")
        return edges

    def _find_conceptnet_relation(
        self,
        concepts_1: List[str],
        concepts_2: List[str]
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        두 개념 리스트 간의 ConceptNet 관계 찾기

        Args:
            concepts_1: 첫 번째 노트의 개념 리스트
            concepts_2: 두 번째 노트의 개념 리스트

        Returns:
            (관계 레이블, [개념1, 개념2]) 또는 (None, None)
        """
        if not self.use_conceptnet:
            return None, None

        # 모든 개념 쌍 조합 확인 (최대 10개씩만 확인하여 API 호출 제한)
        concepts_1_sample = concepts_1[:10]
        concepts_2_sample = concepts_2[:10]

        for concept_1 in concepts_1_sample:
            for concept_2 in concepts_2_sample:
                relation = self._query_conceptnet(concept_1, concept_2)
                if relation:
                    label = self.relation_labels.get(relation, relation)
                    return label, [concept_1, concept_2]

        return None, None

    def _query_conceptnet(self, concept_1: str, concept_2: str) -> Optional[str]:
        """
        ConceptNet API에 두 개념 간 관계 조회

        Args:
            concept_1: 첫 번째 개념
            concept_2: 두 번째 개념

        Returns:
            관계 유형 (예: '/r/RelatedTo') 또는 None
        """
        try:
            # 개념을 URI 형식으로 변환
            c1_uri = f"/c/{self.conceptnet_language}/{concept_1.replace(' ', '_')}"
            c2_uri = f"/c/{self.conceptnet_language}/{concept_2.replace(' ', '_')}"

            # ConceptNet API 호출
            url = f"{self.conceptnet_url}/query"
            params = {
                'start': c1_uri,
                'end': c2_uri,
                'limit': 1
            }

            response = requests.get(url, params=params, timeout=3)

            if response.status_code == 200:
                data = response.json()
                edges = data.get('edges', [])

                if edges:
                    # 첫 번째 관계 반환
                    return edges[0].get('rel', {}).get('@id')

        except Exception as e:
            # API 호출 실패는 조용히 처리
            pass

        return None

    def filter_edges_by_weight(self, edges: List[Edge], min_weight: float = 0.8) -> List[Edge]:
        """
        가중치로 엣지 필터링

        Args:
            edges: Edge 리스트
            min_weight: 최소 가중치

        Returns:
            필터링된 Edge 리스트
        """
        filtered = [edge for edge in edges if edge.weight >= min_weight]
        print(f"가중치 {min_weight} 이상: {len(filtered)}개 엣지")
        return filtered

    def get_top_edges(self, edges: List[Edge], top_k: int = 50) -> List[Edge]:
        """
        가중치가 높은 상위 K개 엣지 반환

        Args:
            edges: Edge 리스트
            top_k: 상위 K개

        Returns:
            상위 K개 Edge 리스트
        """
        sorted_edges = sorted(edges, key=lambda e: e.weight, reverse=True)
        return sorted_edges[:top_k]

    def export_edges_to_dict(self, edges: List[Edge]) -> List[Dict[str, Any]]:
        """
        엣지 리스트를 dict 리스트로 변환 (저장/전송용)

        Args:
            edges: Edge 객체 리스트

        Returns:
            dict 리스트
        """
        return [
            {
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'weight': edge.weight,
                'label': edge.label,
                'reason': edge.reason
            }
            for edge in edges
        ]

    def build_simple_relationships(
        self,
        similarities: Dict[Tuple[int, int], float]
    ) -> List[Edge]:
        """
        ConceptNet 없이 유사도만으로 관계 구축 (빠른 버전)

        Args:
            similarities: {(note_id_1, note_id_2): similarity_score}

        Returns:
            Edge 객체 리스트
        """
        print(f"\n간단 관계 추론 중 (임계값: {self.similarity_threshold})...")

        edges = []

        for (note_id_1, note_id_2), similarity in similarities.items():
            if similarity >= self.similarity_threshold:
                edge = Edge(
                    source_id=note_id_1,
                    target_id=note_id_2,
                    weight=similarity,
                    label="유사함"
                )
                edges.append(edge)

        print(f"총 {len(edges)}개의 관계 생성 완료")
        return edges
