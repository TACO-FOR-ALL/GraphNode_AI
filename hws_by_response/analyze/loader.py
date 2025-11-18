"""
데이터 로더 모듈
conversations.json 파일을 로드하는 기능 제공
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path


class ConversationLoader:
    """대화 데이터를 로드하는 클래스"""

    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: conversations.json 파일 경로
        """
        if data_path is None:
            # 기본 경로: 프로젝트 루트의 data/conversations.json
            project_root = Path(__file__).parent.parent
            data_path = project_root / "data" / "mock_data.json"

        self.data_path = Path(data_path)
        self._validate_path()

    def _validate_path(self):
        """파일 경로 유효성 검사"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.data_path}")

        if not self.data_path.is_file():
            raise ValueError(f"파일이 아닙니다: {self.data_path}")

    def load(self) -> List[Dict[str, Any]]:
        """
        conversations.json 파일을 로드

        Returns:
            대화 리스트 (각 대화는 dict 형태)
        """
        print(f"데이터 로드 중: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            print(f"총 {len(conversations)}개의 대화를 로드했습니다.")
            return conversations

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 오류: {e}")
        except Exception as e:
            raise RuntimeError(f"파일 로드 중 오류 발생: {e}")

    def load_sample(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        샘플 데이터만 로드

        Args:
            n: 로드할 대화 개수

        Returns:
            샘플 대화 리스트
        """
        conversations = self.load()
        return conversations[:min(len(conversations),n)]

    @staticmethod
    def get_conversation_info(conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        대화 메타 정보 추출

        Args:
            conversation: 대화 dict

        Returns:
            메타 정보 (title, create_time, update_time, message_count)
        """
        return {
            'title': conversation.get('title', '제목 없음'),
            'create_time': conversation.get('create_time'),
            'update_time': conversation.get('update_time'),
            'message_count': len(conversation.get('mapping', {}))
        }
