"""
데이터 파서 모듈
conversations.json의 복잡한 구조를 분석하여 노트(Note)로 변환
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class Note:
    """노트 데이터 클래스"""
    note_id: int
    title: str
    content: str
    create_time: float
    update_time: Optional[float] = None
    message_count: int = 0

    def __str__(self):
        return f"Note(id={self.note_id}, title='{self.title}', length={len(self.content)})"


@dataclass
class AIResponse:
    """AI 답변 데이터 클래스"""
    response_id: str  # conversation_id + "_" + message_index
    conversation_id: int  # 어느 대화에 속하는지
    conversation_title: str
    content: str  # AI 답변 내용
    message_index: int  # 해당 대화 내에서 몇 번째 AI 답변인지
    timestamp: Optional[float] = None  # 생성 시간 (선택적)

    def __str__(self):
        return f"AIResponse(id={self.response_id}, conv={self.conversation_id}, idx={self.message_index})"


class NoteParser:
    """대화를 노트로 파싱하는 클래스"""

    def __init__(self, min_content_length: int = 10):
        """
        Args:
            min_content_length: 노트로 인정할 최소 콘텐츠 길이
        """
        self.min_content_length = min_content_length

    def parse_conversations(self, conversations: List[Dict[str, Any]]) -> List[Note]:
        """
        대화 리스트를 노트 리스트로 변환

        Args:
            conversations: 대화 리스트

        Returns:
            Note 객체 리스트
        """
        notes = []

        for idx, conversation in enumerate(conversations, start=1):
            try:
                note = self._parse_single_conversation(conversation, note_id=idx)
                if note and len(note.content) >= self.min_content_length:
                    notes.append(note)
            except Exception as e:
                print(f"경고: 대화 {idx} 파싱 실패 - {e}")
                continue

        print(f"총 {len(notes)}개의 노트를 생성했습니다.")
        return notes

    def _parse_single_conversation(self, conversation: Dict[str, Any], note_id: int) -> Optional[Note]:
        """
        단일 대화를 노트로 변환

        Args:
            conversation: 대화 dict
            note_id: 노트 ID

        Returns:
            Note 객체 또는 None
        """
        title = conversation.get('title', f'대화 {note_id}')
        create_time = conversation.get('create_time', 0)
        update_time = conversation.get('update_time')
        mapping = conversation.get('mapping', {})

        # 대화 내용 추출
        messages = self._extract_messages(mapping)

        if not messages:
            return None

        # 메시지를 하나의 텍스트로 합치기
        content = self._merge_messages(messages)

        return Note(
            note_id=note_id,
            title=title,
            content=content,
            create_time=create_time,
            update_time=update_time,
            message_count=len(messages)
        )

    def _extract_messages(self, mapping: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        mapping에서 실제 메시지 추출

        Args:
            mapping: 대화의 mapping 객체

        Returns:
            메시지 리스트 [{'role': 'user', 'content': '...', 'create_time': ...}, ...]
        """
        messages = []

        for node_id, node_data in mapping.items():
            message = node_data.get('message')

            if not message:
                continue

            # 메시지 정보 추출
            author = message.get('author', {})
            role = author.get('role')

            # system 메시지는 건너뛰기
            if role not in ['user', 'assistant']:
                continue

            content_obj = message.get('content', {})
            parts = content_obj.get('parts', [])

            # content.parts[0]에 실제 텍스트가 있음
            if not parts or not parts[0]:
                continue

            text = str(parts[0]).strip()
            text = self._sanitize_text(text)

            if not text:
                continue

            messages.append({
                'role': role,
                'content': text,
                'create_time': message.get('create_time', 0)
            })

        # create_time 기준으로 정렬
        messages.sort(key=lambda x: x['create_time'])

        return messages

    def _merge_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        메시지들을 하나의 텍스트로 합치기

        Args:
            messages: 메시지 리스트

        Returns:
            합쳐진 텍스트
        """
        merged_parts = []

        for msg in messages:
            role = msg['role']
            content = msg['content']

            # 역할 표시와 함께 내용 추가
            role_label = "사용자" if role == "user" else "어시스턴트"
            merged_parts.append(f"[{role_label}] {content}")

        return "\n\n".join(merged_parts)

    def _sanitize_text(self, text: str) -> str:
        try:
            emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\u2600-\u27BF]+", flags=re.UNICODE)
            text = emoji_pattern.sub("", text)
        except re.error:
            text = re.sub(r"[\u2600-\u27BF]+", "", text)
        fillers = [
            "좋습니다",
            "알겠습니다",
            "아주 좋은 시도",
            "좋은 시도",
            "감사합니다",
        ]
        for f in fillers:
            text = text.replace(f, "")
        return text.strip()

    def export_notes_to_dict(self, notes: List[Note]) -> List[Dict[str, Any]]:
        """
        노트 리스트를 dict 리스트로 변환 (저장/전송용)

        Args:
            notes: Note 객체 리스트

        Returns:
            dict 리스트
        """
        return [
            {
                'note_id': note.note_id,
                'title': note.title,
                'content': note.content,
                'create_time': note.create_time,
                'update_time': note.update_time,
                'message_count': note.message_count
            }
            for note in notes
        ]

    def parse_ai_responses(self, conversations: List[Dict[str, Any]]) -> List[AIResponse]:
        """
        대화 리스트에서 AI 답변만 추출

        각 대화에서 assistant 역할의 메시지만 개별적으로 추출합니다.
        이는 노이즈가 많은 사용자 질문 대신 잘 정제된 AI 답변만을 사용하기 위함입니다.

        Args:
            conversations: 대화 리스트

        Returns:
            AIResponse 객체 리스트
        """
        ai_responses = []

        for conv_idx, conversation in enumerate(conversations, start=1):
            try:
                responses = self._extract_ai_responses_from_conversation(
                    conversation,
                    conversation_id=conv_idx
                )
                ai_responses.extend(responses)
            except Exception as e:
                print(f"경고: 대화 {conv_idx} AI 답변 추출 실패 - {e}")
                continue

        print(f"총 {len(ai_responses)}개의 AI 답변을 추출했습니다.")
        return ai_responses

    def _extract_ai_responses_from_conversation(
        self,
        conversation: Dict[str, Any],
        conversation_id: int
    ) -> List[AIResponse]:
        """
        단일 대화에서 AI 답변들만 추출

        Args:
            conversation: 대화 dict
            conversation_id: 대화 ID

        Returns:
            AIResponse 객체 리스트
        """
        title = conversation.get('title', f'대화 {conversation_id}')
        mapping = conversation.get('mapping', {})

        # 전체 메시지 추출 (시간순 정렬)
        messages = self._extract_messages(mapping)

        # assistant 메시지만 필터링
        ai_responses = []
        ai_message_index = 0

        for msg in messages:
            if msg['role'] == 'assistant':
                content = self._sanitize_text(msg['content']).strip()

                # 최소 길이 체크
                if len(content) < self.min_content_length:
                    continue

                response_id = f"{conversation_id}_{ai_message_index}"

                ai_responses.append(AIResponse(
                    response_id=response_id,
                    conversation_id=conversation_id,
                    conversation_title=title,
                    content=content,
                    message_index=ai_message_index,
                    timestamp=msg.get('create_time')  # 선택적으로 가져오기
                ))

                ai_message_index += 1

        return ai_responses

    def export_ai_responses_to_dict(self, responses: List[AIResponse]) -> List[Dict[str, Any]]:
        """
        AIResponse 리스트를 dict 리스트로 변환 (저장/전송용)

        Args:
            responses: AIResponse 객체 리스트

        Returns:
            dict 리스트
        """
        return [
            {
                'response_id': resp.response_id,
                'conversation_id': resp.conversation_id,
                'conversation_title': resp.conversation_title,
                'content': resp.content,
                'create_time': resp.create_time,
                'message_index': resp.message_index
            }
            for resp in responses
        ]

    def parse_qa_pairs(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        대화 리스트에서 Q-A 쌍 추출
        
        각 대화에서 user-assistant 메시지 쌍을 추출합니다.
        
        Args:
            conversations: 대화 리스트
            
        Returns:
            Q-A 쌍 리스트 [{'qa_id': ..., 'conversation_id': ..., 'question': ..., 'answer': ..., ...}, ...]
        """
        qa_pairs = []
        
        for conv_idx, conversation in enumerate(conversations, start=1):
            try:
                pairs = self._extract_qa_pairs_from_conversation(
                    conversation,
                    conversation_id=conv_idx
                )
                qa_pairs.extend(pairs)
            except Exception as e:
                print(f"경고: 대화 {conv_idx} Q-A 쌍 추출 실패 - {e}")
                continue
        
        print(f"총 {len(qa_pairs)}개의 Q-A 쌍을 추출했습니다.")
        return qa_pairs
    
    def _extract_qa_pairs_from_conversation(
        self,
        conversation: Dict[str, Any],
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """
        단일 대화에서 Q-A 쌍 추출
        
        Args:
            conversation: 대화 dict
            conversation_id: 대화 ID
            
        Returns:
            Q-A 쌍 리스트
        """
        title = conversation.get('title', f'대화 {conversation_id}')
        mapping = conversation.get('mapping', {})
        
        # 전체 메시지 추출 (시간순 정렬)
        messages = self._extract_messages(mapping)
        
        # user-assistant 쌍 매칭
        qa_pairs = []
        qa_index = 0
        
        i = 0
        while i < len(messages):
            # user 메시지 찾기
            if messages[i]['role'] == 'user':
                question = self._sanitize_text(messages[i]['content']).strip()
                
                # 다음 assistant 메시지 찾기
                if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    answer = self._sanitize_text(messages[i + 1]['content']).strip()
                    
                    # 최소 길이 체크
                    if len(question) >= 5 and len(answer) >= self.min_content_length:
                        qa_id = f"{conversation_id}_{qa_index}"
                        
                        qa_pairs.append({
                            'qa_id': qa_id,
                            'conversation_id': conversation_id,
                            'conversation_title': title,
                            'question': question,
                            'answer': answer,
                            'qa_index': qa_index,
                            'timestamp': messages[i + 1].get('create_time')
                        })
                        
                        qa_index += 1
                    
                    i += 2  # user + assistant 건너뛰기
                else:
                    i += 1
            else:
                i += 1
        
        return qa_pairs
