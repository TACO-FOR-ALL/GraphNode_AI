"""Pydantic schemas for validating chat graph input and output."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """Single chat message."""

    id: str
    role: str
    content: str
    timestamp: Optional[str] = None

    @validator("role")
    def validate_role(cls, value: str) -> str:
        allowed = {"user", "assistant", "system", "tool"}
        if value not in allowed:
            raise ValueError(f"role must be one of {allowed}")
        return value


class Conversation(BaseModel):
    """A conversation containing multiple messages."""

    id: str
    title: Optional[str] = None
    messages: List[Message]
    create_time: Optional[int] = None
    update_time: Optional[int] = None

    def get_merged_content(self) -> str:
        """Merge all message contents into a single text."""
        merged_parts = []
        for msg in self.messages:
            merged_parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(merged_parts)

    def get_earliest_timestamp(self) -> Optional[str]:
        """Get the earliest timestamp from messages."""
        timestamps = [msg.timestamp for msg in self.messages if msg.timestamp]
        return min(timestamps) if timestamps else None


class ChatHistory(BaseModel):
    """Input payload: list of messages or conversations."""

    messages: List[Message] = Field(default_factory=list)
    conversations: List[Conversation] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, payload: List[dict]) -> "ChatHistory":
        return cls(messages=[Message(**item) for item in payload])

    @classmethod
    def from_conversations(cls, conversations: List[Conversation]) -> "ChatHistory":
        return cls(conversations=conversations)


class Keyword(BaseModel):
    term: str
    score: float


class Node(BaseModel):
    id: int
    orig_id: str = Field(..., alias="orig_id")
    role: str
    text: str
    timestamp: Optional[str]
    cluster: int
    keywords: List[Keyword]
    num_messages: int = 0
    message_ids: List[str] = Field(default_factory=list)


class Edge(BaseModel):
    source: int
    target: int
    weight: float
    type: str


class KeywordParams(BaseModel):
    top_n: int
    max_ngram: int
    dedup_thresh: float


class GraphParams(BaseModel):
    sim_top_k: Optional[int]
    sim_threshold: Optional[float]


class ClusterParams(BaseModel):
    min_cluster_size: int
    min_samples: int
    metric: str


class PreprocessParams(BaseModel):
    lower: bool
    strip_urls: bool
    strip_code: bool
    strip_punct: bool
    stopwords_langs: List[str]


class Params(BaseModel):
    embedding_model: str
    embedding_model_digest: str
    keyword: KeywordParams
    cluster: ClusterParams
    graph: GraphParams
    preprocess: PreprocessParams


class ClusterSummary(BaseModel):
    size: int
    top_terms: List[Tuple[str, float]]


class Counts(BaseModel):
    nodes: int
    edges: int
    clusters: int
    outliers: int


class Metadata(BaseModel):
    clusters: Dict[str, ClusterSummary]
    params: Params
    counts: Counts


class OutputGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    metadata: Metadata

    @validator("edges", each_item=True)
    def validate_edge_type(cls, value: Edge) -> Edge:
        if value.type != "similarity":
            raise ValueError('edge type must be "similarity"')
        return value
