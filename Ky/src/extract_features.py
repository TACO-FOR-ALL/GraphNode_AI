"""CLI entry point for extracting embeddings and keywords from chat history."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import re
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time

import numpy as np
import yaml
from pydantic import ValidationError

from util.io_schemas import (
    ChatHistory,
    Conversation,
    Keyword,
    Message,
)

np.random.seed(42)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled by fallback
    SentenceTransformer = None  # type: ignore

try:
    from keybert import KeyBERT
except ImportError as exc:  # pragma: no cover - library required
    raise SystemExit(
        "KeyBERT is required. Install dependencies via requirements.txt"
    ) from exc

from stopwordsiso import stopwords as stopwords_iso

MAX_CHARS_PER_CHUNK = 512
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_END_RE = re.compile(r"(?<=[\.\!\?。？！])\s+")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
FILLER_TOKENS = {
    "ㅎㅎ",
    "ㅋㅋ",
    "lol",
    "haha",
    "uh",
    "um",
    "mmm",
    "음",
    "ㅎㅎㅎ",
    "ㅋㅋㅋ",
}


@dataclass
class KeywordConfig:
    top_n: int
    max_ngram: int
    dedup_thresh: float


@dataclass
class PreprocessConfig:
    lower: bool
    strip_urls: bool
    strip_code: bool
    strip_punct: bool
    stopwords_langs: List[str]


@dataclass
class Config:
    embedding_model: str
    keyword: KeywordConfig
    preprocess: PreprocessConfig


@dataclass
class ConversationFeatures:
    """Lightweight conversation feature summary for LLM-based clustering."""

    id: int
    orig_id: str
    keywords: List[Keyword]
    timestamp: Optional[str]
    num_messages: int


@dataclass
class FeatureData:
    """Feature dataset containing inputs needed for LLM clustering and edge generation."""

    conversations: List[ConversationFeatures]
    embeddings: np.ndarray
    metadata: Dict[str, Any]


class DummySentenceTransformer:
    """Deterministic embedding fallback when the real model cannot be loaded."""

    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self.model_name = model_name
        self._dimension = dimension

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for sentence in sentences:
            if not sentence:
                vec = np.zeros(self._dimension, dtype=np.float32)
            else:
                digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()
                seed = int(digest[:16], 16)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self._dimension).astype(np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        stacked = np.vstack(vectors)
        return stacked if convert_to_numpy else stacked.tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


def load_embedding_model(model_name: str):
    """Load a sentence transformer model with a deterministic fallback."""
    if SentenceTransformer is None:
        warnings.warn(
            "sentence-transformers not available; using deterministic fallback embeddings."
        )
        return DummySentenceTransformer(model_name)
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - network/cache failures
        warnings.warn(
            f"Failed to load {model_name}: {exc}. Using deterministic fallback embeddings."
        )
        return DummySentenceTransformer(model_name)


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    keyword_cfg = KeywordConfig(**data["keyword"])
    preprocess_cfg = PreprocessConfig(**data["preprocess"])
    return Config(
        embedding_model=data["embedding_model"],
        keyword=keyword_cfg,
        preprocess=preprocess_cfg,
    )


def _is_message_like(item: Dict[str, Any]) -> bool:
    required = {"id", "role", "content"}
    return required.issubset(item.keys())


def _group_chatgpt_export(payload: Iterable[Dict[str, Any]]) -> List[Conversation]:
    """Group ChatGPT export data into Conversation objects."""
    conversations: List[Conversation] = []

    def _normalize_epoch(value: Any) -> Optional[int]:
        """Cast floats/strings coming from exports into ints for Pydantic."""
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return int(float(value))
            except ValueError:
                return None
        return None

    for conv_idx, conversation in enumerate(payload):
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            continue

        title = conversation.get("title", f"Conversation {conv_idx}")
        create_time = _normalize_epoch(conversation.get("create_time"))
        update_time = _normalize_epoch(conversation.get("update_time"))
        messages: List[Message] = []

        def iter_nodes() -> Iterable[Dict[str, Any]]:
            visited: set[str] = set()
            roots = [node for node in mapping.values() if node.get("parent") is None]

            def sort_key(node: Dict[str, Any]) -> str:
                message = node.get("message")
                if isinstance(message, dict):
                    return str(message.get("id", ""))
                return ""

            roots.sort(key=sort_key)
            queue: deque[Dict[str, Any]] = deque(roots)

            while queue:
                node = queue.popleft()
                node_id = node.get("id")
                if node_id in visited:
                    continue
                visited.add(node_id)
                yield node

                for child_id in node.get("children") or []:
                    child = mapping.get(child_id)
                    if child:
                        queue.append(child)

            for node in mapping.values():
                node_id = node.get("id")
                if node_id not in visited:
                    yield node

        for node in iter_nodes():
            message = node.get("message") or {}
            message_id = message.get("id")
            author = message.get("author") or {}
            role = author.get("role")
            content_obj = message.get("content") or {}

            content: str = ""
            if isinstance(content_obj, dict):
                parts = content_obj.get("parts")
                if isinstance(parts, list):
                    content = "\n".join(str(part) for part in parts)
                elif isinstance(content_obj.get("text"), str):
                    content = content_obj["text"]
            elif isinstance(content_obj, str):
                content = content_obj

            timestamp: Optional[str] = message.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp = None

            if not message_id or role is None or content is None:
                continue

            messages.append(
                Message(
                    id=str(message_id),
                    role=str(role),
                    content=str(content),
                    timestamp=timestamp,
                )
            )

        if messages:
            conv_id = (
                messages[0].id.rsplit("_", 1)[0]
                if "_" in messages[0].id
                else f"conv_{conv_idx}"
            )
            conversations.append(
                Conversation(
                    id=conv_id,
                    title=title,
                    messages=messages,
                    create_time=create_time,
                    update_time=update_time,
                )
            )

    return conversations


def _group_messages_by_conversation(messages: List[Message]) -> List[Conversation]:
    """Group simple message list by conversation ID prefix."""
    from collections import defaultdict

    grouped: Dict[str, List[Message]] = defaultdict(list)

    for msg in messages:
        conv_id = msg.id.rsplit("_", 1)[0] if "_" in msg.id else "conv_0"
        grouped[conv_id].append(msg)

    conversations: List[Conversation] = []
    for conv_id, conv_messages in grouped.items():
        conv_messages.sort(key=lambda m: m.timestamp or "")
        conversations.append(
            Conversation(
                id=conv_id,
                title=f"Conversation {conv_id}",
                messages=conv_messages,
            )
        )

    return conversations


def load_messages(path: Path) -> ChatHistory:
    """
    Load messages and group them into conversations.

    Supports ChatGPT export payloads and flat message lists.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        if isinstance(payload, dict):
            conversations = _group_chatgpt_export([payload])
            if not conversations:
                raise ValueError(
                    "Input JSON must be a list of message objects or a ChatGPT export mapping."
                )
            return ChatHistory.from_conversations(conversations)
        raise ValueError("Input JSON must be a list of message objects.")

    if not payload:
        return ChatHistory.from_conversations([])

    if payload and all(
        isinstance(item, dict) and _is_message_like(item) for item in payload
    ):
        messages = [Message(**item) for item in payload]
        conversations = _group_messages_by_conversation(messages)
        return ChatHistory.from_conversations(conversations)

    try:
        conversations = _group_chatgpt_export(payload)
        if not conversations:
            raise ValueError("Unable to interpret input JSON as chat messages.")
        return ChatHistory.from_conversations(conversations)
    except Exception as exc:
        raise ValueError(
            "Input must be either a list of messages or ChatGPT export format."
        ) from exc


def strip_code_blocks(text: str) -> str:
    return CODE_BLOCK_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def preprocess_text(text: str, cfg: PreprocessConfig) -> str:
    cleaned = text
    if cfg.strip_code:
        cleaned = strip_code_blocks(cleaned)
    if cfg.strip_urls:
        cleaned = strip_urls(cleaned)
    if cfg.lower:
        cleaned = cleaned.lower()
    if cfg.strip_punct:
        cleaned = cleaned.translate(PUNCT_TRANSLATION)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    parts = SENTENCE_END_RE.split(text)
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(part) <= max_chars:
            current = part
        else:
            for start in range(0, len(part), max_chars):
                segment = part[start : start + max_chars].strip()
                if segment:
                    chunks.append(segment)
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def get_embedding_dimension(model) -> int:
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
        if isinstance(dim, int):
            return dim
    sample = model.encode(["dimension_probe"], convert_to_numpy=True)
    if isinstance(sample, list):
        sample = np.array(sample)
    return int(sample.shape[1])


def mean_pool_embeddings(model, texts: Sequence[str]) -> np.ndarray:
    embeddings = model.encode(list(texts), convert_to_numpy=True)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    return embeddings.mean(axis=0)


def generate_embeddings(
    cleaned_texts: Sequence[str],
    model,
    *,
    chunk_cache: Optional[Sequence[Sequence[str]]] = None,
) -> np.ndarray:
    dimension = get_embedding_dimension(model)
    vectors: List[np.ndarray] = []
    for idx, text in enumerate(cleaned_texts):
        cached_chunks: Optional[Sequence[str]] = None
        if chunk_cache is not None:
            try:
                cached_chunks = chunk_cache[idx]
            except (IndexError, TypeError):
                cached_chunks = None
        if cached_chunks is None:
            chunks = chunk_text(text) if text else []
        else:
            chunks = list(cached_chunks)
        if chunks:
            vec = mean_pool_embeddings(model, chunks)
        else:
            vec = np.zeros(dimension, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        vectors.append(vec.astype(np.float32))
    return np.vstack(vectors) if vectors else np.zeros((0, dimension), dtype=np.float32)


def build_stopwords(langs: Iterable[str]) -> List[str]:
    stoplist = set()
    for lang in langs:
        try:
            stoplist.update(stopwords_iso(lang))
        except KeyError:
            warnings.warn(f"No stopwords found for language '{lang}'.")
    stoplist.update(FILLER_TOKENS)
    cleaned = {word.strip().lower() for word in stoplist if word and word.strip()}
    return sorted(cleaned)


def keyword_token_set(term: str) -> set:
    tokens = TOKEN_RE.findall(term.lower())
    if not tokens:
        tokens = list(term.lower())
    return set(tokens)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def deduplicate_keywords(
    candidates: Sequence[Tuple[str, float]],
    *,
    limit: int,
    threshold: float,
) -> List[Keyword]:
    selected: List[Keyword] = []
    seen: List[set] = []
    for term, score in candidates:
        token_set = keyword_token_set(term)
        if any(jaccard(token_set, prev) >= threshold for prev in seen):
            continue
        seen.append(token_set)
        selected.append(Keyword(term=term, score=float(score)))
        if len(selected) >= limit:
            break
    return selected


def extract_keywords(
    texts: Sequence[str],
    config: KeywordConfig,
    stoplist: Optional[List[str]],
    keybert_model: KeyBERT,
    doc_embeddings: Optional[np.ndarray] = None,
) -> List[List[Keyword]]:
    results: List[List[Keyword]] = []
    stop_words = stoplist if stoplist else None
    doc_embeddings_arr = doc_embeddings
    if doc_embeddings_arr is not None:
        if (
            not isinstance(doc_embeddings_arr, np.ndarray)
            or doc_embeddings_arr.ndim < 1
        ):
            warnings.warn(
                "Document embeddings must be a numpy array; ignoring provided embeddings."
            )
            doc_embeddings_arr = None
        elif len(doc_embeddings_arr) != len(texts):
            warnings.warn(
                "Document embeddings count does not match number of texts; ignoring provided embeddings."
            )
            doc_embeddings_arr = None

    for idx, text in enumerate(texts):
        if not text:
            results.append([])
            continue
        doc_embedding: Optional[np.ndarray] = None
        if doc_embeddings_arr is not None:
            doc_embedding = doc_embeddings_arr[idx]
            if isinstance(doc_embedding, np.ndarray):
                if doc_embedding.ndim == 1:
                    doc_embedding = doc_embedding.reshape(1, -1)
                elif doc_embedding.ndim != 2:
                    warnings.warn(
                        "Unexpected document embedding shape; falling back to KeyBERT's internal embeddings."
                    )
                    doc_embedding = None
            else:
                doc_embedding = None
        try:
            raw_keywords = keybert_model.extract_keywords(
                text,
                top_n=config.top_n * 4,
                keyphrase_ngram_range=(1, config.max_ngram),
                stop_words=stop_words,
                doc_embeddings=doc_embedding,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Keyword extraction failed: {exc}")
            raw_keywords = []
        deduped = deduplicate_keywords(
            raw_keywords, limit=config.top_n, threshold=config.dedup_thresh
        )
        results.append(deduped)
    return results


def extract_keywords_and_embeddings(
    chat_history: ChatHistory, config: Config
) -> Tuple[List[ConversationFeatures], np.ndarray, Dict[str, Any]]:
    """
    Execute pipeline up to keyword extraction (before clustering).
    Returns conversation features, embeddings, and metadata with timing.
    """
    overall_start = time.perf_counter()

    model = load_embedding_model(config.embedding_model)
    stoplist = build_stopwords(config.preprocess.stopwords_langs)

    conversations = chat_history.conversations
    if not conversations:
        conversations = [
            Conversation(id=msg.id, messages=[msg]) for msg in chat_history.messages
        ]

    merged_texts = [conv.get_merged_content() for conv in conversations]
    cleaned_texts = [preprocess_text(text, config.preprocess) for text in merged_texts]
    chunked_texts = [chunk_text(text) if text else [] for text in cleaned_texts]
    total_chunk_segments = sum(len(chunks) for chunks in chunked_texts)
    avg_chunks_per_conversation = (
        total_chunk_segments / len(chunked_texts) if chunked_texts else 0.0
    )

    # === STEP 1: Embedding Generation (preprocessing, chunking, mean pooling) ===
    embedding_start = time.perf_counter()
    embeddings = generate_embeddings(cleaned_texts, model, chunk_cache=chunked_texts)
    embedding_time = time.perf_counter() - embedding_start
    num_embeddings = embeddings.shape[0]
    embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 0
    print(f"  ⏱️  Embedding generation: {embedding_time:.1f}s")
    print(
        f"      └─ Generated {num_embeddings} embeddings "
        f"({embedding_dim}-dimensional vectors)"
    )
    print(
        f"      └─ Encoded {total_chunk_segments} chunk segments before mean pooling "
        f"(avg {avg_chunks_per_conversation:.2f} per conversation)"
    )

    # === STEP 2: Keyword Extraction ===
    keyword_start = time.perf_counter()
    keybert_model = KeyBERT(model=model)
    keywords = extract_keywords(
        cleaned_texts,
        config.keyword,
        stoplist,
        keybert_model,
        doc_embeddings=embeddings,
    )
    keyword_time = time.perf_counter() - keyword_start
    print(f"  ⏱️  Keyword extraction: {keyword_time:.1f}s")

    # Build conversation-level feature summaries
    conversation_features: List[ConversationFeatures] = []
    for idx, (conversation, keyword_list) in enumerate(zip(conversations, keywords)):
        conversation_features.append(
            ConversationFeatures(
                id=idx,
                orig_id=conversation.id,
                keywords=keyword_list,
                timestamp=conversation.get_earliest_timestamp(),
                num_messages=len(conversation.messages),
            )
        )

    # Calculate total time
    total_time = time.perf_counter() - overall_start

    # Prepare metadata
    metadata = {
        "total_conversations": len(conversation_features),
        "embedding_model": config.embedding_model,
        "keyword_params": {
            "top_n": config.keyword.top_n,
            "max_ngram": config.keyword.max_ngram,
            "dedup_thresh": config.keyword.dedup_thresh,
        },
        "preprocess_params": {
            "lower": config.preprocess.lower,
            "strip_urls": config.preprocess.strip_urls,
            "strip_code": config.preprocess.strip_code,
            "strip_punct": config.preprocess.strip_punct,
            "stopwords_langs": config.preprocess.stopwords_langs,
        },
        "timing": {
            "embedding_seconds": round(embedding_time, 2),
            "keyword_seconds": round(keyword_time, 2),
            "total_seconds": round(total_time, 2),
        },
        "embedding_stats": {
            "conversations": len(conversation_features),
            "chunk_segments": total_chunk_segments,
            "avg_chunks_per_conversation": avg_chunks_per_conversation,
            "embedding_dimension": int(embedding_dim),
        },
    }

    return conversation_features, embeddings, metadata


def extract_and_save_features(
    input_path: Path,
    output_path: Path,
    config_path: Path,
) -> FeatureData:
    """
    Build feature data for downstream processing.
    Extracts keywords and embeddings, saves to JSON.
    """
    config = load_config(config_path)
    messages = load_messages(input_path)
    try:
        messages.messages  # access to trigger validation
    except ValidationError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid input data: {exc}") from exc

    conversation_features, embeddings, metadata = extract_keywords_and_embeddings(
        messages, config
    )

    # Create FeatureData
    feature_data = FeatureData(
        conversations=conversation_features, embeddings=embeddings, metadata=metadata
    )

    # Convert to dictionary for JSON serialization
    output_data = {
        "conversations": [
            {
                "id": conv.id,
                "orig_id": conv.orig_id,
                "keywords": [
                    {"term": kw.term, "score": kw.score} for kw in conv.keywords
                ],
                "timestamp": conv.timestamp,
                "num_messages": conv.num_messages,
            }
            for conv in feature_data.conversations
        ],
        "embeddings": feature_data.embeddings.tolist(),
        "metadata": feature_data.metadata,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary with timing breakdown
    timing = metadata.get("timing", {})
    total_time = timing.get("total_seconds", 0)
    embedding_time = timing.get("embedding_seconds", 0)
    keyword_time = timing.get("keyword_seconds", 0)

    print(f"\n⏱️  Feature extraction completed in {total_time:.1f}s")
    print(f"    └─ Embedding: {embedding_time:.1f}s, Keyword: {keyword_time:.1f}s")
    print(
        f"    └─ Embeddings count: {feature_data.embeddings.shape[0]} "
        f"({feature_data.embeddings.shape[1]} dims)"
    )
    embedding_stats = metadata.get("embedding_stats", {})
    chunk_segments = embedding_stats.get("chunk_segments")
    avg_chunks = embedding_stats.get("avg_chunks_per_conversation")
    if chunk_segments is not None:
        if avg_chunks is None:
            total_convs = embedding_stats.get("conversations") or len(feature_data.conversations) or 0
            avg_chunks = (chunk_segments / total_convs) if total_convs else 0.0
        print(
            f"    └─ Chunk segments encoded before pooling: {chunk_segments} "
            f"(avg {avg_chunks:.2f} per conversation)"
        )

    return feature_data


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract keywords and embeddings from chat conversations"
    )
    parser.add_argument(
        "--input",
        "--in",
        dest="input_path",
        required=True,
        help="Path to chat history JSON file.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output_path",
        required=True,
        help="Path to write intermediate results JSON.",
    )
    parser.add_argument(
        "--cfg", dest="config_path", required=True, help="Path to YAML config."
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path)

    # Extract feature data (embeddings and keywords)
    feature_data = extract_and_save_features(input_path, output_path, config_path)
    print(
        f"Feature data extracted for {len(feature_data.conversations)} conversations. "
        f"Output saved to {output_path.resolve()}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
