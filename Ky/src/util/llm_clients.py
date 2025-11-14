"""LLM client implementations for multiple providers (OpenAI, Qwen, Groq, Gemini)."""

from __future__ import annotations

import inspect
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv
from httpx import Client, ReadTimeout
import openai
from openai import BadRequestError, OpenAI

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_API_KEY = DASHSCOPE_API_KEY or os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv(
    "DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOW_INSECURE_CONNECTIONS = (
    os.getenv("ALLOW_INSECURE_CONNECTIONS", "false").lower() == "true"
)

# Default models
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-max")
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_MAX_TOKEN_RETRIES = int(os.getenv("GEMINI_MAX_TOKEN_RETRIES", "2"))


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model_name: str, provider_name: str):
        """Initialize the base client.

        Args:
            model_name: Name of the model to use
            provider_name: Name of the provider (for logging)
        """
        self.model_name = model_name
        self.provider_name = provider_name
        self.call_count = 0

    @abstractmethod
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> str:
        """Make a call to the LLM.

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/request
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            LLM response text

        Raises:
            Exception: If API call fails
        """
        pass

    def reset_call_count(self):
        """Reset the call counter."""
        self.call_count = 0

    def get_call_count(self) -> int:
        """Get the number of API calls made."""
        return self.call_count


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (OpenAI, Qwen, Groq, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        provider_name: str,
        allow_insecure: bool = False,
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
        credential_env_var: Optional[str] = None,
    ):
        """Initialize the OpenAI-compatible client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            model_name: Name of the model to use
            provider_name: Name of the provider (for logging)
            allow_insecure: Whether to allow insecure SSL connections
            timeout_seconds: Request timeout in seconds
            max_retries: Number of retry attempts on timeout
            credential_env_var: Name of the environment variable required for credentials
        """
        super().__init__(model_name, provider_name)

        if not api_key:
            env_var_name = credential_env_var or f"{provider_name.upper()}_API_KEY"
            raise ValueError(
                f"API key not found. Please set {env_var_name} in your .env file."
            )

        # Configure SSL verification
        verify_ssl = not allow_insecure

        if allow_insecure:
            print(
                f"\n⚠️ WARNING: SSL certificate verification is disabled for {provider_name}."
            )

        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self._uses_responses_api = model_name.lower().startswith("gpt-5")
        self._supports_response_format = False

        # Create HTTP client
        http_client = Client(
            verify=verify_ssl,
            http1=True,
            http2=False,
            timeout=self.timeout_seconds,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=self.timeout_seconds,
        )
        try:
            signature = inspect.signature(self.client.responses.create)
            self._supports_response_format = "response_format" in signature.parameters
        except (TypeError, ValueError, AttributeError):
            self._supports_response_format = False

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> str:
        """Make a call to the OpenAI-compatible API.

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/request
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            LLM response text

        Raises:
            Exception: If API call fails
        """
        self.call_count += 1
        attempts = 0
        max_attempts = 1 + self.max_retries

        while True:
            attempts += 1
            try:
                if self._uses_responses_api:
                    response_text = self._call_responses_api(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_output_tokens=max_tokens,
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        presence_penalty=1.5,
                    )
                    response_text = completion.choices[0].message.content
                if response_text is None:
                    raise ValueError(f"{self.provider_name} returned empty response")
                return response_text.strip()
            except (openai.APITimeoutError, ReadTimeout) as exc:
                if attempts >= max_attempts:
                    print(f"❌ Error calling {self.provider_name} LLM: {exc}")
                    raise
                wait_seconds = min(10.0, 2.0 * attempts)
                print(
                    f"⚠️ {self.provider_name} request timed out (attempt {attempts}/{max_attempts}). "
                    f"Retrying in {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
            except BadRequestError as exc:
                error_code = None
                error_message = None
                if hasattr(exc, "body"):
                    body = getattr(exc, "body") or {}
                    if isinstance(body, dict):
                        error_payload = body.get("error") or {}
                        error_code = error_payload.get("code")
                        error_message = error_payload.get("message")
                error_code = error_code or getattr(exc, "code", None)
                error_message = error_message or str(exc)

                if error_code == "data_inspection_failed" or (
                    isinstance(error_message, str)
                    and "data_inspection_failed" in error_message
                ):
                    print(
                        f"❌ {self.provider_name} rejected the request due to content inspection. "
                        "Consider sanitizing the prompt or masking sensitive terms."
                    )
                else:
                    print(f"❌ Error calling {self.provider_name} LLM: {error_message}")
                raise
            except Exception as exc:
                print(f"❌ Error calling {self.provider_name} LLM: {exc}")
                raise

    def _call_responses_api(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> str:
        """Invoke the Responses API used by GPT-5 family models."""
        request_kwargs = {
            "model": self.model_name,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "max_output_tokens": max_output_tokens,
        }

        if self._supports_response_format and self._prompts_request_json(
            system_prompt, user_prompt
        ):
            request_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.responses.create(**request_kwargs)

        # Responses objects expose output_text for fast access to aggregated text
        response_text = getattr(response, "output_text", "")
        if response_text:
            return response_text

        # Fallback: walk the structured output to stitch together plain text content
        output = getattr(response, "output", None) or getattr(response, "outputs", None)
        if not output:
            return ""

        json_payload = None
        text_fragments: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                json_data = None
                if isinstance(part, dict):
                    json_data = part.get("json_object")
                    text_value = part.get("text")
                else:
                    json_data = getattr(part, "json_object", None)
                    text_value = getattr(part, "text", None)
                if json_data is not None:
                    json_payload = json_data
                    continue
                if text_value:
                    text_fragments.append(text_value)
        if json_payload is not None:
            try:
                return json.dumps(json_payload, ensure_ascii=False)
            except TypeError:
                return str(json_payload)
        return "\n".join(text_fragments).strip()

    @staticmethod
    def _prompts_request_json(system_prompt: str, user_prompt: str) -> bool:
        """Heuristic to detect prompts that expect JSON output."""
        combined = f"{system_prompt}\n{user_prompt}".lower()
        triggers = (
            "return only valid json",
            "return valid json",
            "output json",
            "respond in json",
            "json format",
            "json object",
            "json array",
            "provide json",
        )
        return any(trigger in combined for trigger in triggers)


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL):
        """Initialize the Gemini client.

        Args:
            api_key: API key for authentication
            model_name: Name of the Gemini model to use

        Raises:
            ImportError: If google-generativeai package is not installed
            ValueError: If API key is not provided
        """
        super().__init__(model_name, "Gemini")

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install it with: pip install google-generativeai"
            )

        if not api_key:
            raise ValueError(
                "API key not found. Please set GEMINI_API_KEY in your .env file."
            )

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Initialize the model (generation config will be set per call)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.max_output_tokens = GEMINI_MAX_OUTPUT_TOKENS
        self.max_token_retries = max(0, GEMINI_MAX_TOKEN_RETRIES)

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> str:
        """Make a call to the Gemini API.

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/request
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            LLM response text

        Raises:
            Exception: If API call fails
        """
        try:
            self.call_count += 1

            # Gemini doesn't have a separate system prompt field,
            # so we combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

            attempt = 0
            retries_left = self.max_token_retries
            requested_tokens = max_tokens
            max_output_tokens = min(requested_tokens, self.max_output_tokens)

            while True:
                attempt += 1

                # Generate content with specified parameters
                response = self.model.generate_content(
                    combined_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_output_tokens,
                    ),
                )

                response_text = self._extract_text(response)
                if response_text:
                    if attempt > 1:
                        print(
                            f"ℹ️ Gemini retry succeeded with max_output_tokens={max_output_tokens}"
                        )
                    return response_text.strip()

                finish_reasons = self._collect_finish_reasons(response)
                prompt_feedback = getattr(response, "prompt_feedback", None)
                block_reason = self._extract_block_reason(prompt_feedback)

                if (
                    "MAX_TOKENS" in finish_reasons
                    and max_output_tokens < self.max_output_tokens
                ):
                    if retries_left <= 0:
                        # No retries remaining
                        break

                    previous_tokens = max_output_tokens
                    max_output_tokens = min(
                        self.max_output_tokens, max_output_tokens * 2
                    )
                    if max_output_tokens == previous_tokens:
                        # Already at cap; no point retrying
                        break

                    retries_left -= 1
                    print(
                        f"⚠️ Gemini hit max tokens (requested={requested_tokens}, used={previous_tokens}). "
                        f"Retrying with max_output_tokens={max_output_tokens}..."
                    )
                    continue

                context_bits = []
                if finish_reasons:
                    context_bits.append("finish_reasons=" + ", ".join(finish_reasons))
                if block_reason:
                    context_bits.append(f"block_reason={block_reason}")

                context = (
                    "; ".join(context_bits) if context_bits else "no additional context"
                )
                raise ValueError(
                    f"Gemini returned no text content ({context}). "
                    "The request may have been blocked or filtered."
                )
        except Exception as exc:
            print(f"❌ Error calling Gemini LLM: {exc}")
            raise

    @staticmethod
    def _extract_text(response) -> str:
        """Extract textual content from a Gemini response."""
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", None)
            if parts is None and isinstance(content, (list, tuple)):
                parts = content
            if parts is None:
                parts = [content]

            text_fragments = []
            for part in parts:
                if part is None:
                    continue
                text_value = None
                if hasattr(part, "text"):
                    text_value = part.text
                elif isinstance(part, dict):
                    text_value = part.get("text")
                elif isinstance(part, str):
                    text_value = part

                if text_value:
                    text_fragments.append(text_value)

            if text_fragments:
                return "".join(text_fragments)

        return ""

    @staticmethod
    def _collect_finish_reasons(response) -> list[str]:
        """Return a list of finish reason names from the response."""
        finish_reasons = []
        for candidate in getattr(response, "candidates", []) or []:
            reason = getattr(candidate, "finish_reason", None)
            if hasattr(reason, "name"):
                finish_reasons.append(reason.name)
            elif reason is not None:
                finish_reasons.append(str(reason))
        return finish_reasons

    @staticmethod
    def _extract_block_reason(prompt_feedback) -> Optional[str]:
        """Extract a textual block reason from prompt feedback."""
        if prompt_feedback is None:
            return None

        block_reason = getattr(prompt_feedback, "block_reason", None)
        if hasattr(block_reason, "name"):
            return block_reason.name
        if block_reason is not None:
            return str(block_reason)
        return None


def create_llm_client(
    provider: str = "openai",
    model_name: Optional[str] = None,
) -> BaseLLMClient:
    """Create an LLM client based on the specified provider.

    Args:
        provider: Provider name ("openai", "qwen", "groq", or "gemini")
        model_name: Optional model name override

    Returns:
        An instance of BaseLLMClient

    Raises:
        ValueError: If provider is unknown or required credentials are missing
    """
    provider = provider.lower()

    if provider == "openai":
        configured_model = model_name or OPENAI_MODEL
        # Normalize common formatting issues (e.g. spaces instead of hyphens)
        normalized_model = configured_model.strip().replace(" ", "-")
        if normalized_model != configured_model:
            print(
                f"ℹ️ Normalizing OpenAI model name from '{configured_model}' to '{normalized_model}'"
            )
        model = normalized_model
        return OpenAICompatibleClient(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_URL,
            model_name=model,
            provider_name="OpenAI",
            allow_insecure=False,
            timeout_seconds=OPENAI_TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
    elif provider == "qwen":
        model = model_name or QWEN_MODEL
        return OpenAICompatibleClient(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            model_name=model,
            provider_name="Qwen",
            allow_insecure=ALLOW_INSECURE_CONNECTIONS,
            credential_env_var="DASHSCOPE_API_KEY",
        )
    elif provider == "groq":
        model = model_name or GROQ_MODEL
        return OpenAICompatibleClient(
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL,
            model_name=model,
            provider_name="Groq",
            allow_insecure=False,  # Groq uses proper SSL
        )
    elif provider == "gemini":
        model = model_name or GEMINI_MODEL
        return GeminiClient(api_key=GEMINI_API_KEY, model_name=model)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported providers: openai, qwen, groq, gemini"
        )
