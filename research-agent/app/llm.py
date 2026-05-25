import os
import logging
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

VLLM_URL = os.environ["VLLM_URL"]
VLLM_MODEL = os.environ["VLLM_MODEL"]

T = TypeVar("T", bound=BaseModel) #input can be any basemodel must be.
logger = logging.getLogger(__name__)


class LLMClients: 
    def __init__(self):
        """Plain chat LLM for prose generation (section writing, stitching)."""
        self._fast = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_URL}/v1",
            api_key="not-needed",
            temperature=0.2,
            max_tokens=6144,
            # Disable Qwen3's <think> tags for structured nodes — they confuse JSON parsing.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )


    
        """For section writers — let Qwen3 reason before writing. Better prose quality."""
        self._writer = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_URL}/v1",
            api_key="not-needed",
            temperature=0.5,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )


    def structured_llm(self, schema: Type[T]) -> BaseChatModel:
        
        return self._fast.with_structured_output(schema, method="json_schema")
    
    def writer_llm(self, **kwargs):
        return self._writer.bind(**kwargs)
    
_clients: LLMClients | None = None

def init_clients() -> LLMClients:
    global _clients
    if _clients is None:
        _clients = LLMClients()
    return _clients

def get_clients() -> LLMClients:
    if _clients is None:
        raise RuntimeError("LLM clients not initialized. Call init_clients() first.")
    return _clients