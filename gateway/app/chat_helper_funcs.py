



import json
from typing import Any, Optional
import os
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TOKENS = 10000
MIN_GEN_TOKENS = 512
MAX_CONTEXT_WINDOW = 8192
MARGIN_SAFETY = 128



_tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
    trust_remote_code=True,
)

def _rough_token_estimate(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(value) // 4)

def count_tokens(messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None) -> int:
    """Prompt count need to render chat template"""
    try: 
        rendered = _tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        return len(_tokenizer.encode(rendered, add_special_tokens=False))
    except Exception as e:
        logger.warning(f"apply_chat_template failed ({e}); falling back to estimate")
        total = 20 * len(messages)
        for message in messages:
            total += _rough_token_estimate(message.get("content"))
            total += _rough_token_estimate(message.get("tool_calls"))
        if tools:
            total += _rough_token_estimate(tools)
        return total

def compute_max_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Compute max_tokens so prompt + generation fits in context window."""
    prompt_tokens = count_tokens(messages, tools=tools)
    available = MAX_CONTEXT_WINDOW - prompt_tokens - MARGIN_SAFETY

    if available < MIN_GEN_TOKENS:
        logger.warning(
            f"Prompt is {prompt_tokens} tokens — only {available} left for "
            f"generation (below MIN_GEN_TOKENS={MIN_GEN_TOKENS}). "
        )
    return max(MIN_GEN_TOKENS, available)

# sse stream done when "finish_reason": "stop" recieved so need to have flag for when recieved
def parse_sse_stream(line: str) -> tuple[str, bool, str, dict]:
    empty = ("", False, "", {})
    if not line.startswith("data:"):
        return empty
    data = line[len("data:"):].strip()
    if not data:
        return empty
    if data == "[DONE]":
        return ("", True, "stop",{})
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return empty
    choices = obj.get("choices") or []
    if not choices:
        return empty
    
    c0 = choices[0] or {}
    delta = c0.get("delta") or {}
    content = delta.get("content") or ""
    reasoning_content = delta.get("reasoning_content") or ""
    
    finish = c0.get("finish_reason") or ""
    done = finish in ("stop", "length") 
    usage =obj.get("usage")
    return (content, reasoning_content ,done, finish, usage)