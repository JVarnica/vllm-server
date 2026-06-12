# vllm-server

The self-hosted backend powering [ExecuChat](https://github.com/JVarnica/Execu_Chat)'s online mode — a multi-user LLM stack with streaming chat, web search, RAG, and a deep research agent. Runs as a single Docker Compose stack on one consumer GPU.

> 📖 **Full write-up — architecture, design decisions, and setup:** [Python Server hub](https://jvarnica.github.io/projects/python-server/)

## Stack

Nine containers on an internal network; only the gateway, Prometheus, and Grafana are exposed.

| Service | Role |
|---|---|
| **vLLM** | Qwen3-8B-NVFP4 inference, OpenAI-compatible API |
| **Gateway** | FastAPI entry point — routing, session validation |
| **Auth** | JWT issue / refresh |
| **SearxNG** | Privacy metasearch, JSON API |
| **deep-research** | LangGraph research agent ([repo](https://github.com/JVarnica/research-agent)) |
| **Redis** | Session context, task queue, streaming chunks |
| **Qdrant** | Vector store for RAG |
| **Prometheus / Grafana** | Metrics and dashboards |

## Requirements

- Linux (tested on Ubuntu 24.04)
- NVIDIA GPU, 16 GB+ VRAM (built for an RTX 5060 Ti)
- NVIDIA Container Toolkit
- Docker & Docker Compose
- A Hugging Face token

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env   # then fill in HF_TOKEN, JWT_SECRET_KEY, QDRANT_API_KEY, etc.

# 2. Bring up the stack
docker compose up -d

# 3. Watch vLLM load the model (first boot is slow)
docker compose logs -f vllm

# 4. Gateway is live once vLLM is healthy
curl http://localhost:8080/health
```

Model weights download into `~/.cache/huggingface` on first boot and are reused on restart. Dashboards at `localhost:3000`.

### vLLM Configuration
 
The inference container is tuned to fit Qwen3-8B-NVFP4 inside 16 GB VRAM while leaving headroom for concurrent users:
 
--max-model-len 16384          # context window
--gpu-memory-utilization 0.9   # leave ~10% VRAM headroom
--kv-cache-dtype fp8           # halve KV-cache memory
--max-num-seqs 4               # cap concurrent sequences
--enable-prefix-caching        # reuse shared prompt prefixes
--enable-auto-tool-choice
--tool-call-parser hermes      # parse Qwen tool calls


## Related

- [ExecuChat](https://github.com/JVarnica/Execu_Chat) — Android frontend
- [research-agent](https://github.com/JVarnica/research-agent) — deep research container
- Write-ups: [why vLLM](https://jvarnica.github.io/inference/) · [building the server](https://jvarnica.github.io/inference-server/) · [web search tool](https://jvarnica.github.io/agentic-search/)
