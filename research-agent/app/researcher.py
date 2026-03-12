"""
Deep Researcher
===============
A multi-step research agent that:
  1. Plans sub-questions from the user query
  2. Searches SearXNG for each sub-question
  3. Summarises each source
  4. Synthesises everything into a structured report

Uses vLLM via the OpenAI-compatible API (through langchain-openai).
Streams structured events so the Android UI can show live progress.

NOTE: We avoid tool-calling here because quantised models (NVFP4) can be
unreliable with structured tool calls. Instead we use simple prompt→JSON
extraction which works much more reliably with smaller models.
"""

import asyncio
import json
import re
from typing import Callable, Coroutine, Any, Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
EventCallback = Callable[[dict], Coroutine[Any, Any, None]]


class DeepResearcher:
    def __init__(
        self,
        vllm_url: str,
        vllm_model: str,
        searxng_url: str,
    ):
        self.searxng_url = searxng_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=30)

        # LLM for planning + summarisation (fast, lower tokens)
        self.llm_fast = ChatOpenAI(
            openai_api_base=f"{vllm_url}/v1",
            openai_api_key="not-needed",
            model_name=vllm_model,
            temperature=0.3,
            max_tokens=1024,
        )

        # LLM for final report (higher token budget)
        self.llm_report = ChatOpenAI(
            openai_api_base=f"{vllm_url}/v1",
            openai_api_key="not-needed",
            model_name=vllm_model,
            temperature=0.4,
            max_tokens=4096,
        )

    async def close(self):
        await self.http.aclose()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def research(
        self,
        query: str,
        max_searches: int = 8,
        max_results_per_search: int = 5,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """Run the full research pipeline, returns a markdown report."""

        async def emit(event: dict):
            if on_event:
                await on_event(event)

        # ---- Phase 1: Planning ----
        await emit({
            "type": "status",
            "phase": "planning",
            "message": "Analysing query and generating research plan…",
        })

        sub_questions = await self._plan(query, max_searches)

        await emit({
            "type": "status",
            "phase": "planning",
            "message": f"Research plan ready — {len(sub_questions)} sub-questions",
            "sub_questions": sub_questions,
        })

        # ---- Phase 2: Search + Summarise per sub-question ----
        all_findings: list[dict] = []

        for i, sq in enumerate(sub_questions):
            await emit({
                "type": "status",
                "phase": "searching",
                "message": f"Searching ({i+1}/{len(sub_questions)}): {sq}",
                "current_question": sq,
                "progress": (i + 1) / len(sub_questions),
            })

            # Search
            results = await self._search(sq, max_results_per_search)

            await emit({
                "type": "search",
                "sub_question": sq,
                "query": sq,
                "num_results": len(results),
                "sources": [
                    {"title": r.get("title", ""), "url": r.get("url", "")}
                    for r in results[:3]
                ],
            })

            # Emit individual sources
            for r in results:
                await emit({
                    "type": "source",
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:200],
                })

            # Summarise this batch of results
            if results:
                await emit({
                    "type": "status",
                    "phase": "reading",
                    "message": f"Summarising findings for: {sq}",
                })

                summary = await self._summarise(sq, results)

                await emit({
                    "type": "summary",
                    "sub_question": sq,
                    "summary": summary[:500],
                })

                all_findings.append({
                    "question": sq,
                    "summary": summary,
                    "sources": [
                        {"title": r.get("title", ""), "url": r.get("url", "")}
                        for r in results
                    ],
                })

        # ---- Phase 3: Synthesise final report ----
        await emit({
            "type": "status",
            "phase": "synthesising",
            "message": "Writing final research report…",
            "progress": 1.0,
        })

        report = await self._synthesise(query, all_findings)

        await emit({
            "type": "status",
            "phase": "done",
            "message": "Research complete",
        })

        return report

    # ------------------------------------------------------------------
    # Step 1 – Planning: break query into sub-questions
    # ------------------------------------------------------------------
    async def _plan(self, query: str, max_questions: int) -> list[str]:
        prompt = f"""You are a research planner. Given the user's research query, generate a list of specific sub-questions that need to be answered to produce a comprehensive research report.

Rules:
- Generate between 3 and {max_questions} sub-questions
- Each sub-question should be specific and searchable
- Cover different angles of the topic
- Return ONLY a JSON array of strings, no other text

User query: {query}

JSON array of sub-questions:"""

        response = await self.llm_fast.ainvoke([HumanMessage(content=prompt)])
        return self._parse_json_array(response.content, fallback=[query])

    # ------------------------------------------------------------------
    # Step 2 – Search SearXNG
    # ------------------------------------------------------------------
    async def _search(self, query: str, max_results: int) -> list[dict]:
        try:
            resp = await self.http.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return results[:max_results]
        except Exception as e:
            print(f"[Search Error] {e}")
            return []

    # ------------------------------------------------------------------
    # Step 3 – Summarise search results for a sub-question
    # ------------------------------------------------------------------
    async def _summarise(self, question: str, results: list[dict]) -> str:
        context = ""
        for i, r in enumerate(results):
            title = r.get("title", "Untitled")
            content = r.get("content", "")[:600]
            url = r.get("url", "")
            context += f"\n[{i+1}] {title}\nURL: {url}\n{content}\n"

        prompt = f"""You are a research analyst. Summarise the following search results to answer the question.

Question: {question}

Search Results:
{context}

Write a concise but thorough summary (2-4 paragraphs). Include key facts, figures, and cite sources by number [1], [2] etc. If results are conflicting, note the disagreement."""

        response = await self.llm_fast.ainvoke([HumanMessage(content=prompt)])
        return response.content

    # ------------------------------------------------------------------
    # Step 4 – Synthesise all findings into a report
    # ------------------------------------------------------------------
    async def _synthesise(self, query: str, findings: list[dict]) -> str:
        findings_text = ""
        all_sources = []

        for i, f in enumerate(findings):
            findings_text += f"\n## Research on: {f['question']}\n{f['summary']}\n"
            for s in f.get("sources", []):
                if s not in all_sources:
                    all_sources.append(s)

        sources_section = "\n".join(
            f"- [{s['title']}]({s['url']})" for s in all_sources if s.get("url")
        )

        prompt = f"""You are a senior research analyst writing a comprehensive report.

    Original research query: {query}

    Research findings from multiple sub-topics:
    {findings_text}

    Write a well-structured markdown research report that:
    1. Has a clear title (# heading)
    2. Starts with an executive summary
    3. Organises findings into logical sections with ## headings
    4. Draws conclusions that directly answer the original query
    5. Ends with a "Sources" section

    Available sources for the Sources section:
    {sources_section}

    Write the full report in markdown:"""

        response = await self.llm_report.ainvoke([
            SystemMessage(content="You are an expert research report writer. Write clear, well-structured markdown reports."),
            HumanMessage(content=prompt),
        ])
        return response.content

    # ------------------------------------------------------------------
    # Utility: parse JSON array from LLM output
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_array(text: str, fallback: list[str]) -> list[str]:
        """Extract a JSON array from potentially messy LLM output."""
        # Try to find JSON array in the text
        # Strip thinking tags if model uses them (Qwen3 /think)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Try direct parse
        text = text.strip()
        if text.startswith("["):
            try:
                result = json.loads(text)
                if isinstance(result, list):
                    return [str(item) for item in result if item]
            except json.JSONDecodeError:
                pass

        # Try to find array in text
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(item) for item in result if item]
            except json.JSONDecodeError:
                pass

        # Fallback: split numbered lines
        lines = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip('" ')
            for line in text.split("\n")
            if line.strip() and not line.strip().startswith("{")
        ]
        return [l for l in lines if len(l) > 5] or fallback