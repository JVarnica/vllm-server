
from __future__ import annotations
import operator
from typing import TypedDict, Annotated, Literal
from typing_extensions import NotRequired
from pydantic import BaseModel, Field


# Custom reducer for cross-node URL dedup
def union_unique(left: list[str], right: list[str]) -> list[str]:
    """Merge two lists preserving order, dropping duplicates. 
    Used by `seen_urls` so parallel search nodes can share what's been scraped."""
    seen = set(left)
    out = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

class Query(BaseModel):
    """A search query with a one-line rationale. The rationale forces 
    the model to commit to *why* it's running this query."""
    id: str
    query: str = Field(description="2-6 keyword search query, no full sentences")
    rationale: str = Field(description="Why this query advances the research, one sentence")


class Document(BaseModel):
    """A raw search hit with scraped content. Cold storage — never sent 
    whole to the LLM. The summarizer reads it one at a time."""
    id: str
    url: str
    title: str
    raw_content: str
    source_query_id: str
    search_score: float = 0.0


class DocSummary(BaseModel):
    """Compressed view of a single doc. ~150-300 tokens.
    Schema fields are deliberate: `relevant` forces a yes/no commitment, 
    `key_findings` forces specificity, `quotes` provide auditable evidence."""
    doc_id: str
    url: str
    relevant: bool = Field(description="Does this doc actually help answer the question?")
    key_findings: list[str] = Field(
        description="3-5 specific findings, each one sentence. Empty list if not relevant.",
        max_length=5,
    )
    quotes: list[str] = Field(
        description="Up to 2 short supporting quotes (<25 words each)",
        max_length=2,
    )


class Claim(BaseModel):
    """A specific factual claim with provenance. Built by aggregating 
    findings across docs. The planner organizes these into sections."""
    id: str
    statement: str = Field(description="One specific factual claim, one sentence")
    source_doc_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class SectionPlan(BaseModel):
    """A planned section. `claim_ids` is the contract — the section writer 
    only gets these claims, nothing else."""
    id: str
    title: str
    angle: str = Field(description="What this section argues or covers, one sentence")
    claim_ids: list[str] = Field(min_length=1)


class ReportPlan(BaseModel):
    title: str
    sections: list[SectionPlan] = Field(min_length=3, max_length=6)


class WrittenSection(BaseModel):
    """A section after the writer has filled it in. `citations_used` lets 
    the stitcher build the reference list and lets us audit grounding."""
    id: str
    title: str
    body_markdown: str
    citations_used: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """The reflection node decides whether to loop or stop."""
    is_sufficient: bool
    knowledge_gap: str = Field(description="What's still missing, one sentence")
    follow_up_queries: list[Query] = Field(default_factory=list, max_length=4)


#Overall state
class OverallState(TypedDict):
    # ---- Input ----
    task_id: str
    original_query: str
    max_research_loops: int

    # ---- Accumulated across the run (parallel-safe via reducers) ----
    search_queries: Annotated[list[Query], operator.add]
    raw_docs: Annotated[list[Document], operator.add]
    doc_summaries: Annotated[list[DocSummary], operator.add]
    claims: Annotated[list[Claim], operator.add]
    seen_urls: Annotated[list[str], union_unique]
    written_sections: Annotated[list[WrittenSection], operator.add]

    # ---- Loop control (single-writer, overwrite semantics) ----
    research_loop_count: int
    is_sufficient: bool

    # ---- Set mid-run by specific nodes ----
    plan: NotRequired[ReportPlan] #not required optional as first nodes wont ouput anything to it
    final_report: NotRequired[str]
