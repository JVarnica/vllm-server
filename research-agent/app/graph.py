from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import AsyncRedisSaver

from .state import OverallState
from .nodes import (
    generate_queries,
    fan_out_search, search_one_query,
    fan_out_summarize, summarize_one_doc,
    extract_claims,
    reflect, reflect_router,
    plan_report,
    fan_out_sections, write_section,
    stitch_report,
)


def build_graph(checkpointer=None):
    g = StateGraph(OverallState)
    
    # ---- Nodes ----
    g.add_node("generate_queries", generate_queries)
    g.add_node("search_one_query", search_one_query)        # parallel target
    g.add_node("summarize_one_doc", summarize_one_doc)      # parallel target
    g.add_node("extract_claims", extract_claims)
    g.add_node("reflect", reflect)
    g.add_node("plan_report", plan_report)
    g.add_node("write_section", write_section)              # parallel target
    g.add_node("stitch_report", stitch_report)
    
    # ---- Edges ----
    g.add_edge(START, "generate_queries")
    
    # Fan-out to search (conditional edge returning list[Send])
    g.add_conditional_edges("generate_queries", fan_out_search, ["search_one_query"])
    
    # After all search branches finish, fan out to summarize
    g.add_conditional_edges("search_one_query", fan_out_summarize, ["summarize_one_doc"])
    
    # After all summarize branches finish, extract claims
    g.add_edge("summarize_one_doc", "extract_claims")
    
    # Then reflect, which decides: loop back to search or move on
    g.add_edge("extract_claims", "reflect")
    g.add_conditional_edges("reflect", reflect_router, ["search_one_query", "plan_report"])
    # Note: on loop-back, reflect_router returns "search_one_query" but we want fan-out behavior.
    # See note below — we route via the fan-out function.
    
    g.add_conditional_edges("plan_report", fan_out_sections, ["write_section"])
    g.add_edge("write_section", "stitch_report")
    g.add_edge("stitch_report", END)
    
    return g.compile(checkpointer=checkpointer)