[1mdiff --git a/home/julien/Documents/vllm_chat/mobile/gateway/app/context2.py b/home/julien/Downloads/context(3).py[m
[1mindex 92bc02c..6591365 100644[m
[1m--- a/home/julien/Documents/vllm_chat/mobile/gateway/app/context2.py[m
[1m+++ b/home/julien/Downloads/context(3).py[m
[36m@@ -1,7 +1,6 @@[m
 import os[m
 import re[m
 import math[m
[31m-import random[m
 from datetime import datetime, timedelta, timezone[m
 from typing import Optional[m
 from urllib.parse import urlsplit[m
[36m@@ -22,7 +21,6 @@[m [mS_CONTEXT_MAX_CHARS = 20000[m
 SEARCH_CANDIDATES = 12          # fetch more, then rerank[m
 SCRAPE_TOP_N = 5               # scrape best reranked results (must be >= CONTEXT_TOP_N)[m
 CONTEXT_TOP_N = 5              # results shown to the model; keep aligned with SCRAPE_TOP_N[m
[31m-SCRAPE_MAX_CHARS = 5000        # legacy per-URL cap (used by scrape_urls)[m
 SCRAPE_RAW_MAX_CHARS = 60000   # safety cap before passage selection[m
 [m
 # ---- Global passage selection -------------------------------------------------[m
[36m@@ -57,30 +55,23 @@[m [m_MONTHS = {[m
     "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,[m
 }[m
 [m
[31m-# ---- Passage-selection tracing ------------------------------------------------[m
[31m-# Emits a nested span whose INPUT is the raw extracted page text - i.e. the actual[m
[31m-# inputs to passage scoring, including the passages that lose. Trace/tool content[m
[31m-# is post-selection, so it cannot be used to replay a different passage scorer.[m
[31m-# Langfuse truncates events at ~1MB and silently drops input/output/metadata to[m
[31m-# fit, so the payload is capped well under that.[m
[31m-PASSAGE_TRACE_ENABLED = os.environ.get("PASSAGE_TRACE_ENABLED", "1") not in ("0", "false", "False")[m
[31m-PASSAGE_TRACE_SAMPLE_RATE = float(os.environ.get("PASSAGE_TRACE_SAMPLE_RATE", "1.0"))[m
[31m-PASSAGE_TRACE_MAX_CHARS_PER_DOC = int(os.environ.get("PASSAGE_TRACE_MAX_CHARS_PER_DOC", "12000"))[m
[31m-PASSAGE_TRACE_MAX_TOTAL_CHARS = int(os.environ.get("PASSAGE_TRACE_MAX_TOTAL_CHARS", "60000"))[m
[31m-[m
[31m-# Phrases that mark a question as historical even if a recency word appears[m
[31m-# nearby ("who was the first current-affairs presenter" etc).[m
[31m-# Only phrases that are unambiguously about the past. The earlier list included[m
[31m-# "during the", "before the" and "the first person to", which fire on plainly[m
[31m-# current questions ("pollen count during the summer", "first person to run a[m
[31m-# sub-2-hour marathon") - and since this list no longer overrides the model, a[m
[31m-# false positive here would only ever mislead the fallback path anyway.[m
[32m+[m[32m# ---- Raw-text tracing --------------------------------------------------------[m
[32m+[m[32m# One span, emitted only when TRACES_DATASET=1. Its input is the raw extracted[m
[32m+[m[32m# page text - the actual input to passage scoring, including the passages that[m
[32m+[m[32m# lose. tool_content is post-selection, so this is the only place they exist.[m
[32m+[m[32mTRACES_DATASET = os.environ.get("TRACES_DATASET", "0") == "1"[m
[32m+[m[32mTRACE_MAX_CHARS_PER_DOC = 12000   # Langfuse truncates ~1MB events, dropping fields silently[m
[32m+[m
[32m+[m
[32m+[m[32m# Phrases that are unambiguously about the past. Deliberately narrow: broader[m
[32m+[m[32m# substrings like "during the" or "the first person to" fire on plainly current[m
[32m+[m[32m# questions ("pollen count during the summer", "first person to run a sub-2-hour[m
[32m+[m[32m# marathon"). Only consulted when the model declares nothing.[m
 _HISTORICAL_PHRASES = ([m
     "history of", "who invented", "who founded", "was born", "were born",[m
     "in ancient", "used to be", "in the 19", "in the 18",[m
 )[m
 [m
[31m-SEARXNG_SAFESEARCH = 1         # 0=off, 1=moderate, 2=strict[m
 SEARXNG_LANGUAGE = "en"[m
 SEARXNG_CATEGORY = "general"[m
 [m
[36m@@ -103,9 +94,8 @@[m [m_STOPWORDS = {[m
 [m
 _CURRENT_TERMS = {[m
     "current", "currently", "lates