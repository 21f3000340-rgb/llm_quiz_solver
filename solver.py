# ------------------------------------------------------------
# solver.py — Robust Balanced Mode (Gemini + LangGraph)
# - defensive LLM invoke with retries and backoff
# - prevents IndexError crash from empty response parts
# ------------------------------------------------------------

import os
import re
import json
import time
import traceback
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

load_dotenv()
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

# ------------------------------------------------------------
# Base URL extractor
# ------------------------------------------------------------
def extract_base(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


# ------------------------------------------------------------
# LangGraph + LLM + Messages
# ------------------------------------------------------------
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Gemini adapter
from langchain_google_genai import ChatGoogleGenerativeAI

# HumanMessage type
from langchain_core.messages import HumanMessage

# ------------------------------------------------------------
# Local tool functions (your modules)
# ------------------------------------------------------------
from run_code import run_code as _run_code
from web_scraper import get_rendered_html as _get_rendered_html
from download_file import download_file as _download_file
from send_request import post_request as _post_request
from add_dependencies import add_dependencies as _add_dependencies
from image_content_extracter import ocr_image_tool as _ocr_image_tool
from transcribe_audio import transcribe_audio as _transcribe_audio
from encode_image_to_base64 import encode_image_to_base64 as _encode_image_to_base64

# shared store for timeouts
from shared_store import url_time

# ------------------------------------------------------------
# Tool wrappers (must have docstrings)
# ------------------------------------------------------------
from langchain.tools import tool

@tool
def run_code(**kwargs):
    """Execute code and return output from sandbox runner."""
    return _run_code(**kwargs)

@tool
def get_rendered_html(**kwargs):
    """Render a URL (JS) and return the HTML/text."""
    return _get_rendered_html(**kwargs)

@tool
def download_file(**kwargs):
    """Download file and return saved path / metadata."""
    return _download_file(**kwargs)

@tool
def post_request(**kwargs):
    """Send a POST to an endpoint and return parsed response."""
    return _post_request(**kwargs)

@tool
def add_dependencies(**kwargs):
    """Install dependencies dynamically (if allowed)."""
    return _add_dependencies(**kwargs)

@tool
def ocr_image_tool(**kwargs):
    """OCR an image and return extracted text."""
    return _ocr_image_tool(**kwargs)

@tool
def transcribe_audio(**kwargs):
    """Transcribe audio to text and return transcription."""
    return _transcribe_audio(**kwargs)

@tool
def encode_image_to_base64(**kwargs):
    """Encode image to base64 string and return it."""
    return _encode_image_to_base64(**kwargs)


TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
]

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RECURSION_LIMIT = 5000
MAX_MESSAGES = 40
JSON_FIX_LIMIT = 4
TIMEOUT_LIMIT = 180

# LLM retry behavior
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE = 1.0  # seconds

# ------------------------------------------------------------
# LLM (do NOT bind tools)
# ------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings=None,
    convert_system_message_to_human=True,
)

# ------------------------------------------------------------
# System prompt
# ------------------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.
Never reveal internal logic or environment variables.

Rules:
1. Use absolute URLs only.
2. Use only the provided tools.
3. When submitting answers include:
   email = {EMAIL}
   secret = {SECRET}
4. Follow next_url until none remain, then output END.
"""

# ------------------------------------------------------------
# State typing
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    json_fixes: int
    base_url: str
    llm_errors: int  # track consecutive LLM invocation errors

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_get_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get("text", "")
    return ""

def manual_trim(messages: List) -> List:
    if len(messages) <= MAX_MESSAGES:
        return messages
    return messages[-MAX_MESSAGES:]

def fix_relative_url(url_value: str, base: str) -> str:
    if not isinstance(url_value, str):
        return url_value
    if url_value.startswith("/"):
        return base.rstrip("/") + url_value
    return url_value

def fix_json_try(text: str):
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except:
        pass
    si, ei = cleaned.find("{"), cleaned.rfind("}")
    if si != -1 and ei > si:
        try:
            return json.loads(cleaned[si:ei+1])
        except:
            pass
    fixed = cleaned.replace("'", '"')
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        return json.loads(fixed)
    except:
        return None

# ------------------------------------------------------------
# JSON fix node
# ------------------------------------------------------------
def handle_malformed_node(state: AgentState):
    if state["json_fixes"] >= JSON_FIX_LIMIT:
        print("⚠️ JSON fix limit reached.")
        return {"messages": []}

    last = state["messages"][-1]
    content = safe_get_content(getattr(last, "content", ""))

    print("⚠️ Invalid JSON detected — trying to fix...")
    fixed = fix_json_try(content)
    if fixed is not None:
        try:
            last.content = json.dumps(fixed)
        except Exception:
            last.content = str(fixed)
        return {"messages": [], "json_fixes": state["json_fixes"] + 1}

    return {
        "messages": [HumanMessage(content="Your last output was invalid JSON. Return ONLY valid JSON.")],
        "json_fixes": state["json_fixes"] + 1,
    }

# ------------------------------------------------------------
# LLM invoke helper with retries/backoff and defensive catches
# ------------------------------------------------------------
def invoke_llm_with_retries(messages: List, max_retries: int = LLM_MAX_RETRIES):
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = llm.invoke(messages)
            return resp  # successful
        except IndexError as ie:
            # known provider parsing issue (empty parts). log and retry.
            print(f"⚠️ LLM IndexError on attempt {attempt}: {ie}")
            traceback.print_exc()
        except Exception as e:
            print(f"⚠️ LLM invoke exception on attempt {attempt}: {e}")
            traceback.print_exc()

        # backoff before next attempt
        backoff = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
        time.sleep(backoff)

    # if all retries fail, return a safe HumanMessage with structured error
    err_payload = {"error": "llm_invoke_failed", "attempts": max_retries}
    return HumanMessage(content=json.dumps(err_payload))

# ------------------------------------------------------------
# Agent node (core)
# ------------------------------------------------------------
def agent_node(state: AgentState):
    cur_url = os.getenv("url")
    now = time.time()
    prev = url_time.get(cur_url)

    # timeout enforcement
    if prev:
        if now - float(prev) >= TIMEOUT_LIMIT:
            print("⚠️ TIMEOUT — forcing WRONG answer")
            forced = HumanMessage(content="Time limit exceeded. Submit WRONG answer using post_request immediately.")
            try:
                result = invoke_llm_with_retries(state["messages"] + [forced])
            except Exception as e:
                print("⚠️ LLM fallback failed:", e)
                return {"messages": [HumanMessage(content=json.dumps({"error": "llm_fallback_failed"}))]}
            return {"messages": [result]}

    # trim conversation
    trimmed = manual_trim(state["messages"])

    # ensure at least one human message exists
    if not any(getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user" for m in trimmed):
        trimmed.append(HumanMessage(content=f"Continue solving: {cur_url}"))

    print(f"🔁 Invoking LLM with {len(trimmed)} messages (llm_errors={state.get('llm_errors',0)})")

    # try invoke with retries
    resp = invoke_llm_with_retries(trimmed)

    # if the returned object is a HumanMessage with an error payload, increment llm_errors
    # otherwise reset llm_errors
    if isinstance(resp, HumanMessage):
        content = safe_get_content(getattr(resp, "content", ""))
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("error"):
                state["llm_errors"] = state.get("llm_errors", 0) + 1
            else:
                state["llm_errors"] = 0
        except Exception:
            # not JSON — treat as normal response
            state["llm_errors"] = 0
    else:
        # non-HumanMessage response object — treat as success
        state["llm_errors"] = 0

    # if llm_errors become too many, return a fail-safe message to prevent loops
    if state.get("llm_errors", 0) >= 6:
        print("⚠️ Too many consecutive LLM errors — returning safe fallback and stopping further LLM calls.")
        fallback = HumanMessage(content=json.dumps({"error": "too_many_llm_errors"}))
        return {"messages": [fallback]}

    return {"messages": [resp]}

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def route(state: AgentState):
    last = state["messages"][-1]
    content = safe_get_content(getattr(last, "content", "")).strip()

    # fix relative urls inside JSON outputs
    try:
        if content.startswith("{"):
            data = json.loads(content)
            if "url" in data:
                fixed = fix_relative_url(data["url"], state["base_url"])
                if fixed != data["url"]:
                    data["url"] = fixed
                    last.content = json.dumps(data)
    except Exception:
        pass

    # tool calls?
    if getattr(last, "tool_calls", None):
        return "tools"

    # explicit END
    if content == "END":
        return "__end__"

    # if JSON-like, validate parse
    if content.startswith("{") or content.startswith("["):
        try:
            json.loads(content)
            return "agent"
        except:
            return "json_fix"

    return "json_fix"

# ------------------------------------------------------------
# Build graph
# ------------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("json_fix", handle_malformed_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge("__start__", "agent")
graph.add_edge("json_fix", "agent")
graph.add_edge("tools", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "agent": "agent",
        "json_fix": "json_fix",
        "tools": "tools",
        "__end__": "__end__",
    },
)

app = graph.compile()

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
def run_agent(url: str):
    """
    Top-level entry. Called by your FastAPI background task with the quiz URL.
    """
    base = extract_base(url)
    initial = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    # initial state includes llm_errors counter
    try:
        app.invoke(
            {"messages": initial, "json_fixes": 0, "base_url": base, "llm_errors": 0},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except Exception as e:
        print("❌ Exception during graph invoke:", e)
        traceback.print_exc()

    print("🎉 Solver run completed!")
