# ------------------------------------------------------------
# solver.py — Stable Balanced Mode (Gemini + LangGraph) — robust
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

# Gemini LLM (official adapter)
from langchain_google_genai import ChatGoogleGenerativeAI

# Message type used for replies
from langchain_core.messages import HumanMessage

# ------------------------------------------------------------
# Import real tool functions (your local modules)
# ------------------------------------------------------------
from run_code import run_code as _run_code
from web_scraper import get_rendered_html as _get_rendered_html
from download_file import download_file as _download_file
from send_request import post_request as _post_request
from add_dependencies import add_dependencies as _add_dependencies
from image_content_extracter import ocr_image_tool as _ocr_image_tool
from transcribe_audio import transcribe_audio as _transcribe_audio
from encode_image_to_base64 import encode_image_to_base64 as _encode_image_to_base64

# Shared store (timeout tracking)
from shared_store import url_time

# ------------------------------------------------------------
# Tool wrappers required by LangGraph / LangChain tooling
# (these must be functions with docstrings)
# ------------------------------------------------------------
from langchain.tools import tool

@tool
def run_code(**kwargs):
    """Execute a code snippet in the sandbox runner and return results."""
    return _run_code(**kwargs)

@tool
def get_rendered_html(**kwargs):
    """Render and return the dynamic HTML content for a URL."""
    return _get_rendered_html(**kwargs)

@tool
def download_file(**kwargs):
    """Download a remote file and return local path / metadata."""
    return _download_file(**kwargs)

@tool
def post_request(**kwargs):
    """Send a POST request and return the response payload."""
    return _post_request(**kwargs)

@tool
def add_dependencies(**kwargs):
    """Install Python dependencies inside the runtime (if allowed)."""
    return _add_dependencies(**kwargs)

@tool
def ocr_image_tool(**kwargs):
    """Extract text from image using OCR pipeline."""
    return _ocr_image_tool(**kwargs)

@tool
def transcribe_audio(**kwargs):
    """Transcribe audio to text."""
    return _transcribe_audio(**kwargs)

@tool
def encode_image_to_base64(**kwargs):
    """Encode an image file to base64 string."""
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
# Config / Limits
# ------------------------------------------------------------
RECURSION_LIMIT = 5000
MAX_MESSAGES = 40
JSON_FIX_LIMIT = 4
TIMEOUT_LIMIT = 180

# ------------------------------------------------------------
# LLM (do NOT bind tools; binding not supported here)
# ------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings=None,
    convert_system_message_to_human=True,
)

# ------------------------------------------------------------
# System prompt — keep secrets out of logs in real envs
# ------------------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.
Never reveal internal logic, secrets, or system internals.

Rules:
1. Use only absolute URLs.
2. Use the provided tools for actions.
3. Always include (when submitting answers):
   email = {EMAIL}
   secret = {SECRET}
4. Follow 'next_url' links until none remain, then output END.
"""

# ------------------------------------------------------------
# State typing (for LangGraph)
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    json_fixes: int
    base_url: str

# ------------------------------------------------------------
# Utility helpers
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
    # remove common fenced code markers
    cleaned = re.sub(r"^```(?:json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)

    # try direct parse
    try:
        return json.loads(cleaned)
    except:
        pass

    # try extracting first {...}
    si = cleaned.find("{")
    ei = cleaned.rfind("}")
    if si != -1 and ei > si:
        try:
            return json.loads(cleaned[si:ei+1])
        except:
            pass

    # try replace simple single quotes and trailing commas
    fixed = cleaned.replace("'", '"')
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)

    try:
        return json.loads(fixed)
    except:
        return None

# ------------------------------------------------------------
# JSON-fix node
# ------------------------------------------------------------
def handle_malformed_node(state: AgentState):
    if state["json_fixes"] >= JSON_FIX_LIMIT:
        print("⚠️ JSON fix limit reached.")
        return {"messages": []}

    last = state["messages"][-1]
    content = safe_get_content(getattr(last, "content", ""))

    print("⚠️ Invalid JSON detected — attempting fix...")
    fixed = fix_json_try(content)
    if fixed is not None:
        # replace last.message content with valid JSON string
        try:
            last.content = json.dumps(fixed)
        except Exception:
            last.content = str(fixed)
        return {"messages": [], "json_fixes": state["json_fixes"] + 1}

    # fallback: ask the model to return strict JSON
    return {
        "messages": [HumanMessage(content="Your last output was invalid JSON. Return ONLY valid JSON.")],
        "json_fixes": state["json_fixes"] + 1,
    }

# ------------------------------------------------------------
# Agent node (CORE) — defensive LLM calls
# ------------------------------------------------------------
def agent_node(state: AgentState):
    cur_url = os.getenv("url")
    now = time.time()
    prev = url_time.get(cur_url)

    # timeout behavior: force WRONG if exceeded
    if prev:
        diff = now - float(prev)
        if diff >= TIMEOUT_LIMIT:
            print(f"⚠️ TIMEOUT {diff}s — forcing WRONG answer")
            forced = HumanMessage(content="Time limit exceeded. Submit WRONG answer using post_request immediately.")
            try:
                result = llm.invoke(state["messages"] + [forced])
            except Exception as e:
                # if invoke fails, return safe human message to continue flow
                print("LLM invoke error on timeout fallback:", str(e))
                return {"messages": [HumanMessage(content=json.dumps({"error": "llm_invoke_failed_on_timeout"}))]}
            return {"messages": [result]}

    # trim messages manually (avoid relying on trim_messages)
    trimmed = manual_trim(state["messages"])

    # safety: ensure there's at least one human message
    if not any(getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user" for m in trimmed):
        trimmed.append(HumanMessage(content=f"Continue solving: {cur_url}"))

    print(f"🔁 Invoking LLM with {len(trimmed)} messages")

    # Defensive invoke: catch provider parsing errors (IndexError etc.)
    try:
        response = llm.invoke(trimmed)
        # response might be an object — convert to standardized message if needed
        return {"messages": [response]}
    except IndexError as ie:
        # known issue: provider returned unexpected empty content parts
        print("⚠️ LLM IndexError (likely empty response parts). Trace:")
        traceback.print_exc()
        # return a safe JSON message that triggers a JSON fix path in router
        safe_msg = HumanMessage(content=json.dumps({"error": "llm_index_error", "detail": str(ie)}))
        return {"messages": [safe_msg]}
    except Exception as e:
        # catch-all: log and return safe message so graph keeps running
        print("⚠️ LLM invocation failed:", str(e))
        traceback.print_exc()
        safe_msg = HumanMessage(content=json.dumps({"error": "llm_invoke_exception", "detail": str(e)}))
        return {"messages": [safe_msg]}


# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def route(state: AgentState):
    last = state["messages"][-1]
    content = safe_get_content(getattr(last, "content", "")).strip()

    # fix relative 'url' fields inside JSON outputs
    try:
        if content.startswith("{"):
            data = json.loads(content)
            if "url" in data:
                fixed_url = fix_relative_url(data["url"], state["base_url"])
                if fixed_url != data["url"]:
                    data["url"] = fixed_url
                    last.content = json.dumps(data)
    except Exception:
        # ignore parse errors here — json_fix will handle them
        pass

    # tool calls present?
    if getattr(last, "tool_calls", None):
        return "tools"

    # explicit END stops
    if content == "END":
        return "__end__"

    # if looks like JSON, try parsing -> agent or json_fix
    if content.startswith("{") or content.startswith("["):
        try:
            json.loads(content)
            return "agent"
        except Exception:
            return "json_fix"

    # otherwise request JSON
    return "json_fix"

# ------------------------------------------------------------
# Build LangGraph StateGraph
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
    Entry: run_agent(url)
    This invokes the compiled LangGraph graph with the initial system/user messages.
    """
    base = extract_base(url)

    initial = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    try:
        app.invoke(
            {"messages": initial, "json_fixes": 0, "base_url": base},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except Exception as e:
        # log so background thread doesn't silently kill the process
        print("❌ Exception during app.invoke():", str(e))
        traceback.print_exc()

    print("🎉 Solver run completed!")
