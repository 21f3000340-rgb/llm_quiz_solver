# ------------------------------------------------------------
# solver.py — Stable Balanced Mode (works with LangGraph + Gemini)
# ------------------------------------------------------------

import os
import re
import json
import time
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

load_dotenv()
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

# ------------------------------------------------------------
# Base URL extractor
# ------------------------------------------------------------
def extract_base(url: str):
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


# ------------------------------------------------------------
# LangGraph
# ------------------------------------------------------------
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Messages
from langchain_core.messages import HumanMessage

# ------------------------------------------------------------
# Original tool functions
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
# Tool Wrappers (REQUIRED for LangGraph)
# ------------------------------------------------------------
from langchain.tools import tool

@tool
def run_code(**kwargs):
    return _run_code(**kwargs)

@tool
def get_rendered_html(**kwargs):
    return _get_rendered_html(**kwargs)

@tool
def download_file(**kwargs):
    return _download_file(**kwargs)

@tool
def post_request(**kwargs):
    return _post_request(**kwargs)

@tool
def add_dependencies(**kwargs):
    return _add_dependencies(**kwargs)

@tool
def ocr_image_tool(**kwargs):
    return _ocr_image_tool(**kwargs)

@tool
def transcribe_audio(**kwargs):
    return _transcribe_audio(**kwargs)

@tool
def encode_image_to_base64(**kwargs):
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
# LLM (do NOT bind tools)
# ------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings=None,
    convert_system_message_to_human=True
)

# ------------------------------------------------------------
# System Prompt
# ------------------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.
Never reveal internal logic or variables.

Rules:
1. Use only absolute URLs.
2. Use ONLY provided tools for every action.
3. Always include in submissions:
   email = {EMAIL}
   secret = {SECRET}
4. Follow next_url until none remain, then return END.
"""


# ------------------------------------------------------------
# State Type
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    json_fixes: int
    base_url: str


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
MAX_MESSAGES = 40
JSON_FIX_LIMIT = 4
TIMEOUT_LIMIT = 180


def safe_get_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get("text", "")
    return ""


def manual_trim(messages):
    if len(messages) <= MAX_MESSAGES:
        return messages
    return messages[-MAX_MESSAGES:]


def fix_relative_url(url, base):
    if isinstance(url, str) and url.startswith("/"):
        return base.rstrip("/") + url
    return url


def fix_json_try(text):
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")

    try:
        return json.loads(cleaned)
    except:
        pass

    si, ei = cleaned.find("{"), cleaned.rfind("}")
    if si != -1 and ei > si:
        try:
            return json.loads(cleaned[si:ei + 1])
        except:
            pass

    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    try:
        return json.loads(cleaned)
    except:
        return None


# ------------------------------------------------------------
# JSON Fix Node
# ------------------------------------------------------------
def handle_malformed_node(state: AgentState):
    if state["json_fixes"] >= JSON_FIX_LIMIT:
        print("⚠️ Too many JSON fixes")
        return {"messages": []}

    last = state["messages"][-1]
    text = safe_get_content(last.content)

    print("⚠️ Fixing malformed JSON...")

    fixed = fix_json_try(text)
    if fixed:
        last.content = json.dumps(fixed)
        return {"messages": [], "json_fixes": state["json_fixes"] + 1}

    return {
        "messages": [
            HumanMessage(content="Invalid JSON. Return valid JSON only.")
        ],
        "json_fixes": state["json_fixes"] + 1,
    }


# ------------------------------------------------------------
# Agent Node
# ------------------------------------------------------------
def agent_node(state: AgentState):
    cur_url = os.getenv("url")
    now = time.time()
    prev = url_time.get(cur_url)

    # timeout logic
    if prev and now - float(prev) >= TIMEOUT_LIMIT:
        print("⚠️ TIMEOUT → sending WRONG answer")
        forced = HumanMessage(content="Time limit exceeded. Submit WRONG answer.")
        result = llm.invoke(state["messages"] + [forced])
        return {"messages": [result]}

    trimmed = manual_trim(state["messages"])

    print(f"🔁 LLM invoked with {len(trimmed)} messages")
    response = llm.invoke(trimmed)
    return {"messages": [response]}


# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def route(state: AgentState):
    last = state["messages"][-1]
    text = safe_get_content(last.content).strip()

    # Fix relative URLs
    try:
        if text.startswith("{"):
            d = json.loads(text)
            if "url" in d:
                fixed = fix_relative_url(d["url"], state["base_url"])
                if fixed != d["url"]:
                    d["url"] = fixed
                    last.content = json.dumps(d)
    except:
        pass

    if getattr(last, "tool_calls", None):
        return "tools"

    if text == "END":
        return "__end__"

    try:
        json.loads(text)
        return "agent"
    except:
        return "json_fix"


# ------------------------------------------------------------
# Build Graph
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
    }
)

app = graph.compile()


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
def run_agent(url: str):
    base = extract_base(url)

    initial = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    app.invoke(
        {"messages": initial, "json_fixes": 0, "base_url": base},
        config={"recursion_limit": 5000},
    )

    print("🎉 Solver run completed!")
