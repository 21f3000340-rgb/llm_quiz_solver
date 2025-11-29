# ------------------------------------------------------------
# solver.py — Robust Balanced Mode (Gemini + LangGraph)
# - defensive LLM invoke with retries and backoff
# - prevents IndexError crash from empty response parts
# - rate limiting protection
# - enhanced error handling
# ------------------------------------------------------------

import os
import re
import json
import time
import traceback
from typing import TypedDict, List, Annotated
from collections import deque
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
# Rate Limiter Class
# ------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_requests=8, time_window=60):
        """
        max_requests: Max requests allowed (set to 8 to stay under 10/min limit)
        time_window: Time window in seconds (60s = 1 minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def wait_if_needed(self):
        now = time.time()
        
        # Remove requests older than time_window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # If at limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0]) + 1
            print(f"⏳ Rate limit protection: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            self.requests.popleft()
        
        # Record this request
        self.requests.append(time.time())

# Create global rate limiter
rate_limiter = RateLimiter(max_requests=8, time_window=60)


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
LLM_BACKOFF_BASE = 2.0  # seconds

# ------------------------------------------------------------
# LLM (do NOT bind tools)
# ------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings=None,
    convert_system_message_to_human=True,
    temperature=0.7,
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
    """Safely extract text content from various content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        if not content:
            return ""
        if isinstance(content[0], dict):
            return content[0].get("text", "")
        if isinstance(content[0], str):
            return content[0]
    return ""

def manual_trim(messages: List) -> List:
    """Trim messages to MAX_MESSAGES limit."""
    if len(messages) <= MAX_MESSAGES:
        return messages
    return messages[-MAX_MESSAGES:]

def fix_relative_url(url_value: str, base: str) -> str:
    """Convert relative URLs to absolute URLs."""
    if not isinstance(url_value, str):
        return url_value
    if url_value.startswith("/"):
        return base.rstrip("/") + url_value
    return url_value

def fix_json_try(text: str):
    """Attempt to fix and parse malformed JSON."""
    if not isinstance(text, str):
        return None
    
    cleaned = text.strip()
    
    # Remove markdown code fences
    cleaned = re.sub(r"^```(?:json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except:
        pass
    
    # Try extracting JSON object/array
    si, ei = cleaned.find("{"), cleaned.rfind("}")
    if si != -1 and ei > si:
        try:
            return json.loads(cleaned[si:ei+1])
        except:
            pass
    
    # Try fixing common issues
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
    """Handle malformed JSON responses from LLM."""
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
        "messages": [HumanMessage(content="Your last output was invalid JSON. Return ONLY valid JSON with no markdown formatting.")],
        "json_fixes": state["json_fixes"] + 1,
    }

# ------------------------------------------------------------
# LLM invoke helper with retries/backoff and defensive catches
# ------------------------------------------------------------
def invoke_llm_with_retries(messages: List, max_retries: int = LLM_MAX_RETRIES):
    """
    Invoke LLM with retry logic, rate limiting, and error handling.
    Returns either a valid response or a HumanMessage with error payload.
    """
    attempt = 0
    
    while attempt < max_retries:
        attempt += 1
        
        # Apply rate limiting before each attempt
        try:
            rate_limiter.wait_if_needed()
        except Exception as e:
            print(f"⚠️ Rate limiter error: {e}")
        
        try:
            print(f"🤖 LLM invoke attempt {attempt}/{max_retries}")
            resp = llm.invoke(messages)
            
            # Validate response has content
            content = safe_get_content(getattr(resp, "content", ""))
            if not content or content.strip() == "":
                print(f"⚠️ Empty response on attempt {attempt}, retrying...")
                raise ValueError("Empty response from LLM")
            
            print(f"✅ LLM responded successfully (content length: {len(content)})")
            return resp  # successful with content
            
        except IndexError as ie:
            # Known Gemini API issue with empty parts
            print(f"⚠️ LLM IndexError on attempt {attempt}: {ie}")
            traceback.print_exc()
            if attempt >= max_retries:
                print("❌ Max retries reached for IndexError")
                break
                
        except ValueError as ve:
            # Empty content error
            print(f"⚠️ Empty content on attempt {attempt}: {ve}")
            if attempt >= max_retries:
                print("❌ Max retries reached for empty content")
                break
                
        except Exception as e:
            error_str = str(e)
            print(f"⚠️ LLM invoke exception on attempt {attempt}: {e}")
            
            # Handle rate limit errors specially
            if "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower():
                wait_time = 65  # Wait just over 1 minute
                print(f"⚠️ Rate limit hit! Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue  # Don't count as a retry attempt
            
            # Handle other API errors
            if "503" in error_str or "500" in error_str:
                print("⚠️ API server error, retrying...")
            
            if attempt >= max_retries:
                print("❌ Max retries reached")
                break
        
        # Exponential backoff before next retry
        backoff = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
        print(f"⏳ Waiting {backoff}s before retry...")
        time.sleep(backoff)

    # All retries failed - return error payload
    print("❌ All LLM invocation attempts failed")
    err_payload = {"error": "llm_invoke_failed", "attempts": max_retries}
    return HumanMessage(content=json.dumps(err_payload))

# ------------------------------------------------------------
# Agent node (core)
# ------------------------------------------------------------
def agent_node(state: AgentState):
    """Main agent logic - invokes LLM and handles responses."""
    cur_url = os.getenv("url")
    now = time.time()
    prev = url_time.get(cur_url)

    # Timeout enforcement
    if prev:
        if now - float(prev) >= TIMEOUT_LIMIT:
            print("⚠️ TIMEOUT — forcing WRONG answer")
            forced = HumanMessage(content="Time limit exceeded. Submit WRONG answer using post_request immediately.")
            result = invoke_llm_with_retries(state["messages"] + [forced])
            return {"messages": [result]}

    # Trim conversation to prevent context overflow
    trimmed = manual_trim(state["messages"])

    # Ensure at least one human message exists
    if not any(getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user" for m in trimmed):
        trimmed.append(HumanMessage(content=f"Continue solving: {cur_url}"))

    print(f"🔁 Invoking LLM with {len(trimmed)} messages (llm_errors={state.get('llm_errors',0)})")

    # Invoke LLM with retry logic
    resp = invoke_llm_with_retries(trimmed)

    # Check if response contains error payload
    if isinstance(resp, HumanMessage):
        content = safe_get_content(getattr(resp, "content", ""))
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "error" in parsed:
                state["llm_errors"] = state.get("llm_errors", 0) + 1
                print(f"❌ LLM error count: {state['llm_errors']}")
            else:
                state["llm_errors"] = 0
        except:
            # Not JSON, treat as normal response
            state["llm_errors"] = 0
    else:
        # Non-HumanMessage response - success
        state["llm_errors"] = 0

    # Stop if too many consecutive errors
    if state.get("llm_errors", 0) >= 3:
        print("❌ Too many consecutive LLM errors — aborting solver")
        return {"messages": [HumanMessage(content="END")]}

    return {"messages": [resp]}

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def route(state: AgentState):
    """Route to next node based on last message content."""
    last = state["messages"][-1]
    content = safe_get_content(getattr(last, "content", "")).strip()

    # Fix relative URLs inside JSON outputs
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

    # Check for tool calls
    if getattr(last, "tool_calls", None):
        print("→ Routing to tools")
        return "tools"

    # Explicit END signal
    if content == "END":
        print("→ Routing to END")
        return "__end__"

    # Validate JSON-like content
    if content.startswith("{") or content.startswith("["):
        try:
            json.loads(content)
            print("→ Valid JSON, routing to agent")
            return "agent"
        except:
            print("→ Invalid JSON, routing to json_fix")
            return "json_fix"

    # Default to JSON fix for non-JSON content
    print("→ Non-JSON content, routing to json_fix")
    return "json_fix"

# ------------------------------------------------------------
# Build graph
# ------------------------------------------------------------
print("🔧 Building LangGraph workflow...")

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
print("✅ LangGraph workflow compiled successfully")

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
def run_agent(url: str):
    """
    Top-level entry. Called by your FastAPI background task with the quiz URL.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting solver for: {url}")
    print(f"{'='*60}\n")
    
    base = extract_base(url)
    initial = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    # Initial state includes llm_errors counter
    try:
        app.invoke(
            {"messages": initial, "json_fixes": 0, "base_url": base, "llm_errors": 0},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        print("\n🎉 Solver run completed successfully!")
    except Exception as e:
        print(f"\n❌ Exception during graph invoke: {e}")
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Solver finished")
    print(f"{'='*60}\n")
