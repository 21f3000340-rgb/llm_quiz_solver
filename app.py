from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os, re, json, logging
from solver import solve_quiz_task

# ----------------------------------------------------
# ENVIRONMENT & SETUP
# ----------------------------------------------------
load_dotenv()
SECRET = os.getenv("USER_SECRET")
EMAIL = os.getenv("USER_EMAIL")
GITHUB_REPO = os.getenv("GITHUB_REPO")

logger = logging.getLogger("quiz_guard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)

app = FastAPI(
    title="Quiz Solver API (Gemini Edition)",
    description="Solves quizzes & analyzes data safely using Gemini.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ----------------------------------------------------
# LEAK DETECTION HELPERS
# ----------------------------------------------------
def detect_leak(text: str, secrets: list[str]) -> list[str]:
    found = []
    t = text.lower()
    for s in secrets:
        if not s:
            continue
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", t):
            found.append(s)
    return found

def sanitize_output(data: dict, secrets: list[str]):
    combined = json.dumps(data)
    leaked = detect_leak(combined, secrets)
    if leaked:
        logger.warning(f"🚨 Leak detected: {leaked}")
        for key, val in data.items():
            if isinstance(val, str):
                for s in leaked:
                    data[key] = re.sub(r"\b" + re.escape(s) + r"\b", "[REDACTED]", val, flags=re.IGNORECASE)
        data["safety_action"] = {
            "action": "redacted",
            "leaked_tokens": leaked,
            "note": "Sensitive content removed from output."
        }
    return data

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to the Quiz Solver API (Gemini Edition)! 🎯",
        "powered_by": "Google Gemini + FastAPI",
        "repo": GITHUB_REPO or "Not linked",
        "example_payload": {
            "email": EMAIL or "example@example.com",
            "secret": "your_secret_here",
            "url": "https://example.com/quiz"
        },
        "endpoints": {
            "solve_quiz": "/solve_quiz (POST)",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/solve_quiz")
async def solve_quiz(payload: QuizRequest):
    if payload.secret != SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret")

    secrets_to_check = ["elephant", "tiger", "umbrella"]  # during tests, replace dynamically

    result = await solve_quiz_task(payload.dict())
    data = result.get("data", {})
    clean_data = sanitize_output(data, secrets_to_check)
    return {"status": "success", "data": clean_data}

@app.get("/health")
def health():
    return {"status": "ok", "message": "Quiz Solver API running safely ✅"}

# ----------------------------------------------------
# FAVICON ROUTE
# ----------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    file_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(file_path)
