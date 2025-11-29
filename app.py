from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from solver import run_agent
from dotenv import load_dotenv
import uvicorn
import os
import time

from shared_store import url_time, BASE64_STORE

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


# ------------------------------------------------------------
# HEALTH CHECK ENDPOINT (Railway needs /health)
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }


# Optional extra endpoint you already had
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }


# ------------------------------------------------------------
# SOLVER ENDPOINT
# ------------------------------------------------------------
@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):

    # Parse JSON body
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    secret = data.get("secret")

    if not url or not secret:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Auth check
    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Reset per-run caches
    url_time.clear()
    BASE64_STORE.clear()

    # Pass to solver
    os.environ["url"] = url
    os.environ["offset"] = "0"
    url_time[url] = time.time()

    background_tasks.add_task(run_agent, url)

    return JSONResponse(status_code=200, content={"status": "ok"})


# ------------------------------------------------------------
# RUN SERVER (Railway-compatible)
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
