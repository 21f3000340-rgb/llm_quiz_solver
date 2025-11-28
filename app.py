# app.py
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
import os
import time

from solver import run_agent
from shared_store import url_time, BASE64_STORE

# Load .env variables
load_dotenv()
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

app = FastAPI()

# ---------------------------------------------------------
# FAVICON SUPPORT
# ---------------------------------------------------------
# Mount static folder (make sure you have static/favicon.ico)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


# ---------------------------------------------------------
# CORS (allow all for testing)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


@app.get("/healthz")
def healthz():
    """Basic server health check."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }


@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    """Start solving the quiz from a given URL."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON received")

    url = data.get("url")
    secret = data.get("secret")

    if not url or not secret:
        raise HTTPException(status_code=400, detail="Missing 'url' or 'secret' in request")

    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret key")

    # Reset previous state
    url_time.clear()
    BASE64_STORE.clear()

    # Store context for solver
    os.environ["url"] = url
    os.environ["offset"] = "0"
    url_time[url] = time.time()

    print(f"Starting solve job for URL: {url}")

    # Run agent in background
    background_tasks.add_task(run_agent, url)

    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)


