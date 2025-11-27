<p align="center">
  <img src="logo.png" alt="Quiz Solver Logo" width="180"/>
</p>

<h1 align="center">🌑🧠 Quiz Solver API — Gemini Edition</h1>

<p align="center">
A secure, containerized AI backend that solves quizzes, scrapes web data, processes files,  
and generates intelligent insights — powered by Google Gemini and FastAPI.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-Framework-009688?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Docker-Ready-0db7ed?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Railway-Deployed-111111?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## 🌌 Overview (Dark Theme Styled)

This backend is built for **Data Science Project 2**, designed to autonomously:

✨ Scrape quizzes (even JavaScript-rendered)  
✨ Parse CSV, XLSX, PDFs, text  
✨ Analyze and transform datasets  
✨ Run statistical or ML-like reasoning  
✨ Generate slide-style summaries  
✨ Return charts as base64 images  

Everything runs safely with:

- Secret-leak prevention  
- 3-minute retry logic (matching instructor rules)  
- Input validation  
- Clean Docker deployment  

---

## 🧠 Key Features

### **✔ Autonomous multi-page quiz solving**
Follows the chain of pages until no next URL is given.

### **✔ True 3-minute retry window**
If you answer wrong → retries allowed for 3 minutes.  
Your latest answer overrides the previous ones.

### **✔ Safe output sanitization**
Blocks accidental reveal of secret words.

### **✔ Multi-modal and multi-source data handling**

- HTML (static + JS rendered with Playwright)  
- JSON APIs  
- CSV / Excel  
- PDF extraction  
- DataFrames  

### **✔ Fully containerized & cloud ready**
Runs seamlessly on **Railway**, **Docker Desktop**, **Render**, **Azure**, etc.

---

## ⚙️ Tech Stack (Dark Mode)

| Component | Technology |
|----------|------------|
| Backend | FastAPI |
| AI Model | Gemini 2.5 Flash |
| Web Scraping | Playwright (Chromium) |
| Deployment | Docker + Railway |
| Language | Python 3.12 |
| Server | Uvicorn |

---

## 📁 Project Structure
llm_quiz_solver/
│
├── app.py # FastAPI backend + retry logic + leak detection
├── solver.py # Core quiz solver using Gemini
│
├── tools/ # Optional helper tool scripts (scraper, downloader etc.)
│
├── Dockerfile # Production-ready Docker build
├── requirements.txt # Python dependencies
├── Procfile # Railway start command (optional)
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md

---

# 📦 Installation

### **Prerequisites**
- Python **3.12+**
- Docker (optional)
- Railway account (deployment)
- Google Gemini API key

---

# 🛠️ Installation Steps

## **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/llm_quiz_solver.git
cd llm_quiz_solver


---
2. Install dependencies
Option A: Using pip
pip install -r requirements.txt
playwright install chromium
Option B: Using Docker (recommended)
docker build -t quiz-solver .

⚙️ Configuration

Create a .env file:
USER_EMAIL=your_email@example.com
USER_SECRET=your_secret_key
GITHUB_REPO=https://github.com/yourusername/llm_quiz_solver
GEMINI_API_KEY=your_gemini_api_key_here
Never commit .env to GitHub.

🚀 Usage
Start the server (pip)
uvicorn app:app --host 0.0.0.0 --port 8000

Start the server (Docker)
docker run --env-file .env -p 8000:8000 quiz-solver

Test API
curl -X POST http://localhost:8000/solve_quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your.email@example.com",
    "secret": "your_secret_string",
    "url": "https://example.com/quiz"
  }'

🌐 API Endpoints
POST /solve_quiz
Starts solving a quiz.

GET /health

Returns:
{"status":"ok","message":"Quiz Solver API running safely ✅"}

🛠 Tools & Capabilities

Your solver can:

1. Scrape JS websites

via Playwright Chromium.

2. Load APIs

Handles JSON endpoints; auto-detects list/dict structures.

3. Parse Data Files

CSV, XLSX, PDF via PyPDF2.

4. Run LLM reasoning

Gemini analyzes patterns, quizzes, slides, insights.

5. Generate charts

Charts returned as data:image/png;base64,....

6. Follow next_url chains

Until quiz ends.

🧠 How It Works
1. Request Received

FastAPI validates email + secret
Starts 3-minute session window.

2. Solver Loads Content

Depending on type:

HTML

JS-rendered page

CSV/XLSX

JSON API

PDF

3. Gemini Processing

LLM creates:

summary

analysis

QA

slides

chart

next_url

4. Session Memory

Your API tracks
"latest submission within 3 minutes"
matching official rules.

5. Continue Chain

If next_url exists → solve next URL.

6. End Condition

When LLM returns no new URL → quiz completed.

## 📄 License

This project is licensed under the **[MIT License](LICENSE)**.  
Click to view the full license text.

---

### 👤 Author  
**Sanjeev Kumar Gogoi**  
Course: Data Science Project 2

📌 **GitHub Repository:**  
👉 [https://github.com/21f3000340-rgb/llm_quiz_solver](https://github.com/21f3000340-rgb/llm_quiz_solver)

For questions or issues, please open an issue on the GitHub repository.

