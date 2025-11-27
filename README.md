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
├── app.py # FastAPI service + retry window logic
├── solver.py # Gemini-based quiz solver
├── agent.py # (optional) LangGraph autonomous agent
│
├── tools/ # Modular scraping/execution tools
│ ├── get_rendered_html.py
│ ├── download_file.py
│ ├── run_code.py
│ ├── post_request.py
│ └── add_dependencies.py
│
├── Dockerfile
├── requirements.txt
├── Procfile
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md

---

## 🔐 Environment Variables

Create a `.env` file:

```env
USER_EMAIL=your_email@example.com
USER_SECRET=your_secret_key
GITHUB_REPO=https://github.com/yourusername/llm_quiz_solver
GEMINI_API_KEY=your_gemini_api_key_here
⚠️ Never commit .env files to GitHub.

🧩 Local Development
1. Clone
git clone https://github.com/yourusername/llm_quiz_solver.git
cd llm_quiz_solver

2. Build Docker
docker build -t quiz-solver:latest .

3. Run
docker run --env-file .env -p 8000:8000 quiz-solver:latest

4. Access

Home → http://localhost:8000

Docs → http://localhost:8000/docs

Health → http://localhost:8000/health

☁️ Deployment on Railway (Dark Mode)

Push repo to GitHub

Create new Railway project → select your repo

Add environment variables

Railway auto-builds your Dockerfile

Open deployed URL 🎉

📡 API Endpoints
POST /solve_quiz

Input fields:

email

secret

url

Returns:

summary

analysis

qa pairs

slides

chart (base64)

answer

next_url

GET /health

Quick readiness probe.

GET /favicon.ico

Supports custom favicon.

🛡 Security & Reliability

🛡 Strict secret enforcement
🛡 Leak detection for code words
🛡 Sanitizes LLM outputs
🛡 3-minute retry guarantee
🛡 No secrets stored inside Docker

👤 Author

Sanjeev Kumar Gogoi
Working Professional • Data Science Project 2
Focused on automation, agents, and applied data workflows.

📜 License

Licensed under MIT License.
Feel free to use, extend, or distribute with attribution.
