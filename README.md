<p align="center">
  <img src="./logo.png" alt="Quiz Solver Logo" width="200" style="border-radius: 50%;"/>
</p>

<h1 align="center">🌑🧠 Quiz Solver API — Gemini Edition</h1>

<p align="center" style="font-size: 1.1rem;">
A secure, containerized AI backend that solves quizzes, scrapes dynamic web pages,  
processes files, and generates intelligent insights — powered by Google Gemini.
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

# 📋 Table of Contents  

- [Overview](#-overview-dark-theme-styled)  
- [Features](#-key-features)  
- [Tech Stack](#-tech-stack-dark-mode)  
- [Project Structure](#-project-structure)  
- [Installation](#-installation)  
- [Configuration](#️-configuration)  
- [Usage](#-usage)  
- [API Endpoints](#-api-endpoints)  
- [Tools  Capabilities](#-tools--capabilities)  
- [How It Works](#-how-it-works)  
- [License](#-license)  
- [Author](#-author)

---

## 🌌 Overview (Dark Theme Styled)

This backend is built for **Data Science Project 2**, designed to autonomously:

✨ Scrape quizzes (including JavaScript-rendered pages via Playwright)  
✨ Parse CSV, XLSX, PDFs, and APIs  
✨ Clean and transform datasets  
✨ Perform reasoning and lightweight ML-style analysis  
✨ Generate slides & base64 charts  
✨ Follow multi-step quiz chains until the final task  

The system also supports:

- **3-minute retry logic** (instructor requirement)  
- **Secret leak prevention**  
- **Railway-ready Docker deployment**

---

## 🧠 Key Features

### ✔ Autonomous multi-page quiz solving  
Follows every `next_url` until the quiz is completed.

### ✔ 3-minute retry window  
Your latest submission within 3 minutes overrides all previous answers.

### ✔ Safe output sanitization  
Blocks forbidden code-words (`elephant`, `tiger`, `umbrella`, etc).

### ✔ Multi-modal parsing  
Supports:  
HTML • JS-rendered HTML • JSON APIs • CSV • Excel • PDF (PyPDF2)

### ✔ Clean visualization output  
Generates base64 charts + short slide-style narratives.

### ✔ Containerized & cloud-ready  
Deployable to Railway with a single Dockerfile.

---

## ⚙️ Tech Stack (Dark Mode)

| Component      | Technology            |
|----------------|------------------------|
| Backend        | FastAPI                |
| AI Model       | Gemini 2.5 Flash       |
| Scraping       | Playwright Chromium    |
| Deployment     | Docker + Railway       |
| Language       | Python 3.12            |
| Server         | Uvicorn                |

---

## 📁 Project Structure

```
llm_quiz_solver/
│
├── app.py                 # FastAPI backend with session logic, leak checks
├── solver.py              # Core Gemini-based quiz solving engine
├── tools/                 # (Optional) helper utilities
│
├── Dockerfile             # Production-ready container
├── requirements.txt       # Python dependencies
├── Procfile               # Railway process definition
├── .dockerignore
├── .gitignore
├── .env.example           # Example environment variables
└── README.md              # This documentation
```

---

# 📦 Installation

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/21f3000340-rgb/llm_quiz_solver.git
cd llm_quiz_solver
```

## 2️⃣ Install Dependencies (Option A — pip)
```bash
pip install -r requirements.txt
playwright install chromium
```

## 3️⃣ Install with Docker (Option B — recommended)
```bash
docker build -t quiz-solver .
```

---

# 🛠 Configuration

Create a `.env` file:

```env
USER_EMAIL=your_email@example.com
USER_SECRET=your_secret_key
GITHUB_REPO=https://github.com/21f3000340-rgb/llm_quiz_solver
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ **Never commit `.env` to GitHub**

---

# 🚀 Usage

## Run (pip)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Run (Docker)
```bash
docker run --env-file .env -p 8000:8000 quiz-solver
```

## Test API
```bash
curl -X POST http://localhost:8000/solve_quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your.email@example.com",
    "secret": "your_secret_string",
    "url": "https://example.com/quiz"
  }'
```

---

# 🌐 API Endpoints

### **POST /solve_quiz**
Starts solving a quiz.

### **GET /health**
Returns:
```json
{"status":"ok","message":"Quiz Solver API running safely ✅"}
```

### **GET /favicon.ico**
Loads your custom icon.

---

# 🛠 Tools & Capabilities

Your solver supports:

### **1. JavaScript-rendered scraping**  
Playwright Chromium → full DOM extraction.

### **2. API loading**  
JSON, nested structures, auto-normalization.

### **3. File parsing**  
CSV, Excel, PDF (PyPDF2).

### **4. LLM data reasoning**  
Summary • QA • Insight • Table analysis • ML-style reasoning.

### **5. Chart generation**  
Returned as `"data:image/png;base64,..."`.

### **6. Multi-page chaining**  
Follows `next_url` until quiz ends.

---

# 🧠 How It Works

### **1. FastAPI receives request**
Validates secret & email  
Starts 3-minute retry window.

### **2. Solver loads data**
HTML / JS / PDFs / APIs → cleaned → passed to Gemini.

### **3. Gemini analyzes**
Generates:
- summary  
- analysis  
- QA  
- slides  
- chart  
- next_url  

### **4. Session memory**
Maintains latest answer for 3 minutes.

### **5. Multi-page solving**
If `next_url` → continue  
If none → quiz finished.

---

# 📄 License

This project is licensed under the **[MIT License](LICENSE)**.

---

# 👤 Author

**Sanjeev Kumar Gogoi**  
Working Professional • Data Science Project 2

📌 **GitHub Repository:**  
👉 https://github.com/21f3000340-rgb/llm_quiz_solver  

For issues or suggestions, please open an Issue in the repository.

