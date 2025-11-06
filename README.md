# 🧠 Quiz Solver API (Gemini Edition)

A production-ready **FastAPI** application developed as part of **Data Science Project 2**.
This tool automates quiz solving and data interpretation using advanced AI models, built with a focus on security, scalability, and deployment readiness.

---

## 🚀 Overview

The **Quiz Solver API** provides a secure backend for solving and analyzing quiz-related tasks.
It includes safety checks for data leaks, efficient task handling, and ready-to-use Docker deployment for Railway or any containerized environment.

---

## ⚙️ Tech Stack

* **Python 3.12**
* **FastAPI** – backend framework
* **Uvicorn** – ASGI web server
* **Docker** – containerization
* **Playwright (Chromium)** – for automation tasks
* **Railway** – hosting & deployment

---

## 📁 Project Structure

```
llm_quiz_solver/
│
├── app.py               # Main FastAPI application
├── solver.py            # Core quiz-solving logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build instructions
├── .dockerignore        # Ignored files for Docker build
├── .env                 # Local environment variables (not pushed to GitHub)
├── Procfile             # Optional process definition (for non-Docker deploys)
├── runtime.txt          # Optional Python runtime version
└── README.md            # Project documentation
```

---

## 🔐 Environment Variables

Before running or deploying, create a `.env` file in the project root with the following keys:

```
USER_EMAIL=your_email@example.com
USER_SECRET=your_secret_key
GITHUB_REPO=https://github.com/yourusername/llm_quiz_solver
API_KEY=your_api_key_here   # optional if external API is used
```

> ⚠️ **Do not** commit or upload your `.env` file to GitHub.
> It contains sensitive credentials and should remain private.

---

## 🧩 Local Development

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/llm_quiz_solver.git
   cd llm_quiz_solver
   ```

2. **Build Docker Image**

   ```bash
   docker build -t quiz-solver:latest .
   ```

3. **Run the Container**

   ```bash
   docker run --env-file .env -p 8000:8000 quiz-solver:latest
   ```

4. **Access the API**

   * Home: [http://localhost:8000](http://localhost:8000)
   * Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   * Health: [http://localhost:8000/health](http://localhost:8000/health)

---

## ☁️ Deployment on Railway

1. Push your project to **GitHub**.
2. On Railway:

   * Create a **New Project → Deploy from GitHub Repo**.
   * Add environment variables (`USER_EMAIL`, `USER_SECRET`, `GITHUB_REPO`, and any others).
3. Railway automatically builds your Dockerfile and runs your container.
4. Once deployed, access your app at your generated Railway URL.

---

## 🧠 Security Notes

* Sensitive values are never stored inside the Docker image.
* The `.env` file is excluded using `.dockerignore` and `.gitignore`.
* Environment variables are securely injected at runtime (both locally and on Railway).

---

## 👤 Author

**Sanjeev Kumar Gogoi**
Working Professional | Data Science Project 2
💼 Exploring tools and technologies in applied data science
🌐 Developed as part of hands-on learning and automation research

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
You’re free to modify and distribute with attribution.
