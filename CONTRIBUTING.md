# 🤝 Contributing to Quiz Solver API (Data Science Project 2)

Welcome!
This project is part of **Data Science Project 2**, created to explore, build, and refine automation tools for intelligent quiz solving and data handling using FastAPI and modern AI integrations.

Contributions are always appreciated — especially those that improve stability, performance, or clarity.

---

## 🧩 Getting Started

1. **Fork the repository**
   Click the “Fork” button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/yourusername/llm_quiz_solver.git
   cd llm_quiz_solver
   ```

3. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Create your `.env` file**

   ```
   USER_EMAIL=your_email@example.com
   USER_SECRET=your_secret_key
   GITHUB_REPO=https://github.com/yourusername/llm_quiz_solver
   ```

   > The `.env` file stores credentials safely — never include it in commits.

5. **Run locally**

   ```bash
   uvicorn app:app --reload
   ```

   Then open [http://localhost:8000](http://localhost:8000).

---

## 🛠 Development Workflow

* Create a new branch before making changes:

  ```bash
  git checkout -b feature/your-feature-name
  ```

* After your changes, commit clearly:

  ```bash
  git commit -m "Add: short description of the update"
  ```

* Push to your fork:

  ```bash
  git push origin feature/your-feature-name
  ```

* Submit a **Pull Request (PR)** with a short summary of what you changed and why.

---

## ✅ Code Standards

* Follow **PEP 8** styling guidelines.
* Keep code **modular** and well-commented.
* Avoid hard-coding secrets or file paths.
* Include docstrings for clarity.
* Test before committing.

---

## 🧪 Testing Your Changes

Run your containerized version to keep environments consistent:

```bash
docker build -t quiz-solver:latest .
docker run --env-file .env -p 8000:8000 quiz-solver:latest
```

Check the following endpoints:

* `/` → API Welcome Message
* `/health` → Health status
* `/docs` → Interactive documentation

---

## 💬 Communication

This repository supports collaboration primarily for **educational and research learning**.
If you encounter issues or wish to suggest improvements:

* Open a GitHub **Issue** describing the problem or idea.
* Be respectful, concise, and solution-oriented.

---

## 🧠 Quick Summary

✅ Keep commits small and clear
✅ Test thoroughly before pushing
✅ Don’t commit `.env` or secrets
✅ Write clean, documented code

---

Thanks for contributing to **Quiz Solver API (Tools in Data Science Project 2)** —
your input helps refine and improve real-world data science tools.
