# 🧠 RMIT Academic Policies Chatbot
**AI-Assisted Supporter Chatbot for New RMIT Students**

This chatbot helps new RMIT students understand **academic policies** such as academic integrity, assessment extensions, appeals, and course progress.  
It provides **accurate, citation-backed answers** based on official RMIT policies using **Retrieval-Augmented Generation (RAG)** with AWS Bedrock (Claude Sonnet / Haiku).

---

## ✨ Features
- 🎨 Streamlit-based modern UI (mobile-friendly)
- ⚙️ RAG with FAISS + sentence-transformers
- 🧠 Context-aware conversation memory
- 💬 Automatic topic/entity clarification
- 📚 Auto-indexing of uploaded policy JSON files
- 🧩 Responsible-AI guardrails
- 📖 Clause-level citations for every factual answer

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/cloudwhj/DataCommAssm3.git
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the chatbot
```bash
streamlit run app.py
```

The app will start on **http://localhost:8501**

---

## 📂 Project Structure
```
academic-policies-chatbot/
├── app.py                     # Main Streamlit application
├── requirements.txt
├── .gitignore
├── data/
│   ├── policies/              # Policy JSON files
│   └── index/                 # Auto-generated FAISS index
└── README.md
```

---

## 🧩 How It Works
1. **Policy Data** → PDFs are converted into JSON (`metadata`, `structure`, `clauses`, `subclauses`).
2. **Indexing** → Text chunks embedded with `sentence-transformers` and stored in FAISS.
3. **Retrieval** → Queries are embedded and matched against the index.
4. **Generation** → Claude (Sonnet/Haiku) answers using the retrieved clauses.
5. **Clarification** → The chatbot rewrites vague queries (“this”, “it”) and asks clarifying questions when needed.

---

## 🔒 Responsible-AI Rules
- Responds only using official RMIT policies.
- Adds citations to every factual statement.
- Politely refuses unrelated or personal queries.
- Displays a clear disclaimer for transparency.

---

## 🧠 Example Queries
Try:
- “What counts as plagiarism?”
- “How do I apply for an assessment extension?”
- “How do I appeal a final course result?”
- “What happens if I fail the same course twice?”

---

## ⚙️ Default Settings
| Parameter | Default | Description |
|------------|----------|-------------|
| Top-k | 6 | Number of retrieved clauses |
| Min similarity | 0.35 | Confidence threshold |
| Temperature | 0.2 | Creativity control |
| Top-p | 0.9 | Response diversity |

---

## 🧰 Technologies Used
- **Streamlit** — Web UI framework  
- **AWS Bedrock (Claude 3)** — LLM inference  
- **boto3** — AWS SDK for Python  
- **sentence-transformers** — Embeddings  
- **FAISS** — Vector similarity search  
- **PyPDF2** — PDF parsing  

---

## 🏗 Architecture
```
 ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
 │  Streamlit UI │──→──│  Retrieval (FAISS) │──→──│  Bedrock Claude API │
 └──────────────┘      └──────────────┘      └──────────────────┘
        ▲                                         │
        │                                         ▼
        └───── Temporary AWS Credentials via boto3
```

---

## 🧪 Testing
Once the app is running, go to:
```
http://localhost:8501
```
and test various queries.  
Each response includes relevant policy citations for verification.

---

## 👥 Authors
- **Wong Hon Jun** – s4060180  
- **Student 2** – s
- **Student 3** – s
RMIT University, Bachelor of Computer Science

---

## 🪪 License
This project is for **educational purposes** under RMIT’s Academic Use Policy.  
Do not redistribute or deploy publicly with institutional credentials.
