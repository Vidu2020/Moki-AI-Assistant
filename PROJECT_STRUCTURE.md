

## 🧩 Module Breakdown

### ✅ `app/main.py`
- Entry point of the application
- Controls flow of the voicebot
- Integrates voice, memory, and LLM

---

### ✅ `app/voice.py`
Handles audio interactions:

- `listen()` → Converts speech to text
- `speak()` → Converts text to speech

---

### ✅ `app/memory.py`
Handles Qdrant database:

- `setup_qdrant()` → Initializes DB
- `store_memory()` → Saves conversation
- `retrieve_documents()` → Fetches context

---

### ✅ `app/rag.py`
Handles intelligence layer:

- Builds prompts using retrieved context
- Adds fallback for general knowledge

---

## 🧠 Data Flow


User Speech
↓
voice.py → Text
↓
memory.py → Embedding + Retrieval
↓
rag.py → Prompt
↓
LLM (Ollama)
↓
Response
↓
voice.py → Speech Output

---

## 🚀 Scaling Plan

Future modular expansion:


app/
├── api/                # FastAPI backend
├── avatar/             # Unity / 3D integration
├── utils/              # Shared helpers
├── config/             # Environment configs

---

## ✅ Best Practices Implemented

- Modular code structure ✅
- Separation of concerns ✅
- Reusable components ✅
- Production-ready layout ✅
- Scalable architecture ✅