# 🎤 AI Voicebot with Memory + RAG + 3D Avatar Ready

## 🚀 Overview

This project is a **local AI Voice Assistant** with:

- 🎤 Speech-to-Text (Voice Input)
- 🔊 Text-to-Speech (Voice Output)
- 🧠 Persistent Memory using Qdrant (Vector DB)
- 📚 Hybrid Intelligence (RAG + General Knowledge)
- 🤖 Local LLM (Ollama - LLaMA3)
- 🎭 Ready for 3D Avatar Integration

---

## 🧱 Tech Stack

- Python 3.11+
- Qdrant (Vector Database)
- SentenceTransformers (Embeddings)
- Ollama (LLM - `llama3:8b`)
- SpeechRecognition + PyAudio (Voice Input)
- pyttsx3 (Voice Output)

---

## ⚙️ Installation

### ✅ 1. Install Python dependencies

```bash
pip install -r requirements.txt

#If PyAudio fails on Windows:
pip install pipwin
pipwin install pyaudio

#Start Qdrant (Vector DB)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

#✅ 3. Start Ollama model

ollama run llama3:8b

✅ 4. Run the Voicebot
python -m app.main

 Features

✅ Voice-based interaction (mic input + speech output)
✅ Long-term memory using vector database
✅ Semantic search (embeddings)
✅ RAG (Retrieval-Augmented Generation)
✅ Fallback to general LLM knowledge
✅ Modular architecture (production-ready)
✅ Ready for 3D avatar integration

🧪 Example Usage

You: My name is Parth
You: What is my name?
Bot: Your name is Parth

You: What is the capital of India?
Bot: New Delhi


🧩 Architecture

Voice Input 🎤
   ↓
Speech → Text
   ↓
Embedding (SentenceTransformers)
   ↓
Qdrant Search (Memory)
   ↓
RAG Prompt Builder
   ↓
LLM (Ollama - LLaMA3)
   ↓
Response
   ↓
Text-to-Speech 