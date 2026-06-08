from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import sys
import os

# Add the parent directory to sys.path to allow 'from app' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.voice import listen, speak
from app.memory import setup_qdrant, store_memory
from app.rag import build_prompt


class VoiceBot:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def answer(self, user_query):
        prompt = build_prompt(self.client, self.model, user_query)

        # ✅ FIXED indentation
        response = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        # ✅ store memory
        store_memory(self.client, self.model, user_query, "user")
        store_memory(self.client, self.model, answer, "assistant")

        return answer

    def start(self):
        print("✅ Voice AI Assistant Ready! Say 'quit' to exit\n")

        while True:
            user_input = listen()

            if not user_input:
                continue

            if "quit" in user_input.lower():
                speak("Goodbye!")
                break

            reply = self.answer(user_input)
            speak(reply)


# ✅ INIT QDRANT
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=6333
)

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Setup DB
setup_qdrant(qdrant_client)

# ✅ RUN BOT
bot = VoiceBot(qdrant_client, embedding_model)
bot.start()