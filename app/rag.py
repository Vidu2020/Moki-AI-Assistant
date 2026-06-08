from app.memory import retrieve_documents


def build_prompt(client, model, user_query):
    docs = retrieve_documents(client, model, user_query)

    context = "\n".join([doc["content"] for doc in docs])

    # ✅ fallback → general knowledge
    if len(context.strip()) < 10:
        return f"""
You are a smart AI assistant.

Answer using your general knowledge.

Question: {user_query}

Answer:
"""

    return f"""
You are a smart AI assistant.

Use the context if relevant; otherwise use your general knowledge.

Context:
{context}

Question: {user_query}

Answer:
"""