from qdrant_client.models import VectorParams, Distance
import uuid
from datetime import datetime
import PyPDF2
import io


# ✅ Extract text from PDF
def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return ""


# ✅ Store PDF content
def store_pdf_content(client, model, pdf_text, filename="uploaded.pdf"):
    """Store PDF content in Qdrant as chunks"""
    # Split text into chunks (simple approach by paragraphs)
    chunks = [chunk.strip() for chunk in pdf_text.split("\n\n") if len(chunk.strip()) > 50]

    # If no chunks found, split by sentences
    if not chunks:
        import re
        chunks = re.split(r'(?<=[.!?]) +', pdf_text)
        chunks = [c for c in chunks if len(c) > 50]

    points = []
    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": chunk,
                "source": filename,
                "chunk_index": i,
                "timestamp": str(datetime.now()),
                "type": "pdf_content"
            }
        })

    if points:
        client.upsert(
            collection_name="documents",
            points=points
        )
        return len(points)
    return 0


# ✅ Setup Qdrant
def setup_qdrant(client):
    if not client.collection_exists("documents"):
        client.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print("✅ Collection created")
    else:
        print("✅ Collection already exists")


# ✅ Store memory
def store_memory(client, model, text, speaker="user"):
    vector = model.encode(text).tolist()

    client.upsert(
        collection_name="documents",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": text,
                "speaker": speaker,
                "timestamp": str(datetime.now()),
                "type": "conversation"
            }
        }]
    )


# ✅ Retrieve documents
def retrieve_documents(client, model, query, top_k=3):
    try:
        query_vector = model.encode(query).tolist()

        results = client.query_points(
            collection_name="documents",
            query=query_vector,
            limit=top_k
        )

        docs = []
        for r in results.points:
            docs.append({
                "content": r.payload.get("text", "")
            })

        return docs

    except Exception as e:
        print("❌ Qdrant error:", e)
        return []