import chromadb
from gigachat import GigaChat
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / ".." / ".." / "data" / "vector_db"
COLLECTION_NAME = "lenta_clean_index"

GIGACHAT_CREDENTIALS = ""

chroma_client = chromadb.PersistentClient(path= DB_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")

def search_in_db(query: str, k: int=3):
    query_vector = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=k
    )

    retrieved_data = []
    for i in range(len(results['documents'][0])):
        retrieved_data.append({
            "text": results['documents'][0][i],
            "distance": results['distances'][0][i],
            "meta": results['metadatas'][0][i]
        })
    return retrieved_data

def 