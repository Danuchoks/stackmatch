from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import faiss
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import os

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="StackMatch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "model_artifacts"
SBERT_MODEL   = "all-MiniLM-L6-v2"
VALID_TAGS    = sorted([
    "assembly", "c#", "c++", "dart", "go", "haskell", "java",
    "javascript", "kotlin", "lua", "objective-c", "perl", "php",
    "python", "r", "ruby", "rust", "scala", "swift", "typescript"
])
FUSION_WEIGHTS = {"alpha": 0.4, "beta": 0.5, "gamma": 0.1}

# ── Load Artifacts on Startup ─────────────────────────────────────────────────
print("Loading model artifacts...")

with open(f"{ARTIFACTS_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(f"{ARTIFACTS_DIR}/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

index = faiss.read_index(f"{ARTIFACTS_DIR}/faiss_index.faiss")

df = pd.read_parquet(f"{ARTIFACTS_DIR}/dataframe.parquet")

embedding_model = SentenceTransformer(SBERT_MODEL)
translator      = GoogleTranslator(source="auto", target="en")

print(f"✅ Loaded {len(df)} rows | {index.ntotal} vectors in FAISS")

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_score(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)

def translate_query(text: str) -> str:
    try:
        return translator.translate(text)
    except:
        return text

# ── Search Logic ──────────────────────────────────────────────────────────────
def hybrid_search(query: str, tag: str, top_k: int = 5):
    translated = translate_query(query)

    # Filter by tag
    mask           = df["Tags"] == tag.lower()
    subset_df      = df[mask]
    subset_indices = df[mask].index.tolist()

    if len(subset_df) == 0:
        raise HTTPException(status_code=404, detail=f"Tag '{tag}' not found")

    # TF-IDF
    q_vec        = tfidf.transform([translated])
    tfidf_all    = cosine_similarity(q_vec, tfidf_matrix).flatten()
    tfidf_scores = tfidf_all[subset_indices]

    # SBERT via FAISS
    q_emb = embedding_model.encode([translated], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    k_search       = min(len(df), top_k * 20)
    sbert_all      = np.zeros(len(df))
    scores_f, idx_f = index.search(q_emb, k_search)
    for sc, ix in zip(scores_f[0], idx_f[0]):
        if ix >= 0:
            sbert_all[ix] = float(sc)
    sbert_scores = sbert_all[subset_indices]

    # AnswerScore
    ans_scores = np.log1p(np.clip(subset_df["AnswerScore"].values.astype(float), 0, None))

    # Fusion
    fusion = (
        FUSION_WEIGHTS["alpha"] * normalize_score(tfidf_scores) +
        FUSION_WEIGHTS["beta"]  * normalize_score(sbert_scores) +
        FUSION_WEIGHTS["gamma"] * normalize_score(ans_scores)
    )

    top_local = fusion.argsort()[-top_k:][::-1]

    results = []
    for li in top_local:
        gi  = subset_indices[li]
        row = df.iloc[gi]
        results.append({
            "id":               int(gi),
            "title":            str(row["Title"]),
            "tags":             str(row["Tags"]),
            "answer":           str(row["AnswerBody"]),
            "answer_score":     int(row["AnswerScore"]),
            "fusion_score":     round(float(fusion[li]), 4),
            "tfidf_score":      round(float(normalize_score(tfidf_scores)[li]), 4),
            "sbert_score":      round(float(normalize_score(sbert_scores)[li]), 4),
            "translated_query": translated,
        })

    return results

# ── Schemas ───────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query:  str
    tag:    str
    top_k:  Optional[int] = 5

class TranslateRequest(BaseModel):
    text:        str
    target_lang: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/api/tags")
def get_tags():
    """Return list of available tags with row counts."""
    counts = df["Tags"].value_counts().to_dict()
    return {
        "tags": [
            {"name": t, "count": counts.get(t, 0)}
            for t in VALID_TAGS
        ]
    }

@app.get("/api/stats")
def get_stats():
    """Overall dataset statistics."""
    return {
        "total_questions": len(df),
        "total_tags":      len(VALID_TAGS),
        "top_tags":        df["Tags"].value_counts().head(5).to_dict(),
        "avg_answer_score": round(float(df["AnswerScore"].mean()), 2),
    }

@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.tag.lower() not in VALID_TAGS:
        raise HTTPException(status_code=400, detail=f"Invalid tag. Choose from: {VALID_TAGS}")
    top_k = max(1, min(req.top_k, 10))
    results = hybrid_search(req.query, req.tag, top_k)
    return {"query": req.query, "tag": req.tag, "results": results}

@app.post("/api/translate")
def translate(req: TranslateRequest):
    try:
        t = GoogleTranslator(source="auto", target=req.target_lang)
        translated = t.translate(req.text)
        return {"original": req.text, "translated": translated, "target_lang": req.target_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/random")
def random_questions(tag: Optional[str] = None, n: int = 3):
    """Return random questions, optionally filtered by tag."""
    subset = df[df["Tags"] == tag] if tag and tag in VALID_TAGS else df
    sample = subset.sample(min(n, len(subset)))
    return {
        "questions": [
            {"title": row["Title"], "tags": row["Tags"], "answer_score": int(row["AnswerScore"])}
            for _, row in sample.iterrows()
        ]
    }

# ── Static files (frontend) ───────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
