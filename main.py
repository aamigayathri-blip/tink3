from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the AI Brain
from legal_brain import legal_engine

app = FastAPI(
    title="Justicia",
    description="Sovereign AI Legal Assistant for India (BNS 2023)",
    version="1.0"
)

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class UserQuery(BaseModel):
    query_text: str
    language: str = "en"

class LegalResponse(BaseModel):
    status: str
    query_received: str
    matched_section: str
    legal_title: str
    simplified_advice: str
    confidence_score: float

# --- Endpoints ---

@app.get("/")
def health_check():
    return {
        "status": "online",
        "system": "Justicia Cloud"
    }

@app.post("/consult", response_model=LegalResponse)
def consult_lawyer(user_query: UserQuery):

    if not user_query.query_text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"Processing Query: {user_query.query_text}")

    results = legal_engine.search(user_query.query_text, k=1)

    if not results:
        return LegalResponse(
            status="no_match",
            query_received=user_query.query_text,
            matched_section="N/A",
            legal_title="Consult a Human Lawyer",
            simplified_advice="Our AI could not find a specific section for this issue. Please visit a legal aid clinic.",
            confidence_score=0.0
        )

    best_match = results[0]

    return LegalResponse(
    status="success",
    query_received=user_query.query_text,
    matched_section=best_match.get("section", "N/A"),
    legal_title=best_match.get("title", "Unknown"),
    simplified_advice=best_match.get("simplified", "No explanation available."),
    confidence_score=best_match.get("score", 0.0)
)

