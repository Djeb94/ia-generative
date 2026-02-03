import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyDWXKyDN6nHg1Kwkxmq6AbpU8u_VgOjkQ4"
genai.configure(api_key=GEMINI_API_KEY)
GEN_MODEL = genai.GenerativeModel("gemini-2.5-flash")




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "csv")
COMP_PATH = os.path.join(CSV_DIR, "competences.csv")
METIERS_PATH = os.path.join(CSV_DIR, "metiers.csv")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    free_text: str
    other_answers: Optional[Dict[str, Any]] = None

def _load_csvs():
    comp_df = pd.read_csv(COMP_PATH, dtype=str).fillna("") if os.path.exists(COMP_PATH) else pd.DataFrame()
    met_df = pd.read_csv(METIERS_PATH, dtype=str).fillna("") if os.path.exists(METIERS_PATH) else pd.DataFrame()
    return comp_df, met_df

COMP_DF, METIERS_DF = _load_csvs()

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

COMPETENCE_NAMES: List[str] = COMP_DF["CompetencyName"].tolist() if not COMP_DF.empty else []
COMP_EMBS = MODEL.encode(COMPETENCE_NAMES, convert_to_tensor=True) if COMPETENCE_NAMES else None

JOB_EMBS = []
JOB_ROWS = []

if COMP_EMBS is not None and not METIERS_DF.empty:
    for _, row in METIERS_DF.iterrows():
        ids = [i.strip() for i in str(row["RequiredCompetencies"]).split(";") if i.strip()]
        emb_list = []

        for cid in ids:
            match = COMP_DF[COMP_DF["CompetencyID"] == cid]
            if not match.empty:
                emb_list.append(COMP_EMBS[int(match.index[0])])

        if emb_list:
            JOB_EMBS.append(torch.mean(torch.stack(emb_list), dim=0))
            JOB_ROWS.append({"job_id": row["JobID"], "title": row["JobTitle"], "required": ids})

def _combine_text(free_text: str, other_answers=None) -> str:
    parts = [free_text or ""]
    if other_answers:
        for v in other_answers.values():
            parts.append(" ".join(v) if isinstance(v, list) else str(v))
    return " \n ".join(parts)

def generate_career_advice(matched, block_scores, jobs, profile):
    try:
        skills = ", ".join([m["name"] for m in matched])
        blocks = ", ".join([f"{k}: {round(v,2)}" for k,v in block_scores.items()])
        job_titles = ", ".join([j["job"] for j in jobs])

        prompt = f"""
You are an AI career advisor.

User skills detected: {skills}
Skill block scores: {blocks}
Recommended jobs: {job_titles}
Profile level: {profile}

Give short career advice:
- Why these jobs match
- Strong areas
- What to improve
- Encouraging tone
"""

        response = GEN_MODEL.generate_content(prompt)

        print("GEMINI RAW RESPONSE:", response)

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        print("🔥 GEMINI ERROR:", str(e))
        return f"AI error: {str(e)}"

@app.post("/analyse")
def analyse_endpoint(data: UserInput):
    combined = _combine_text(data.free_text, data.other_answers)

    if COMP_EMBS is None:
        return {"matched_competences": [], "block_scores": {}, "recommended_jobs": []}

    user_emb = MODEL.encode(combined, convert_to_tensor=True)
    sims = util.cos_sim(user_emb, COMP_EMBS)[0].cpu().numpy()

    top_idx = np.argsort(sims)[-5:][::-1]
    matched = [{
        "id": COMP_DF.iloc[i]["CompetencyID"],
        "name": COMP_DF.iloc[i]["CompetencyName"],
        "block": COMP_DF.iloc[i]["BlockName"],
        "score": float(sims[i])
    } for i in top_idx]

    block_map = {}
    for i, sc in enumerate(sims):
        blk = COMP_DF.iloc[i]["BlockName"]
        block_map.setdefault(blk, []).append(float(sc))

    block_scores = {b: float(np.mean(sorted(v, reverse=True)[:3])) for b, v in block_map.items()}

    coverage_score = np.mean(list(block_scores.values())) if block_scores else 0

    profile_level = (
        "Data Scientist" if coverage_score >= 0.7 else
        "ML Engineer" if coverage_score >= 0.5 else
        "Entry-level Analyst"
    )

    recommended = []
    for jemb, jinfo in zip(JOB_EMBS, JOB_ROWS):
        score = util.cos_sim(user_emb, jemb)[0][0].item()
        recommended.append({"job_id": jinfo["job_id"], "job": jinfo["title"], "match": round(score*100,2)})

    recommended = sorted(recommended, key=lambda x: x["match"], reverse=True)[:3]

    career_advice = generate_career_advice(matched, block_scores, recommended, profile_level)

    return {
        "matched_competences": matched,
        "block_scores": block_scores,
        "coverage_score": round(float(coverage_score), 3),
        "profile_recommendation": profile_level,
        "recommended_jobs": recommended,
        "career_advice": career_advice
    }

@app.get("/health")
def health():
    return {"status": "ok"}
