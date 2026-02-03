import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


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
    if os.path.exists(COMP_PATH):
        comp_df = pd.read_csv(COMP_PATH, dtype=str)
    else:
        comp_df = pd.DataFrame(columns=["CompetencyID", "CompetencyName", "BlockID", "BlockName"])

    if os.path.exists(METIERS_PATH):
        met_df = pd.read_csv(METIERS_PATH, dtype=str)
    else:
        met_df = pd.DataFrame(columns=["JobID", "JobTitle", "RequiredCompetencies"])

    return comp_df.fillna(""), met_df.fillna("")


COMP_DF, METIERS_DF = _load_csvs()

MODEL_NAME = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
MODEL = SentenceTransformer(MODEL_NAME)

COMPETENCE_NAMES: List[str] = COMP_DF["CompetencyName"].astype(str).tolist() if not COMP_DF.empty else []
COMP_EMBS = MODEL.encode(COMPETENCE_NAMES, convert_to_tensor=True) if COMPETENCE_NAMES else None


JOB_EMBS = []
JOB_ROWS = []

if COMP_EMBS is not None and not METIERS_DF.empty:
    for _, row in METIERS_DF.iterrows():
        reqs = str(row.get("RequiredCompetencies", "")).strip()
        ids = [i.strip() for i in reqs.replace(",", ";").split(";") if i.strip()]
        emb_list = []

        for cid in ids:
            matches = COMP_DF[COMP_DF["CompetencyID"].astype(str) == str(cid)]
            if not matches.empty:
                idx = matches.index[0]
                emb_list.append(COMP_EMBS[int(idx)])

        if emb_list:
            mean_emb = torch.mean(torch.stack(emb_list), dim=0)
            if torch.isnan(mean_emb).any() or torch.isinf(mean_emb).any():
                continue

            JOB_EMBS.append(mean_emb)
            JOB_ROWS.append({
                "job_id": row.get("JobID"),
                "title": row.get("JobTitle"),
                "required": ids
            })


def _combine_text(free_text: str, other_answers: Optional[Dict[str, Any]] = None) -> str:
    parts = [free_text or ""]
    if other_answers:
        for v in other_answers.values():
            if v is None:
                continue
            if isinstance(v, list):
                parts.append(" ".join([str(x) for x in v]))
            else:
                parts.append(str(v))
    return " \n ".join(p for p in parts if p)


@app.post("/analyse")
def analyse_endpoint(data: UserInput):
    try:
        combined = _combine_text(data.free_text, data.other_answers)

        if COMP_EMBS is None:
            return {"matched_competences": [], "block_scores": {}, "recommended_jobs": []}

        user_emb = MODEL.encode(combined, convert_to_tensor=True)
        sims = util.cos_sim(user_emb, COMP_EMBS)[0].cpu().numpy()

        top_k = min(5, len(sims))
        top_idx = list(np.argsort(sims)[-top_k:][::-1])
        matched = []

        for idx in top_idx:
            row = COMP_DF.iloc[int(idx)]
            matched.append({
                "id": str(row.get("CompetencyID", str(idx))),
                "name": str(row.get("CompetencyName", "")),
                "block": str(row.get("BlockName", "")),
                "score": float(sims[int(idx)])
            })

        block_map: Dict[str, List[float]] = {}
        for i, sc in enumerate(sims):
            blk = str(COMP_DF.iloc[int(i)].get("BlockName", ""))
            block_map.setdefault(blk, []).append(float(sc))

        block_scores = {}

        for b, v in block_map.items():
            v_sorted = sorted(v, reverse=True)

            top_n = v_sorted[:3]
            block_scores[b] = float(np.mean(top_n))


        BLOCK_WEIGHTS = {
            "Data Analysis": 1,
            "Machine Learning": 1,
            "NLP": 1
        }

        weighted_sum = 0
        total_weight = 0

        for block, score in block_scores.items():
            w = BLOCK_WEIGHTS.get(block, 1)
            weighted_sum += w * score
            total_weight += w

        coverage_score = weighted_sum / total_weight if total_weight else 0


        if coverage_score >= 0.7:
            profile_level = "Data Scientist"
        elif coverage_score >= 0.5:
            profile_level = "ML Engineer"
        else:
            profile_level = "Entry-level Analyst"

        recommended = []
        if JOB_EMBS:
            job_sims = []

            for jemb, jinfo in zip(JOB_EMBS, JOB_ROWS):
                score = util.cos_sim(user_emb, jemb)[0][0].item()
                if np.isnan(score) or np.isinf(score):
                    score = 0.0
                job_sims.append((score, jinfo))

            job_sims = sorted(job_sims, key=lambda x: x[0], reverse=True)[:3]

            for s, info in job_sims:
                recommended.append({
                    "job_id": str(info.get("job_id")),
                    "job": str(info.get("title")),
                    "match": round(float(s) * 100, 2)
                })

        return {
            "matched_competences": matched,
            "block_scores": block_scores,
            "coverage_score": round(float(coverage_score), 3),
            "profile_recommendation": profile_level,
            "recommended_jobs": recommended
        }

    except Exception as e:
        return {"error": str(e), "matched_competences": [], "block_scores": {}, "recommended_jobs": []}


@app.get("/health")
def health():
    return {"status": "ok"}
