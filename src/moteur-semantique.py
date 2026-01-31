import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "csv")
COMP_PATH = os.path.join(CSV_DIR, "competences.csv")
METIERS_PATH = os.path.join(CSV_DIR, "metiers.csv")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Simple payload model
class UserInput(BaseModel):
	free_text: str
	other_answers: Optional[Dict[str, Any]] = None


# Load CSVs
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

# Load model once
MODEL_NAME = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
print("Loading SBERT model:", MODEL_NAME)
MODEL = SentenceTransformer(MODEL_NAME)

# Compute competence embeddings if available
COMPETENCE_NAMES: List[str] = COMP_DF["CompetencyName"].astype(str).tolist() if not COMP_DF.empty else []
if len(COMPETENCE_NAMES) > 0:
	COMP_EMBS = MODEL.encode(COMPETENCE_NAMES, convert_to_tensor=True)
else:
	COMP_EMBS = None


# Precompute job embeddings (mean of required competence embeddings)
JOB_EMBS = []
JOB_ROWS = []
if COMP_EMBS is not None and not METIERS_DF.empty:
	for _, row in METIERS_DF.iterrows():
		reqs = str(row.get("RequiredCompetencies", "")).strip()
		# Expect format like "C01;C02;C03"
		ids = [i.strip() for i in reqs.replace(",", ";").split(";") if i.strip()]
		emb_list = []
		for cid in ids:
			matches = COMP_DF[COMP_DF["CompetencyID"].astype(str) == str(cid)]
			if not matches.empty:
				idx = matches.index[0]
				emb_list.append(COMP_EMBS[int(idx)])
		if emb_list:
			mean_emb = np.mean([e.cpu().numpy() for e in emb_list], axis=0)
			JOB_EMBS.append(mean_emb)
			JOB_ROWS.append({"job_id": row.get("JobID"), "title": row.get("JobTitle"), "required": ids})


def _combine_text(free_text: str, other_answers: Optional[Dict[str, Any]] = None) -> str:
	parts = [free_text or ""]
	if other_answers:
		for k, v in other_answers.items():
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

		# Encode user text
		user_emb = MODEL.encode(combined, convert_to_tensor=True)

		# similarity to each competence
		sims = util.cos_sim(user_emb, COMP_EMBS)[0].cpu().numpy()

		# top 5 competences
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

		# block scores: average similarity per block
		block_map: Dict[str, List[float]] = {}
		for i, sc in enumerate(sims):
			blk = str(COMP_DF.iloc[int(i)].get("BlockName", ""))
			block_map.setdefault(blk, []).append(float(sc))
		block_scores = {b: float(np.mean(v)) for b, v in block_map.items()}

		# job similarities: compute cos between user_emb and each job mean emb
		recommended = []
		if JOB_EMBS:
			job_sims = []
			for jemb, jinfo in zip(JOB_EMBS, JOB_ROWS):
				# jemb is numpy array
				score = util.cos_sim(user_emb, jemb)[0]
				# cos_sim handles tensors/numpy - coerce to float
				try:
					s = float(score)
				except Exception:
					s = float(score.cpu().numpy())
				job_sims.append((s, jinfo))

			job_sims = sorted(job_sims, key=lambda x: x[0], reverse=True)[:3]
			for s, info in job_sims:
				recommended.append({"job_id": str(info.get("job_id")), "job": str(info.get("title")), "match": float(s)})

		return {"matched_competences": matched, "block_scores": block_scores, "recommended_jobs": recommended}
	except Exception as e:
		return {"error": str(e), "matched_competences": [], "block_scores": {}, "recommended_jobs": []}


@app.get("/health")
def health():
	return {"status": "ok"}
