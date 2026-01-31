Backend FastAPI (moteur sémantique)

Prérequis
- Python 3.8+

Installer et lancer (PowerShell)
```powershell
# depuis le dossier du projet (où se trouve requirements.txt)
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
# Lancer le serveur FastAPI
uvicorn src.moteur-semantique:app --reload --host 127.0.0.1 --port 8000
```

Endpoints
- POST /analyse  -> attend JSON {"free_text": "...", "other_answers": {...}}
- GET  /health   -> {"status":"ok"}

Remarques
- Le fichier `src/moteur-semantique.py` charge le modèle Sentence-BERT et lit `competences.csv` et `metiers.csv`. Assure-toi que ces CSV existent et sont accessibles.
- Si ton frontend React tourne sur un autre host/port, ajoute l'origine correspondante à la variable `origins` dans `moteur-semantique.py`.
- Si tu veux autoriser toutes les origines en développement, remplace `allow_origins=origins` par `allow_origins=["*"]` (à éviter en production).
