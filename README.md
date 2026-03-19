# RAG Informatique Maroc - Streamlit Cloud

Assistant pedagogique base sur les instructions officielles MEN.

## Structure du projet

```
rag_deploy/
├── app.py                    <- Application principale
├── build_index.py            <- Script indexation (une seule fois sur PC)
├── requirements.txt          <- Dependances Python
├── data/
│   ├── college.index         <- Index FAISS College (commite sur GitHub)
│   ├── college.json          <- Passages College
│   ├── tc.index              <- Index FAISS TC
│   └── tc.json               <- Passages TC
├── pdfs/                     <- PDFs sources (NON commite, .gitignore)
│   ├── IO_college_2006.pdf
│   └── IO_TC_2005.pdf
└── .streamlit/
    ├── config.toml           <- Config Streamlit
    └── secrets.toml          <- Cles API (NON commite, .gitignore)
```

## Deploiement - Etapes

### Etape 1 : Preparer l'index sur votre PC

```bash
# 1. Creer le dossier pdfs/ et y copier vos PDFs
mkdir pdfs
copy "IO_college_2006.pdf" pdfs\
copy "instruction_officiles_TC_2005.pdf" pdfs\IO_TC_2005.pdf

# 2. Installer les dependances
pip install -r requirements.txt

# 3. Lancer le script d'indexation
python build_index.py
```

Cela genere les fichiers dans data/ (college.index, college.json, tc.index, tc.json).

### Etape 2 : Pousser sur GitHub

```bash
git init
git add .
git commit -m "Initial deploy RAG Informatique Maroc"
git remote add origin https://github.com/VOTRE_USERNAME/rag-maroc.git
git push -u origin main
```

IMPORTANT : data/ est commite (index pre-calcule).
IMPORTANT : pdfs/ et .streamlit/secrets.toml ne sont PAS commites (.gitignore).

### Etape 3 : Deployer sur Streamlit Cloud

1. Allez sur share.streamlit.io
2. Connectez-vous avec GitHub
3. Cliquez "New app"
4. Selectionnez votre repo et app.py
5. Allez dans Settings > Secrets et ajoutez :

```toml
GEMINI_API_KEY = "AIza..."
OPENROUTER_API_KEY = "sk-or-..."
```

6. Cliquez Deploy !

### Obtenir les cles API gratuites

**Gemini** : aistudio.google.com > Get API Key
**OpenRouter** : openrouter.ai > Keys (acces DeepSeek + Llama gratuits)

## Pour l'utilisateur final

L'utilisateur accede directement a l'URL Streamlit Cloud.
- Aucune cle API a entrer
- Aucun PDF a charger
- Choix du modele IA via bouton radio
- Telechargement des fiches/evaluations/cours en .txt
