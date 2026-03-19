<div align="center">

# 🇲🇦 RAG Informatique Maroc

### Assistant pédagogique intelligent basé sur les Instructions Officielles MEN

[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/cloud)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/Licence-MIT-2E7D32)](./LICENSE)
[![Gratuit](https://img.shields.io/badge/100%25-Gratuit-F9A825)](.)

**Application web publique** — aucune installation, aucune clé API, aucun upload pour l'utilisateur final.

[Voir l'application →](https://meninfo.streamlit.app/)

</div>

---

## 🎯 Présentation du projet

**RAG Informatique Maroc** est un assistant pédagogique intelligent destiné aux **enseignants d'informatique du système éducatif marocain**. Il utilise la technique RAG (*Retrieval-Augmented Generation*) pour répondre à toutes les questions pédagogiques en s'appuyant **exclusivement** sur les instructions officielles du Ministère de l'Education Nationale (MEN).

### Pourquoi ce projet ?

Les enseignants d'informatique au Maroc disposent de deux documents officiels fondamentaux :
- **Instructions Officielles — Secondaire Collégial** (2006) : programme 1AC, 2AC, 3AC
- **Instructions Officielles — Troncs Communs** (2005) : programme TC Sciences, Lettres, Technologique

Ces documents sont denses et complexes. Ce projet permet à tout enseignant de **questionner directement ces textes** et de générer automatiquement des ressources pédagogiques complètes et conformes au programme officiel.

---

## ✨ Fonctionnalités

| Onglet | Description |
|--------|-------------|
| 💬 **Chat libre** | Posez n'importe quelle question sur le programme officiel |
| 📋 **Fiche pédagogique** | Génère une fiche complète (objectifs, déroulement, évaluation) |
| 📝 **Évaluation** | Crée contrôles, QCM, TP notés avec barème et grille de correction |
| 🗺️ **Scénario pédagogique** | Planifie une séquence multi-séances avec méthodes actives |
| 📖 **Cours complet** | Rédige un cours structuré avec exercices, glossaire et auto-évaluation |

### Ce que l'application garantit

- ✅ **Fidélité au PDF** — chaque réponse cite la source (fichier + page)
- ✅ **Séparation stricte** — l'index Collège et l'index TC ne se mélangent jamais
- ✅ **Fallback automatique** — si un modèle IA est indisponible, l'app bascule sur le suivant
- ✅ **Export** — téléchargement de chaque document généré en `.txt`
- ✅ **Zéro configuration** pour l'utilisateur final

---

## 🏗️ Architecture technique

```
┌─────────────────────────────────────────────────────────┐
│                    UTILISATEUR FINAL                     │
│              (navigateur web, aucune install)            │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────┐
│                  STREAMLIT CLOUD                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                   app.py                        │    │
│  │                                                 │    │
│  │  1. Embed query (Sentence-Transformers local)   │    │
│  │  2. Recherche FAISS (index pre-calcule)         │    │
│  │  3. Construction du prompt avec contexte PDF    │    │
│  │  4. Appel LLM (Gemini ou OpenRouter)            │    │
│  │  5. Affichage + telechargement                  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  data/college.index  ←──── IO_college_2006.pdf           │
│  data/tc.index       ←──── IO_TC_2005.pdf                │
│  .streamlit/secrets  ←──── Cles API (invisibles)         │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Google Gemini API         OpenRouter API
   (Gemini 2.5 Flash)    (Llama, Mistral, etc.)
```

### Stack technologique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Interface | Streamlit | UI web responsive |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Vectorisation multilingue Fr/Ar |
| Index vectoriel | FAISS (IndexFlatIP) | Recherche de similarité cosinus |
| LLM principal | Google Gemini 2.5 Flash | Génération de contenu |
| LLM alternatifs | OpenRouter (Llama 3.3, Mistral...) | Fallback automatique |
| Déploiement | Streamlit Cloud | Hébergement gratuit |

---

## 📚 Base de connaissances

### Deux index FAISS totalement séparés

```
data/
├── college.index     ← FAISS exclusif IO_college_2006.pdf
├── college.json      ← 63 passages extraits (chunking 1500 chars, overlap 300)
├── tc.index          ← FAISS exclusif IO_TC_2005.pdf
└── tc.json           ← 61 passages extraits
```

### Niveaux couverts

| Niveau |  Compétences clés |
|--------|-----------------|
| **1AC** | C0, C11, C21, C22, C31 |
| **2AC** | C0, C12, C13, C31, C32 |
| **3AC** | C0, C11, C13, C23, C33 |
| **TC** | Modules 1→4 (68h annuelles) |

### Modèles IA disponibles

| Modèle | Fournisseur | Contexte | Statut |
|--------|-------------|---------|--------|
| Gemini 2.5 Flash | Google | 1M tokens | ✅ Recommandé |
| Llama 3.3 70B | Meta / OpenRouter | 128K | ✅ Stable |
| Mistral Small 3.1 | Mistral / OpenRouter | 128K | ✅ Français natif |
| Gemma 3 27B | Google / OpenRouter | 131K | ✅ Multilingue |
| Hermes 3 Llama 405B | Nous Research / OpenRouter | 131K | ✅ Puissant |
| Nemotron 120B | NVIDIA / OpenRouter | 262K | ✅ Raisonnement |
| OpenRouter Auto | OpenRouter | 200K | ✅ Fallback auto |

> **Fallback automatique** : si le modèle choisi retourne une erreur 429 (quota dépassé), l'app essaie automatiquement Gemini puis les autres modèles.

---

## 🚀 Déploiement — Guide complet

### Prérequis

- Python 3.10+ sur votre PC
- Compte GitHub
- Compte Streamlit Cloud (gratuit) : [share.streamlit.io](https://share.streamlit.io)
- Clé API Gemini (gratuite) : [aistudio.google.com](https://aistudio.google.com)
- Clé API OpenRouter (gratuite) : [openrouter.ai](https://openrouter.ai)

---

### Étape 1 — Préparer l'index vectoriel (sur votre PC, une seule fois)

```bash
# 1. Cloner ou télécharger ce repo
git clone https://github.com/VOTRE_USERNAME/rag-maroc.git
cd rag-maroc

# 2. Créer le dossier pdfs/ et y copier vos PDFs
mkdir pdfs
# Copiez IO_college_2006.pdf → pdfs/IO_college_2006.pdf
# Copiez IO_TC_2005.pdf     → pdfs/IO_TC_2005.pdf

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le script d'indexation
python build_index.py
```

Le script va :
1. Extraire le texte de chaque PDF page par page
2. Découper en passages de 1500 caractères avec 300 de chevauchement
3. Calculer les embeddings multilingues (modèle ~90MB, téléchargé automatiquement)
4. Sauvegarder l'index FAISS dans `data/`

**Durée** : 2 à 5 minutes selon votre machine.

**Résultat attendu** :
```
[College (1AC/2AC/3AC)]
  248 passages créés.
  Encodage des embeddings...
  OK - 248 passages sauvegardés.

[Tronc Commun]
  186 passages créés.
  OK - 186 passages sauvegardés.

SUCCÈS: Les deux index sont prêts !
```

---

### Étape 2 — Pousser sur GitHub

```bash
git init
git add .
# Vérifier que pdfs/ et secrets.toml ne sont PAS inclus
git status
git commit -m "Initial deploy - RAG Informatique Maroc"
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/rag-maroc.git
git push -u origin main
```

> ⚠️ **Important** : Le fichier `.gitignore` exclut automatiquement `pdfs/` et `.streamlit/secrets.toml`. Vos PDFs et clés API ne sont **jamais** uploadés sur GitHub.

> ✅ **data/** est commité sur GitHub — c'est voulu. L'index pré-calculé permet un chargement instantané sans recalcul sur le serveur.

---

### Étape 3 — Déployer sur Streamlit Cloud

1. Allez sur **[share.streamlit.io](https://share.streamlit.io)**
2. Connectez-vous avec votre compte GitHub
3. Cliquez **"New app"**
4. Remplissez :
   - **Repository** : `VOTRE_USERNAME/rag-maroc`
   - **Branch** : `main`
   - **Main file path** : `app.py`
5. Cliquez **"Advanced settings"** → **"Secrets"** et collez :

```toml
GEMINI_API_KEY = "AIza..."
OPENROUTER_API_KEY = "sk-or-..."
```

6. Cliquez **"Deploy !"**

L'application sera disponible sur une URL du type :
`https://rag-maroc-XXXX.streamlit.app`

---

### Étape 4 — Obtenir les clés API gratuites

#### Clé Gemini (Google AI Studio)
1. Allez sur [aistudio.google.com](https://aistudio.google.com)
2. Connectez-vous avec un **compte Gmail personnel** (pas école/travail)
3. Cliquez **"Get API key"** → **"Create API key in new project"**
4. Copiez la clé `AIza...`

#### Clé OpenRouter
1. Allez sur [openrouter.ai](https://openrouter.ai)
2. Créez un compte (email ou GitHub)
3. Allez dans **Settings → Keys → Create Key**
4. Copiez la clé `sk-or-...`
5. Pas besoin d'ajouter des crédits — les modèles `:free` sont gratuits

---

### Mises à jour futures

Pour mettre à jour l'app après un changement de code :

```bash
git add app.py
git commit -m "Mise a jour app.py"
git push
```

Streamlit Cloud redéploie automatiquement en 1 à 2 minutes.

Pour reconstruire l'index (si vous changez les PDFs) :

```bash
python build_index.py
git add data/
git commit -m "Mise a jour index"
git push
```

---

## 📁 Structure du projet

```
rag-maroc/
│
├── app.py                  ← Application Streamlit principale
├── build_index.py          ← Script d'indexation (PC uniquement)
├── requirements.txt        ← Dépendances Python
├── README.md               ← Ce fichier
│
├── data/                   ← Index vectoriels (commités sur GitHub)
│   ├── college.index       ← Index FAISS — Collège (1AC/2AC/3AC)
│   ├── college.json        ← Passages + métadonnées — Collège
│   ├── tc.index            ← Index FAISS — Tronc Commun
│   └── tc.json             ← Passages + métadonnées — TC
│
├── pdfs/                   ← PDFs sources (dans .gitignore, jamais commité)
│   ├── IO_college_2006.pdf
│   └── IO_TC_2005.pdf
│
└── .streamlit/
    ├── config.toml         ← Thème et configuration Streamlit
    └── secrets.toml        ← Clés API (dans .gitignore, jamais commité)
```

---

## 🔒 Sécurité et confidentialité

| Élément | Statut | Détail |
|---------|--------|--------|
| Clés API | 🔒 Secrets Streamlit | Jamais exposées, jamais dans le code |
| PDFs sources | 🔒 Local uniquement | Dans `.gitignore`, jamais sur GitHub |
| Index FAISS | ✅ Public | Données dérivées, pas les PDFs originaux |
| Données utilisateur | ✅ Aucune | Pas de base de données, pas de logs utilisateur |

---

## 🤝 Contribution

Ce projet est destiné à la communauté des enseignants d'informatique marocains. Toute contribution est bienvenue :

- 🐛 **Signaler un bug** : ouvrez une Issue GitHub
- 💡 **Suggérer une amélioration** : ouvrez une Discussion
- 🔧 **Contribuer au code** : Fork → Branch → Pull Request


<div align="center">

**RAG Informatique Maroc** | Conforme aux programmes MEN 2005-2006

Propulsé par [Streamlit](https://streamlit.io) · [Google Gemini](https://ai.google.dev) · [OpenRouter](https://openrouter.ai) · [FAISS](https://faiss.ai) · [Sentence-Transformers](https://sbert.net)

</div>
