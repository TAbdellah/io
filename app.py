"""
app.py - RAG Informatique Maroc - Streamlit Cloud
PDFs pre-indexes, cles API dans Secrets. Zero configuration user.
"""
import sys, os, re, json
from pathlib import Path
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR    = Path(__file__).parent / "data"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 10

CORPUS = {
    "college": {
        "label":   "College (1AC / 2AC / 3AC)",
        "niveaux": ["1AC","2AC","3AC"],
        "index":   DATA_DIR / "college.index",
        "chunks":  DATA_DIR / "college.json",
    },
    "tc": {
        "label":   "Tronc Commun (TC)",
        "niveaux": ["TC"],
        "index":   DATA_DIR / "tc.index",
        "chunks":  DATA_DIR / "tc.json",
    },
}

NIVEAU_LABELS = {
    "1AC":"1ere Annee College (6eme)",
    "2AC":"2eme Annee College (5eme)",
    "3AC":"3eme Annee College (4eme/3eme)",
    "TC" :"Tronc Commun",
    "tous":"Tous les niveaux",
}

PROGRAMME = {
    "1AC":{"unites":["U1: Systeme informatique + SE","U2: Traitement de texte","U3: Recherche documentaire"],
           "competences":["C0","C11","C21","C22","C31"],"heures":"30h",
           "logiciels":["SE","Texteur","Utilitaires"]},
    "2AC":{"unites":["U1: Materiel + Reseau local","U2: Echange reseau","U3: Tableur","U4: Logo"],
           "competences":["C0","C12","C13","C31","C32"],"heures":"30h",
           "logiciels":["Tableur","Logo"]},
    "3AC":{"unites":["U1: Reseaux LAN/MAN/WAN","U2: PAO","U3: Logo avance","U4: Web","U5: Messagerie"],
           "competences":["C0","C11","C13","C23","C33"],"heures":"30h",
           "logiciels":["PAO","Logo","Navigateur","Messagerie"]},
    "TC" :{"unites":["Module 1: Systemes informatiques","Module 2: Logiciels","Module 3: Algorithmique","Module 4: Reseaux"],
           "competences":["Gerer fichiers","Texteur","Tableur","Graphiques","Internet","Algorithmique","Programmation"],
           "heures":"68h","logiciels":["SE","Texteur","Tableur","IDE Pascal","Navigateur"]},
}

LLM_PROVIDERS = {
    "Gemini 2.5 Flash":{
        "id":"gemini","model":"gemini-2.5-flash","secret":"GEMINI_API_KEY",
        "desc":"Rapide, Fr/Ar, 1M tokens","icon":"G"},
    "DeepSeek V3":{
        "id":"openrouter","model":"deepseek/deepseek-chat-v3-0324:free",
        "secret":"OPENROUTER_API_KEY","base_url":"https://openrouter.ai/api/v1",
        "desc":"Excellent raisonnement","icon":"D"},
    "Llama 3.3 70B":{
        "id":"openrouter","model":"meta-llama/llama-3.3-70b-instruct:free",
        "secret":"OPENROUTER_API_KEY","base_url":"https://openrouter.ai/api/v1",
        "desc":"Open source Meta","icon":"L"},
}
DEFAULT_PROVIDER = "Gemini 2.5 Flash"

st.set_page_config(page_title="RAG Informatique Maroc",page_icon="M",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""<style>
.header-banner{background:linear-gradient(135deg,#1B5E20,#2E7D32 55%,#C62828);
  border-radius:14px;padding:20px 28px;margin-bottom:18px;display:flex;
  align-items:center;gap:16px;box-shadow:0 4px 18px rgba(0,0,0,0.18);}
.header-banner h1{color:#fff;margin:0;font-size:1.7rem;}
.header-banner p{color:#e8f5e9;margin:4px 0 0;font-size:0.9rem;}
.user-bubble{background:#E3F2FD;border-radius:14px 14px 4px 14px;
  padding:12px 16px;margin:8px 0;max-width:82%;margin-left:auto;color:#0D47A1;font-weight:500;}
.assistant-bubble{background:#F1F8E9;border-radius:14px 14px 14px 4px;
  padding:14px 18px;margin:8px 0;max-width:90%;border-left:4px solid #2E7D32;color:#1B5E20;}
.source-badge{background:#FFF8E1;border:1px solid #F9A825;border-radius:8px;
  padding:4px 10px;font-size:0.78rem;color:#6D4C41;margin:3px 2px;display:inline-block;}
.fiche-card{background:linear-gradient(135deg,#E8F5E9,#F1F8E9);
  border-radius:12px;padding:16px;margin:8px 0;border-left:4px solid #2E7D32;}
.eval-card{background:linear-gradient(135deg,#FFF3E0,#FFF8E1);
  border-radius:12px;padding:16px;margin:8px 0;border-left:4px solid #F9A825;}
.cours-card{background:linear-gradient(135deg,#E8EAF6,#EDE7F6);
  border-radius:12px;padding:16px;margin:8px 0;border-left:4px solid #3949AB;}
.model-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.8rem;
  font-weight:700;margin:2px;border:2px solid;}
</style>""", unsafe_allow_html=True)

# ── Load indexes at startup ───────────────────
@st.cache_resource(show_spinner="Chargement du modele d'embeddings...")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner="Chargement de l'index College...")
def load_corpus_college():
    cfg = CORPUS["college"]
    if cfg["index"].exists() and cfg["chunks"].exists():
        idx   = faiss.read_index(str(cfg["index"]))
        data  = json.loads(cfg["chunks"].read_text(encoding="utf-8"))
        return idx, data["chunks"], data["metadata"], data.get("pdf_name","IO_college_2006.pdf")
    return None, [], [], ""

@st.cache_resource(show_spinner="Chargement de l'index Tronc Commun...")
def load_corpus_tc():
    cfg = CORPUS["tc"]
    if cfg["index"].exists() and cfg["chunks"].exists():
        idx   = faiss.read_index(str(cfg["index"]))
        data  = json.loads(cfg["chunks"].read_text(encoding="utf-8"))
        return idx, data["chunks"], data["metadata"], data.get("pdf_name","IO_TC_2005.pdf")
    return None, [], [], ""

# ── Get API key from Streamlit Secrets ────────
def get_api_key(provider_name):
    cfg = LLM_PROVIDERS[provider_name]
    secret_name = cfg["secret"]
    try:
        return st.secrets[secret_name]
    except Exception:
        return None

# ── Retrieve from correct index ───────────────
def retrieve(query, model, provider, niveau, k=TOP_K):
    idx_col, ch_col, meta_col, _ = load_corpus_college()
    idx_tc,  ch_tc,  meta_tc,  _ = load_corpus_tc()

    if niveau in ["1AC","2AC","3AC"]:
        if idx_col is None: return []
        idx, chunks, meta = idx_col, ch_col, meta_col
    elif niveau == "TC":
        if idx_tc is None: return []
        idx, chunks, meta = idx_tc, ch_tc, meta_tc
    else:  # tous
        if idx_col is None and idx_tc is None: return []
        chunks = (ch_col or []) + (ch_tc or [])
        meta   = (meta_col or []) + (meta_tc or [])
        if not chunks: return []
        # Build combined index on-the-fly (cached via session)
        if "idx_all" not in st.session_state or \
           st.session_state.get("idx_all_size") != len(chunks):
            emb = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
            ix  = faiss.IndexFlatIP(emb.shape[1])
            ix.add(emb.astype("float32"))
            st.session_state["idx_all"] = ix
            st.session_state["idx_all_size"] = len(chunks)
        idx = st.session_state["idx_all"]

    qe = model.encode([query], normalize_embeddings=True).astype("float32")
    sc, ids = idx.search(qe, min(k, len(chunks)))
    return [{"chunk":chunks[i],"score":float(s),"source":meta[i]["source"],"page":meta[i]["page"]}
            for s,i in zip(sc[0],ids[0]) if i>=0 and float(s)>0.1]

# ── Call LLM ──────────────────────────────────
def call_llm(prompt, provider_name):
    cfg = LLM_PROVIDERS[provider_name]
    api_key = get_api_key(provider_name)
    if not api_key:
        raise ValueError(f"Cle API {cfg['secret']} non configuree dans les Secrets.")

    if cfg["id"] == "gemini":
        from google import genai
        from google.genai import types as gtypes
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=cfg["model"], contents=prompt,
            config=gtypes.GenerateContentConfig(max_output_tokens=8192, temperature=0.2))
        return response.text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"],
                        default_headers={"HTTP-Referer":"https://rag-maroc.streamlit.app"})
        resp = client.chat.completions.create(
            model=cfg["model"], max_tokens=8192, temperature=0.2,
            messages=[{"role":"user","content":prompt}])
        return resp.choices[0].message.content

# ── Build prompt ──────────────────────────────
def ctx_str(results):
    return "\n\n---\n\n".join(
        f"[{c['source']}, p.{c['page']}, score:{c['score']:.2f}]\n{c['chunk']}"
        for c in results)

def niv_context(niveau):
    p = PROGRAMME.get(niveau, {})
    if not p: return ""
    return (f"\nPROGRAMME OFFICIEL {NIVEAU_LABELS.get(niveau,niveau)} ({p.get('heures','')}):\n"
            f"Unites: {', '.join(p.get('unites',[]))}\n"
            f"Competences: {', '.join(p.get('competences',[]))}\n"
            f"Logiciels: {', '.join(p.get('logiciels',[]))}\n")

def build_prompt(query, results, mode, niveau):
    ctx  = ctx_str(results)
    prog = niv_context(niveau)
    niv_label = NIVEAU_LABELS.get(niveau, niveau)
    base = (f"Tu es un expert pedagogique senior en informatique pour le MEN Maroc.\n"
            f"{prog}\nEXTRAITS INSTRUCTIONS OFFICIELLES:\n{ctx}\n\n"
            f"REGLES: Cite source+page, ne jamais inventer, reponds en francais, complet.\n")

    STRUCTURES = {
        "fiche": f"""Genere une FICHE PEDAGOGIQUE COMPLETE pour {niv_label}:
# FICHE PEDAGOGIQUE
**Niveau:** {niv_label} | **Matiere:** Informatique

## 1. Competences et Objectifs
**Competence(s):** | **Objectifs (L'apprenant sera capable de...):** | **Prerequis:**

## 2. Materiel et Ressources
| Ressource | Quantite | Remarque |

## 3. Deroulement
| Phase | Duree | Activite Enseignant | Activite Apprenant | Support |
(mise en situation / decouverte / structuration / application / evaluation)

## 4. Differentiation
**En difficulte:** | **Avances:**

## 5. Trace Ecrite
(ce que l'apprenant recopie)

## 6. Evaluation Formative
(3 questions rapides avec reponses)

## 7. Prolongements
""",
        "eval": f"""Cree un OUTIL D'EVALUATION COMPLET pret a photocopier pour {niv_label}:
# EVALUATION - [TITRE]
**Classe:** {niv_label} | **Date:** ___ | **Duree:** ___ | **Note:** ___/20

## Competences evaluees
## Consignes generales
## Exercice 1 (/__ pts) - [Titre]
## Exercice 2 (/__ pts) - [Titre]
## Exercice 3 (/__ pts) - [Titre/Bonus]
---
## Bareme detaille
| Question | Points | Criteres |
## Grille de correction
## Grille de competences
| Competence | Non acquis | En cours | Acquis | Depasse |
""",
        "scenario": f"""Redige un SCENARIO PEDAGOGIQUE COMPLET pour {niv_label}:
# SCENARIO PEDAGOGIQUE - [TITRE]
**Niveau:** {niv_label} | **Duree totale:** ___ seances

## Situation Declenchante
## Competences et Objectifs de la Sequence
## Progression des Seances
### Seance 1: [Titre]
- Objectif | Prerequis | Deroulement | Ressources | Evaluation
(repeter pour chaque seance)
## Evaluation Sommative
## Ressources et Materiels
## Differenciation
""",
        "cours": f"""Redige un COURS COMPLET pret a utiliser pour {niv_label}:
# [TITRE DU COURS]
**Niveau:** {niv_label} | **Duree:** ___ | **Prerequis:** ___

## Introduction
(accroche + objectifs + lien competence officielle)

## I. [Premiere Notion]
### Definition | Explication | Exemple concret | Schema/Tableau

## II. [Deuxieme Notion]
(meme structure)

## Activites Pratiques
### Activite 1 (guidee): etapes pas-a-pas
### Activite 2 (semi-autonome): consigne + criteres

## Exercices d'Entrainement
### Exercice 1 - Facile: enonce + correction
### Exercice 2 - Moyen: enonce + correction
### Exercice 3 - Difficile: enonce + correction

## Synthese - Ce qu'il faut retenir
(5 points cles minimum)

## Glossaire
| Terme | Definition simple |

## Auto-evaluation
(5 questions de niveaux varies)
""",
    }

    if mode in STRUCTURES:
        return base + STRUCTURES[mode] + f"\nDEMANDE: {query}\n\nREPONSE COMPLETE:"
    else:
        return (base +
                f"Reponds completement, cite les sources, structure avec titres et listes.\n"
                f"QUESTION: {query}\n\nREPONSE:")

# ── Generate ──────────────────────────────────
def generate(query, mode, niveau, provider_name, k=TOP_K):
    model = load_embed_model()
    results = retrieve(query, model, provider_name, niveau, k=k)

    if not results:
        corpus = "College" if niveau in ["1AC","2AC","3AC"] else "Tronc Commun" if niveau=="TC" else "les PDFs"
        st.error(f"Aucun passage trouve dans {corpus}. Verifiez le deploiement.")
        return None, []

    if results[0]["score"] < 0.2:
        st.warning(f"Pertinence faible ({results[0]['score']:.2f}). Le PDF contient peu d'info sur ce sujet.")

    prompt = build_prompt(query, results, mode, niveau)
    try:
        return call_llm(prompt, provider_name), results
    except Exception as e:
        st.error(f"Erreur LLM ({provider_name}): {e}")
        return None, []

# ── Session state ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state["provider"] = list(LLM_PROVIDERS.keys())[0]

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("## Parametres")

    # Model selector
    st.markdown("### Modele IA")
    provider = st.radio(
        "Choisir le modele",
        list(LLM_PROVIDERS.keys()),
        format_func=lambda x: f"{LLM_PROVIDERS[x]['icon']} {x} — {LLM_PROVIDERS[x]['desc']}",
        label_visibility="collapsed",
    )
    st.session_state.provider = provider

    # Check key availability
    key_ok = get_api_key(provider) is not None
    if key_ok:
        st.success(f"Modele disponible")
    else:
        st.error("Modele non disponible (cle manquante)")

    st.divider()

    # Index status
    st.markdown("### Base de connaissances")
    idx_col, ch_col, _, pname_col = load_corpus_college()
    idx_tc,  ch_tc,  _, pname_tc  = load_corpus_tc()

    if idx_col is not None:
        st.success(f"College: {len(ch_col)} passages")
        st.caption(f"Source: {pname_col}")
    else:
        st.error("College: index manquant")

    if idx_tc is not None:
        st.success(f"Tronc Commun: {len(ch_tc)} passages")
        st.caption(f"Source: {pname_tc}")
    else:
        st.error("Tronc Commun: index manquant")

    st.divider()
    top_k = st.slider("Passages recuperes", 5, 15, TOP_K)

    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ────────────────────────────────────
_col_ok  = f"College OK ({len(ch_col)} passages)" if idx_col else "College: non disponible"
_tc_ok   = f"TC OK ({len(ch_tc)} passages)"       if idx_tc  else "TC: non disponible"
st.markdown(f"""<div class="header-banner">
  <span style="font-size:2.4rem">🇲🇦</span>
  <div><h1>RAG Informatique Maroc</h1>
  <p>Assistant pedagogique MEN | Instructions officielles integrees | College & Tronc Commun</p>
  <p style="font-size:0.78rem;color:#A5D6A7;margin-top:4px">{_col_ok}  |  {_tc_ok}</p>
  </div></div>""", unsafe_allow_html=True)

# ── Niveau selector helper ────────────────────
def niveau_selector(key, include_tous=False):
    opts = []
    if include_tous: opts.append("tous")
    if idx_col: opts += ["1AC","2AC","3AC"]
    if idx_tc:  opts.append("TC")
    if not opts: opts = list(NIVEAU_LABELS.keys())
    return st.selectbox("Niveau", opts,
        format_func=lambda x: NIVEAU_LABELS[x], key=key)

# ── Tabs ──────────────────────────────────────
tab_chat, tab_fiche, tab_eval, tab_scenario, tab_cours = st.tabs([
    "Chat libre","Fiche pedagogique","Evaluation","Scenario pedagogique","Cours complet"])

# ── TAB 1: CHAT ───────────────────────────────
with tab_chat:
    col_q, col_niv = st.columns([3,1])
    with col_niv:
        chat_niv = niveau_selector("chat_niv", include_tous=True)

    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(f'<div class="user-bubble">Vous : {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"Sources utilisees ({len(msg['sources'])} passages)"):
                    for i,s in enumerate(msg["sources"],1):
                        st.markdown(f"**{i}.** `{s['source']}` p.{s['page']} (score:{s['score']:.2f})")
                        st.caption(s["chunk"][:300]+"...")
                        st.divider()

    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([5,1])
        query = cols[0].text_input("question", label_visibility="collapsed",
            placeholder="Ex: Quelles sont les competences C13 en 2AC ?")
        submitted = cols[1].form_submit_button("Envoyer")

    if submitted and query.strip():
        st.session_state.messages.append({"role":"user","content":query})
        with st.spinner(f"Recherche via {st.session_state.provider}..."):
            answer, sources = generate(query,"chat",chat_niv,st.session_state.provider,k=top_k)
        if answer:
            st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
            st.rerun()

# ── TAB 2: FICHE ─────────────────────────────
with tab_fiche:
    st.markdown("### Generateur de Fiches Pedagogiques")
    col1, col2 = st.columns(2)
    with col1:
        f_niv   = niveau_selector("f_niv")
        f_unite = st.text_input("Unite / Module", placeholder="Ex: U4 Programmation Logo")
    with col2:
        f_seance = st.text_input("Titre de la seance", placeholder="Ex: Primitives de base Logo")
        f_duree  = st.selectbox("Duree", ["1h","2h","3h","Sequence"])
    f_ctx = st.text_area("Contexte (optionnel)", height=60,
        placeholder="Ex: 30 apprenants, 15 PC, debut de module...")

    p = PROGRAMME.get(f_niv, {})
    if p:
        with st.expander(f"Programme officiel {NIVEAU_LABELS[f_niv]}"):
            st.markdown(f"**Duree:** {p['heures']} | **Logiciels:** {', '.join(p['logiciels'])}")
            for u in p["unites"]: st.markdown(f"- {u}")

    if st.button("Generer la fiche", use_container_width=True, type="primary"):
        q = f"Fiche pedagogique {NIVEAU_LABELS[f_niv]} - Unite '{f_unite}' - Seance '{f_seance}' - Duree {f_duree}. {f_ctx}"
        with st.spinner("Generation de la fiche..."):
            answer, sources = generate(q,"fiche",f_niv,st.session_state.provider,k=top_k)
        if answer:
            st.markdown('<div class="fiche-card">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("Sources utilisees"):
                for s in sources:
                    st.markdown(f'<span class="source-badge">{s["source"]} p.{s["page"]}</span>',
                                unsafe_allow_html=True)
            fname = f"fiche_{f_niv}_{f_seance}".replace(" ","_")
            st.download_button("Telecharger (.txt)", data=answer.encode("utf-8"),
                file_name=fname+".txt", mime="text/plain", use_container_width=True)

# ── TAB 3: EVAL ──────────────────────────────
with tab_eval:
    st.markdown("### Generateur d'Evaluations")
    col1, col2, col3 = st.columns(3)
    with col1:
        e_niv  = niveau_selector("e_niv")
        e_type = st.selectbox("Type", ["Controle ecrit","QCM","TP note",
                                        "Grille de competences","Evaluation diagnostique"])
    with col2:
        e_chap  = st.text_input("Chapitre / Competence", placeholder="Ex: C13 Programmation Logo")
        e_duree = st.selectbox("Duree", ["20 min","30 min","1h","2h"])
    with col3:
        e_note = st.number_input("Note sur", 10, 40, 20, 5)
        e_diff = st.select_slider("Difficulte", ["Facile","Moyen","Difficile","Mixte"], value="Moyen")
    e_obj = st.text_area("Objectifs specifiques (optionnel)", height=60)

    if st.button("Generer l'evaluation", use_container_width=True, type="primary"):
        q = f"{e_type} {NIVEAU_LABELS[e_niv]} - '{e_chap}' - {e_duree} - /{e_note} - {e_diff}. {e_obj}"
        with st.spinner("Generation de l'evaluation..."):
            answer, sources = generate(q,"eval",e_niv,st.session_state.provider,k=top_k)
        if answer:
            st.markdown('<div class="eval-card">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)
            st.download_button("Telecharger (.txt)", data=answer.encode("utf-8"),
                file_name=f"eval_{e_niv}_{e_chap}.txt".replace(" ","_"), mime="text/plain",
                use_container_width=True)

# ── TAB 4: SCENARIO ──────────────────────────
with tab_scenario:
    st.markdown("### Generateur de Scenarios Pedagogiques")
    col1, col2 = st.columns(2)
    with col1:
        s_niv   = niveau_selector("s_niv")
        s_unite = st.text_input("Unite thematique", placeholder="Ex: U3 Tableur")
        s_nb    = st.number_input("Nombre de seances", 1, 10, 3)
    with col2:
        s_meth = st.multiselect("Methodes", ["Resolution de problemes","Pedagogie de projet",
            "Methode de decouverte","Travail collaboratif","Classe inversee"],
            default=["Resolution de problemes"])
        s_res  = st.multiselect("Ressources", ["Salle informatique","Tableau","Manuel",
            "Videoprojecteur","Reseau local","Internet"], default=["Salle informatique"])
    s_notes = st.text_area("Notes (optionnel)", height=60)

    if st.button("Generer le scenario", use_container_width=True, type="primary"):
        q = (f"Scenario {NIVEAU_LABELS[s_niv]} - '{s_unite}' - {s_nb} seances - "
             f"Methodes: {', '.join(s_meth)} - Ressources: {', '.join(s_res)}. {s_notes}")
        with st.spinner("Elaboration du scenario..."):
            answer, sources = generate(q,"scenario",s_niv,st.session_state.provider,k=top_k)
        if answer:
            st.markdown(answer)
            st.download_button("Telecharger (.txt)", data=answer.encode("utf-8"),
                file_name=f"scenario_{s_niv}_{s_unite}.txt".replace(" ","_"), mime="text/plain",
                use_container_width=True)

# ── TAB 5: COURS ─────────────────────────────
with tab_cours:
    st.markdown("### Generateur de Cours Complets")
    col1, col2 = st.columns(2)
    with col1:
        c_niv   = niveau_selector("c_niv")
        c_titre = st.text_input("Titre du cours", placeholder="Ex: Le tableur - Formules et fonctions")
        p = PROGRAMME.get(c_niv, {})
        if p:
            with st.expander(f"Programme {NIVEAU_LABELS[c_niv]}", expanded=False):
                st.markdown(f"**Duree:** {p['heures']}")
                for u in p["unites"]: st.markdown(f"- {u}")
    with col2:
        c_unite = st.selectbox("Unite", ["---"] + PROGRAMME.get(c_niv,{}).get("unites",["---"]),
                               key="c_unite")
        c_comp  = st.selectbox("Competence", ["---"] + PROGRAMME.get(c_niv,{}).get("competences",["---"]),
                               key="c_comp")
        c_pub   = st.selectbox("Profil", ["Debutants","Quelques notions","Intermediaire"])
        c_lang  = st.selectbox("Langue", ["Francais","Arabe","Bilingue Fr/Ar"])
    c_obj = st.text_area("Objectifs (optionnel)", height=60)
    st.info("Le cours sera genere avec toutes les sections : Intro, Notions, Activites, "
            "Exercices (3 niveaux + corrections), Synthese, Glossaire, Auto-evaluation.")

    if st.button("Generer le cours complet", use_container_width=True, type="primary"):
        if not c_titre.strip():
            st.error("Saisissez le titre du cours.")
        else:
            q = (f"Cours complet '{c_titre}' - {NIVEAU_LABELS[c_niv]} - Unite: {c_unite} - "
                 f"Competence: {c_comp} - Public: {c_pub} - Langue: {c_lang}. {c_obj}")
            with st.spinner("Redaction du cours (30-60s)..."):
                answer, sources = generate(q,"cours",c_niv,st.session_state.provider,k=top_k)
            if answer:
                st.markdown('<div class="cours-card">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)
                fname = f"cours_{c_niv}_{c_titre}".replace(" ","_")
                st.download_button("Telecharger (.txt)", data=answer.encode("utf-8"),
                    file_name=fname+".txt", mime="text/plain", use_container_width=True)

# ── Footer ────────────────────────────────────
st.divider()
st.markdown("<p style='text-align:center;color:#888;font-size:0.82rem;'>"
    "RAG Informatique Maroc | MEN 2005-2006 | Instructions officielles integrees | "
    "College 1AC/2AC/3AC + Tronc Commun | Streamlit Cloud</p>",
    unsafe_allow_html=True)
