"""
build_index.py
==============
A lancer UNE SEULE FOIS sur votre PC avant de pousser sur GitHub.
Il lit les deux PDFs et sauvegarde les index FAISS dans data/

Usage:
    python build_index.py

Les fichiers generes (data/college.*, data/tc.*) sont commites sur GitHub.
Streamlit Cloud les charge directement au demarrage - aucun upload d'user necessaire.
"""

import sys
from pathlib import Path

# ── Check deps ────────────────────────────────
try:
    import pdfplumber, faiss, numpy as np, json, re
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"ERREUR: {e}")
    print("Installez les dependances: pip install pdfplumber pypdf faiss-cpu sentence-transformers")
    sys.exit(1)

# ── Config ────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 300
DATA_DIR      = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS = {
    "college": {
        "pdf":    Path(__file__).parent / "pdfs" / "IO_college_2006.pdf",
        "index":  DATA_DIR / "college.index",
        "chunks": DATA_DIR / "college.json",
        "label":  "College (1AC/2AC/3AC)",
    },
    "tc": {
        "pdf":    Path(__file__).parent / "pdfs" / "IO_TC_2005.pdf",
        "index":  DATA_DIR / "tc.index",
        "chunks": DATA_DIR / "tc.json",
        "label":  "Tronc Commun",
    },
}

# ── Functions ─────────────────────────────────
def extract_pdf(path):
    pages = []
    print(f"  Lecture de {path.name}...")
    try:
        with pdfplumber.open(path) as pdf:
            for i, pg in enumerate(pdf.pages):
                txt = pg.extract_text(x_tolerance=2, y_tolerance=2) or ""
                for tbl in (pg.extract_tables() or []):
                    for row in tbl:
                        if row:
                            r = " | ".join(str(c or "").strip() for c in row)
                            if r.strip("| "): txt += "\n" + r
                txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
                if txt:
                    pages.append({"page": i + 1, "text": txt, "source": path.name})
    except Exception as e:
        print(f"  ERREUR pdfplumber: {e} - tentative pypdf...")
        reader = PdfReader(str(path))
        for i, pg in enumerate(reader.pages):
            txt = pg.extract_text() or ""
            if txt.strip():
                pages.append({"page": i + 1, "text": txt, "source": path.name})
    print(f"  {len(pages)} pages extraites.")
    return pages

def chunk_pages(pages):
    chunks, meta = [], []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            chunk = text[start: start + CHUNK_SIZE]
            if chunk.strip():
                chunks.append(chunk)
                meta.append({"source": p["source"], "page": p["page"]})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks, meta

def build_and_save(corpus_id, cfg, model):
    print(f"\n[{cfg['label']}]")
    if not cfg["pdf"].exists():
        print(f"  ERREUR: PDF non trouve: {cfg['pdf']}")
        print(f"  Placez le PDF dans: {cfg['pdf'].parent}/")
        return False

    pages = extract_pdf(cfg["pdf"])
    chunks, meta = chunk_pages(pages)
    print(f"  {len(chunks)} passages crees.")

    print(f"  Encodage des embeddings (peut prendre 1-2 minutes)...")
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)

    print(f"  Construction de l'index FAISS...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    print(f"  Sauvegarde dans {cfg['index'].name} + {cfg['chunks'].name}...")
    faiss.write_index(index, str(cfg["index"]))
    cfg["chunks"].write_text(
        json.dumps({"chunks": chunks, "metadata": meta, "pdf_name": cfg["pdf"].name},
                   ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  OK - {len(chunks)} passages sauvegardes.")
    return True

# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Build Index - RAG Informatique Maroc")
    print("=" * 55)

    # Check pdfs/ folder
    pdf_dir = Path(__file__).parent / "pdfs"
    if not pdf_dir.exists():
        pdf_dir.mkdir()
        print(f"\nDossier 'pdfs/' cree.")
        print(f"Copiez vos PDFs ici:")
        print(f"  {pdf_dir / 'IO_college_2006.pdf'}")
        print(f"  {pdf_dir / 'IO_TC_2005.pdf'}")
        sys.exit(0)

    print("\nChargement du modele d'embeddings (premiere fois: telechargement ~90MB)...")
    model = SentenceTransformer(EMBED_MODEL)
    print("Modele charge.")

    ok_count = 0
    for corpus_id, cfg in CORPUS.items():
        if build_and_save(corpus_id, cfg, model):
            ok_count += 1

    print("\n" + "=" * 55)
    if ok_count == 2:
        print("  SUCCES: Les deux index sont prets !")
        print("  Vous pouvez maintenant pousser sur GitHub.")
        print()
        print("  Fichiers generes:")
        for cfg in CORPUS.values():
            print(f"    {cfg['index']}  ({cfg['index'].stat().st_size // 1024} KB)")
            print(f"    {cfg['chunks']}  ({cfg['chunks'].stat().st_size // 1024} KB)")
    else:
        print(f"  {ok_count}/2 index construits. Verifiez les erreurs ci-dessus.")
    print("=" * 55)
