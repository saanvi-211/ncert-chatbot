"""Build the FAISS textbook index. Run: python ingest.py  -> writes ./index/"""
import pathlib
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from backend.embed import get_embeddings

CHAPTERS = {
    "jesc101": "Chemical Reactions and Equations",
    "jesc102": "Acids, Bases and Salts",
    "jesc103": "Metals and Non-metals",
    "jesc104": "Carbon and its Compounds",
    "jesc105": "Life Processes",
    "jesc106": "Control and Coordination",
    "jesc107": "How do Organisms Reproduce?",
    "jesc108": "Heredity",
    "jesc109": "Light – Reflection and Refraction",
    "jesc110": "The Human Eye and the Colourful World",
    "jesc111": "Electricity",
    "jesc112": "Magnetic Effects of Electric Current",
    "jesc113": "Our Environment",
}

docs = []
for stem, name in CHAPTERS.items():
    path = pathlib.Path("data") / f"{stem}.pdf"
    if not path.exists():
        print("MISSING", path)
        continue
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, 1):
            try:
                pages.append(page.get_text())
            except Exception as e:
                print(f"  page {i} skipped: {e}")
    text = "\n".join(pages)
    docs.append(Document(page_content=text, metadata={"chapter": name}))
    print(f"read {name}: {len(text)} chars")

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
print(f"embedding {len(chunks)} chunks (first run downloads the model, please wait)...")
FAISS.from_documents(chunks, get_embeddings()).save_local("index")
print(f"Indexed {len(chunks)} chunks from {len(docs)} chapters")