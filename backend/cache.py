"""Semantic cache = embedding match (recall) + deterministic guards (safety). No LLM is ever called here."""
import re, json, os, threading
import numpy as np, faiss

def _n(t): return t[:-1] if t.endswith("s") and len(t) > 3 else t
def toks(s): return [_n(t) for t in re.findall(r"[a-z]+", s.lower())]
def nums(s): return sorted(re.findall(r"\d+(?:\.\d+)?", s))
def formulas(s): return sorted(re.findall(r"\b(?:[A-Z][a-z]?\d*){2,}\b", s))

# Words that flip the meaning of a question while barely moving its embedding.
CONTRAST = [set(map(_n, g.split())) for g in [
    "concave convex plane", "real virtual", "acid base alkali salt", "endothermic exothermic",
    "oxidation reduction oxidising reducing", "mitosis meiosis", "series parallel", "ac dc",
    "metal nonmetal", "saturated unsaturated", "alkane alkene alkyne", "artery vein",
    "dominant recessive", "reflection refraction dispersion scattering", "myopia hypermetropia presbyopia",
    "sexual asexual", "autotrophic heterotrophic", "aerobic anaerobic",
    "increase decrease", "anode cathode", "ionic covalent", "male female", "xylem phloem",
    "genotype phenotype", "voltage current resistance", "mirror lens", "magnification power focal"]]

STOP = set("what is are the a an of does do mean define explain tell me about how please can you give".split())
STYLE = re.compile(r"\b(simpl\w*|shorter|briefly|in detail|elaborate|another example|rephrase|summari[sz]e|again|in hindi|like i'?m|eli5)\b", re.I)
ANAPHORA = re.compile(r"\b(it|its|it's|they|their|them|this|these|those|he|she|above|previous|earlier|same|former|latter|more|further|another|also)\b|^\s*(and|so|then|what about|how about)\b", re.I)

def key(q): return " ".join(t for t in toks(q) if t not in STOP) + "|" + ",".join(nums(q))

def safe(a, b):
    if nums(a) != nums(b) or formulas(a) != formulas(b): return False
    ta, tb = set(toks(a)), set(toks(b))
    return all((ta & g) == (tb & g) for g in CONTRAST)

class SemanticCache:
    def __init__(self, embedder, path="cache_store", threshold=0.88, dim=384):
        self.emb, self.path, self.thr, self.lock = embedder, path, threshold, threading.Lock()
        self.index, self.items = faiss.IndexFlatIP(dim), []
        os.makedirs(path, exist_ok=True)
        if os.path.exists(f"{path}/meta.json"):
            self.index = faiss.read_index(f"{path}/cache.faiss")
            self.items = json.load(open(f"{path}/meta.json"))

    def _vec(self, q): return np.array([self.emb.embed_query(q)], dtype="float32")

    def lookup(self, q):
        with self.lock:
            if not self.items: return None
            k = key(q)
            for it in self.items:                      # fast path: same words, same numbers
                if it["key"] == k: return it
            sims, ids = self.index.search(self._vec(q), 5)
            for s, i in zip(sims[0], ids[0]):
                if i >= 0 and s >= self.thr and safe(q, self.items[i]["q"]): return self.items[i]
        return None

    def add(self, q, reply, citations):
        with self.lock:
            self.index.add(self._vec(q))
            self.items.append({"q": q, "key": key(q), "reply": reply, "citations": citations})
            faiss.write_index(self.index, f"{self.path}/cache.faiss")
            json.dump(self.items, open(f"{self.path}/meta.json", "w"))
