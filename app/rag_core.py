"""Core RAG pipeline for supplier PDFs (digital or scanned).

ingest:  PDF -> page text (PyMuPDF, or EasyOCR for scanned pages) -> page classification + document
         segmentation (Mistral, constrained JSON) -> chunks with metadata -> BGE-small vector index
answer:  question -> route to a document type (Mistral) -> filtered retrieval (with fallbacks)
         -> numbered-source prompt -> Mistral answer with [n] citations -> sources + confidence
"""
import json
import os
import re
import time
from functools import lru_cache

import fitz  # PyMuPDF
import llama_cpp
from llama_cpp import Llama
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MODEL_PATH = os.environ.get("MISTRAL_GGUF", os.path.join(
    "/content/models" if os.path.isdir("/content") else "models",   # Colab, or a local models/ folder
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"))
TOP_K = 4
MIN_TEXT_CHARS = 50      # a page with less extractable text than this is treated as scanned
ROUTE_MIN_SCORE = 0.45   # if the best routed chunk scores below this, search everything instead

DOC_TYPES = ["Cover Letter", "Certificate of Quality", "Packaging Specification", "BSE/TSE Declaration",
             "Material Description", "Supplier Qualification", "Chain of Custody", "Other"]
TYPE_DEFINITIONS = """- Cover Letter: a letter or email addressed to the reader, e.g. about storage, shipping or what a package contains
- Certificate of Quality: certifies one product lot or batch (lot/batch numbers, manufacture/expiry dates, release tests), including third-party processing certificates
- Packaging Specification: packaging materials, configuration, packaging part numbers, drawings, packaging change history
- BSE/TSE Declaration: statement that products contain no animal-origin materials (BSE/TSE), animal origin statements
- Material Description: materials of construction, composition and physical properties of a product
- Supplier Qualification: supplier or vendor approval status, audits, certifications, sub-suppliers, quality agreement
- Chain of Custody: where assemblies are manufactured and how they move to the customer
- Other: none of the above"""

# ---------------------------------------------------------------- models (loaded once, lazily)
Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    query_instruction="Represent this sentence for searching relevant passages: ")

_llm, _ocr = None, None

def llm():
    global _llm
    if _llm is None:
        if not os.path.exists(MODEL_PATH):
            from huggingface_hub import hf_hub_download
            hf_hub_download("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                            local_dir=os.path.dirname(MODEL_PATH))
        gpu = -1 if llama_cpp.llama_supports_gpu_offload() else 0
        _llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=gpu, seed=42, verbose=False)
    return _llm

def ocr_reader():
    global _ocr
    if _ocr is None:
        import easyocr
        import torch
        _ocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _ocr

@lru_cache(maxsize=None)
def ask_json(prompt, schema_json):
    # temperature 0 + JSON schema grammar: the answer is always valid JSON with allowed values only
    r = llm().create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                     response_format={"type": "json_object", "schema": json.loads(schema_json)},
                                     temperature=0.0, max_tokens=80)
    return json.loads(r["choices"][0]["message"]["content"])

# ---------------------------------------------------------------- 1. text extraction (+ OCR)
def clean_ocr(text):
    """Fix the commonest OCR confusion in identifiers: letters O/o, l/I inside number-like tokens."""
    swap = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})
    def fix(m):
        tok = m.group(0)
        if sum(c.isdigit() for c in tok) / len(tok) >= 0.6:
            # keep a genuine letter prefix (the D in D000291847); fix confusable characters everywhere else
            keep = 1 if tok[0].isalpha() and tok[0] not in "OoIl" else 0
            return tok[:keep] + tok[keep:].translate(swap)
        return tok
    return re.sub(r"\b[A-Za-z0-9]{6,}\b", fix, text)

def deskew(gray):
    """Straighten a slightly rotated scan (rotation makes table rows drift across the page)."""
    import cv2
    import numpy as np
    ink = np.column_stack(np.where(gray < 128))[:, ::-1].astype(np.float32)   # (x, y) of dark pixels
    if len(ink) < 100:
        return gray
    angle = cv2.minAreaRect(ink)[-1]
    # OpenCV versions report this angle in [-90, 0) or (0, 90]; bring it into [-45, 45]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.1 or abs(angle) > 10:               # already straight, or not a simple skew
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

def ocr_text(page):
    """OCR a page and rebuild table rows: words whose boxes sit at the same height are joined into one line.
    Without this, every table cell comes out on its own line and a batch number loses its values."""
    import numpy as np
    pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
    gray = deskew(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width))
    results = ocr_reader().readtext(gray, detail=1)
    boxes = sorted(((sum(pt[1] for pt in b) / 4, min(pt[0] for pt in b), max(pt[1] for pt in b) - min(pt[1] for pt in b), t)
                    for b, t, _ in results), key=lambda x: x[0])
    rows, current = [], []
    for y, x, h, t in boxes:
        if current and abs(y - current[-1][0]) > 0.6 * max(h, current[-1][2]):
            rows.append(current)
            current = []
        current.append((y, x, h, t))
    if current:
        rows.append(current)
    return "\n".join(" | ".join(t for _, _, _, t in sorted(r, key=lambda c: c[1])) for r in rows)

def extract_pages(path):
    """One record per page; scanned pages (no text layer) are OCR'd."""
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            text, ocr = page.get_text(), False
            if len(text.strip()) < MIN_TEXT_CHARS:
                text = clean_ocr(ocr_text(page))
                ocr = True
            pages.append({"source_file": os.path.basename(path), "page": i + 1, "text": text, "ocr": ocr})
    return pages

# ---------------------------------------------------------------- 2. classification + segmentation
PAGE_SCHEMA = json.dumps({"type": "object", "properties": {"doc_type": {"type": "string", "enum": DOC_TYPES},
                          "identifier": {"type": "string"}}, "required": ["doc_type", "identifier"]})

def describe_page(text):
    return ask_json(f"""Read this page from a bundle of pharmaceutical supplier documents.

Document types:
{TYPE_DEFINITIONS}

PAGE:
{text[:3000]}

Return JSON with:
"doc_type": the document type from the list above
"identifier": the document number, lot number or record code printed on this page that identifies this specific document. Copy it exactly as printed. Use "" if the page has none; do not make one up.""", PAGE_SCHEMA)

def verified_identifier(model_identifier, page_text):
    # keep only an ID-like token that really appears on the page (never a date)
    page = re.sub(r"\s+", " ", page_text)
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", model_identifier):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
            continue
        for cand in [token] + token.split("-"):
            if len(cand) >= 5 and re.search(r"\d", cand) and cand in page:
                return cand
    return ""

def continues_previous(text):
    head = text[:400].lower()
    m = re.search(r"page\s+(\d+)\s*(?:of|/)\s*(\d+)", head)
    return "continued" in head or bool(m and int(m.group(1)) > 1)

def segment(pages):
    """Assign doc_type and doc_id to every page (rules from the Project 7 segmentation pipeline)."""
    prev, doc_n = None, 0
    for p in pages:
        info = describe_page(p["text"])
        ident = verified_identifier(info["identifier"], p["text"])
        cont = continues_previous(p["text"])
        if prev is None or prev["source_file"] != p["source_file"]:
            new = True
        elif cont:
            new = False
        elif info["doc_type"] != prev["doc_type"]:
            new = not (len(p["text"].split()) < 30 and not ident)   # short page without ID: keep with current doc
        else:
            new = bool(ident and prev["identifier"] and ident != prev["identifier"])
        if new:
            doc_n += 1
            p["doc_type"] = info["doc_type"]
        else:
            p["doc_type"] = prev["doc_type"]
        p["doc_id"] = f"{p['source_file']}#{doc_n}"
        p["identifier"] = ident or (prev["identifier"] if prev and not new else "")
        prev = p
    # page range of each document, stored on every page of it
    ranges = {}
    for p in pages:
        ranges.setdefault(p["doc_id"], []).append(p["page"])
    for p in pages:
        r = ranges[p["doc_id"]]
        p["page_range"] = f"{r[0]}" if len(r) == 1 else f"{r[0]}-{r[-1]}"
    return pages

# ---------------------------------------------------------------- 3. chunking + index
def build_index(pages):
    docs = []
    for p in pages:
        meta = {k: p[k] for k in ["source_file", "page", "page_range", "doc_id", "doc_type", "identifier", "ocr"]}
        docs.append(Document(text=p["text"], metadata=meta,
                             excluded_embed_metadata_keys=[k for k in meta if k != "doc_type"]))
    # split at line breaks, so a table row (one line) is never cut in half between two chunks
    nodes = SentenceSplitter(chunk_size=256, chunk_overlap=30, paragraph_separator="\n").get_nodes_from_documents(docs)
    for i, n in enumerate(nodes):
        n.metadata["chunk_index"] = i
        n.excluded_embed_metadata_keys = n.excluded_embed_metadata_keys + ["chunk_index"]
    return VectorStoreIndex(nodes), len(nodes)

CACHE_DIR = os.environ.get("RAG_CACHE", ".rag_cache")

def cache_key(paths):
    # same files in the same order -> same key, whatever their names or upload location
    import hashlib
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:16]

def ingest(paths, progress=None):
    """Full ingestion. Returns (index, pages, stats).
    The slow steps (OCR and page classification) are cached on disk, so the same files load in seconds next time."""
    t0 = time.perf_counter()
    paths = sorted(paths, key=os.path.basename)   # upload order doesn't matter
    cache_file = os.path.join(CACHE_DIR, cache_key(paths) + ".json")
    if os.path.exists(cache_file):
        if progress: progress(0.5, desc="Loading processed pages from cache")
        with open(cache_file, encoding="utf-8") as f:
            pages = json.load(f)
        t_extract = t_segment = 0.0
        cached = True
    else:
        pages, cached = [], False
        for i, path in enumerate(paths):
            if progress: progress(i / max(len(paths), 1) * 0.5, desc=f"Reading {os.path.basename(path)}")
            pages += extract_pages(path)
        t_extract = time.perf_counter() - t0
        if progress: progress(0.5, desc="Classifying pages and finding document boundaries")
        segment(pages)
        t_segment = time.perf_counter() - t0 - t_extract
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(pages, f)
    if progress: progress(0.9, desc="Embedding and indexing")
    index, n_chunks = build_index(pages)
    stats = {"files": len(paths), "pages": len(pages), "ocr_pages": sum(p["ocr"] for p in pages),
             "documents": len({p["doc_id"] for p in pages}), "chunks": n_chunks,
             "extract_s": round(t_extract, 1), "segment_s": round(t_segment, 1),
             "total_s": round(time.perf_counter() - t0, 1), "cached": cached}
    return index, pages, stats

# ---------------------------------------------------------------- 4. routing + retrieval
QUERY_SCHEMA = json.dumps({"type": "object", "properties": {"doc_type": {"type": "string", "enum": DOC_TYPES}},
                           "required": ["doc_type"]})

def route_query(question):
    t = ask_json(f"""Which type of document in a pharmaceutical supplier bundle is most likely to answer this question?

Document types:
{TYPE_DEFINITIONS}

QUESTION: {question}

Return JSON with "doc_type". Use "Other" if the question does not point to one type.""", QUERY_SCHEMA)["doc_type"]
    return None if t == "Other" else t

def identifiers_in(question):
    return re.findall(r"\b(?:[A-Z]{1,}-?[A-Z0-9-]*\d[A-Z0-9-]*|\d{6,})\b", question)

def exact_matches(index, question, ids):
    """Chunks containing an ID from the question, ranked by their embedding similarity to the question."""
    from llama_index.core.schema import NodeWithScore
    hits = [n for n in index.docstore.docs.values() if any(i in n.get_content() for i in ids)]
    if not hits:
        return []
    q = Settings.embed_model.get_query_embedding(question)
    def sim(n):
        e = Settings.embed_model.get_text_embedding(n.get_content(metadata_mode="embed"))
        return sum(a * b for a, b in zip(q, e)) / ((sum(a * a for a in q) ** 0.5) * (sum(b * b for b in e) ** 0.5))
    return sorted((NodeWithScore(node=n, score=sim(n)) for n in hits), key=lambda x: -x.score)[:TOP_K]

def retrieve(index, question, mode="Auto"):
    """mode: 'Auto' (route by document type), 'All documents', or a specific document type.
    Returns (nodes, route_info)."""
    search = lambda dt: index.as_retriever(similarity_top_k=TOP_K, filters=None if dt is None else
                                           MetadataFilters(filters=[MetadataFilter(key="doc_type", value=dt)])
                                           ).retrieve(question)
    if mode == "All documents":
        return search(None), {"doc_type": None, "note": "searched all documents"}
    routed = route_query(question) if mode == "Auto" else mode
    if routed is None:
        return search(None), {"doc_type": None, "note": "question not specific to one document type; searched all"}
    nodes = search(routed)
    text = " ".join(n.get_content() for n in nodes)
    ids = [i for i in identifiers_in(question) if len(i) >= 5]
    if mode == "Auto" and ids and not any(i in text for i in ids):
        # exact-match lookup: chunks that literally contain the ID come first, then the best unfiltered matches
        exact = exact_matches(index, question, ids)
        rest = [n for n in search(None) if n.node.node_id not in {e.node.node_id for e in exact}]
        note = f"routed to {routed}, but {ids[0]} was not there; " + (
            f"found it by exact match" if exact else "searched all")
        return (exact + rest)[:TOP_K], {"doc_type": routed, "note": note}
    if mode == "Auto" and (not nodes or nodes[0].score < ROUTE_MIN_SCORE):
        return search(None), {"doc_type": routed, "note": f"weak match in {routed}; searched all"}
    return nodes, {"doc_type": routed, "note": f"routed to {routed}"}

# ---------------------------------------------------------------- 5. prompt + answer
def build_prompt(question, nodes):
    sources = "\n\n".join(
        f"[{i}] {n.metadata['doc_type']}, {n.metadata['source_file']}, page {n.metadata['page']}\n{n.get_content()}"
        for i, n in enumerate(nodes, 1))
    return f"""You are a pharmaceutical quality assistant. Answer the question using ONLY the numbered sources below.

Rules:
- After every fact, cite the source number in square brackets, e.g. [1] or [2].
- Copy numbers, dates, lot numbers and units exactly as written in the sources.
- Tables are written one row per line with cells separated by "|"; the first row of a table names the columns. Read a value from the same row as the item it belongs to.
- If the sources do not contain the answer, reply exactly: "I could not find this in the uploaded documents."
- Answer only the question asked. Do not comment on sources that are not relevant, and do not add further questions.
- Be concise.

SOURCES:
{sources}

QUESTION: {question}
ANSWER:"""

def stream_answer(prompt):
    for chunk in llm().create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                              temperature=0.0, max_tokens=350, stream=True,
                                              stop=["QUESTION:", "\nQ:", "SOURCES:"]):
        yield chunk["choices"][0]["delta"].get("content", "")

NOT_FOUND = "could not find this in the uploaded documents"

DATE = r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"

def numbers_in(text):
    text = re.sub(r"\[\d+\]", " ", text)
    # a reformatted date (2021-07-12 for 20210712) counts as its digits joined, not as three numbers
    dates = {re.sub(r"\D", "", d) for d in re.findall(DATE, text)}
    return set(re.findall(r"\d+(?:\.\d+)?", re.sub(DATE, " ", text))) | dates

def assess(answer, nodes):
    """Confidence from two checks that can be computed, not guessed:
    retrieval similarity of the best chunk, and whether every number in the answer appears in the sources."""
    cited = sorted({int(c) for c in re.findall(r"\[(\d+)\]", answer) if 1 <= int(c) <= len(nodes)})
    source_text = " ".join(n.get_content() for n in nodes)
    src_numbers = numbers_in(source_text) | numbers_in(source_text.replace(" ", "")) | set(re.findall(r"\d+", source_text))
    ans_numbers = numbers_in(answer)
    unsupported = sorted(ans_numbers - src_numbers)
    grounding = 1.0 if not ans_numbers else 1 - len(unsupported) / len(ans_numbers)
    top = nodes[0].score if nodes else 0.0
    if "could not find" in answer.lower():
        level = "Low"   # a full or partial "not found" answer
    elif top >= 0.60 and grounding == 1.0 and cited:
        level = "High"
    elif top >= 0.50 and grounding >= 0.8:
        level = "Medium"
    else:
        level = "Low"
    return {"level": level, "top_similarity": round(top, 3), "grounding": round(grounding, 2),
            "unsupported_numbers": unsupported, "cited": cited, "chunks": len(nodes)}
