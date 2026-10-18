"""Evaluate the full RAG pipeline on questions with known answers.

usage: python evaluate.py <digital pdf> <scanned pdf> <digital original of the scanned pdf> <output json>
"""
import json
import re
import sys
import time

import fitz

import rag_core as rc

digital_pdf, scanned_pdf, scanned_original, out_path = sys.argv[1:5]
SAMPLE, SCANNED = "pharma-blob-sample.pdf", scanned_pdf.replace("\\", "/").split("/")[-1]

def pages_of(where):
    return [] if where is None else (where if isinstance(where, list) else [where])

def digits(t):
    return re.sub(r"[-/. ]", "", t.lower())

# question, strings the answer must contain, (file, page) or list of them that hold the answer
TESTS = [
    ("What is the recommended storage temperature for the AKTA ready flow kits?", ["+5"], (SAMPLE, 1)),
    ("When does the BSE/TSE risk assessment for the AKTA flow kits expire?", ["15 March 2028"], (SAMPLE, 6)),
    ("What is the expiration date of lot 15102934?", ["20210712"], (SAMPLE, 3)),
    ("When was Cytiva last audited?", ["20 November 2023"], (SAMPLE, 8)),
    ("What gamma dose range did batch D000291847 receive?", ["27.3", "38.1"], (SCANNED, 10)),
    ("Until when is the West animal origin statement valid?", ["2029"], (SCANNED, 6)),
    # "re:" = regular expression; "not:" = must NOT appear (stricter checks after reading the first run's answers)
    ("Which sub-suppliers provide materials for the West stoppers?",
     [r"re:ExxonMobil[^.;]{0,40}chlorobutyl", r"re:Daikin[^.;]{0,40}PTFE"], (SCANNED, 9)),
    ("What change was made under ECN-2024-0027?", ["28", "32", "ECT", "not:minute"], (SCANNED, 5)),
    ("What is the Shore A hardness specification of the West stopper?", ["50"], [(SCANNED, 4), (SCANNED, 7)]),
    ("What is the capital of France?", [rc.NOT_FOUND], None),
]

t0 = time.perf_counter()
index, pages, stats = rc.ingest([digital_pdf, scanned_pdf])
print("ingest:", json.dumps(stats), flush=True)

# OCR quality: word recall of the OCR'd pages against the digital original
def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())
with fitz.open(scanned_original) as pdf:
    truth = [words(p.get_text()) for p in pdf]
ocr_pages = [p for p in pages if p["ocr"]]
recalls = []
for p, t in zip(ocr_pages, truth):
    got = set(words(p["text"]))
    recalls.append(sum(w in got for w in t) / len(t))
ocr_recall = sum(recalls) / len(recalls)
print(f"OCR word recall: {ocr_recall:.1%} (min page {min(recalls):.1%})", flush=True)

# retrieval metrics: auto-routing vs searching everything (no generation needed)
# a retrieved chunk is relevant if it comes from a page that holds the answer
retrieval = []
for q, _, where in TESTS:
    if where is None:
        continue
    row = {"question": q}
    relevant_pages = pages_of(where)
    for mode in ["All documents", "Auto"]:
        t1 = time.perf_counter()
        nodes, route = rc.retrieve(index, q, mode)   # Auto includes the (uncached) routing LLM call
        latency = time.perf_counter() - t1
        found = [(n.metadata["source_file"], n.metadata["page"]) for n in nodes]
        ranks = [r for r, p in enumerate(found, 1) if p in relevant_pages]
        row[mode] = bool(ranks)
        row[mode + " metrics"] = {
            "recall": len(set(found) & set(relevant_pages)) / len(relevant_pages),
            "precision": len(ranks) / len(nodes),
            "rr": 1 / ranks[0] if ranks else 0.0,
            "latency_s": round(latency, 2)}
        if mode == "Auto":
            row["route"] = route["note"]
    retrieval.append(row)
    print("retrieval:", row, flush=True)

def mean(xs):
    return sum(xs) / len(xs)
retrieval_summary = {mode: {m: round(mean([r[mode + " metrics"][m] for r in retrieval]), 3)
                            for m in ["recall", "precision", "rr", "latency_s"]} | {"hit_rate": mean([r[mode] for r in retrieval])}
                     for mode in ["All documents", "Auto"]}
print("retrieval summary:", retrieval_summary, flush=True)

# full answers
results = []
for q, must, where in TESTS:
    t1 = time.perf_counter()
    nodes, route = rc.retrieve(index, q, "Auto")   # routing answer is cached from the retrieval pass above
    t_retrieve = time.perf_counter() - t1
    t2 = time.perf_counter()
    answer = "".join(rc.stream_answer(rc.build_prompt(q, nodes))).strip()
    t_generate = time.perf_counter() - t2
    t_total = time.perf_counter() - t1
    a = rc.assess(answer, nodes)
    def check(m):
        if m.startswith("re:"):
            return bool(re.search(m[3:], answer, re.I))
        if m.startswith("not:"):
            return m[4:].lower() not in answer.lower()
        return m.lower() in answer.lower() or digits(m) in digits(answer)
    correct = all(check(m) for m in must)
    cited_right = None if where is None else any(
        (nodes[c - 1].metadata["source_file"], nodes[c - 1].metadata["page"]) in pages_of(where) for c in a["cited"])
    results.append({"question": q, "answer": answer, "correct": correct, "cited_right_page": cited_right,
                    "route": route["note"], **a, "retrieve_s": round(t_retrieve, 2), "generate_s": round(t_generate, 1),
                    "total_s": round(t_total, 1),
                    "sources": [f"{n.metadata['source_file']} p{n.metadata['page']} ({n.score:.2f})" for n in nodes]})
    print(json.dumps(results[-1]), flush=True)

json.dump({"ingest": stats, "ocr_word_recall": ocr_recall, "ocr_page_recalls": recalls,
           "retrieval": retrieval, "retrieval_summary": retrieval_summary, "answers": results},
          open(out_path, "w"), indent=2)
print("done in", round(time.perf_counter() - t0), "s")
