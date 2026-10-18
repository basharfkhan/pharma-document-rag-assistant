"""Pharma Supplier Document Assistant: Gradio UI for the full RAG pipeline in rag_core.py."""
import os
import time
from datetime import datetime

import gradio as gr
import pandas as pd

import rag_core as rc

SCOPES = ["Auto (route by document type)", "All documents"] + [t for t in rc.DOC_TYPES if t != "Other"]
BADGE = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}
EMPTY_SOURCES = pd.DataFrame(columns=["#", "Cited", "Document type", "File", "Page", "Similarity"])

# ---------------------------------------------------------------- callbacks
def process(files, progress=gr.Progress()):
    if not files:
        return None, "⚠️ Upload at least one PDF first.", pd.DataFrame()
    paths = [f if isinstance(f, str) else f.name for f in files]
    index, pages, s = rc.ingest(paths, progress=progress)
    library = (pd.DataFrame(pages)
               .groupby("doc_id", sort=False)
               .agg(Type=("doc_type", "first"), Pages=("page_range", "first"), File=("source_file", "first"),
                    ID=("identifier", "first"), OCR=("ocr", "any"))
               .reset_index(drop=True))
    library["OCR"] = library["OCR"].map({True: "scanned (OCR)", False: "digital"})
    status = (f"✅ **{s['files']} file(s) · {s['pages']} pages · {s['documents']} documents · {s['chunks']} chunks** "
              f"({s['ocr_pages']} pages OCR'd) in {s['total_s']:.0f}s" + (" · loaded from cache" if s["cached"] else ""))
    return index, status, library

def ask(question, history, index, scope):
    question = (question or "").strip()
    if not question:
        yield history, "", gr.update(), EMPTY_SOURCES
        return
    history = history + [{"role": "user", "content": question}]
    if index is None:
        history.append({"role": "assistant", "content": "Please upload PDFs and click **Process documents** first."})
        yield history, "", "⚠️ No documents indexed yet.", EMPTY_SOURCES
        return

    history.append({"role": "assistant", "content": "⏳ Finding the right document..."})
    yield history, "", "🔎 Routing and retrieving...", EMPTY_SOURCES
    t0 = time.perf_counter()
    mode = "Auto" if scope.startswith("Auto") else scope
    nodes, route = rc.retrieve(index, question, mode)
    t_retrieve = time.perf_counter() - t0

    sources = pd.DataFrame([{"#": i, "Cited": "", "Document type": n.metadata["doc_type"],
                             "File": n.metadata["source_file"], "Page": n.metadata["page"],
                             "Similarity": f"{n.score:.0%}"} for i, n in enumerate(nodes, 1)])
    history[-1]["content"] = "✍️ Writing the answer..."
    yield history, "", f"✍️ {route['note'].capitalize()}. Writing the answer from {len(nodes)} chunks...", sources

    answer = ""
    for piece in rc.stream_answer(rc.build_prompt(question, nodes)):
        answer += piece
        history[-1]["content"] = answer + " ▌"
        yield history, "", gr.update(), sources

    a = rc.assess(answer, nodes)
    sources["Cited"] = ["✔" if i in a["cited"] else "" for i in sources["#"]]
    cited_refs = " · ".join(f"[{i}] {nodes[i-1].metadata['source_file']} p.{nodes[i-1].metadata['page']}" for i in a["cited"])
    history[-1]["content"] = (answer.strip() + f"\n\n**Confidence: {BADGE[a['level']]}** · {a['chunks']} chunks"
                              + (f" · 📎 {cited_refs}" if cited_refs else ""))
    warn = f" · ⚠️ numbers not found in sources: {', '.join(a['unsupported_numbers'])}" if a["unsupported_numbers"] else ""
    details = (f"**Confidence {BADGE[a['level']]}** · best match {a['top_similarity']:.0%} · "
               f"numbers grounded {a['grounding']:.0%} · {a['chunks']} chunks · {route['note']} · "
               f"retrieval {t_retrieve:.1f}s, total {time.perf_counter() - t0:.0f}s{warn}")
    yield history, "", details, sources

def export_chat(history):
    if not history:
        return None, "Nothing to save yet."
    path = os.path.abspath(f"chat_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Supplier document chat, {datetime.now():%d %B %Y %H:%M}\n\n")
        for m in history:
            f.write(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}\n\n")
    return path, f"💾 Saved {len(history) // 2} question(s)."

# ---------------------------------------------------------------- layout
CSS = """
#hero h1 {margin-bottom: 0}
#hero p {color: #64748b; margin-top: 4px}
#details {min-height: 2.6em; font-size: 0.95em}
"""
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate")

with gr.Blocks(title="Pharma Supplier Document Assistant") as demo:
    index_state = gr.State(None)
    gr.Markdown("# 💊 Pharma Supplier Document Assistant\n"
                "Upload supplier PDFs (digital or scanned). Every answer cites its sources and shows how confident it is.",
                elem_id="hero")
    with gr.Row():
        with gr.Column(scale=4):
            files = gr.File(label="📄 Supplier PDFs", file_types=[".pdf"], file_count="multiple")
            process_btn = gr.Button("⚙️ Process documents", variant="primary")
            status = gr.Markdown("Upload PDFs to begin. Scanned pages are read with OCR.")
            library = gr.Dataframe(label="📚 Document library (detected automatically)", interactive=False, wrap=True)
            scope = gr.Dropdown(SCOPES, value=SCOPES[0], label="🔍 Search scope")
            with gr.Row():
                save_btn = gr.Button("💾 Save chat")
                clear_btn = gr.Button("🗑️ Clear")
            saved = gr.File(label="Chat export", interactive=False, visible=True)
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Assistant", height=470,
                                 placeholder="Ask about storage conditions, lot expiry dates, sterilization doses, audits...")
            with gr.Row():
                question = gr.Textbox(placeholder="Ask a question about your documents...", show_label=False, scale=6)
                send = gr.Button("Ask", variant="primary", scale=1)
            details = gr.Markdown("", elem_id="details")
            sources = gr.Dataframe(EMPTY_SOURCES, label="📎 Sources for the last answer", interactive=False, wrap=True)
            gr.Examples(["What is the recommended storage temperature for the AKTA ready flow kits?",
                         "What gamma dose range did batch D000291847 receive?",
                         "When does the BSE/TSE risk assessment expire?",
                         "Which sub-suppliers provide materials for the West stoppers?"], inputs=question)

    process_btn.click(process, inputs=files, outputs=[index_state, status, library])
    for trigger in (send.click, question.submit):
        trigger(ask, inputs=[question, chatbot, index_state, scope], outputs=[chatbot, question, details, sources])
    save_btn.click(export_chat, inputs=chatbot, outputs=[saved, details])
    clear_btn.click(lambda: ([], "", EMPTY_SOURCES), outputs=[chatbot, details, sources])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
