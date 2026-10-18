"""Pharma Document Q&A Chatbot: ask questions about uploaded PDFs, answered from the documents with page citations.

Pipeline: PyMuPDF (one Document per page) -> LlamaIndex + BGE-small embeddings -> top-4 chunks
-> Mistral-7B-Instruct (local, llama.cpp) or Gemini (if GOOGLE_API_KEY is set) -> streamed answer + sources.
"""
import os
import time
from datetime import datetime

import fitz  # PyMuPDF
import gradio as gr
import llama_cpp
from llama_cpp import Llama
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MODEL_PATH = os.environ.get("MISTRAL_GGUF", "/content/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")
TOP_K = 4
ALL_FILES = "All uploaded files"

# ---------- Models (loaded once) ----------
Settings.llm = None  # retrieval only; answers are generated below
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    query_instruction="Represent this sentence for searching relevant passages: ")

_mistral = None

def mistral():
    # Loaded on first use, so the UI starts instantly
    global _mistral
    if _mistral is None:
        if not os.path.exists(MODEL_PATH):
            from huggingface_hub import hf_hub_download
            hf_hub_download("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                            local_dir=os.path.dirname(MODEL_PATH))
        gpu_layers = -1 if llama_cpp.llama_supports_gpu_offload() else 0
        _mistral = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=gpu_layers, seed=42, verbose=False)
    return _mistral

MODELS = ["Mistral-7B (local, no API key)"] + (["Gemini (API)"] if GEMINI_KEY else [])

STYLES = {
    "Concise": "Answer in one or two sentences.",
    "Detailed": "Answer thoroughly, including every relevant value from the context.",
    "Bullet points": "Answer as a short bulleted list.",
}

# ---------- Document processing ----------
def process_pdfs(files):
    """Read every uploaded PDF page by page and build a searchable index. Returns (index, status, file choices)."""
    if not files:
        return None, "⚠️ Please upload at least one PDF first.", gr.update(choices=[ALL_FILES], value=ALL_FILES)
    t0 = time.perf_counter()
    docs, names = [], []
    for f in files:
        path = f if isinstance(f, str) else f.name
        name = os.path.basename(path)
        names.append(name)
        with fitz.open(path) as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    # metadata travels with every chunk, so answers can cite file and page
                    docs.append(Document(text=text, metadata={"file_name": name, "page": i + 1},
                                         excluded_embed_metadata_keys=["file_name", "page"]))
    nodes = SentenceSplitter(chunk_size=256, chunk_overlap=30).get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)
    status = (f"✅ Indexed **{len(names)} file(s)**, {len(docs)} pages, {len(nodes)} chunks "
              f"in {time.perf_counter() - t0:.1f}s. Ask away!")
    return index, status, gr.update(choices=[ALL_FILES] + names, value=ALL_FILES)

# ---------- Question answering ----------
def build_prompt(question, nodes, style):
    context = "\n\n".join(f"[{n.metadata['file_name']}, page {n.metadata['page']}]\n{n.get_content()}" for n in nodes)
    return (f"You answer questions about pharmaceutical supplier documents using ONLY the context below.\n"
            f"If the answer is not in the context, say you could not find it in the uploaded documents.\n"
            f"{STYLES[style]}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}")

def stream_answer(prompt, model):
    """Yield the answer text as it is generated."""
    if model.startswith("Gemini"):
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        for chunk in client.models.generate_content_stream(model="gemini-3.8-flash", contents=prompt):
            if chunk.text:
                yield chunk.text
    else:
        for chunk in mistral().create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                                      temperature=0.1, max_tokens=400, stream=True):
            yield chunk["choices"][0]["delta"].get("content", "")

def chat(question, history, index, file_choice, style, model):
    """Generator: shows a status message, streams the answer, then appends the sources."""
    question = (question or "").strip()
    if not question:
        yield history, "", "Type a question first."
        return
    history = history + [{"role": "user", "content": question}]
    if index is None:
        history.append({"role": "assistant", "content": "Please upload a PDF and click **Process documents** first."})
        yield history, "", "⚠️ No documents indexed yet."
        return

    history.append({"role": "assistant", "content": "⏳ Searching your documents..."})
    yield history, "", "🔎 Searching..."

    # restrict the search to one file if the user picked one in the dropdown
    filters = None if file_choice == ALL_FILES else MetadataFilters(filters=[MetadataFilter(key="file_name", value=file_choice)])
    nodes = index.as_retriever(similarity_top_k=TOP_K, filters=filters).retrieve(question)

    yield history, "", f"✍️ Writing the answer with {model.split(' (')[0]}..."
    t0 = time.perf_counter()
    answer = ""
    for piece in stream_answer(build_prompt(question, nodes, style), model):
        answer += piece
        history[-1]["content"] = answer + " ▌"
        yield history, "", f"✍️ Writing the answer with {model.split(' (')[0]}..."

    # cite where the context came from, in retrieval order, without duplicates
    sources = list(dict.fromkeys(f"{n.metadata['file_name']} p.{n.metadata['page']}" for n in nodes))
    history[-1]["content"] = answer.strip() + "\n\n📎 *Sources: " + " · ".join(sources) + "*"
    yield history, "", f"✅ Answered in {time.perf_counter() - t0:.0f}s from {len(nodes)} passages."

def export_chat(history):
    """Save the conversation to a Markdown file the user can download."""
    if not history:
        return None, "Nothing to save yet."
    path = os.path.abspath(f"chat_history_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Document Q&A chat, {datetime.now():%d %B %Y %H:%M}\n\n")
        for m in history:
            f.write(f"**{'You' if m['role'] == 'user' else 'Bot'}:** {m['content']}\n\n")
    return path, f"💾 Saved {len(history) // 2} question(s) to {os.path.basename(path)}"

# ---------- UI ----------
CSS = """
#title {text-align: center; margin-bottom: 0}
#subtitle {text-align: center; color: #64748b; margin-top: 0}
#status {min-height: 2.5em}
"""

with gr.Blocks(title="Pharma Document Q&A") as demo:
    index_state = gr.State(None)  # the vector index for this user's uploads

    gr.Markdown("# 💊 Pharma Document Q&A", elem_id="title")
    gr.Markdown("Upload supplier PDFs, then ask questions. Every answer cites the file and page it came from.",
                elem_id="subtitle")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📄 Upload PDFs", file_types=[".pdf"], file_count="multiple")
            process_btn = gr.Button("🔄 Process documents", variant="primary")
            file_choice = gr.Dropdown([ALL_FILES], value=ALL_FILES, label="🔍 Search in")
            style = gr.Dropdown(list(STYLES), value="Concise", label="📝 Answer style")
            model = gr.Dropdown(MODELS, value=MODELS[0], label="🤖 Answer model")
            with gr.Row():
                save_btn = gr.Button("💾 Save chat")
                clear_btn = gr.Button("🗑️ Clear chat")
            saved_file = gr.File(label="Chat history file", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=520,
                                 placeholder="Upload a PDF and ask something like *What sterilization method was used?*")
            with gr.Row():
                user_input = gr.Textbox(placeholder="Ask a question about your documents...", show_label=False, scale=5)
                send_btn = gr.Button("📤 Send", variant="primary", scale=1)
            status = gr.Markdown("Upload a PDF to begin.", elem_id="status")
            gr.Examples(["What sterilization method was used for this product?",
                         "What are the storage conditions for the flow kits?",
                         "When does the BSE/TSE risk assessment expire?",
                         "Which sub-suppliers are approved?"], inputs=user_input)

    # wire the UI to the functions
    process_btn.click(process_pdfs, inputs=pdf_input, outputs=[index_state, status, file_choice])
    chat_inputs = [user_input, chatbot, index_state, file_choice, style, model]
    send_btn.click(chat, inputs=chat_inputs, outputs=[chatbot, user_input, status])
    user_input.submit(chat, inputs=chat_inputs, outputs=[chatbot, user_input, status])
    save_btn.click(export_chat, inputs=chatbot, outputs=[saved_file, status])
    clear_btn.click(lambda: ([], "Chat cleared."), outputs=[chatbot, status])

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="sky")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
