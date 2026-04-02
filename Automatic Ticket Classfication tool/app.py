import gradio as gr
from dotenv import load_dotenv
import os
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────

ENV_PATH  = "/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Automatic Ticket Classfication tool/.env"
CSV_PATH  = "/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Automatic Ticket Classfication tool/customer_support_tickets.csv"
DOCS_PATH = "/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Automatic Ticket Classfication tool/Documents"
MODEL_PATH = "modelsvm.pk1"

load_dotenv(ENV_PATH)

from user_utils import create_embeddings, pull_from_pinecone, get_similar_docs, get_answer, predict
from admin_utils import (
    read_pdf_data, split_data, create_embeddings_load_data,
    push_to_pinecone, read_data, get_embeddings,
    create_embeddings_for_df, split_train_test__data, get_score
)

# ── Tab 1: Ask a Question ─────────────────────────────────────────────────────

def handle_query(user_input):
    if not user_input.strip():
        return "⚠️ Please enter a question.", ""
    embeddings = create_embeddings()
    index = pull_from_pinecone(os.getenv("PINECONE_API_KEY"), "gcp-starter", "tickets", embeddings)
    relevant_docs = get_similar_docs(index, user_input)
    response = get_answer(relevant_docs, user_input)
    return response, user_input


def submit_ticket(user_input, hr_tickets, it_tickets, transport_tickets):
    if not user_input.strip():
        return "⚠️ Please enter a complaint first.", hr_tickets, it_tickets, transport_tickets
    embeddings = create_embeddings()
    query_result = embeddings.embed_query(user_input)
    department = predict(query_result)
    if department == "HR":
        hr_tickets = hr_tickets + [user_input]
    elif department == "IT":
        it_tickets = it_tickets + [user_input]
    else:
        transport_tickets = transport_tickets + [user_input]
    return f"✅ Ticket submitted to: **{department}**", hr_tickets, it_tickets, transport_tickets

# ── Tab 2: Pending Tickets ────────────────────────────────────────────────────

def show_tickets(hr, it, transport):
    def fmt(label, tickets):
        if not tickets:
            return f"**{label}** — No tickets yet.\n\n"
        lines = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tickets)])
        return f"**{label} Tickets:**\n{lines}\n\n"
    return fmt("HR Support", hr) + "---\n\n" + fmt("IT Support", it) + "---\n\n" + fmt("Transportation Support", transport)

# ── Tab 3: Load Knowledge Base ────────────────────────────────────────────────

def load_documents_to_pinecone():
    if not os.path.exists(DOCS_PATH):
        return f"⚠️ Documents folder not found at:\n{DOCS_PATH}"
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        return f"⚠️ No PDF files found in:\n{DOCS_PATH}"
    embeddings = create_embeddings_load_data()
    log = []
    for pdf_file in pdf_files:
        full_path = os.path.join(DOCS_PATH, pdf_file)
        text = read_pdf_data(full_path)
        docs_chunks = split_data(text)
        push_to_pinecone(os.getenv("PINECONE_API_KEY"), "gcp-starter", "tickets", embeddings, docs_chunks)
        log.append(f"✅ Pushed: {pdf_file}")
    return "\n".join(log) + "\n\n✅ All documents pushed to Pinecone successfully!"

# ── Tab 4: Train ML Model ─────────────────────────────────────────────────────

def load_csv_and_embed():
    if not os.path.exists(CSV_PATH):
        return f"⚠️ CSV not found at:\n{CSV_PATH}", None
    df = read_data(CSV_PATH)
    embeddings = get_embeddings()
    cleaned = create_embeddings_for_df(df, embeddings)
    return f"✅ Loaded {len(cleaned)} rows and created embeddings.", cleaned


def train_model(cleaned_data):
    if cleaned_data is None:
        return "⚠️ Please load data first.", None, None, None, None, None
    s_train, s_test, l_train, l_test = split_train_test__data(cleaned_data)
    clf = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    clf.fit(s_train, l_train)
    return "✅ Model trained successfully!", clf, s_train, s_test, l_train, l_test


def evaluate_model(clf, s_test, l_test):
    if clf is None:
        return "⚠️ Please train the model first."
    score = get_score(clf, s_test, l_test)
    sample_text = "Rude driver with scary driving"
    emb = get_embeddings()
    query_result = emb.embed_query(sample_text)
    prediction = clf.predict([query_result])
    return (
        f"✅ Validation accuracy: **{100 * score:.2f}%**\n\n"
        f"**Sample complaint:** {sample_text}\n\n"
        f"**Predicted department:** {prediction[0]}"
    )


def save_model(clf):
    if clf is None:
        return "⚠️ Please train the model first."
    joblib.dump(clf, MODEL_PATH)
    return f"✅ Model saved as `{MODEL_PATH}`"

# ── Build App ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Automatic Ticket Classification Tool") as demo:

    # Shared state
    hr_state         = gr.State([])
    it_state         = gr.State([])
    transport_state  = gr.State([])
    cleaned_state    = gr.State(None)
    clf_state        = gr.State(None)
    s_train_state    = gr.State(None)
    s_test_state     = gr.State(None)
    l_train_state    = gr.State(None)
    l_test_state     = gr.State(None)
    last_input_state = gr.State("")

    gr.Markdown("# 🎫 Automatic Ticket Classification Tool")
    gr.Markdown("We are here to help you. Describe your issue and get an instant answer.")

    with gr.Tabs():

        # ── Tab 1 ─────────────────────────────────────────────────────────
        with gr.Tab("🔍 Ask a Question"):
            user_input = gr.Textbox(
                label="Your complaint or question",
                placeholder="e.g. My laptop is not turning on...",
                lines=3
            )
            with gr.Row():
                ask_btn    = gr.Button("Get Answer", variant="primary")
                submit_btn = gr.Button("Submit Ticket", variant="secondary")

            answer_output = gr.Markdown(label="Answer")
            ticket_status = gr.Markdown()

            ask_btn.click(
                fn=handle_query,
                inputs=user_input,
                outputs=[answer_output, last_input_state]
            )
            submit_btn.click(
                fn=submit_ticket,
                inputs=[user_input, hr_state, it_state, transport_state],
                outputs=[ticket_status, hr_state, it_state, transport_state]
            )

        # ── Tab 2 ─────────────────────────────────────────────────────────
        with gr.Tab("📋 Pending Tickets"):
            refresh_btn     = gr.Button("Refresh Tickets")
            tickets_display = gr.Markdown()
            refresh_btn.click(
                fn=show_tickets,
                inputs=[hr_state, it_state, transport_state],
                outputs=tickets_display
            )

        # ── Tab 3 ─────────────────────────────────────────────────────────
        with gr.Tab("📁 Load Knowledge Base"):
            gr.Markdown(f"Loads all PDFs from:\n\n`{DOCS_PATH}`")
            load_docs_btn  = gr.Button("Push PDFs to Pinecone", variant="primary")
            load_docs_status = gr.Markdown()
            load_docs_btn.click(
                fn=load_documents_to_pinecone,
                inputs=[],
                outputs=load_docs_status
            )

        # ── Tab 4 ─────────────────────────────────────────────────────────
        with gr.Tab("🤖 Train ML Model"):
            gr.Markdown(f"Using dataset:\n\n`{CSV_PATH}`")

            with gr.Tabs():

                with gr.Tab("1 · Load Data"):
                    load_data_btn  = gr.Button("Load CSV & Create Embeddings", variant="primary")
                    load_data_status = gr.Markdown()
                    load_data_btn.click(
                        fn=load_csv_and_embed,
                        inputs=[],
                        outputs=[load_data_status, cleaned_state]
                    )

                with gr.Tab("2 · Train"):
                    train_btn    = gr.Button("Train SVM Classifier", variant="primary")
                    train_status = gr.Markdown()
                    train_btn.click(
                        fn=train_model,
                        inputs=cleaned_state,
                        outputs=[train_status, clf_state, s_train_state, s_test_state, l_train_state, l_test_state]
                    )

                with gr.Tab("3 · Evaluate"):
                    eval_btn    = gr.Button("Evaluate Model", variant="primary")
                    eval_output = gr.Markdown()
                    eval_btn.click(
                        fn=evaluate_model,
                        inputs=[clf_state, s_test_state, l_test_state],
                        outputs=eval_output
                    )

                with gr.Tab("4 · Save"):
                    save_btn    = gr.Button("Save Model", variant="primary")
                    save_status = gr.Markdown()
                    save_btn.click(
                        fn=save_model,
                        inputs=clf_state,
                        outputs=save_status
                    )

if __name__ == "__main__":
    demo.launch()