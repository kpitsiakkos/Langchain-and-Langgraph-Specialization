import gradio as gr
from dotenv import load_dotenv
import os
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────

ENV_PATH   = "/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Automatic Ticket Classfication tool/.env"
CSV_PATH   = "/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Automatic Ticket Classfication tool/Tickets.csv"
DOCS_PATH  = "/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Automatic Ticket Classfication tool/Documents"
MODEL_PATH = "modelsvm.pk1"

load_dotenv(ENV_PATH)

from user_utils import create_embeddings, pull_from_pinecone, get_similar_docs, get_answer, predict
from admin_utils import (
    read_pdf_data, split_data, create_embeddings_load_data,
    push_to_pinecone, read_data, get_embeddings,
    create_embeddings_for_df, split_train_test__data, get_score
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

.gradio-container {
    background: #f0f4ff !important;
    min-height: 100vh;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    padding: 36px 44px 28px;
    margin-bottom: 0;
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.5px;
}
.app-header p {
    color: rgba(255,255,255,0.8) !important;
    font-size: 14px !important;
    margin: 0 !important;
}

/* ── Setup banner (first time) ── */
.setup-required {
    background: #fff7ed;
    border: 2px solid #f97316;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.setup-required h3 {
    color: #c2410c !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    display: flex;
    align-items: center;
    gap: 6px;
}
.setup-required p {
    color: #7c2d12 !important;
    font-size: 13px !important;
    margin: 0 0 14px 0 !important;
    line-height: 1.6 !important;
}
.prereq-steps {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.prereq-step {
    background: #ffffff;
    border: 1px solid #fed7aa;
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1;
    min-width: 160px;
}
.prereq-step .pnum {
    background: #f97316;
    color: white !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    margin-bottom: 6px;
}
.prereq-step .ptitle {
    color: #1e1b4b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    display: block;
    margin-bottom: 2px;
}
.prereq-step .pdesc {
    color: #6b7280 !important;
    font-size: 11px !important;
    line-height: 1.4 !important;
}

/* ── Welcome banner ── */
.welcome-banner {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    border: 1px solid #c7d2fe;
    border-top: 4px solid #4f46e5;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 22px;
}
.welcome-banner h3 {
    color: #1e1b4b !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    margin: 0 0 10px 0 !important;
}
.welcome-banner p {
    color: #4b5563 !important;
    font-size: 13px !important;
    margin: 0 0 16px 0 !important;
    line-height: 1.7 !important;
}
.step-cards {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.step-card {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    padding: 14px 16px;
    flex: 1;
    min-width: 160px;
    box-shadow: 0 1px 4px rgba(79,70,229,0.08);
}
.step-card .sc-num {
    background: #4f46e5;
    color: white !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}
.step-card .sc-title {
    color: #1e1b4b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    display: block;
    margin-bottom: 4px;
}
.step-card .sc-desc {
    color: #6b7280 !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

/* ── Info card ── */
.info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #4f46e5;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.info-card .card-title {
    color: #1e1b4b !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    margin: 0 0 14px 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.tag-admin {
    background: #fee2e2;
    color: #b91c1c !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}
.tag-once {
    background: #dcfce7;
    color: #15803d !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 10px;
}
.step-circle {
    background: #4f46e5;
    color: white !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-circle.warn { background: #f97316; }
.step-circle.info { background: #0ea5e9; }
.step-text {
    color: #4b5563 !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    flex: 1;
}
.step-text strong { color: #1e1b4b !important; }
.step-text code {
    background: #f3f4f6;
    color: #4f46e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    padding: 1px 6px;
    border-radius: 4px;
    border: 1px solid #e5e7eb;
}
.tip-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 14px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.tip-box .tip-text {
    color: #0369a1 !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
}
.tip-box .tip-text strong { color: #0c4a6e !important; }

/* ── Tab bar ── */
.tabs > .tab-nav {
    background: #ffffff !important;
    border-bottom: 2px solid #e5e7eb !important;
    padding: 0 20px !important;
    gap: 2px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
.tabs > .tab-nav button {
    color: #6b7280 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 22px !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
.tabs > .tab-nav button:hover {
    color: #4f46e5 !important;
    background: #f5f3ff !important;
}
.tabs > .tab-nav button.selected {
    color: #4f46e5 !important;
    border-bottom: 3px solid #4f46e5 !important;
    font-weight: 600 !important;
    background: transparent !important;
}

.tabitem {
    background: #f0f4ff !important;
    padding: 28px 32px !important;
}

/* ── Inputs ── */
textarea, input[type="text"] {
    background: #ffffff !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #111827 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    outline: none !important;
}
label span { color: #374151 !important; font-size: 13px !important; font-weight: 500 !important; }

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 11px 28px !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
}
button.secondary {
    background: #ffffff !important;
    border: 2px solid #4f46e5 !important;
    border-radius: 10px !important;
    color: #4f46e5 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 11px 28px !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    background: #eef2ff !important;
    transform: translateY(-1px) !important;
}

/* ── Markdown output ── */
.prose p, .prose li {
    color: #374151 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f4ff; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f46e5; }
"""

# ── Backend ────────────────────────────────────────────────────────────────────

def handle_query(user_input):
    if not user_input.strip():
        return "⚠️ Please type your question first.", ""
    embeddings = create_embeddings()
    index = pull_from_pinecone(os.getenv("PINECONE_API_KEY"), "gcp-starter", "tickets", embeddings)
    relevant_docs = get_similar_docs(index, user_input)
    response = get_answer(relevant_docs, user_input)
    return response, user_input

def submit_ticket(user_input, hr_tickets, it_tickets, transport_tickets):
    if not user_input.strip():
        return "⚠️ Please type your complaint first.", hr_tickets, it_tickets, transport_tickets
    embeddings = create_embeddings()
    query_result = embeddings.embed_query(user_input)
    department = predict(query_result)
    if department == "HR":
        hr_tickets = hr_tickets + [user_input]
    elif department == "IT":
        it_tickets = it_tickets + [user_input]
    else:
        transport_tickets = transport_tickets + [user_input]
    return f"✅ Ticket submitted to the **{department}** department. Go to 📋 Pending Tickets to see it.", hr_tickets, it_tickets, transport_tickets

def show_tickets(hr, it, transport):
    def fmt(icon, label, tickets):
        if not tickets:
            return f"### {icon} {label}\n*No tickets yet.*\n\n"
        lines = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tickets)])
        return f"### {icon} {label} - {len(tickets)} ticket(s)\n{lines}\n\n"
    return fmt("👥", "HR Support", hr) + "---\n\n" + fmt("💻", "IT Support", it) + "---\n\n" + fmt("🚌", "Transportation Support", transport)

def load_documents_to_pinecone():
    if not os.path.exists(DOCS_PATH):
        return f"⚠️ Documents folder not found at: `{DOCS_PATH}`"
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        return "⚠️ No PDF files found. Add your policy PDFs to the Documents folder and try again."
    embeddings = create_embeddings_load_data()
    log = []
    for pdf_file in pdf_files:
        full_path = os.path.join(DOCS_PATH, pdf_file)
        text = read_pdf_data(full_path)
        docs_chunks = split_data(text)
        push_to_pinecone(os.getenv("PINECONE_API_KEY"), "gcp-starter", "tickets", embeddings, docs_chunks)
        log.append(f"✅ Indexed: **{pdf_file}**")
    return "\n\n".join(log) + "\n\n---\n\n✅ All documents pushed to Pinecone. Knowledge base is ready - go to **🔍 Ask a Question** to test it."

def load_csv_and_embed():
    if not os.path.exists(CSV_PATH):
        return f"⚠️ CSV not found at: `{CSV_PATH}`", None
    df = read_data(CSV_PATH)
    embeddings = get_embeddings()
    cleaned = create_embeddings_for_df(df, embeddings)
    return f"✅ Loaded **{len(cleaned)} training examples**. Now go to **2 · Train** →", cleaned

def train_model(cleaned_data):
    if cleaned_data is None:
        return "⚠️ Complete Step 1 - Load Data first.", None, None, None, None, None
    s_train, s_test, l_train, l_test = split_train_test__data(cleaned_data)
    clf = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    clf.fit(s_train, l_train)
    return f"✅ Model trained on **{len(s_train)} examples**. Now go to **3 · Evaluate** →", clf, s_train, s_test, l_train, l_test

def evaluate_model(clf, s_test, l_test):
    if clf is None:
        return "⚠️ Complete Step 2 - Train Model first."
    score = get_score(clf, s_test, l_test)
    sample_text = "Rude driver with scary driving"
    emb = get_embeddings()
    result = clf.predict([emb.embed_query(sample_text)])
    return (f"✅ Accuracy: **{100*score:.2f}%**\n\n"
            f"**Sample:** \"{sample_text}\" → **{result[0]}**\n\n"
            f"If this looks correct, go to **4 · Save** →")

def save_model(clf):
    if clf is None:
        return "⚠️ Complete Step 2 - Train Model first."
    joblib.dump(clf, MODEL_PATH)
    return f"✅ Model saved as `{MODEL_PATH}`.\n\n🎉 Setup complete! Now go to **📁 Load Knowledge Base** → then **🔍 Ask a Question**."

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Ticket Classification Tool") as demo:

    hr_state = gr.State([])
    it_state = gr.State([])
    transport_state = gr.State([])
    cleaned_state = gr.State(None)
    clf_state = gr.State(None)
    s_train_state = gr.State(None)
    s_test_state = gr.State(None)
    l_train_state = gr.State(None)
    l_test_state = gr.State(None)
    last_input = gr.State("")

    gr.HTML("""
    <div class="app-header">
        <h1>🎫 Automatic Ticket Classification Tool</h1>
        <p>AI-powered support - describe your issue in plain English and we'll answer your question and route your ticket automatically.</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Ask a Question ──────────────────────────────────────────────
        with gr.Tab("🔍 Ask a Question"):

            gr.HTML("""
            <div class="setup-required">
                <h3>⚠️ Before using this tab - complete the setup first</h3>
                <p>
                    This app requires a one-time setup before it can answer questions or route tickets.
                    If you are using this app for the first time, please complete the steps below <strong>in order</strong>
                    before typing anything here. Once the setup is done, you never need to do it again.
                </p>
                <div class="prereq-steps">
                    <div class="prereq-step">
                        <span class="pnum">STEP 1 - DO FIRST</span>
                        <span class="ptitle">🤖 Train ML Model</span>
                        <span class="pdesc">Go to the Train ML Model tab and complete all 4 sub-steps: Load Data → Train → Evaluate → Save.</span>
                    </div>
                    <div class="prereq-step">
                        <span class="pnum">STEP 2 - DO SECOND</span>
                        <span class="ptitle">📁 Load Knowledge Base</span>
                        <span class="pdesc">Go to the Load Knowledge Base tab and click Push PDFs to Pinecone. Wait for the success message.</span>
                    </div>
                    <div class="prereq-step">
                        <span class="pnum">STEP 3 - YOU ARE HERE</span>
                        <span class="ptitle">🔍 Ask a Question</span>
                        <span class="pdesc">Once setup is done, come back here to ask questions and submit tickets.</span>
                    </div>
                </div>
            </div>

            <div class="welcome-banner">
                <h3>👋 How to use this tab</h3>
                <p>
                    Type your complaint or question below. The AI will search the company policy documents and give you an instant answer.
                    Once you have read the answer, click <strong>Submit Ticket</strong> to log a formal support request - it will be
                    automatically sent to the right department (HR, IT or Transportation) without you having to decide.
                </p>
                <div class="step-cards">
                    <div class="step-card">
                        <span class="sc-num">STEP 1</span>
                        <span class="sc-title">Type your issue</span>
                        <span class="sc-desc">Write your complaint in plain English - no special format needed. E.g. "My laptop won't turn on."</span>
                    </div>
                    <div class="step-card">
                        <span class="sc-num">STEP 2</span>
                        <span class="sc-title">Click Get Answer</span>
                        <span class="sc-desc">The AI searches the policy documents and gives you an instant answer based on company policy.</span>
                    </div>
                    <div class="step-card">
                        <span class="sc-num">STEP 3</span>
                        <span class="sc-title">Click Submit Ticket</span>
                        <span class="sc-desc">Your ticket is automatically routed to HR, IT or Transportation. Check Pending Tickets to confirm.</span>
                    </div>
                </div>
            </div>
            """)

            user_input = gr.Textbox(
                label="What is your complaint or question?",
                placeholder='e.g. "I have not received my salary this month" or "My laptop is not turning on"',
                lines=4
            )
            with gr.Row():
                ask_btn    = gr.Button("💬 Get Answer", variant="primary")
                submit_btn = gr.Button("📨 Submit Ticket", variant="secondary")

            answer_output = gr.Markdown(label="Answer")
            ticket_status = gr.Markdown()

            gr.HTML("""
            <div class="tip-box">
                <span>💡</span>
                <span class="tip-text">
                    <strong>Tip:</strong> Always click <strong>Get Answer</strong> first - the AI may already have the answer from the policy documents.
                    Only click <strong>Submit Ticket</strong> if you still need support from a human team member.
                    After submitting, check the <strong>📋 Pending Tickets</strong> tab to confirm your ticket was received.
                </span>
            </div>
            """)

            ask_btn.click(fn=handle_query, inputs=user_input, outputs=[answer_output, last_input])
            submit_btn.click(fn=submit_ticket, inputs=[user_input, hr_state, it_state, transport_state],
                             outputs=[ticket_status, hr_state, it_state, transport_state])

        # ── Tab 2: Pending Tickets ─────────────────────────────────────────────
        with gr.Tab("📋 Pending Tickets"):

            gr.HTML("""
            <div class="info-card">
                <div class="card-title">📋 View all submitted tickets <span class="tag-admin">READ ONLY</span></div>
                <div class="step-row">
                    <div class="step-circle">1</div>
                    <div class="step-text">Click <strong>Refresh Tickets</strong> below to load all tickets submitted during this session.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">2</div>
                    <div class="step-text">Tickets are grouped by department - <strong>HR</strong>, <strong>IT</strong> and <strong>Transportation</strong>. The AI automatically decided which department each ticket belongs to when it was submitted.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle warn">!</div>
                    <div class="step-text">If you do not see your ticket here, go back to <strong>🔍 Ask a Question</strong> and make sure you clicked <strong>Submit Ticket</strong>, not just Get Answer.</div>
                </div>
                <div class="tip-box">
                    <span>⚠️</span>
                    <span class="tip-text"><strong>Please note:</strong> Tickets are stored in memory for this session only. If the app is restarted, the ticket list will reset to empty.</span>
                </div>
            </div>
            """)

            refresh_btn     = gr.Button("🔄 Refresh Tickets", variant="primary")
            tickets_display = gr.Markdown()
            refresh_btn.click(fn=show_tickets, inputs=[hr_state, it_state, transport_state], outputs=tickets_display)

        # ── Tab 3: Load Knowledge Base ─────────────────────────────────────────
        with gr.Tab("📁 Load Knowledge Base"):

            gr.HTML(f"""
            <div class="info-card">
                <div class="card-title">📁 Set up the AI knowledge base <span class="tag-once">DO ONCE</span></div>
                <div class="step-row">
                    <div class="step-circle info">?</div>
                    <div class="step-text"><strong>What is this?</strong> This is the second setup step. The knowledge base is what the AI reads when you click "Get Answer". It is built from your company's PDF policy documents (HR handbook, travel policy, etc.). Without this, the AI cannot answer any questions.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">1</div>
                    <div class="step-text">Make sure you have already completed <strong>🤖 Train ML Model</strong> first. If not, do that tab first and then come back here.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">2</div>
                    <div class="step-text">Make sure your PDF policy documents are in the <code>Documents</code> folder:<br><code>{DOCS_PATH}</code></div>
                </div>
                <div class="step-row">
                    <div class="step-circle">3</div>
                    <div class="step-text">Click <strong>Push PDFs to Pinecone</strong> below. The app will read every PDF, break it into pieces and upload them to the cloud. This takes a few minutes - wait for the green success message before doing anything else.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">4</div>
                    <div class="step-text">Once you see the success message, the setup is fully complete. Go to <strong>🔍 Ask a Question</strong> and test with something like: <code>How many days of annual leave am I entitled to?</code></div>
                </div>
                <div class="tip-box">
                    <span>✅</span>
                    <span class="tip-text"><strong>You only need to do this once.</strong> The documents stay in Pinecone even after the app restarts. Only redo this step if you add new documents to the folder.</span>
                </div>
            </div>
            """)

            load_docs_btn    = gr.Button("📤 Push PDFs to Pinecone", variant="primary")
            load_docs_status = gr.Markdown()
            load_docs_btn.click(fn=load_documents_to_pinecone, inputs=[], outputs=load_docs_status)

        # ── Tab 4: Train ML Model ──────────────────────────────────────────────
        with gr.Tab("🤖 Train ML Model"):

            gr.HTML(f"""
            <div class="info-card">
                <div class="card-title">🤖 Train the ticket routing AI <span class="tag-admin">START HERE - DO FIRST</span></div>
                <div class="step-row">
                    <div class="step-circle info">?</div>
                    <div class="step-text"><strong>What is this?</strong> This is the first setup step. This tab builds the AI brain that reads a complaint and decides which department to send it to - HR, IT or Transportation. You must train and save the model here before the <strong>Submit Ticket</strong> button will work.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">1</div>
                    <div class="step-text">Click the <strong>1 · Load Data</strong> sub-tab → click the button. The app reads your training file and converts each example complaint into numbers the AI can understand. <strong>This takes 1–2 minutes</strong> - the page may look frozen but it is working. Watch the terminal for progress.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">2</div>
                    <div class="step-text">Click the <strong>2 · Train</strong> sub-tab → click the button. The AI learns the difference between HR, IT and Transportation complaints. Takes about <strong>30 seconds</strong>.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">3</div>
                    <div class="step-text">Click the <strong>3 · Evaluate</strong> sub-tab → click the button. You will see an accuracy score. Anything <strong>above 80%</strong> means the model is working well.</div>
                </div>
                <div class="step-row">
                    <div class="step-circle">4</div>
                    <div class="step-text">Click the <strong>4 · Save</strong> sub-tab → click the button. <strong>This step is mandatory.</strong> If you skip it, the model will be lost the next time the app starts and you will have to train it again from scratch.</div>
                </div>
                <div class="tip-box">
                    <span>➡️</span>
                    <span class="tip-text"><strong>After saving the model:</strong> Go to the <strong>📁 Load Knowledge Base</strong> tab to complete the second part of the setup. Do not skip it - the AI cannot answer questions without it.</span>
                </div>
            </div>
            """)

            with gr.Tabs():

                with gr.Tab("1 · Load Data"):
                    gr.Markdown("**Step 1 of 4** - Reads the training data and converts complaints into numbers the AI can learn from. Takes 1–2 minutes. The button may appear frozen - this is normal, keep waiting.")
                    load_data_btn    = gr.Button("Load CSV & Create Embeddings", variant="primary")
                    load_data_status = gr.Markdown()
                    load_data_btn.click(fn=load_csv_and_embed, inputs=[], outputs=[load_data_status, cleaned_state])

                with gr.Tab("2 · Train"):
                    gr.Markdown("**Step 2 of 4** - Trains the classifier on the loaded data. Only click this after Step 1 shows a green tick.")
                    train_btn    = gr.Button("Train SVM Classifier", variant="primary")
                    train_status = gr.Markdown()
                    train_btn.click(fn=train_model, inputs=cleaned_state,
                                    outputs=[train_status, clf_state, s_train_state, s_test_state, l_train_state, l_test_state])

                with gr.Tab("3 · Evaluate"):
                    gr.Markdown("**Step 3 of 4** - Tests the model on complaints it has never seen and shows you the accuracy. Only click after Step 2 is complete.")
                    eval_btn    = gr.Button("Evaluate Model", variant="primary")
                    eval_output = gr.Markdown()
                    eval_btn.click(fn=evaluate_model, inputs=[clf_state, s_test_state, l_test_state], outputs=eval_output)

                with gr.Tab("4 · Save"):
                    gr.Markdown("**Step 4 of 4** - Saves the model to disk. **You must click this.** If you skip it, the model will be gone after the app restarts and Submit Ticket will not work.")
                    save_btn    = gr.Button("💾 Save Model", variant="primary")
                    save_status = gr.Markdown()
                    save_btn.click(fn=save_model, inputs=clf_state, outputs=save_status)

if __name__ == "__main__":
    demo.launch()