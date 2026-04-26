import html
import streamlit as st
from transformers import pipeline

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

EXAMPLES = {
    "English": ["You are amazing and kind.", "I hate you."],
    "Arabic": ["أنت شخص رائع", "أكرهك"],
    "French": ["Tu es gentil.", "Je te déteste."],
    "Turkish": ["Sen çok naziksin.", "Senden nefret ediyorum."],
    "Spanish": ["Eres muy amable.", "Te odio."],
}

st.set_page_config(
    page_title="Multilingual Hate Speech Detector",
    page_icon="🌍",
    layout="wide",
)

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("text-classification", model=MODEL_NAME)

def classify_label(label):
    label_lower = label.lower()
    if "very negative" in label_lower:
        return "Hate Speech", "danger", "High-risk harmful language detected."
    if "negative" in label_lower:
        return "Offensive", "warning", "Potentially offensive language detected."
    return "Not Hate", "success", "No hateful or offensive signal detected."

def set_example(text):
    st.session_state.input_text = text

def css():
    st.markdown("""
    <style>
    :root {
        --ink:#101828; --muted:#667085; --blue:#2364ff; --teal:#0f9f8f;
        --green:#14804a; --amber:#b76e00; --red:#c92a2a; --line:#d8dee8;
    }
    .stApp {
        background:
          radial-gradient(circle at 10% 5%, rgba(35,100,255,.16), transparent 30rem),
          radial-gradient(circle at 90% 8%, rgba(15,159,143,.12), transparent 24rem),
          linear-gradient(135deg,#f7f9fc 0%,#edf3f8 50%,#fafafa 100%);
        color:var(--ink);
    }
    [data-testid="stHeader"] { background:transparent; }
    .main .block-container { max-width:1180px; padding-top:1.7rem; padding-bottom:3rem; }
    h1,h2,h3,p { letter-spacing:0; }

    .hero {
        border-radius:8px; padding:2rem; color:white;
        background:linear-gradient(135deg,rgba(16,24,40,.96),rgba(23,78,219,.92));
        box-shadow:0 24px 70px rgba(16,24,40,.20);
        margin-bottom:1rem;
    }
    .hero-kicker {
        display:inline-flex; border:1px solid rgba(255,255,255,.24);
        border-radius:999px; background:rgba(255,255,255,.12);
        color:#eaf1ff; padding:.35rem .75rem; font-size:.82rem;
        font-weight:800; margin-bottom:1rem;
    }
    .hero h1 { color:white; font-size:3.1rem; line-height:1.03; margin:0; font-weight:900; }
    .hero p { color:#dbe7ff; max-width:720px; margin:1rem 0 1.3rem; font-size:1.08rem; line-height:1.65; }
    .badges { display:flex; flex-wrap:wrap; gap:.55rem; }
    .badge {
        border-radius:999px; background:rgba(255,255,255,.13);
        border:1px solid rgba(255,255,255,.20); color:white;
        padding:.35rem .8rem; font-size:.86rem; font-weight:800;
    }

    .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:.85rem; margin:1rem 0 1.25rem; }
    .card, .result {
        background:rgba(255,255,255,.94); border:1px solid rgba(16,24,40,.10);
        border-radius:8px; box-shadow:0 16px 48px rgba(16,24,40,.10);
    }
    .stat { padding:1rem; }
    .stat-label { color:var(--muted); font-size:.78rem; font-weight:850; text-transform:uppercase; }
    .stat-value { color:var(--ink); font-size:1.45rem; font-weight:900; margin-top:.2rem; }

    .section-title { margin:.55rem 0 .8rem; }
    .section-title h2 { margin:0 0 .2rem; font-size:1.35rem; }
    .section-title p { margin:0; color:var(--muted); }

    .card { padding:1.1rem; }
    .card h3 { margin:0 0 .35rem; font-size:1rem; }
    .card p { margin:0 0 .9rem; color:var(--muted); line-height:1.55; }

    .guide { display:grid; gap:.7rem; }
    .guide-item { display:grid; grid-template-columns:.65rem 1fr; gap:.7rem; border-top:1px solid rgba(16,24,40,.08); padding-top:.75rem; }
    .dot { width:.65rem; height:.65rem; border-radius:50%; margin-top:.38rem; }
    .success-dot { background:var(--green); }
    .warning-dot { background:var(--amber); }
    .danger-dot { background:var(--red); }
    .guide-item strong { display:block; margin-bottom:.12rem; }
    .guide-item span { color:var(--muted); font-size:.92rem; }

    .stTextArea textarea {
        border-radius:8px; border:1px solid var(--line); min-height:210px;
        font-size:1rem; background:white; color:var(--ink);
    }
    .stTextArea textarea:focus {
        border-color:var(--blue); box-shadow:0 0 0 3px rgba(35,100,255,.14);
    }

    div.stButton > button {
        width:100%; border-radius:8px; border:1px solid rgba(16,24,40,.12);
        background:white; color:#182230; font-weight:850; min-height:2.75rem;
        box-shadow:0 5px 16px rgba(16,24,40,.07);
    }
    div.stButton > button:hover {
        border-color:rgba(35,100,255,.45); background:rgba(35,100,255,.08); color:#174edb;
    }
    div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"] {
        border-color:#174edb; background:linear-gradient(135deg,var(--blue),var(--teal));
        color:white; min-height:3.1rem;
    }

    .result { padding:1.25rem; margin-top:1.2rem; }
    .pill { display:inline-flex; border-radius:999px; padding:.4rem .8rem; font-weight:900; font-size:.9rem; margin-bottom:.8rem; }
    .status-success { color:#0f6b3c; background:#dcfce7; }
    .status-warning { color:#8a5200; background:#fef3c7; }
    .status-danger { color:#b42318; background:#fee2e2; }
    .result h3 { margin:0 0 .4rem; font-size:1.35rem; }
    .result p { margin:0; color:var(--muted); }

    .confidence { display:grid; grid-template-columns:1fr auto; gap:1rem; align-items:center; margin:1rem 0; }
    .track { height:.7rem; border-radius:999px; overflow:hidden; background:#e6eaf0; }
    .fill { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--teal),var(--blue)); }
    .percent { font-size:1.2rem; font-weight:950; }

    .metrics { display:grid; grid-template-columns:repeat(2,1fr); gap:.75rem; margin-top:1rem; }
    .metric { background:#f8fafc; border:1px solid #e6eaf0; border-radius:8px; padding:.85rem; }
    .metric.full { grid-column:1/-1; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; font-weight:850; }
    .metric-value { font-weight:850; margin-top:.2rem; overflow-wrap:anywhere; }

    .footer { text-align:center; color:#667085; margin-top:2rem; font-size:.9rem; }
    @media(max-width:760px) {
        .hero h1 { font-size:2.2rem; }
        .stats, .metrics { grid-template-columns:1fr; }
    }
    </style>
    """, unsafe_allow_html=True)

def render_result(text, label, score, decision, tone, explanation):
    confidence = score * 100
    st.markdown(f"""
    <div class="result">
        <div class="pill status-{tone}">Final Decision: {html.escape(decision)}</div>
        <h3>{html.escape(explanation)}</h3>
        <p>This score is generated by the model and should be treated as a review aid.</p>
        <div class="confidence">
            <div class="track"><div class="fill" style="width:{confidence:.1f}%"></div></div>
            <div class="percent">{confidence:.1f}%</div>
        </div>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Model Output</div>
                <div class="metric-value">{html.escape(label)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Decision Type</div>
                <div class="metric-value">{html.escape(decision)}</div>
            </div>
            <div class="metric full">
                <div class="metric-label">Input Text</div>
                <div class="metric-value">{html.escape(text)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    css()

    badges = "".join(f'<span class="badge">{lang}</span>' for lang in EXAMPLES)
    st.markdown(f"""
    <div class="hero">
        <div class="hero-kicker">AI moderation workspace</div>
        <h1>Multilingual Hate Speech Detector</h1>
        <p>Review short messages across five languages with a transformer model, a clear final decision, and confidence details designed for fast scanning.</p>
        <div class="badges">{badges}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats">
        <div class="card stat"><div class="stat-label">Languages</div><div class="stat-value">{len(EXAMPLES)}</div></div>
        <div class="card stat"><div class="stat-label">Model</div><div class="stat-value">Transformer</div></div>
        <div class="card stat"><div class="stat-label">Output</div><div class="stat-value">3 States</div></div>
    </div>
    """, unsafe_allow_html=True)

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.markdown("""
    <div class="section-title">
        <h2>Analyze a message</h2>
        <p>Paste text, or load one of the examples below.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        st.markdown("""
        <div class="card">
            <h3>Message input</h3>
            <p>Short, direct messages usually produce the clearest signal.</p>
        </div>
        """, unsafe_allow_html=True)

        text = st.text_area(
            "Text to analyze",
            key="input_text",
            height=210,
            placeholder="Type or paste a message in English, Arabic, French, Turkish, or Spanish...",
        )

        analyze = st.button("Analyze Text", type="primary")

    with right:
        st.markdown("""
        <div class="card">
            <h3>Detection guide</h3>
            <p>Use the result as a moderation signal, then review context before making a final call.</p>
            <div class="guide">
                <div class="guide-item"><div class="dot success-dot"></div><div><strong>Not Hate</strong><span>Positive or neutral signal</span></div></div>
                <div class="guide-item"><div class="dot warning-dot"></div><div><strong>Offensive</strong><span>Negative signal that needs review</span></div></div>
                <div class="guide-item"><div class="dot danger-dot"></div><div><strong>Hate Speech</strong><span>Very negative high-risk signal</span></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">
        <h2>Examples</h2>
        <p>Click any sample to place it in the analyzer.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(list(EXAMPLES.keys()))
    for tab, (language, examples) in zip(tabs, EXAMPLES.items()):
        with tab:
            cols = st.columns(len(examples))
            for col, example in zip(cols, examples):
                with col:
                    st.button(
                        example,
                        key=f"{language}-{example}",
                        on_click=set_example,
                        args=(example,),
                    )

    if analyze:
        cleaned = text.strip()
        if not cleaned:
            st.warning("Please enter text first.")
        else:
            with st.spinner("Analyzing text..."):
                model = load_model()
                result = model(cleaned)[0]

            label = result["label"]
            score = result["score"]
            decision, tone, explanation = classify_label(label)
            render_result(cleaned, label, score, decision, tone, explanation)

    st.markdown("""
    <div class="footer">
        Built with Python, Streamlit, Hugging Face Transformers, and PyTorch.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
