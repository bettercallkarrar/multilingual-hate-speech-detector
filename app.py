import streamlit as st
from transformers import pipeline

try:
    from src.predict import load_model as project_load_model
except ImportError:
    project_load_model = None


MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

EXAMPLES = {
    "English": [
        "You are amazing and kind.",
        "I hate you.",
    ],
    "Arabic": [
        "أنت شخص رائع",
        "أكرهك",
    ],
    "French": [
        "Tu es gentil.",
        "Je te déteste.",
    ],
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    if project_load_model is not None:
        return project_load_model()

    return pipeline("text-classification", model=MODEL_NAME)


def normalize_result(model_label):
    label = model_label.lower()

    if "very negative" in label:
        return "Hate Speech", "danger", "High-risk harmful language detected."
    if "negative" in label:
        return "Offensive", "warning", "Potentially offensive language detected."
    return "Safe", "success", "No hateful or offensive signal detected."


def analyze_text(text):
    classifier = load_classifier()
    result = classifier(text)[0]
    status, tone, explanation = normalize_result(result["label"])

    return {
        "status": status,
        "tone": tone,
        "explanation": explanation,
        "model_output": result["label"],
        "confidence": result["score"],
    }


def apply_page_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #14213d;
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #14213d;
            letter-spacing: 0;
        }

        .hero {
            border-left: 5px solid #2563eb;
            padding: 1.2rem 1.4rem;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.2rem;
        }

        .hero p {
            color: #526070;
            margin-bottom: 0;
            font-size: 1.02rem;
        }

        .result-card {
            background: #ffffff;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
            margin-top: 1rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .status-success {
            color: #166534;
            background: #dcfce7;
        }

        .status-warning {
            color: #92400e;
            background: #fef3c7;
        }

        .status-danger {
            color: #991b1b;
            background: #fee2e2;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .metric-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.85rem;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
        }

        .metric-value {
            color: #14213d;
            font-size: 1.02rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #1d4ed8;
            background: #2563eb;
            color: #ffffff;
            font-weight: 700;
            min-height: 2.8rem;
        }

        div.stButton > button:hover {
            border-color: #1e40af;
            background: #1d4ed8;
            color: #ffffff;
        }

        @media (max-width: 640px) {
            .metric-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result(prediction):
    confidence_percent = prediction["confidence"] * 100
    st.markdown(
        f"""
        <div class="result-card">
            <div class="status-pill status-{prediction['tone']}">
                {prediction['status']}
            </div>
            <h3>{prediction['explanation']}</h3>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Model Output</div>
                    <div class="metric-value">{prediction['model_output']}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence_percent:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Multilingual Hate Speech Detector",
        page_icon=":globe_with_meridians:",
        layout="centered",
    )
    apply_page_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Multilingual Hate Speech Detector</h1>
            <p>Analyze English, Arabic, and French text with a transformer-based multilingual model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.subheader("Try an example")
    example_columns = st.columns(3)
    for column, (language, examples) in zip(example_columns, EXAMPLES.items()):
        with column:
            st.caption(language)
            for example in examples:
                if st.button(example, key=f"{language}-{example}"):
                    st.session_state.input_text = example

    text = st.text_area(
        "Enter text to analyze",
        key="input_text",
        height=160,
        placeholder="Type or paste a message in English, Arabic, or French...",
    )

    analyze_clicked = st.button("Analyze Text", type="primary")

    if analyze_clicked:
        cleaned_text = text.strip()

        if not cleaned_text:
            st.warning("Please enter text before analyzing.")
            return

        with st.spinner("Analyzing text..."):
            try:
                prediction = analyze_text(cleaned_text)
            except Exception as exc:
                st.error(
                    "The model could not be loaded or used. Check your internet connection "
                    "and installed dependencies, then try again."
                )
                st.exception(exc)
                return

        render_result(prediction)


if __name__ == "__main__":
    main()

import streamlit as st
from transformers import pipeline

try:
    from src.predict import load_model as project_load_model
except ImportError:
    project_load_model = None


MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

EXAMPLES = {
    "English": [
        "You are amazing and kind.",
        "I hate you.",
    ],
    "Arabic": [
        "أنت شخص رائع",
        "أكرهك",
    ],
    "French": [
        "Tu es gentil.",
        "Je te déteste.",
    ],
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    if project_load_model is not None:
        return project_load_model()

    return pipeline("text-classification", model=MODEL_NAME)


def normalize_result(model_label):
    label = model_label.lower()

    if "very negative" in label:
        return "Hate Speech", "danger", "High-risk harmful language detected."
    if "negative" in label:
        return "Offensive", "warning", "Potentially offensive language detected."
    return "Safe", "success", "No hateful or offensive signal detected."


def analyze_text(text):
    classifier = load_classifier()
    result = classifier(text)[0]
    status, tone, explanation = normalize_result(result["label"])

    return {
        "status": status,
        "tone": tone,
        "explanation": explanation,
        "model_output": result["label"],
        "confidence": result["score"],
    }


def apply_page_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #14213d;
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #14213d;
            letter-spacing: 0;
        }

        .hero {
            border-left: 5px solid #2563eb;
            padding: 1.2rem 1.4rem;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.2rem;
        }

        .hero p {
            color: #526070;
            margin-bottom: 0;
            font-size: 1.02rem;
        }

        .result-card {
            background: #ffffff;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
            margin-top: 1rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .status-success {
            color: #166534;
            background: #dcfce7;
        }

        .status-warning {
            color: #92400e;
            background: #fef3c7;
        }

        .status-danger {
            color: #991b1b;
            background: #fee2e2;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .metric-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.85rem;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
        }

        .metric-value {
            color: #14213d;
            font-size: 1.02rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #1d4ed8;
            background: #2563eb;
            color: #ffffff;
            font-weight: 700;
            min-height: 2.8rem;
        }

        div.stButton > button:hover {
            border-color: #1e40af;
            background: #1d4ed8;
            color: #ffffff;
        }

        @media (max-width: 640px) {
            .metric-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result(prediction):
    confidence_percent = prediction["confidence"] * 100
    st.markdown(
        f"""
        <div class="result-card">
            <div class="status-pill status-{prediction['tone']}">
                {prediction['status']}
            </div>
            <h3>{prediction['explanation']}</h3>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Model Output</div>
                    <div class="metric-value">{prediction['model_output']}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence_percent:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Multilingual Hate Speech Detector",
        page_icon=":globe_with_meridians:",
        layout="centered",
    )
    apply_page_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Multilingual Hate Speech Detector</h1>
            <p>Analyze English, Arabic, and French text with a transformer-based multilingual model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.subheader("Try an example")
    example_columns = st.columns(3)
    for column, (language, examples) in zip(example_columns, EXAMPLES.items()):
        with column:
            st.caption(language)
            for example in examples:
                if st.button(example, key=f"{language}-{example}"):
                    st.session_state.input_text = example

    text = st.text_area(
        "Enter text to analyze",
        key="input_text",
        height=160,
        placeholder="Type or paste a message in English, Arabic, or French...",
    )

    analyze_clicked = st.button("Analyze Text", type="primary")

    if analyze_clicked:
        cleaned_text = text.strip()

        if not cleaned_text:
            st.warning("Please enter text before analyzing.")
            return

        with st.spinner("Analyzing text..."):
            try:
                prediction = analyze_text(cleaned_text)
            except Exception as exc:
                st.error(
                    "The model could not be loaded or used. Check your internet connection "
                    "and installed dependencies, then try again."
                )
                st.exception(exc)
                return

        render_result(prediction)


if __name__ == "__main__":
    main()
