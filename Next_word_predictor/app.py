
import streamlit as st
import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="NextWord AI",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------
# Custom styling
# ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #f6f3ff, #f8faff);
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 760px;
    padding-top: 3rem;
    padding-bottom: 2rem;
}

.hero {
    text-align: center;
    padding: 20px 0 30px;
}

.badge {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 30px;
    background: #e9e2ff;
    color: #6941c6;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 44px);
    color: #24213b;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0 0 12px;
}

.hero h1 span {
    color: #7956e8;
}

.hero p {
    color: #77758b;
    font-size: 15px;
    line-height: 1.7;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #eae6f7;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 12px 35px rgba(65, 45, 120, 0.06);
}

div[data-testid="stTextInput"] input {
    background: #faf9ff;
    border: 1px solid #e4dff2;
    border-radius: 12px;
    min-height: 48px;
    font-size: 15px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #7956e8;
    box-shadow: 0 0 0 1px #7956e8;
}

div.stButton > button {
    border-radius: 11px;
    min-height: 45px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button[kind="primary"] {
    background: #7956e8;
    border: 1px solid #7956e8;
    color: white;
}

div.stButton > button[kind="primary"]:hover {
    background: #6341d1;
    border-color: #6341d1;
}

.result {
    margin-top: 24px;
    padding: 24px;
    background: linear-gradient(135deg, #eee8ff, #f7f4ff);
    border: 1px solid #ddd1ff;
    border-radius: 18px;
}

.result-label {
    color: #7054be;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-word {
    color: #5536bd;
    font-size: 36px;
    font-weight: 800;
    overflow-wrap: anywhere;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #9290a4;
    font-size: 12px;
    padding-top: 28px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model(str(BASE_DIR / "lstm_model.h5"))

    with open(BASE_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(BASE_DIR / "max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return ""

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    preds = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(preds[0]))

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return ""


# ------------------------------
# Header
# ------------------------------
st.markdown("""
<div class="hero">
    <div class="badge">✦ AI-POWERED LANGUAGE MODEL</div>
    <h1>NextWord <span>AI.</span></h1>
    <p>
        Every sentence has a next word.<br>
        Let your LSTM model predict what comes next.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Input card
# ------------------------------
with st.container(border=True):
    st.markdown("### ✍️ Enter your text")
    st.caption("Type a phrase and let the model complete it.")

    user_input = st.text_input(
        "Your sentence",
        placeholder="e.g. I am not a",
        label_visibility="collapsed"
    )

    st.caption("TRY AN EXAMPLE")

    col1, col2, col3 = st.columns(3)

    if col1.button("I am not a", use_container_width=True):
        st.session_state["example_text"] = "I am not a"

    if col2.button("The future of", use_container_width=True):
        st.session_state["example_text"] = "The future of"

    if col3.button("Machine learning", use_container_width=True):
        st.session_state["example_text"] = "Machine learning"

    # Use the selected example on the next rerun.
    if "example_text" in st.session_state:
        st.info(
            f"Example selected: {st.session_state['example_text']}"
        )

    predict_clicked = st.button(
        "✦  Predict Next Word",
        type="primary",
        use_container_width=True
    )

# ------------------------------
# Prediction output
# ------------------------------
if predict_clicked:
    text_to_predict = st.session_state.get(
        "example_text", ""
    ) if not user_input.strip() else user_input.strip()

    if not text_to_predict:
        st.warning("Please enter some text first.")
    else:
        try:
            model, tokenizer, max_len = load_resources()

            # Use the same prediction logic with loaded resources.
            sequence = tokenizer.texts_to_sequences(
                [text_to_predict]
            )[0]

            if not sequence:
                st.warning(
                    "No recognized words found. Try another phrase."
                )
            else:
                sequence = pad_sequences(
                    [sequence],
                    maxlen=max_len - 1,
                    padding="pre"
                )

                preds = model.predict(sequence, verbose=0)
                predicted_index = int(np.argmax(preds[0]))

                next_word = next(
                    (
                        word
                        for word, index in tokenizer.word_index.items()
                        if index == predicted_index
                    ),
                    ""
                )

                if next_word:
                    st.markdown(f"""
                    <div class="result">
                        <div class="result-label">
                            ✦ PREDICTED NEXT WORD
                        </div>
                        <div class="result-word">
                            {next_word}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No next word was found.")

        except FileNotFoundError as e:
            st.error(f"Missing file: {e.filename}")
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div class="footer">
    Built with Python · TensorFlow · LSTM · Streamlit
    <br><br>
    Predicting one word at a time.
</div>
""", unsafe_allow_html=True)
