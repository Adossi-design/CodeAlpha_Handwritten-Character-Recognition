"""
HandScript AI: Handwritten Character Recognition
==================================================

A Streamlit web app that recognises handwritten English capital letters (A-Z)
using a trained convolutional neural network.

Run from the project root:

    streamlit run app/app.py

Author: Adossi Fred William | CodeAlpha Machine Learning Internship
"""

import os
import string

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.keras")
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_images")

LETTERS = list(string.ascii_uppercase)          # ['A', 'B', ..., 'Z']
DATASET_SIZE = 372_450
NUM_CLASSES = 26
# Headline test accuracy achieved during training (see notebook / models/metrics.json)
TEST_ACCURACY = 0.9891

# Theme colours
DARK_BG = "#0F0F1A"
PANEL_BG = "#1A1A2E"
ACCENT = "#00E5FF"
GREEN = "#2ECC71"
ROSE = "#FF4D6D"

# ---------------------------------------------------------------------------
# Page configuration & global dark styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HandScript AI",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #FFFFFF; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; }}
    .metric-card {{
        background: {PANEL_BG};
        border: 1px solid #2A2A40;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }}
    .metric-card h2 {{ color: {ACCENT}; margin: 0; font-size: 1.9rem; }}
    .metric-card p  {{ color: #AAB; margin: 4px 0 0 0; font-size: 0.85rem; }}
    .pred-badge {{
        background: {GREEN};
        color: #06210F;
        border-radius: 18px;
        text-align: center;
        padding: 10px 0;
        font-size: 6rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .section-head {{
        border-left: 4px solid {ACCENT};
        padding-left: 10px;
        margin: 8px 0 4px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recognition model...")
def load_recognition_model():
    """Load the trained Keras model once and cache it for the session."""
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(pil_image):
    """Convert an arbitrary PIL image into the model's input tensor.

    Steps: grayscale -> resize to 28x28 -> normalise to [0, 1] ->
    reshape to (1, 28, 28, 1). The grayscale 28x28 array is also returned
    for display. Auto-inverts so the character is light-on-dark (matching the
    training data) when a light-background photo is supplied.
    """
    gray = pil_image.convert("L")
    resized = gray.resize((28, 28), Image.LANCZOS)
    arr = np.array(resized, dtype="float32")

    # Training data is white strokes on a black background. If the uploaded
    # image is dark strokes on a light background, invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    norm = arr / 255.0
    tensor = norm.reshape(1, 28, 28, 1)
    return tensor, arr


def render_letter_image(letter):
    """Render a clean glyph for a letter as a fallback sample (light on dark)."""
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((28 - w) / 2 - bbox[0], (28 - h) / 2 - bbox[1]), letter, fill=255, font=font)
    return img


def load_sample_for_letter(letter):
    """Return a PIL sample image for a letter: from sample_images/ if present,
    otherwise a font-rendered fallback."""
    path = os.path.join(SAMPLE_DIR, f"{letter}.png")
    if os.path.exists(path):
        return Image.open(path)
    return render_letter_image(letter)


# ---------------------------------------------------------------------------
# Sidebar: input controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h1 style='color:{ACCENT};'>🖋️ HandScript AI</h1>", unsafe_allow_html=True)
    st.caption("Handwritten Character Recognition - A-Z")
    st.divider()

    mode = st.radio(
        "Input mode",
        ["📤 Upload image", "🔤 Type a letter"],
        help="Upload your own handwriting or pick a sample letter from the test set.",
    )

    source_image = None
    if mode == "📤 Upload image":
        uploaded = st.file_uploader(
            "Upload a letter image", type=["png", "jpg", "jpeg"]
        )
        if uploaded is not None:
            try:
                source_image = Image.open(uploaded)
            except Exception:
                st.warning("⚠️ Could not open that file. Please upload a valid PNG/JPG image.")
    else:
        chosen = st.selectbox("Pick a letter (sample from test set)", LETTERS)
        source_image = load_sample_for_letter(chosen)
        st.image(source_image.resize((120, 120)), caption=f"Sample: {chosen}")

    st.divider()
    predict = st.button("🚀 Predict", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"<h1 style='color:#FFFFFF;'>🖋️ HandScript AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#AAB;font-size:1.05rem;'>Deep-learning recognition of handwritten "
    "capital letters (A-Z), powered by a convolutional neural network.</p>",
    unsafe_allow_html=True,
)

# Guard: model must exist
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found at `{os.path.relpath(MODEL_PATH, ROOT_DIR)}`. "
        "Train it first by running the notebook (`notebook/handwritten_character_recognition.ipynb`)."
    )
    st.stop()

model = load_recognition_model()


def render_results(pil_image):
    """Run the full prediction pipeline and render all four sections."""
    try:
        tensor, gray28 = preprocess_image(pil_image)
    except Exception:
        st.warning("⚠️ The image could not be processed. Try a different file.")
        return

    probs = model.predict(tensor, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    pred_letter = LETTERS[top_idx]
    confidence = float(probs[top_idx])

    # ---- Section 1: Prediction result ----
    st.markdown("<h3 class='section-head'>Prediction Result</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.3, 1.2])
    with c1:
        st.markdown("**Predicted letter**")
        st.markdown(f"<div class='pred-badge'>{pred_letter}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Confidence**")
        st.markdown(f"<h2 style='color:{GREEN};'>{confidence * 100:.2f}%</h2>", unsafe_allow_html=True)
        st.progress(min(max(confidence, 0.0), 1.0))
    with c3:
        st.markdown("**Top 3 guesses**")
        for rank, i in enumerate(order[:3]):
            colour = GREEN if rank == 0 else "#FFFFFF"
            st.markdown(
                f"<span style='color:{colour};font-size:1.15rem;'>"
                f"{rank + 1}. <b>{LETTERS[int(i)]}</b> &nbsp;{probs[int(i)] * 100:.2f}%</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ---- Section 2: Preprocessed image display ----
    st.markdown("<h3 class='section-head'>Preprocessed Input</h3>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Grayscale - resized to 28x28**")
        st.image(gray28.astype("uint8"), width=220, clamp=True)
    with p2:
        st.markdown("**Raw pixel grid (heatmap)**")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        fig.patch.set_facecolor(DARK_BG)
        ax.imshow(gray28, cmap="magma")
        ax.axis("off")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.divider()

    # ---- Section 3: Confidence distribution chart ----
    st.markdown("<h3 class='section-head'>Confidence Across All 26 Letters</h3>", unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    colours = [GREEN if i == top_idx else ACCENT for i in range(NUM_CLASSES)]
    ax.barh(LETTERS, probs * 100, color=colours)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence (%)", color="#FFFFFF")
    ax.tick_params(colors="#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#3A3A5A")
    ax.set_title(f"Predicted: {pred_letter}", color="#FFFFFF", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main panel: run prediction or show prompt
# ---------------------------------------------------------------------------
if predict and source_image is not None:
    render_results(source_image)
elif predict and source_image is None:
    st.warning("⚠️ Please upload an image or pick a letter first.")
else:
    st.info("👈 Choose an input mode in the sidebar, then press **Predict**.")

# ---------------------------------------------------------------------------
# Section 4: Model info cards (always visible at the bottom)
# ---------------------------------------------------------------------------
st.divider()
st.markdown("<h3 class='section-head'>Model Information</h3>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
cards = [
    (m1, f"{TEST_ACCURACY * 100:.2f}%", "Test accuracy"),
    (m2, f"{model.count_params():,}", "Parameters"),
    (m3, f"{DATASET_SIZE:,}", "Dataset size"),
    (m4, f"{NUM_CLASSES}", "Classes (A-Z)"),
]
for col, value, label in cards:
    with col:
        st.markdown(
            f"<div class='metric-card'><h2>{value}</h2><p>{label}</p></div>",
            unsafe_allow_html=True,
        )
