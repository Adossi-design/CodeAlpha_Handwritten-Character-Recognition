"""
HandScript AI: Handwritten Character Recognition
==================================================

A Streamlit web app that recognises handwritten English capital letters (A-Z)
using a trained convolutional neural network. Draw a letter, upload a photo of
one, or try a sample, and the model predicts it instantly.

Run from the project root:

    streamlit run app/app.py

Author: Adossi Fred William | CodeAlpha Machine Learning Internship
"""

import base64
import io
import json
import os
import string

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.keras")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.json")
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_images")

LETTERS = list(string.ascii_uppercase)
DATASET_SIZE = 372_450
LOW_CONFIDENCE = 0.60

INK = "#4338CA"      # primary accent (matches .streamlit/config.toml)
INK_SOFT = "#C7C9F5"  # non-winning bars
TEXT = "#1C1B22"
MUTED = "#6B6A75"

INPUT_MODES = {
    "draw": ":material/draw: Draw",
    "upload": ":material/upload: Upload",
    "sample": ":material/grid_view: Samples",
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HandScript AI",
    page_icon=":material/stylus_note:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@700&family=Inter:wght@400;600;800&display=swap');

    .block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }}

    h1.hs-brand {{
        font-family: 'Caveat', 'Segoe Print', cursive !important;
        font-size: 3.2rem !important; font-weight: 700 !important;
        line-height: 1 !important; color: {TEXT}; margin: 0; padding: 0 !important;
    }}
    .hs-brand span {{ color: {INK}; }}
    .hs-tagline {{ color: {MUTED}; font-size: 1rem; margin: .6rem 0 1.4rem 0; }}

    .hs-label {{
        font-size: .75rem; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; color: {MUTED}; margin: 0 0 .5rem 0;
    }}

    .hs-result {{ display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap; }}
    .hs-letter {{
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 5.5rem; font-weight: 800; line-height: 1;
        color: #FFFFFF; background: {INK};
        width: 8.5rem; height: 8.5rem; border-radius: 1.25rem;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }}
    .hs-conf {{ font-size: 2rem; font-weight: 800; color: {TEXT}; line-height: 1.1; }}
    .hs-conf-note {{ color: {MUTED}; font-size: .9rem; }}

    .hs-alt {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .25rem; }}
    .hs-chip {{
        border: 1px solid #E3E2DC; border-radius: 999px; padding: .2rem .7rem;
        font-size: .9rem; color: {TEXT}; background: #FFFFFF;
    }}
    .hs-chip b {{ color: {INK}; }}

    .hs-empty {{ text-align: center; color: {MUTED}; padding: 3.5rem 1rem; }}
    .hs-empty .hs-ghost {{
        font-family: 'Caveat', cursive; font-size: 5rem; color: #D9D8D1; line-height: 1;
    }}


    @media (max-width: 640px) {{
        .block-container {{ padding-top: 3.5rem; }}
        h1.hs-brand {{ font-size: 2.6rem !important; }}
        .hs-letter {{ font-size: 4.2rem; width: 6.5rem; height: 6.5rem; }}
        .hs-conf {{ font-size: 1.6rem; }}
        .st-key-hs_actions {{ justify-content: flex-start !important; margin-bottom: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Drawing pad (a small custom component that returns the sketch as a PNG)
# ---------------------------------------------------------------------------
PAD_HTML = """
<div class="pad-wrap">
  <canvas class="pad" width="280" height="280"></canvas>
  <div class="pad-bar">
    <span class="pad-hint">Draw one capital letter, large and centered</span>
    <button class="pad-clear" type="button">Clear</button>
  </div>
</div>
"""

PAD_CSS = """
.pad-wrap { width: 100%; max-width: 360px; margin: 0 auto; font-family: inherit; }
.pad {
  width: 100%; aspect-ratio: 1 / 1; display: block; touch-action: none; cursor: crosshair;
  background: #FFFFFF; border: 1.5px dashed #CFCDC4; border-radius: 16px;
}
.pad-bar { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 8px; }
.pad-hint { font-size: 13px; color: #6B6A75; }
.pad-clear {
  font: inherit; font-size: 14px; font-weight: 600; color: #4338CA; background: transparent;
  border: 1px solid #C7C9F5; border-radius: 10px; padding: 6px 14px; cursor: pointer;
}
.pad-clear:hover { background: #EEF0FF; }
"""

PAD_JS = """
export default function ({ parentElement, setStateValue }) {
  const canvas = parentElement.querySelector(".pad");
  const clear = parentElement.querySelector(".pad-clear");
  const ctx = canvas.getContext("2d");
  let drawing = false, dirty = false, last = null;

  const reset = () => {
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };
  if (!canvas.dataset.ready) { reset(); canvas.dataset.ready = "1"; }

  const point = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * canvas.width / r.width,
            (e.clientY - r.top) * canvas.height / r.height];
  };
  const stroke = (a, b) => {
    ctx.strokeStyle = "#1C1B22"; ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  };
  const down = (e) => {
    e.preventDefault(); canvas.setPointerCapture(e.pointerId);
    drawing = true; last = point(e); stroke(last, last); dirty = true;
  };
  const move = (e) => { if (!drawing) return; const p = point(e); stroke(last, p); last = p; };
  const up = () => {
    if (!drawing) return; drawing = false;
    if (dirty) setStateValue("image", canvas.toDataURL("image/png"));
  };
  const wipe = () => { reset(); dirty = false; setStateValue("image", ""); };

  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  clear.addEventListener("click", wipe);
  return () => {
    canvas.removeEventListener("pointerdown", down);
    canvas.removeEventListener("pointermove", move);
    canvas.removeEventListener("pointerup", up);
    canvas.removeEventListener("pointercancel", up);
    clear.removeEventListener("click", wipe);
  };
}
"""

try:
    drawing_pad = st.components.v2.component(
        "handscript_pad", html=PAD_HTML, css=PAD_CSS, js=PAD_JS
    )
except AttributeError:  # older Streamlit without custom components v2
    drawing_pad = None
    INPUT_MODES.pop("draw")


def image_from_data_url(data_url):
    """Decode a canvas PNG data URL into a PIL image (None if empty)."""
    if not data_url or "," not in data_url:
        return None
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


# ---------------------------------------------------------------------------
# Model & data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up the model...")
def load_recognition_model():
    """Load the trained Keras model once and share it across sessions."""
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_metrics():
    """Read the training metrics saved by the notebook."""
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_sample(letter):
    """Return the sample image for a letter from sample_images/, if present."""
    path = os.path.join(SAMPLE_DIR, f"{letter}.png")
    return Image.open(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(pil_image):
    """Convert an arbitrary PIL image into the model's input tensor.

    The training letters are framed MNIST-style: the glyph sits in roughly a
    20x20 box centered inside the 28x28 frame, with a margin around it. To make
    inputs match that layout, we: convert to grayscale, put the strokes in
    white on black, crop to the letter, scale its longest side to 20px (keeping
    the aspect ratio), and center it in a 28x28 frame. Returns the input tensor
    of shape (1, 28, 28, 1) and the 28x28 array for display, or (None, None)
    when the image is blank.
    """
    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGBA")
        flat = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        pil_image = Image.alpha_composite(flat, pil_image)
    arr = np.array(pil_image.convert("L"), dtype="float32")

    # Training data is white strokes on a black background. If the image is
    # dark strokes on a light background, invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    if arr.max() < 30:  # nothing drawn / blank image
        return None, None

    threshold = arr.max() * 0.25
    coords = np.argwhere(arr > threshold)
    (y0, x0), (y1, x1) = coords.min(0), coords.max(0) + 1
    crop = arr[y0:y1, x0:x1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    glyph = np.array(
        Image.fromarray(crop.astype("uint8")).resize((nw, nh), Image.LANCZOS),
        dtype="float32",
    )
    frame = np.zeros((28, 28), dtype="float32")
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    frame[oy:oy + nh, ox:ox + nw] = glyph

    return (frame / 255.0).reshape(1, 28, 28, 1), frame


def predict(pil_image):
    """Return (probabilities, 28x28 frame) for an image, or (None, None)."""
    tensor, frame = preprocess_image(pil_image)
    if tensor is None:
        return None, None
    probs = load_recognition_model().predict(tensor, verbose=0)[0]
    return probs, frame


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_empty_state(mode):
    hint = {
        "draw": "Draw a letter on the pad to see what the model reads.",
        "upload": "Upload a photo or scan of a single capital letter.",
        "sample": "Pick a letter to test the model on a real sample.",
    }[mode]
    st.markdown(
        f"<div class='hs-empty'><div class='hs-ghost'>Aa</div><p>{hint}</p></div>",
        unsafe_allow_html=True,
    )


def render_prediction(probs, frame):
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    confidence = float(probs[top])
    note = (
        "Confident match"
        if confidence >= LOW_CONFIDENCE
        else "Not sure. Try writing it larger and clearer."
    )
    chips = "".join(
        f"<span class='hs-chip'><b>{LETTERS[int(i)]}</b> {probs[int(i)] * 100:.1f}%</span>"
        for i in order[1:4]
    )
    st.markdown(
        f"""
        <p class='hs-label'>Prediction</p>
        <div class='hs-result'>
            <div class='hs-letter'>{LETTERS[top]}</div>
            <div>
                <div class='hs-conf'>{confidence * 100:.1f}%</div>
                <div class='hs-conf-note'>{note}</div>
                <p class='hs-label' style='margin-top:1rem;'>Next closest</p>
                <div class='hs-alt'>{chips}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(confidence, 0.0), 1.0))

    st.markdown("<p class='hs-label' style='margin-top:1rem;'>What the model sees</p>", unsafe_allow_html=True)
    seen = Image.fromarray(frame.astype("uint8")).resize((112, 112), Image.NEAREST)
    st.image(seen, width=112)
    st.caption("Your letter, cropped, centered, and shrunk to 28 x 28 pixels.")


def render_distribution(probs):
    df = pd.DataFrame({"Letter": LETTERS, "Confidence": probs * 100})
    top_letter = LETTERS[int(np.argmax(probs))]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Letter:N", sort=LETTERS, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=False, labelFontSize=11)),
            y=alt.Y("Confidence:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.condition(
                alt.datum.Letter == top_letter, alt.value(INK), alt.value(INK_SOFT)
            ),
            tooltip=[alt.Tooltip("Letter:N"), alt.Tooltip("Confidence:Q", format=".2f")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@st.dialog("About the model", icon=":material/info:", width="medium")
def show_about():
    metrics = load_metrics()
    stats = [
        ("Test accuracy", f"{metrics.get('test_accuracy', 0.9891) * 100:.2f}%"),
        ("Parameters", f"{metrics.get('parameters', 443002):,}"),
        ("Training images", f"{DATASET_SIZE:,}"),
        ("Classes", "26 (A-Z)"),
    ]
    for row in (stats[:2], stats[2:]):
        for col, (label, value) in zip(st.columns(2), row):
            with col:
                with st.container(border=True):
                    st.metric(label, value)
    st.caption(
        "A convolutional neural network trained on the Kaggle A-Z Handwritten "
        "dataset. Built by Adossi Fred William for the CodeAlpha Machine "
        "Learning Internship."
    )


@st.dialog("How it works", icon=":material/lightbulb:", width="medium")
def show_how_it_works():
    st.markdown(
        """
1. **You give it a letter.** Draw one on the pad, upload a photo, or pick one of the samples.
2. **The app tidies it up.** It removes the color, trims away the empty space, and shrinks your letter down to a tiny 28 x 28 pixel square. That's the same size and style as the letters the model learned from.
3. **The model takes a look.** It has studied more than 370,000 handwritten letters, so it has a good idea of what each one usually looks like. It compares your letter against all 26 and decides how likely each one is.
4. **You see the answer.** The letter it's most sure about is shown in big type, together with how confident it is and the other letters it thought about.

**Tip:** for the best results, write a single capital letter, make it big, and keep it in the middle.
        """
    )


brand, actions = st.columns([3, 2], vertical_alignment="center")
with brand:
    st.markdown(
        "<h1 class='hs-brand'>HandScript <span>AI</span></h1>"
        "<p class='hs-tagline'>Write a capital letter and a neural network reads it back.</p>",
        unsafe_allow_html=True,
    )
with actions:
    with st.container(horizontal=True, horizontal_alignment="right", key="hs_actions"):
        if st.button("How it works", icon=":material/lightbulb:"):
            show_how_it_works()
        if st.button("About", icon=":material/info:"):
            show_about()

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found at `{os.path.relpath(MODEL_PATH, ROOT_DIR)}`. "
        "Train it first by running `notebook/handwritten_character_recognition.ipynb`."
    )
    st.stop()

st.session_state.setdefault("pad_id", 0)


def reset_pad():
    """Give the drawing pad a fresh key so an old sketch is not reused."""
    st.session_state.pad_id += 1


left, right = st.columns([1.05, 1], gap="medium")

with left:
    with st.container(border=True):
        mode = st.segmented_control(
            "Input",
            options=list(INPUT_MODES),
            format_func=INPUT_MODES.get,
            default=next(iter(INPUT_MODES)),
            label_visibility="collapsed",
            key="mode",
            on_change=reset_pad,
        ) or next(iter(INPUT_MODES))

        source = None
        if mode == "draw":
            pad = drawing_pad(
                key=f"pad_{st.session_state.pad_id}", on_image_change=lambda: None
            )
            source = image_from_data_url(getattr(pad, "image", None))
        elif mode == "upload":
            uploaded = st.file_uploader(
                "Upload a letter image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    source = Image.open(uploaded)
                    st.image(source, width=200)
                except Exception:
                    st.warning("That file could not be opened. Please upload a PNG or JPG image.")
        else:
            letter = st.pills(
                "Sample letter", LETTERS, default="A", key="sample_letter"
            ) or "A"
            source = load_sample(letter)
            if source is not None:
                st.image(source.resize((140, 140), Image.NEAREST), width=140)
                st.caption(f"Sample '{letter}' from the test set.")

with right:
    with st.container(border=True):
        probs = frame = None
        if source is not None:
            try:
                probs, frame = predict(source)
            except Exception:
                st.warning("This image could not be processed. Try a different one.")
        if probs is None:
            render_empty_state(mode)
        else:
            render_prediction(probs, frame)

if probs is not None:
    st.markdown("<p class='hs-label' style='margin-top:1.5rem;'>Confidence for every letter</p>", unsafe_allow_html=True)
    with st.container(border=True):
        render_distribution(probs)
