import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Fashion CNN Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CLASS_EMOJI = ["👕", "👖", "👚", "👗", "🧥", "🩴", "👔", "👟", "👜", "👢"]
MODEL_CANDIDATES = ["fashion_cnn.keras", os.path.join("models", "fashion_cnn.keras"), "fashion_cnn.h5"]
EXAMPLE_DIR = "samples"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; max-width: 1100px;}
      h1 {letter-spacing: -0.5px;}
      .pred-card {
          background: #ffffff;
          border: 1px solid #e6e9ef;
          border-left: 6px solid #6c5ce7;
          border-radius: 12px;
          padding: 18px 22px;
          box-shadow: 0 1px 4px rgba(0,0,0,.06);
          margin-bottom: 12px;
      }
      .pred-label {font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; color: #8b8fa3;}
      .pred-class  {font-size: 2rem; font-weight: 700; line-height: 1.3; color: #1e1e2f;}
      .pred-conf   {font-size: 1.05rem; color: #5d6070;}
      .img-caption {font-size: .85rem; color: #8b8fa3; margin-top: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


def find_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path)


def preprocess_image(img):
    im = ImageOps.exif_transpose(img).convert("L")
    a = np.array(im)
    bg = np.median(a)
    mask = np.abs(a.astype(int) - int(bg)) > 25
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) and len(cols):
        im = im.crop((int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))
    side = max(im.size)
    sq = Image.new("L", (side, side), int(bg))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    sq = sq.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(sq).astype("float32") / 255.0
    if arr[0].mean() > 0.5:
        arr = 1.0 - arr
    return arr[np.newaxis, ..., np.newaxis]


def confidence_color(conf):
    if conf >= 0.85:
        return "#00b894"
    if conf >= 0.55:
        return "#fdcb6e"
    return "#e17055"


def top3_chart(probs):
    top = np.argsort(probs)[::-1][:3]
    df = pd.DataFrame({
        "Class": [CLASS_NAMES[i] for i in top],
        "Probability": probs[top],
    })
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Class:N", sort="-x", title=None),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.Color("Class:N", scale=alt.Scale(scheme="purpleorange"), legend=None),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".2%")],
    )
    labels = alt.Chart(df).mark_text(align="left", dx=6, color="#3a3a4a").encode(
        y=alt.Y("Class:N", sort="-x"),
        x="Probability:Q",
        text=alt.Text("Probability:Q", format=".1%"),
    )
    return (bars + labels).properties(height=150)


def example_images():
    if not os.path.isdir(EXAMPLE_DIR):
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return sorted(
        os.path.join(EXAMPLE_DIR, f) for f in os.listdir(EXAMPLE_DIR) if f.lower().endswith(exts)
    )


model_path = find_model()
model = None
setup_problems = []
if not TF_AVAILABLE:
    setup_problems.append(
        "TensorFlow is not installed in this Python environment — you're probably running "
        "the system/Homebrew Python. Activate the project venv first: "
        "`source .venv/bin/activate`, then `streamlit run app.py`."
    )
if model_path is None:
    setup_problems.append(
        "No saved model found. Open `fashion_mnist_cnn.ipynb`, run it through training, "
        "then run the **Save the trained model** cell to create `fashion_cnn.keras`."
    )

with st.sidebar:
    st.title("👕 Fashion CNN")
    st.caption("Image classifier trained on Fashion MNIST")
    st.markdown(
        """
        **How to use**
        1. Upload a clothing photo (or pick an example)
        2. See the predicted class and confidence
        3. Compare the top-3 class probabilities

        **Tips for best results**
        - One item per photo, centred
        - Plain, uncluttered background
        - The photo is cropped, squared and
          converted to 28×28 automatically
        """
    )
    st.divider()
    st.markdown("**Classes**")
    for emoji, name in zip(CLASS_EMOJI, CLASS_NAMES):
        st.markdown(f"{emoji} {name}")

st.title("👕 Fashion CNN Classifier")
st.caption(
    "Upload a photo of clothing — the CNN predicts the class, its probability, "
    "and the top-3 class distribution."
)

if setup_problems:
    st.error("**Setup needed before predicting**")
    for p in setup_problems:
        st.markdown(f"- {p}")
    st.info(
        "Quick start: `source .venv/bin/activate` → `streamlit run app.py` "
        "(first time: `python3 -m venv .venv && source .venv/bin/activate && "
        "pip install -r requirements.txt`)"
    )
    st.stop()

model = load_model(model_path) if (TF_AVAILABLE and model_path) else None

col_left, col_right = st.columns([1, 1.15], gap="large")

with col_left:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    examples = example_images()
    example = None
    if examples and uploaded is None:
        choice = st.selectbox(
            "…or try an example", [""] + [os.path.basename(p) for p in examples]
        )
        if choice:
            example = os.path.join(EXAMPLE_DIR, choice)

    source = uploaded if uploaded is not None else example
    image = None
    if source is not None:
        try:
            image = Image.open(source)
        except Exception:
            st.error("That file could not be read as an image.")
    if image is None:
        st.info("Upload an image or choose an example to classify it.")

with col_right:
    st.subheader("Prediction")
    if image is None:
        st.write("The prediction card, probability and chart will appear here.")

if image is not None:
    x = preprocess_image(image)
    with st.spinner("Classifying..."):
        probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    st.divider()
    col_img, col_pred = st.columns([1, 1.15], gap="large")

    with col_img:
        st.markdown("**Original**")
        st.image(image, width=280)
        st.markdown("**Model input (28×28, inverted to match training style)**")
        st.image(np.squeeze(x), width=280, clamp=True)

    with col_pred:
        color = confidence_color(conf)
        st.markdown(
            f"""
            <div class="pred-card" style="border-left-color: {color};">
              <div class="pred-label">Predicted class</div>
              <div class="pred-class">{CLASS_EMOJI[pred]} {CLASS_NAMES[pred]}</div>
              <div class="pred-conf">Confidence: <b>{conf:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Top-3 classes**")
        st.altair_chart(top3_chart(probs), use_container_width=True)

        with st.expander("All class probabilities"):
            table = pd.DataFrame({
                "Class": [f"{e}  {n}" for e, n in zip(CLASS_EMOJI, CLASS_NAMES)],
                "Probability": [f"{p:.2%}" for p in probs],
            }).sort_values("Probability", ascending=False, key=lambda s: s.str.rstrip("%").astype(float))
            st.dataframe(table, hide_index=True, use_container_width=True)
