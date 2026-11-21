"""
Malaria Cell Detection — Streamlit prototype
Upload a thin-blood-smear cell image and the MobileNetV2 transfer-learning
model predicts whether the cell is Parasitized or Uninfected, with a Grad-CAM
heatmap showing which region drove the decision.
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "malaria_mobilenet_model.keras"
SAMPLES_DIR = ROOT / "samples"
IMG_SIZE = (128, 128)

# Keras flow_from_directory indexes classes alphabetically:
#   0 = Parasitized, 1 = Uninfected
# so the sigmoid output is P(Uninfected).
CLASS_NAMES = {0: "Parasitized", 1: "Uninfected"}


# ------------------------------------------------------------------
# Model loading & inference
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Split the model into the MobileNetV2 backbone and the classifier head
    # so Grad-CAM can take gradients w.r.t. the backbone's last feature map.
    base_idx = next(
        i for i, layer in enumerate(model.layers)
        if isinstance(layer, tf.keras.Model)
    )
    base = model.layers[base_idx]
    head = model.layers[base_idx + 1:]
    return model, base, head


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_with_gradcam(base, head, x: np.ndarray):
    """Returns P(Uninfected) and a Grad-CAM heatmap (H×W in [0, 1])."""
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        feature_map = base(x, training=False)
        tape.watch(feature_map)
        y = feature_map
        for layer in head:
            y = layer(y, training=False)
        p_uninfected = y[:, 0]
        # Explain whichever class was predicted
        score = tf.where(p_uninfected >= 0.5, p_uninfected, 1.0 - p_uninfected)

    grads = tape.gradient(score, feature_map)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(feature_map[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    return float(p_uninfected[0]), cam.numpy()


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha=0.45):
    base_img = np.asarray(img.convert("RGB").resize((256, 256)))
    hm = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_CUBIC)
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    hm = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_img, 1 - alpha, hm, alpha, 0)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Malaria Cell Detection", page_icon="🔬", layout="centered")

st.title("🔬 Malaria Cell Detection")
st.write(
    "Upload a microscope image of a single red blood cell. A **MobileNetV2** "
    "transfer-learning model classifies it as **Parasitized** or **Uninfected**, "
    "and **Grad-CAM** highlights the region the model focused on."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold for 'Uninfected'",
        min_value=0.10, max_value=0.95, value=0.83, step=0.01,
        help="The model outputs P(Uninfected). Above this value the cell is "
             "called Uninfected; otherwise Parasitized. Raising it makes the "
             "model flag more cells as Parasitized (higher sensitivity). "
             "Default 0.83 is the F1-optimal threshold found on the test set.",
    )
    st.markdown("---")
    st.markdown(
        "**Model:** MobileNetV2 (ImageNet) fine-tuned on the NIH Malaria "
        "Cell Images dataset (27,558 images).\n\n"
        "⚠️ Research prototype — not a medical device."
    )

model, base, head = load_model()

samples = sorted(SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
tab_upload, tab_sample = st.tabs(["📤 Upload image", "🧪 Try a sample"])

image = None
with tab_upload:
    uploaded = st.file_uploader("Cell image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded is not None:
        image = Image.open(uploaded)
with tab_sample:
    if samples:
        choice = st.selectbox("Sample image", [p.name for p in samples])
        if st.button("Analyze sample", type="primary"):
            st.session_state["sample"] = choice
        if uploaded is None and st.session_state.get("sample"):
            image = Image.open(SAMPLES_DIR / st.session_state["sample"])
    else:
        st.info("No sample images found in the `samples/` folder.")

if image is not None:
    p_uninfected, heatmap = predict_with_gradcam(base, head, preprocess(image))
    is_uninfected = p_uninfected > threshold
    label = CLASS_NAMES[int(is_uninfected)]
    confidence = p_uninfected if is_uninfected else 1 - p_uninfected

    st.markdown("---")
    if is_uninfected:
        st.success(f"### Prediction: {label}")
    else:
        st.error(f"### Prediction: {label}")

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{confidence * 100:.2f}%")
    c2.metric("P(Parasitized)", f"{(1 - p_uninfected) * 100:.2f}%")
    st.progress(float(1 - p_uninfected), text="Parasitized probability")

    col_a, col_b = st.columns(2)
    col_a.image(image.convert("RGB").resize((256, 256)), caption="Input image", width="stretch")
    col_b.image(overlay_heatmap(image, heatmap), caption="Grad-CAM overlay", width="stretch")
    st.caption(
        "Grad-CAM: red/yellow regions contributed most to the prediction. "
        "For parasitized cells it should light up the stained parasite."
    )
else:
    st.info("Upload an image or pick a sample to get a prediction.")
