"""
Metal Defect Detection — Streamlit App

Loads a trained ResNet18 checkpoint (saved by train_model.py) and lets users
upload one or many images to classify them into one of six NEU-DET surface
defect categories: crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches.


Expects a checkpoint file named 'defect_model.pth' in the same folder as
this script, saved as a dict with keys: model_state_dict, classes,
num_classes. If the file is missing, the app still launches in DEMO MODE
so you can see and test the interface (predictions in demo mode are
random and clearly labeled as such — they are NOT real detections).
"""

import io
import os
import random
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

st.set_page_config(
    page_title="Metal Detection Maintenance",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "defect_model.pth")
IMG_SIZE = 128
DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}
UNCERTAIN_THRESHOLD = 0.85

# Theme — blue, injected once via CSS
st.markdown(
    """
    <style>
        :root {
            --navy: #0B2545;
            --deep-blue: #13315C;
            --accent-blue: #1E6FD9;
            --bright-blue: #3B8FF3;
            --pale-blue: #EAF2FC;
            --steel: #8FA8C7;
        }

        .stApp {
            background: linear-gradient(180deg, #0B2545 0%, #13315C 320px, #F4F8FD 320px, #F4F8FD 100%);
        }

        /* Header banner */
        .app-header {
            padding: 2.2rem 1rem 1.6rem 1rem;
            text-align: center;
        }
        .app-header h1 {
            color: #FFFFFF;
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }
        .app-header p {
            color: var(--pale-blue);
            font-size: 1.05rem;
            opacity: 0.85;
            margin-top: 0;
        }

        /* Card container look for main content blocks */
        .blue-card {
            background: #FFFFFF;
            border: 1px solid #D7E4F5;
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 4px 18px rgba(11, 37, 69, 0.08);
            margin-bottom: 1.2rem;
        }

        /* Big bold result banner */
        .defect-result {
            text-align: center;
            padding: 2.2rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--deep-blue), var(--accent-blue));
            box-shadow: 0 6px 24px rgba(19, 49, 92, 0.25);
            margin: 1.2rem 0;
        }
        .defect-result .label {
            color: var(--pale-blue);
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .defect-result .defect-name {
            color: #FFFFFF;
            font-size: 3rem;
            font-weight: 900;
            letter-spacing: 1px;
            margin: 0.2rem 0;
            line-height: 1.1;
        }
        .defect-result .confidence {
            color: var(--pale-blue);
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }
        .defect-result.uncertain {
            background: linear-gradient(135deg, #5A4A0E, #B8860B);
        }

        /* Section headers */
        .section-title {
            color: var(--navy);
            font-weight: 700;
            font-size: 1.3rem;
            margin-bottom: 0.6rem;
            border-left: 5px solid var(--accent-blue);
            padding-left: 0.6rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: var(--navy);
        }
        section[data-testid="stSidebar"] * {
            color: #EAF2FC !important;
        }

        /* Buttons */
        .stButton button, .stDownloadButton button {
            background-color: var(--accent-blue);
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background-color: var(--bright-blue);
        }

        /* Demo mode banner */
        .demo-banner {
            background: #FFF4E5;
            border: 1px solid #F0B860;
            color: #7A4B00;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Model definition (must match training script architecture)
class DefectModel(nn.Module):
    def __init__(self, num_classes):
        super(DefectModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads the trained checkpoint if present. Returns (model, classes, demo_mode).
    If no checkpoint is found, falls back to an untrained model in demo mode
    so the UI remains fully testable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        model = DefectModel(num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, classes, device, False

    # Demo / fallback mode — no checkpoint available yet
    model = DefectModel(num_classes=len(DEFAULT_CLASSES))
    model.to(device)
    model.eval()
    return model, DEFAULT_CLASSES, device, True


model, CLASSES, DEVICE, DEMO_MODE = load_model()

inference_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(img: Image.Image):
    """Run inference on a single PIL image. Returns (class_name, confidence, status)."""
    if DEMO_MODE:
        # Clearly-labeled placeholder behavior — no checkpoint loaded yet.
        cls = random.choice(CLASSES)
        confidence = random.uniform(0.55, 0.99)
    else:
        img_tensor = inference_transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            cls = CLASSES[predicted_idx.item()]

    status = "UNCERTAIN" if confidence < UNCERTAIN_THRESHOLD else "OK"
    return cls, confidence, status



# Session state for accumulated results (so the CSV can include every
# image detected so far in this session, not just the latest batch)
if "results" not in st.session_state:
    st.session_state.results = []  # list of dicts


# Header
st.markdown(
    """
    <div class="app-header">
        <h1>🔩 Metal Detection Maintenance</h1>
        <p>Upload surface images to automatically detect crazing, inclusion, patches,
        pitted surfaces, rolled-in scale, or scratches.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if DEMO_MODE:
    st.markdown(
        f"""
        <div class="demo-banner">
        ⚠️ <strong>Demo mode:</strong> no trained checkpoint found at
        <code>{os.path.basename(CHECKPOINT_PATH)}</code>. Predictions shown below are
        random placeholders so you can preview the interface. Run your training script
        (saving to <code>defect_model.pth</code>) and place it next to this app to get
        real predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Top-right download button placeholder — filled in after detection runs
top_left, top_right = st.columns([4, 1])
download_slot = top_right.empty()

# Sidebar — settings
with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider(
        "Confidence threshold for 'uncertain' flag",
        min_value=0.50,
        max_value=0.99,
        value=UNCERTAIN_THRESHOLD,
        step=0.01,
    )
    st.markdown("---")
    st.markdown("### Defect classes")
    for c in CLASSES:
        st.markdown(f"- {DISPLAY_NAMES.get(c, c)}")
    st.markdown("---")
    if st.session_state.results:
        if st.button("Clear session results"):
            st.session_state.results = []
            st.rerun()

# Upload section
st.markdown('<div class="section-title">Upload images</div>', unsafe_allow_html=True)
st.markdown('<div class="blue-card">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload one image or several at once",
    type=["jpg", "jpeg", "png", "jfif", "bmp"],
    accept_multiple_files=True,
)

run_detection = st.button("Run detection", type="primary", disabled=not uploaded_files)

st.markdown("</div>", unsafe_allow_html=True)

# Run detection
if run_detection and uploaded_files:
    new_results = []
    with st.spinner(f"Analyzing {len(uploaded_files)} image(s)..."):
        for uploaded_file in uploaded_files:
            img_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(img_bytes))
            cls, confidence, status = predict(img)
            if confidence < threshold:
                status = "UNCERTAIN"
            new_results.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": uploaded_file.name,
                    "predicted_class": cls,
                    "predicted_class_display": DISPLAY_NAMES.get(cls, cls),
                    "confidence": round(confidence, 4),
                    "status": status,
                    "_thumb": img,
                }
            )
    st.session_state.results.extend(new_results)
    st.session_state.last_batch = new_results

# Show results for the most recent batch
if st.session_state.get("last_batch"):
    last_batch = st.session_state.last_batch

    st.markdown('<div class="section-title">Detection result</div>', unsafe_allow_html=True)

    if len(last_batch) == 1:
        r = last_batch[0]
        css_class = "defect-result uncertain" if r["status"] == "UNCERTAIN" else "defect-result"
        col_img, col_result = st.columns([1, 2])
        with col_img:
            st.image(r["_thumb"], caption=r["image_name"], width="stretch")
        with col_result:
            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="label">{'Needs human review' if r['status'] == 'UNCERTAIN' else 'Defect detected'}</div>
                    <div class="defect-name">{r['predicted_class_display'].upper()}</div>
                    <div class="confidence">Confidence: {r['confidence']*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f"Analyzed **{len(last_batch)}** images:")
        cols_per_row = 3
        for i in range(0, len(last_batch), cols_per_row):
            row = last_batch[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, row):
                css_class = "defect-result uncertain" if r["status"] == "UNCERTAIN" else "defect-result"
                with col:
                    st.image(r["_thumb"], caption=r["image_name"], width="stretch")
                    st.markdown(
                        f"""
                        <div class="{css_class}" style="padding: 1.2rem 0.6rem;">
                            <div class="label" style="font-size:0.75rem;">{'Review needed' if r['status'] == 'UNCERTAIN' else 'Detected'}</div>
                            <div class="defect-name" style="font-size:1.6rem;">{r['predicted_class_display'].upper()}</div>
                            <div class="confidence" style="font-size:0.9rem;">{r['confidence']*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# Top-right CSV download (renders once any results exist this session)
if st.session_state.results:
    df = pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"],
                "image_name": r["image_name"],
                "predicted_class": r["predicted_class_display"],
                "confidence": r["confidence"],
                "status": r["status"],
            }
            for r in st.session_state.results
        ]
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    with download_slot:
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name=f"defect_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download a CSV log of every image detected this session",
        )

# Full session history table at the bottom
if st.session_state.results:
    st.markdown('<div class="section-title">Session history</div>', unsafe_allow_html=True)
    st.markdown('<div class="blue-card">', unsafe_allow_html=True)
    history_df = pd.DataFrame(
        [
            {
                "Timestamp": r["timestamp"],
                "Image": r["image_name"],
                "Defect": r["predicted_class_display"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Status": r["status"],
            }
            for r in st.session_state.results
        ]
    )
    st.dataframe(history_df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)