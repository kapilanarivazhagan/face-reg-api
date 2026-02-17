# ======================================================
# FACE VERIFICATION + LIVENESS — API CORE (NO UI)
# ======================================================

import time
import torch
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------- CONFIG ----------------
BASE_THRESHOLD = 0.75
LIGHTING_MARGIN = 0.05

MODEL_VERSION = "facenet_v1_liveness"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS (ONCE) ----------------
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    select_largest=True,
    keep_all=False,
    device=device
)

model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

# ---------------- PREPROCESS VARIANTS ----------------
def gamma_correct(img, gamma):
    table = np.array(
        [(i / 255.0) ** (1 / gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)

def generate_variants(img):
    return {
        "original": img,
        "brightened": gamma_correct(img, 0.7),
        "darkened": gamma_correct(img, 1.3)
    }

# ---------------- FACE EMBEDDING ----------------
def extract_single(img_np):
    img = Image.fromarray(img_np)
    face = mtcnn(img)

    if face is None:
        return None

    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(device))[0].cpu().numpy()

    return emb / np.linalg.norm(emb)

def extract_multi(img_np):
    variants = generate_variants(img_np)
    embeddings = {}

    for name, v in variants.items():
        emb = extract_single(v)
        if emb is not None:
            embeddings[name] = emb

    return embeddings

# ---------------- SIMILARITY ----------------
def cosine_sim(a, b):
    return float(np.dot(a, b))

# ======================================================
# 🔥 LIVENESS ANALYSIS (LIGHTWEIGHT & EXPLAINABLE)
# ======================================================

def blur_score(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def lighting_stats(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return np.mean(gray), np.std(gray)

def evaluate_liveness(img_np, embedding_spread):
    """
    Returns:
      liveness_score (0–1)
      liveness_flags (list)
    """

    flags = []
    score = 1.0

    blur = blur_score(img_np)
    mean_light, std_light = lighting_stats(img_np)

    # ---- Anti-photo / anti-screen ----
    if blur < 80:
        flags.append("LOW_SHARPNESS")
        score *= 0.7

    # ---- Flat lighting (screen glow) ----
    if std_light < 35:
        flags.append("FLAT_LIGHTING")
        score *= 0.75

    # ---- Embedding instability ----
    if embedding_spread > 0.06:
        flags.append("EMBEDDING_INCONSISTENT")
        score *= 0.8

    return round(score, 2), flags

# ======================================================
# 🚀 MAIN VERIFICATION FUNCTION (API CORE)
# ======================================================

def verify_face(onboard_img_bytes, live_img_bytes):
    start_time = time.time()

    try:
        from io import BytesIO

        onboard_bytes = onboard_img_bytes.read()
        live_bytes = live_img_bytes.read()

        onboard_np = np.array(Image.open(onboard_img_bytes).convert("RGB"))
        live_np    = np.array(Image.open(live_img_bytes).convert("RGB"))

        data1 = extract_multi(onboard_np)
        data2 = extract_multi(live_np)

        if not data1 or not data2:
            return {
                "matched": False,
                "risk_level": "BLOCK",
                "raw_score": 0.0,
                "liveness_score": 0.0,
                "liveness_flags": ["FACE_NOT_DETECTED"],
                "decision_reason": "FACE_NOT_DETECTED",
                "model_version": MODEL_VERSION,
                "processing_time_sec": round(time.time() - start_time, 2)
            }

        # ---- Similarity matrix ----
        results = []
        for e1 in data1.values():
            for e2 in data2.values():
                results.append(cosine_sim(e1, e2))

        raw_score = max(results)
        spread = max(results) - min(results)

        # ---------------- LIVENESS ----------------
        liveness_score, liveness_flags = evaluate_liveness(live_np, spread)

        # ---------------- DECISION ----------------
        if liveness_score < 0.6:
            matched = False
            decision_reason = "LIVENESS_FAILED"
            risk_level = "BLOCK"

        elif raw_score >= BASE_THRESHOLD:
            matched = True
            decision_reason = "CLEAR_MATCH"
            risk_level = "LOW"

        elif raw_score >= BASE_THRESHOLD - LIGHTING_MARGIN:
            matched = True
            decision_reason = "LIGHTING_COMPENSATED"
            risk_level = "MEDIUM"

        elif spread > 0.05:
            matched = False
            decision_reason = "UNSTABLE_LIGHTING"
            risk_level = "HIGH"

        else:
            matched = False
            decision_reason = "FACE_MISMATCH"
            risk_level = "BLOCK"

        return {
            "matched": matched,
            "risk_level": risk_level,
            "raw_score": round(raw_score, 4),
            "liveness_score": liveness_score,
            "liveness_flags": liveness_flags,
            "decision_reason": decision_reason,
            "model_version": MODEL_VERSION,
            "processing_time_sec": round(time.time() - start_time, 2)
        }

    except Exception as e:
        return {
            "matched": False,
            "risk_level": "ERROR",
            "raw_score": 0.0,
            "liveness_score": 0.0,
            "liveness_flags": ["INTERNAL_ERROR"],
            "decision_reason": "INTERNAL_ERROR",
            "error": str(e),
            "model_version": MODEL_VERSION,
            "processing_time_sec": round(time.time() - start_time, 2)
        }

# ======================================================
# LOCAL TEST (REMOVE IN PRODUCTION)
# ======================================================

if __name__ == "__main__":
    onboard_path = r"C:\path\to\onboard.jpg"
    live_path    = r"C:\path\to\live.jpg"

    with open(onboard_path, "rb") as f1, open(live_path, "rb") as f2:
        result = verify_face(f1, f2)

    print("\n================ FACE VERIFICATION RESULT ================\n")
    for k, v in result.items():
        print(f"{k:20}: {v}")
    print("\n==========================================================\n")
