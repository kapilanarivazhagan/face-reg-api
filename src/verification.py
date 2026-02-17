import time
import numpy as np
from PIL import Image

from src.embedding import extract_multi
from src.similarity import cosine_sim
from src.liveness import evaluate_liveness
from src.config import BASE_THRESHOLD, LIGHTING_MARGIN, MODEL_VERSION

def verify_face(onboard_file, live_file):
    start = time.time()

    try:
        onboard_np = np.array(Image.open(onboard_file).convert("RGB"))
        live_np = np.array(Image.open(live_file).convert("RGB"))

        data1 = extract_multi(onboard_np)
        data2 = extract_multi(live_np)

        if not data1 or not data2:
            return _fail("FACE_NOT_DETECTED", start)

        scores = [
            cosine_sim(e1, e2)
            for e1 in data1.values()
            for e2 in data2.values()
        ]

        raw = max(scores)
        spread = max(scores) - min(scores)

        live_score, live_flags = evaluate_liveness(live_np, spread)

        if live_score < 0.6:
            return _decision(False, "BLOCK", raw, live_score, live_flags, "LIVENESS_FAILED", start)

        if raw >= BASE_THRESHOLD:
            return _decision(True, "LOW", raw, live_score, live_flags, "CLEAR_MATCH", start)

        if raw >= BASE_THRESHOLD - LIGHTING_MARGIN:
            return _decision(True, "MEDIUM", raw, live_score, live_flags, "LIGHTING_COMPENSATED", start)

        if spread > 0.05:
            return _decision(False, "HIGH", raw, live_score, live_flags, "UNSTABLE_LIGHTING", start)

        return _decision(False, "BLOCK", raw, live_score, live_flags, "FACE_MISMATCH", start)

    except Exception as e:
        return {
            "matched": False,
            "risk_level": "ERROR",
            "error": str(e),
            "model_version": MODEL_VERSION,
            "processing_time_sec": round(time.time() - start, 2),
        }

def _fail(reason, start):
    return {
        "matched": False,
        "risk_level": "BLOCK",
        "decision_reason": reason,
        "model_version": MODEL_VERSION,
        "processing_time_sec": round(time.time() - start, 2),
    }

def _decision(matched, risk, raw, live, flags, reason, start):
    return {
        "matched": matched,
        "risk_level": risk,
        "raw_score": round(raw, 4),
        "liveness_score": live,
        "liveness_flags": flags,
        "decision_reason": reason,
        "model_version": MODEL_VERSION,
        "processing_time_sec": round(time.time() - start, 2),
    }
