from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
import os

from src.verification import verify_face

# ======================================================
# 🔐 API KEY CONFIG (ENV-FIRST, CLOUD SAFE)
# ======================================================

API_KEY = os.getenv("FACE_API_KEY")
API_KEY_HEADER = "X-API-Key"

def verify_api_key(x_api_key: str = Header(...)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ======================================================
# 🚀 FASTAPI APP
# ======================================================

app = FastAPI(
    title="Face Verification API",
    description="Verify live driver face against onboarded image with liveness detection",
    version="1.0.0"
)

# ======================================================
# 🩺 HEALTH CHECK (PUBLIC)
# ======================================================

@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "face-verification"
    }

# ======================================================
# 🔍 FACE VERIFICATION (SECURED)
# ======================================================

@app.post(
    "/verify-face",
    tags=["Face Verification"],
    summary="Verify onboarded face vs live face"
)
async def verify_face_api(
    onboarded_image: UploadFile = File(...),
    live_image: UploadFile = File(...),
    _: None = Depends(verify_api_key)
):
    try:
        result = verify_face(
            onboard_file=onboarded_image.file,
            live_file=live_image.file
        )
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "matched": False,
                "risk_level": "ERROR",
                "decision_reason": "INTERNAL_ERROR",
                "error": str(e)
            }
        )

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
