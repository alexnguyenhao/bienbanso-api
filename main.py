import os
import shutil
import tempfile
import json
from typing import Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import requests

app = FastAPI(title="Biên Bản Số - Lightweight Cloud AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Sử dụng faster-whisper tối ưu int8: chạy cực nhanh và chỉ tốn ~150MB RAM (hoàn toàn an toàn trong 512MB của Render)
print(f"Đang tải mô hình Faster-Whisper ({MODEL_NAME}, int8)...")
whisper_model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
print("-> Mô hình Faster-Whisper đã sẵn sàng!")

def check_auth(authorization: Optional[str]):
    if ACCESS_TOKEN:
        if not authorization or authorization != f"Bearer {ACCESS_TOKEN}":
            raise HTTPException(status_code=401, detail="Token bảo mật không chính xác")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Bien Ban So Lightweight Cloud AI",
        "whisper_model": MODEL_NAME,
        "engine": "faster-whisper-int8",
        "auth_required": bool(ACCESS_TOKEN),
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)

    suffix = os.path.splitext(file.filename or "")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Faster-whisper phiên bản tối ưu int8
        segments_gen, _ = whisper_model.transcribe(
            tmp_path,
            language="vi",
            beam_size=1,
            best_of=1,
            temperature=0.0,
        )

        segments = []
        for s in segments_gen:
            text = s.text.strip()
            if text:
                segments.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": text,
                    "speaker": None,
                    "uncertain": False,
                })

        return {"segments": segments}
    except Exception as e:
        print(f"Lỗi chép lời: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class DraftRequest(BaseModel):
    system: str
    user: str
    language: Optional[str] = "vi"

@app.post("/draft")
async def generate_draft(
    req: DraftRequest,
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)

    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{req.system}\n\n{req.user}"}]}
                ],
                "generationConfig": {"temperature": 0.2},
            }
            res = requests.post(url, json=payload, timeout=60)
            if res.ok:
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"text": text}
        except Exception as e:
            print(f"Lỗi Gemini: {e}")

    # Fallback tự động không cần LLM
    fallback_blocks = [
        {
            "kind": "heading",
            "text": "I. NỘI DUNG VÀ Ý KIẾN PHÁT BIỂU",
            "sources": [],
            "uncertain": False,
        },
        {
            "kind": "paragraph",
            "text": "Nội dung trích xuất tự động từ bản chép lời cuộc họp. Vui lòng rà soát và điều chỉnh chi tiết.",
            "sources": [],
            "uncertain": True,
            "note": "Cần rà soát",
        },
        {
            "kind": "heading",
            "text": "II. KẾT LUẬN CỦA CHỦ TRÌ",
            "sources": [],
            "uncertain": False,
        },
        {
            "kind": "conclusion",
            "text": "Chủ trì kết luận các nội dung trọng tâm và yêu cầu các đơn vị phối hợp triển khai thực hiện [Cần bổ sung].",
            "sources": [],
            "uncertain": True,
            "note": "Cần bổ sung",
        },
    ]
    return {"text": json.dumps({"blocks": fallback_blocks}, ensure_ascii=False)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
