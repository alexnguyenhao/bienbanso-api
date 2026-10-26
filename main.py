"""
Biên Bản Số - Lightweight Cloud AI
- Speech-to-Text: Groq (whisper-large-v3-turbo) hoặc faster-whisper local.
- Tổng hợp biên bản: Gemini (JSON mode) với fallback không cần LLM.

Các biến môi trường quan trọng:
  ACCESS_TOKEN        Token Bearer bắt buộc (để trống = không yêu cầu)
  STT_PROVIDER        auto | groq | local (auto: Groq nếu có key)
  GROQ_API_KEY        Khóa Groq (free tier) — khuyến nghị cho Render free
  GROQ_MODEL          Mặc định whisper-large-v3 (chính xác hơn turbo)
  GROQ_FALLBACK_MODEL Mặc định whisper-large-v3-turbo (dùng khi model chính lỗi)
  GROQ_MAX_MB         Giới hạn tệp Groq free (mặc định 25 MB)
  WHISPER_MODEL       tiny | base | small | medium | large-v3 | <repo PhoWhisper>
  WHISPER_COMPUTE_TYPE int8 | int8_float16 | float16 | float32
  WHISPER_BEAM_SIZE   Mặc định 3 (1 = nhanh nhất, 5 = chính xác hơn)
  WHISPER_CONCURRENCY Số request chép lời chạy song song (mặc định 1, giữ RAM an toàn)
  MAX_UPLOAD_MB       Giới hạn dung lượng mỗi tệp gửi lên
  GEMINI_API_KEY      Khóa Google AI Studio
  GEMINI_MODEL        Mặc định gemini-2.5-flash
"""

import asyncio
import hmac
import importlib.util
import json
import os
import re
import tempfile
import threading
import time
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Biên Bản Số - Lightweight Cloud AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

# ---------------------------------------------------------------------------
# Chọn nguồn chép lời:
#   auto  = dùng Groq nếu có GROQ_API_KEY, ngược lại chạy model local
#   groq  = luôn gọi Groq (khuyến nghị cho Render free 512 MB / 0.1 CPU)
#   local = chạy faster-whisper trong container (cần nhiều RAM/CPU)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# large-v3 chính xác hơn turbo (WER 10,3% so với 12%) mà vẫn nhanh gấp ~189 lần
# thời gian thực; turbo là phương án dự phòng / nhanh hơn.
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "whisper-large-v3-turbo")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "180"))
GROQ_MAX_BYTES = max(1, int(os.getenv("GROQ_MAX_MB", "25"))) * 1024 * 1024

# Định dạng Groq/Whisper chấp nhận (kèm MIME để gửi multipart đúng chuẩn).
_AUDIO_MIME = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".flac": "audio/flac",
}

_requested_provider = os.getenv("STT_PROVIDER", "auto").strip().lower()
if _requested_provider == "auto":
    STT_PROVIDER = "groq" if GROQ_API_KEY else "local"
else:
    STT_PROVIDER = _requested_provider if _requested_provider in ("groq", "local") else "local"
if STT_PROVIDER == "groq" and not GROQ_API_KEY:
    print("CẢNH BÁO: STT_PROVIDER=groq nhưng thiếu GROQ_API_KEY → tạm dùng model local.")
    STT_PROVIDER = "local"

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "3")))
STT_CONCURRENCY = max(1, int(os.getenv("WHISPER_CONCURRENCY", "1")))
MAX_UPLOAD_BYTES = max(1, int(os.getenv("MAX_UPLOAD_MB", "25" if STT_PROVIDER == "groq" else "60"))) * 1024 * 1024

# Ngưỡng đánh dấu "nghe không rõ". Điều chỉnh được qua biến môi trường.
UNCERTAIN_LOGPROB = float(os.getenv("UNCERTAIN_LOGPROB", "-0.70"))
UNCERTAIN_NO_SPEECH = float(os.getenv("UNCERTAIN_NO_SPEECH", "0.60"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "120"))

print(f"Chế độ chép lời: {STT_PROVIDER.upper()}" + (f" ({GROQ_MODEL})" if STT_PROVIDER == "groq" else f" ({MODEL_NAME}, {COMPUTE_TYPE})"))
if STT_PROVIDER == "local" and MODEL_NAME == "tiny":
    print(
        "CẢNH BÁO: model 'tiny' có độ chính xác tiếng Việt rất thấp. "
        "Nên dùng 'small' hoặc 'medium' (hoặc PhoWhisper) nếu đủ RAM."
    )

_whisper_model = None
_model_lock = threading.Lock()


def get_local_model():
    """Tải model một lần, chỉ khi thật sự dùng chế độ local (tiết kiệm RAM trên Render)."""
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as e:
                    raise RuntimeError(
                        "Chế độ local cần faster-whisper. Chạy "
                        "'pip install -r requirements-local.txt' hoặc đặt GROQ_API_KEY."
                    ) from e

                print(f"Đang tải Faster-Whisper ({MODEL_NAME}, {COMPUTE_TYPE})...")
                _whisper_model = WhisperModel(
                    MODEL_NAME,
                    device="cpu",
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
                )
                print("-> Faster-Whisper đã sẵn sàng.")
    return _whisper_model


# Nếu chạy local, tải model trong nền để request đầu tiên không phải chờ quá lâu.
if STT_PROVIDER == "local":
    threading.Thread(target=get_local_model, daemon=True).start()

# Giới hạn số phép suy luận đồng thời để không nổ RAM trên gói miễn phí.
_stt_semaphore = asyncio.Semaphore(STT_CONCURRENCY)


def check_auth(authorization: Optional[str]):
    if ACCESS_TOKEN:
        expected = f"Bearer {ACCESS_TOKEN}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Token bảo mật không chính xác")


def clean_text(text: str) -> str:
    """Chuẩn hóa nhẹ, không làm thay đổi nội dung: gộp khoảng trắng, sửa khoảng trắng trước dấu câu."""
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def _run_transcribe(tmp_path: str, language: str, prompt: str):
    """Chạy trong threadpool: không chặn event loop của FastAPI."""
    segments_gen, info = get_local_model().transcribe(
        tmp_path,
        language=language or "vi",
        task="transcribe",
        beam_size=BEAM_SIZE,
        best_of=BEAM_SIZE,
        temperature=0.0,
        # Tránh hiện tượng lặp/hallucination khi âm thanh im lặng hoặc ồn.
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        # VAD bỏ khoảng lặng: nhanh hơn và giảm hẳn câu bịa trong im lặng.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        # Gợi ý từ khóa/tên riêng của cuộc họp (do ứng dụng gửi lên).
        initial_prompt=prompt or None,
    )

    segments = []
    for s in segments_gen:
        text = clean_text(s.text or "")
        if not text:
            continue
        avg_logprob = getattr(s, "avg_logprob", 0.0) or 0.0
        no_speech = getattr(s, "no_speech_prob", 0.0) or 0.0
        uncertain = avg_logprob < UNCERTAIN_LOGPROB or no_speech > UNCERTAIN_NO_SPEECH
        segments.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": text,
                "speaker": None,
                "uncertain": bool(uncertain),
            }
        )
    return segments, info


def _audio_suffix(path: str) -> str:
    """Đuôi tệp hợp lệ cho Groq; mặc định .m4a vì app ghi bằng expo-audio."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in _AUDIO_MIME:
        return suffix
    return ".m4a"


def _audio_mime(filename: str) -> str:
    return _AUDIO_MIME.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")


def _transcribe_groq(tmp_path: str, language: str, prompt: str):
    """
    Gửi âm thanh tới Groq (Whisper large-v3) — nhanh gấp hàng trăm lần
    so với chạy model trên 0.1 CPU của Render free, và tốn rất ít RAM.
    """
    size = os.path.getsize(tmp_path)
    if size > GROQ_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Đoạn âm thanh {size / 1024 / 1024:.1f} MB vượt giới hạn "
                f"{GROQ_MAX_BYTES // (1024 * 1024)} MB của Groq free."
            ),
        )

    # Groq (giống OpenAI) nhận dạng định dạng qua phần mở rộng của tên tệp,
    # nên giữ đúng đuôi và khai báo MIME tương ứng.
    def post(model: str) -> requests.Response:
        fields = {
            "model": (None, model),
            "language": (None, language or "vi"),
            "response_format": (None, "verbose_json"),
            "temperature": (None, "0"),
        }
        if prompt:
            fields["prompt"] = (None, prompt[:800])

        filename = f"audio{_audio_suffix(tmp_path)}"
        with open(tmp_path, "rb") as fh:
            files = {"file": (filename, fh, _audio_mime(filename)), **fields}
            return requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files=files,
                timeout=GROQ_TIMEOUT,
            )

    try:
        res = post(GROQ_MODEL)
        # Nếu model chính bị gỡ/không còn hỗ trợ, tự chuyển sang model dự phòng.
        if res.status_code in (400, 404) and GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
            print(f"Groq không dùng được {GROQ_MODEL} ({res.status_code}) → thử {GROQ_FALLBACK_MODEL}.")
            res = post(GROQ_FALLBACK_MODEL)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Groq phản hồi quá lâu. Thử lại đoạn này.")
    except requests.RequestException as e:
        print(f"Không kết nối được Groq: {e}")
        raise HTTPException(status_code=502, detail="Không kết nối được Groq. Kiểm tra mạng rồi thử lại.")

    if res.status_code == 429:
        retry_after = res.headers.get("retry-after")
        wait = f" Thử lại sau {retry_after} giây." if retry_after else " Thử lại sau ít phút."
        raise HTTPException(status_code=429, detail="Groq đang giới hạn tần suất." + wait)
    if not res.ok:
        print(f"Groq trả lỗi {res.status_code}: {res.text[:300]}")
        raise HTTPException(
            status_code=502,
            detail=f"Groq trả lỗi ({res.status_code}). Kiểm tra GROQ_API_KEY và hạn mức free tier.",
        )

    data = res.json()
    raw_segments = data.get("segments")
    segments = []
    if isinstance(raw_segments, list):
        for s in raw_segments:
            text = clean_text(str(s.get("text") or ""))
            if not text:
                continue
            avg_logprob = float(s.get("avg_logprob") or 0.0)
            no_speech = float(s.get("no_speech_prob") or 0.0)
            segments.append(
                {
                    "start": float(s.get("start") or 0.0),
                    "end": float(s.get("end") or 0.0),
                    "text": text,
                    "speaker": None,
                    "uncertain": bool(avg_logprob < UNCERTAIN_LOGPROB or no_speech > UNCERTAIN_NO_SPEECH),
                }
            )
    if not segments:
        text = clean_text(str(data.get("text") or ""))
        if text:
            segments.append(
                {
                    "start": 0.0,
                    "end": float(data.get("duration") or 1.0),
                    "text": text,
                    "speaker": None,
                    "uncertain": False,
                }
            )
    info = {
        "language": data.get("language", language or "vi"),
        "duration": float(data.get("duration") or 0.0),
    }
    return segments, info


def _info_get(info, key: str, default=None):
    if isinstance(info, dict):
        return info.get(key, default)
    return getattr(info, key, default)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Bien Ban So Lightweight Cloud AI",
        "provider": STT_PROVIDER,
        "model": GROQ_MODEL if STT_PROVIDER == "groq" else MODEL_NAME,
        "fallback_model": GROQ_FALLBACK_MODEL if STT_PROVIDER == "groq" else None,
        "whisper_model": MODEL_NAME,
        "groq_configured": bool(GROQ_API_KEY),
        "engine": "groq-whisper" if STT_PROVIDER == "groq" else "faster-whisper-int8",
        "beam_size": BEAM_SIZE,
        "vad_filter": STT_PROVIDER == "local",
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "local_stt_installed": importlib.util.find_spec("faster_whisper") is not None,
        "auth_required": bool(ACCESS_TOKEN),
        "gemini_model": GEMINI_MODEL if GEMINI_API_KEY else None,
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    prompt: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)

    suffix = os.path.splitext(file.filename or "")[1] or ".m4a"
    tmp_path = ""
    started = time.monotonic()
    size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Tệp vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Hãy chia nhỏ bản ghi.",
                    )
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Tệp âm thanh rỗng.")

        async with _stt_semaphore:
            runner = _transcribe_groq if STT_PROVIDER == "groq" else _run_transcribe
            segments, info = await run_in_threadpool(runner, tmp_path, language, prompt)

        elapsed = time.monotonic() - started
        print(
            f"[stt:{STT_PROVIDER}] {size / 1024 / 1024:.1f} MB, {elapsed:.1f}s, "
            f"{len(segments)} đoạn, ngôn ngữ={_info_get(info, 'language', language)}"
        )
        return {
            "segments": segments,
            "language": _info_get(info, "language", language),
            "duration": float(_info_get(info, "duration", 0.0) or 0.0),
            "provider": STT_PROVIDER,
            "elapsed_seconds": round(elapsed, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Lỗi chép lời: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi chép lời: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


class DraftRequest(BaseModel):
    system: str
    user: str
    language: Optional[str] = "vi"


def _call_gemini(req: DraftRequest) -> Optional[str]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    generation_config = {
        "temperature": 0.2,
        "maxOutputTokens": 8192,
        "responseMimeType": "application/json",
        "thinkingConfig": {"thinkingBudget": GEMINI_THINKING_BUDGET},
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{req.system}\n\n{req.user}"}]}],
        "generationConfig": generation_config,
    }
    res = requests.post(url, json=payload, timeout=GEMINI_TIMEOUT)
    # Một số model (vd. bản Pro) không cho tắt "thinking" → thử lại không kèm thinkingConfig.
    if res.status_code == 400 and "thinkingConfig" in payload["generationConfig"]:
        payload["generationConfig"].pop("thinkingConfig", None)
        res = requests.post(url, json=payload, timeout=GEMINI_TIMEOUT)
    if not res.ok:
        print(f"Gemini trả lỗi {res.status_code}: {res.text[:300]}")
        return None
    data = res.json()
    candidates = data.get("candidates") or []
    if not candidates:
        print(f"Gemini không trả về candidates: {str(data)[:300]}")
        return None
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or None


@app.post("/draft")
async def generate_draft(
    req: DraftRequest,
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)

    if GEMINI_API_KEY:
        try:
            text = await run_in_threadpool(_call_gemini, req)
            if text:
                return {"text": text, "engine": GEMINI_MODEL}
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
    return {
        "text": json.dumps({"blocks": fallback_blocks}, ensure_ascii=False),
        "engine": "fallback",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
