"""FastAPI application for document upload and grounded questions."""
from pathlib import Path
from uuid import uuid4
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from retriever.ingest import ingest_file
from orchestrator import run_workflow
from memory_cache import (get_history, remember, cache_key, get_cached,
                          put_cached, invalidate_cache, clear_session)

app = FastAPI(title="Week 7 - Document Chat", version="1.0.0")
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
ALLOWED = {".pdf", ".docx", ".txt"}
MAX_UPLOAD = 20 * 1024 * 1024

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    original = Path(file.filename or "").name
    if Path(original).suffix.lower() not in ALLOWED:
        raise HTTPException(400, "Only PDF, DOCX and TXT files are supported")
    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / original
    temp = UPLOAD_DIR / (uuid4().hex + Path(original).suffix.lower())
    try:
        with temp.open("wb") as target:
            shutil.copyfileobj(file.file, target, length=1024 * 1024)
        if temp.stat().st_size > MAX_UPLOAD:
            raise HTTPException(413, "File exceeds 20 MB")
        # Preserve original name in metadata; ingest uses the original path.
        temp.replace(path)
        count = ingest_file(str(path))
        invalidate_cache()
        return {"filename": original, "chunks": count, "status": "indexed"}
    except HTTPException:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        temp.unlink(missing_ok=True)
        file.file.close()

@app.post("/ask")
def ask(payload: AskRequest):
    history = get_history(payload.session_id)
    key = cache_key(payload.question, history)
    result = get_cached(key)
    cached = result is not None
    if result is None:
        try:
            result = run_workflow(payload.question, history=history)
        except Exception as exc:
            raise HTTPException(500, f"Workflow failed: {exc}") from exc
        put_cached(key, result)
    remember(payload.session_id, payload.question, result["answer"])
    return {**result, "session_id": payload.session_id, "cached": cached}

@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
