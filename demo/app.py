"""Demo web app: Sentiment analysis (IMDB) with Naive Bayes.

Run:  python demo/app.py   ->  http://127.0.0.1:8000
Loads pre-trained model artifact or automatically trains if not yet present.
"""
import json
from pathlib import Path
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "IMDB Dataset.csv"
if not DATA_PATH.exists():
    fallback = PROJECT_ROOT / "input" / "IMDB Dataset.csv"
    if fallback.exists():
        DATA_PATH = fallback

HOST, PORT = "127.0.0.1", 8000  # localhost only
MAX_TEXT_LEN = 5000
MAX_BODY_BYTES = 32 * 1024

# Allow-list of static files (never build paths from user input)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def load_or_train_model():
    if ARTIFACT_PATH.exists():
        print(f"Loading cached model from: {ARTIFACT_PATH}", flush=True)
        data = joblib.load(ARTIFACT_PATH)
        return data["vectorizer"], data["model"], data.get("accuracy", 0.8929)

    print(f"Artifact not found at {ARTIFACT_PATH}. Training model now...", flush=True)
    from src.train import train
    data = train(DATA_PATH, PROJECT_ROOT / "artifacts")
    return data["vectorizer"], data["model"], data["accuracy"]


VEC, CLF, ACC = load_or_train_model()
VOCAB = np.array(VEC.get_feature_names_out())
# per-feature evidence for positive vs negative class
LOG_RATIO = CLF.feature_log_prob_[1] - CLF.feature_log_prob_[0]


def predict(text: str) -> dict:
    cleaned = clean_text(text)
    x = VEC.transform([cleaned])
    proba = CLF.predict_proba(x)[0]
    pos, neg = float(proba[1]), float(proba[0])
    coo = x.tocoo()
    contrib = coo.data * LOG_RATIO[coo.col]
    order = np.argsort(contrib)
    words = [(VOCAB[coo.col[i]], float(contrib[i])) for i in order]
    top_neg = [{"word": w, "score": round(-s, 3)} for w, s in words if s < 0][:5]
    top_pos = [{"word": w, "score": round(s, 3)} for w, s in reversed(words) if s > 0][:5]
    return {
        "label": "positive" if pos >= neg else "negative",
        "positive": round(pos, 4),
        "negative": round(neg, 4),
        "cleaned": cleaned,
        "known_features": int(x.nnz),
        "top_positive": top_pos,
        "top_negative": top_neg,
    }


SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SentimentDemo"
    sys_version = ""

    def _send(self, status, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/info":
            return self._json(200, {
                "accuracy": round(ACC, 4),
                "model": "TF-IDF (1-2 gram) + MultinomialNB"
            })
        entry = STATIC_FILES.get(path)
        if not entry:
            return self._json(404, {"error": "Not found"})
        name, ctype = entry
        self._send(200, (BASE_DIR / "static" / name).read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/predict":
            return self._json(404, {"error": "Not found"})

        # Reject cross-origin browser requests (basic CSRF/DNS-rebinding guard)
        host_ok = self.headers.get("Host", "") in (f"{HOST}:{PORT}", f"localhost:{PORT}")
        origin = self.headers.get("Origin")
        origin_ok = origin is None or origin in (f"http://{HOST}:{PORT}", f"http://localhost:{PORT}")
        if not (host_ok and origin_ok):
            return self._json(403, {"error": "Forbidden"})

        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._json(415, {"error": "Content-Type must be application/json"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "Bad request"})

        if length <= 0 or length > MAX_BODY_BYTES:
            return self._json(413, {"error": "Request body too large or empty"})

        try:
            data = json.loads(self.rfile.read(length))
            text = data.get("text") if isinstance(data, dict) else None
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "Invalid JSON"})

        if not isinstance(text, str) or not text.strip():
            return self._json(400, {"error": "Please enter some text"})

        if len(text) > MAX_TEXT_LEN:
            return self._json(400, {"error": f"Text must be <= {MAX_TEXT_LEN} characters"})

        try:
            self._json(200, predict(text))
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "Internal error"})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
