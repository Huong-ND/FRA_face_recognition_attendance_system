
import os
import cv2
import numpy as np
from config import Config


try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False
    print("[face_service] WARNING: insightface not installed – recognition disabled.")


_face_app = None


def _get_app() -> "FaceAnalysis":
    global _face_app
    if _face_app is None:
        if not _INSIGHTFACE_OK:
            raise RuntimeError("insightface is not installed.")
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        _face_app = FaceAnalysis(
            name="buffalo_s",               # buffalo_s = SCRFD + ArcFace-R50  (~170 MB)
            root=Config.MODEL_DIR,
            # providers=["CPUExecutionProvider"],   # change to CUDAExecutionProvider if GPU
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=Config.DET_SIZE)
        print("[face_service] insightface model loaded.")
    return _face_app


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_and_embed(image_bgr: np.ndarray) -> list[dict]:
    """
    Run the full pipeline on a BGR frame.

    Returns a list of dicts (one per detected face, up to MAX_FACES):
    {
        "bbox"      : [x1, y1, x2, y2],
        "landmarks" : [[x,y]×5],
        "det_score" : float,
        "embedding" : np.ndarray  shape (512,)  L2-normalised
    }
    """
    app = _get_app()
    faces = app.get(image_bgr)

    # Sort by detection confidence desc, keep top MAX_FACES
    faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
    faces = faces[: Config.MAX_FACES]

    results = []
    for f in faces:
        results.append({
            "bbox"      : [float(v) for v in f.bbox],            # [x1,y1,x2,y2]
            "landmarks" : [[float(p[0]), float(p[1])] for p in f.kps],  # 5 pts
            "det_score" : float(f.det_score),
            "embedding" : np.array(f.embedding, dtype=np.float32),      # (512,)
        })
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a / (np.linalg.norm(a) + 1e-8),
                        b / (np.linalg.norm(b) + 1e-8)))


def match_embedding(
    query_emb: np.ndarray,
    gallery_names: list[str],
    gallery_matrix: np.ndarray,
) -> dict:
    """
    Compare query embedding against gallery.

    Returns:
    {
        "name"       : str,        # matched name or "Unknown" / "Uncertain"
        "confidence" : float,      # cosine similarity [0,1]
        "status"     : "confident" | "uncertain" | "unknown"
    }
    """
    if gallery_matrix.shape[0] == 0:
        return {"name": "No Gallery", "confidence": 0.0, "status": "unknown"}

    # Normalise query
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    # Normalise gallery rows
    norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True) + 1e-8
    normed = gallery_matrix / norms

    sims = normed @ q                           # (N,)
    idx  = int(np.argmax(sims))
    best = float(sims[idx])

    if best >= Config.THRESHOLD_HIGH:
        status = "confident"
        name   = gallery_names[idx]
    elif best >= Config.THRESHOLD_LOW:
        status = "uncertain"
        name   = f"~{gallery_names[idx]}"
    else:
        status = "unknown"
        name   = "Unknown"

    return {"name": name, "confidence": round(best, 4), "status": status}


def extract_embedding_from_path(image_path: str) -> np.ndarray | None:
    """
    Convenience: load an image file, detect first face, return its embedding.
    Returns None if no face found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = detect_and_embed(img)
    if not faces:
        return None
    return faces[0]["embedding"]


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average multiple embeddings and L2-normalise the result."""
    mat = np.stack(embeddings, axis=0).astype(np.float32)
    avg = mat.mean(axis=0)
    avg /= np.linalg.norm(avg) + 1e-8
    return avg
