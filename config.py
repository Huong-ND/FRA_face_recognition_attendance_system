import os

class Config:
    # ── MySQL ──────────────────────────────────────────────────────────
    MYSQL_HOST     = os.environ.get("MYSQL_HOST", "localhost")
    MYSQL_PORT     = int(os.environ.get("MYSQL_PORT", 3306))
    MYSQL_USER     = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "root")
    MYSQL_DB       = os.environ.get("MYSQL_DB", "fra_db")

    # ── Dataset ────────────────────────────────────────────────────────
    DATASET_PATH   = os.environ.get(
        "DATASET_PATH",
        r"D:\face_project\dataset\lfw-dataset\lfw-deepfunneled"
    )
    BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
    GALLERY_NPZ  = os.path.join(BASE_DIR, "data", "gallery_lfw.npz")
    DATA_DIR     = os.path.join(BASE_DIR, "data")

    # ── insightface model cache ────────────────────────────────────────
    MODEL_DIR    = os.path.join(BASE_DIR, "models", "insightface")

    # ── Recognition thresholds ────────────────────────────────────────
    # cosine similarity  ≥ HIGH  → confident match
    # cosine similarity  in [LOW, HIGH) → uncertain
    # cosine similarity  <  LOW  → unknown
    THRESHOLD_HIGH = float(os.environ.get("THRESHOLD_HIGH", 0.70))
    THRESHOLD_LOW  = float(os.environ.get("THRESHOLD_LOW",  0.50))

    # ── Frame processing ──────────────────────────────────────────────
    DET_SIZE    = (640, 640)   # SCRFD detection input size
    MAX_FACES   = 3            # max simultaneous faces per requirement
