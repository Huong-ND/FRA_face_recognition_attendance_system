"""
High-level attendance logic.
Every face scan → recognition_logs (ALL results).
Only confident (>=0.70) + first time today → attendance.
"""
import cv2, base64
import numpy as np
from services.face_service import detect_and_embed, match_embedding
from services import embedding_cache
from models.attendance_model import mark_attendance, log_scan
from database.db import get_connection


def decode_frame(b64_image: str) -> np.ndarray:
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    raw = base64.b64decode(b64_image)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def process_frame(b64_image: str) -> list[dict]:
    """
    Full pipeline. Returns per-face dicts with recognition results.
    Logs every detection to recognition_logs.
    Marks attendance only for confident + first time today.
    """
    img = decode_frame(b64_image)
    if img is None:
        return []

    faces  = detect_and_embed(img)
    names, matrix = embedding_cache.get_gallery()
    results = []

    for face in faces:
        match = match_embedding(face["embedding"], names, matrix)
        student_id = _get_student_id_by_name(match["name"]) if match["status"] == "confident" else None

        # Log EVERY scan to recognition_logs
        log_scan(
            student_id=student_id,
            predicted_name=match["name"],
            confidence=match["confidence"],
            scan_result=match["status"],
        )

        marked = False
        if match["status"] == "confident" and student_id:
            marked = mark_attendance(student_id, match["confidence"])

        results.append({
            "bbox"       : face["bbox"],
            "landmarks"  : face["landmarks"],
            "det_score"  : face["det_score"],
            "name"       : match["name"],
            "confidence" : match["confidence"],
            "status"     : match["status"],
            "marked"     : marked,
        })

    return results


def _get_student_id_by_name(name: str):
    # strip leading ~ from uncertain matches
    name = name.lstrip("~").strip()
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM students WHERE name=%s LIMIT 1", (name,))
            row = cur.fetchone()
            return row["id"] if row else None
