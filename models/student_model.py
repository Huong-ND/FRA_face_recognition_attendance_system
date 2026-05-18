import json
import numpy as np
from database.db import get_connection


# ── Create / Update ───────────────────────────────────────────────────────────

def create_student(name: str, student_code: str, embedding: np.ndarray,
                   source="manual", photo_count: int = 1) -> int:
    emb_json = json.dumps(embedding.tolist())
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO students (name, student_code, embedding, photo_count, source) "
                "VALUES (%s, %s, %s, %s, %s)",
                (name, student_code, emb_json, photo_count, source),
            )
        conn.commit()
        sid = conn.insert_id()
    return sid


def update_embedding(student_id: int, embedding: np.ndarray, photo_count: int = None):
    """Replace the averaged embedding. Optionally update photo_count."""
    emb_json = json.dumps(embedding.tolist())
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            if photo_count is not None:
                cur.execute(
                    "UPDATE students SET embedding=%s, photo_count=%s WHERE id=%s",
                    (emb_json, photo_count, student_id),
                )
            else:
                cur.execute(
                    "UPDATE students SET embedding=%s WHERE id=%s",
                    (emb_json, student_id),
                )
        conn.commit()


def save_individual_embedding(student_id: int, embedding: np.ndarray, photo_index: int):
    """
    Insert one photo's embedding into student_embeddings.
    This is the per-photo audit trail — separate from the averaged vector in students.embedding.
    """
    emb_json = json.dumps(embedding.tolist())
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO student_embeddings (student_id, embedding, photo_index) "
                "VALUES (%s, %s, %s)",
                (student_id, emb_json, photo_index),
            )
        conn.commit()


def get_individual_embeddings(student_id: int) -> list[np.ndarray]:
    """
    Load all per-photo embeddings for one student.
    Used to re-compute the average when adding/removing photos.
    """
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM student_embeddings "
                "WHERE student_id=%s ORDER BY photo_index ASC",
                (student_id,),
            )
            rows = cur.fetchall()
    result = []
    for r in rows:
        try:
            vec = np.array(json.loads(r["embedding"]), dtype=np.float32)
            if vec.shape == (512,):
                result.append(vec)
        except Exception:
            pass
    return result


def recompute_average_embedding(student_id: int) -> np.ndarray | None:
    """
    Pull all individual embeddings from student_embeddings,
    average them, L2-normalise, write back to students.embedding.
    Returns the new averaged vector, or None if no embeddings found.
    """
    from services.face_service import average_embeddings
    vecs = get_individual_embeddings(student_id)
    if not vecs:
        return None
    avg = average_embeddings(vecs)
    update_embedding(student_id, avg, photo_count=len(vecs))
    return avg


# ── Read ──────────────────────────────────────────────────────────────────────

def get_all_students():
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, student_code, photo_count, source, created_at FROM students"
            )
            return cur.fetchall()


def get_student_by_id(student_id: int):
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM students WHERE id=%s", (student_id,))
            return cur.fetchone()


def get_student_by_code(code: str):
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM students WHERE student_code=%s", (code,))
            return cur.fetchone()


def delete_student(student_id: int):
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM students WHERE id=%s", (student_id,))
        conn.commit()


def load_gallery() -> tuple[list[str], np.ndarray]:
    """
    Load the averaged embedding for every student.
    Returns (names, matrix)  shape: (N, 512)
    """
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, embedding FROM students WHERE embedding IS NOT NULL"
            )
            rows = cur.fetchall()

    if not rows:
        return [], np.empty((0, 512), dtype=np.float32)

    names, vecs = [], []
    for r in rows:
        try:
            vec = np.array(json.loads(r["embedding"]), dtype=np.float32)
            if vec.shape == (512,):
                names.append(r["name"])
                vecs.append(vec)
        except Exception:
            pass
    return names, np.stack(vecs, axis=0) if vecs else np.empty((0, 512), dtype=np.float32)
