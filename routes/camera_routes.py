"""
Camera / Recognition API
========================
POST /api/camera/recognize          – process one frame, return face results
POST /api/camera/recognize-test     – same but is_test=True
POST /api/camera/enroll-manual      – enroll from a SINGLE photo (legacy)
POST /api/camera/enroll-multi       – enroll from MULTIPLE photos (new)
POST /api/camera/enroll-add-photo   – add extra photo to existing student
GET  /api/camera/gallery-info       – enrolled count
POST /api/camera/refresh-cache      – reload embedding cache
"""

from flask import Blueprint, request, jsonify
from services.attendance_service import process_frame
from services import embedding_cache

camera_bp = Blueprint("camera", __name__)


@camera_bp.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame")
    if not frame_b64:
        return jsonify({"error": "No frame provided"}), 400
    return jsonify({"faces": process_frame(frame_b64)})


@camera_bp.route("/recognize-test", methods=["POST"])
def recognize_test():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame")
    if not frame_b64:
        return jsonify({"error": "No frame provided"}), 400
    return jsonify({"faces": process_frame(frame_b64)})


# ── helpers ────────────────────────────────────────────────────────────────────
def _extract_embeddings_from_frames(frames_b64: list) -> list:
    """
    Given a list of base64 images, return list of (embedding, det_score) tuples.
    Skips frames where no face is detected.
    """
    from services.face_service import detect_and_embed
    from services.attendance_service import decode_frame
    results = []
    for b64 in frames_b64:
        img = decode_frame(b64)
        if img is None:
            continue
        faces = detect_and_embed(img)
        if faces:
            best = max(faces, key=lambda f: f["det_score"])
            results.append((best["embedding"], best["det_score"]))
    return results


# ── Single photo enroll (legacy, kept for compatibility) ───────────────────────
@camera_bp.route("/enroll-manual", methods=["POST"])
def enroll_manual():
    data         = request.get_json(silent=True) or {}
    frames_b64   = [data.get("frame")] if data.get("frame") else []
    name         = data.get("name", "").strip()
    student_code = data.get("student_code", "").strip()
    if not frames_b64 or not name:
        return jsonify({"error": "frame and name are required"}), 400
    return _do_enroll(frames_b64, name, student_code, source="manual")


# ── Multi-photo enroll ─────────────────────────────────────────────────────────
@camera_bp.route("/enroll-multi", methods=["POST"])
def enroll_multi():
    """
    Body: {
        "frames":       ["<base64>", "<base64>", ...],  // 1-10 photos
        "name":         "Nguyen Van A",
        "student_code": "SV001"                          // optional
    }

    Pipeline:
      For each frame → SCRFD detect → best face → ArcFace embedding
      Average all embeddings → L2-normalise → store as students.embedding
      Also store each individual embedding in student_embeddings (audit trail)
    """
    data         = request.get_json(silent=True) or {}
    frames_b64   = data.get("frames", [])
    name         = data.get("name", "").strip()
    student_code = data.get("student_code", "").strip()

    if not frames_b64:
        return jsonify({"error": "frames list is required"}), 400
    if not name:
        return jsonify({"error": "name is required"}), 400
    if len(frames_b64) > 15:
        return jsonify({"error": "Maximum 15 photos allowed"}), 400

    return _do_enroll(frames_b64, name, student_code, source="manual")


def _do_enroll(frames_b64: list, name: str, student_code: str, source: str):
    """
    Shared enroll logic for both single and multi-photo routes.
    Writes to:
      students            → averaged embedding + photo_count
      student_embeddings  → one row per photo (individual embeddings)
    """
    import numpy as np
    from services.face_service import average_embeddings
    from models.student_model import (
        create_student, update_embedding,
        get_student_by_code, save_individual_embedding,
        recompute_average_embedding, get_individual_embeddings
    )

    extracted = _extract_embeddings_from_frames(frames_b64)
    if not extracted:
        return jsonify({"error": "No face detected in any of the provided images"}), 422

    embeddings = [e for e, _ in extracted]
    avg_emb    = average_embeddings(embeddings)
    n_photos   = len(embeddings)

    existing = get_student_by_code(student_code) if student_code else None

    if existing:
        sid = existing["id"]
        # Get existing individual embeddings from DB
        old_vecs = get_individual_embeddings(sid)
        start_idx = len(old_vecs) + 1
        # Append new individual embeddings
        for i, emb in enumerate(embeddings, start=start_idx):
            save_individual_embedding(sid, emb, photo_index=i)
        # Re-average ALL (old + new)
        recompute_average_embedding(sid)
        new_count = len(old_vecs) + n_photos
        embedding_cache.refresh()
        return jsonify({
            "message"    : f"Added {n_photos} photo(s) to existing student",
            "student_id" : sid,
            "total_photos": new_count,
            "action"     : "updated",
        })

    # New student
    sid = create_student(name, student_code or None, avg_emb,
                         source=source, photo_count=n_photos)
    for i, emb in enumerate(embeddings, start=1):
        save_individual_embedding(sid, emb, photo_index=i)

    embedding_cache.refresh()
    return jsonify({
        "message"    : f"Student enrolled with {n_photos} photo(s)",
        "student_id" : sid,
        "total_photos": n_photos,
        "action"     : "created",
    }), 201


# ── Add more photos to existing student ────────────────────────────────────────
@camera_bp.route("/enroll-add-photo", methods=["POST"])
def enroll_add_photo():
    """
    Add extra photos to an already-enrolled student.
    Body: { "student_id": 42, "frames": ["<base64>", ...] }
    Re-computes the averaged embedding after adding.
    """
    from models.student_model import (
        get_student_by_id, save_individual_embedding,
        recompute_average_embedding, get_individual_embeddings
    )

    data       = request.get_json(silent=True) or {}
    student_id = data.get("student_id")
    frames     = data.get("frames", [])

    if not student_id or not frames:
        return jsonify({"error": "student_id and frames are required"}), 400

    student = get_student_by_id(int(student_id))
    if not student:
        return jsonify({"error": "Student not found"}), 404

    extracted = _extract_embeddings_from_frames(frames)
    if not extracted:
        return jsonify({"error": "No face detected"}), 422

    old_count = len(get_individual_embeddings(student_id))
    for i, (emb, _) in enumerate(extracted, start=old_count + 1):
        save_individual_embedding(student_id, emb, photo_index=i)

    new_avg = recompute_average_embedding(student_id)
    embedding_cache.refresh()
    total = old_count + len(extracted)
    return jsonify({
        "message"     : f"Added {len(extracted)} photo(s)",
        "student_id"  : student_id,
        "total_photos": total,
    })


# ── Gallery info ───────────────────────────────────────────────────────────────
@camera_bp.route("/gallery-info", methods=["GET"])
def gallery_info():
    names, matrix = embedding_cache.get_gallery()
    return jsonify({"count": len(names), "students": names})


@camera_bp.route("/refresh-cache", methods=["POST"])
def refresh_cache():
    embedding_cache.refresh()
    names, _ = embedding_cache.get_gallery()
    return jsonify({"message": "Cache refreshed", "count": len(names)})
