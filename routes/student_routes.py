"""
Student management API
GET    /api/students/           – list all
GET    /api/students/<id>       – get one
DELETE /api/students/<id>       – delete
"""

from flask import Blueprint, jsonify, request
from models.student_model import get_all_students, get_student_by_id, delete_student
from services import embedding_cache

student_bp = Blueprint("students", __name__)


@student_bp.route("/", methods=["GET"])
def list_students():
    students = get_all_students()
    # Remove embedding field from response (too large)
    for s in students:
        s.pop("embedding", None)
        if hasattr(s.get("created_at"), "isoformat"):
            s["created_at"] = s["created_at"].isoformat()
    return jsonify(students)


@student_bp.route("/<int:student_id>", methods=["GET"])
def get_student(student_id):
    s = get_student_by_id(student_id)
    if not s:
        return jsonify({"error": "Not found"}), 404
    s.pop("embedding", None)
    return jsonify(s)


@student_bp.route("/<int:student_id>", methods=["DELETE"])
def remove_student(student_id):
    delete_student(student_id)
    embedding_cache.refresh()
    return jsonify({"message": "Deleted"})
