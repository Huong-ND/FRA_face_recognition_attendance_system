"""
Admin API
GET  /api/admin/top-stats         – 4 stat cards
GET  /api/admin/chart-attendance  – Điểm danh thành công per day
GET  /api/admin/chart-scans       – Tất cả lần quét per day
GET  /api/admin/logs              – All scan detail rows (with filters)
GET  /api/admin/summary           – legacy daily summary
POST /api/admin/change-password
GET  /api/admin/students
"""
from flask import Blueprint, request, jsonify, session
from utils.auth_utils import admin_required
from models.attendance_model import (
    get_top_stats, get_attendance_daily_for_chart,
    get_logs_daily_summary, get_logs, get_daily_summary,
)

admin_bp = Blueprint("admin", __name__)


def _ser(rows):
    out = []
    for r in rows:
        row = dict(r)
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
        out.append(row)
    return out


@admin_bp.route("/top-stats", methods=["GET"])
@admin_required
def top_stats():
    return jsonify(get_top_stats())


@admin_bp.route("/chart-attendance", methods=["GET"])
@admin_required
def chart_attendance():
    date_from = request.args.get("from")
    date_to   = request.args.get("to")
    return jsonify(get_attendance_daily_for_chart(date_from, date_to))


@admin_bp.route("/chart-scans", methods=["GET"])
@admin_required
def chart_scans():
    date_from = request.args.get("from")
    date_to   = request.args.get("to")
    return jsonify(get_logs_daily_summary(date_from, date_to))


@admin_bp.route("/logs", methods=["GET"])
@admin_required
def scan_logs():
    date_from     = request.args.get("from")
    date_to       = request.args.get("to")
    result_filter = request.args.get("result")      # confident/uncertain/unknown/all
    name_search   = request.args.get("name")
    limit         = int(request.args.get("limit", 500))
    rows = get_logs(date_from, date_to, result_filter, name_search, limit)
    return jsonify(_ser(rows))


@admin_bp.route("/summary", methods=["GET"])
@admin_required
def summary():
    return jsonify(get_daily_summary())


@admin_bp.route("/change-password", methods=["POST"])
@admin_required
def change_password():
    from database.db import get_connection
    data     = request.get_json(silent=True) or {}
    old_pw   = data.get("old_password", "")
    new_pw   = data.get("new_password", "")
    username = data.get("username", session.get("username"))
    if not new_pw or len(new_pw) < 4:
        return jsonify({"error": "New password must be at least 4 characters"}), 400
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s AND password=%s", (username, old_pw))
            if not cur.fetchone():
                return jsonify({"error": "Current password is incorrect"}), 401
            cur.execute("UPDATE users SET password=%s WHERE username=%s", (new_pw, username))
        conn.commit()
    return jsonify({"message": "Password updated"})


@admin_bp.route("/students", methods=["GET"])
@admin_required
def list_students_admin():
    from database.db import get_connection
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, student_code, source, photo_count, created_at,
                       (embedding IS NOT NULL) AS has_embedding
                FROM students ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
    return jsonify(_ser(rows))
