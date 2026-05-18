from flask import Blueprint, request, jsonify, session
from database.db import get_connection

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, role FROM users WHERE username=%s AND password=%s",
                (username, password),
            )
            user = cur.fetchone()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    session["role"]     = user["role"]
    return jsonify({"message": "OK", "role": user["role"]})


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@auth_bp.route("/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({"username": session["username"], "role": session["role"]})
