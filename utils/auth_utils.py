"""
utils/auth_utils.py
Decorators for route-level access control.
"""
from functools import wraps
from flask import session, jsonify, request


def login_required(f):
    """Block unauthenticated requests → 401."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "code": "NOT_LOGGED_IN"}), 401
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Block non-admin requests → 401 / 403."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "code": "NOT_LOGGED_IN"}), 401
        if session.get("role") != "admin":
            return jsonify({"error": "Admin access required", "code": "FORBIDDEN"}), 403
        return f(*args, **kwargs)
    return decorated
