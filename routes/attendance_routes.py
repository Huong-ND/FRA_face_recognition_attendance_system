from flask import Blueprint, jsonify, request
from models.attendance_model import get_attendance_by_date, get_attendance_dates

attendance_bp = Blueprint("attendance", __name__)


def _ser(rows):
    out = []
    for r in rows:
        row = dict(r)
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
        out.append(row)
    return out


@attendance_bp.route("/", methods=["GET"])
def list_attendance():
    target_date = request.args.get("date")
    return jsonify(_ser(get_attendance_by_date(target_date)))


@attendance_bp.route("/dates", methods=["GET"])
def list_dates():
    return jsonify(get_attendance_dates())
