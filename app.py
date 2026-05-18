
import os

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from database.db import init_db
from routes.auth_routes       import auth_bp
from routes.student_routes    import student_bp
from routes.attendance_routes import attendance_bp
from routes.camera_routes     import camera_bp
from routes.admin_routes      import admin_bp
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "fra_secret_2024_change_me")
CORS(app, supports_credentials=True)

app.register_blueprint(auth_bp,       url_prefix="/api/auth")
app.register_blueprint(student_bp,    url_prefix="/api/students")
app.register_blueprint(attendance_bp, url_prefix="/api/attendance")
app.register_blueprint(camera_bp,     url_prefix="/api/camera")
app.register_blueprint(admin_bp,      url_prefix="/api/admin")

@app.route("/")
def index():
    return send_from_directory("templates", "dashboard.html")

@app.route("/<path:filename>")
def serve_template(filename):
    html_path = os.path.join("templates", filename)
    if os.path.isfile(html_path):
        return send_from_directory("templates", filename)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "FRA"})

@app.route("/api/charts/<filename>")
def serve_chart(filename):
    chart_dir = os.path.join(os.path.dirname(__file__), "data", "charts")
    return send_from_directory(chart_dir, filename)

if __name__ == "__main__":
    init_db()
    from services import embedding_cache  # noqa

    # Use waitress (production WSGI) to avoid Flask dev-server warnings
    try:
        from waitress import serve
        print("[FRA] Starting on http://0.0.0.0:5000  (waitress)")
        serve(app, host="0.0.0.0", port=5000, threads=4)
    except ImportError:
        # Fallback – install waitress: pip install waitress
        print("[FRA] waitress not found, using Flask dev server")
        app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
