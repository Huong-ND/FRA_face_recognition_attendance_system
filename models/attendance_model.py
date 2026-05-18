from datetime import date
from database.db import get_connection


# ── attendance (one record per person per day) ────────────────────────────────

def mark_attendance(student_id: int, confidence: float) -> bool:
    today = date.today().isoformat()
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO attendance (student_id, date, confidence) VALUES (%s,%s,%s)",
                (student_id, today, confidence),
            )
        conn.commit()
        return conn.affected_rows() > 0


def get_attendance_by_date(target_date: str = None):
    if target_date is None:
        target_date = date.today().isoformat()
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.id, a.date, a.recognized_at, a.confidence,
                       s.name, s.student_code
                FROM attendance a
                JOIN students s ON s.id = a.student_id
                WHERE a.date = %s
                ORDER BY a.recognized_at DESC
            """, (target_date,))
            return cur.fetchall()


def get_attendance_dates():
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
            return [
                r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"])
                for r in cur.fetchall()
            ]


# ── recognition_logs (every single scan) ─────────────────────────────────────

def log_scan(student_id, predicted_name: str, confidence: float, scan_result: str):
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO recognition_logs (student_id, predicted_name, confidence, scan_result) "
                "VALUES (%s,%s,%s,%s)",
                (student_id, predicted_name, confidence, scan_result),
            )
        conn.commit()


def get_logs(date_from: str = None, date_to: str = None,
             result_filter: str = None, name_search: str = None,
             limit: int = 500) -> list:
    """
    Fetch recognition_logs with optional filters.
    result_filter: 'confident' | 'uncertain' | 'unknown' | None (all)
    """
    conn = get_connection()
    conditions = []
    params = []
    if date_from:
        conditions.append("DATE(rl.scanned_at) >= %s"); params.append(date_from)
    if date_to:
        conditions.append("DATE(rl.scanned_at) <= %s"); params.append(date_to)
    if result_filter and result_filter != 'all':
        conditions.append("rl.scan_result = %s"); params.append(result_filter)
    if name_search:
        conditions.append("rl.predicted_name LIKE %s"); params.append(f"%{name_search}%")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    with conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT rl.id, rl.predicted_name, rl.confidence, rl.scan_result,
                       rl.scanned_at, s.student_code
                FROM recognition_logs rl
                LEFT JOIN students s ON s.id = rl.student_id
                {where}
                ORDER BY rl.scanned_at DESC
                LIMIT %s
            """, params)
            return cur.fetchall()


def get_logs_daily_summary(date_from: str = None, date_to: str = None) -> list:
    """
    Per-day aggregation of recognition_logs for charts.
    Returns: date, confident_count, uncertain_count, unknown_count, total
    """
    conn = get_connection()
    conditions = []
    params = []
    if date_from:
        conditions.append("DATE(scanned_at) >= %s"); params.append(date_from)
    if date_to:
        conditions.append("DATE(scanned_at) <= %s"); params.append(date_to)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    DATE(scanned_at) AS date,
                    SUM(scan_result='confident') AS confident_count,
                    SUM(scan_result='uncertain') AS uncertain_count,
                    SUM(scan_result='unknown')   AS unknown_count,
                    COUNT(*) AS total
                FROM recognition_logs
                {where}
                GROUP BY DATE(scanned_at)
                ORDER BY date ASC
            """, params)
            rows = cur.fetchall()
    return [{
        "date"            : r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"]),
        "confident_count" : int(r["confident_count"] or 0),
        "uncertain_count" : int(r["uncertain_count"] or 0),
        "unknown_count"   : int(r["unknown_count"]   or 0),
        "total"           : int(r["total"] or 0),
    } for r in rows]


def get_attendance_daily_for_chart(date_from: str = None, date_to: str = None) -> list:
    """
    For 'Điểm danh thành công' chart.
    Returns per-day: date, count of distinct students, avg confidence of first scan.
    """
    conn = get_connection()
    conditions = []
    params = []
    if date_from:
        conditions.append("a.date >= %s"); params.append(date_from)
    if date_to:
        conditions.append("a.date <= %s"); params.append(date_to)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    a.date,
                    COUNT(DISTINCT a.student_id) AS student_count,
                    AVG(a.confidence)            AS avg_confidence
                FROM attendance a
                {where}
                GROUP BY a.date
                ORDER BY a.date ASC
            """, params)
            rows = cur.fetchall()
    return [{
        "date"          : r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"]),
        "student_count" : int(r["student_count"] or 0),
        "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
    } for r in rows]


# ── admin top-stats ───────────────────────────────────────────────────────────

def get_top_stats() -> dict:
    """
    Returns the 4 admin stat card values:
    - total_students
    - avg_conf_first_scan  (avg confidence of first successful scan per person per day)
    - success_rate         (distinct students attended / distinct predicted names scanned)
    - avg_conf_all_scans   (avg of all recognition_logs – may have duplicates)
    - total_scans
    """
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM students")
            total_students = cur.fetchone()["n"]

            cur.execute("SELECT AVG(confidence) AS ac, COUNT(*) AS n FROM attendance")
            r = cur.fetchone()
            avg_conf_first = float(r["ac"] or 0)
            total_attendance = int(r["n"] or 0)

            cur.execute("SELECT AVG(confidence) AS ac, COUNT(*) AS n FROM recognition_logs")
            r = cur.fetchone()
            avg_conf_all = float(r["ac"] or 0)
            total_scans = int(r["n"] or 0)

            # success rate = people successfully attended / total unique people scanned today (all time)
            cur.execute("""
                SELECT
                    COUNT(DISTINCT student_id) AS success_people,
                    COUNT(*) AS total_confident
                FROM recognition_logs WHERE scan_result='confident'
            """)
            r = cur.fetchone()
            success_people = int(r["success_people"] or 0)

            cur.execute("SELECT COUNT(*) AS n FROM recognition_logs")
            total_logs = int(cur.fetchone()["n"] or 0)

    success_rate = (success_people / total_logs * 100) if total_logs else 0

    return {
        "total_students"    : total_students,
        "avg_conf_first"    : round(avg_conf_first * 100, 1),   # as percentage
        "total_attendance"  : total_attendance,
        "success_rate"      : round(success_rate, 1),
        "avg_conf_all"      : round(avg_conf_all * 100, 1),
        "total_scans"       : total_scans,
    }


def get_daily_summary() -> list:
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    d.date,
                    d.total_tests,
                    d.avg_confidence,
                    d.correct_count,
                    COALESCE(e.enrolled_count, 0) AS enrolled_count
                FROM (
                    SELECT DATE(tested_at) AS date, COUNT(*) AS total_tests,
                           AVG(confidence) AS avg_confidence, SUM(is_correct) AS correct_count
                    FROM recognition_tests GROUP BY DATE(tested_at)
                ) d
                LEFT JOIN (
                    SELECT DATE(created_at) AS date, COUNT(*) AS enrolled_count
                    FROM students GROUP BY DATE(created_at)
                ) e ON e.date = d.date
                ORDER BY d.date DESC
            """)
            rows = cur.fetchall()
    out = []
    for r in rows:
        row = dict(r)
        if hasattr(row.get("date"), "isoformat"):
            row["date"] = row["date"].isoformat()
        if row.get("avg_confidence") is not None:
            row["avg_confidence"] = round(float(row["avg_confidence"]), 4)
        out.append(row)
    return out
