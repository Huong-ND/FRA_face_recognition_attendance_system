from database.db import get_connection


def get_all_classes():
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM classes ORDER BY name")
            return cur.fetchall()


def create_class(name: str, description: str = ""):
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO classes (name, description) VALUES (%s,%s)",
                (name, description),
            )
        conn.commit()
        return conn.insert_id()
