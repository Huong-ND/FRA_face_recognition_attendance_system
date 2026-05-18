import pymysql
from config import Config

def get_connection():
    return pymysql.connect(
        host=Config.MYSQL_HOST, port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER, password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, autocommit=False,
    )

def init_db():
    conn = pymysql.connect(
        host=Config.MYSQL_HOST, port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER, password=Config.MYSQL_PASSWORD,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
    )
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{Config.MYSQL_DB}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        conn.commit()

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            # students
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    name         VARCHAR(255) NOT NULL,
                    student_code VARCHAR(80)  UNIQUE,
                    embedding    LONGTEXT,
                    photo_count  INT DEFAULT 1,
                    source       VARCHAR(50) DEFAULT 'manual',
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            # per-photo embeddings
            cur.execute("""
                CREATE TABLE IF NOT EXISTS student_embeddings (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    student_id  INT NOT NULL,
                    embedding   LONGTEXT NOT NULL,
                    photo_index INT DEFAULT 1,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            # attendance – one row per person per day (first successful scan only)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    student_id    INT NOT NULL,
                    date          DATE NOT NULL,
                    recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence    FLOAT NOT NULL DEFAULT 0,
                    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                    UNIQUE KEY uq_student_date (student_id, date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            # recognition_logs – every single face scan (success OR fail)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    student_id    INT,
                    predicted_name VARCHAR(255),
                    confidence    FLOAT NOT NULL,
                    scan_result   ENUM('confident','uncertain','unknown') NOT NULL,
                    scanned_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            # recognition_tests – kept for backwards compat, can be ignored
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recognition_tests (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    student_id  INT,
                    confidence  FLOAT NOT NULL,
                    label       VARCHAR(255),
                    predicted   VARCHAR(255),
                    is_correct  TINYINT(1),
                    tested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            # users
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INT AUTO_INCREMENT PRIMARY KEY,
                    username   VARCHAR(80) UNIQUE NOT NULL,
                    password   VARCHAR(255) NOT NULL,
                    role       ENUM('admin','viewer') DEFAULT 'viewer',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            # migrations
            for col_sql in [
                "ALTER TABLE students ADD COLUMN photo_count INT DEFAULT 1",
                "ALTER TABLE attendance DROP INDEX uq_student_date",
                "ALTER TABLE attendance ADD UNIQUE KEY uq_student_date (student_id, date)",
            ]:
                try:
                    cur.execute(col_sql)
                except Exception:
                    pass
            cur.execute("""
                INSERT IGNORE INTO users (username, password, role)
                VALUES ('admin', 'admin123', 'admin');
            """)
        conn.commit()
    print("[DB] Tables ready.")
