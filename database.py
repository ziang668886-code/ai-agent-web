import os

import pymysql
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    """Create and return a new MySQL database connection."""
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("DB_PASSWORD environment variable is not set.")

    port_value = os.getenv("DB_PORT", "3306")
    try:
        port = int(port_value)
    except ValueError as error:
        raise RuntimeError("DB_PORT must be an integer.") from error

    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=port,
        database=os.getenv("DB_NAME", "ai_agent"),
        user=os.getenv("DB_USER", "ai_agent_user"),
        password=password,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
        autocommit=True,
    )


def test_connection():
    """Return True when MySQL responds successfully to SELECT 1."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 AS result")
            row = cursor.fetchone()
            return bool(row and row["result"] == 1)
    finally:
        connection.close()
