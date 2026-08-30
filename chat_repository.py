import uuid

from database import get_connection


def create_or_update_visitor(visitor_id: str) -> None:
    """Create a visitor or refresh the visitor's last-seen time."""
    sql = """
        INSERT INTO visitors (visitor_id, created_at, last_seen_at)
        VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE last_seen_at = CURRENT_TIMESTAMP
    """

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (visitor_id,))
    finally:
        connection.close()


def create_conversation(
    visitor_id: str,
    title: str = "新对话",
    conversation_id: str | None = None,
) -> str:
    """Create a conversation and return its conversation ID."""
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    sql = """
        INSERT INTO conversations (
            conversation_id,
            visitor_id,
            title,
            created_at,
            updated_at
        )
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (conversation_id, visitor_id, title))
    finally:
        connection.close()

    return conversation_id


def save_message(conversation_id: str, role: str, content: str) -> int:
    """Save one message and return the generated message ID."""
    insert_message_sql = """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
    """
    update_conversation_sql = """
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE conversation_id = %s
    """

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(insert_message_sql, (conversation_id, role, content))
            message_id = cursor.lastrowid
            cursor.execute(update_conversation_sql, (conversation_id,))
            return message_id
    finally:
        connection.close()


def get_latest_conversation_id(visitor_id: str) -> str | None:
    """Return the visitor's most recently updated conversation ID."""
    sql = """
        SELECT conversation_id
        FROM conversations
        WHERE visitor_id = %s
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1
    """

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (visitor_id,))
            row = cursor.fetchone()
            return row["conversation_id"] if row else None
    finally:
        connection.close()


def get_messages(conversation_id: str) -> list[dict]:
    """Return a conversation's messages in chronological order."""
    sql = """
        SELECT message_id, conversation_id, role, content, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY created_at ASC, message_id ASC
    """

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (conversation_id,))
            return cursor.fetchall()
    finally:
        connection.close()
