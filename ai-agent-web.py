import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

from chat_repository import (
    conversation_belongs_to_visitor,
    create_conversation,
    create_or_update_visitor,
    get_conversations_by_visitor,
    get_latest_conversation_id,
    get_messages,
    save_message,
)
from visitor_identity import get_or_create_visitor_id

load_dotenv()

api_key = os.getenv("ARK_API_KEY")

client = Ark(api_key=api_key)

SYSTEM_PROMPT = "你的名字是子昂，你的小名是克里斯蒂亚诺。无论任何时候，当用户问你叫什么、叫什么名字、你是谁时，你都回答：我叫子昂，你也可以叫我克里斯蒂亚诺，很高兴认识你。当用户问你的小名、昵称叫什么时，你必须回答：我的小名叫克里斯蒂亚诺。你需要用清晰、友好、简洁的中文回答用户的问题。不要主动说自己是豆包。"


def create_initial_messages():
    """Return a fresh message list containing the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def create_messages_from_history(history):
    """Build chat messages from database history without duplicate systems."""
    messages = create_initial_messages()
    for message in history:
        if message["role"] in ("user", "assistant"):
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )
    return messages


def mark_database_error():
    """Record a database failure without interrupting the chat experience."""
    st.session_state.database_warning = True


def register_current_visitor():
    """Create or refresh the current visitor in MySQL."""
    try:
        create_or_update_visitor(st.session_state.visitor_id)
        st.session_state.visitor_registered = True
    except Exception:
        mark_database_error()


def create_current_conversation():
    """Create the current conversation in MySQL when possible."""
    try:
        create_conversation(
            visitor_id=st.session_state.visitor_id,
            conversation_id=st.session_state.conversation_id,
        )
        st.session_state.conversation_registered = True
    except Exception:
        mark_database_error()


def restore_latest_conversation():
    """Restore the current visitor's latest conversation and messages."""
    try:
        conversation_id = get_latest_conversation_id(
            st.session_state.visitor_id
        )
        if not conversation_id:
            return False

        history = get_messages(conversation_id)
    except Exception:
        mark_database_error()
        return False

    messages = create_messages_from_history(history)

    st.session_state.conversation_id = conversation_id
    st.session_state.conversation_registered = True
    st.session_state.messages = messages
    return True


def initialize_database_session():
    """Prepare visitor and conversation IDs for this Streamlit session."""
    if "database_warning" not in st.session_state:
        st.session_state.database_warning = False

    if "visitor_id" not in st.session_state:
        st.session_state.visitor_id = get_or_create_visitor_id()
        st.session_state.visitor_registered = False

    if not st.session_state.get("visitor_registered", False):
        register_current_visitor()

    if "conversation_id" not in st.session_state:
        restored = False
        if st.session_state.get("visitor_registered", False):
            restored = restore_latest_conversation()

        if not restored:
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.conversation_registered = False

    if (
        st.session_state.get("visitor_registered", False)
        and not st.session_state.get("conversation_registered", False)
    ):
        create_current_conversation()


def start_new_conversation():
    """Start a new conversation without deleting previous database records."""
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.conversation_registered = False

    if st.session_state.get("visitor_registered", False):
        create_current_conversation()


def save_message_safely(role, content):
    """Save a message without allowing database errors to stop the chat."""
    if not st.session_state.get("conversation_registered", False):
        mark_database_error()
        return

    try:
        save_message(
            conversation_id=st.session_state.conversation_id,
            role=role,
            content=content,
        )
    except Exception:
        mark_database_error()


def switch_conversation(conversation_id):
    """Switch conversations after confirming ownership for this visitor."""
    if conversation_id == st.session_state.conversation_id:
        return True

    try:
        belongs_to_visitor = conversation_belongs_to_visitor(
            conversation_id,
            st.session_state.visitor_id,
        )
        if not belongs_to_visitor:
            return False

        history = get_messages(conversation_id)
    except Exception:
        return False

    st.session_state.conversation_id = conversation_id
    st.session_state.conversation_registered = True
    st.session_state.messages = create_messages_from_history(history)
    st.rerun()


def show_history_sidebar():
    """Display clickable conversations belonging to the current visitor."""
    with st.sidebar:
        st.subheader("历史会话")

        try:
            conversations = get_conversations_by_visitor(
                st.session_state.visitor_id
            )
        except Exception:
            st.caption("历史会话暂时无法加载。")
            return

        if not conversations:
            st.caption("暂无历史会话")
            return

        for conversation in conversations:
            title = conversation.get("title") or "新对话"
            conversation_id = conversation["conversation_id"]
            if st.button(
                title,
                key=f"history_conversation_{conversation_id}",
                use_container_width=True,
            ):
                if not switch_conversation(conversation_id):
                    st.caption("该历史会话暂时无法加载。")


# 设置网页基本信息
st.set_page_config(
    page_title="AI 智能助手",
    page_icon="🤖"
)

# 网页标题
st.title("🤖 AI 智能助手")

st.write("我叫子昂，你也可以叫我克里斯蒂亚诺，很高兴认识你 😊")

initialize_database_session()
show_history_sidebar()

if st.button("🗑️ 清空聊天记录"):
    st.session_state.messages = create_initial_messages()
    start_new_conversation()
    st.rerun()

if st.session_state.get("database_warning", False):
    st.caption("⚠️ 聊天记录暂时无法保存，但不影响 AI 对话功能。")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = create_initial_messages()
# 显示历史聊天记录
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
# 创建聊天输入框
question = st.chat_input("请输入你的问题...")

# 如果用户输入了内容
if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    save_message_safely("user", question)

    # 显示用户发送的消息
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("🤔 子昂正在思考..."):
        response = client.chat.completions.create(
            model="doubao-seed-2-0-lite-260215",
            messages=st.session_state.messages
        )

        answer = response.choices[0].message.content
    # 暂时模拟 AI 回复
    with st.chat_message("assistant"):
        st.write(answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        save_message_safely("assistant", answer)