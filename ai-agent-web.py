import streamlit as st
import hashlib
import math
import os
import uuid
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

from chat_repository import (
    conversation_belongs_to_visitor,
    create_conversation,
    create_or_update_visitor,
    delete_conversation,
    get_conversations_by_visitor,
    get_latest_conversation_id,
    get_messages,
    rename_conversation,
    save_message,
    update_conversation_title,
)
from visitor_identity import get_or_create_visitor_id
from agent_service import run_agent_turn
from document_processor import (
    DocumentProcessingError,
    MAX_DOCUMENT_SIZE_BYTES,
    process_document,
)
from embedding_service import EmbeddingServiceError, embed_chunks, embed_text
from rag_service import answer_with_rag
from vector_store import (
    VectorStoreError,
    add_chunks,
    has_document,
    has_knowledge_base,
    search,
)

load_dotenv()

api_key = os.getenv("ARK_API_KEY")

client = Ark(api_key=api_key)

SYSTEM_PROMPT = "你的名字是子昂，你的小名是克里斯蒂亚诺。无论任何时候，当用户问你叫什么、叫什么名字、你是谁时，你都回答：我叫子昂，你也可以叫我克里斯蒂亚诺，很高兴认识你。当用户问你的小名、昵称叫什么时，你必须回答：我的小名叫克里斯蒂亚诺。你需要用清晰、友好、简洁的中文回答用户的问题。不要主动说自己是豆包。"
CONVERSATION_TITLE_MAX_LENGTH = 18
MANUAL_TITLE_MAX_LENGTH = 30
RAG_ROUTING_THRESHOLD = 0.35
RAG_TOP_K = 4

NAVIGATION_ITEMS = (
    ("home", "首页", ":material/home:"),
    ("chat", "AI 助手", ":material/auto_awesome:"),
    ("explore", "城市探索", ":material/explore:"),
    ("knowledge", "我的知识库", ":material/library_books:"),
    ("history", "历史记录", ":material/history:"),
    ("settings", "设置", ":material/settings:"),
)

SCENE_CARDS = (
    (
        "dining",
        "今天吃什么",
        "根据城市、预算和口味寻找餐厅",
        "🍽️",
    ),
    (
        "coffee",
        "找咖啡店",
        "寻找附近适合休息或约会的咖啡店",
        "☕",
    ),
    (
        "weekend",
        "周末去哪玩",
        "发现城市景点和休闲活动",
        "🗺️",
    ),
    (
        "weather",
        "天气规划",
        "结合实时天气安排今天的活动",
        "☁️",
    ),
    (
        "nearby",
        "附近推荐",
        "根据城市和地标寻找附近地点",
        "📍",
    ),
    (
        "knowledge",
        "我的资料",
        "从个人知识库获取信息",
        "📁",
    ),
)

TOOL_STATUS_LABELS = {
    "get_weather": "已查询实时天气",
    "search_poi": "已搜索城市地点",
    "knowledge_base_search": "已查询个人知识库",
}


def apply_product_styles():
    """Apply restrained product styling without relying on brittle DOM paths."""
    st.markdown(
        """
        <style>
        :root {
            --city-primary: #2563eb;
            --city-ink: #172033;
            --city-muted: #667085;
            --city-border: #e6eaf0;
            --city-surface: #ffffff;
        }
        .stApp { background: #f7f8fa; color: var(--city-ink); }
        [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--city-border);
        }
        section[data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            border-radius: 10px;
            min-height: 2.65rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--city-border);
            border-radius: 16px;
            background: var(--city-surface);
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.04);
        }
        div[data-testid="stChatMessage"] {
            border-radius: 16px;
            border: 1px solid var(--city-border);
            background: #ffffff;
            padding: 0.35rem 0.75rem;
        }
        .city-hero {
            padding: clamp(1.5rem, 3.5vw, 2.75rem);
            border: 1px solid var(--city-border);
            border-radius: 24px;
            background: linear-gradient(145deg, #ffffff 0%, #f3f7ff 100%);
            box-shadow: 0 18px 48px rgba(37, 99, 235, 0.08);
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
        .city-hero h1 {
            margin: 0 0 0.4rem 0;
            font-size: clamp(2rem, 5vw, 3.25rem);
            letter-spacing: -0.04em;
        }
        .city-hero p {
            margin: 0;
            color: var(--city-muted);
            font-size: 1.05rem;
        }
        .city-eyebrow {
            color: var(--city-primary);
            font-weight: 650;
            margin-bottom: 0.65rem;
        }
        .city-section-copy { color: var(--city-muted); margin-top: -0.5rem; }
        .city-card-copy {
            min-height: 6.75rem;
        }
        .city-card-title {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            min-height: 2rem;
            margin: 0 0 0.65rem 0;
            color: var(--city-ink);
            font-size: 1.18rem;
            font-weight: 700;
            line-height: 1.35;
        }
        .city-card-icon {
            display: inline-flex;
            flex: 0 0 1.75rem;
            width: 1.75rem;
            height: 1.75rem;
            align-items: center;
            justify-content: center;
            font-size: 1.35rem;
            line-height: 1;
        }
        .city-card-description {
            min-height: 2.8rem;
            margin: 0;
            color: var(--city-muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
            .city-hero { padding: 1.5rem; border-radius: 18px; }
            .city-card-copy { min-height: auto; }
            .city-card-description { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_initial_messages():
    """Return a fresh message list containing the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def blank_chat_requested():
    """Return whether the URL requests a persistent blank chat page."""
    return st.query_params.get("new_chat") == "1"


def set_blank_chat_requested(requested):
    """Persist or clear the blank-chat marker without storing identity data."""
    try:
        if requested:
            st.query_params["new_chat"] = "1"
        else:
            st.query_params.pop("new_chat", None)
    except Exception:
        pass


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


def generate_conversation_title(first_message):
    """Create a short title directly from the first user message."""
    normalized_message = " ".join(first_message.split())
    if not normalized_message:
        return None

    if len(normalized_message) > CONVERSATION_TITLE_MAX_LENGTH:
        shortened_message = normalized_message[
            :CONVERSATION_TITLE_MAX_LENGTH
        ].rstrip()
        return f"{shortened_message}..."

    return normalized_message


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
    st.session_state.suppress_history_restore = False
    set_blank_chat_requested(False)
    st.session_state.messages = messages
    return True


def initialize_database_session():
    """Prepare visitor and conversation IDs for this Streamlit session."""
    if "database_warning" not in st.session_state:
        st.session_state.database_warning = False

    url_requests_blank_chat = blank_chat_requested()
    if "suppress_history_restore" not in st.session_state:
        st.session_state.suppress_history_restore = url_requests_blank_chat
    elif url_requests_blank_chat:
        st.session_state.suppress_history_restore = True

    if "visitor_id" not in st.session_state:
        st.session_state.visitor_id = get_or_create_visitor_id()
        st.session_state.visitor_registered = False

    if not st.session_state.get("visitor_registered", False):
        register_current_visitor()

    if "conversation_id" not in st.session_state:
        restored = False
        if (
            st.session_state.get("visitor_registered", False)
            and not st.session_state.suppress_history_restore
        ):
            restored = restore_latest_conversation()

        if not restored:
            st.session_state.conversation_id = None
            st.session_state.conversation_registered = False
            st.session_state.suppress_history_restore = True


def start_new_conversation():
    """Show a blank chat without creating a database conversation yet."""
    st.session_state.conversation_id = None
    st.session_state.conversation_registered = False
    st.session_state.suppress_history_restore = True
    st.session_state.pending_prompt = None
    st.session_state.messages = create_initial_messages()
    st.session_state.renaming_conversation_id = None
    set_blank_chat_requested(True)


def start_new_chat():
    """Open a blank chat while preserving all existing conversation history."""
    st.session_state.pending_delete_conversation_id = None
    st.session_state.active_view = "chat"
    start_new_conversation()
    st.rerun()


def ensure_current_conversation():
    """Create the database conversation when the user first sends a message."""
    if (
        st.session_state.get("conversation_registered", False)
        and st.session_state.get("conversation_id")
    ):
        return True

    if not st.session_state.get("visitor_registered", False):
        mark_database_error()
        return False

    if not st.session_state.get("conversation_id"):
        st.session_state.conversation_id = str(uuid.uuid4())

    create_current_conversation()
    if st.session_state.get("conversation_registered", False):
        st.session_state.suppress_history_restore = False
        set_blank_chat_requested(False)
        return True

    return False


def save_message_safely(role, content):
    """Save a message without allowing database errors to stop the chat."""
    if not st.session_state.get("conversation_registered", False):
        mark_database_error()
        return False

    try:
        save_message(
            conversation_id=st.session_state.conversation_id,
            role=role,
            content=content,
        )
        return True
    except Exception:
        mark_database_error()
        return False


def update_title_for_first_user_message(first_message):
    """Set the default title from the first successfully saved user message."""
    title = generate_conversation_title(first_message)
    if not title:
        return False

    try:
        return update_conversation_title(
            conversation_id=st.session_state.conversation_id,
            visitor_id=st.session_state.visitor_id,
            title=title,
        )
    except Exception:
        mark_database_error()
        return False


def switch_conversation(conversation_id):
    """Switch conversations after confirming ownership for this visitor."""
    st.session_state.renaming_conversation_id = None
    if conversation_id == st.session_state.conversation_id:
        st.session_state.active_view = "chat"
        st.session_state.suppress_history_restore = False
        set_blank_chat_requested(False)
        st.rerun()

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
    st.session_state.suppress_history_restore = False
    st.session_state.active_view = "chat"
    set_blank_chat_requested(False)
    st.session_state.messages = create_messages_from_history(history)
    st.rerun()


def rename_history_conversation(conversation_id, title):
    """Rename one visitor-owned conversation without exposing DB errors."""
    try:
        return rename_conversation(
            conversation_id=conversation_id,
            visitor_id=st.session_state.visitor_id,
            title=title,
        )
    except Exception:
        return False


def delete_history_conversation(conversation_id):
    """Delete one visitor-owned conversation without exposing DB errors."""
    try:
        deleted = delete_conversation(
            conversation_id=conversation_id,
            visitor_id=st.session_state.visitor_id,
        )
    except Exception:
        return False

    if not deleted:
        return False

    st.session_state.pending_delete_conversation_id = None
    if conversation_id == st.session_state.conversation_id:
        start_new_conversation()

    st.rerun()


def format_conversation_time(value):
    """Format a database timestamp for the history page."""
    if not value:
        return ""
    try:
        return value.strftime("%Y-%m-%d %H:%M")
    except AttributeError:
        return str(value)


def render_history():
    """Render visitor-owned conversations and their management actions."""
    st.title("历史记录")
    st.caption("继续之前的对话，或整理你的聊天标题。")

    try:
        conversations = get_conversations_by_visitor(
            st.session_state.visitor_id
        )
    except Exception:
        st.warning("历史记录暂时无法加载，请稍后重试。")
        return

    if not conversations:
        with st.container(border=True):
            st.subheader("还没有历史记录")
            st.caption("完成第一次对话后，会话会出现在这里。")
            if st.button(
                "开始聊天",
                key="empty_history_start_chat",
                type="primary",
                icon=":material/chat:",
            ):
                st.session_state.active_view = "chat"
                st.rerun()
        return

    for conversation in conversations:
        title = conversation.get("title") or "新对话"
        conversation_id = conversation["conversation_id"]
        updated_at = format_conversation_time(conversation.get("updated_at"))

        with st.container(border=True):
            st.subheader(title)
            if updated_at:
                st.caption(f"最近更新：{updated_at}")

            with st.container(horizontal=True):
                if st.button(
                    "继续聊天",
                    key=f"history_conversation_{conversation_id}",
                    type="primary",
                    icon=":material/chat:",
                ):
                    if not switch_conversation(conversation_id):
                        st.error("该历史会话暂时无法加载。")

                if st.button(
                    "重命名",
                    key=f"rename_conversation_{conversation_id}",
                    icon=":material/edit:",
                ):
                    st.session_state.pending_delete_conversation_id = None
                    st.session_state.renaming_conversation_id = conversation_id
                    st.session_state[f"rename_title_input_{conversation_id}"] = title
                    st.rerun()

                if st.button(
                    "删除",
                    key=f"delete_conversation_{conversation_id}",
                    icon=":material/delete:",
                ):
                    st.session_state.renaming_conversation_id = None
                    st.session_state.pending_delete_conversation_id = conversation_id
                    st.rerun()

            if st.session_state.get("renaming_conversation_id") == conversation_id:
                input_key = f"rename_title_input_{conversation_id}"
                edited_title = st.text_input(
                    "会话标题",
                    key=input_key,
                    max_chars=MANUAL_TITLE_MAX_LENGTH,
                )
                with st.container(horizontal=True):
                    if st.button(
                        "保存",
                        key=f"save_rename_{conversation_id}",
                        type="primary",
                        icon=":material/save:",
                    ):
                        normalized_title = edited_title.strip()
                        if not normalized_title:
                            st.error("标题不能为空。")
                        elif len(normalized_title) > MANUAL_TITLE_MAX_LENGTH:
                            st.error("标题不能超过 30 个字符。")
                        elif normalized_title == title:
                            st.session_state.renaming_conversation_id = None
                            st.rerun()
                        elif rename_history_conversation(
                            conversation_id,
                            normalized_title,
                        ):
                            st.session_state.renaming_conversation_id = None
                            st.toast("会话标题已更新。", icon=":material/check_circle:")
                            st.rerun()
                        else:
                            st.error("重命名失败，请稍后重试。")

                    if st.button(
                        "取消",
                        key=f"cancel_rename_{conversation_id}",
                    ):
                        st.session_state.renaming_conversation_id = None
                        st.rerun()

            if st.session_state.get("pending_delete_conversation_id") == conversation_id:
                st.warning("确认删除这个会话？删除后无法恢复。")
                with st.container(horizontal=True):
                    if st.button(
                        "确认删除",
                        key=f"confirm_delete_{conversation_id}",
                        type="primary",
                    ):
                        if not delete_history_conversation(conversation_id):
                            st.error("删除失败，请稍后重试。")

                    if st.button(
                        "取消",
                        key=f"cancel_delete_{conversation_id}",
                    ):
                        st.session_state.pending_delete_conversation_id = None
                        st.rerun()


def render_knowledge_base():
    """Render the multi-format in-memory knowledge-base upload workflow."""
    st.title("我的知识库")
    st.caption("把个人资料添加到 AI 助手，后续可以基于这些资料回答问题。")

    try:
        knowledge_base_exists = has_knowledge_base(
            st.session_state.visitor_id
        )
    except Exception:
        knowledge_base_exists = False
        st.warning("知识库状态暂时无法读取。")

    if knowledge_base_exists:
        st.success("当前知识库已建立。", icon=":material/check_circle:")

    with st.container(border=True):
        st.subheader("添加资料")
        st.caption("支持 PDF、Word、TXT、Markdown。")
        uploaded_document = st.file_uploader(
            "选择资料文件",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=False,
            key="knowledge_base_document_uploader",
        )
        st.caption("PDF 需包含可复制文字，TXT 和 Markdown 需使用 UTF-8 编码；单个文件最大 10MB。")

        document_too_large = bool(
            uploaded_document
            and uploaded_document.size > MAX_DOCUMENT_SIZE_BYTES
        )
        if document_too_large:
            st.error("文件不能超过 10MB。")

        add_to_knowledge_base = st.button(
            "添加到知识库",
            key="add_document_to_knowledge_base",
            type="primary",
            icon=":material/upload_file:",
            disabled=uploaded_document is None or document_too_large,
        )
        if not add_to_knowledge_base or document_too_large:
            return

        file_bytes = uploaded_document.getvalue()
        document_id = hashlib.sha256(file_bytes).hexdigest()
        visitor_id = st.session_state.visitor_id

        try:
            if has_document(visitor_id, document_id):
                st.info("该文档已经存在，无需重复添加。")
                return
        except Exception:
            st.error("知识库状态检查失败，请稍后重试。")
            return

        try:
            document_result = process_document(
                file_bytes,
                source_file=uploaded_document.name,
                content_type=uploaded_document.type,
            )
        except DocumentProcessingError as exc:
            st.error(str(exc))
            return
        except Exception:
            st.error("文档解析失败，请确认文件完整且格式正确。")
            return

        try:
            with st.spinner("正在整理资料并建立知识库…"):
                embeddings = embed_chunks(document_result["chunks"])
        except (EmbeddingServiceError, ValueError):
            st.error("资料处理暂时失败，请稍后重试。")
            return
        except Exception:
            st.error("资料处理暂时失败，请稍后重试。")
            return

        try:
            add_chunks(visitor_id, document_result["chunks"], embeddings)
        except (VectorStoreError, ValueError):
            st.error("知识库保存失败，请稍后重试。")
            return
        except Exception:
            st.error("知识库保存失败，请稍后重试。")
            return

        st.success("已添加到知识库。", icon=":material/check_circle:")
        st.write(f"文件：{document_result['source_file']}")
        st.write(f"类型：{document_result['source_type'].upper()}")
        if document_result.get("total_pages") is not None:
            st.write(f"页数：{document_result['total_pages']}")
        st.write(f"文本片段：{len(document_result['chunks'])}")


def initialize_ui_state():
    """Initialize product navigation and one-shot prompt state."""
    st.session_state.setdefault("active_view", "home")
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("home_scenario", None)
    st.session_state.setdefault("renaming_conversation_id", None)
    st.session_state.setdefault("pending_delete_conversation_id", None)


def select_view(view_name):
    """Switch product views and clear transient history editing state."""
    st.session_state.active_view = view_name
    st.session_state.renaming_conversation_id = None
    st.session_state.pending_delete_conversation_id = None


def queue_prompt(prompt):
    """Queue one prompt for the shared chat flow and open the assistant view."""
    normalized_prompt = " ".join(str(prompt).split())
    if not normalized_prompt:
        return False
    st.session_state.pending_prompt = normalized_prompt
    st.session_state.active_view = "chat"
    return True


def render_sidebar_navigation():
    """Render app identity and the single-view product navigation."""
    with st.sidebar:
        st.markdown("### AI 城市生活助手")
        st.caption("发现城市里的吃喝玩乐")

        for view_name, label, icon in NAVIGATION_ITEMS:
            if st.button(
                label,
                key=f"navigation_{view_name}",
                icon=icon,
                type=(
                    "primary"
                    if st.session_state.active_view == view_name
                    else "secondary"
                ),
                width="stretch",
            ):
                select_view(view_name)
                st.rerun()

        st.space("small")
        if st.button(
            "新建聊天",
            key="sidebar_new_chat",
            icon=":material/add_comment:",
            width="stretch",
        ):
            start_new_chat()


def render_home_scenario_form(scenario):
    """Collect enough context to create a user-confirmed natural prompt."""
    labels = {item[0]: item[1] for item in SCENE_CARDS}
    with st.container(border=True):
        st.subheader(labels.get(scenario, "告诉我你的需求"))

        if scenario == "dining":
            with st.form("home_dining_form"):
                city = st.text_input("城市", placeholder="例如：郑州")
                landmark = st.text_input("区域 / 地标（可选）", placeholder="例如：二七广场")
                people = st.number_input("人数", min_value=1, max_value=30, value=2)
                budget = st.number_input("人均预算（元）", min_value=0, max_value=5000, value=100, step=20)
                cuisine = st.text_input("想吃", placeholder="例如：火锅")
                extra = st.text_input("其他要求（可选）", placeholder="例如：不要太辣")
                submitted = st.form_submit_button("让 AI 帮我推荐", type="primary", icon=":material/auto_awesome:")
            if submitted:
                if not city.strip() or not cuisine.strip():
                    st.error("请填写城市和想吃的类型。")
                    return
                place = f"{landmark.strip()}附近" if landmark.strip() else ""
                prompt = f"我在{city.strip()}{place}，{people}个人，人均预算{budget}元，想吃{cuisine.strip()}"
                if extra.strip():
                    prompt += f"，{extra.strip()}"
                queue_prompt(prompt + "，请帮我推荐。")
                st.rerun()

        elif scenario == "coffee":
            with st.form("home_coffee_form"):
                city = st.text_input("城市", placeholder="例如：上海")
                landmark = st.text_input("附近地标（可选）", placeholder="例如：静安寺")
                purpose = st.selectbox("主要需求", ["休息聊天", "约会", "安静办公", "特色咖啡"])
                submitted = st.form_submit_button("寻找咖啡店", type="primary", icon=":material/search:")
            if submitted:
                if not city.strip():
                    st.error("请填写城市。")
                    return
                place = f"{landmark.strip()}附近" if landmark.strip() else ""
                queue_prompt(f"帮我找{city.strip()}{place}适合{purpose}的咖啡店。")
                st.rerun()

        elif scenario == "weekend":
            with st.form("home_weekend_form"):
                city = st.text_input("城市", placeholder="例如：杭州")
                preference = st.text_input("兴趣偏好", placeholder="例如：展览、公园、亲子活动")
                companion = st.selectbox("同行人", ["一个人", "朋友", "情侣", "家人", "带孩子"])
                submitted = st.form_submit_button("生成周末建议", type="primary", icon=":material/explore:")
            if submitted:
                if not city.strip():
                    st.error("请填写城市。")
                    return
                detail = preference.strip() or "休闲活动"
                queue_prompt(f"这个周末我想和{companion}在{city.strip()}玩，偏好{detail}，请帮我规划。")
                st.rerun()

        elif scenario == "weather":
            with st.form("home_weather_form"):
                city = st.text_input("城市", placeholder="例如：郑州")
                activity = st.text_input("想安排的活动（可选）", placeholder="例如：户外散步")
                submitted = st.form_submit_button("查看天气建议", type="primary", icon=":material/cloud:")
            if submitted:
                if not city.strip():
                    st.error("请填写城市。")
                    return
                prompt = f"{city.strip()}今天的实时天气怎么样？"
                if activity.strip():
                    prompt += f"我想安排{activity.strip()}，请给我建议。"
                queue_prompt(prompt)
                st.rerun()

        elif scenario == "nearby":
            with st.form("home_nearby_form"):
                city = st.text_input("城市", placeholder="例如：北京")
                landmark = st.text_input("地标", placeholder="例如：故宫")
                target = st.text_input("想找", placeholder="例如：餐厅")
                submitted = st.form_submit_button("搜索附近", type="primary", icon=":material/near_me:")
            if submitted:
                if not city.strip() or not landmark.strip() or not target.strip():
                    st.error("请填写城市、地标和想找的地点。")
                    return
                queue_prompt(f"帮我找{city.strip()}{landmark.strip()}附近的{target.strip()}。")
                st.rerun()

        elif scenario == "knowledge":
            with st.form("home_knowledge_form"):
                query = st.text_area("想查询什么？", placeholder="例如：根据我的资料，总结项目中用到的 RAG 流程。")
                submitted = st.form_submit_button("查询我的资料", type="primary", icon=":material/library_books:")
            if submitted:
                if not query.strip():
                    st.error("请输入要查询的问题。")
                    return
                queue_prompt(query)
                st.rerun()


def render_home():
    """Render the city-life product landing page and guided scenarios."""
    st.markdown(
        """
        <div class="city-hero">
          <div class="city-eyebrow">AI 城市生活助手</div>
          <h1>今天想去哪里？</h1>
          <p>帮你发现城市里的吃喝玩乐，也能读懂你的个人资料。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("home_quick_question", border=False):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            quick_question = st.text_input(
                "直接告诉 AI 你的需求",
                placeholder="例如：今晚郑州两个人，人均 100，想吃火锅",
                key="home_quick_question_input",
            )
            submitted = st.form_submit_button(
                "问 AI",
                type="primary",
                icon=":material/arrow_forward:",
            )
    if submitted:
        if queue_prompt(quick_question):
            st.rerun()
        st.error("请先输入你的需求。")

    st.subheader("从一个场景开始")
    st.markdown('<p class="city-section-copy">先补充几项信息，再由 AI 帮你完成规划。</p>', unsafe_allow_html=True)

    for row_start in range(0, len(SCENE_CARDS), 3):
        columns = st.columns(3)
        for column, (scenario, title, description, icon) in zip(
            columns,
            SCENE_CARDS[row_start : row_start + 3],
        ):
            with column.container(border=True, height="stretch"):
                st.markdown(
                    f"""
                    <div class="city-card-copy">
                      <div class="city-card-title">
                        <span class="city-card-icon" aria-hidden="true">{icon}</span>
                        <span>{title}</span>
                      </div>
                      <p class="city-card-description">{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "开始",
                    key=f"home_scene_{scenario}",
                    width="stretch",
                ):
                    st.session_state.home_scenario = scenario
                    st.rerun()

    scenario = st.session_state.get("home_scenario")
    if scenario:
        render_home_scenario_form(scenario)


def render_city_explore():
    """Render a guided POI request that still goes through the Agent."""
    st.title("城市探索")
    st.caption("告诉我你想找什么，AI 会从真实城市地点中帮你筛选。")

    with st.container(border=True):
        with st.form("city_explore_form", border=False):
            city = st.text_input("城市", placeholder="例如：郑州")
            target = st.text_input("想找", placeholder="例如：咖啡店、餐厅、景点")
            landmark = st.text_input("附近地标（可选）", placeholder="例如：二七广场")
            preference = st.text_input("偏好（可选）", placeholder="例如：适合约会、人均 100 元以内")
            submitted = st.form_submit_button("开始探索", type="primary", icon=":material/search:")

    if submitted:
        if not city.strip() or not target.strip():
            st.error("请填写城市和想找的地点。")
            return
        place = f"{landmark.strip()}附近" if landmark.strip() else ""
        prompt = f"帮我找{city.strip()}{place}的{target.strip()}"
        if preference.strip():
            prompt += f"，希望{preference.strip()}"
        queue_prompt(prompt + "。")
        st.rerun()

    with st.container(border=True):
        st.caption("地点结果卡片将在后续版本提供。当前由 AI 结合真实地点数据给出自然语言建议。")


def render_settings():
    """Render a safe, user-facing project overview."""
    st.title("设置")
    st.caption("了解当前助手已开启的能力。")

    with st.container(border=True):
        st.subheader("AI 城市生活助手")
        st.markdown(
            """
            - :material/check_circle: AI 多轮对话
            - :material/check_circle: 实时天气
            - :material/check_circle: 城市地点搜索
            - :material/check_circle: 个人知识库
            """
        )

    with st.container(border=True):
        st.subheader("关于本项目")
        st.write("这是一个面向城市生活场景的 AI 助手，可以帮你查询天气、发现地点、规划活动，并从你的个人资料中寻找答案。")


def get_rag_retrieval_results(question):
    """Route one question using the current visitor's knowledge base."""
    try:
        if not has_knowledge_base(st.session_state.visitor_id):
            return None, False

        query_embedding = embed_text(question)
        retrieval_results = search(
            st.session_state.visitor_id,
            query_embedding,
            top_k=RAG_TOP_K,
        )
        if (
            retrieval_results
            and float(retrieval_results[0]["score"])
            >= RAG_ROUTING_THRESHOLD
        ):
            return retrieval_results, False
    except Exception:
        return None, True

    return None, False


def format_knowledge_source(source):
    """Format a real page or non-page locator without exposing internals."""
    source_file = str(source.get("source_file") or "未知文件")
    source_type = source.get("source_type")
    locator_type = source.get("locator_type")
    locator_value = source.get("locator_value")
    page_number = source.get("page_number")

    if (
        source_type == "pdf"
        or locator_type == "page"
        or (
            source_type is None
            and isinstance(page_number, int)
            and not isinstance(page_number, bool)
            and page_number > 0
        )
    ):
        page = page_number or locator_value
        return f"{source_file} · 第{page}页" if page else source_file

    if locator_type == "paragraph" and locator_value:
        value = str(locator_value)
        label = f"第{value}段" if value.isdigit() else value
        return f"{source_file} · {label}"
    if locator_type == "chunk" and locator_value:
        return f"{source_file} · Chunk {locator_value}"
    if locator_type == "section" and locator_value:
        return f"{source_file} · {locator_value}"
    return source_file


def add_sources_to_answer(answer, sources):
    """Append deduplicated, display-safe document source labels."""
    source_lines = []
    seen_sources = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        label = format_knowledge_source(source)
        if label in seen_sources:
            continue
        seen_sources.add(label)
        source_lines.append(f"- {label}")

    if not source_lines:
        return answer
    return f"{answer.rstrip()}\n\n📎 来源：\n\n" + "\n".join(source_lines)


def escape_markdown(value):
    """Escape provider text before inserting it into Markdown labels."""
    text = str(value)
    for character in "\\`*_{}[]<>()#+-.!|":
        text = text.replace(character, f"\\{character}")
    return text


def format_poi_number(value):
    """Format one finite provider number without inventing precision."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def format_poi_distance(distance_m):
    """Format metres as metres or kilometres for display only."""
    if isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)):
        return None
    if not math.isfinite(distance_m) or distance_m < 0:
        return None
    if distance_m < 1000:
        return f"{distance_m:.0f} m"
    return f"{distance_m / 1000:.1f} km"


def simplify_poi_category(category):
    """Show the most specific category while preserving raw display_data."""
    if not isinstance(category, str):
        return None
    parts = [part.strip() for part in category.split(";") if part.strip()]
    return parts[-1] if parts else None


def render_poi_cards(display_data):
    """Render safe POI display metadata without calling the provider again."""
    if not isinstance(display_data, dict):
        return
    if display_data.get("type") != "poi_results":
        return
    items = display_data.get("items")
    if not isinstance(items, list) or not items:
        return

    safe_items = [item for item in items[:5] if isinstance(item, dict)]
    if not safe_items:
        return

    st.subheader("地点信息")
    context_parts = []
    for field in ("city", "query"):
        value = display_data.get(field)
        if isinstance(value, str) and value.strip():
            context_parts.append(escape_markdown(value.strip()))
    anchor = display_data.get("anchor")
    if isinstance(anchor, str) and anchor.strip():
        context_parts.append(f"{escape_markdown(anchor.strip())}附近")
    if context_parts:
        st.caption(" · ".join(context_parts))

    for row_start in range(0, len(safe_items), 2):
        columns = st.columns(2)
        for column, item in zip(columns, safe_items[row_start : row_start + 2]):
            with column.container(border=True, height="stretch"):
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                rank = item.get("rank")
                rank_label = (
                    f"{rank}. "
                    if isinstance(rank, int) and not isinstance(rank, bool)
                    else ""
                )
                st.markdown(f"### {rank_label}{escape_markdown(name.strip())}")

                facts = []
                rating = format_poi_number(item.get("rating"))
                if rating is not None:
                    facts.append(f"★ {rating}")
                cost = format_poi_number(item.get("cost_per_person"))
                if cost is not None:
                    facts.append(f"人均 ¥{cost}")
                distance = format_poi_distance(item.get("distance_m"))
                if distance:
                    facts.append(f"距离 {distance}")
                if facts:
                    st.markdown(" &nbsp; · &nbsp; ".join(facts))

                place_parts = []
                district = item.get("district")
                if isinstance(district, str) and district.strip():
                    place_parts.append(escape_markdown(district.strip()))
                category = simplify_poi_category(item.get("category"))
                if category:
                    place_parts.append(escape_markdown(category))
                if place_parts:
                    st.caption(" · ".join(place_parts))

                address = item.get("address")
                if isinstance(address, str) and address.strip():
                    st.write(f"地址：{escape_markdown(address.strip())}")

                opening_hours = item.get("opening_hours")
                if isinstance(opening_hours, str) and opening_hours.strip():
                    st.write(
                        f"营业时间：{escape_markdown(opening_hours.strip())}"
                    )

                tags = item.get("tags")
                if isinstance(tags, list):
                    safe_tags = [
                        escape_markdown(tag.strip())
                        for tag in tags
                        if isinstance(tag, str) and tag.strip()
                    ]
                    if safe_tags:
                        st.caption("标签：" + " · ".join(safe_tags))

    st.caption("地点数据由高德地图提供")


def handle_user_message(question):
    """Run the shared lazy-create, persistence, Agent, and response flow."""
    is_first_user_message = (
        not st.session_state.get("conversation_registered", False)
        or not any(
            message["role"] == "user"
            for message in st.session_state.messages
        )
    )
    if not st.session_state.get("conversation_registered", False):
        ensure_current_conversation()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    user_message_saved = save_message_safely("user", question)
    title_updated = False
    if is_first_user_message and user_message_saved:
        title_updated = update_title_for_first_user_message(question)

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("正在为你整理建议…"):
        try:
            agent_result = run_agent_turn(
                visitor_id=st.session_state.visitor_id,
                user_question=question,
                chat_messages=st.session_state.messages,
            )
        except Exception:
            agent_result = {"ok": False}

    if not agent_result.get("ok", False):
        st.error("AI 服务暂时不可用，请稍后重试。")
        return

    answer = agent_result["answer"]
    if agent_result.get("mode") == "tool" and agent_result.get("sources"):
        answer = add_sources_to_answer(answer, agent_result["sources"])

    tool_name = agent_result.get("tool_name")
    tool_status_label = TOOL_STATUS_LABELS.get(tool_name)
    if tool_status_label:
        st.caption(f"✓ {tool_status_label}")

    with st.chat_message("assistant"):
        st.write(answer)

    render_poi_cards(agent_result.get("display_data"))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    save_message_safely("assistant", answer)

    if title_updated:
        st.rerun()


def render_chat():
    """Render the free-form assistant using the single shared chat flow."""
    title_column, action_column = st.columns([5, 1], vertical_alignment="center")
    with title_column:
        st.title("AI 助手")
        st.caption("可以问我城市、美食、天气，也可以查询你的个人资料。")
    with action_column:
        if st.button(
            "清空当前对话",
            key="clear_current_chat",
            icon=":material/delete_sweep:",
            width="stretch",
        ):
            start_new_conversation()
            st.rerun()

    visible_messages = [
        message
        for message in st.session_state.messages
        if message["role"] != "system"
    ]
    if not visible_messages:
        with st.container(border=True):
            st.subheader("从一个问题开始")
            st.caption("例如：“北京故宫附近有什么餐厅？”或“郑州今天适合户外活动吗？”")

    for message in visible_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    pending_prompt = st.session_state.pop("pending_prompt", None)
    if pending_prompt:
        handle_user_message(pending_prompt)

    question = st.chat_input(
        "输入你的问题…",
        key="assistant_chat_input",
        submit_mode="disable",
    )
    if question:
        handle_user_message(question)


st.set_page_config(
    page_title="AI 城市生活助手",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_product_styles()
initialize_database_session()
initialize_ui_state()
if "messages" not in st.session_state:
    st.session_state.messages = create_initial_messages()

render_sidebar_navigation()

if st.session_state.get("database_warning", False):
    st.caption("聊天记录暂时无法保存，但不影响 AI 对话。")

active_view = st.session_state.active_view
if active_view == "home":
    render_home()
elif active_view == "chat":
    render_chat()
elif active_view == "explore":
    render_city_explore()
elif active_view == "knowledge":
    render_knowledge_base()
elif active_view == "history":
    render_history()
elif active_view == "settings":
    render_settings()
else:
    st.session_state.active_view = "home"
    st.rerun()
