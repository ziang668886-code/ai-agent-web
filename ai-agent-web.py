import streamlit as st
import hashlib
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
from embedding_service import EmbeddingServiceError, embed_chunks, embed_text
from pdf_processor import PDFProcessingError, SCANNED_PDF_ERROR, process_pdf
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
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
RAG_ROUTING_THRESHOLD = 0.35
RAG_TOP_K = 4


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
    st.session_state.messages = create_initial_messages()
    st.session_state.renaming_conversation_id = None
    set_blank_chat_requested(True)


def start_new_chat():
    """Open a blank chat while preserving all existing conversation history."""
    st.session_state.pending_delete_conversation_id = None
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
    was_renaming = st.session_state.get("renaming_conversation_id") is not None
    st.session_state.renaming_conversation_id = None
    if conversation_id == st.session_state.conversation_id:
        if was_renaming:
            st.rerun()
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
    st.session_state.suppress_history_restore = False
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


def show_history_sidebar():
    """Display clickable conversations belonging to the current visitor."""
    with st.sidebar:
        if st.button(
            "➕ 新建聊天",
            key="new_chat",
            type="primary",
            use_container_width=True,
        ):
            start_new_chat()

        st.divider()
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
            title_column, rename_column, delete_column = st.columns(
                [5, 1, 1],
                gap="small",
            )

            with title_column:
                if st.button(
                    title,
                    key=f"history_conversation_{conversation_id}",
                    use_container_width=True,
                ):
                    if not switch_conversation(conversation_id):
                        st.caption("该历史会话暂时无法加载。")

            with rename_column:
                if st.button(
                    "✏️",
                    key=f"rename_conversation_{conversation_id}",
                    help=f"重命名会话：{title}",
                    use_container_width=True,
                ):
                    st.session_state.pending_delete_conversation_id = None
                    st.session_state.renaming_conversation_id = conversation_id
                    st.session_state[
                        f"rename_title_input_{conversation_id}"
                    ] = title
                    st.rerun()

            with delete_column:
                if st.button(
                    "🗑️",
                    key=f"delete_conversation_{conversation_id}",
                    help=f"删除会话：{title}",
                    use_container_width=True,
                ):
                    st.session_state.renaming_conversation_id = None
                    st.session_state.pending_delete_conversation_id = (
                        conversation_id
                    )
                    st.rerun()

            if (
                st.session_state.get("renaming_conversation_id")
                == conversation_id
            ):
                input_key = f"rename_title_input_{conversation_id}"
                edited_title = st.text_input(
                    "会话标题",
                    value=title,
                    key=input_key,
                )
                save_column, cancel_rename_column = st.columns(2)

                with save_column:
                    if st.button(
                        "保存",
                        key=f"save_rename_{conversation_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        normalized_title = edited_title.strip()
                        if not normalized_title:
                            st.caption("标题不能为空。")
                        elif len(normalized_title) > MANUAL_TITLE_MAX_LENGTH:
                            st.caption("标题不能超过 30 个字符。")
                        elif normalized_title == title:
                            st.session_state.renaming_conversation_id = None
                            st.rerun()
                        elif rename_history_conversation(
                            conversation_id,
                            normalized_title,
                        ):
                            st.session_state.renaming_conversation_id = None
                            st.rerun()
                        else:
                            st.caption("重命名失败，请稍后重试。")

                with cancel_rename_column:
                    if st.button(
                        "取消",
                        key=f"cancel_rename_{conversation_id}",
                        use_container_width=True,
                    ):
                        st.session_state.renaming_conversation_id = None
                        st.rerun()

            if (
                st.session_state.get("pending_delete_conversation_id")
                == conversation_id
            ):
                st.caption("确认删除这个会话？删除后无法恢复。")
                confirm_column, cancel_column = st.columns(2)

                with confirm_column:
                    if st.button(
                        "确认删除",
                        key=f"confirm_delete_{conversation_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        if not delete_history_conversation(conversation_id):
                            st.caption("删除失败，请稍后重试。")

                with cancel_column:
                    if st.button(
                        "取消",
                        key=f"cancel_delete_{conversation_id}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_delete_conversation_id = None
                        st.rerun()


def show_knowledge_base_sidebar():
    """Display PDF upload controls for the current visitor's knowledge base."""
    with st.sidebar:
        st.divider()
        st.subheader("📚 我的知识库")

        try:
            if has_knowledge_base(st.session_state.visitor_id):
                st.caption("📚 当前知识库已建立")
        except Exception:
            st.caption("知识库状态暂时无法读取。")

        uploaded_pdf = st.file_uploader(
            "上传 PDF",
            type=["pdf"],
            accept_multiple_files=False,
            key="knowledge_base_pdf_uploader",
        )
        st.caption("当前仅支持可以复制文字的 PDF，暂不支持纯扫描件。")

        pdf_too_large = bool(
            uploaded_pdf
            and uploaded_pdf.size > MAX_PDF_SIZE_BYTES
        )
        if pdf_too_large:
            st.error("PDF 文件不能超过 10MB。")

        add_to_knowledge_base = st.button(
            "📥 添加到知识库",
            key="add_pdf_to_knowledge_base",
            use_container_width=True,
            disabled=uploaded_pdf is None or pdf_too_large,
        )
        if not add_to_knowledge_base or pdf_too_large:
            return

        pdf_bytes = uploaded_pdf.getvalue()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        visitor_id = st.session_state.visitor_id

        try:
            if has_document(visitor_id, document_id):
                st.info("ℹ️ 该文档已经存在于知识库中，无需重复添加。")
                return
        except Exception:
            st.error("知识库状态检查失败，请稍后重试。")
            return

        try:
            pdf_result = process_pdf(
                pdf_bytes,
                source_file=uploaded_pdf.name,
            )
        except PDFProcessingError as exc:
            if str(exc) == SCANNED_PDF_ERROR:
                st.error(SCANNED_PDF_ERROR)
            else:
                st.error("PDF 解析失败，请确认文件完整且包含可复制文字。")
            return
        except Exception:
            st.error("PDF 解析失败，请确认文件完整且包含可复制文字。")
            return

        try:
            with st.spinner("正在生成知识库索引，请稍候..."):
                embeddings = embed_chunks(pdf_result["chunks"])
        except (EmbeddingServiceError, ValueError):
            st.error("文本向量生成失败，请稍后重试。")
            return
        except Exception:
            st.error("文本向量生成失败，请稍后重试。")
            return

        try:
            add_chunks(visitor_id, pdf_result["chunks"], embeddings)
        except (VectorStoreError, ValueError):
            st.error("知识库索引保存失败，请稍后重试。")
            return
        except Exception:
            st.error("知识库索引保存失败，请稍后重试。")
            return

        st.success("✅ 已添加到知识库")
        st.write(f"文件：{pdf_result['source_file']}")
        st.write(f"页数：{pdf_result['total_pages']}")
        st.write(f"Chunk：{len(pdf_result['chunks'])}")


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


def add_sources_to_answer(answer, sources):
    """Append deduplicated, display-safe source labels to a RAG answer."""
    source_lines = []
    seen_sources = set()
    for source in sources:
        source_key = (
            source["source_file"],
            source["page_number"],
        )
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_lines.append(
            f"- {source['source_file']} · 第{source['page_number']}页"
        )

    if not source_lines:
        return answer
    return f"{answer.rstrip()}\n\n📎 来源：\n\n" + "\n".join(source_lines)


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
show_knowledge_base_sidebar()

if st.button("🗑️ 清空聊天记录"):
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

    # 显示用户发送的消息
    with st.chat_message("user"):
        st.write(question)

    retrieval_results, retrieval_failed = get_rag_retrieval_results(question)
    if retrieval_failed:
        st.caption("知识库检索暂时不可用，本次已使用普通回答。")

    with st.spinner("🤔 子昂正在思考..."):
        if retrieval_results:
            rag_result = answer_with_rag(
                visitor_id=st.session_state.visitor_id,
                user_question=question,
                chat_messages=st.session_state.messages,
                top_k=RAG_TOP_K,
                retrieval_results=retrieval_results,
            )
            answer = add_sources_to_answer(
                rag_result["answer"],
                rag_result["sources"],
            )
        else:
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

    if title_updated:
        st.rerun()
