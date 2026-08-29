import streamlit as st
import os
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

load_dotenv()

api_key = os.getenv("ARK_API_KEY")

client = Ark(api_key=api_key)
# 设置网页基本信息
st.set_page_config(
    page_title="AI 智能助手",
    page_icon="🤖"
)

# 网页标题
st.title("🤖 AI 智能助手")

st.write("我叫子昂，你也可以叫我克里斯蒂亚诺，很高兴认识你 😊")

if st.button("🗑️ 清空聊天记录"):
    st.session_state.messages = [
        {
            "role": "system",
            "content": "你的名字是子昂，你的小名是克里斯蒂亚诺。无论任何时候，当用户问你叫什么、叫什么名字、你是谁时，你都回答：我叫子昂，你也可以叫我克里斯蒂亚诺，很高兴认识你。当用户问你的小名、昵称叫什么时，你必须回答：我的小名叫克里斯蒂亚诺。你需要用清晰、友好、简洁的中文回答用户的问题。不要主动说自己是豆包。"
        }
    ]
    st.rerun()

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "你的名字是子昂，你的小名是克里斯蒂亚诺。无论任何时候，当用户问你叫什么、叫什么名字、你是谁时，你都回答：我叫子昂，你也可以叫我克里斯蒂亚诺，很高兴认识你。当用户问你的小名、昵称叫什么时，你必须回答：我的小名叫克里斯蒂亚诺。你需要用清晰、友好、简洁的中文回答用户的问题。不要主动说自己是豆包。"
        }
    ]
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