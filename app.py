"""
项目入口 + 前端界面
作用：整个项目的启动文件，用户看到的聊天界面。
功能：
启动 Streamlit 网页界面
显示聊天历史
接收用户问题
调用智能体（Agent）
把回答流式输出给用户看
保存对话记录
"""

import os

# 绕过系统代理（代理自签证书会导致 SSL 验证失败）
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import time
import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
