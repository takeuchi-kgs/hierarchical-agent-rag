"""チャットインターフェースモジュール"""

from typing import Callable

import streamlit as st
from google.adk.runners import InMemoryRunner


def render_chat_interface(
    runner: InMemoryRunner, send_message_fn: Callable[[InMemoryRunner, str], str]
):
    """チャットUIのレンダリング"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # チャット履歴用のコンテナ
    chat_container = st.container(height=1_000)

    # 会話履歴のレンダリング
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 入力処理（コンテナの外に配置することで固定位置に）
    if user_input := st.chat_input("メッセージを入力してください"):
        handle_user_input(chat_container, runner, user_input, send_message_fn)


def handle_user_input(
    chat_container,
    runner: InMemoryRunner,
    user_input: str,
    send_message_fn: Callable[[InMemoryRunner, str], str],
):
    """ユーザー入力の処理"""
    # ユーザーメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # コンテナ内にユーザーメッセージを表示
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

        # アシスタントの応答を生成・表示
        with st.chat_message("assistant"):
            with st.spinner("考え中...", show_time=True):
                try:
                    response = send_message_fn(runner, user_input)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    # エラーメッセージを表示して履歴に追加
                    error_msg = f"❌ エラー: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
