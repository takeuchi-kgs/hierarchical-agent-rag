import asyncio

import streamlit as st
from google.adk.runners import InMemoryRunner
from google.genai import types

from video_index.agent import call_agent_async
from video_index.indexer import index_video
from video_index.ui import render_chat_interface, render_video_tree

APP_NAME = "sample_adk_app"
USER_ID = "user"
SESSION_ID = "session_1"

st.set_page_config(page_title="動画インデックス & AI対話", layout="wide")

st.title("動画インデックス & AI対話")


# ヘルパー関数
def initialize_video_session(video_bytes: bytes) -> InMemoryRunner:
    """
    動画セッションを初期化し、エージェントをウォームアップする

    Args:
        video_bytes: 動画のバイトデータ

    Returns:
        InMemoryRunner: 初期化済みのrunner
    """
    indexed_video = st.session_state["indexed_video"]
    video_agent = indexed_video.to_agent()
    video_runner = InMemoryRunner(agent=video_agent, app_name=APP_NAME)

    # 動画をアーティファクトとして保存
    asyncio.run(
        video_runner.artifact_service.save_artifact(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
            filename="uploaded_video",
            artifact=types.Part(
                inline_data=types.Blob(
                    data=video_bytes,
                    display_name="uploaded_video",
                    mime_type="video/mp4",
                )
            ),
        )
    )

    # セッション初期化
    asyncio.run(
        video_runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )

    st.session_state["video_agent"] = video_agent
    st.session_state["video_runner"] = video_runner

    return video_runner


async def warmup_agent(runner: InMemoryRunner):
    """
    エージェントをウォームアップして初回応答を高速化

    Args:
        runner: ウォームアップするrunner
    """
    warmup_content = types.Content(
        role="user",
        parts=[types.Part(text="準備完了")],
    )

    # ダミーメッセージで初回のLLM呼び出しを実行
    async for _ in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=warmup_content,
    ):
        pass  # イベントは無視


def send_message(runner: InMemoryRunner, query: str) -> str:
    """エージェントにメッセージを送信する"""
    return asyncio.run(
        call_agent_async(
            query=query,
            runner=runner,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )


# メインロジック
# 1. 動画アップロード
video_file = st.file_uploader(
    label="動画をアップロードしてください", type=["mp4", "mov", "avi", "mkv"]
)

# 2. 動画インデックス化と2カラムレイアウト
if video_file is None:
    st.info(
        "右上の「Browse files」ボタンをクリックして動画をアップロードしてください。"
    )
else:
    # インデックス化とセッション初期化（1回のみ）
    if "indexed_video" not in st.session_state:
        with st.spinner("動画をインデックス化中...", show_time=True):
            video_bytes = video_file.read()
            st.session_state["indexed_video"] = index_video(video_bytes=video_bytes)
            st.session_state["video_bytes"] = video_bytes

        # エージェント初期化とウォームアップ
        with st.spinner("AIエージェントを準備中...", show_time=True):
            video_runner = initialize_video_session(video_bytes)
            # ウォームアップでコールドスタートを解消
            asyncio.run(warmup_agent(video_runner))

        st.success("動画のインデックス化が完了しました！")

    # セッション状態からrunnerを取得
    video_runner = st.session_state["video_runner"]

    # 2カラムレイアウト
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 動画")
        st.video(data=video_file)

        st.divider()

        st.subheader("🌳 構造")
        render_video_tree(st.session_state["indexed_video"])

    with col2:
        st.subheader("💬 チャット")
        render_chat_interface(video_runner, send_message)
