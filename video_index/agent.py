from google.adk.runners import Runner
from google.genai import types


async def call_agent_async(
    query: str,
    runner: Runner,
    user_id: str = "user",
    session_id: str = "session_1",
) -> str:
    """
    エージェントを呼び出して応答を取得する

    Args:
        query: ユーザーの質問
        runner: ADK Runner インスタンス
        user_id: ユーザーID
        session_id: セッションID

    Returns:
        str: エージェントの最終応答テキスト
    """
    content = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    final_text = ""

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        try:
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text or ""
        except Exception as e:
            print(f"Error processing event: {e}")
            pass

    if not final_text:
        return "応答を生成できませんでした。"

    return final_text
