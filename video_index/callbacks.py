from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

# AIDEV-NOTE: spec-sensitive;
# 前段エージェント結果を統合してマニュアル生成用LLMリクエストを完成


class AttachVideoToLlmRequestCallback:
    def __init__(self, start_time: str, end_time: str):
        self.start_time = start_time
        self.end_time = end_time

    async def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """
            前段エージェント結果を統合してマニュアル生成用LLMリクエストを完成

            Args:
                callback_context: コールバックコンテキスト
                llm_request: LLMリクエスト

        Returns:
            LLMリクエスト
        """

        video_part = await callback_context.load_artifact("uploaded_video")

        inline_data = types.Blob(
            data=video_part.inline_data.data,
            display_name="uploaded_video",
            mime_type="video/mp4",
        )

        start_offset = int(self.start_time.split(":")[0]) * 60 + int(
            self.start_time.split(":")[1]
        )
        end_offset = int(self.end_time.split(":")[0]) * 60 + int(
            self.end_time.split(":")[1]
        )

        video_metadata = types.VideoMetadata(
            start_offset=f"{start_offset}s",
            end_offset=f"{end_offset}s",
            fps=1,
        )

        video_content = types.Part(
            inline_data=inline_data,
            video_metadata=video_metadata,
        )

        llm_request.contents.append(
            types.UserContent(
                parts=[video_content],
            )
        )
