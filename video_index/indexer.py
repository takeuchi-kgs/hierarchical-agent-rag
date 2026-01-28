from typing import cast

from dotenv import load_dotenv
from google import genai
from google.genai import types

from video_index.models import VideoAnalysisResult

load_dotenv()

client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))

VIDEO_TREE_INDEXER_PROMPT = """
あなたは熟練した**動画コンテンツアナリスト**です。

[入力]
**動画ファイル**を受け取ります。

[目標]
動画コンテンツを構造化された**コンテンツツリー**にインデックス化します。
線形的な動画タイムラインを論理的な階層ノードに変換します。

[ノード構造]
1. **SegmentNode（リーフ）**: 最小の意味単位。特定のシーン、会話のやり取り、視覚的イベント、スライドプレゼンテーションなど。**必ず`time_span`を含める必要があります。**
2. **ChapterNode（コンテナ）**: 動画の主要な区分。例：イントロダクション、メイントピックの変更、結論。

[分析戦略]
- **ボトムアップ**: まず原子的なセグメントを特定し、論理的な単位を形成する場合にセクション/チャプターにグループ化します。
- **柔軟性**: 
    - セグメントが十分に独立している場合、ルートまたはチャプター直下に配置できます。
    - 構造上重要でない限り、不要な階層（例：セグメントが1つだけのセクション）の作成は避けてください。
- **精度**: セグメントの`start_time`と`end_time`が正確であることを確認してください。
"""


def index_video(video_bytes: bytes) -> VideoAnalysisResult:
    """
    動画をインデックス化して構造化されたVideoAnalysisResultを返す

    Args:
        video_bytes: 動画ファイルのバイトデータ

    Returns:
        VideoAnalysisResult: 構造化された動画分析結果

    Raises:
        ValueError: 動画の分析に失敗した場合、またはchildrenが空の場合
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.UserContent(
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        data=video_bytes,
                        mime_type="video/mp4",
                    )
                ),
                types.Part(text=VIDEO_TREE_INDEXER_PROMPT),
            ]
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoAnalysisResult,
        ),
    )

    result = cast(VideoAnalysisResult, response.parsed)

    # 結果の検証
    if not result.children:
        raise ValueError(
            f"動画の分析結果にセグメントが含まれていません。\n"
            f"タイトル: {result.video_title}\n"
            f"概要: {result.overview}\n"
            f"動画が短すぎるか、分析に失敗した可能性があります。"
        )

    return result
