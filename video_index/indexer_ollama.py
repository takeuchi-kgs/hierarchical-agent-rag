"""
Ollama + Qwen3-VL を使用した動画インデックス処理

動画からフレームを抽出し、各フレームをVLMで解析して階層構造を生成する。
"""

import base64
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx

from video_index.models import (
    ChapterNode,
    SegmentNode,
    TimeSpan,
    VideoAnalysisResult,
)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-vl:8b"  # 48GBメモリ向け（高精度）
OLLAMA_TIMEOUT = 300.0  # 5分


def extract_frames(
    video_bytes: bytes,
    interval_seconds: int = 10,
    max_frames: int = 30,
) -> list[tuple[str, bytes]]:
    """
    動画からフレームを抽出する

    Args:
        video_bytes: 動画ファイルのバイトデータ
        interval_seconds: フレーム抽出間隔（秒）
        max_frames: 最大フレーム数

    Returns:
        list of (timestamp_str, frame_bytes): タイムスタンプとフレーム画像のペア
    """
    frames: list[tuple[str, bytes]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        video_path = tmpdir_path / "input.mp4"
        video_path.write_bytes(video_bytes)

        # 動画の長さを取得
        duration_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

        # フレーム数を計算
        num_frames = min(int(duration / interval_seconds) + 1, max_frames)

        # フレームを抽出
        for i in range(num_frames):
            timestamp = i * interval_seconds
            if timestamp >= duration:
                break

            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"

            frame_path = tmpdir_path / f"frame_{i:04d}.jpg"

            extract_cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(frame_path),
            ]
            subprocess.run(
                extract_cmd,
                capture_output=True,
                check=True,
            )

            if frame_path.exists():
                frames.append((timestamp_str, frame_path.read_bytes()))

    return frames


def analyze_frame_with_ollama(
    frame_bytes: bytes,
    timestamp: str,
    context: str = "",
) -> dict:
    """
    Ollamaを使用してフレームを解析する

    Args:
        frame_bytes: フレーム画像のバイトデータ
        timestamp: フレームのタイムスタンプ（MM:SS形式）
        context: 前のフレームからの文脈情報

    Returns:
        dict: 解析結果
    """
    image_base64 = base64.b64encode(frame_bytes).decode("utf-8")

    prompt = f"""このフレームは動画の {timestamp} 時点のものです。

{f"前のシーンの文脈: {context}" if context else ""}

以下の情報をJSON形式で回答してください（日本語で）:
{{
    "scene_description": "このシーンで何が起きているかの詳細な説明",
    "visual_elements": "画面に映っている主要な要素（人物、物体、テキストなど）",
    "audio_hint": "このシーンで想定される音声や会話の内容（推測）",
    "is_scene_change": true/false（前のシーンから大きく変化したか）,
    "scene_type": "intro/main/transition/conclusion/other"
}}

JSONのみを出力してください。"""

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.3,
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    result_text = response.json()["response"]

    # JSONを抽出
    try:
        # ```json ... ``` ブロックを除去
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        return json.loads(result_text.strip())
    except json.JSONDecodeError:
        return {
            "scene_description": result_text,
            "visual_elements": "",
            "audio_hint": "",
            "is_scene_change": False,
            "scene_type": "other",
        }


def build_video_structure(
    frames_analysis: list[tuple[str, dict]],
    video_duration_seconds: float,
) -> VideoAnalysisResult:
    """
    フレーム解析結果から階層構造を構築する

    Args:
        frames_analysis: (timestamp, analysis_result) のリスト
        video_duration_seconds: 動画の長さ（秒）

    Returns:
        VideoAnalysisResult: 構造化された動画分析結果
    """
    if not frames_analysis:
        raise ValueError("フレーム解析結果が空です")

    # セグメントを作成
    segments: list[SegmentNode] = []
    current_chapter_segments: list[SegmentNode] = []
    chapters: list[ChapterNode] = []

    for i, (timestamp, analysis) in enumerate(frames_analysis):
        # 次のフレームのタイムスタンプを取得（終了時刻として使用）
        if i + 1 < len(frames_analysis):
            end_timestamp = frames_analysis[i + 1][0]
        else:
            # 最後のフレームは動画の終わりまで
            end_minutes = int(video_duration_seconds // 60)
            end_seconds = int(video_duration_seconds % 60)
            end_timestamp = f"{end_minutes:02d}:{end_seconds:02d}"

        # start_time と end_time が同じ場合はスキップ
        if timestamp == end_timestamp:
            continue

        segment = SegmentNode(
            title=analysis.get("scene_type", "シーン").title() + f" ({timestamp})",
            description=analysis.get("scene_description", ""),
            time_span=TimeSpan(
                start_time=timestamp,
                end_time=end_timestamp,
            ),
        )

        # シーン変化があればチャプターを区切る
        if analysis.get("is_scene_change", False) and current_chapter_segments:
            chapter = ChapterNode(
                title=f"チャプター {len(chapters) + 1}",
                summary=current_chapter_segments[0].description[:100] + "...",
                children=current_chapter_segments,
            )
            chapters.append(chapter)
            current_chapter_segments = []

        current_chapter_segments.append(segment)
        segments.append(segment)

    # 残りのセグメントをチャプターにまとめる
    if current_chapter_segments:
        chapter = ChapterNode(
            title=f"チャプター {len(chapters) + 1}",
            summary=current_chapter_segments[0].description[:100] + "...",
            children=current_chapter_segments,
        )
        chapters.append(chapter)

    # 全体の概要を生成
    overview_parts = [seg.description for seg in segments[:3]]
    overview = " ".join(overview_parts)[:200] + "..."

    return VideoAnalysisResult(
        video_title="動画分析結果",
        overview=overview,
        children=chapters if len(chapters) > 1 else segments,
    )


def get_video_duration(video_bytes: bytes) -> float:
    """動画の長さを取得する（秒）"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            temp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    finally:
        Path(temp_path).unlink(missing_ok=True)


def index_video_ollama(
    video_bytes: bytes,
    interval_seconds: int = 10,
    max_frames: int = 30,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> VideoAnalysisResult:
    """
    Ollama + Qwen3-VLを使用して動画をインデックス化する

    Args:
        video_bytes: 動画ファイルのバイトデータ
        interval_seconds: フレーム抽出間隔（秒）
        max_frames: 最大フレーム数
        progress_callback: 進捗コールバック (current, total, message)

    Returns:
        VideoAnalysisResult: 構造化された動画分析結果

    Raises:
        ValueError: 動画の分析に失敗した場合
    """
    def report_progress(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)
        print(f"[{current}/{total}] {message}")  # ログ出力

    # 動画の長さを取得
    report_progress(0, 3, "動画情報を取得中...")
    duration = get_video_duration(video_bytes)

    # フレームを抽出
    report_progress(1, 3, f"フレームを抽出中（{interval_seconds}秒間隔）...")
    frames = extract_frames(
        video_bytes,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )

    if not frames:
        raise ValueError("動画からフレームを抽出できませんでした")

    report_progress(2, 3, f"{len(frames)}フレームを抽出完了。解析開始...")

    # 各フレームを解析
    frames_analysis: list[tuple[str, dict]] = []
    context = ""
    total_frames = len(frames)

    for i, (timestamp, frame_bytes) in enumerate(frames):
        report_progress(i + 1, total_frames, f"フレーム {timestamp} を解析中...")
        analysis = analyze_frame_with_ollama(
            frame_bytes,
            timestamp,
            context=context,
        )
        frames_analysis.append((timestamp, analysis))

        # 次のフレームの文脈として使用
        context = analysis.get("scene_description", "")

    # 階層構造を構築
    result = build_video_structure(frames_analysis, duration)

    if not result.children:
        raise ValueError(
            f"動画の分析結果にセグメントが含まれていません。\n"
            f"タイトル: {result.video_title}\n"
            f"概要: {result.overview}\n"
            f"動画が短すぎるか、分析に失敗した可能性があります。"
        )

    return result
