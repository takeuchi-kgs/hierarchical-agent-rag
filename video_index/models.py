import re
from typing import List, Union

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from .callbacks import AttachVideoToLlmRequestCallback


# [時間ポインタ]
class TimeSpan(BaseModel):
    start_time: str = Field(..., description="開始タイムスタンプ（MM:SS形式）")
    end_time: str = Field(..., description="終了タイムスタンプ（MM:SS形式）")

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """start_time、end_timeがMM:SS形式であることを検証"""
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError(f"無効な時間形式: {v}。MM:SS形式である必要があります。")
        return v

    @model_validator(mode="after")
    def validate_time_order(self) -> "TimeSpan":
        """start_timeがend_timeより厳密に前であることを確認"""
        if self.start_time >= self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) はend_time ({self.end_time}) より前である必要があります"
            )
        return self


# [レベル2: リーフ] 原子的コンテンツ単位
class SegmentNode(BaseModel):
    node_type: str = "Segment"
    title: str = Field(..., description="この動画セグメントの簡潔なタイトル")
    description: str = Field(
        ..., description="コンテンツの詳細な説明（映像、音声、イベント）"
    )
    # リーフノードのみ時間情報を持つ
    time_span: TimeSpan = Field(..., description="正確なタイムスタンプ範囲")

    @computed_field
    @property
    def id(self) -> str:
        start = self.time_span.start_time.replace(":", "")
        end = self.time_span.end_time.replace(":", "")
        return f"{self.node_type}_{start}_{end}"

    def to_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-flash",
            name=self.id,
            description=f"「{self.title}」の専門エージェント。時間範囲: {self.time_span.start_time} 〜 {self.time_span.end_time}。このセグメントの映像・音声・イベントに関する質問に回答できます。",
            instruction=f"""あなたは動画セグメント「{self.title}」の専門エージェントです。

## 担当範囲
- 時間: {self.time_span.start_time} 〜 {self.time_span.end_time}

## セグメント内容
{self.description}

## 行動指針
1. 上記のセグメント内容に基づいて、ユーザーの質問に正確に回答してください
2. このセグメントに含まれる情報のみを使用し、推測や外部知識は避けてください
3. 質問がこのセグメントの範囲外の場合は、その旨を明確に伝えてください
4. 映像の視覚的要素、音声、発言内容、イベントの時系列について詳細に説明できます""",
            before_model_callback=AttachVideoToLlmRequestCallback(
                start_time=self.time_span.start_time, end_time=self.time_span.end_time
            ),
            output_key=f"{self.id}_response",
        )


# [レベル1: メインユニット] 主要区分（例：チャプター、トピック）
class ChapterNode(BaseModel):
    node_type: str = "Chapter"
    title: str = Field(..., description="この主要チャプターまたはトピックのタイトル")
    summary: str = Field(..., description="チャプターの要約")
    # Time Spanは削除（子要素から自動計算）
    # [柔軟性] SectionまたはSegmentを子要素として持てる
    children: List[SegmentNode] = Field(default_factory=list)

    @computed_field
    @property
    def time_span(self) -> TimeSpan:
        if not self.children:
            # デフォルトの時間範囲を返す（childrenが空の場合）
            return TimeSpan(start_time="00:00", end_time="00:01")
        min_start_time = min(child.time_span.start_time for child in self.children)
        max_end_time = max(child.time_span.end_time for child in self.children)
        return TimeSpan(start_time=min_start_time, end_time=max_end_time)

    @computed_field
    @property
    def id(self) -> str:
        start = self.time_span.start_time.replace(":", "")
        end = self.time_span.end_time.replace(":", "")
        return f"{self.node_type}_{start}_{end}"

    def to_agent(self) -> Agent:
        # 子要素の情報を整形
        children_info = "\n".join(
            f"- [{child.time_span.start_time}〜{child.time_span.end_time}] {child.title}: {child.summary if hasattr(child, 'summary') else child.description}"
            for child in self.children
        )
        agent_tools = [AgentTool(agent=child.to_agent()) for child in self.children]
        return Agent(
            model="gemini-2.5-flash",
            name=self.id,
            description=f"「{self.title}」の専門エージェント。時間範囲: {self.time_span.start_time} 〜 {self.time_span.end_time}。このチャプターの映像・音声・イベントに関する質問に回答できます。",
            instruction=f"""あなたは動画チャプター「{self.title}」の専門エージェントです。

## 担当範囲
- 時間: {self.time_span.start_time} 〜 {self.time_span.end_time}

## チャプター概要
{self.summary}

## 含まれる子要素
{children_info}

## 行動指針
1. 上記のチャプター概要と子要素情報に基づいて、ユーザーの質問に正確に回答してください
2. このチャプターに含まれる情報のみを使用し、推測や外部知識は避けてください
3. 質問がこのチャプターの範囲外の場合は、その旨を明確に伝えてください

## ツール活用指針
あなたは各セグメントの専門エージェントをツールとして持っています。
- **詳細な情報が必要な場合は、必ず該当する子エージェントを呼び出してください**
- 質問内容が特定の時間範囲に関連する場合、その時間を担当する子エージェントに委譲してください
- 複数の子要素に関わる質問の場合、関連する全ての子エージェントを順次呼び出して情報を統合してください
- 概要レベルで回答できる質問でも、正確性を高めるために子エージェントで裏付けを取ることを推奨します
- 階層的に深掘りが必要な場合、子エージェントがさらにその子エージェントを呼び出すことで詳細情報を取得できます""",
            output_key=f"{self.id}_response",
            tools=agent_tools,
        )


# [ルート] 動画分析結果
class VideoAnalysisResult(BaseModel):
    video_title: str = Field(..., description="動画のタイトル/トピック")
    overview: str = Field(..., description="全体の概要")
    # [完全自律] Chapter, Section, Segmentのいずれも最上位に配置可能
    children: List[Union[ChapterNode, SegmentNode]] = Field(default_factory=list)

    @computed_field
    @property
    def time_span(self) -> TimeSpan:
        if not self.children:
            # デフォルトの時間範囲を返す（childrenが空の場合）
            return TimeSpan(start_time="00:00", end_time="00:01")
        min_start_time = min(child.time_span.start_time for child in self.children)
        max_end_time = max(child.time_span.end_time for child in self.children)
        return TimeSpan(start_time=min_start_time, end_time=max_end_time)

    @computed_field
    @property
    def id(self) -> str:
        start = self.time_span.start_time.replace(":", "")
        end = self.time_span.end_time.replace(":", "")
        return f"Video_{start}_{end}"

    def to_agent(self) -> Agent:
        # 子要素の情報を整形
        children_info = "\n".join(
            f"- [{child.time_span.start_time}〜{child.time_span.end_time}] {child.title}: {child.summary if hasattr(child, 'summary') else child.description}"
            for child in self.children
        )
        agent_tools = [AgentTool(agent=child.to_agent()) for child in self.children]
        return Agent(
            model="gemini-2.5-flash",
            name=self.id,
            description=f"「{self.video_title}」の専門エージェント。時間範囲: {self.time_span.start_time} 〜 {self.time_span.end_time}。この動画全体の映像・音声・イベントに関する質問に回答できます。",
            instruction=f"""あなたは動画「{self.video_title}」の専門エージェントです。

## 担当範囲
- 時間: {self.time_span.start_time} 〜 {self.time_span.end_time}

## 動画概要
{self.overview}

## 含まれる子要素
{children_info}

## 行動指針
1. 上記の動画概要と子要素情報に基づいて、ユーザーの質問に正確に回答してください
2. この動画に含まれる情報のみを使用し、推測や外部知識は避けてください

## ツール活用指針
あなたは各チャプター/セグメントの専門エージェントをツールとして持っています。
- **詳細な情報が必要な場合は、必ず該当する子エージェントを呼び出してください**
- 質問内容が特定の時間範囲に関連する場合、その時間を担当する子エージェントに委譲してください
- 複数の子要素に関わる質問の場合、関連する全ての子エージェントを順次呼び出して情報を統合してください
- 概要レベルで回答できる質問でも、正確性を高めるために子エージェントで裏付けを取ることを推奨します
- 階層的に深掘りが必要な場合、子エージェントがさらにその子エージェントを呼び出すことで詳細情報を取得できます
- 動画全体に関する質問でも、まず関連しそうな子エージェントから情報を収集してから回答を構築してください""",
            output_key=f"{self.id}_response",
            tools=agent_tools,
        )
