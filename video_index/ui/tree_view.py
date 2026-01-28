"""ツリービューレンダリングモジュール"""

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import streamlit as st

from ..models import ChapterNode, SegmentNode, VideoAnalysisResult


@dataclass
class NodeConfig:
    """ノードタイプ別設定"""

    node_class: str
    icon: str
    title: str
    description: str
    time_span: "TimeSpan"
    has_children: bool
    children: list


def get_node_config(
    node: Union[VideoAnalysisResult, ChapterNode, SegmentNode],
) -> NodeConfig:
    """ノードタイプに応じた設定を返す"""
    if isinstance(node, VideoAnalysisResult):
        return NodeConfig(
            node_class="video-node",
            icon="📹",
            title=html.escape(node.video_title),
            description=html.escape(node.overview),
            time_span=node.time_span,
            has_children=len(node.children) > 0,
            children=node.children,
        )
    elif isinstance(node, ChapterNode):
        return NodeConfig(
            node_class="chapter-node",
            icon="📚",
            title=html.escape(node.title),
            description=html.escape(node.summary),
            time_span=node.time_span,
            has_children=len(node.children) > 0,
            children=node.children,
        )
    elif isinstance(node, SegmentNode):
        return NodeConfig(
            node_class="segment-node",
            icon="🎬",
            title=html.escape(node.title),
            description=html.escape(node.description),
            time_span=node.time_span,
            has_children=False,
            children=[],
        )
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def render_tree_node(
    node: Union[VideoAnalysisResult, ChapterNode, SegmentNode],
    level: int = 0,
    node_counter: list = None,
) -> str:
    """再帰的にツリーノードをHTMLでレンダリングする"""
    if node_counter is None:
        node_counter = [0]

    node_id = f"node-{node_counter[0]}"
    node_counter[0] += 1

    config = get_node_config(node)

    time_badge = f"{config.time_span.start_time} - {config.time_span.end_time}"
    indent = level * 20

    # 子ノードHTML生成
    children_html = ""
    if config.has_children:
        children_html = f'<div class="node-children" id="{node_id}-children">'
        for child in config.children:
            children_html += render_tree_node(child, level + 1, node_counter)
        children_html += "</div>"

    # トグルアイコン（子がある場合のみ）
    toggle_icon = (
        f'<span class="toggle-icon" id="{node_id}-icon">▼</span>'
        if config.has_children
        else '<span class="toggle-icon-spacer"></span>'
    )

    # HTML生成
    html_str = f"""
    <div class="tree-node {config.node_class}" style="margin-left: {indent}px;">
        <div class="node-header" onclick="toggleNode('{node_id}-children', '{node_id}-icon')">
            {toggle_icon}
            <span class="node-icon">{config.icon}</span>
            <span class="node-title">{config.title}</span>
            <span class="time-badge">{time_badge}</span>
        </div>
        <div class="node-content">
            <p class="node-description">{config.description}</p>
        </div>
        {children_html}
    </div>
    """

    return html_str


def _load_asset_file(filename: str) -> str:
    """assetsディレクトリからファイルをロードする"""
    assets_dir = Path(__file__).parent / "assets"
    file_path = assets_dir / filename
    return file_path.read_text(encoding="utf-8")


def render_video_tree(indexed_video: VideoAnalysisResult):
    """動画ツリー構造をHTML/CSS/JSでレンダリングする"""
    # ツリーHTML生成
    tree_html = render_tree_node(indexed_video)

    # CSSとJSファイルのロード
    css_content = _load_asset_file("tree_styles.css")
    js_content = _load_asset_file("tree_script.js")

    # 完全なHTMLドキュメント生成
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            {css_content}
        </style>
    </head>
    <body>
        <div class="tree-container">
            {tree_html}
        </div>
        
        <script>
            {js_content}
        </script>
    </body>
    </html>
    """

    # StreamlitにHTML表示
    st.components.v1.html(full_html, height=600, scrolling=True)
