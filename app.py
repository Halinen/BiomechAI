"""
app.py — Gradio Web 界面

功能：
  - 文字输入（描述姿势问题或动作异常）
  - 图片上传（姿势照片、动作视频截图）
  - 层级选择（可选：让用户指定评估方向）
  - 流式回答（打字机效果）
  - 多轮对话（带"清空对话"按钮）

运行方式：
  python app.py
  → 浏览器打开 http://localhost:7860

依赖：
  pip install gradio anthropic chromadb sentence-transformers python-dotenv
"""

import sys
import os
from pathlib import Path

# 加入项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载 .env 文件中的 ANTHROPIC_API_KEY
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from rag_system.agent import AssessmentAgent


# ── 初始化 Agent ────────────────────────────────────────────────────────────

# 全局 agent 实例，整个 UI 会话共用
# （多用户场景下需要每个用户独立实例，这里是单用户本地工具）
agent = AssessmentAgent()


# ── Gradio 回调函数 ─────────────────────────────────────────────────────────

def chat(
    user_input: str,
    image: gr.Image | None,
    layer_choice: str,
    history: list[dict],
):
    """
    Gradio 6 聊天回调。history 格式为 messages 列表：
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    if not user_input.strip():
        yield history
        return

    # 处理图片
    image_path = None
    if image is not None:
        import tempfile
        from PIL import Image as PILImage
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        PILImage.fromarray(image).save(tmp.name)
        image_path = tmp.name

    layer = None if layer_choice == "自动判断" else layer_choice

    # 追加用户消息，assistant 先占位
    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ""},
    ]

    accumulated = ""
    for chunk in agent.ask_stream(user_input, image_path, layer):
        accumulated += chunk
        history[-1]["content"] = accumulated
        yield history

    if image_path:
        try:
            os.unlink(image_path)
        except Exception:
            pass


def clear_chat():
    """清空对话历史（重置 agent + UI）。"""
    agent.reset()
    return [], ""


# ── Gradio UI 布局 ──────────────────────────────────────────────────────────

with gr.Blocks(title="AI 动作评估助手") as demo:

    gr.Markdown("""
    # 🏃 AI 动作评估助手
    描述你的姿势问题或动作代偿，可选上传照片，系统会从专业知识库中检索相关内容并给出评估建议。

    **知识库涵盖：** FMS/SFMA（动作筛查）· NASM CES（纠正训练）· PRI（呼吸/骨盆）· 红旗症状
    """)

    with gr.Row():
        # 左侧：对话区域
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话",
                height=520,
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="描述你的问题，例如：我深蹲时膝盖总是内扣，背部也会圆...",
                    label="",
                    scale=5,
                    lines=2,
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

        # 右侧：辅助输入区
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="上传姿势图片（可选）",
                type="numpy",
                elem_classes=["upload-btn"],
                height=200,
            )
            layer_selector = gr.Dropdown(
                choices=["自动判断", "控制层", "结构层", "输出层"],
                value="自动判断",
                label="评估方向",
                info="选择后只检索对应知识域，不确定请选'自动判断'"
            )
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

            gr.Markdown("""
            **三层框架说明：**
            - **控制层** → PRI：呼吸模式、骨盆感知
            - **结构层** → FMS/SFMA：关节活动度、软组织
            - **输出层** → NASM CES：肌肉失衡、代偿模式
            """)

    # ── 事件绑定 ────────────────────────────────────────────────────────────

    # 点击"发送"按钮
    submit_btn.click(
        fn=chat,
        inputs=[user_input, image_input, layer_selector, chatbot],
        outputs=[chatbot],
    ).then(
        # 发送后清空输入框
        fn=lambda: "",
        outputs=[user_input],
    )

    # 按 Enter 键也能发送（Shift+Enter 换行）
    user_input.submit(
        fn=chat,
        inputs=[user_input, image_input, layer_selector, chatbot],
        outputs=[chatbot],
    ).then(
        fn=lambda: "",
        outputs=[user_input],
    )

    # 清空对话
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, user_input],
    )


# ── 启动 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # share=False：只在本机运行（本地工具，不需要公网分享）
    # server_port=7860：默认端口，浏览器访问 http://localhost:7860
    demo.launch(
        server_port=7863,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=".chatbot { font-size: 15px; }",
    )
