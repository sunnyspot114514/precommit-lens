from __future__ import annotations

from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent


def read_text(path: str) -> str:
    target = ROOT / path
    if not target.exists():
        return f"Missing file: {path}"
    return target.read_text(encoding="utf-8", errors="replace")


SUMMARY_PATH = "results/dense_jlens_qwen_fulllayers_4fit/Qwen__Qwen3-0.6B/dense_jlens_summary.md"
INTERPRETATION_PATH = "results/QWEN3_DENSE_JLENS_INTERPRETATION.md"


with gr.Blocks(title="PreCommitLens") as demo:
    gr.Markdown(
        """
        # PreCommitLens

        Lightweight Jacobian-lens reproduction and runtime governance probe.

        This preview is a static result browser. It does not download models or
        fit Jacobians on the free Space runtime.
        """
    )
    with gr.Tabs():
        with gr.Tab("Dense J-Lens Summary"):
            gr.Markdown(read_text(SUMMARY_PATH))
        with gr.Tab("Interpretation"):
            gr.Markdown(read_text(INTERPRETATION_PATH))
        with gr.Tab("中文说明"):
            gr.Markdown(read_text("README.zh-CN.md"))


if __name__ == "__main__":
    demo.launch()
