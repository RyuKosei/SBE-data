from __future__ import annotations

import math
from typing import Any


GENERATION_SYSTEM_PROMPT = (
    "你是中文短文本生成器。不要展示推理、分析或中间步骤；任何情况下都只输出最终中文正文。"
)


def ratio_text(alpha: float) -> str:
    known = {
        0.0: "alpha = 0（100% 风格A + 0% 风格B）",
        1 / 6: "alpha = 1/6（约 83.33% 风格A + 16.67% 风格B）",
        2 / 6: "alpha = 2/6（约 66.67% 风格A + 33.33% 风格B）",
        3 / 6: "alpha = 3/6（50% 风格A + 50% 风格B）",
        4 / 6: "alpha = 4/6（约 33.33% 风格A + 66.67% 风格B）",
        5 / 6: "alpha = 5/6（约 16.67% 风格A + 83.33% 风格B）",
        1.0: "alpha = 1（0% 风格A + 100% 风格B）",
    }
    for value, label in known.items():
        if math.isclose(alpha, value, abs_tol=1e-12):
            return label
    raise ValueError(f"unsupported target ratio {alpha}")


def render_qwen_prompt(scenario: dict[str, Any], alpha: float, template_id: str) -> str:
    minimum, maximum = scenario.get("length_range_zh_chars", [80, 140])
    if template_id == "computers_explicit_ratio_v1":
        return (
            "/no_think\n"
            "严禁输出 Thinking Process、Analysis、推理过程、步骤说明、标题或 Markdown 列表。\n"
            "你只能输出最终中文正文，第一句话就必须是正文内容。\n\n"
            "你将根据一个固定核心语义生成中文短文本。\n\n"
            f"核心语义：\n{scenario['content']}\n\n"
            f"风格A：\n{scenario['style_a']}\n\n"
            f"风格B：\n{scenario['style_b']}\n\n"
            f"目标风格比例：\n{ratio_text(alpha)}。\n\n"
            "硬性要求：\n"
            "1. 必须保留核心语义，不添加与核心语义冲突的新事实。\n"
            f"2. 正文必须至少 {minimum} 个汉字，最多 {maximum} 个汉字；少于 {minimum} 个汉字不合格。\n"
            "3. 只输出正文，不要解释比例，不要列标题。\n"
            "4. 不要输出任何英文分析，不要输出 Thinking Process。"
        )
    if template_id == "computers_explicit_ratio_v2":
        return (
            "/no_think\n只提交最终中文正文，不展示任何构思、推理、分析、标题或列表。\n\n"
            "请忠实表达下列固定内容，并将语言风格调到指定的连续位置。\n\n"
            f"固定内容：\n{scenario['content']}\n\n"
            f"连续风格区间的A端：\n{scenario['style_a']}\n\n"
            f"连续风格区间的B端：\n{scenario['style_b']}\n\n"
            f"所需B端强度：\n{ratio_text(alpha)}。\n\n"
            "写作约束：\n"
            "- 完整保留固定内容，不引入冲突事实。\n"
            f"- 正文必须在 {minimum} 至 {maximum} 个汉字之间。\n"
            "- 不解释比例，不使用标题或 Markdown，只输出正文。"
        )
    raise ValueError(f"unknown prompt template: {template_id}")


def render_external_prompt(scenario: dict[str, Any], alpha: float, template_id: str) -> tuple[str, str]:
    original = render_qwen_prompt(scenario, alpha, template_id)
    sent = original.removeprefix("/no_think\n")
    if sent == original:
        raise ValueError("expected Qwen /no_think wrapper was not present")
    return original, sent
