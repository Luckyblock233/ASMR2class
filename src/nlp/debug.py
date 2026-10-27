"""调试输出：格式化每次 LLM 块翻译的请求与响应（--debug 模式）。

仅输出 prompt 与响应内容本身，绝不包含 api_key（密钥不参与 prompt，亦不打印）。
输出到 stdout，可重定向：uv run python -m src.main --debug ... > debug.log 2>&1
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm_client import ChatResult


@dataclass
class Attempt:
    """一次块翻译尝试的记录（含重试与逐句兜底）。"""

    ids: list[int]
    with_context: bool
    prompt: str | None = None
    response: ChatResult | None = None
    error: str | None = None  # 请求异常或解析/对齐失败原因
    pairs: list[tuple[str, str]] | None = None  # 对齐成功的 (原文, 译文)

    @property
    def ok(self) -> bool:
        return self.pairs is not None


def format_block_exchange(
    block_no: int,
    texts: list[str],
    attempts: list[Attempt],
    memory_note: str,
) -> str:
    """把一个块的完整交涉过程格式化为可读的调试文本。"""
    lines: list[str] = []
    bar = "=" * 70
    lines.append(bar)
    lines.append(f"LLM 块翻译 #{block_no}（{len(texts)} 句，尝试 {len(attempts)} 次）")
    lines.append(bar)
    for i, att in enumerate(attempts, 1):
        kind = "块翻译" if att.with_context else "逐句兜底"
        lines.append(f"\n----- 尝试 {i}/{len(attempts)}：{kind}（{len(att.ids)} 句，ids={att.ids}）-----")
        if att.prompt is not None:
            lines.append("[请求 prompt]")
            lines.append(att.prompt)
        if att.response is not None:
            r = att.response
            lines.append(
                f"[响应] 耗时 {r.latency_sec:.2f}s，"
                f"tokens {r.prompt_tokens}+{r.completion_tokens}={r.total_tokens}，"
                f"json_mode={'开' if r.json_mode else '关'}，第 {r.attempts} 次网络尝试"
            )
            lines.append(att.response.content)
        if att.error:
            lines.append(f"[失败] {att.error}")
        if att.pairs is not None:
            lines.append(f"[对齐] 成功 {len(att.pairs)}/{len(att.ids)} 句")
    lines.append("\n[记忆回写] " + (memory_note or "无变更"))
    lines.append("[最终输出]")
    final = attempts[-1].pairs if attempts else None
    if final and len(final) == len(texts):
        for src, tgt in final:
            lines.append(f"  {src}  →  {tgt or '(仅原文)'}")
    else:
        for t in texts:
            lines.append(f"  {t}  →  (见各次尝试)")
    lines.append("")
    return "\n".join(lines)


@dataclass
class GrammarAttempt:
    """一次语法分析尝试的记录。"""

    attempt_no: int
    prompt: str | None = None
    response: ChatResult | None = None
    error: str | None = None
    stage: str = ""


def format_grammar_exchange(
    segment_id: int,
    sentence: str,
    attempts: list[GrammarAttempt],
) -> str:
    """把一句语法分析的完整交涉过程格式化为可读的调试文本（--debug 模式）。"""
    lines: list[str] = []
    bar = "=" * 70
    lines.append(bar)
    lines.append(f"语法分析 句段#{segment_id}（尝试 {len(attempts)} 次）")
    lines.append(f"原文：{sentence}")
    lines.append(bar)
    for att in attempts:
        lines.append(f"\n----- 尝试 {att.attempt_no}/{len(attempts)} -----")
        if att.prompt is not None:
            lines.append("[请求 prompt]")
            lines.append(att.prompt)
        if att.response is not None:
            r = att.response
            lines.append(
                f"[响应] 耗时 {r.latency_sec:.2f}s，"
                f"tokens {r.prompt_tokens}+{r.completion_tokens}={r.total_tokens}，"
                f"第 {r.attempts} 次网络尝试"
            )
            lines.append(r.content)
        if att.error:
            lines.append(f"[失败] {att.error}")
    lines.append("")
    return "\n".join(lines)
