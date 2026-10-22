"""Prompt 模板引擎（FR-5）：外部模板文件 + 变量渲染 + mtime 热重载。

- 模板为纯文本文件，`#` 开头的行视为注释（用于说明可用变量），渲染前剔除。
- 变量语法：{name}。未提供值的变量会抛出带文件与行号的 PromptTemplateError。
- 每次渲染前检查 mtime，修改后无需重启即生效（FR-5.3）。
- 语言专属模板缺失时回落到 _default/ 目录（FR-5.2）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_VAR_RE = re.compile(r"\{(\w+)\}")


class PromptTemplateError(Exception):
    pass


@dataclass
class _CachedTemplate:
    mtime: float
    body: str  # 已剔除注释行的模板正文


class PromptEngine:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[Path, _CachedTemplate] = {}

    def resolve(self, rel_path: str | None, kind: str) -> Path:
        """解析模板路径：档案指定路径 → _default/<kind>.txt 兜底。"""
        candidates = []
        if rel_path:
            candidates.append(self.prompts_dir / rel_path)
        candidates.append(self.prompts_dir / "_default" / f"{kind}.txt")
        for c in candidates:
            if c.is_file():
                return c
        raise PromptTemplateError(
            f"未找到 prompt 模板：{rel_path or kind}（已尝试："
            + ", ".join(str(c) for c in candidates)
            + "）"
        )

    def resolve_pair(
        self, kind: str, rel_path: str | None, source_lang: str, target_lang: str
    ) -> Path:
        """语言对感知解析：<源>-<目标>/<kind>.txt → 包内路径 → _default 兜底。

        自由语言对模式下，pair 目录（如 ja-en/）承载特定语言对的定制模板；
        源语言为 auto 时跳过 pair 目录。
        """
        candidates = []
        if source_lang and source_lang != "auto" and target_lang:
            candidates.append(
                self.prompts_dir / f"{source_lang}-{target_lang}" / f"{kind}.txt"
            )
        if rel_path:
            candidates.append(self.prompts_dir / rel_path)
        candidates.append(self.prompts_dir / "_default" / f"{kind}.txt")
        for c in candidates:
            if c.is_file():
                return c
        raise PromptTemplateError(
            f"未找到 prompt 模板：{kind}（{source_lang}→{target_lang}，已尝试："
            + ", ".join(str(c) for c in candidates)
            + "）"
        )

    def _load(self, path: Path) -> str:
        mtime = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached and cached.mtime == mtime:
            return cached.body
        text = path.read_text(encoding="utf-8")
        body = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        ).strip()
        self._cache[path] = _CachedTemplate(mtime=mtime, body=body)
        return body

    def template_version(self, path: Path) -> int:
        """模板版本号（FR-4.5 缓存 key 用）：以 mtime 的整数秒近似。"""
        return int(path.stat().st_mtime)

    def variables(self, path: Path) -> set[str]:
        """模板中声明的变量名集合（用于 FR-C4.1 的机制自动停用检测）。"""
        return set(_VAR_RE.findall(self._load(path)))

    def render(self, path: Path, **variables: str) -> str:
        body = self._load(path)

        def _sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name in variables:
                return str(variables[name])
            line_no = body[: m.start()].count("\n") + 1
            raise PromptTemplateError(
                f"模板 {path} 第 {line_no} 行引用了未提供的变量 {{{name}}}"
            )

        return _VAR_RE.sub(_sub, body)
