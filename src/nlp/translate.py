"""块级上下文感知翻译调度（专项文档 FR-C1/C2/C3/C4 + 模型可插拔架构）。

链路：blocker 成块 → 取翻译槽当前 Provider → 渲染 prompt（注入历史 + 滚动记忆）
     → LLM（JSON 契约 / instTrans 纯文本契约）→ 对齐校验 → 回写历史与记忆。

Provider 接入（模型可插拔 §1.1/§4）：
- Translator 不缓存 Provider：每块翻译经 provider_getter() 取当前绑定，
  取引用与发请求之间被换绑是合法状态，请求失败走降级链，不得崩溃
- Provider 声明 glossary_constraint 能力时走 instTrans 映射（术语表→【硬性要求】、
  风格→【注意】，响应为多行纯文本按行对齐）；否则走通用 JSON 契约模板
- 翻译槽未绑定（provider_getter 返回 None）→ 仅显示原文，不发起请求

降级链（FR-C4.3）：整块 JSON/对齐失败 → 重试 1 次 → 拆为逐句无上下文翻译
→ 单句仍失败则该句仅保留原文（主 PRD FR-1.6）。任何环节都不打断字幕流。

机制开关（FR-C4.1/4.2）：
- 模板缺失 {context_history}/{glossary}/{style_notes} 变量 → 对应机制自动停用（日志提示）
- context.tail_sentences=0 关闭滑动窗口；memory.enabled=false 关闭滚动记忆；
  block.max_sentences=1 时 blocker 自然退化为逐句
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from ..config import ContextConfig
from ..profiles.loader import LanguageProfile
from .context_memory import RollingMemory
from .debug import Attempt, format_block_exchange
from .prompt_engine import PromptEngine
from .providers.capabilities import CAP_GLOSSARY_CONSTRAINT

log = logging.getLogger(__name__)

# instTrans 模板用中文语言名（官方客户端约定）；未收录的代码回落为原样
LANG_NAMES_ZH = {"ja": "日语", "en": "英语", "zh": "中文", "ko": "韩语", "fr": "法语",
                 "de": "德语", "es": "西班牙语", "ru": "俄语"}


def lang_name_zh(code: str) -> str:
    return LANG_NAMES_ZH.get(code, code)


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 字符按 1，其余按 ~4 字符/token。"""
    cjk = sum(1 for c in text if ord(c) > 0x2E7F)
    return cjk + (len(text) - cjk + 3) // 4


def _extract_json(content: str) -> dict:
    """从响应中提取 JSON 对象（容忍 ```json 围栏与前后杂散文本）。"""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        return json.loads(m.group(1) if m.lastindex else m.group(0))
    raise json.JSONDecodeError("no JSON object found", content, 0)


@dataclass
class TranslationResult:
    source: str
    translation: str | None  # None = 降级（离线/失败），仅保留原文


class Translator:
    """块级翻译器：持有滑动历史与滚动记忆，对外接口为 translate_block。

    provider_getter：零参 callable，每次返回翻译槽当前绑定的 Provider
    （具备 chat()/online；None = 槽位未绑定，翻译停用仅显示原文）。
    兼容旧用法：直接传一个带 chat() 的客户端会自动包装为固定 getter。
    """

    def __init__(
        self,
        provider_getter,
        prompts: PromptEngine,
        profile: LanguageProfile,
        context_cfg: ContextConfig | None = None,
        memory: RollingMemory | None = None,
        debug=None,
    ):
        if callable(provider_getter):
            self._get_provider: Callable[[], Any] = provider_getter
        else:  # 兼容：直接传 LLMClient/Provider 对象
            self._get_provider = lambda: provider_getter  # noqa: E731
        self.prompts = prompts
        self.profile = profile
        self.debug = debug  # 可调用：接收格式化调试文本（--debug 模式）
        self._block_no = 0
        self.cfg = (context_cfg or ContextConfig()).merged(profile.context)
        self.memory = memory or RollingMemory(
            glossary_max_entries=self.cfg.memory.glossary_max_entries,
            style_max_chars=self.cfg.memory.style_max_chars,
            style_preset=profile.style_preset,
        )
        self._history: deque[tuple[str, str]] = deque(maxlen=max(self.cfg.tail_sentences, 1))
        self._tokens_used = 0
        self._lock = threading.Lock()
        self._sentence_id = 0

        # FR-C4.1：检测模板实际声明的变量，缺失的机制自动停用（JSON 与 instTrans 两套模板分别检测）
        self._tpl_json = self.prompts.resolve(self.profile.prompt_path("translate"), "translate")
        self._flags_json = self._mechanism_flags(self._tpl_json)
        # instTrans 模板惰性解析：仅在 Provider 声明 glossary_constraint 时需要
        self._tpl_insttrans = None
        self._flags_insttrans: tuple[bool, bool, bool] | None = None

    def _insttrans_template(self):
        """解析 instTrans 模板（含变量检测），缓存结果；模板缺失抛 PromptTemplateError。"""
        if self._tpl_insttrans is None:
            self._tpl_insttrans = self.prompts.resolve(
                self.profile.prompt_path("translate_insttrans"), "translate_insttrans"
            )
            self._flags_insttrans = self._mechanism_flags(self._tpl_insttrans)
        flags = self._flags_insttrans
        assert flags is not None
        return self._tpl_insttrans, flags

    def _mechanism_flags(self, path) -> tuple[bool, bool, bool]:
        """(use_history, use_glossary, use_style)，模板缺变量则对应机制停用。"""
        declared = self.prompts.variables(path)
        use_history = self.cfg.window_enabled and "context_history" in declared
        use_glossary = self.cfg.memory.enabled and "glossary" in declared
        use_style = self.cfg.memory.enabled and "style_notes" in declared
        if self.cfg.window_enabled and not use_history:
            log.info("模板 %s 未声明 {context_history}，滑动窗口机制停用", path.name)
        if self.cfg.memory.enabled and not use_glossary:
            log.info("模板 %s 未声明 {glossary}，术语表照常采集但不注入 prompt", path.name)
        if self.cfg.memory.enabled and not use_style:
            log.info("模板 %s 未声明 {style_notes}，风格摘要照常采集但不注入 prompt", path.name)
        return use_history, use_glossary, use_style

    @staticmethod
    def _is_insttrans(provider) -> bool:
        """Provider 声明术语表硬约束能力 → 走 instTrans 契约（纯文本响应）。"""
        caps = getattr(getattr(provider, "cfg", None), "capabilities", None) or []
        return CAP_GLOSSARY_CONSTRAINT in caps

    @property
    def online(self) -> bool:
        provider = self._get_provider()
        return bool(provider is not None and getattr(provider, "online", False))

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    # ---- 对外接口 ----

    def translate(self, text: str) -> TranslationResult:
        """单句便捷接口（等价于 translate_block([text])[0]）。"""
        return self.translate_block([text])[0]

    def translate_block(self, texts: list[str]) -> list[TranslationResult]:
        texts = [t.strip() for t in texts]
        if not texts or not any(texts):
            return [TranslationResult(source=t, translation=None) for t in texts]
        if self._get_provider() is None:
            # 翻译槽未绑定：仅显示原文，不发起请求
            return [TranslationResult(source=t, translation=None) for t in texts]

        trace: list[Attempt] | None = [] if self.debug else None
        glossary_before = (
            {t: e.translation for t, e in self.memory.glossary.items()} if trace is not None else {}
        )
        style_before = self.memory.style_notes if trace is not None else ""

        results = self._translate_block_inner(texts, with_context=True, trace=trace)
        if results is None:
            # 整块失败：重试 1 次
            log.info("块翻译失败，整块重试 1 次（%d 句）", len(texts))
            results = self._translate_block_inner(texts, with_context=True, trace=trace)
        if results is None:
            # 仍失败：逐句无上下文兜底（FR-C4.3）
            log.warning("块翻译重试仍失败，降级为逐句无上下文翻译")
            out: list[TranslationResult] = []
            for t in texts:
                single = self._translate_block_inner([t], with_context=False, trace=trace)
                out.append(single[0] if single is not None
                            else TranslationResult(source=t, translation=None))
            results = out

        if trace is not None and self.debug:
            self._block_no += 1
            self.debug(format_block_exchange(
                self._block_no, texts, trace,
                self._memory_diff_note(glossary_before, style_before),
            ))
        return results

    def _memory_diff_note(self, before: dict[str, str], style_before: str) -> str:
        notes = []
        for term, trans in self.memory.glossary.items():
            if term not in before:
                notes.append(f"+术语 {term}→{trans.translation}")
            elif before[term] != trans.translation:
                notes.append(f"~术语 {term}: {before[term]}→{trans.translation}")
        if self.memory.style_notes != style_before:
            notes.append(f"风格更新：{self.memory.style_notes or '(清空)'}")
        return "；".join(notes)

    # ---- 内部 ----

    def _next_ids(self, n: int) -> list[int]:
        with self._lock:
            ids = list(range(self._sentence_id + 1, self._sentence_id + 1 + n))
            self._sentence_id += n
        return ids

    def _history_text(self, use_history: bool) -> str:
        if not use_history or not self._history:
            return ""
        # FR-C1.3：token 上限内从最新往最早保留
        budget = self.cfg.history_max_tokens
        kept: list[tuple[str, str]] = []
        for pair in reversed(self._history):
            cost = estimate_tokens(pair[0]) + estimate_tokens(pair[1])
            if budget - cost < 0:
                break
            kept.append(pair)
            budget -= cost
        lines = [f"{src} → {tgt}" for src, tgt in reversed(kept)]
        return "\n".join(lines)

    def _translate_block_inner(
        self, texts: list[str], with_context: bool, trace: list[Attempt] | None = None
    ) -> list[TranslationResult] | None:
        """返回 None = 整块失败（调用方决定重试/兜底）。"""
        provider = self._get_provider()
        if provider is None:
            return None  # 槽位在取引用间隙被解绑：走调用方降级链
        insttrans = self._is_insttrans(provider)
        if insttrans:
            path, flags = self._insttrans_template()
        else:
            path, flags = self._tpl_json, self._flags_json
        use_history, use_glossary, use_style = flags

        ids = self._next_ids(len(texts))
        self.memory.begin_block()
        att = Attempt(ids=ids, with_context=with_context)
        if trace is not None:
            trace.append(att)

        joined_sources = "\n".join(texts)
        self.memory.mark_hits(joined_sources)

        if insttrans:
            # instTrans 契约：多行原文，每行一句（FR-C1 历史同样以多行文本注入）
            sentences_payload = joined_sources
        else:
            sentences_payload = json.dumps(
                [{"id": i, "source": t} for i, t in zip(ids, texts)],
                ensure_ascii=False,
            )
        history = self._history_text(use_history) if with_context else ""
        glossary = self.memory.glossary_prompt() if (with_context and use_glossary) else ""
        style = self.memory.style_prompt() if (with_context and use_style) else ""
        source_lang, target_lang = self.profile.source_lang, self.profile.target_lang
        if insttrans:
            # 官方客户端约定：中文语言名 + 一行式术语对照，空约束显示（无）
            source_lang, target_lang = lang_name_zh(source_lang), lang_name_zh(target_lang)
            glossary = (self.memory.glossary_inline() if (with_context and use_glossary) else "") or "（无）"
            style = style or "（无）"
        # 模板错误（如引用未声明变量）直接抛出并带行号，不走降级链（主 PRD §9）
        att.prompt = self.prompts.render(
            path,
            source_lang=source_lang,
            target_lang=target_lang,
            sentences=sentences_payload,
            context_history=history,
            glossary=glossary,
            style_notes=style,
        )
        try:
            # instTrans 为纯文本契约，不用 json_mode；JSON 契约保持 json_mode
            resp = provider.chat(att.prompt, json_mode=not insttrans)
        except Exception as e:  # noqa: BLE001 —— 网络/服务错误走降级链
            att.error = f"请求异常：{type(e).__name__}: {e}"
            log.warning("块翻译请求失败：%s", type(e).__name__)
            return None
        att.response = resp

        with self._lock:
            self._tokens_used += resp.total_tokens

        # 解析与逐句对齐（FR-C2.4）
        data: dict = {}
        try:
            if insttrans:
                pairs = self._align_lines(resp.content, texts)
            else:
                data = _extract_json(resp.content)
                pairs = self._align(data, ids, texts)
        except Exception as e:  # noqa: BLE001
            log.warning("块翻译响应解析/对齐失败：%s", type(e).__name__)
            # 单句兜底宽容：非 JSON 纯文本直接当作译文（兼容旧式模板，仅 JSON 契约路径；
            # instTrans 路径不允许——多行回显垃圾会经此混进历史，污染后续所有块的上下文）
            if not insttrans and len(texts) == 1 and resp.content and "{" not in resp.content:
                pairs = [(texts[0], resp.content.strip())]
            else:
                att.error = f"解析/对齐失败：{type(e).__name__}: {e}"
                return None
        att.pairs = pairs

        # 回写历史与记忆（只登记确认译文，FR-C2.3 重叠句不重复翻译）
        for src, tgt in pairs:
            if tgt and self._is_clean_translation(src, tgt):
                self._history.append((src, tgt))
            elif tgt:
                log.info("历史回写拦截可疑译文：%r → %r", src, tgt[:50])
        if insttrans:
            # instTrans 纯文本契约无 glossary_updates/style_updates 字段，不回写记忆
            pass
        elif self.cfg.memory.enabled:
            self.memory.merge_updates(
                data.get("glossary_updates") if isinstance(data, dict) else None,
                data.get("style_updates") if isinstance(data, dict) else None,
            )
        return [TranslationResult(source=s, translation=t) for s, t in pairs]

    @staticmethod
    def _is_clean_translation(src: str, tgt: str) -> bool:
        """历史回写卫生检查：prompt 回显/多行杂讯/原文原样返回 不进历史。"""
        if "\n" in tgt or "【" in tgt:
            return False
        return "".join(tgt.split()) != "".join(src.split())

    @staticmethod
    def _align_lines(content: str, texts: list[str]) -> list[tuple[str, str]]:
        """instTrans 纯文本响应按行对齐到请求句数；行数不符抛异常走降级链。

        对齐前先清洗：剥离思考块、剔除 prompt 回显行（含【标记）、折叠连续重复行——
        小模型偶尔会把约束块或上下文行抄进输出，不清洗会污染对齐与历史。
        """
        content = content.strip()
        if "</think>" in content:  # 思考块泄露时剥离（官方客户端同款防御）
            content = content.split("</think>", 1)[1].strip()
        lines: list[str] = []
        for ln in content.splitlines():
            ln = ln.strip()
            if not ln or "【" in ln:
                continue
            if lines and lines[-1] == ln:  # 折叠连续重复行
                continue
            lines.append(ln)
        if len(lines) != len(texts):
            raise ValueError(f"译文行数不匹配：期望 {len(texts)}，实际 {len(lines)}")
        return list(zip(texts, lines))

    def _align(
        self, data: dict, ids: list[int], texts: list[str]
    ) -> list[tuple[str, str]]:
        """响应 JSON 与请求逐句对齐。对齐失败抛异常走降级链。"""
        arr = data.get("sentences")
        if not isinstance(arr, list) or len(arr) != len(texts):
            raise ValueError(f"句数不匹配：期望 {len(texts)}，实际 {len(arr) if isinstance(arr, list) else 'N/A'}")
        by_id = {}
        for item in arr:
            if isinstance(item, dict) and "translation" in item:
                by_id[item.get("id")] = str(item.get("translation") or "")
        pairs = []
        for pos, (sid, src) in enumerate(zip(ids, texts)):
            if sid in by_id:
                pairs.append((src, by_id[sid]))
            elif isinstance(arr[pos], dict) and "translation" in arr[pos]:
                # id 对不上但顺序可用：按位置对齐
                pairs.append((src, str(arr[pos].get("translation") or "")))
            else:
                raise ValueError(f"句 id={sid} 无对应译文")
        return pairs
