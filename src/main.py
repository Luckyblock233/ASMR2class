"""主程序：实时识别翻译、M3 异步语法学习闭环及会话重载预览。

用法示例：
  uv run python -m src.main                          # 麦克风 + 默认语言档案
  uv run python -m src.main --source loopback        # 系统音内录（仅 Windows）
  uv run python -m src.main --source file --input a.wav --profile ja --grammar
  uv run python -m src.main --replay-session output/<会话目录>
  uv run python -m src.main --no-translate           # 仅识别（离线调试）
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .asr.engine import ASREngine
from .asr.streamer import ASRStreamer, Sentence
from .audio.base import RingBuffer
from .config import GlobalConfig, load_config
from .nlp.prompt_engine import PromptEngine
from .nlp.translate import Translator
from .profiles.loader import get_profile

log = logging.getLogger(__name__)

RING_SECONDS = 300  # 环形缓冲 5 分钟，超出自动丢弃最旧数据


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_source(cfg: GlobalConfig, source: str, input: str | None, loopback_device: str = ""):
    if source == "file":
        from .audio.file_source import FileSource

        if not input:
            raise SystemExit("--source file 需要 --input 指定音频文件")
        return FileSource(input)
    if source == "loopback":
        from .audio.loopback import LoopbackSource

        device = loopback_device or cfg.audio.loopback_device
        return LoopbackSource(
            sample_rate=cfg.audio.sample_rate,
            block_ms=cfg.audio.block_ms,
            device_name=device,
        )
    from .audio.mic import MicSource

    return MicSource(sample_rate=cfg.audio.sample_rate, block_ms=cfg.audio.block_ms)


def list_devices() -> int:
    """列出麦克风与可内录的输出设备（loopback），供设备选择参考。"""
    print("== 麦克风（输入设备）==")
    try:
        from .audio.mic import list_input_devices

        for d in list_input_devices():
            print(f"  [{d['index']:>2}] {d['name']}  ({d['channels']}ch @ {int(d['sample_rate'])}Hz)")
    except Exception as e:
        print(f"  无法枚举：{e}")

    print("\n== 系统音内录（loopback 输出设备，仅 Windows）==")
    try:
        from .audio.loopback import list_loopback_devices

        devices = list_loopback_devices()
        if not devices:
            print("  未发现 loopback 设备")
        for d in devices:
            mark = "  ← 默认输出" if d["is_default"] else ""
            print(f"  [{d['index']:>2}] {d['name']}  ({d['channels']}ch @ {d['sample_rate']}Hz){mark}")
    except Exception as e:
        print(f"  无法枚举：{e}")

    print(
        "\n选择方式：把设备名的一部分填入 config/config.yaml 的 audio.loopback_device，"
        "或运行时加参数 --loopback-device \"名称关键词\""
    )
    return 0


def _apply_profile_llm_override(cfg: GlobalConfig, profile) -> None:
    """旧版档案级 llm 覆盖（FR-1.7）：作用于简易模式合成的 default provider。"""
    if not profile.llm_override:
        return
    if len(cfg.providers) == 1 and cfg.providers[0].id == "default":
        merged = cfg.llm.merged(profile.llm_override)
        p = cfg.providers[0]
        p.base_url, p.api_key, p.model = merged.base_url, merged.api_key, merged.model
        p.timeout_sec, p.max_retries = merged.timeout_sec, merged.max_retries


def list_providers(cfg: GlobalConfig) -> int:
    """列出全部 Provider 及槽位绑定状态（离线，不做网络健康检查）。"""
    from .nlp.providers import filter_for_slot
    from .nlp.slots import resolve_slots
    from .nlp.providers.registry import ProviderRegistry

    registry = ProviderRegistry.from_config(cfg)
    res = resolve_slots(cfg.slots, {}, registry)
    if not cfg.providers:
        print("未配置任何 Provider（见 config/config.yaml 的 providers 节，或运行向导）")
        return 0
    bound: dict[str, list[str]] = {}
    for slot, label in (("translation", "翻译槽"), ("grammar", "语法槽")):
        pid = res.bindings.get(slot)
        if pid:
            bound.setdefault(pid, []).append(label)
    print("== 已注册的 Provider ==")
    for p in cfg.providers:
        marks = bound.get(p.id, [])
        local = ""
        if p.local is not None:
            from pathlib import Path as _P

            gguf = _P(p.local.gguf_path)
            gguf = gguf if gguf.is_absolute() else cfg.root / gguf
            local = f"，GGUF {'✓ 已下载' if gguf.is_file() else '✗ 缺失（--download-model ' + p.id + '）'}"
        caps = ",".join(p.capabilities) or "(无)"
        mark = f"  ← 绑定：{'、'.join(marks)}" if marks else ""
        print(f"  [{p.id}] {p.name}（{p.type}，能力：{caps}{local}）{mark}")
    print(f"\n可绑翻译槽：{[p.id for p in filter_for_slot(cfg.providers, 'translation')]}")
    print(f"可绑语法槽：{[p.id for p in filter_for_slot(cfg.providers, 'grammar')]}")
    return 0


def _build_vad(cfg: GlobalConfig):
    """按 vad 配置创建 VAD 后端；加载失败报错并回退 Silero（不阻断主流程）。"""
    from .asr.engine import setup_cuda_dll_dirs
    from .asr.vad import create_vad_fn

    # onnxruntime-gpu 的 CUDA EP 依赖 pip 版 nvidia DLL（cudart/cudnn/cublas），
    # 需在创建推理 session 前注册 DLL 目录（幂等；非 Windows 为 no-op）
    setup_cuda_dll_dirs()
    try:
        vad = create_vad_fn(cfg.vad, cfg.segmentation)
    except Exception as e:  # noqa: BLE001
        print(f"⚠ VAD 后端 {cfg.vad.backend} 加载失败：{e}\n  已回退内置 Silero VAD。")
        from .asr.vad import SileroVad

        vad = SileroVad(min_silence_ms=cfg.segmentation.silence_ms, pad_ms=200)
        cfg.vad.backend = "silero"
    active_provider = getattr(vad, "active_provider", "")
    device_note = f"（{active_provider}）" if active_provider else ""
    print(f"VAD：{cfg.vad.backend}{device_note}")
    return vad


def rollover_profile(
    runtime: dict,
    new_profile_name: str,
    *,
    cfg: GlobalConfig,
    slots,
    streamer,
    debug_hook=None,
    output_root: Path | None = None,
    session_name: str = "",
    auto_lang: bool = False,
    rollover_lock: threading.Lock | None = None,
) -> tuple[object | None, str | None]:
    """语言档案运行中切换的会话滚动（交接文档 §4.2）。

    流程：暂停接收新 ASR 句 → 冲刷旧 Blocker → 排空旧语法任务 → 保存旧会话
    → 重新解析档案级槽位并换绑 → 重建 Translator/RollingMemory/GrammarAnalyzer/
    SessionRecorder/Blocker → 更新 ASR 语种提示 → 恢复运行。
    旧会话完整保留，新会话使用新档案，绝不混写两种源语言的分析文件。

    runtime 是可变字典，键：profile/translator/grammar/recorder/blocker/on_block；
    回调闭包必须读 runtime 而不是捕获旧实例。返回 (新档案, None) 或 (None, 错误信息)。
    """
    from .nlp.blocker import Blocker
    from .nlp.grammar import GrammarAnalyzer
    from .nlp.slots import resolve_slots
    from .subtitle.session import SessionRecorder

    lock = rollover_lock or threading.Lock()
    with lock:
        old_profile = runtime["profile"]
        if new_profile_name == old_profile.name:
            return old_profile, None
        try:
            new_profile = get_profile(cfg.languages_dir, new_profile_name)
        except KeyError as e:
            return None, str(e)

        streamer.set_paused(True)
        try:
            blocker = runtime.get("blocker")
            if blocker is not None:
                blocker.flush()  # 冲刷残余句（用旧 Translator 完成在途翻译）
            grammar = runtime.get("grammar")
            if grammar is not None:
                grammar.drain()
            translator = runtime.get("translator")
            out_dir = runtime["recorder"].save(
                output_root or cfg.root / "output",
                session_name or old_profile.name,
                memory=translator.memory
                if translator is not None and translator.cfg.memory.enabled
                else None,
            )
            if out_dir:
                print(f"旧会话已保存：{out_dir}")

            if slots is not None:
                slots.rebind(resolve_slots(cfg.slots, new_profile.slots, slots.registry))

            tpl_version = 0
            if translator is not None:
                from .nlp.translate import Translator

                new_translator = Translator(
                    slots.translation_provider,
                    PromptEngine(cfg.prompts_dir),
                    new_profile,
                    cfg.context,
                    debug=debug_hook,
                    lease=slots.translation_lease,
                )
                try:
                    tpl_path = new_translator.prompts.resolve(
                        new_profile.prompt_path("translate"), "translate"
                    )
                    tpl_version = new_translator.prompts.template_version(tpl_path)
                except Exception:  # noqa: BLE001
                    pass
                runtime["translator"] = new_translator

            recorder = SessionRecorder(new_profile, prompt_template_version=tpl_version)
            runtime["recorder"] = recorder
            runtime["grammar"] = (
                GrammarAnalyzer(
                    slots.grammar_provider,
                    PromptEngine(cfg.prompts_dir),
                    new_profile,
                    recorder.document,
                    on_update=runtime.get("on_grammar_update"),
                    lease=slots.grammar_lease,
                )
                if slots is not None
                else None
            )
            if runtime.get("blocker") is not None:
                merged_ctx = cfg.context.merged(new_profile.context)
                runtime["blocker"] = Blocker(merged_ctx.block, runtime["on_block"])
            runtime["profile"] = new_profile
            streamer.language_hint = None if auto_lang else new_profile.language_hint
            print(f"已切换语言档案：{new_profile.display_name}（{old_profile.display_name} 会话已保存）")
            return new_profile, None
        finally:
            streamer.set_paused(False)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.root) if args.root else None)
    profile = get_profile(cfg.languages_dir, args.profile or cfg.default_profile)
    print(f"语言档案：{profile.display_name}（{profile.source_lang} → {profile.target_lang}）")

    # ---- 模型槽位（Provider Registry + 双槽位，模型可插拔架构）----
    registry = None
    slots = None
    translator = None
    debug_hook: Callable[[str], None] | None = None
    if not args.no_translate or args.grammar:
        from .nlp.providers import SLOT_GRAMMAR, SLOT_TRANSLATION, ProviderRegistry
        from .nlp.slots import ModelSlots, resolve_slots

        _apply_profile_llm_override(cfg, profile)
        registry = ProviderRegistry.from_config(cfg)
        resolution = resolve_slots(cfg.slots, profile.slots, registry)

        # 首次启动向导：无任何可用 Provider 时引导注册并绑定翻译槽（FR-1.5）
        if not args.no_translate and resolution.provider_id(SLOT_TRANSLATION) is None and not any(
            p.is_complete for p in cfg.providers
        ):
            from .ui.setup_wizard import run_setup_wizard

            provider, new_slots = run_setup_wizard(cfg)
            cfg.providers.append(provider)
            cfg.slots = new_slots
            registry = ProviderRegistry.from_config(cfg)
            resolution = resolve_slots(cfg.slots, profile.slots, registry)

        slots = ModelSlots(registry, resolution)
        for line in slots.status_lines():
            print(line)

        # 启动健康检查（§4）：本地托管 Provider 在此拉起子进程；云端做轻量 ping
        for slot in (SLOT_TRANSLATION, SLOT_GRAMMAR):
            if args.no_translate and slot == SLOT_TRANSLATION:
                continue
            pid = resolution.provider_id(slot)
            if not pid:
                continue
            err = registry.check_health(pid)
            if err:
                hint = "将仅显示原文，恢复后自动续翻" if slot == SLOT_TRANSLATION else "语法分析功能停用"
                print(f"⚠ {err}\n  {hint}。")

        debug_hook: Callable[[str], None] | None = None
        if args.debug:
            _print_lock = threading.Lock()

            def _print_debug(text: str) -> None:
                with _print_lock:
                    print(text, flush=True)

            debug_hook = _print_debug

        if not args.no_translate:
            translator = Translator(
                slots.translation_provider,
                PromptEngine(cfg.prompts_dir),
                profile,
                cfg.context,
                debug=debug_hook,
                lease=slots.translation_lease,
            )
            if args.debug:
                print("⚙ debug 模式：每次 LLM 请求/响应将完整输出（可重定向 > debug.log 2>&1 离线分析）")

    # ---- ASR 引擎（GPU/CPU 自适应，§4.2）----
    # 非 TTY 环境下 tqdm 下载进度条会被静默，至少保证 huggingface_hub 日志可见
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    print("正在加载 ASR 模型…")
    from .asr.engine import ModelLoadError

    try:
        engine = ASREngine(cfg.asr)
    except ModelLoadError as e:
        print(f"\n✗ ASR 模型加载失败：\n{e}")
        return 1
    info = engine.info
    print(f"ASR：{info.model_size} @ {info.device}（{info.compute_type}）")
    if info.downgraded:
        print(f"⚠ 已自动降级：{info.reason}")

    # ---- 音频源与环形缓冲 ----
    source = build_source(cfg, args.source, args.input, args.loopback_device or "")
    ring = RingBuffer(capacity_samples=cfg.audio.sample_rate * RING_SECONDS)

    # ---- 会话记录器（M2：结束一键落盘）----
    from .subtitle.session import SessionRecorder

    tpl_version = 0
    if translator is not None:
        try:
            tpl_path = translator.prompts.resolve(
                profile.prompt_path("translate"), "translate"
            )
            tpl_version = translator.prompts.template_version(tpl_path)
        except Exception:
            pass
    recorder = SessionRecorder(profile, prompt_template_version=tpl_version)

    # 运行时指标：采集线程只写快照，UI 每秒读取（见 telemetry 模块）
    from .telemetry import RuntimeMetrics

    metrics = RuntimeMetrics()

    # 可滚动运行态：档案切换（rollover_profile）整体替换这里的实例，
    # 所有回调闭包必须读 runtime 而不是捕获旧实例。
    runtime: dict = {
        "profile": profile,
        "translator": translator,
        "grammar": None,
        "recorder": recorder,
        "blocker": None,
        "on_block": None,
        "on_grammar_update": None,
    }
    rollover_lock = threading.Lock()

    # ---- 悬浮窗（FR-3.1/3.2 + M3 完整 A–D 栏）----
    overlay = None
    sampler = None
    use_overlay = cfg.ui.overlay and not args.no_overlay
    if use_overlay:
        from .telemetry import SystemSampler
        from .ui.i18n import set_language
        from .ui.overlay import OverlayUnavailableError, create_overlay
        from .ui.settings import SettingsController

        set_language(cfg.ui.ui_language, cfg.root / "config" / "locales")
        merged_ui = cfg.ui.merged(profile.appearance)  # 档案外观覆盖 > 全局
        controller = (
            SettingsController(
                cfg,
                registry,
                slots,
                profile,
                memory_provider=lambda: (
                    runtime["translator"].memory
                    if runtime["translator"] is not None
                    else None
                ),
            )
            if slots is not None and registry is not None
            else None
        )
        sampler = SystemSampler()
        try:
            overlay = create_overlay(
                display_order=profile.subtitle.get("display_order", "source_first"),
                opacity=cfg.ui.opacity,
                profile=profile,
                ui=merged_ui,
                controller=controller,
                metrics=metrics,
                sampler=sampler,
                confirm_exit=True,
                profiles=controller.available_profiles() if controller is not None else None,
            )
        except OverlayUnavailableError as e:
            print(f"⚠ 悬浮窗不可用（{e}），退回纯控制台模式")

    # ---- M3 语法分析：始终异步，自动分析需 --grammar 显式开启 ----
    # 分析器在 slots 可用时始终创建（手动右键分析入口始终在线）；
    # 自动提交前检查当前绑定，未绑定时不制造必然失败的任务。
    grammar = None
    on_grammar_update: Callable[[object], None] | None = None
    if slots is not None:
        from .nlp.grammar import GrammarAnalyzer

        def _emit_grammar_update(segment):
            if overlay is not None:
                overlay.analysis_ready.emit(segment)

        on_grammar_update = _emit_grammar_update

        grammar = GrammarAnalyzer(
            slots.grammar_provider,
            PromptEngine(cfg.prompts_dir),
            profile,
            recorder.document,
            on_update=on_grammar_update,
            lease=slots.grammar_lease,
        )
        if overlay is not None:
            overlay.set_grammar_enabled(grammar.available)
            overlay.analysis_requested.connect(grammar.submit)
        if args.grammar and not grammar.available:
            print("⚠ 语法槽未绑定，自动语法分析停用；识别与翻译不受影响。")
    elif args.grammar:
        print("⚠ 语法槽未绑定，自动语法分析停用；识别与翻译不受影响。")

    # 填充可滚动运行态（runtime 已在悬浮窗之前创建）
    runtime["grammar"] = grammar
    runtime["on_grammar_update"] = on_grammar_update

    state = {"online": True}
    state_lock = threading.Lock()

    from .subtitle.document import TRANSLATION_FAILED, TRANSLATION_NONE, TRANSLATION_READY

    def print_sentence(s: Sentence, translation: str | None) -> None:
        line = f"[{fmt_ts(s.start)} → {fmt_ts(s.end)}] {s.text}"
        if translation:
            line += f"\n{'':>16}{translation}"
        print(line, flush=True)

    def register_sentence(s: Sentence) -> None:
        """ASR 句首次出现：立即登记稳定 ID 并显示原文/翻译占位。"""
        recorder = runtime["recorder"]
        grammar = runtime["grammar"]
        if runtime["translator"] is not None:
            segment_id = recorder.add_pending(s.start, s.end, s.text, asr_ms=s.asr_ms)
        else:
            segment_id = recorder.add(s.start, s.end, s.text, None, asr_ms=s.asr_ms)
            print_sentence(s, None)
        s.segment_id = segment_id
        if overlay is not None:
            overlay.upsert_segment.emit(recorder.get_segment(segment_id))
        if grammar is not None and args.grammar and runtime["translator"] is None and grammar.available:
            grammar.submit(segment_id)

    def emit_status(online: bool) -> None:
        with state_lock:
            if online != state["online"]:
                state["online"] = online
                tag = "ONLINE" if online else "OFFLINE（仅显示原文）"
                print(f"\n--- LLM 状态：{tag} ---", flush=True)
                if overlay is not None:
                    overlay.set_status.emit("ONLINE" if online else "OFFLINE")

    if translator is not None:
        from .nlp.blocker import Blocker

        def on_block(block: list[Sentence]) -> None:
            # 入口快照当前运行态：与档案切换并发时，旧块仍回写旧文档/旧语法器
            translator = runtime["translator"]
            recorder = runtime["recorder"]
            grammar = runtime["grammar"]
            # Blocker 携带登记时的稳定 ID，翻译完成后按 ID 回写（不靠文本/时间匹配）
            started = time.perf_counter()
            results = translator.translate_block([s.text for s in block])
            block_ms = (time.perf_counter() - started) * 1000
            emit_status(translator.online)
            bound = translator.bound
            for s, r in zip(block, results):
                ok = bool(r.translation)
                metrics.record_translation(block_ms, ok)
                if not bound:
                    status, error = TRANSLATION_NONE, ""
                elif ok:
                    status, error = TRANSLATION_READY, ""
                else:
                    status, error = TRANSLATION_FAILED, "翻译失败或服务不可用"
                segment = recorder.update_translation(
                    s.segment_id, r.translation,
                    status=status, error=error, translate_ms=block_ms,
                )
                print_sentence(s, r.translation)
                if overlay is not None:
                    overlay.upsert_segment.emit(segment)
                if grammar is not None and args.grammar and grammar.available:
                    grammar.submit(s.segment_id)

        runtime["on_block"] = on_block
        # Blocker 与 Translator 使用同一份档案合并后的 context 配置
        blocker = Blocker(cfg.context.merged(profile.context).block, on_block)
        runtime["blocker"] = blocker

        def feed_sentence(s: Sentence) -> None:
            register_sentence(s)
            runtime["blocker"].feed(s)

        on_sentence = feed_sentence
    else:
        on_sentence = register_sentence

    streamer = ASRStreamer(
        engine,
        cfg.segmentation,
        language_hint=None if args.auto_lang else profile.language_hint,
        on_sentence=on_sentence,
        vad_fn=_build_vad(cfg),
        metrics=metrics,
    )
    if overlay is not None:
        overlay.pause_toggled.connect(streamer.set_paused)
        overlay.set_asr_info(f"ASR {info.model_size} @ {info.device}（{info.compute_type}）")

        def _do_profile_switch(name: str) -> None:
            """worker 线程执行会话滚动；GUI 更新经 post_to_gui 回主线程。"""
            new_profile, err = rollover_profile(
                runtime,
                name,
                cfg=cfg,
                slots=slots,
                streamer=streamer,
                debug_hook=debug_hook if args.debug else None,
                output_root=Path(args.output) if args.output else cfg.root / "output",
                session_name=args.session_name or "",
                auto_lang=args.auto_lang,
                rollover_lock=rollover_lock,
            )
            if err is not None:
                log.warning("档案切换失败：%s", err)
                overlay.post_to_gui(lambda: overlay.notify_profile_error(name, err))
                return
            assert new_profile is not None
            merged = cfg.ui.merged(new_profile.appearance)

            def apply():
                overlay.apply_profile(
                    new_profile,
                    merged_ui=merged,
                    notice=f"已切换语言档案：{new_profile.display_name}",
                )
                grammar = runtime["grammar"]
                overlay.set_grammar_enabled(bool(grammar and grammar.available))

            overlay.post_to_gui(apply)

        overlay.profile_change_requested.connect(
            lambda name: threading.Thread(
                target=_do_profile_switch, args=(name,),
                name="profile-switch", daemon=True,
            ).start()
        )

    def metered_write(chunk) -> None:
        """采集回调只写 RMS 快照（telemetry），不触碰 UI。"""
        ring.write(chunk)
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.size:
            metrics.record_audio_block(float(np.sqrt(np.mean(arr * arr))))

    print(f"音频来源：{args.source}。开始识别，{'关闭悬浮窗' if overlay else 'Ctrl+C'} 结束。\n")
    started = time.monotonic()
    try:
        source.start(metered_write)
        streamer.start(ring)
        if overlay is not None:
            _run_overlay_loop(overlay, source, args.source)
        elif args.source == "file":
            getattr(source, "finished").wait()
            time.sleep(0.5)  # 尾部静音推入缓冲
        else:
            while True:
                # 采集流被系统中止/回调线程崩溃时及时退出，而不是空转
                is_active = getattr(source, "is_active", None)
                if callable(is_active) and not is_active():
                    print("\n⚠ 音频采集流已停止（设备被拔出或回调异常），正在结束会话…")
                    break
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("运行中断")
    finally:
        source.stop()
        streamer.stop()
        try:
            blocker = runtime["blocker"]
            grammar = runtime["grammar"]
            translator = runtime["translator"]
            recorder = runtime["recorder"]
            if blocker is not None:
                blocker.flush()  # 先冲刷残余句，再停止接收语法任务
            if grammar is not None:
                if not grammar.drain():
                    print("⚠ 部分排队语法分析已取消；运行中的请求已安全结束。")
            elapsed = time.monotonic() - started
            print(f"\n会话结束，时长 {fmt_ts(elapsed)}")
            if translator is not None:
                print(f"本次会话 token 用量估算：{translator.tokens_used}")
                if translator.cfg.memory.enabled:
                    n = len(translator.memory.glossary)
                    print(f"滚动记忆：术语 {n} 条"
                          + (f"，风格：{translator.memory.style_notes}" if translator.memory.style_notes else ""))
            out_dir = recorder.save(
                Path(args.output) if args.output else cfg.root / "output",
                args.session_name or profile.name,
                memory=translator.memory if translator and translator.cfg.memory.enabled else None,
            )
            if out_dir:
                print(f"会话文件已保存：{out_dir}")
        finally:
            if sampler is not None:
                sampler.close()
            if registry is not None:
                registry.close_all()  # 所有请求和落盘结束后回收本地 Provider
    return 0


def _run_overlay_loop(overlay, source, source_kind: str) -> None:
    """Qt 事件循环（GUI 必须跑在主线程）；定时器驱动结束条件与 SIGINT 处理。"""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    overlay.show()
    grace: dict[str, float | None] = {"deadline": None}

    def tick():
        if source_kind == "file":
            # 文件源播完后给翻译/冲刷一段宽限时间再关窗
            finished = getattr(source, "finished", None)
            if finished is not None and finished.is_set():
                if grace["deadline"] is None:
                    grace["deadline"] = time.monotonic() + 3.0
                elif time.monotonic() >= grace["deadline"]:
                    overlay.close()
        else:
            is_active = getattr(source, "is_active", None)
            if callable(is_active) and not is_active():
                overlay.close()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(500)
    app.exec()


def run_replay_session(args: argparse.Namespace) -> int:
    """重载已保存会话（或外部 SRT），不初始化 ASR/Provider；--media 启用媒体回放。"""
    from .subtitle.document import load_session_document
    from .ui.replay import document_from_srt, run_session_preview

    cfg = load_config(Path(args.root) if args.root else None)
    if args.replay_session:
        document = load_session_document(args.replay_session)
        profile_name = args.profile or document.language_profile or cfg.default_profile
        profile = get_profile(cfg.languages_dir, profile_name)
        if document.language_profile and document.language_profile != profile.name:
            raise SystemExit(
                f"会话使用语言档案 {document.language_profile}，不能按 {profile.name} 重载"
            )
        if document.source_lang and document.source_lang != profile.source_lang:
            raise SystemExit(
                f"会话源语言 {document.source_lang} 与档案 {profile.name} 的 {profile.source_lang} 不一致"
            )
    else:
        # 外部 SRT 独立回放：完整块读取，保留多行原文
        document = document_from_srt(args.srt)
        profile = get_profile(cfg.languages_dir, args.profile or cfg.default_profile)
    return run_session_preview(
        document, profile, cfg.ui.opacity, media_path=args.media, ui=cfg.ui
    )


def run_batch(args: argparse.Namespace) -> int:
    """真正的离线批处理（阶段 5）：独立于 FileSource/RingBuffer 实时模拟。"""
    from .offline.runner import OfflineRunner

    cfg = load_config(Path(args.root) if args.root else None)
    profile = get_profile(cfg.languages_dir, args.profile or cfg.default_profile)
    print(f"语言档案：{profile.display_name}（{profile.source_lang} → {profile.target_lang}）")

    from .nlp.providers import SLOT_GRAMMAR, SLOT_TRANSLATION, ProviderRegistry
    from .nlp.slots import ModelSlots, resolve_slots

    _apply_profile_llm_override(cfg, profile)
    registry = ProviderRegistry.from_config(cfg)
    resolution = resolve_slots(cfg.slots, profile.slots, registry)
    slots = ModelSlots(registry, resolution)
    for line in slots.status_lines():
        print(line)
    for slot in (SLOT_TRANSLATION, SLOT_GRAMMAR):
        pid = resolution.provider_id(slot)
        if not pid:
            continue
        if slot == SLOT_GRAMMAR and not args.grammar:
            continue
        err = registry.check_health(pid)
        if err:
            print(f"⚠ {err}")

    print("正在加载 ASR 模型…")
    from .asr.engine import ModelLoadError

    try:
        engine = ASREngine(cfg.asr)
    except ModelLoadError as e:
        print(f"\n✗ ASR 模型加载失败：\n{e}")
        return 1
    info = engine.info
    print(f"ASR：{info.model_size} @ {info.device}（{info.compute_type}）")

    runner = OfflineRunner(
        cfg,
        profile,
        output_root=Path(args.output) if args.output else cfg.root / "output",
        engine=engine,
        vad_fn=_build_vad(cfg),
        slots=slots,
        grammar_enabled=args.grammar,
        session_name=args.session_name or "",
    )
    try:
        return runner.run(args.batch)
    finally:
        registry.close_all()  # 全部文件完成（翻译/语法排空）后回收本地 Provider


def main() -> int:
    parser = argparse.ArgumentParser(description="实时语音识别翻译工具（M3 学习闭环）")
    parser.add_argument("--source", choices=["mic", "loopback", "file"], default=None,
                        help="音频来源（默认取 config.yaml 的 audio.source）")
    parser.add_argument("--input", help="--source file 时的音频文件路径")
    parser.add_argument("--loopback-device", default="",
                        help="loopback 输出设备名关键词（覆盖 config 的 audio.loopback_device）")
    parser.add_argument("--list-devices", action="store_true",
                        help="列出麦克风与 loopback 设备后退出")
    parser.add_argument("--profile", help="语言档案名（config/languages/<name>.yaml）")
    parser.add_argument("--root", help="项目根目录（默认自动查找）")
    parser.add_argument("--no-translate", action="store_true", help="仅识别不翻译")
    parser.add_argument("--grammar", action="store_true",
                        help="自动后台语法分析；未开启时仍可在悬浮窗右键逐句分析")
    parser.add_argument("--replay-session", metavar="SESSION_DIR",
                        help="重载会话目录中的 analysis JSON/SRT，显示句块和语法卡")
    parser.add_argument("--media", metavar="MEDIA_PATH",
                        help="回放模式加载的媒体文件（mp4/mkv/mp3/wav/m4a 等）")
    parser.add_argument("--srt", metavar="SRT_PATH",
                        help="外部 SRT 字幕回放（完整块读取，保留多行；可配合 --media）")
    parser.add_argument("--batch", nargs="+", metavar="MEDIA_FILE",
                        help="离线批处理：一个或多个媒体文件（PyAV 解码，非实时模拟）")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：完整输出每次 LLM 请求/响应（prompt 全文、原始响应、token 用量）")
    parser.add_argument("--auto-lang", action="store_true", help="ASR 自动检测语种（默认用档案指定）")
    parser.add_argument("--no-overlay", action="store_true", help="禁用悬浮字幕窗（纯控制台）")
    parser.add_argument("--session-name", help="会话名（输出目录 output/<日期>-<会话名>）")
    parser.add_argument("--output", help="输出根目录（默认 <项目根>/output）")
    parser.add_argument("--list-providers", action="store_true",
                        help="列出全部 Provider 与槽位绑定状态后退出")
    parser.add_argument("--download-model", metavar="PROVIDER_ID",
                        help="按 Provider 的 local.download 配置下载 GGUF 模型后退出")
    parser.add_argument("--download-source", choices=["auto", "hf", "modelscope"], default="auto",
                        help="--download-model 的下载源（默认 auto：hf 失败自动试 modelscope）")
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.replay_session or args.srt:
        return run_replay_session(args)
    if args.batch:
        return run_batch(args)
    if args.list_devices:
        return list_devices()
    if args.list_providers:
        return list_providers(load_config(Path(args.root) if args.root else None))
    if args.download_model:
        from .nlp.providers.download import DownloadError, download_model

        cfg = load_config(Path(args.root) if args.root else None)
        provider = next((p for p in cfg.providers if p.id == args.download_model), None)
        if provider is None:
            ids = ", ".join(p.id for p in cfg.providers) or "(无)"
            print(f"Provider '{args.download_model}' 不存在，已注册：{ids}")
            return 1
        try:
            download_model(provider, cfg.root, source=args.download_source)
        except DownloadError as e:
            print(f"✗ {e}")
            return 1
        return 0
    if args.source is None:
        cfg = load_config(Path(args.root) if args.root else None)
        args.source = cfg.audio.source
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
