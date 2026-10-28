# ASR2class — 实时语音识别翻译与语言学习工具

实时语音识别（faster-whisper）+ 可插拔云端/本地模型翻译的桌面学习工具。
当前进度：**M3 第一轮（语法学习闭环）已完成**；完整音视频回放、设置页与完整外观系统属于后续 M3。需求文档见 `doc/实时语音翻译学习工具-需求文档.md`。

- 🎙 麦克风 + 系统音内录（WASAPI loopback，Windows）双来源
- ⚡ 流式识别：VAD 断句（可插拔：内置 Silero 或耳语专用 Whisper-Vad-ASMR ONNX），GPU（int8_float16）/ CPU（int8）自适应，显存超预算自动降级；幻觉句门控（质量阈值 + 短语黑名单）
- 🧩 **模型可插拔**：Provider Registry + 翻译/语法双槽位，云端 OpenAI 兼容服务与本地 llama.cpp GGUF 模型混绑，语言档案可覆盖绑定（见下文「模型提供方与槽位」）
- 🌐 翻译完全配置化：`base_url / api_key / model` 三项接入任意 OpenAI 兼容服务，**无默认服务商**
- 🧠 **上下文感知翻译**：块级整体翻译（语义分块）+ 滑动历史窗口 + 滚动记忆（术语表/风格摘要），解决代词断裂、术语不一致、语域漂移（见 `doc/20261014/上下文感知翻译-专项需求文档.md`）
- 🧑‍🏫 **M3 语法学习闭环**：可选异步语法分析、按 ID 原位更新句块、语言档案驱动的语法卡、分析随会话落盘并支持重载预览
- 🪟 置顶半透明悬浮窗：可拖拽、透明度可调（控制条按钮）、点击穿透（托盘菜单恢复）、逐句滚动双语字幕
- 💾 会话结束一键落盘：`output/<日期>-<会话名>/` 下产出 transcript.txt、original.srt、bilingual.srt、analysis.<源语言>.json、memory.json
- 🗂 语言档案（`config/languages/*.yaml`）+ 外部 prompt 模板（`config/prompts/`），新增语言零代码改动
- 🔌 断网自动降级为仅显示原文，恢复后自动续翻；控制台显示 ONLINE/OFFLINE 与 token 用量

## 快速开始（Windows 11 + RTX 3060）

```powershell
# 1. 安装 uv（https://docs.astral.sh/uv/），然后：
git clone <repo> && cd ASMR2class
uv sync                      # 自动安装 Python 3.12 + 全部依赖（含 pyaudiowpatch）

# 2. 首次运行会进入设置向导，填写你的 LLM base_url / api_key / model
uv run python -m src.main --source loopback --profile ja
```

首次使用某个模型档位需下载（small 约 466MB，medium 约 1.5GB）。直连 HuggingFace 失败且未设镜像时，程序会**自动切换到 hf-mirror.com 重试**；也可手动预设（更快）：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
```

## 使用

```
uv run python -m src.main                        # 启动入口窗口（选择模式/档案/参数）
uv run python -m src.main --source loopback      # 实时：系统音内录（仅 Windows），跳过启动器
uv run python -m src.main --list-devices         # 列出麦克风与可内录的输出设备
uv run python -m src.main --source loopback --loopback-device "扬声器"  # 指定输出设备
uv run python -m src.main --source file --input a.wav --profile en   # 文件模拟（档案默认语言对）
uv run python -m src.main --source-lang ja --target-lang en          # 自由语言对：日译英
uv run python -m src.main --auto-lang --target-lang zh               # 源语言自动识别
uv run python -m src.main --no-translate         # 仅识别不翻译（离线调试）
uv run python -m src.main --grammar              # 自动后台语法分析（默认不开启，避免额外 API 费用）
uv run python -m src.main --replay-session output/<日期>-<会话名>            # 重载句块与语法卡
uv run python -m src.main --replay-session output/<日期>-<会话名> --media a.mp4  # 会话 + 音视频同步回放
uv run python -m src.main --srt a.srt --media a.mp4                        # 外部 SRT（保留多行）+ 媒体回放
uv run python -m src.main --batch a.mp4 b.mkv                              # 真正的离线批处理（PyAV 解码，非实时模拟）
uv run python -m src.main --debug                # 完整输出每次 LLM 请求/响应（翻译块与语法分析的 prompt 全文、原始响应、token 用量、降级链）
uv run python -m src.main --auto-lang            # ASR 自动检测语种
uv run python -m src.main --no-overlay           # 禁用悬浮窗（纯控制台）
uv run python -m src.main --list-providers       # 列出全部 Provider 与槽位绑定状态
uv run python -m src.main --download-model <provider_id>  # 下载本地 GGUF 模型（HF/ModelScope 双源自动回退）
uv run python -m src.main --session-name 动漫01  # 自定义会话名（输出目录 output/<日期>-动漫01）
uv run python -m src.main -v                     # 调试日志（可见实际发送的 prompt，密钥除外）
```

不带模式参数启动时弹出**启动入口窗口**：实时（mic/loopback，进悬浮窗）/ 离线批处理（进独立进度窗口，逐文件状态 + 日志 + 取消）/ 回放（会话目录或外部 SRT + 媒体）。带任何模式参数（`--source`/`--batch`/`--replay-session`/`--srt`/`--profile` 等）则跳过启动器，保持脚本化直启。

loopback 设备选择：`--loopback-device` 用设备名的任意一段做（大小写不敏感的）包含匹配；也可写入 `config.yaml` 的 `audio.loopback_device` 作为默认。留空则跟随系统默认输出设备。

运行中按 `Ctrl+C` 结束会话。

## 配置

| 文件                              | 作用                                                                                                                                   |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `config/config.example.yaml`      | 入库的配置模板（不含任何密钥）                                                                                                         |
| `config/config.yaml`              | 本地配置，**已 gitignore**：首次运行由 example 自动复制创建，向导/手动编辑都写入它                                                     |
| `config/languages/<lang>.yaml`    | 语言包：语种提示、默认目标语言、语法 prompt/语法卡、档案级槽位覆盖                                                                              |
| `config/prompts/<src>-<tgt>/`     | 特定语言对的定制翻译模板（可选；解析顺序：pair 目录 → 包内路径 → `_default/`）                                                                |
| `config/prompts/**/translate.txt` | 翻译模板，`#` 开头为注释，支持 `{source_lang}/{target_lang}/{sentences}/{context_history}/{glossary}/{style_notes}` 变量，改后即时生效 |
| `config/prompts/**/grammar.txt`   | M3 语法分析模板，支持 `{source_lang}/{target_lang}/{sentence}/{translation}/{level_hint}`；缺失时回落 `_default/grammar.txt` |

### 上下文感知翻译（config.yaml `context:` 节）

| 配置                                  | 默认 | 说明                                       |
| ------------------------------------- | ---- | ------------------------------------------ |
| `context.tail_sentences`              | 3    | 滑动窗口：注入前块尾部历史句数，`0` = 关闭 |
| `context.history_max_tokens`          | 512  | 历史注入的 token 上限（粗估），超出裁最早  |
| `context.block.max_sentences`         | 8    | 块句数上限；`1` = 退化为逐句翻译           |
| `context.block.silence_gap_sec`       | 1.5  | 静音间隙超过该值即成块（语义边界）         |
| `context.memory.enabled`              | true | 滚动记忆（术语表 + 风格摘要）总开关        |
| `context.memory.glossary_max_entries` | 100  | 术语表上限，LRU 淘汰                       |

### VAD 后端（config.yaml `vad:` 节）

| 配置             | 默认   | 说明                                                                                     |
| ---------------- | ------ | ---------------------------------------------------------------------------------------- |
| `vad.backend`    | silero | `silero`（内置，零下载）或 `whisper_asmr`（耳语/软语专用 ONNX，20ms 帧高精度）           |
| `vad.threshold`  | 0.5    | whisper_asmr 语音帧判定阈值（调低更灵敏，负阈值自动 -0.15 滞后防抖）                     |
| `vad.model_path` | -      | whisper_asmr 的 ONNX 路径；缺失时按 `vad.download` 配置的 HF/ModelScope 仓库自动下载     |
| `vad.device`     | auto   | whisper_asmr 推理设备：`auto`（有 CUDA 则用 GPU）/ `cuda` / `cpu`                        |
| `vad.recompute_ms` | 1000 | whisper_asmr 增量推理的重算节流间隔：新增音频不足该值时复用缓存帧概率（切句确认不受限） |

ASMR/耳语内容建议 `whisper_asmr`：判别式模型对耳语段检出率显著高于 Silero，且不受语音内容影响（纯声学帧分类）。切换后端只需改 `vad.backend`，重启生效。whisper_asmr 采用增量推理：完整的 30s 块概率定稿缓存、尾部块按 `recompute_ms` 节流重算、切句确认点立即重算，分段结果与全量重算完全一致，流式 CPU 开销约降为原来的 1/5。

**GPU 推理（仅 whisper_asmr）**：Windows 上依赖已锁定为 `onnxruntime-gpu==1.26.*`（CUDA 12 + cuDNN 9 构建，与项目内 ctranslate2 的 cu12 栈一致），`uv sync` 后自动生效，无需手动换装。实现方式：pyproject 用 `override-dependencies` 把 faster-whisper 传递依赖的 CPU 版 `onnxruntime` 限定为非 Windows 平台（两个发行版安装同名模块文件，不能共存；GPU 版是 CPU 版超集），Windows 改由 `onnxruntime-gpu` + `nvidia-*-cu12` 组件库（cuDNN/cuFFT/cuRAND 等，启动时自动注册 DLL 目录）提供。版本钉 1.26.x 的原因：1.27 起 PyPI 默认包改为 CUDA 13 构建，与 cu12 栈不兼容。

启动时显示 `VAD：whisper_asmr（CUDAExecutionProvider）` 即 GPU 生效；CUDA 初始化失败时自动回退 CPU 并打印针对性诊断。若历史上手动换装过导致文件残留（表现：装了 GPU 版却显示 CPU），运行 `scripts\enable-vad-gpu.ps1` 一键修复并验证。显存紧张时（本地 LLM + medium ASR 同驻）可设 `vad.device: cpu`——增量推理下 CPU 开销已很小，GPU 主要进一步降低单次前向延迟。

三个机制可任意组合；模板中删除对应变量即静默停用该机制（日志有提示）。语言档案可用同名 `context:` 节逐项覆盖，并可用 `style_preset:` 锁定风格设定（档案值优先于 LLM 推断）。翻译响应 JSON 中的 `glossary_updates`/`style_updates` 自动回写记忆，不增加额外请求。

API key 可用环境变量覆盖（不写入文件）：`ASR2CLASS_LLM_API_KEY`（同时填充所有 api_key 为空的 Provider）。

## 模型提供方与槽位（Provider / Slot）

翻译与语法分析是两个独立槽位，各自按 provider id 绑定，可绑同一个也可分开（如本地翻译模型 + 云端语法分析）。**代码中无任何具体模型/服务商硬编码**，全部 Provider 在 `config.yaml` 的 `providers:` 节声明：

```yaml
providers:
  - id: local-translator           # 本地 GGUF 翻译模型（llama.cpp server，应用托管子进程）
    type: llama_cpp_server
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "EMPTY"
    model: "local-translator"
    capabilities: [translation, glossary_constraint]  # 翻译专用 + 术语表硬约束
    params: {temperature: 0, max_tokens: 1024, enable_thinking: false}  # 按模型官方推荐
    local:
      managed: true                # 启动应用且被绑定时自动拉起，换绑/退出时回收
      gguf_path: "models/translator-Q4_K_M.gguf"
      server_args: "--port 8080 --ctx-size 4096"
      vram_budget_mb: 1500         # 空闲显存不足时警告（不静默抢占）
      download: {hf_repo: "org/repo", hf_file: "x.gguf",
                 modelscope_repo: "org/repo", modelscope_file: "x.gguf"}
slots:
  translation: local-translator
  grammar: my-cloud                # 留空 = 语法分析功能停用
```

- **能力标签过滤**：`translation` 可绑翻译槽、`grammar` 可绑语法槽、`glossary_constraint` 启用术语表硬约束（instTrans 映射：术语表→【硬性要求】、风格→【注意】，模板 `config/prompts/_default/translate_insttrans.txt` 可改）。无 `grammar` 能力的翻译专用模型无法绑到语法槽。
- **档案覆盖**：语言档案内 `slots.translation` / `slots.grammar` 优先于全局，切换档案即切换模型；档案覆盖槽位时设置页/B 栏会显示锁定标识，不会假报全局换绑已生效。
- **安全热换绑**：设置页或悬浮窗 B 栏改绑即生效（不重启）。请求经租约（`slots.acquire`）使用 Provider：换绑后旧实例等在途请求结束才回收；同 id 编辑、A→B→A 都产生新实例代际；健康检查或配置保存失败时保留旧运行态。本地 llama-server 的启动/关闭等待都在 worker 线程，不卡 GUI。
- **向后兼容**：不填 `providers:` 时，旧 `llm:` 节自动合成为 `default` Provider 并绑到两个槽位，现有配置零改动。
- 本地 GGUF 下载：`--download-model <provider_id>`（HF 直连失败自动切镜像，再失败自动试 ModelScope；需先在 `local.download` 配好仓库信息）。llama-server 二进制需自备（装到 PATH 或 `local.server_bin` 填绝对路径）。

## M3 第一轮：语法学习闭环

- 语法槽绑定具有 `grammar` 能力的 Provider 后，`--grammar` 会对新句自动排队分析；不加该参数时不会自动产生语法 API 请求，但可在句块右键选择“为此句生成语法分析”。
- 分析在后台执行，按稳定句段 ID 回写，乱序完成不会改变字幕顺序。未绑定、超时或 JSON 契约失败只影响对应句子的语法状态，不阻断识别和翻译。
- 有分析数据的句块可点击弹出语法卡（实时模式）。卡片区块、字段、标题和徽章颜色由 `config/languages/<lang>.yaml` 的 `grammar_card.sections` 声明；日语/英语字段只存在 YAML 和 prompt 中。
- **分析笔记**：A 栏「笔记」按钮打开独立半透明悬浮窗（与主悬浮窗同风格），按识别顺序集中查看已分析句（时间戳+原句+译文，默认折叠）；点击单句弹出同款语法卡；可按识别顺序/分析顺序排序；「导出 Markdown…」把全部分析写成结构化 .md 文件（原子写入），档案未声明的自定义分析字段也会兜底导出。
- **气泡模式（漫画对话框）**：悬浮窗菜单开启后，窗口变为漫画对话框外形（圆角气泡 + 尾巴），实时/回放模式都可用；只显示当前句的原文一行 + 译文一行（超出宽度自动跑马灯滚动），其余元素全部隐藏。右键气泡可调整尾巴方向（八方向）、显示内容（双语/仅原文/仅译文）或退出；托盘菜单也可退出，退出后恢复正常悬浮窗。
- 会话结束时会先停止接收新任务并限期排空，再保存 `analysis.<源语言>.json`。

## M3 后续：完整悬浮窗 / 设置页 / 回放 / 离线批处理

- **A–D 悬浮窗**：A 栏（菜单：穿透/置顶/自动滚动/紧凑/字幕显隐 + 运行暂停/清空显示/设置/关闭）、B 栏（档案切换 + 槽位下拉，能力过滤，主行模型名、细节 tooltip）、C 栏（RMS/VAD、CPU/RAM/GPU/VRAM、句数与 ASR/翻译耗时，阈值着色可配置 `ui.latency_*` / `ram_warn_percent`；服务状态点 ●运行中 ●降级 ●错误 ○停止，点击查看最近错误）、D 区（翻译占位/失败状态行、上翻暂停跟随 + 回到底部、容量上限 `ui.view_capacity`、条目分隔线）。
- **外观**：白天/夜间两套预定义主题（`ui.theme`）+ 背景不透明度（RGBA 背景，文字不随窗口变淡）、行距、窗口尺寸、栏目显隐、紧凑模式、中/英界面文案（`config/locales/*.yaml`）。设置页修改即时预览、应用后原子落盘；语言档案 `appearance:` 节可覆盖。
- **档案运行中切换**：B 栏可分别切换源语言（含「自动识别」）与目标语言，即「会话滚动」——暂停接收新句 → 冲刷/排空 → 旧会话落盘 → 新语言对重建翻译/语法/记忆/文档，绝不混写两种源语言的 analysis 文件。保存的分析文件名标注语言对：`analysis.<源>.<目标>.json`（源为自动识别时是 `analysis.auto.<目标>.json`）。
- **回放**：`--replay-session <dir> --media <媒体>` 或界面「打开媒体」；媒体时钟驱动当前句高亮，单击句块 seek（双击或右键弹语法卡，两种手势分离）；`--srt` 外部字幕完整块读取保留多行。
- **离线批处理**：`--batch 文件…` 走 PyAV 流式解码（拉取式背压，不丢音频），离线块上限 `context.block.max_sentences_offline`（默认 20）实际生效；顺序翻译保持滚动记忆顺序，语法排空后才保存；逐文件失败隔离，Ctrl+C 保存已完成部分后退出。
- **记忆编辑**：设置页「记忆」页签查看/增删改术语与风格（`style_preset` 锁定时只读）；编辑实时会话同一实例，下一次翻译立即生效；`memory.json` 严格校验（空字段/重复术语/档案不匹配报错）、原子保存；「Prompt」页签可打开模板目录、恢复默认模板（自动 .bak 备份，mtime 热重载即时生效）。

## M1 验收清单（Windows 真机）

- [ ] mic 与 loopback 各跑 10 分钟日语/英语材料，识别与翻译正常
- [ ] GPU 模式首字延迟 ≤ 1.5s；`nvidia-smi` 观察 ASR 显存 ≤ 2.5GB
- [ ] 显存不足 / 拔掉网线时自动降级（CPU 模式 / 仅显示原文），恢复网络后续翻
- [ ] 首次启动设置向导可完成 LLM 配置

### loopback 排障

提示 loopback 不可用时：确认默认输出设备正常 → 开启「立体声混音」或更换输出设备 → 仍不行在 `config.yaml` 把 `audio.source` 改为 `mic` 兜底。

### GPU 报 cublas64_12.dll / cudnn 缺失

Windows 上 ctranslate2 不自带 CUDA 运行时。`uv sync` 已会自动安装 `nvidia-cublas-cu12` 与 `nvidia-cudnn-cu12`（仅 Windows），程序启动时自动把它们的 DLL 目录加入搜索路径。若仍失败：确认这两个包已安装（`uv pip list | findstr nvidia`），或安装 CUDA 12 Toolkit + cuDNN 9；程序会自动降级到 CPU 模式保证可用。

## 开发（WSL / Linux）

WSL 无 PortAudio 与 Windows 音频栈，mic/loopback 模块延迟导入、不参与测试；

```bash
uv venv --python 3.12 .venv && uv pip sync uv.lock
.venv/bin/python -m pytest tests/             # 单元测试（无需音频设备/GPU）
ASR2CLASS_E2E=1 .venv/bin/python -m pytest tests/test_pipeline_e2e.py   # 真实模型端到端
```

## 路线图

- [x] **M1** 核心链路：双来源采集 → 流式 ASR → LLM 翻译 → 控制台输出
- [x] **M2** 字幕落盘（txt/原文 srt/双语 srt/analysis.json/memory.json）+ PyQt6 置顶半透明悬浮窗（拖拽/透明度/点击穿透）
- [x] **M3 第一轮** 异步语法分析 + 稳定 ID 原位更新 + 动态语法卡 + 分析落盘/重载预览
- [x] **M3 后续** 完整 A–D 悬浮窗、设置页、安全热换绑、档案会话滚动切换、音视频同步回放、真正离线批处理、记忆编辑（Windows 真机验收逐项进行中）
- [ ] M4 生词本导出、缓存优化、PyInstaller 打包
