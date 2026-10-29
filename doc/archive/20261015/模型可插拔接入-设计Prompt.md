# 翻译/语法模型可插拔接入 —— Coding Agent 设计 Prompt

- 版本：v1.0
- 日期：2026-10-13
- 关联文档：《实时语音翻译学习工具-需求文档（PRD）v1.1》（FR-1.5、FR-1.7、§4.2、§5.4、§6）、《上下文感知翻译-专项需求文档》（FR-C3 术语表/风格摘要）
- 定位：替代并废弃原 PRD 中「云端超时 → 本地降级」的故障转移思路，改为**用户主动选择的双模型槽位架构**

---

## 0. 角色指令（Prompt 开头直接使用）

> 你是一名资深 Python 桌面应用架构师。请在《实时语音识别翻译与语言学习工具》项目中实现一套**模型提供方（Model Provider）可插拔架构**。核心要求：翻译模型与语法分析模型是两个完全独立的「槽位」，用户在应用启动时分别选择；架构必须为「运行时热切换」预留能力，任何代码不得假设模型在会话期间固定不变；首个接入的本地模型为 B 站开源的 Index-Translate-2B（GGUF + llama.cpp）。以下为完整规格，请严格实现。

## 1. 架构总则（取代故障转移思路）

**禁止设计**：云端 API 超时自动切换本地模型的故障转移链。废弃该思路。

**必须设计**：Provider Registry（提供方注册表）＋ Model Slot（模型槽位）模式：

```
┌─────────────────────────────────────────────┐
│              Provider Registry               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │云端服务A  │ │云端服务B  │ │本地llama │     │
│  │(OpenAI   │ │(OpenAI   │ │.cpp服务  │     │
│  │兼容API)  │ │兼容API)  │ │(Index-   │     │
│  │          │ │          │ │Translate-2B)   │
│  └──────────┘ └──────────┘ └──────────┘     │
│   统一接口：OpenAI 兼容 chat/completions       │
└──────┬──────────────────┬───────────────────┘
       │ 绑定              │ 绑定
┌──────▼───────┐   ┌──────▼───────┐
│ 翻译槽位      │   │ 语法分析槽位  │   ← 两个槽位完全独立
│ Translation  │   │ Grammar      │     可绑定同一 Provider，
│ Slot         │   │ Slot         │     也可分别绑定不同 Provider
└──────────────┘   └──────────────┘
```

### 1.1 关键设计约束

| 约束 | 说明 |
|---|---|
| 槽位独立 | 翻译模型 ≠ 语法分析模型，用户分别选择。允许两个槽绑同一个 Provider（如都指向同一个云端模型），也允许分离（本地 Index-Translate-2B 翻译 + 云端模型做语法分析） |
| 不写死 | 代码中不得出现任何具体模型名/服务商名的硬编码；所有 Provider 由配置文件声明，启动时加载进注册表 |
| 热切换就绪 | 槽位对 Provider 的引用必须是**可替换的运行时引用**（如 Registry 内的间接寻址），而非启动时固化的对象。本期只需在「设置页重新选择并应用」级别实现切换，但内部结构必须保证未来加一个下拉框即可实现运行中即时切换——**不得存在"绑定后缓存连接/句柄导致无法换绑"的写法** |
| 统一接口 | 所有 Provider 对外暴露 OpenAI 兼容 `chat/completions` 接口（本地模型通过 llama.cpp server / vLLM 自带端点满足）；llm_client 只认该接口，不感知本地/云端差异 |
| 能力声明 | 每个 Provider 在配置中声明能力标签：`translation`（可绑翻译槽）、`grammar`（可绑语法槽，即支持结构化 JSON 输出）、`glossary_constraint`（支持术语表硬约束，如 Index-Translate 家族）。UI 选择器按槽位过滤可选 Provider |

## 2. Provider 配置规范

`config.yaml` 新增 `providers` 节（声明式，用户可加任意多个）：

```yaml
providers:
  - id: index-translate-2b-local      # 唯一标识，槽位按 id 绑定
    name: "Index-Translate-2B（本地）"
    type: llama_cpp_server            # llama_cpp_server | openai_compatible | vllm
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "EMPTY"
    model: "index-translate-2b"       # llama.cpp server 加载的模型别名
    capabilities: [translation, glossary_constraint]
    local:
      managed: true                   # 应用负责启动/停止该本地服务
      gguf_path: "models/Index-Translate-2B-Q4_K_M.gguf"
      server_args: "--port 8080 --ctx-size 4096"
      vram_budget_mb: 1500            # 启动前检测，超预算则警告（PRD §4.2）

  - id: my-cloud-llm
    name: "云端通用模型（用户自填）"
    type: openai_compatible
    base_url: ""                      # 首次启动向导中填写
    api_key: ""
    model: ""
    capabilities: [translation, grammar]

slots:
  translation: index-translate-2b-local   # 翻译槽当前绑定
  grammar: my-cloud-llm                   # 语法槽当前绑定
```

- 语言档案（PRD §5.3）可覆盖槽位绑定：档案内 `slots.translation` / `slots.grammar` 字段优先于全局，切换语言档案时槽位绑定随之切换（日语档案绑 Sakura-1.5B、英语档案绑云端模型这类玩法必须支持）
- 首次启动设置向导：引导用户①注册至少一个 Provider ②为两个槽位各选一个绑定，未完成的槽位对应功能置灰并提示

## 3. Index-Translate-2B 接入规格（首个本地 Provider）

| 项目 | 规格 |
|---|---|
| 模型来源 | HuggingFace `IndexTeam/Index-Translate-2B` 官方 GGUF 仓库，默认拉取 **Q4_K_M** 档（约 1.5GB）；设置页提供 HuggingFace / ModelScope 双下载源（国内镜像友好） |
| 推理后端 | **llama.cpp server**（`llama-server`），应用以子进程托管：启动应用且该 Provider 被任一槽位绑定时拉起，退出或换绑时回收；也允许用户改用自行部署的 vLLM（type: vllm，应用不托管进程，只连端点） |
| 推理参数 | 沿用官方推荐：**贪心解码（temperature=0）、enable_thinking=False、max_tokens=1024**；请求 150 语言任意方向，目标语言由当前语言档案注入 |
| 术语表对接 | 该模型原生支持术语表硬约束与语气软约束（instTrans 格式）。实现时将《上下文感知翻译》FR-C3 的 `{glossary}`（实体登记表）与 `{style_notes}`（风格设定）映射为其 instTrans 规范中的 `【硬性要求】`/`【注意】` 区块；Provider 声明 `glossary_constraint` 能力时优先走该映射，普通 Provider 走通用 prompt 注入 |
| 上下文窗口对接 | FR-C1 的历史对以多行文本块形式随当前块一并送入（该模型支持多行输入） |
| 资源约束 | 启动前检测剩余显存：若低于 `vram_budget_mb` + 2GB 保留量，弹出警告并建议改用 CPU 推理（server_args 去掉 GPU 层卸载）或改绑其他 Provider，不得静默抢显存（PRD §4.2） |
| 能力边界 | 该模型为翻译专用，**禁止绑定到语法分析槽位**（capabilities 不含 grammar，UI 过滤保证用户选不了）；语法分析始终要求支持结构化输出的通用模型 |

## 4. 运行时行为

| 场景 | 行为 |
|---|---|
| 启动 | 读 `slots` 配置 → 校验 Provider 可用性（本地：GGUF 文件存在 + 端口可拉起；云端：一次轻量 ping 请求）→ 不可用的槽位功能置灰并在状态条提示（悬浮窗 UI 设计文档 §5 的降级标记复用） |
| 翻译请求 | translate.py 从「翻译槽」取当前 Provider 引用发请求；取引用与发请求之间 Provider 被换绑是合法状态，请求失败按原 PRD 降级规则处理，不得崩溃 |
| 语法分析请求 | grammar.py 从「语法槽」取引用；槽位未绑定或 Provider 无 grammar 能力时，语法分析功能整体停用（语法卡不弹、后台任务不排队），其余功能不受影响 |
| 设置页改绑 | 保存后立即生效：新请求走新 Provider；进行中的请求跑完自然结束；本地托管 Provider 被换绑出所有槽位后关闭其子进程释放显存 |
| 每档案覆盖 | 切换语言档案时按档案 `slots` 覆盖值重新解析绑定，无覆盖则沿用全局 |

## 5. 模块改动（对接 PRD §6 包结构）

```
src/nlp/
  providers/
    registry.py        # Provider 注册表：加载配置、健康检查、按 id 寻址
    base.py            # ProviderBase 抽象（统一 chat/completions 调用）
    openai_compatible.py
    llama_cpp_local.py # 子进程托管：拉起/守护/回收 llama-server，显存预检
    capabilities.py    # 能力标签定义与槽位过滤逻辑
  slots.py             # 翻译槽/语法槽：绑定解析（档案覆盖 > 全局）、热替换引用
  translate.py         # 改为从翻译槽取 Provider；Index-Translate 的 instTrans 映射在此实现
  grammar.py           # 改为从语法槽取 Provider；无绑定时整体停用
  llm_client.py        # 保留为底层 HTTP 客户端，被 ProviderBase 复用
config/
  config.yaml          # 新增 providers 与 slots 两节
src/ui/
  settings.py          # 设置页：Provider 管理（增删改/测试连接）+ 槽位绑定下拉（按能力过滤）
  setup_wizard.py      # 首次启动向导增加「注册 Provider → 绑定槽位」两步
```

## 6. 验收清单

- [ ] 启动时可分别选择翻译模型与语法分析模型，两者可以是同一 Provider 也可以是不同 Provider
- [ ] 翻译槽绑定 Index-Translate-2B（本地 llama.cpp）后，日→中字幕翻译正常出结果，端到端延迟 ≤ 1s/句
- [ ] 语法槽绑定通用云端模型后语法卡正常；语法槽留空时语法功能整体停用且不影响翻译
- [ ] 设置页改绑翻译 Provider 保存后立即生效，无需重启；进行中的会话不崩溃
- [ ] 代码全局搜索无任何具体模型名/服务商名硬编码（配置示例除外）
- [ ] 语言档案内覆盖槽位绑定后，切换档案即切换模型
- [ ] Index-Translate-2B 的术语表约束生效：登记表中的人名在后续译文中保持一致
- [ ] 显存不足时启动警告且不静默抢占；Index-Translate-2B 无法被绑到语法槽
- [ ] 新增一个 Provider 仅需在 config.yaml 加一段声明，重启（未来：不重启）即可出现在选择器中
