# ASR2class — 实时语音识别翻译与语言学习工具

看片/看直播时的实时双语字幕工具：识别语音（faster-whisper）→ 翻译成中文（云端 API 或本地模型）→ 置顶悬浮窗显示，可选语法分析讲解（语言学习向）。M1–M3 功能完整可用。

- 🎙 麦克风 / 系统音内录（Windows WASAPI loopback）
- ⚡ 流式识别：GPU/CPU 自适应，显存不足自动降级
- 🌐 翻译/语法模型可插拔：默认 DeepSeek API，可选本地 Index-Translate-2B（离线免费）
- 🪟 置顶半透明字幕窗 + 漫画气泡模式，外观可调
- 🧑‍🏫 可选语法分析：句块点语法卡、分析笔记、Markdown 导出
- 💾 会话自动落盘（txt / 双语 SRT / 语法分析 JSON）

## 快速开始（Windows，零开发基础）

1. **装 uv**（Python 包管理器，装一次即可）：打开 <https://docs.astral.sh/uv/getting-started/installation/> 按指引安装。
2. **下载本项目**：`git clone` 本仓库，或点 GitHub 页面 Code → Download ZIP 后解压。
3. **双击项目文件夹里的 `启动.bat`**：首次运行会自动安装依赖（约 1-2 GB，耐心等），然后弹出启动窗口。
4. **首次启动**：按提示下载语音识别模型（约 466MB，只需下载一次）。翻译默认走 DeepSeek 云端 API，**需要一个 API key**——到 <https://platform.deepseek.com/> 注册并在「API keys」页创建，粘贴到启动器的「API 密钥」框（只填一次，自动保存）。没有 key 也能启动，但只能看到原文字幕。
5. 在启动窗口选好语言（如 日语 → 中文），点「启动」，把悬浮窗拖到合适位置即可。

> 磁盘占用：完整安装约 5-6 GB（依赖 + 模型）。没有 NVIDIA 显卡也能用（自动 CPU 模式，速度略慢）。

**想用本地翻译模型（离线、免费、不吃 API 额度）**：设置页把翻译模型改绑到 Index-Translate-2B，按提示下载模型文件（约 1.3GB）即可；需要先安装 llama.cpp（见下方常见问题第 3 条）。

## 常见问题（FAQ）

**Q：字幕只有原文，没有翻译？**
A：翻译 API 密钥没填或不对。打开 设置 → Provider，检查 DeepSeek 的 api_key；密钥在 <https://platform.deepseek.com/> 创建。语法分析（句子成分讲解）也用这个 key，并需在启动器勾选「语法分析」才开启。

**Q：启动时一直卡在下载/下载失败？**
A：多为网络问题。模型直连 HuggingFace 失败会自动切换国内镜像重试；仍失败可先设镜像再启动（PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。`uv sync` 慢可给 uv 配国内 PyPI 镜像。

**Q：想用本地翻译模型，提示找不到 llama-server？**
A：本地模型依赖 llama.cpp：到 <https://github.com/ggml-org/llama.cpp/releases> 下载 Windows 版压缩包，解压后把文件夹加入系统 PATH（或在 provider 配置里填 llama-server 的绝对路径）。

**Q：Windows SmartScreen / 杀毒软件拦截？**
A：本项目及 llama.cpp 均为开源软件，属误报。SmartScreen 点「仍要运行」；杀毒软件把项目文件夹加入信任区。

**Q：显卡不是 NVIDIA / 显存不够？**
A：自动降级到 CPU 运行，功能不变、速度略慢，无需任何操作。

**Q：内录没声音？**
A：启动器音频来源选「系统音内录」，设备名留空用默认输出设备；选错了设备名关键词会导致录不到。

## 更多文档

- 命令行参数、配置文件、Provider/槽位、回放/批处理等进阶内容：[doc/使用与配置详解.md](doc/使用与配置详解.md)
- 开发环境、测试、设计约定、路线图：[doc/开发.md](doc/开发.md)
- 历史设计文档（需求文档、各阶段设计 Prompt，仅供参考）：[doc/archive/](doc/archive/)
