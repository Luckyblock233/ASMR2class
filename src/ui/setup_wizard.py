"""首次启动设置向导（控制台版，FR-1.5 + 模型可插拔架构 §2）。

无任何可用翻译绑定时引导用户：①注册一个 OpenAI 兼容 Provider ②绑定到翻译槽。
api_key 输入不回显；用户可选择仅写入环境变量而非配置文件。
图形设置页（Provider 管理 + 槽位下拉）规划在 M3 与语法卡一起实现。
"""

from __future__ import annotations

import getpass

from ..config import (
    API_KEY_ENV,
    GlobalConfig,
    ProviderConfig,
    SlotsConfig,
    save_provider_config,
)


def _next_provider_id(cfg: GlobalConfig) -> str:
    """生成不冲突的 provider id：cloud-1、cloud-2…"""
    existing = {p.id for p in cfg.providers}
    n = 1
    while f"cloud-{n}" in existing:
        n += 1
    return f"cloud-{n}"


def run_setup_wizard(cfg: GlobalConfig) -> tuple[ProviderConfig, SlotsConfig]:
    """注册 Provider + 绑定翻译槽，写回 config.yaml。返回 (provider, 新槽位绑定)。"""
    print("=" * 60)
    print("首次启动设置向导：注册模型提供方（Provider）并绑定槽位")
    print("=" * 60)
    print("本工具不在代码中预设任何服务商，请填写你自己的 OpenAI 兼容服务信息。")
    print("（将写入 config/config.yaml 的 providers 与 slots 节；")
    print(" api_key 也可选仅使用环境变量；语法槽可稍后编辑 config.yaml 绑定）\n")

    base_url = input("base_url（如 https://your-provider.example.com/v1）: ").strip()
    while not base_url:
        base_url = input("base_url 不能为空，请重新输入: ").strip()

    model = input("model（模型名）: ").strip()
    while not model:
        model = input("model 不能为空，请重新输入: ").strip()

    print("\napi_key 处理方式：")
    print("  1) 写入 config.yaml（方便，注意不要提交到版本库）")
    print(f"  2) 不写入文件，每次运行前设置环境变量 {API_KEY_ENV}")
    choice = input("请选择 [1/2，默认 1]: ").strip() or "1"
    api_key = getpass.getpass("api_key（输入不回显）: ").strip()
    while not api_key:
        api_key = getpass.getpass("api_key 不能为空，请重新输入: ").strip()

    provider = ProviderConfig(
        id=_next_provider_id(cfg),
        name="云端通用模型（向导注册）",
        type="openai_compatible",
        base_url=base_url,
        api_key=api_key if choice == "1" else "",
        model=model,
        capabilities=["translation", "grammar"],
        timeout_sec=cfg.llm.timeout_sec,
        max_retries=cfg.llm.max_retries,
    )
    slots = SlotsConfig(translation=provider.id, grammar=cfg.slots.grammar)
    save_provider_config(cfg, provider, slots)

    if choice == "1":
        print("\n已写入 config/config.yaml。")
    else:
        print("\n已写入 base_url 与 model；下次运行前请设置环境变量：")
        print(f"  export {API_KEY_ENV}=<your-key>   # Linux/WSL")
        print(f'  setx {API_KEY_ENV} "<your-key>"   # Windows（新终端生效）')
    print(f"翻译槽已绑定：{provider.id}。语法槽可在 config.yaml 的 slots.grammar 绑定。")
    print("设置完成，正在启动…\n")
    # 返回内存中的完整 provider 供本次会话使用（文件中是否含 key 取决于上面的选择）
    if choice != "1":
        provider.api_key = api_key
    return provider, slots
