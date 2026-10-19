"""instTrans 契约测试：声明 glossary_constraint 能力的 Provider 走术语表硬约束映射。"""

from pathlib import Path

from src.config import ProviderConfig
from src.nlp.llm_client import ChatResult
from src.nlp.prompt_engine import PromptEngine
from src.nlp.translate import Translator
from src.profiles.loader import get_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FakeInstTransProvider:
    """模拟翻译专用模型：chat 返回多行纯文本译文；记录 json_mode 是否被误用。"""

    def __init__(self, script):
        self.cfg = ProviderConfig(
            id="local-tr", capabilities=["translation", "glossary_constraint"]
        )
        self.script = list(script)
        self.calls: list[dict] = []
        self.online = True

    def chat(self, prompt, *, system=None, json_mode=False):
        self.calls.append({"prompt": prompt, "json_mode": json_mode})
        item = self.script.pop(0) if self.script else ""
        if isinstance(item, Exception):
            self.online = False
            raise item
        return ChatResult(item, prompt_tokens=10, completion_tokens=5)


def make(script):
    provider = FakeInstTransProvider(script)
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(lambda: provider, prompts, profile)
    return t, provider


def test_insttrans_prompt_and_line_alignment():
    t, provider = make(["早上好\n已经回去了？"])
    results = t.translate_block(["おはよう", "もう帰った？"])
    assert [r.translation for r in results] == ["早上好", "已经回去了？"]
    call = provider.calls[0]
    assert call["json_mode"] is False  # 纯文本契约不得用 json_mode
    prompt = call["prompt"]
    # 官方 canonical 结构：【源文】+【约束要求】（1.硬性要求 2.注意）
    assert "【源文】" in prompt and "【硬性要求】" in prompt and "【注意】" in prompt
    assert "日语" in prompt and "中文" in prompt  # 语言代码映射为中文名
    assert "おはよう" in prompt and "もう帰った？" in prompt


def test_insttrans_glossary_injected_as_hard_constraint():
    t, provider = make(["小雪已经回去了", "请转告小雪"])
    t.memory.set_term("ユキ", "小雪")
    t.translate_block(["ユキはもう帰った"])
    prompt = provider.calls[0]["prompt"]
    # 术语表映射为【硬性要求】的一行式对照（官方格式：A→B、C→D）
    assert "【硬性要求】" in prompt and "ユキ→小雪" in prompt


def test_insttrans_think_block_stripped():
    """思考块泄露时剥离后再按行对齐（官方客户端同款防御）。"""
    t, _ = make(["<think>想了一下</think>\n早上好"])
    results = t.translate_block(["おはよう"])
    assert results[0].translation == "早上好"


def test_insttrans_history_injected():
    t, provider = make(["早上好", "已经回去了？"])
    t.translate_block(["おはよう"])
    t.translate_block(["もう帰った？"])
    assert "おはよう → 早上好" in provider.calls[1]["prompt"]


def test_insttrans_no_glossary_writeback():
    """纯文本契约无 glossary_updates：记忆不从此类响应回写。"""
    t, _ = make(["早上好"])
    t.translate_block(["おはよう"])
    assert len(t.memory.glossary) == 0


def test_insttrans_line_mismatch_falls_back():
    """行数不符 → 整块重试 → 逐句无上下文兜底（降级链完整保留）。"""
    t, provider = make([
        "只有一行译文",          # 块 2 句 → 行数不符，失败
        "仍然只有一行",          # 重试仍失败
        "早上好",                # 逐句兜底：第 1 句
        "已经回去了？",          # 逐句兜底：第 2 句
    ])
    results = t.translate_block(["おはよう", "もう帰った？"])
    assert [r.translation for r in results] == ["早上好", "已经回去了？"]
    assert len(provider.calls) == 4


def test_unbound_slot_returns_source_only():
    """翻译槽未绑定：不发起请求，仅显示原文。"""
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(lambda: None, prompts, profile)
    results = t.translate_block(["おはよう"])
    assert results[0].translation is None
    assert t.online is False


def test_hot_swap_between_providers():
    """换绑：getter 每次取当前 Provider，新请求立即走新 Provider。"""
    from src.nlp.translate import Translator  # noqa: F811

    current = {"p": FakeInstTransProvider(["译文A"])}
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(lambda: current["p"], prompts, profile)
    assert t.translate_block(["第一句"])[0].translation == "译文A"

    class JsonProvider(FakeInstTransProvider):
        def __init__(self):
            super().__init__([])
            self.cfg = ProviderConfig(id="cloud", capabilities=["translation", "grammar"])

    import json

    cloud = JsonProvider()
    cloud.script = [json.dumps(
        {"sentences": [{"id": 2, "source": "第二句", "translation": "译文B"}],
         "glossary_updates": [], "style_updates": ""}, ensure_ascii=False)]
    current["p"] = cloud
    assert t.translate_block(["第二句"])[0].translation == "译文B"
    assert cloud.calls[0]["json_mode"] is True  # 普通 Provider 走 JSON 契约


def test_insttrans_echo_lines_cleaned_before_align():
    """模型把约束块/回显行抄进输出时，清洗后再对齐（不进历史、不污染后续上下文）。"""
    t, provider = make([
        "【约束要求】\n死都这样啦\n感谢您一直看到最后",  # 回显行 + 2 行真译文
    ])
    results = t.translate_block(["死ぬかくそうだよ", "最後まで視聴してくださって"])
    assert [r.translation for r in results] == ["死都这样啦", "感谢您一直看到最后"]


def test_insttrans_consecutive_duplicates_collapsed():
    """连续重复行折叠后再对齐。"""
    t, _ = make(["蓬松柔软地\n蓬松柔软地\n轻轻抚摸也没关系哦"])
    results = t.translate_block(["ふわふわ", "撫でちゃってもいいですよ"])
    assert [r.translation for r in results] == ["蓬松柔软地", "轻轻抚摸也没关系哦"]


def test_history_rejects_junk_translation():
    """历史回写卫生：含换行/【】/与原文相同的译文不进历史（防污染级联）。"""
    import json

    class JunkLLM:
        cfg = None  # 无 capabilities → JSON 契约路径
        online = True

        def __init__(self):
            self.calls = 0

        def chat(self, prompt, **kw):
            self.calls += 1
            # 单句 + 纯文本 → 命中 JSON 路径的宽容分支，整段被当译文
            from src.nlp.llm_client import ChatResult

            return ChatResult("第一行\n第二行\n【约束要求】杂讯", prompt_tokens=1, completion_tokens=1)

    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(JunkLLM(), prompts, profile)
    results = t.translate_block(["おはよう"])
    assert results[0].translation is not None  # 译文本身照常输出（宽松）
    assert len(t._history) == 0  # 但不进历史，后续块上下文不受污染


def test_history_skips_source_echo():
    """译文与原文相同（未翻译回显）不进历史。"""
    from src.nlp.llm_client import ChatResult

    class EchoLLM:
        cfg = None
        online = True

        def chat(self, prompt, **kw):
            return ChatResult("おはよう", prompt_tokens=1, completion_tokens=1)

    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(EchoLLM(), prompts, profile)
    t.translate_block(["おはよう"])
    assert len(t._history) == 0
