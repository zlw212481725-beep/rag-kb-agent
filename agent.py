"""
agent.py —— RAG 知识库问答 Agent（项目 1 正式版）

和 mini-agent 的区别（mini-agent 是学习轨迹，这个是简历主菜）：
  1. 回答强制带出处：每句话能说清来自「哪个文件 › 哪个标题」，可核查
  2. 引用校验：回答落地前把每条出处拿回检索结果里对账，
     编造的标题当场告警并记进 trace（实测模型真会编出处，见 check_citations）
  3. 多轮追问记忆：能接着上一轮继续问（"那它什么时候开始？"），不用重述背景
  4. 查不到就明说：三段式证据门控（硬拒答 / 证据偏弱警告 / 证据充分）压幻觉
  5. 砍掉 write_summary：正式版聚焦"问答 + 引用"，写文件是 mini-agent 的学习用例

跑法：
  python agent.py "我的六级什么时候补考"     # 问一句就走
  python agent.py                            # 进入连续对话（支持追问），输入 exit 退出
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag_lib import semantic_search, NOTES_DIR, TOP_K

# Windows 控制台默认 GBK 码页，下面的 🎯 🔧 💬 会抛 UnicodeEncodeError，先切 UTF-8
# stdin 也要切：否则管道/重定向传入的中文会被当 GBK 解码而变成乱码
for _stream in (sys.stdout, sys.stdin):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------- 1. 配置 ----------
load_dotenv()
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),   # 换厂商 = 改 .env 两行，代码不动
    timeout=30.0,      # 30 秒没响应就放弃，别干等
    max_retries=2,     # 网络抖动/限流时 SDK 自动重试 2 次
)
MODEL = os.environ["LLM_MODEL"]
OUTPUT_DIR = Path("output")
TRACE_FILE = OUTPUT_DIR / "trace.jsonl"
MAX_STEPS = 6   # 步数预算 ≈ 工具数(3) + 最终回答(1) + 余量(2)；mini-agent 实战调参得出的算法


# ---------- 2. 工具层 ----------
def clip(text: str, limit: int = 800) -> str:
    """工具结果统一限长。超长就截断，并告诉模型下一步怎么办。
    为什么：模型上下文又贵又有限，塞一大坨会烧钱还干扰判断。
    实战教训：mini-agent 里 1453 字的笔记被截到 800 字后，模型自己发现不完整、
    主动补搜关键词核实，还在回答里说明了补救策略——截断提示能引导模型行为。"""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(结果太长已截断，原文共 {len(text)} 字；建议缩小范围或分次读取)"


def kb_search(query: str, top_k: int = None) -> str:
    """语义检索知识库（主力工具）。返回带出处和相似度的候选段落。
    库里没有时返回"无证据"提示，不给模型任何可编造的素材。"""
    try:
        return semantic_search(query, top_k)
    except Exception as e:
        return f"语义检索暂不可用：{e}"


def kb_read(filename: str) -> str:
    """按文件名读整篇笔记（和 kb_search 配套：先搜到出处，再读全文看细节）。"""
    matches = [md for md in NOTES_DIR.rglob("*.md") if md.name == filename]
    if not matches:
        # 错误信息要能指导模型的下一步动作，而不是一句"出错了"
        return f"没有找到名为「{filename}」的笔记。可以先用 kb_search 搜一句自然语言描述，从结果的出处里复制文件名"
    if len(matches) > 1:
        paths = "\n".join(str(m.relative_to(NOTES_DIR)) for m in matches)
        return f"有 {len(matches)} 个同名文件，请用相对路径精确指定：\n{paths}"
    try:
        return matches[0].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return f"读取「{filename}」失败：{e}"


def kb_grep(keyword: str, max_hits: int = 8) -> str:
    """关键词兜底：在所有笔记里找字面包含 keyword 的行。
    什么时候用：语义检索没结果，但你知道某个专有名词的确切写法（如版本号、命令名）。"""
    hits = []
    for md in NOTES_DIR.rglob("*.md"):
        if ".obsidian" in md.parts:
            continue
        if len(hits) >= max_hits:
            break
        try:
            lines = md.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines, 1):
            if keyword.lower() in line.lower():
                hits.append(f"{md.name} 第{i}行: {line.strip()[:150]}")
                if len(hits) >= max_hits:
                    break
    return "\n".join(hits) if hits else f"笔记里没找到字面包含「{keyword}」的行"


# 给模型看的"工具说明书"（Function Calling 标准 JSON Schema）
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": (
                "在用户知识库里做语义检索（主力工具，回答任何问题前先调它）。"
                "query 必须是一句完整的自然语言（例如：英语六级什么时候补考），"
                "不要堆砌关键词——堆关键词会稀释语义、降低检索质量。"
                "返回结果带出处（文件名 › 标题）和相似度分数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "想找的内容，用自然语言一整句描述"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_read",
            "description": "读取指定笔记的完整内容。先用 kb_search 拿到出处文件名，再用本工具读全文看细节。",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "笔记文件名，例如 用户画像.md"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_grep",
            "description": "关键词兜底检索：找字面包含某个词的行。仅在 kb_search 没结果、且你确定某个专有名词确切写法时用。",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string", "description": "要精确匹配的词，如版本号、命令名"}},
                "required": ["keyword"],
            },
        },
    },
]
TOOL_FUNCS = {"kb_search": kb_search, "kb_read": kb_read, "kb_grep": kb_grep}


# ---------- 2.5 引用校验（把“编造出处”从静默幻觉变成可见告警） ----------
# 踩到的真坑：要求模型标出处之后，它居然会编出不存在的标题。
# 实测一例：回答里写了「来源：使用说明.md › 目录结构」，
# 而《使用说明》根本没有“目录结构”这个标题——文件名对、标题是它自己加的。
# 强制引用提高了可信度，但也开了一个新的幻觉面，所以得把引用反查一遍。
#
# 校验器自己也踩过一个坑：最早写成 r"来源：([^）)\n]+)"，遇到第一个右括号就截断。
# 可笔记标题里带全角括号是很常见的形态（实测一例：项目看板.md › 健身计划（⏸ 暂缓）），
# 标题被截残成「健身计划（⏸ 暂缓」，和真实检索结果对不上 → 把合法引用误报成编造。
# 所以下面允许标题内含一层配对括号（全角/半角都行）；两层以上嵌套不支持，实际数据里没出现过。
CITATION_RE = re.compile(                      # 抓回答里的（来源：XX › YY）
    r"来源：((?:[^（）()\n]|（[^（）()\n]*）|\([^（）()\n]*\))+)"
)
EVIDENCE_RE = re.compile(r"\[出处：(.+?)｜")    # 抓工具返回的[出处：XX › YY｜分数]


def _split_where(where: str):
    """把「文件.md › 标题」拆成 (文件, 标题)；没标题时标题为空字串。"""
    where = where.strip()
    if "›" in where:
        f, h = where.split("›", 1)
        return f.strip(), h.strip()
    return where, ""


def collect_evidence(observations: list) -> set:
    """从本轮工具原始返回里取出所有真正检索到过的 (文件, 标题)。"""
    pairs = set()
    for obs in observations:
        for m in EVIDENCE_RE.finditer(obs):
            pairs.add(_split_where(m.group(1)))
    return pairs


def check_citations(answer: str, evidence: set) -> list:
    """把回答里的每条出处拿去和真实检索结果对账，返回对不上的那些。

    注意用未截断的工具原文来建 evidence：clip() 只给模型看 800 字，
    拿截断后的文本对账会把合法引用误判成编造。
    """
    bad = []
    for m in CITATION_RE.finditer(answer):
        f, h = _split_where(m.group(1))
        if (f, h) in evidence:
            continue
        if any(f == ef for ef, _ in evidence):
            bad.append(f"{f} › {h}（文件确实检索到了，但这个标题不在检索结果里）")
        else:
            bad.append(f"{f}（整场对话里这个文件从没被检索到过）")
    return bad


# ---------- 3. Policy：模型的行为准则 ----------
SYSTEM_PROMPT = f"""你是一个运行在用户本机、只读用户 Obsidian 知识库的问答 Agent。

规则（Policy）：
1. 回答任何问题前，先用 kb_search 检索真实笔记。query 用完整自然语句，不要堆关键词。
2. 每条结论都要标出处，格式：（来源：文件名 › 标题）。出处直接取自检索结果，不许自己编。
3. kb_search 的返回分三种情况，分开处理：
   · 正常返回段落 → 正常回答，每条结论标出处。
   · 返回「知识库里没有相关内容」 → 就如实说没有，不许凭记忆猜、
     不许拿常识冒充用户的笔记内容。可以在说明“库里没有”之后另外用常识补充，
     但必须明确区分哪句来自笔记、哪句是你的常识。
   · 返回「⚠️ 证据偏弱」 → 说明检索到的段落可能答非所问。先自己判断这些材料
     能不能真正支撑答案：能就答并标出处，不能就明说库里没有，不要硬凑。
4. 想看细节就用 kb_read 读全文；语义搜不到但确定专有名词写法时用 kb_grep 兜底。
5. 用户追问时（如"那它什么时候开始？"），结合对话历史理解指代，必要时重新检索。
   沿用上一轮的内容时，出处必须在本轮重新检索确认过——不许直接抄上一轮回答里的
   （来源：……）当证据。抄来的出处会被程序对账，对不上就当场标记为编造。
6. 你每轮最多执行 {MAX_STEPS} 步，动作要节约，不要反复搜同一个意思。
7. 回答用中文，简洁直给，先给结论再给依据。"""


# ---------- 4. Trace：每步留痕 ----------
def dump_trace(records: list):
    """追加写 JSONL。每行一条记录，可审计、可复盘、可统计 token 花在哪。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- 5. Agent Loop ----------
def run_turn(history: list, goal: str, turn: int, evidence_pool: list) -> str:
    """跑一轮：模型思考 → 调工具 → 看结果 → 再思考，直到给出最终回答或步数用完。

    history 是跨轮记忆，只装「用户问题 + 最终回答」。
    为什么不把中间的工具调用也留着？—— ①省 token；②上一轮的旧检索结果
    会干扰这一轮的判断（模型可能直接抄旧段落回答新问题）。
    中间过程另存 trace.jsonl，一样可复盘。

    evidence_pool 反过来是「整场会话」级的，由 main() 建好后逐轮传进来累积。
    为什么和 history 不同尺度？—— 跨轮追问时沿用上一轮真检索到的出处是合法行为，
    每轮清零会把它误报成幻觉（实测第2轮问「那它什么时候开始」，模型一次工具都没调，
    直接沿用上一轮的笔记，池子空了就全判编造）。累积之后误报消失，而真编造照样抓得住：
    实测模型引用的 训练计划.md 从头到尾没被任何一轮检索到，累积了也还是被揪出来。
    """
    print(f"\n🎯 [第{turn}轮] {goal}")
    messages = history + [{"role": "user", "content": goal}]
    trace_records = []
    OUTPUT_DIR.mkdir(exist_ok=True)

    for step in range(1, MAX_STEPS + 1):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SPECS)
        except Exception as e:
            return f"❌ 第 {step} 步调用模型失败：{e}"

        msg = resp.choices[0].message
        tokens = resp.usage.total_tokens if resp.usage else 0

        if not msg.tool_calls:      # 模型不再调工具 = 它认为可以回答了
            answer = msg.content or "（模型没有返回内容）"
            # 回答落地前先对账出处：编造的引用要当场揭穿，不能静默放过去
            bad = check_citations(answer, collect_evidence(evidence_pool))
            if bad:
                print(f"  ⚠️ 引用校验：{len(bad)} 条出处对不上检索结果")
                for b in bad:
                    print(f"     ✗ {b}")
            trace_records.append({
                "turn": turn, "step": step, "tool": "FINAL_ANSWER", "args": None,
                "observation": answer[:200], "tokens": tokens, "ts": int(time.time()),
                "bad_citations": bad,
            })
            dump_trace(trace_records)
            # 只有问答对进长期记忆，中间脚手架不进
            history.append({"role": "user", "content": goal})
            history.append({"role": "assistant", "content": answer})
            return answer

        messages.append(msg)        # 把"我要调工具"这个决定记回对话
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  [第{step}步] 🔧 {name}({args})")

            func = TOOL_FUNCS.get(name)
            if func is None:
                observation = f"错误：没有名为 {name} 的工具，可用工具：{list(TOOL_FUNCS)}"
            else:
                try:
                    raw = str(func(**args))
                    evidence_pool.append(raw)   # 存未截断原文（clip 后的文本不能用来对账）
                    # 只追加不清空：这一轮检索到的出处，后续轮次沿用也算合法
                    observation = clip(raw)
                except Exception as e:
                    observation = f"工具执行出错：{e}"   # 出错不崩溃，喂回给模型自己换路子
            print(f"          -> {observation[:100]}")

            trace_records.append({
                "turn": turn, "step": step, "tool": name, "args": args,
                "observation": observation[:200], "tokens": tokens, "ts": int(time.time()),
            })
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

    dump_trace(trace_records)
    return f"⚠️ 达到最大步数 {MAX_STEPS} 仍未完成，中间过程见 {TRACE_FILE}"


# ---------- 6. 入口 ----------
def main():
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    evidence_pool = []      # 整场会话累积的工具原文，跨轮引用校验靠它

    if len(sys.argv) > 1:                       # 单问模式
        answer = run_turn(history, " ".join(sys.argv[1:]), turn=1, evidence_pool=evidence_pool)
        print(f"\n💬 回答：\n{answer}")
        return

    print("📚 知识库问答 Agent（连续对话模式，输入 exit / quit 退出）")
    print(f"   知识库：{NOTES_DIR}｜召回条数 top_k={TOP_K}")
    turn = 0
    while True:
        try:
            goal = input("\n你问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not goal:
            continue
        if goal.lower() in {"exit", "quit", "q"}:
            print("再见")
            break
        turn += 1
        answer = run_turn(history, goal, turn, evidence_pool)
        print(f"\n💬 回答：\n{answer}")


if __name__ == "__main__":
    main()
