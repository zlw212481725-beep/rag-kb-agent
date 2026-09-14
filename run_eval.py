"""
run_eval.py —— 检索层评测（项目 1 的"体检仪"）

为什么要有它：改切块策略、加 rerank、调阈值，凭感觉说"好像变准了"不算数。
AgentGuide 那句话是对的——**没有 Eval 的 Agent 只是 demo**。
这个脚本用 20 条固定 case 打分，改一次跑一次，涨了还是跌了一眼看出来。

关键取舍：只测「检索层」，不测「生成层」。
  · 不调 LLM 来评判答案好坏 → 不烧钱、跑得快、结果 100% 可复现
  · 检索错了生成一定错；检索对了生成才有机会对 → 先守住检索这道关
  · 代价：测不出"答得好不好读"，那部分靠人工验收（README 有记录）

用法：
  python run_eval.py                     # 跑分（默认开 rerank），存 output/eval_report_<tag>.md
  python run_eval.py --both              # ★ 粗排 vs 精排 同场对比，一张表看涨幅
  python run_eval.py --no-rerank         # 只跑纯向量粗排
  python run_eval.py --tag rerank        # 给本轮贴标签，方便留档对比
  python run_eval.py --top-k 3           # 换个召回条数看指标怎么变

指标怎么读（分两个层级，别混）：
  Hit@1/@3/@k   文件级——期望文件有没有进前 N 名（面试常问的 Recall@k 就是这个）
  MRR           文件级平均倒数排名：第 1 名得 1 分，第 2 名 0.5，第 3 名 0.33，没进榜 0
  Ans@3         答案级——前 3 名里有没有"既是期望文件、又真含期望关键词"的块
                （光找对文件不够，找对文件里的错段落照样答错，这才是真正的合格线）
  误拒率        本来能答的问题，被证据门控硬判成"库里没有"的比例（越低越好）
  拒答拦截率    库里没有的问题，被门控拦住（硬拒答或降级为"证据偏弱"警告）的比例
  漏网率        库里没有的问题却拿到"证据充分"判定的比例 —— 这就是幻觉的入口
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Windows 控制台默认 GBK 码页，本脚本要打印 ✅ ❌ › 这类字符，
# 不强制切 UTF-8 会直接抛 UnicodeEncodeError，把跑分中断在半路。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
from rag_lib import (  # noqa: E402
    retrieve, evidence_gate, evidence_score, USE_RERANK, TOP_K, GATE_LOW, GATE_SOFT,
)

CASE_FILE = Path("eval_cases.jsonl")
OUTPUT_DIR = Path("output")


def load_cases() -> list:
    """读 eval case。一行一个 JSON，加 case 只要往文件末尾追加一行。"""
    cases = []
    with CASE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def rank_of(hits: list, files: list, need_kw: bool = False, kw: list = None):
    """在结果里找期望文件的排名（0 起算）。
    need_kw=True 时更严格：还要该块正文里真含期望关键词（= 找到了答案本身）。"""
    for i, h in enumerate(hits):
        if h["file"] not in files:
            continue
        if not need_kw:
            return i
        if any(k in h["text"] for k in (kw or [])):
            return i
    return None


def evaluate(top_k: int, use_rerank: bool) -> list:
    """逐条跑 case。每条 case 花 1 次 embedding（+ 开精排时 1 次 rerank）请求。"""
    rows = []
    for c in load_cases():
        hits = retrieve(c["q"], top_k, use_rerank)
        top1 = hits[0]["score"] if hits else 0.0   # 排序分（开精排时=rerank 相关性）
        ev = evidence_score(hits)                  # 门控分（永远=cosine，两种模式可比）
        gate = evidence_gate(ev) if hits else "none"

        if c["must_hit"]:
            r_file = rank_of(hits, c["files"])
            r_ans = rank_of(hits, c["files"], need_kw=True, kw=c["kw"])
            # 合格线：前 3 名里真有含答案的块，且没被门控误判成"库里没有"
            passed = r_ans is not None and r_ans < 3 and gate != "none"
            rows.append({
                "id": c["id"], "cat": c["cat"], "q": c["q"], "must_hit": True,
                "r_file": r_file, "r_ans": r_ans,
                "rr": round(1 / (r_file + 1), 3) if r_file is not None else 0.0,
                "hit1": r_file == 0, "hit3": r_file is not None and r_file < 3,
                "hitk": r_file is not None,
                "ans3": r_ans is not None and r_ans < 3,
                "top1": top1, "ev": ev, "gate": gate, "got_file": hits[0]["file"] if hits else "-",
                "passed": passed, "note": c.get("note", ""),
            })
        else:
            # 拒答类：gate=ok 就是漏网（模型会拿着无关段落硬答）；none/weak 都算拦住
            rows.append({
                "id": c["id"], "cat": c["cat"], "q": c["q"], "must_hit": False,
                "r_file": None, "r_ans": None, "rr": 0.0,
                "hit1": False, "hit3": False, "hitk": False, "ans3": False,
                "top1": top1, "ev": ev, "gate": gate, "got_file": hits[0]["file"] if hits else "-",
                "passed": gate != "ok", "note": c.get("note", ""),
            })
    return rows


def summarize(rows: list, top_k: int, use_rerank: bool) -> dict:
    hits = [r for r in rows if r["must_hit"]]
    refs = [r for r in rows if not r["must_hit"]]
    n, m = len(hits), len(refs)
    pct = lambda a, b: round(a / b, 3) if b else 0.0
    s = {
        "total": len(rows), "n_hit": n, "n_refuse": m,
        "top_k": top_k, "use_rerank": use_rerank,
        "mode": "粗排(纯向量)" if not use_rerank else "粗排+精排(rerank)",
        "hit1": pct(sum(r["hit1"] for r in hits), n),
        "hit3": pct(sum(r["hit3"] for r in hits), n),
        "hitk": pct(sum(r["hitk"] for r in hits), n),
        "mrr": round(sum(r["rr"] for r in hits) / n, 3) if n else 0,
        "ans3": pct(sum(r["ans3"] for r in hits), n),
        "false_refuse": pct(sum(r["gate"] == "none" for r in hits), n),
        "refuse_block": pct(sum(r["passed"] for r in refs), m),
        "hard_block": pct(sum(r["gate"] == "none" for r in refs), m),
        "leak": pct(sum(r["gate"] == "ok" for r in refs), m),
        "pass_rate": pct(sum(r["passed"] for r in rows), len(rows)),
    }
    # 分数分布统一看门控分（cosine），这样开不开精排都能直接比
    hit_scores = sorted(r["ev"] for r in hits)
    ref_scores = sorted(r["ev"] for r in refs)
    s["hit_min"] = hit_scores[0] if hit_scores else 0
    s["hit_max"] = hit_scores[-1] if hit_scores else 0
    s["ref_min"] = ref_scores[0] if ref_scores else 0
    s["ref_max"] = ref_scores[-1] if ref_scores else 0
    s["overlap"] = round(min(s["hit_max"], s["ref_max"]) - max(s["hit_min"], s["ref_min"]), 3)
    return s


def print_console(rows: list, s: dict):
    print("\n" + "=" * 86)
    print(f"逐条结果 · {s['mode']} · top_k={s['top_k']}（排名从 1 起算，— = 没进榜）")
    print("=" * 86)
    for r in rows:
        mark = "✅" if r["passed"] else "❌"
        if r["must_hit"]:
            rf = str(r["r_file"] + 1) if r["r_file"] is not None else "—"
            ra = str(r["r_ans"] + 1) if r["r_ans"] is not None else "—"
            detail = (f"文件rank={rf:>2} 答案rank={ra:>2} 排序分={r['top1']:.3f} "
                      f"证据分={r['ev']:.3f} 门控={r['gate']}")
        else:
            detail = (f"证据分={r['ev']:.3f} 门控={r['gate']:<5} 最像={r['got_file']}")
        print(f"{mark} {r['id']} [{r['cat']}] {r['q']}")
        print(f"     {detail}")
        if not r["passed"] and r["note"]:
            print(f"     ↳ 备注：{r['note']}")

    print("\n" + "=" * 86)
    print(f"汇总指标 · {s['mode']}")
    print("=" * 86)
    print(f"  Hit@1 / Hit@3 / Hit@{s['top_k']}   {s['hit1']:.1%} / {s['hit3']:.1%} / {s['hitk']:.1%}   （文件级，{s['n_hit']} 条）")
    print(f"  MRR                {s['mrr']:.3f}")
    print(f"  Ans@3              {s['ans3']:.1%}   （答案级：找对文件且找对段落）")
    print(f"  误拒率             {s['false_refuse']:.1%}   （能答却被硬判「库里没有」）")
    print(f"  拒答拦截率         {s['refuse_block']:.1%}   （{s['n_refuse']} 条库里没有的问题；其中硬拦截 {s['hard_block']:.1%}，余为降级警告）")
    print(f"  漏网率             {s['leak']:.1%}   （无关问题拿到「证据充分」判定 = 幻觉入口）")
    print(f"  总通过率           {s['pass_rate']:.1%}   （{s['total']} 条）")
    print(f"\n  证据分（cosine）分布：能答类 {s['hit_min']:.3f}~{s['hit_max']:.3f} ｜ 拒答类 {s['ref_min']:.3f}~{s['ref_max']:.3f}")
    print(f"            重叠宽度 {s['overlap']:+.3f}；当前门控 LOW={GATE_LOW:.2f} / SOFT={GATE_SOFT:.2f}")
    print(f"            → {'两类分得开' if s['overlap'] <= 0 else '两类重叠：单阈值无解，靠三段式把模糊地带交给模型判断'}")


def print_compare(a: dict, b: dict):
    """粗排 vs 精排 同场对比。"""
    print("\n" + "=" * 86)
    print("A/B 对比：纯向量粗排  →  粗排+rerank 精排")
    print("=" * 86)
    rows = [
        ("Hit@1", "hit1", True), ("Hit@3", "hit3", True), (f"Hit@{a['top_k']}", "hitk", True),
        ("MRR", "mrr", False), ("Ans@3", "ans3", True), ("误拒率", "false_refuse", False),
        ("拒答拦截率", "refuse_block", True), ("硬拦截率", "hard_block", True),
        ("漏网率", "leak", False), ("总通过率", "pass_rate", True),
    ]
    print(f"{'指标':<14}{'粗排':>10}{'精排':>10}{'变化':>12}")
    print("-" * 48)
    for label, key, higher_better in rows:
        va, vb = a[key], b[key]
        d = vb - va
        arrow = "→" if abs(d) < 1e-9 else ("↑" if (d > 0) == higher_better else "↓")
        fmt = (lambda v: f"{v:.3f}") if key == "mrr" else (lambda v: f"{v:.1%}")
        print(f"{label:<14}{fmt(va):>10}{fmt(vb):>10}{arrow + ' ' + fmt(abs(d)):>12}")


def write_report(rows: list, s: dict, tag: str, compare: dict = None) -> Path:
    """存成 markdown，方便直接贴进 README 当指标证据。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"eval_report_{tag}.md"
    L = [
        f"# Eval 报告 · {tag}",
        "",
        f"- 跑分时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 模式：{s['mode']}｜top_k={s['top_k']}",
        f"- case 数：{s['total']}（检索类 {s['n_hit']} + 拒答类 {s['n_refuse']}）",
        "",
        "## 指标",
        "",
        "| 指标 | 数值 | 含义 |",
        "| --- | --- | --- |",
        f"| Hit@1 | {s['hit1']:.1%} | 期望文件排第 1 |",
        f"| Hit@3 | {s['hit3']:.1%} | 期望文件进前 3 |",
        f"| Hit@{s['top_k']} | {s['hitk']:.1%} | 期望文件进前 {s['top_k']} |",
        f"| MRR | {s['mrr']:.3f} | 文件级平均倒数排名 |",
        f"| Ans@3 | {s['ans3']:.1%} | 前 3 名里真有含答案的块 |",
        f"| 误拒率 | {s['false_refuse']:.1%} | 能答却被硬判「库里没有」 |",
        f"| 拒答拦截率 | {s['refuse_block']:.1%} | 库里没有的问题被拦住（硬拦截 {s['hard_block']:.1%} + 降级警告）|",
        f"| 漏网率 | {s['leak']:.1%} | 无关问题拿到「证据充分」（幻觉入口）|",
        f"| **总通过率** | **{s['pass_rate']:.1%}** | 全部 {s['total']} 条 |",
        "",
    ]
    if compare:
        L += ["## A/B 对比：纯向量粗排 → 粗排+rerank 精排", "",
              "| 指标 | 粗排 | 精排 | 变化 |", "| --- | --- | --- | --- |"]
        for label, key, hb in [("Hit@1", "hit1", 1), ("Hit@3", "hit3", 1), (f"Hit@{s['top_k']}", "hitk", 1),
                               ("MRR", "mrr", 1), ("Ans@3", "ans3", 1), ("误拒率", "false_refuse", 0),
                               ("拒答拦截率", "refuse_block", 1), ("硬拦截率", "hard_block", 1),
                               ("漏网率", "leak", 0), ("总通过率", "pass_rate", 1)]:
            va, vb = compare["a"][key], compare["b"][key]
            d = vb - va
            arrow = "→" if abs(d) < 1e-9 else ("↑" if (d > 0) == hb else "↓")
            f_ = (lambda v: f"{v:.3f}") if key == "mrr" else (lambda v: f"{v:.1%}")
            L.append(f"| {label} | {f_(va)} | {f_(vb)} | {arrow} {f_(abs(d))} |")
        L.append("")
    L += ["## 逐条明细", "",
          "| ID | 类别 | 问题 | 结果 | 文件rank | 答案rank | 排序分 | 证据分(cos) | 门控 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        rf = str(r["r_file"] + 1) if r["r_file"] is not None else "—"
        ra = str(r["r_ans"] + 1) if r["r_ans"] is not None else "—"
        L.append(f"| {r['id']} | {r['cat']} | {r['q']} | {'✅' if r['passed'] else '❌'} "
                 f"| {rf} | {ra} | {r['top1']:.3f} | {r['ev']:.3f} | {r['gate']} |")
    fails = [r for r in rows if not r["passed"]]
    L += ["", "## 失败案例与病因", ""]
    if fails:
        for r in fails:
            L += [f"- **{r['id']}**（{r['cat']}）{r['q']}",
                  f"  - 精排/粗排第一名：`{r['got_file']}`，top1={r['top1']:.3f}，门控={r['gate']}",
                  f"  - 病因：{r['note'] or '待分析'}"]
    else:
        L.append("- 本轮无失败案例")
    L += ["", "## 分数分布与门控分析", "",
          f"- 门控依据：整池最高 cosine（不用 rerank 分数，原因见 `rag_lib.evidence_score`）",
          f"- 能答类证据分区间：{s['hit_min']:.3f} ~ {s['hit_max']:.3f}",
          f"- 拒答类证据分区间：{s['ref_min']:.3f} ~ {s['ref_max']:.3f}",
          f"- 重叠宽度：{s['overlap']:+.3f}",
          f"- 当前门控：LOW={GATE_LOW:.2f}（低于=硬拒答） / SOFT={GATE_SOFT:.2f}（低于=证据偏弱）",
          "",
          "> **为何单阈值不可行**：要不误拒就得 T ≤ 能答类最低分，要不漏网就得 T > 拒答类最高分，"
          "两者重叠时无解。三段式把矛盾拆开：模糊地带不硬判，"
          "而是把分数和出处一起交给模型，让它自己判断材料够不够。",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description="RAG 检索层评测")
    ap.add_argument("--tag", default=None, help="本轮跑分标签（默认按模式自动取 baseline / rerank）")
    ap.add_argument("--top-k", type=int, default=TOP_K, help=f"最终召回条数（默认 {TOP_K}）")
    ap.add_argument("--no-rerank", action="store_true", help="关掉精排，只测纯向量粗排")
    ap.add_argument("--both", action="store_true", help="粗排和精排各跑一遍，输出 A/B 对比")
    args = ap.parse_args()

    cases = load_cases()
    use_rr = USE_RERANK and not args.no_rerank

    if args.both:
        print(f"A/B 评测：{len(cases)} 条 case × 2 种模式，top_k={args.top_k}")
        rows_a = evaluate(args.top_k, False)
        sa = summarize(rows_a, args.top_k, False)
        print_console(rows_a, sa)
        rows_b = evaluate(args.top_k, True)
        sb = summarize(rows_b, args.top_k, True)
        print_console(rows_b, sb)
        print_compare(sa, sb)
        tag = args.tag or "ab_compare"
        path = write_report(rows_b, sb, tag, compare={"a": sa, "b": sb})
        print(f"\n报告已保存：{path}")
        return

    print(f"开始评测：{len(cases)} 条 case，top_k={args.top_k}，rerank={'开' if use_rr else '关'}")
    try:
        rows = evaluate(args.top_k, use_rr)
    except RuntimeError as e:
        raise SystemExit(f"评测中止：{e}")
    s = summarize(rows, args.top_k, use_rr)
    print_console(rows, s)
    tag = args.tag or ("rerank" if use_rr else "baseline")
    path = write_report(rows, s, tag)
    print(f"\n报告已保存：{path}")


if __name__ == "__main__":
    main()
