"""
rag_lib.py —— RAG 检索层（项目 1 正式版）

整条链路：
  建索引：md 文件 → 切块 → 向量化(embedding) → 存进 Chroma 向量库
  查询时：问题 → 向量化 → 粗排召回 RERANK_POOL 块 → reranker 精排 → 取前 top_k → 交给 Agent

为什么能按"意思"找？—— embedding 把每段文字变成一串数字（坐标），
意思相近的文字坐标也相近。所以问"学校层次会不会影响投简历"能搜到
"民办本科 → 分层投递"，哪怕一个字都对不上。关键词搜索做不到这件事。

为什么要两阶段（粗排 + 精排）？
  向量检索是"拿问题坐标和块坐标比距离"，快但粗——它把整块文字压成一个坐标，
  块里只要混进别的话题，坐标就被拉偏。实测（20 条 eval）：
    · "建索引切了多少块" 的正确块掉到第 6 名
    · "写知识库前要不要先问我" 被《使用说明》里语义相近但不含答案的段落挤到第 6 名
  reranker 不一样：它把「问题 + 每个候选块」成对送进模型打分，能看清
  "这块到底答没答这个问题"。加上精排后上面两条都回到第 1 名。
  代价：每次查询多一次 API 调用（约 +0.3~1s）。粗排负责"别漏"，精排负责"排准"。

正式版相比 mini-agent 的四处改动（都是为了"能测、能引用、敢拒答"）：
  1. 检索与展示解耦：retrieve() 只返回结构化结果（list of dict），
     semantic_search() 才负责拼成给模型看的文本。
     好处：eval 脚本能直接调 retrieve() 打分，不必启动 Agent、不烧 LLM 的钱。
  2. 粗排 + rerank 精排两阶段（USE_RERANK=0 可退回纯向量，方便 A/B 对比）。
  3. 元数据带 heading（这段属于哪个标题）：引用能说"求职总览.md › 投递策略"，
     而不是干巴巴的"第 3 块"。
  4. 三段式证据门控取代单阈值拒答，且门控用 cosine、排序用 rerank（两个实测结论，
     原因见 evidence_gate / evidence_score 的注释）。
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from db import db_manager

load_dotenv()

NOTES_DIR = Path(os.environ.get("NOTES_DIR", "./notes"))
CHROMA_DIR = Path("chroma_db")   # 向量库存盘位置（自动生成，别手改）
COLLECTION = "vault"


def _flag(name: str, default: bool) -> bool:
    """读 .env 里的开关。写成 0/false/no 都算关。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ---------- 检索参数（都能用 .env 覆盖） ----------
TOP_K = int(os.environ.get("TOP_K", "6"))            # 最终交给模型的块数
USE_RERANK = _flag("USE_RERANK", True)               # 精排开关（关掉=退回纯向量粗排）
RERANK_POOL = int(os.environ.get("RERANK_POOL", "16"))  # 粗排召回池：比 top_k 大，给精排留充足翻盘空间
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 证据门控阈值——注意：永远用 cosine 尺度的分数来门控，不用 rerank 分数（原因见 evidence_score）。
# 这两个值由 run_eval.py 的分数分布反推出来，不是拍脑袋定的：
#   GATE_LOW  必须 ≤ 「能答类」最低分（实测 0.506），否则会把能答的问题误拒
#   GATE_SOFT 必须 ≥ 「拒答类」最高分（实测 0.592），否则会把无关问题当成证据充分
# 单阈值做不到（要求 T ≤ 0.506 且 T > 0.592，矛盾无解）；三段式可以。
GATE_LOW = float(os.environ.get("GATE_LOW", "0.45"))    # 低于它=硬拒答，不给模型任何材料
GATE_SOFT = float(os.environ.get("GATE_SOFT", "0.60"))  # 低于它=证据偏弱，给材料但警告模型自己判断


# ---------- 第 1 步：切块 ----------
def split_sections(text: str):
    """按 Markdown 标题把全文切成"一段一个主题"，产出 (标题, 该段全文)。

    为什么按标题切？—— v1 用固定滑窗（每 500 字一刀）把"六级补考"那一行
    和一堆无关内容切进了同一块，整块坐标被稀释，检索排名直接掉出前三。
    改成按标题切，一块只说一件事，坐标就干净了。
    （实战教训：这是调试"检索不准"时用真金白银换来的结论）
    """
    for sec in re.split(r"\n(?=#{1,4} )", text):   # 在标题行前面断开
        sec = sec.strip()
        if not sec:
            continue
        first_line = sec.splitlines()[0]
        heading = first_line.lstrip("#").strip() if first_line.startswith("#") else ""
        yield heading, sec


def chunk_text(text: str, size: int = 450) -> list:
    """切块主函数：返回 [{"heading": 所属标题, "text": 块内容}, ...]

    规则：
    1. 先按 Markdown 标题切成主题段落；
    2. 主题段落内部，若包含列表项（支持任意缩进 \\n\\s*[-*] ），则按列表项精准切分；
    3. 彻底解决多级缩进超长 bullet 导致的关键词被平均稀释问题（治愈 Case E07）。
    """
    chunks = []
    for heading, sec in split_sections(text):
        sec = sec.strip()
        if len(sec) < 20:
            continue
        prefix = f"[{heading}] " if heading and not sec.startswith(f"# {heading}") else ""

        if len(sec) <= size:
            content = sec if sec.startswith(prefix) else (prefix + sec)
            chunks.append({"heading": heading, "text": content})
            continue

        # 检查是否包含列表项（包含带缩进的列表项）
        bullets = re.split(r"\n(?=\s*[-*] )", sec)
        if len(bullets) > 1:
            parent_bullet = ""
            for b in bullets:
                b_str = b.strip()
                if not b_str:
                    continue
                # 判断是否是顶级列表项还是缩进子列表
                is_sub_bullet = b.startswith("  ") or b.startswith("\t")
                if not is_sub_bullet and len(b_str) < 100:
                    parent_bullet = b_str.split("\n")[0] + "\n"

                # 继承父级列表上下文
                full_text = b_str if not is_sub_bullet or not parent_bullet or b_str.startswith(parent_bullet.strip()) else (parent_bullet + b_str)

                # 若单条较长且包含分号，按分号拆分子句
                if len(full_text) > 120 and ("；" in full_text or ";" in full_text):
                    sub_parts = re.split(r"[；;]\s*", full_text)
                    sub_prefix = sub_parts[0].split("：")[0] + "：" if "：" in sub_parts[0] else ""
                    for p in sub_parts:
                        p_clean = p.strip()
                        if len(p_clean) >= 15:
                            item_text = p_clean if p_clean.startswith(sub_prefix) else (sub_prefix + p_clean)
                            chunks.append({"heading": heading, "text": prefix + item_text})
                else:
                    if len(full_text) >= 15:
                        chunks.append({"heading": heading, "text": prefix + full_text})
            continue

        # 普通段落按自然换行切分
        lines = sec.split("\n")
        curr = []
        curr_len = 0
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            line_len = len(line_str) + 1
            if curr_len + line_len <= size:
                curr.append(line_str)
                curr_len += line_len
            else:
                if curr:
                    chunks.append({"heading": heading, "text": prefix + "\n".join(curr)})
                curr = [line_str]
                curr_len = line_len
        if curr:
            chunks.append({"heading": heading, "text": prefix + "\n".join(curr)})
    return chunks


# ---------- 第 2 步：向量化（文字 → 意思坐标） ----------
def get_embed_client() -> OpenAI:
    """拿到向量化 API 客户端。硅基流动 / 百炼都是 OpenAI 兼容接口，换厂商只改 .env。"""
    key = os.environ.get("EMBED_API_KEY", "")
    if not key or "在这里填" in key:
        raise RuntimeError(
            "向量化 API 还没配置：去 cloud.siliconflow.cn 注册（免费），"
            "在 API 密钥页新建一个 key，填进 .env 的 EMBED_API_KEY"
        )
    return OpenAI(api_key=key, base_url=os.environ["EMBED_BASE_URL"])


def embed_texts(texts: list, client=None, batch: int = 32, quiet: bool = False) -> list:
    """把一批文本变成坐标列表。一次请求打包 32 条，省请求次数（也省钱）。"""
    client = client or get_embed_client()
    model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    vectors = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vectors.extend(d.embedding for d in resp.data)
        if not quiet:
            print(f"    已向量化 {min(i + batch, len(texts))}/{len(texts)} 块")
    return vectors


# ---------- 第 3 步：向量库 ----------
def get_collection():
    """打开（没有就创建）向量库。cosine = 用夹角算相似，坐标方向越近意思越像。"""
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def list_note_files() -> list:
    """要建索引的笔记清单：所有 .md，跳过 Obsidian 自己的配置文件夹。"""
    return [md for md in NOTES_DIR.rglob("*.md") if ".obsidian" not in md.parts]


def build_index(force_rebuild: bool = False):
    """
    把知识库切块、向量化、入库。
    加入 MySQL / 数据库增量索引治理：
    通过计算文件 SHA256 哈希比对，仅对有改动的文件进行向量化，未改动文件直接秒级跳过。
    若 force_rebuild=True 则强制对全库重新建索引。
    """
    col = get_collection()
    ec = get_embed_client()
    files = list_note_files()
    print(f"开始建索引：知识库共有 {len(files)} 个文档")
    indexed_count = 0
    skipped_count = 0
    total_chunks = 0

    for md in files:
        rel = md.relative_to(NOTES_DIR)
        rel_str = str(rel).replace("\\", "/")
        current_hash = db_manager.calc_file_hash(md)

        # 增量检测：文件未发生任何修改且非强制重建时跳过
        if not force_rebuild and not db_manager.check_need_reindex(rel_str, current_hash):
            skipped_count += 1
            continue

        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue                               # 读不了的文件跳过，不崩溃
        pieces = chunk_text(text)
        if not pieces:
            continue

        vectors = embed_texts([p["text"] for p in pieces], ec)
        col.upsert(                                # upsert：有则覆盖，无则新增（重复跑不炸）
            ids=[f"{rel}#{i}" for i in range(len(pieces))],
            embeddings=vectors,
            documents=[p["text"] for p in pieces],
            metadatas=[{
                "file": md.name,
                "rel": rel_str,
                "part": i,
                "heading": p["heading"],
            } for i, p in enumerate(pieces)],
        )
        total_chunks += len(pieces)
        indexed_count += 1
        print(f"  [索引更新] {md.name}: 新切块 {len(pieces)} 块")

        # 将文档最新元数据与哈希更新入库
        file_size = md.stat().st_size
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        db_manager.upsert_document(
            doc_id=rel_str,
            title=md.stem,
            category=category,
            file_path=str(md.resolve()),
            file_hash=current_hash,
            chunk_count=len(pieces),
            file_size=file_size,
        )

    print(f"索引构建完成：更新 {indexed_count} 个文档，跳过 {skipped_count} 个未改动文档，本次处理 {total_chunks} 块，数据库当前存有 {col.count()} 块。")
    return total_chunks


# ---------- 第 4 步：粗排（纯向量召回） ----------
def vector_search(query: str, n: int) -> list:
    """按坐标距离召回 n 个候选块。score = 1 - cosine距离，越接近 1 越相关。"""
    col = get_collection()
    if col.count() == 0:
        raise RuntimeError("向量索引还是空的。请先运行：python build_index.py")
    qv = embed_texts([query], quiet=True)[0]
    res = col.query(query_embeddings=[qv], n_results=n)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "file": meta.get("file", ""),
            "rel": meta.get("rel", ""),
            "part": meta.get("part", 0),
            "heading": meta.get("heading", ""),
            "text": doc,
            "score": round(1 - dist, 4),     # 粗排分（cosine 相似度）
        })
    return hits


def keyword_search(query: str, n: int = 5) -> list:
    """轻量关键词精准匹配：补充纯向量检索对专有名词、型号、数字的盲区。"""
    col = get_collection()
    all_data = col.get()
    if not all_data or not all_data["documents"]:
        return []

    # 提取查询中的英文单词、数字和 2 字以上中文词
    tokens = re.findall(r"[A-Za-z0-9_-]{2,}|[\u4e00-\u9fa5]{2,}", query)
    if not tokens:
        return []

    scored = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        doc_lower = doc.lower()
        # 统计命中的关键词数量及频次
        score = sum(doc_lower.count(t.lower()) for t in tokens)
        if score > 0:
            scored.append({
                "file": meta.get("file", ""),
                "rel": meta.get("rel", ""),
                "part": meta.get("part", 0),
                "heading": meta.get("heading", ""),
                "text": doc,
                "score": round(min(score * 0.1, 0.99), 4),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n]


# ---------- 第 5 步：精排（reranker 成对打分） ----------
def rerank_hits(query: str, hits: list, top_n: int) -> list:
    """把粗排的候选块交给 reranker 重新排序。

    用标准库 urllib 直接发 HTTP，不引第三方 SDK——openai 官方包没有 rerank 接口，
    为一个接口多装一个依赖不划算。

    出错就退回粗排顺序：精排是"锦上添花"，不该让整个检索因为它挂掉。
    """
    if not hits:
        return hits
    key = os.environ.get("EMBED_API_KEY", "")
    base = os.environ.get("EMBED_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [h["text"] for h in hits],
        "top_n": min(top_n, len(hits)),
    }
    req = urllib.request.Request(
        base + "/rerank",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        detail = e.read().decode("utf-8", "replace")[:200] if isinstance(e, urllib.error.HTTPError) else str(e)
        print(f"⚠️ 精排失败，退回粗排顺序：{detail}")
        return hits[:top_n]

    out = []
    for r in data.get("results", []):
        h = dict(hits[r["index"]])
        h["vec_score"] = h["score"]                       # 保留粗排分，便于对比调试
        h["score"] = round(r["relevance_score"], 4)       # 最终分换成精排相关性
        out.append(h)
    return out


def retrieve(query: str, top_k: int = None, use_rerank: bool = None) -> list:
    """检索主入口：返回 top_k 个候选块，每个是 dict：
       {file, rel, part, heading, text, score, vec_score?, pool_cosine_max}

    score = 最终排序分（开精排时是 rerank 相关性，否则是 cosine）
    vec_score = 该块的 cosine 相似度（只在开精排时存在，用于对比调试）
    pool_cosine_max = 整个粗排候选池里最高的 cosine 分（门控就看它，见 evidence_score）

    只负责"找"，不负责"排版"——排版是 semantic_search 的事。
    这样 eval 脚本能直接用它打分，不用启动 Agent。
    """
    top_k = TOP_K if top_k is None else top_k
    rerank_on = USE_RERANK if use_rerank is None else use_rerank
    pool = max(RERANK_POOL, top_k) if rerank_on else top_k
    
    # 两路召回：向量粗排 + 关键词精准补充
    vec_hits = vector_search(query, pool)
    kw_hits = keyword_search(query, n=5) if rerank_on else []

    # 去重合并送入候选池
    seen_ids = set()
    hits = []
    for h in kw_hits + vec_hits:
        uid = (h["file"], h["part"])
        if uid not in seen_ids:
            seen_ids.add(uid)
            hits.append(h)

    # 先记下整池的最高 cosine：精排会把池子截断到 top_k，
    # 截断后再算就晚了（高 cosine 的块可能被精排挤出去）。
    pool_max = round(max((h["score"] for h in vec_hits), default=0.0), 4)
    if rerank_on:
        hits = rerank_hits(query, hits, top_k)
    hits = hits[:top_k]
    for h in hits:
        h["pool_cosine_max"] = pool_max
    return hits


# ---------- 第 6 步：证据门控（防幻觉的关键设计） ----------
def evidence_score(hits: list) -> float:
    """门控该看哪个分数？—— 实测结论：看 cosine，不看 rerank。

    rerank 只擅长"排序"（哪一块更可能答得上），它的绝对分数量和
    "库里到底有没有证据"并不对应。20 条 eval 实测：
      · 有正确答案只拿到 0.015（E11“做菜为何不答”），被硬拒答线误杀
      · 有无关问题却拿到 0.502（E19“雅思成绩”撞上笔记里的打分段落）
      两类分数重叠宽达 0.487，拿它门控会同时造成误拒和漏网。
    cosine 虽然排序粗，但"整库最像的一块有多像"这件事它量得稳（重叠只有 0.086）。
    所以：**排序用 rerank，门控用 cosine**，两个分数各干各的活。
    """
    if not hits:
        return 0.0
    if "pool_cosine_max" in hits[0]:
        return hits[0]["pool_cosine_max"]
    return max(h.get("vec_score", h["score"]) for h in hits)


def evidence_gate(score: float) -> str:
    """把 cosine 证据分判成三档：'none'（没证据）/ 'weak'（偏弱）/ 'ok'（充分）。

    为什么不用"单阈值一刀切拒答"？—— 我原先就是这么设计的，eval 直接把它证伪了：
      17 条能答的问题 top1 cosine 最低 0.506，3 条库里没有的问题 top1 最高 0.592。
      单阈值 T 要不误拒就得 T ≤ 0.506，要不漏网就得 T > 0.592 —— 无解。
    三段式把矛盾拆开了：极低分硬拒答兜底，高分放行，
    中间那段（0.45~0.60）把分数和出处一起交给模型，
    让它自己判断"这些材料够不够回答"——判断语义相关性本来就是 LLM 比阈值擅长的事。
    """
    if score < GATE_LOW:
        return "none"
    return "ok" if score >= GATE_SOFT else "weak"


def semantic_search(query: str, top_k: int = None, use_rerank: bool = None) -> str:
    """把 retrieve() 的结果排版成给模型看的文本（Agent 的工具就用这个）。

    注意两个分数分工：排序看 score（开精排时=rerank 相关性），
    门控看 evidence_score（永远=cosine）。理由见 evidence_score 注释。
    """
    rerank_on = USE_RERANK if use_rerank is None else use_rerank
    try:
        hits = retrieve(query, top_k, rerank_on)
    except RuntimeError as e:
        return str(e)
    if not hits:
        return "没有检索到任何内容。"

    ev = evidence_score(hits)      # 门控依据：整池最高 cosine
    gate = evidence_gate(ev)
    kind = "精排相关性" if rerank_on else "语义相似度"

    if gate == "none":
        return (
            f"知识库里没有相关内容（整库最像的段落相似度也只有 {ev:.3f}，"
            f"低于硬拒答线 {GATE_LOW:.2f}）。"
            f"请如实告诉用户「笔记里没有相关内容」，不要凭记忆编造。"
        )

    lines = []
    for h in hits:
        where = h["file"] + (f" › {h['heading']}" if h["heading"] else f" 第{h['part'] + 1}块")
        extra = f"，向量分 {h['vec_score']:.3f}" if "vec_score" in h else ""
        lines.append(f"[出处：{where}｜{kind} {h['score']:.3f}{extra}]\n{h['text'][:300]}")
    body = "\n\n".join(lines)

    if gate == "weak":
        return (
            f"⚠️ 证据偏弱：整库最像的段落相似度 {ev:.3f}，落在 {GATE_LOW:.2f}~{GATE_SOFT:.2f} 的模糊地带，"
            f"下面的材料可能答非所问。\n"
            f"请先判断它们能不能真正支撑答案：能就正常回答并标出处；"
            f"不能就如实说「知识库里没有相关内容」，不要用这些段落硬凑。\n\n"
            f"{body}"
        )
    return body
