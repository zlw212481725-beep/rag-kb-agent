"""
build_index.py —— 建向量索引
用法：python build_index.py
跑一次即可；笔记大改后重跑也安全（upsert 幂等，不会产生重复块）。
"""
from dotenv import load_dotenv

load_dotenv()
from rag_lib import build_index, list_note_files, NOTES_DIR  # noqa: E402

if __name__ == "__main__":
    if not NOTES_DIR.exists():
        raise SystemExit(
            f"找不到知识库目录：{NOTES_DIR}\n"
            f"请在 .env 里把 NOTES_DIR 改成你的 Obsidian 库根目录（参考 .env.example）"
        )
    print(f"知识库目录：{NOTES_DIR}")
    print(f"待处理文件：{len(list_note_files())} 个 .md")
    build_index()
