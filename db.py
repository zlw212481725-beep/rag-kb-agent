"""
db.py —— rag-kb-agent 数据库管理模块
支持 MySQL 8.0+ 与 本地 SQLite 双模式平滑自适应。
若未配置 MySQL 连接信息，自动降级为 SQLite 本地存储 (output/kb_meta.db)。
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# 尝试导入 pymysql
try:
    import pymysql
    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False

# 配置路径
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SQLITE_DB_PATH = OUTPUT_DIR / "kb_meta.db"

# 环境变量读取
DB_TYPE = os.getenv("DB_TYPE", "auto").lower()  # auto, mysql, sqlite
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "rag_kb")


class DatabaseManager:
    """自适应数据库管理器 (MySQL / SQLite)"""

    def __init__(self):
        self.is_mysql = False
        self._init_connection()
        self._create_tables()

    def _init_connection(self):
        """探测并建立连接"""
        if DB_TYPE in ["mysql", "auto"] and HAS_PYMYSQL and MYSQL_HOST and MYSQL_USER:
            try:
                conn = pymysql.connect(
                    host=MYSQL_HOST,
                    port=MYSQL_PORT,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    database=MYSQL_DB,
                    charset="utf8mb4",
                    autocommit=True,
                    connect_timeout=3,
                )
                conn.close()
                self.is_mysql = True
                print(f"[DB] 成功连接至 MySQL 数据库: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
                return
            except Exception as e:
                print(f"[DB] 连接 MySQL 失败 ({e})，自动平滑降级为本地 SQLite: {SQLITE_DB_PATH}")

        self.is_mysql = False

    def get_connection(self):
        """获取数据库连接对象"""
        if self.is_mysql:
            return pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                charset="utf8mb4",
                autocommit=True,
                cursorclass=pymysql.cursors.DictCursor,
            )
        else:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            conn.row_factory = sqlite3.Row
            return conn

    def _create_tables(self):
        """初始化建表 (兼顾 MySQL 与 SQLite 语法)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if self.is_mysql:
                # 遵循 schema.sql 的标准 MySQL 语法
                schema_path = Path(__file__).resolve().parent / "schema.sql"
                if schema_path.exists():
                    sql_text = schema_path.read_text(encoding="utf-8")
                    for statement in sql_text.split(";"):
                        stmt = statement.strip()
                        if stmt:
                            cursor.execute(stmt)
            else:
                # SQLite 专用语法 (自适应类型)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT DEFAULT 'default',
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    file_size_bytes INTEGER DEFAULT 0,
                    last_indexed_at DATETIME,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE NOT NULL,
                    doc_id TEXT NOT NULL,
                    title_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES kb_documents(doc_id) ON DELETE CASCADE
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS qa_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    matched_chunks TEXT,
                    max_cosine_score REAL,
                    max_rerank_score REAL,
                    gate_decision TEXT NOT NULL,
                    has_bad_citation INTEGER DEFAULT 0,
                    bad_citations_detail TEXT,
                    response_text TEXT,
                    latency_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                conn.commit()
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def calc_file_hash(file_path: Path) -> str:
        """计算文件的 SHA256 哈希值"""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def check_need_reindex(self, doc_id: str, current_hash: str) -> bool:
        """
        核心增量索引检测：
        对比数据库中保存的 hash 值，如果 hash 一致则无需重新向量化 (返回 False)；
        若不存在或 hash 已改变则需要重新建索引 (返回 True)。
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            sql = "SELECT file_hash FROM kb_documents WHERE doc_id = %s" if self.is_mysql else "SELECT file_hash FROM kb_documents WHERE doc_id = ?"
            cursor.execute(sql, (doc_id,))
            row = cursor.fetchone()
            if not row:
                return True  # 新文档，需要索引
            stored_hash = row["file_hash"] if isinstance(row, dict) or self.is_mysql else row[0]
            return stored_hash != current_hash
        finally:
            cursor.close()
            conn.close()

    def upsert_document(self, doc_id: str, title: str, category: str, file_path: str, file_hash: str, chunk_count: int, file_size: int):
        """新增或更新文档元数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        try:
            if self.is_mysql:
                sql = """
                INSERT INTO kb_documents (doc_id, title, category, file_path, file_hash, chunk_count, file_size_bytes, last_indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    title=VALUES(title), category=VALUES(category), file_path=VALUES(file_path),
                    file_hash=VALUES(file_hash), chunk_count=VALUES(chunk_count),
                    file_size_bytes=VALUES(file_size_bytes), last_indexed_at=VALUES(last_indexed_at),
                    updated_at=NOW();
                """
                cursor.execute(sql, (doc_id, title, category, file_path, file_hash, chunk_count, file_size, now))
            else:
                sql = """
                INSERT INTO kb_documents (doc_id, title, category, file_path, file_hash, chunk_count, file_size_bytes, last_indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title, category=excluded.category, file_path=excluded.file_path,
                    file_hash=excluded.file_hash, chunk_count=excluded.chunk_count,
                    file_size_bytes=excluded.file_size_bytes, last_indexed_at=excluded.last_indexed_at,
                    updated_at=CURRENT_TIMESTAMP;
                """
                cursor.execute(sql, (doc_id, title, category, file_path, file_hash, chunk_count, file_size, now.isoformat()))
                conn.commit()
        finally:
            cursor.close()
            conn.close()

    def log_qa_audit(
        self,
        question: str,
        matched_chunks: List[Dict[str, Any]],
        max_cosine: float,
        max_rerank: float,
        gate_decision: str,
        has_bad_citation: bool,
        bad_citations_detail: List[str],
        response_text: str,
        latency_ms: int = 0,
        session_id: str = "default",
    ):
        """记录问答审计日志 (支持企业级 SQL 聚合分析与幻觉监控)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        chunks_json = json.dumps(matched_chunks, ensure_ascii=False)
        bad_cit_json = json.dumps(bad_citations_detail, ensure_ascii=False)
        try:
            if self.is_mysql:
                sql = """
                INSERT INTO qa_audit_logs (
                    session_id, question, matched_chunks, max_cosine_score, max_rerank_score,
                    gate_decision, has_bad_citation, bad_citations_detail, response_text, latency_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                cursor.execute(sql, (
                    session_id, question, chunks_json, max_cosine, max_rerank,
                    gate_decision, has_bad_citation, bad_cit_json, response_text, latency_ms
                ))
            else:
                sql = """
                INSERT INTO qa_audit_logs (
                    session_id, question, matched_chunks, max_cosine_score, max_rerank_score,
                    gate_decision, has_bad_citation, bad_citations_detail, response_text, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                cursor.execute(sql, (
                    session_id, question, chunks_json, max_cosine, max_rerank,
                    gate_decision, 1 if has_bad_citation else 0, bad_cit_json, response_text, latency_ms
                ))
                conn.commit()
        finally:
            cursor.close()
            conn.close()


# 单例实例供全局使用
db_manager = DatabaseManager()

if __name__ == "__main__":
    print("=== 测试 DatabaseManager 运行状态 ===")
    print(f"数据库引擎: {'MySQL' if db_manager.is_mysql else 'SQLite (本地保底)'}")
    
    # 测试写入一条审计测试日志
    db_manager.log_qa_audit(
        question="测试问题：六级补考时间是什么时候？",
        matched_chunks=[{"title": "今日任务.md", "snippet": "2026-12 六级补考"}],
        max_cosine=0.506,
        max_rerank=0.853,
        gate_decision="WEAK",
        has_bad_citation=False,
        bad_citations_detail=[],
        response_text="根据知识库记录，六级补考预计于 2026 年 12 月进行。",
        latency_ms=120,
    )
    print("测试审计日志写入成功！已完成数据持久化闭环验证。")
