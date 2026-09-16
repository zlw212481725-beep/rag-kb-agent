-- ==========================================================
-- rag-kb-agent 企业级元数据治理与问答审计数据库 Schema
-- 适用数据库：MySQL 8.0+ / MariaDB
-- 字符集：utf8mb4 (支持各种符号与 Emoji)
-- ==========================================================

-- 1. 知识库文档元数据表 (用于增量索引检测与文件版本管理)
CREATE TABLE IF NOT EXISTS kb_documents (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键自增 ID',
    doc_id VARCHAR(64) NOT NULL UNIQUE COMMENT '文档唯一标识 (如文件名)',
    title VARCHAR(255) NOT NULL COMMENT '文档标题',
    category VARCHAR(64) DEFAULT 'default' COMMENT '业务分类 (如：求职/技术/日常/规章)',
    file_path VARCHAR(512) NOT NULL COMMENT '文件在本地或服务器的绝对路径',
    file_hash VARCHAR(64) NOT NULL COMMENT '文件内容的 SHA256 哈希值 (用于检测文件是否被修改)',
    chunk_count INT DEFAULT 0 COMMENT '切块数量',
    file_size_bytes INT DEFAULT 0 COMMENT '文件大小 (字节)',
    last_indexed_at DATETIME COMMENT '最后一次建立向量索引的时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '文档入库时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '文档最后更新时间',
    INDEX idx_category (category),
    INDEX idx_file_hash (file_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库源文档元数据表';

-- 2. 知识库切块元数据表 (配合 Chroma 向量库实现结构化元数据联动)
CREATE TABLE IF NOT EXISTS kb_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键自增 ID',
    chunk_id VARCHAR(64) NOT NULL UNIQUE COMMENT '切块唯一 ID (与 Chroma 中一致)',
    doc_id VARCHAR(64) NOT NULL COMMENT '关联所属文档 doc_id',
    title_path VARCHAR(255) NOT NULL COMMENT 'Markdown 标题层级路径 (如: 面试复盘 › 技术问答)',
    chunk_index INT NOT NULL COMMENT '在原文档中的切块序号 (从 0 开始)',
    char_count INT NOT NULL COMMENT '切块字符数',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '切块时间',
    INDEX idx_doc_id (doc_id),
    CONSTRAINT fk_chunk_doc FOREIGN KEY (doc_id) REFERENCES kb_documents(doc_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库切块元数据表';

-- 3. 问答全链路审计与运维监控表 (替代原零散 JSONL 文件，支持 SQL 聚合分析)
CREATE TABLE IF NOT EXISTS qa_audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键自增 ID',
    session_id VARCHAR(64) COMMENT '会话 ID',
    question TEXT NOT NULL COMMENT '用户输入的提问内容',
    matched_chunks JSON COMMENT '检索出的 Top 证据池元数据 (JSON 数组格式)',
    max_cosine_score FLOAT COMMENT '粗排检索到的最高余弦相似度分数 (Cosine)',
    max_rerank_score FLOAT COMMENT '精排重排最高分数 (Rerank)',
    gate_decision ENUM('ACCEPT', 'WEAK', 'REJECT') NOT NULL COMMENT '三段式门控决策结果',
    has_bad_citation BOOLEAN DEFAULT FALSE COMMENT '是否存在编造/虚构引用出处',
    bad_citations_detail JSON COMMENT '编造的出处明细列表',
    response_text TEXT COMMENT '大模型最终输出的回答正文',
    latency_ms INT COMMENT '全链路耗时 (毫秒)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '提问发生时间',
    INDEX idx_gate_decision (gate_decision),
    INDEX idx_has_bad_citation (has_bad_citation),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG 问答审计与全链路监控表';
