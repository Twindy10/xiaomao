-- =============================================================================
-- 小贸同学（xiaomao）MySQL 8.4 建表脚本
-- -----------------------------------------------------------------------------
-- 使用方式：先选中目标库（例如 USE xiaomao;），再整体执行本文件。
-- 本脚本不包含 CREATE DATABASE / USE，可重复执行而不报错。
-- 约定：
--   * 引擎 InnoDB，字符集 utf8mb4，排序规则 utf8mb4_0900_ai_ci
--   * 不建外键约束，关联关系由应用层保证，关联字段一律建普通索引
--   * 枚举类字段用 VARCHAR + COMMENT 说明取值，不使用 ENUM
--   * 所有业务查询都带 user_id，卖家之间严格隔离
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. plan：套餐
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `plan` (
    `id`               INT UNSIGNED NOT NULL AUTO_INCREMENT      COMMENT '套餐 ID',
    `code`             VARCHAR(32)  NOT NULL                      COMMENT '套餐代码，取值 free / pro',
    `name`             VARCHAR(64)  NOT NULL                      COMMENT '套餐显示名称',
    `daily_quota`      INT          NOT NULL DEFAULT 0            COMMENT '每日提问次数上限',
    `max_history_days` INT          NOT NULL DEFAULT 0            COMMENT '对话历史保留天数',
    `created_at`       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                     ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_plan_code` (`code`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '套餐表：定义各套餐的每日提问额度和历史保留天数';


-- -----------------------------------------------------------------------------
-- 2. user：用户（表名是 MySQL 关键字，统一加反引号）
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `user` (
    `id`            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT       COMMENT '用户 ID',
    `phone`         VARCHAR(20)     NOT NULL                      COMMENT '手机号，登录账号',
    `password_hash` VARCHAR(100)    NOT NULL                      COMMENT '密码哈希，bcrypt 结果',
    `nickname`      VARCHAR(64)     DEFAULT NULL                  COMMENT '昵称，可空',
    `plan_id`       INT UNSIGNED    NOT NULL DEFAULT 1            COMMENT '套餐 ID，对应 plan.id',
    `status`        TINYINT         NOT NULL DEFAULT 1            COMMENT '账号状态：1=正常，0=封禁',
    `last_login_at` DATETIME(3)     DEFAULT NULL                  COMMENT '最近一次登录时间，可空',
    `created_at`    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                     ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_user_phone` (`phone`),
    KEY `idx_user_plan_id` (`plan_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '用户表：中小跨境卖家的账号';


-- -----------------------------------------------------------------------------
-- 3. seller_profile：卖家画像，和 user 一对一
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `seller_profile` (
    `user_id`    BIGINT UNSIGNED NOT NULL                          COMMENT '用户 ID，对应 user.id，一对一',
    `platforms`  JSON            DEFAULT NULL                      COMMENT '主营平台，JSON 数组，例如 ["amazon","temu"]',
    `sites`      JSON            DEFAULT NULL                      COMMENT '主营站点，JSON 数组，例如 ["US","DE"]',
    `categories` JSON            DEFAULT NULL                      COMMENT '主营品类，JSON 数组，例如 ["3C","家居"]',
    `stage`      VARCHAR(16)     DEFAULT NULL                      COMMENT '经营阶段，取值 newbie / growing，可空',
    `created_at` DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at` DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                  ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`user_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '卖家画像表：用于自动补全检索过滤条件；选填、常改，故与 user 分表';


-- -----------------------------------------------------------------------------
-- 4. conversation：会话
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `conversation` (
    `id`         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT          COMMENT '会话 ID',
    `user_id`    BIGINT UNSIGNED NOT NULL                         COMMENT '所属用户 ID，对应 user.id',
    `title`      VARCHAR(128)    NOT NULL DEFAULT ''              COMMENT '会话标题，通常取首问的截断',
    `is_deleted` TINYINT         NOT NULL DEFAULT 0               COMMENT '是否软删除：0=正常，1=已删除',
    `created_at` DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at` DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                  ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_conversation_user_updated` (`user_id`, `updated_at`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '会话表：一个用户的一个对话线程';


-- -----------------------------------------------------------------------------
-- 5. message：消息
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `message` (
    `id`              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT      COMMENT '消息 ID',
    `conversation_id` BIGINT UNSIGNED NOT NULL                     COMMENT '所属会话 ID，对应 conversation.id',
    `user_id`         BIGINT UNSIGNED NOT NULL                     COMMENT '所属用户 ID，冗余字段，方便权限校验',
    `role`            VARCHAR(16)     NOT NULL                     COMMENT '角色，取值 user / assistant',
    `content`         MEDIUMTEXT      NOT NULL                     COMMENT '消息正文',
    `citations`       JSON            DEFAULT NULL                 COMMENT '引用列表，仅 assistant 消息有；元素含 chunk_id、collection、source_url、effective_date、rank',
    `created_at`      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                       ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_message_conversation` (`conversation_id`, `id`),
    KEY `idx_message_user` (`user_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '消息表：会话里的每一条提问和回答';


-- -----------------------------------------------------------------------------
-- 6. qa_trace：问答链路日志，每条 assistant 消息一行
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `qa_trace` (
    `id`                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT   COMMENT '链路日志 ID',
    `message_id`         BIGINT UNSIGNED NOT NULL                  COMMENT '对应的 assistant 消息 ID，对应 message.id',
    `user_id`            BIGINT UNSIGNED NOT NULL                  COMMENT '所属用户 ID，对应 user.id',
    `query_raw`          TEXT            NOT NULL                  COMMENT '用户原始提问',
    `query_rewritten`    TEXT            DEFAULT NULL              COMMENT '改写或补全过滤条件后、能独立看懂的问题',
    `intent`             VARCHAR(32)     DEFAULT NULL              COMMENT '最终意图，例如 platform_policy / compliance / logistics / tax / ops / account / chitchat / out_of_scope',
    `strategy`           VARCHAR(16)     DEFAULT NULL              COMMENT '检索策略，取值 direct / subquery / hyde',
    `decisions`          JSON            DEFAULT NULL              COMMENT '决策模型每个问题的概率分布、置信度、所用后端、是否降级',
    `cache_hit`          TINYINT         NOT NULL DEFAULT 0        COMMENT '是否命中语义缓存：0=否，1=是',
    `chunk_ids`          JSON            DEFAULT NULL              COMMENT '本次召回的片段 ID 列表',
    `rerank_scores`      JSON            DEFAULT NULL              COMMENT '重排得分列表',
    `evidence_sufficient` TINYINT        DEFAULT NULL              COMMENT '证据是否充分：1=充分，0=不充分，NULL=未判断',
    `backtracked`        TINYINT         NOT NULL DEFAULT 0        COMMENT '是否走了 Backtrack：0=否，1=是',
    `llm_model`          VARCHAR(64)     DEFAULT NULL              COMMENT '生成模型名称，可空',
    `prompt_tokens`      INT             NOT NULL DEFAULT 0        COMMENT '输入 token 数',
    `completion_tokens`  INT             NOT NULL DEFAULT 0        COMMENT '输出 token 数',
    `latency_ms`         JSON            DEFAULT NULL              COMMENT '各阶段耗时，例如 {"cache":3,"decide":41,"retrieve":85,"rerank":120,"llm":2300}',
    `total_latency_ms`   INT             NOT NULL DEFAULT 0        COMMENT '端到端总耗时（毫秒），便于直接排序统计 P95',
    `created_at`         DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`         DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                          ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_qa_trace_message` (`message_id`),
    KEY `idx_qa_trace_user` (`user_id`),
    KEY `idx_qa_trace_created` (`created_at`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '问答链路日志表：每次回答是怎么生成的，用于复盘、评估与成本统计';

-- 说明：本表增长最快且 JSON 字段大，数据量大后可按月做 RANGE 分区，
--       MVP 阶段不建分区，先靠 idx_qa_trace_created 做按时间归档。


-- -----------------------------------------------------------------------------
-- 7. feedback：反馈
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `feedback` (
    `id`         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT           COMMENT '反馈 ID',
    `message_id` BIGINT UNSIGNED NOT NULL                          COMMENT '被反馈的 assistant 消息 ID，对应 message.id',
    `user_id`    BIGINT UNSIGNED NOT NULL                          COMMENT '反馈人用户 ID，对应 user.id',
    `rating`     TINYINT         NOT NULL                          COMMENT '评价：1=赞，-1=踩',
    `reason`     VARCHAR(32)     DEFAULT NULL                      COMMENT '点踩原因，取值 wrong / outdated / off_topic / other，可空',
    `comment`    VARCHAR(500)    DEFAULT NULL                      COMMENT '用户补充说明，可空',
    `status`     VARCHAR(16)     NOT NULL DEFAULT 'pending'        COMMENT '处理状态，取值 pending / kb_added / eval_added / ignored',
    `created_at` DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at` DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                  ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_feedback_message_user` (`message_id`, `user_id`),
    KEY `idx_feedback_status` (`status`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '反馈表：串联“点踩 → 复核 → 补知识库或进评估集”的闭环';


-- -----------------------------------------------------------------------------
-- 8. usage_daily：每日用量
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `usage_daily` (
    `user_id`     BIGINT UNSIGNED NOT NULL                         COMMENT '用户 ID，对应 user.id',
    `day`         DATE            NOT NULL                         COMMENT '统计日期',
    `query_count` INT             NOT NULL DEFAULT 0               COMMENT '当日提问次数',
    `tokens`      INT             NOT NULL DEFAULT 0               COMMENT '当日消耗 token 数',
    `created_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                   ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`user_id`, `day`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '每日用量表：Redis 实时计数定时写回，用于配额对账与用量统计';


-- -----------------------------------------------------------------------------
-- 9. kb_document：知识库文档登记
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `kb_document` (
    `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT       COMMENT '文档 ID，对应 Milvus 里的 doc_id',
    `collection`     VARCHAR(64)     NOT NULL                      COMMENT '所属知识库集合，例如 xm_platform_policy / xm_compliance / xm_faq',
    `source_url`     VARCHAR(1024)   NOT NULL                      COMMENT '原文地址',
    `title`          VARCHAR(512)    NOT NULL DEFAULT ''           COMMENT '文档标题',
    `platform`       VARCHAR(32)     DEFAULT NULL                  COMMENT '适用平台，可空',
    `site`           VARCHAR(16)     DEFAULT NULL                  COMMENT '适用站点，可空',
    `doc_type`       VARCHAR(32)     NOT NULL DEFAULT ''           COMMENT '文档类型，例如 policy / guide / regulation / faq',
    `lang`           VARCHAR(8)      NOT NULL DEFAULT ''           COMMENT '文档语言，例如 zh / en / de / ja',
    `effective_date` DATE            DEFAULT NULL                  COMMENT '生效日期，可空',
    `expire_date`    DATE            DEFAULT NULL                  COMMENT '失效日期，空表示长期有效',
    `content_hash`   CHAR(64)        NOT NULL                      COMMENT '正文 SHA-256 十六进制，用于去重与增量更新；同一集合内唯一',
    `version`        INT             NOT NULL DEFAULT 1            COMMENT '文档版本号，每次内容变化递增',
    `status`         VARCHAR(16)     NOT NULL DEFAULT 'active'     COMMENT '文档状态，取值 active / expired / superseded',
    `chunk_count`    INT             NOT NULL DEFAULT 0            COMMENT '入库片段数，用于与 Milvus 条数对账',
    `ingested_at`    DATETIME(3)     DEFAULT NULL                  COMMENT '最近一次入库时间，可空',
    `created_at`     DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`     DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                      ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_kb_document_collection_hash` (`collection`, `content_hash`),
    KEY `idx_kb_document_collection_status` (`collection`, `status`),
    KEY `idx_kb_document_source_url` (`source_url`(255))
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '知识库文档登记表：文档级的版本与状态放在这里，Milvus 只负责片段检索';


-- -----------------------------------------------------------------------------
-- 种子数据：套餐
-- -----------------------------------------------------------------------------
INSERT INTO `plan` (`id`, `code`, `name`, `daily_quota`, `max_history_days`)
VALUES (1, 'free', '免费版', 20, 30),
       (2, 'pro', '专业版', 200, 365) AS new
ON DUPLICATE KEY UPDATE `name`             = new.`name`,
                        `daily_quota`      = new.`daily_quota`,
                        `max_history_days` = new.`max_history_days`;
