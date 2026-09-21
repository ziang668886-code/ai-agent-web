# AI 城市生活助手部署与迁移指南

本文档用于在 Ubuntu 24.04 LTS 上重新部署或迁移 AI 城市生活助手。
示例中的 `<APP_DIR>`、`<APP_USER>`、`<ENV_FILE>` 等均为占位符，执行前必须替换为新服务器的实际值。

本文档不包含生产密码、API Key、聊天数据或固定服务器 IP。生产备份和真实 `.env` 不得提交到 GitHub。

## 1. 部署架构

```text
Internet
  -> HTTPS 443
  -> Nginx
  -> 127.0.0.1:8501
  -> Streamlit (ai-agent-web.py)
       -> MySQL: visitors / conversations / messages
       -> RAG: data/knowledge_bases/<visitor_id>/index.npz
       -> Volcengine Ark API
       -> QWeather API
       -> AMap Web Service API
```

对外仅开放必要端口：

- `80/tcp`：HTTP，仅用于跳转 HTTPS 和证书验证。
- `443/tcp`：HTTPS 正式访问。
- `22/tcp`：仅按运维需要开放，并限制来源。
- `3306/tcp`：不得对公网开放。
- `8501/tcp`：不得对公网开放，Streamlit 仅监听 `127.0.0.1`。

目标域名：

- `ziang-ai.online`
- `www.ziang-ai.online`

## 2. 新服务器基础准备

### 2.1 系统更新与基础软件

先确认当前处于新服务器，并使用具有 sudo 权限的运维账户：

```bash
sudo apt update
sudo apt upgrade
sudo apt install git nginx mysql-server mysql-client rsync ca-certificates certbot python3-certbot-nginx
```

根据云平台安全组和 Ubuntu 防火墙策略开放 `22`、`80`、`443`，不要开放 `3306` 和 `8501`。

### 2.2 Python 3.11

本项目经过验证的运行版本为 Python 3.11。Ubuntu 24.04 的默认 Python 版本不应被假定为 3.11。

部署前先确认：

```bash
python3.11 --version
```

如果命令不存在，应根据服务器实际可用的软件源、组织镜像、容器方案或可信的 Python 安装方式提供 Python 3.11。不要在未确认软件源的情况下照搬特定第三方仓库命令。

同时确认所选 Python 3.11 安装包含：

- `venv`
- `pip`
- 必要时的 Python 开发头文件和编译工具

不要替换或删除 Ubuntu 系统依赖的默认 Python。

## 3. 获取项目

### 3.1 创建专用运行账户和目录

应用不应以 root 身份运行。先按服务器规范创建专用账户 `<APP_USER>` 和应用目录 `<APP_DIR>`，并确保该账户拥有应用目录。

### 3.2 克隆代码并固定版本

```bash
sudo -u <APP_USER> git clone <REPOSITORY_URL> <APP_DIR>
cd <APP_DIR>
git fetch --all --tags
git checkout <VERIFIED_COMMIT_OR_TAG>
```

生产部署应固定到已经测试通过的 commit 或 tag。不要永久依赖 `main` 当前最新提交，也不要把本文档编写时的 commit 当作永远唯一版本。

确认版本：

```bash
git status --short
git rev-parse HEAD
```

### 3.3 创建虚拟环境并安装依赖

```bash
cd <APP_DIR>
python3.11 -m venv .venv
<APP_DIR>/.venv/bin/python -m pip install --upgrade pip
<APP_DIR>/.venv/bin/python -m pip install -r requirements.txt
```

安装后进行基础导入检查。不要在生产服务器上运行会调用真实外部 API 的实验测试脚本。

项目入口是：

```text
ai-agent-web.py
```

`main.py` 不是 Streamlit 生产入口。

## 4. 环境变量

仓库中的 `.env.example` 只提供变量名称和非敏感默认值。创建生产环境文件时，不要把真实值写入 Git。

应用使用以下变量：

| 变量 | 用途 |
| --- | --- |
| `ARK_API_KEY` | 火山方舟聊天与 Embedding API |
| `DB_HOST` | MySQL 地址，同机部署通常为 `127.0.0.1` |
| `DB_PORT` | MySQL 端口，默认 `3306` |
| `DB_NAME` | 数据库名，当前为 `ai_agent` |
| `DB_USER` | 项目专用 MySQL 用户 |
| `DB_PASSWORD` | 项目专用 MySQL 用户密码 |
| `QWEATHER_API_KEY` | QWeather API 凭证 |
| `QWEATHER_API_HOST` | QWeather 分配或要求使用的 API Host |
| `AMAP_API_KEY` | 高德 Web Service API 凭证 |

建议把生产环境文件放在仅运维账户和应用账户可读取的位置，例如 `<ENV_FILE>`，并限制权限：

```bash
sudo chown <APP_USER>:<APP_GROUP> <ENV_FILE>
sudo chmod 600 <ENV_FILE>
```

可以从模板开始填写，但不得覆盖已有生产配置：

```bash
cp <APP_DIR>/.env.example <ENV_FILE>
```

填写时所有 Secret 应通过安全渠道取得。不要在终端历史、工单、聊天消息或 Git 中粘贴真实值。

## 5. MySQL

### 5.1 网络与账户原则

- MySQL 建议只监听本机或私有网络。
- 云安全组和主机防火墙不得向公网开放 `3306`。
- 为应用创建专用低权限用户。
- 运行期通常只需要对 `ai_agent` 的 `SELECT`、`INSERT`、`UPDATE`、`DELETE` 权限。
- 数据库和连接字符集使用 `utf8mb4`。

### 5.2 场景 A：全新空环境

仅当不需要恢复任何生产数据时，使用仓库中的 `schema.sql` 初始化：

```bash
cd <APP_DIR>
mysql -u <DB_ADMIN_USER> -p < schema.sql
```

该命令会提示输入数据库管理员密码，不要把密码写在命令行中。

之后创建项目专用账户并设置最小权限。账户名、密码和 Host 应按实际环境确定，不要复制示例占位符作为真实凭证。

### 5.3 场景 B：服务器迁移

迁移生产环境时，应优先恢复旧服务器生成的完整 `mysqldump`，不要先用 `schema.sql` 建空表后再自行拼接数据。

在迁移或维护窗口执行恢复：

```bash
mysql -u <DB_ADMIN_USER> -p < <MYSQL_DUMP_FILE>
```

恢复后至少验证：

```bash
mysql -u <DB_ADMIN_USER> -p -D ai_agent -e "SHOW TABLES;"
mysql -u <DB_ADMIN_USER> -p -D ai_agent -e "SELECT COUNT(*) FROM visitors; SELECT COUNT(*) FROM conversations; SELECT COUNT(*) FROM messages;"
```

必须存在：

- `visitors`
- `conversations`
- `messages`

同时检查主键、外键、索引、字符集和 `messages.message_id` 的自增状态。

## 6. RAG 知识库迁移

知识库不在 MySQL 中，必须单独迁移：

```text
<APP_DIR>/data/knowledge_bases/<visitor_id>/index.npz
```

注意事项：

- `index.npz` 包含 Chunk、Embedding 和文档元数据，是生产数据。
- Git clone 不会恢复 `data/knowledge_bases/`，因为该目录被 `.gitignore` 忽略。
- `.lock` 文件是运行时锁文件，不需要迁移。
- 复制期间应避免旧应用同时写入索引。
- 迁移后应用账户必须拥有目录和文件的读写权限。

初次同步示例：

```bash
rsync -a --exclude='*.lock' <OLD_KB_DIR>/ <APP_DIR>/data/knowledge_bases/
sudo chown -R <APP_USER>:<APP_GROUP> <APP_DIR>/data/knowledge_bases
```

最终同步应在短维护窗口执行，并再次排除 `.lock` 文件。

## 7. systemd

以下为模板，不应直接照搬用户名或路径：

```ini
[Unit]
Description=AI City Life Assistant Streamlit Service
Wants=network-online.target
After=network-online.target mysql.service

[Service]
Type=simple
User=<APP_USER>
Group=<APP_GROUP>
WorkingDirectory=<APP_DIR>
EnvironmentFile=<ENV_FILE>
ExecStart=<APP_DIR>/.venv/bin/streamlit run <APP_DIR>/ai-agent-web.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

如果 MySQL 不在同一台服务器，应根据实际架构调整 `After`，不应保留不存在的本机 MySQL 服务依赖。

把 unit 保存到系统规定的位置后，验证配置并启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable <SERVICE_NAME>
sudo systemctl start <SERVICE_NAME>
sudo systemctl status <SERVICE_NAME>
```

检查日志时不要把包含敏感配置的输出复制到公开渠道：

```bash
sudo journalctl -u <SERVICE_NAME> --since "10 minutes ago"
```

确认 Streamlit 只监听本机：

```bash
ss -lntp | grep 8501
```

预期监听地址为 `127.0.0.1:8501`，而不是 `0.0.0.0:8501`。

## 8. Nginx

以下模板把两个域名反向代理到本机 Streamlit。此模板本身不代表 HTTPS 已配置完成。

`map` 通常放在 Nginx `http` 上下文中：

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
```

站点模板：

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name ziang-ai.online www.ziang-ai.online;

    client_max_body_size 15M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_buffering off;
    }
}
```

启用前检查：

```bash
sudo nginx -t
```

`client_max_body_size 15M` 略高于应用的单文件 10MB 限制，用于容纳 multipart 请求开销。不要通过 Nginx 直接暴露 Streamlit 8501。

## 9. HTTPS

可以使用 Certbot、腾讯云 SSL 证书或组织统一证书方案。证书签发和 Nginx TLS 配置必须根据 DNS、证书来源和服务器环境实际完成。

使用 Certbot 时，可在 DNS 已指向新服务器且 HTTP 验证可达后执行相应的 Nginx 集成流程。执行前先确认 Nginx 配置测试通过。

上线要求：

- `https://ziang-ai.online` 可用。
- `https://www.ziang-ai.online` 可用或明确重定向到主域名。
- HTTP 最终重定向到 HTTPS。
- TLS 证书链有效且自动续期方案可验证。
- 不应声称 HTTPS 完成，直到浏览器和证书检查均通过。

浏览器 `navigator.geolocation` 在正式环境要求 HTTPS secure context；localhost 只是浏览器开发环境的例外。若响应头配置了 `Permissions-Policy`，应允许本站使用定位，例如按安全策略设置 `geolocation=(self)`，不能误将其完全禁用。

## 10. GPS / Geolocation 验收

分别使用桌面和手机浏览器检查：

- 权限已经允许：`permission=granted`。
- 尚未决定权限：`permission=prompt`。
- 明确拒绝权限：`permission=denied`，页面应友好降级。
- “我现在在哪”能返回粗粒度位置。
- “我这里天气怎么样”使用当前授权位置。
- “我附近有什么咖啡店”使用附近 POI 检索。
- “我这里天气怎么样？如果天气不错，帮我找附近咖啡店。”能完成天气到 POI 的有界多工具流程。
- 明确城市应优先于 GPS，例如“北京天气怎么样”。
- 页面刷新、切换页面和 Streamlit rerun 不应无限重复请求定位。
- 拒绝定位后，不应猜测用户位置。

更换域名或子域名后，浏览器可能要求用户重新授权定位。

## 11. 正式迁移流程

以下流程用于降低停机和数据遗漏风险：

1. 保持旧服务器继续运行，不立即修改 DNS。
2. 在新服务器安装系统软件、Python 3.11 和项目依赖。
3. 检出经过验证的项目 commit。
4. 创建生产环境变量文件和数据库专用账户。
5. 从旧服务器生成一次初始 MySQL dump。
6. 初次同步 `data/knowledge_bases/`，排除 `.lock`。
7. 在新服务器恢复初始数据库和知识库。
8. 仅通过本机或受限测试入口验证应用、数据库和外部 API。
9. 降低 DNS TTL，并等待旧 TTL 生效。
10. 进入短维护窗口，暂停旧站的新写入。
11. 生成最终 MySQL dump。
12. 最终同步 `data/knowledge_bases/`。
13. 对 dump 和知识库归档执行 SHA-256 校验。
14. 在新服务器执行最终恢复和文件权限设置。
15. 验证表数量、历史会话、知识库、天气、POI 和 GPS。
16. 修改 `ziang-ai.online` 与 `www.ziang-ai.online` 的 DNS A 记录。
17. 从公网验证域名、HTTPS、WebSocket、手机端和桌面端。
18. 保留旧服务器作为短期回滚环境，但避免新旧两端同时接受写入。
19. 观察稳定后，再按正式变更流程停止或释放旧服务器。

切换期间必须确定唯一写入端，避免 MySQL 数据和知识库索引在两台服务器产生分叉。

## 12. 备份与恢复

### 12.1 MySQL 备份

在旧服务器上生成一致性备份：

```bash
mysqldump -u <DB_BACKUP_USER> -p --single-transaction --routines --events --triggers --databases ai_agent > <MYSQL_DUMP_FILE>
sha256sum <MYSQL_DUMP_FILE> > <MYSQL_DUMP_FILE>.sha256
```

不要把密码直接写在命令参数中。

### 12.2 知识库备份

在应用没有写入知识库的维护窗口中创建归档：

```bash
tar --exclude='*.lock' -czf <KB_BACKUP_FILE> -C <APP_DIR>/data knowledge_bases
sha256sum <KB_BACKUP_FILE> > <KB_BACKUP_FILE>.sha256
```

传输到新服务器后验证：

```bash
sha256sum -c <MYSQL_DUMP_FILE>.sha256
sha256sum -c <KB_BACKUP_FILE>.sha256
```

### 12.3 恢复原则

- 数据库恢复属于迁移/维护窗口操作。
- 恢复前确认目标服务器、目标数据库和备份文件。
- 不要把生产 dump、知识库归档或真实 `.env` 上传到 GitHub。
- 备份文件应加密存储，并限制访问权限。
- 定期进行恢复演练，不能只确认备份命令成功。

## 13. 安全检查

- SSH `22` 端口仅按需开放，优先使用密钥认证并限制来源。
- `80/443` 对公网提供 Web 服务。
- MySQL `3306` 不对公网开放。
- Streamlit `8501` 不对公网开放。
- 应用、数据库和备份使用最小权限账户。
- 应用不以 root 运行。
- `.env`、数据库 dump、知识库备份和证书私钥不进入 Git。
- 所有 API Key 和数据库密码只通过安全渠道配置。
- 正式访问强制 HTTPS。
- 检查 Nginx 和 systemd 日志中没有 Secret。
- 验证匿名 visitor Cookie 在 HTTPS 下的 `Secure` 属性；当前 visitor ID 承担数据隔离身份作用。
- 保持原域名有利于现有浏览器继续携带 visitor Cookie。更换域名会导致 Cookie 无法自动迁移。
- 定期更新系统安全补丁，但依赖升级应先在测试环境验证。

## 14. 上线验收清单

### 系统与网络

- [ ] `git rev-parse HEAD` 等于计划部署的稳定 commit。
- [ ] systemd 服务使用专用非 root 用户。
- [ ] Streamlit 仅监听 `127.0.0.1:8501`。
- [ ] MySQL 3306 未向公网开放。
- [ ] Nginx 80/443 工作正常。
- [ ] HTTPS 证书有效，HTTP 自动跳转 HTTPS。
- [ ] systemd 重启后应用能够恢复运行。

### 核心页面与会话

- [ ] 首页正常显示并可直接提问。
- [ ] 新建聊天正常。
- [ ] 历史会话列表、切换、重命名和删除正常。
- [ ] MySQL 能保存用户消息和最终 assistant 回答。
- [ ] 页面刷新后可恢复最近会话。
- [ ] visitor Cookie 能保持匿名身份和数据隔离。

### 知识库

- [ ] PDF、DOCX、TXT、Markdown 上传正常。
- [ ] 单文件 10MB 限制正常。
- [ ] 已迁移文档列表和 Chunk 数量正常。
- [ ] RAG 检索与来源显示正常。
- [ ] 删除及重新上传文档正常。
- [ ] 不同 visitor 的知识库保持隔离。

### Agent 与外部服务

- [ ] 普通对话正常。
- [ ] 实时天气查询正常。
- [ ] POI 城市搜索和附近搜索正常。
- [ ] GPS 定位与逆地理编码正常。
- [ ] Weather -> POI 多 Tool Workflow 正常。
- [ ] KB -> Weather -> POI 有界多 Tool Workflow 正常。
- [ ] Tool Result、坐标和内部上下文不进入永久聊天历史。

### 客户端

- [ ] 桌面浏览器验证通过。
- [ ] 手机浏览器验证通过。
- [ ] GPS permission 的 granted、prompt、denied 均有正确行为。
- [ ] WebSocket 长连接稳定。
- [ ] 文件上传和较长的 Embedding 请求不会被 Nginx 提前中断。

