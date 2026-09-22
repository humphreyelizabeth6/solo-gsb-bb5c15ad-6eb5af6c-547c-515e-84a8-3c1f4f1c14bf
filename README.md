# 多租户 GPU 作业公平排队 API

基于 **Python FastAPI + SQLite** 的多租户 GPU 作业调度服务。租户提交需要的 GPU 数量、
预计时长与权重，服务使用**可复现的加权赤字轮询（Weighted Deficit Round-Robin, WDRR）**
选择可运行作业，并按等待时间提供**老化补偿（aging）**，避免小作业被长期饿死；
调度时严格遵守集群 GPU 容量，**绝不越过容量约束**。

- 调度核心是纯函数、无随机数、不读墙钟 —— 相同输入永远产出相同选择顺序
- 赤字余额（deficit / balance）与轮询游标持久化在 SQLite，跨批次、跨重启连续
- 全部写操作经单把异步锁串行化，容量检查、规划、落库原子完成
- 内置 OpenAPI 交互文档（`/docs`）与健康检查

---

## 一条命令启动

需要安装 Docker（含 Compose 插件）。

```bash
docker compose up --build
```

启动后：

| 项目 | 地址 |
| --- | --- |
| API 基址 | <http://localhost:8000> |
| Swagger 交互文档 | <http://localhost:8000/docs> |
| OpenAPI 描述 | <http://localhost:8000/openapi.json> |
| 健康检查 | <http://localhost:8000/health> |

数据保存在命名卷 `gpu_scheduler_data`（容器内 `/data/scheduler.db`），重启/重建容器不丢失。
修改 GPU 总数等参数可直接编辑 `docker-compose.yml` 的 `environment`，或复制
`.env.example` 为 `.env` 后调整，然后重新 `docker compose up -d`。

---

## 快速上手

```bash
# 1) 租户 alpha 提交一个需要 2 GPU、预计 120 秒、权重 1 的作业
curl -X POST http://localhost:8000/jobs \
  -H 'content-type: application/json' \
  -d '{"tenant_id":"alpha","gpus":2,"duration_seconds":120,"weight":1}'

# 2) 租户 beta（权重 3）提交作业；weight 仅首次提交必带，之后可省略
curl -X POST http://localhost:8000/jobs \
  -H 'content-type: application/json' \
  -d '{"tenant_id":"beta","gpus":1,"duration_seconds":60,"weight":3}'

# 3) 调度下一批（返回选择顺序与每租户余额）
curl -X POST http://localhost:8000/schedule

# 4) 查询队列与余额
curl http://localhost:8000/queue
curl http://localhost:8000/balances
```

---

## 调度算法

### 成本模型

每个作业的成本为 **GPU-秒**：

```
cost = gpus × duration_seconds
```

### 加权赤字轮询（WDRR）

每一轮（pass）按 **租户名字典序**（游标位置跨批次轮转）访问每个有排队作业的租户，
向其赤字余额发放一笔配额（quantum）：

```
quantum = BASE_QUANTUM × weight + aging_bonus
```

随后该租户从队首（FIFO）开始，连续运行所有满足条件的作业：

1. 作业需要的 GPU 数 ≤ 当前空闲 GPU 数；
2. 作业成本 ≤ 当前赤字余额。

每运行一个作业就从余额扣除其成本。当租户队列清空时，按标准 DRR 规则清零剩余余额，
防止长期“囤积分”。**赤字不会为负**：余额不够的作业必须等后续轮次累积配额。

### 老化补偿（防止小作业饿死）

轮询中每个租户每轮固定获得一次配额，因此在“作业长期跑不动”的场景下，等待是有界的；
此外按队首作业的等待时间发放线性老化补偿：

```
aging_bonus = min(AGING_RATE × weight × wait_seconds,
                  AGING_CAP_FACTOR × BASE_QUANTUM × weight)
```

等待越久，每轮积累的赤字越多，能更快达到大作业成本，从而避免小租户的小作业
被大作业无限期压住。补偿设有上限（默认基础配额的 10 倍），避免单个超长等待作业独占集群。

### 容量安全

- 仅当 `job.gpus <= free_gpus` 时作业才会被选中；放不下时该租户的赤字与作业原样保留，
  在后续轮次/批次重试。
- 当剩余空闲 GPU 装不下任何队首作业时，本批结束，这些作业在响应的 `blocked` 中返回。
- 运行中的作业经 `/complete` 或 `/cancel` 释放 GPU 后，下一次 `/schedule` 才会启用新容量。

### 可复现性

- 租户访问顺序：字典序 + 持久化游标（每轮轮转一个位置）；
- 作业顺序：按提交序号 FIFO；
- 不使用随机数；等待时长只来自数据库中记录的 `submitted_at` 与调度时刻之差。

同一份持久化状态连续调用，结果确定。内部为避免长作业成本导致上千次空转，
当一整轮无人可运行时，会用闭式公式**快进**到“最早有队首可负担”的轮数
（轮空期间队首不动，配额线性增长），数学上等价于逐轮循环。

---

## API 用法

### 租户

#### `POST /tenants` —— 创建租户 / 更新权重（可选，首次提交作业时也可隐式创建）

```json
{ "tenant_id": "acme", "weight": 2.5 }
```

`tenant_id` 仅允许字母、数字、`_`、`-`；`weight > 0`。已存在则更新权重。

### 作业

#### `POST /jobs` —— 提交作业

```json
{
  "tenant_id": "acme",
  "gpus": 4,
  "duration_seconds": 3600,
  "weight": 2.5
}
```

- `gpus > 0`、`duration_seconds > 0`；`gpus` 超过集群总数返回 `422`。
- 新租户必须带 `weight`（否则 `400`）；老租户可省略，带上则同时更新权重。
- 返回含服务端生成的 `job_id`、`cost = gpus*duration_seconds`、状态 `queued`。

#### `POST /schedule` —— 调度下一批

无请求体。在当前空闲容量内选出并**原子地**把作业置为 `running`，返回：

```json
{
  "scheduled": [
    {
      "order": 0,
      "job_id": "job-...",
      "tenant_id": "beta",
      "gpus": 1,
      "duration_seconds": 60,
      "cost": 60.0,
      "wait_seconds": 0.12,
      "quantum": 180.0,
      "deficit_after": 120.0,
      "passes_before_run": 0
    }
  ],
  "selection_order": ["job-..."],
  "blocked": [],
  "passes": 1,
  "next_cursor": "alpha",
  "capacity": { "total_gpus": 8, "running_gpus": 1, "free_gpus": 7 },
  "balances": [
    {
      "tenant_id": "beta",
      "weight": 3.0,
      "deficit": 120.0,
      "queued_jobs": 0,
      "running_jobs": 1,
      "queued_gpus": 0,
      "running_gpus": 1
    }
  ]
}
```

- `selection_order`：本批选择顺序（job_id 列表）。
- `scheduled[].passes_before_run`：该作业在第几轮被选中（可直观看出老化加速效果）。
- `blocked`：因剩余容量不足无法启动的队首作业。
- `balances[].deficit`：调度后每租户赤字余额（正=结余，可在后续批次优先消费）。
- 没有任何作业可运行时返回空列表，仍然安全可重复调用。

#### `POST /jobs/{job_id}/complete` —— 完成作业（释放 GPU）

仅 `running` 作业可完成，否则 `409`；不存在 `404`。

#### `POST /jobs/{job_id}/cancel` —— 取消作业（释放 GPU）

可取消 `queued` 或 `running` 作业；对已取消作业重复调用幂等（`200`）；
已完成的作业不能取消（`409`）。

### 查询

| 方法/路径 | 说明 |
| --- | --- |
| `GET /queue` | 容量快照、全部排队作业（FIFO）、每租户余额、下一游标 |
| `GET /balances` | 每租户权重、赤字余额、排队/运行作业数与占用 GPU 数 |
| `GET /jobs` | 全部作业，支持 `?tenant_id=acme&status=running` 过滤 |
| `GET /jobs/{job_id}` | 单个作业详情 |
| `GET /schedule/history?limit=20` | 最近调度批次（选择顺序、阻塞作业） |
| `GET /health` | 健康检查与容量快照 |

`status` 取值：`queued` / `running` / `completed` / `cancelled`。

### 状态码约定

- `201` 资源创建成功；`200` 成功
- `400` 新租户未带权重
- `404` 作业或租户不存在
- `409` 状态冲突（如完成一个排队作业、取消已完成作业）
- `422` 参数校验失败（GPU 数非法、超过集群容量等）

---

## 配置项（环境变量）

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `TOTAL_GPUS` | `8` | 集群 GPU 总数，所有租户共享 |
| `BASE_QUANTUM` | `60` | 每轮每单位权重的基础配额（GPU-秒） |
| `AGING_RATE` | `1` | 每等待 1 秒、每单位权重增加的配额（GPU-秒） |
| `AGING_CAP_FACTOR` | `10` | 老化补偿上限 = 该倍数 × 基础配额 |
| `DB_PATH` | `/data/scheduler.db` | SQLite 数据库路径 |

> 含义示例：权重 3 的租户每轮基础配额为 `60×3=180` GPU-秒；其队首作业已等待
> 600 秒时，每轮配额为 `180 + min(1×3×600, 10×180) = 360` GPU-秒。

---

## 本地开发与测试（无需 Docker）

需要 Python 3.11+：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest httpx

# 运行测试（20 个：纯函数调度算法、朴素实现等价性、端到端 API 与并发安全）
python -m pytest -q

# 本地启动（默认 8000 端口）
TOTAL_GPUS=8 DB_PATH=./local.db uvicorn app.main:app --reload
```

项目结构：

```
app/
  config.py       # 环境变量配置
  database.py     # 异步 SQLite 层（aiosqlite，单连接 + 显式事务）
  scheduler.py    # 纯函数 WDRR 规划器（无 I/O、无随机数）
  schemas.py      # Pydantic 请求/响应模型
  main.py         # FastAPI 路由与应用装配
tests/
  test_scheduler.py
  test_api.py
Dockerfile
docker-compose.yml
```

---

## 设计说明

- **多租户隔离与公平性**：租户权重只影响每轮配额大小；每轮每个有队列的租户至少被访问一次，
  低权重租户的等待时间有上界；老化进一步补偿长时间等待者。
- **持久化一致性**：一次批次的“作业置 running、赤字更新、游标推进、历史写入”在单个
  `BEGIN IMMEDIATE` 事务中提交；并发请求由应用层锁 + 事务双重保护。
- **取消语义**：取消排队作业不发放/回收任何配额（标准 DRR 的队首阻塞语义）；
  租户队列清空时余额清零。
- **可演进点**：多 GPU 机型/拓扑亲和、每租户容量上限、按实际而非预计时长结算，
  可在 `scheduler.py` 的成本与适配判定处扩展，不影响 API 形态。
