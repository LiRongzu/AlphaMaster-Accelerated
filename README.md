# AlphaMaster-Accelerated

AlphaMaster 的**非官方性能导向 Fork / 变体**。本仓库基于 [rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)，目标是在尽量保持原始 reward、公式搜索空间和 walk-forward 评分语义不变的前提下，减少 evaluator 与训练过程中的重复计算和执行开销。

> **Upstream attribution**
>
> 原始项目及主要设计归属于 [rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)。本仓库不是 upstream 官方发布，也不代表原作者对本变体的背书。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

- **本变体仓库**：[LiRongzu/AlphaMaster-Accelerated](https://github.com/LiRongzu/AlphaMaster-Accelerated)
- **原始 upstream**：[rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)
- **Upstream PR**：[PR #6 — Optimize exact EMA and walk-forward evaluation](https://github.com/rosemarycox5334-debug/AlphaMaster/pull/6)
- **原项目 QQ 交流群**：1063897401

![Web 控制台总览](docs/images/00_hero.png)

---

## 本变体修改了什么

当前 `main` 只包含已经通过专项一致性测试的 **exact CPU acceleration**，没有把尚在研究中的 GPU / WF-score fusion 代码混入稳定分支。

| 优化 | 原始做法 | 当前做法 | 科学语义 |
|------|----------|----------|----------|
| **Exact EMA CPU recurrence** | Python / PyTorch 顺序递推 | CPU 上使用 `scipy.signal.lfilter` 的 compiled recurrence；CUDA / autograd / 不支持场景保留 reference fallback | 保持原递推与初值定义 |
| **Exact walk-forward evaluator cache** | rolling WF 中重叠窗口会重复计算 | 每条公式先计算一次完整 `position / turnover / pnl`，再复用 unique windows、Sortino、fold IC 等结果 | 不改变 reward / search space / WF 定义 |

基于当时最新 upstream `main`（`4a0e851`）重新移植后，专项测试结果：

```text
14 passed
```

覆盖：

```text
tests/unit/test_ema_exact_backend.py
tests/unit/test_p1b_wf_cache_equivalence.py
```

### 为什么会快很多

原版 evaluator 的主要问题不是模型本身太大，而是**大量小计算被重复执行**：长序列 EMA 逐点递推，walk-forward 的重叠窗口又反复计算相同的中间量。本变体把这些“重复劳动”改成 compiled recurrence + 缓存 / 复用，因此不需要通过改变公式评分标准来换速度。

开发阶段在 ETHUSDT 15m 长历史数据上的代表性 EMA 测量为：

```text
EMA_5:  ~6.54 s -> ~0.052 s cold
EMA_20: ~6.60 s -> ~0.00175 s
```

这些数字只用于说明原始热点的量级，**不是对所有数据、硬件或完整训练流程的通用加速承诺**。

### GPU 路线状态

我们还单独研究了 SDPA、elite single-pass、`torch.compile` 和 WF-score fusion。GPU compile 路线在独立实验中表现出额外的工程加速，但 CPU / GPU 会因为随机采样拓扑与浮点执行路径不同而产生 trajectory divergence，因此当前公开稳定 `main` 仍以 exact CPU evaluator 为基准。

WF-score fusion 目前也**没有**合入稳定分支：它短期数值非常接近，但长轨迹中微小 scorer 浮点差异会被搜索过程递归放大，仍需要 population-level 结果验证。

---

## 它做什么

AlphaMaster 把「挖因子」做成一条可操作的流水线：

1. **训练**：用强化学习在特征 + 算子空间里搜索公式，按验证集表现选优  
2. **回测**：用 `tanh(因子)` 连续仓位在历史行情上模拟交易，看资金曲线与绩效  
3. **实时分析**：按周期收盘后重算信号，展示方向与把握；方向转折可推飞书提醒  

公式以 token 序列保存（如 `strategies/best_BTCUSDT.json`），可用 StackVM 解释执行，训练 / 回测 / 实时共用同一套信号逻辑。

---

## Web 控制台（推荐入口）

```bash
pip install -r requirements.txt
python run_web.py --port 8765
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。界面分三步：

| 步骤 | 作用 |
|------|------|
| **01 模型训练** | 选 Parquet、开始 / 重新训练、看曲线与日志、导出策略与检查点 |
| **02 策略回测** | 选策略 JSON，设手续费 / 滑点，看绩效与资金曲线 |
| **03 实时分析** | 多数据源监控，收盘后更新信号；可选飞书转折提醒 |

### 模型训练

![训练页](docs/images/01_train.png)

- Parquet 命名：`{品种}_{周期}.parquet`，例如 `BTCUSDT_H1.parquet`、`XAUUSD_H1.parquet`  
- **开始训练**：有检查点则断点续训  
- **重新训练**：清除检查点从头搜索；已有更优策略作为分数下限，不会被弱结果覆盖  
- 展示最优分数、验证分数、训练曲线与最优公式；可选 AI 分析当前训练情况  

### 策略回测

![回测页](docs/images/02_backtest.png)

- 仓位：`position = tanh(factor)`，信号越强仓位越大  
- 成本：手续费 + 滑点（默认约 0.02% / 0.01%）  
- 输出：总收益、夏普、索提诺、盈亏比、滚动夏普与资金曲线  

![资金曲线示例](docs/images/04_equity.png)

### 实时分析

![实时分析页](docs/images/03_realtime.png)

- 数据源：MT5 / OKX 等（以界面可用源为准）  
- **只在当前周期 K 线收盘后**重新判断；未收盘 bar 不参与信号  
- 卡片展示方向（看涨 / 看跌 / 不确定）与把握程度  
- 可选飞书 Webhook：仅在方向转折时推送文字提醒  

---

## 项目结构

```
AlphaMaster/
├── web/                 # FastAPI Web UI（训练 / 回测 / 实时）
├── model_core/          # 特征、算子、StackVM、训练引擎、回测评分
├── data_pipeline/       # Parquet / MT5 K 线加载与对齐
├── strategy_manager/    # 实盘信号与仓位逻辑（与回测口径一致）
├── execution/           # MT5 下单接口
├── backtest_viz/        # 回测引擎与图表
├── strategies/          # best_{symbol}.json 策略文件
├── checkpoints/         # 训练检查点
├── run_web.py           # 启动 Web 控制台
├── train_file.py        # CLI：从单个 Parquet 训练
└── requirements.txt
```

---

## 环境要求

- Python **3.10+**（建议 3.11）  
- PyTorch、pandas、FastAPI、uvicorn 等（见 `requirements.txt`）  
- 可选：MetaTrader 5 终端（实时 MT5 源 / 实盘相关脚本）  
- 复制 `.env.example` 为 `.env` 填写 MT5 等凭证（`.env` 已 gitignore）  

```bash
python -m pip install -r requirements.txt
# TradingView 若上面 git 行失败，可单独装：
# python -m pip install git+https://github.com/rongardF/tvdatafeed.git
```

---

## 常用命令

```bash
# Web 控制台
python run_web.py --port 8765

# CLI 训练（自动续训；加 --from-scratch 则重新训练）
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet --from-scratch
```

策略输出默认在 `strategies/best_{symbol}.json`。

---

## 信号口径（训练 / 回测 / 实时一致）

- 因子经 StackVM 算出标量序列  
- `position = tanh(factor)` ∈ (-1, 1)  
- `|position|` 小于阈值时视为无信号（观望）  
- 实时侧只用**已收盘** K 线，避免盘中抖动与回测不一致  

---

## 截图更新

仓库内展示图由当前 Web UI 截取，可用：

```bash
python scripts/capture_readme_shots.py
```

（需本机已启动 `python run_web.py --port 8765`，并已安装 Playwright + Chromium。）

---

## License

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)。  
修改、分发或通过网络提供服务时，须按相同协议公开对应源代码。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LiRongzu/AlphaMaster-Accelerated&type=date&legend=top-left)](https://www.star-history.com/#LiRongzu/AlphaMaster-Accelerated&type=date&legend=top-left)
