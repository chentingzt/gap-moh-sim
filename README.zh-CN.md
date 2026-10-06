# gapmoh_sim — GAP-MOH 切换判据的独立验证

一套自包含、仅依赖 NumPy 的仿真器：**按论文自身写明的规格**独立实现 GAP-MOH 的切换模型，
再检验论文自己的判据能否复现论文报告的数字。

> **这不是论文仿真器的复原**，而是仅凭稿件文本对其判据的独立实现。
> 本包数值与论文不符时，如实报告差异，**绝不调参去凑**。完整溯源记录（含逐条偏差）可向作者索取。

> 英文版见 [`README.md`](README.md)（内容更全，含完整脚本清单与环境说明）。

---

## 首要结论

按论文自己的时序可行性判据

```
Δt_exec  <  t_exit − t_init          （稿件 §3.4）
```

六方案**一律 100.0%，极差 0.00 个百分点**——连均匀随机策略也是 100%。
该判据是**二值**的：从不切换 = 0%，只要切换 = 100%，**无任何梯度可供强化学习攀爬**。

原因是结构性的，不是随机性：GBPT 触发定义即 $t_{\text{init}} = t_{\text{exit}} - \Delta t_{\text{ho}}$，
故可用预算恒等于 $\Delta t_{\text{ho}} = \Delta t_{\text{exec}} + 2$ ms $> \Delta t_{\text{exec}}$；
时长类方案的预算同样恒等于其阈值。实测预算比各方案自报的 $\Delta t_{\text{exec}}$
高 2～4 个数量级。

推导、三张探针表、以及四个被逐一实测证伪的候选走向，见作者处的溯源记录（可向作者索取）。

---

## 快速开始

```bash
pip install -r requirements.txt

python scripts/00_verify_geometry.py            # 几何回归（对已发布锚点）
python -m pytest tests/ -q                      # 42 项断言，单核约 70 s
python scripts/01_diagnostic.py --terminals 24 --horizon 3600 \
       --out diagnostic_stage0.json             # Stage 0 闸门；不训练
```

---

## 目录

```
gapmoh/                 仿真器包
  constants.py          全部规格常量，逐条带 [SPEC]/[CHOICE]/[DEVIATION] 标签
  orbit.py              Walker-delta 星座、ECEF 传播、仰角、多普勒
  channel.py            Ka 链路预算（对齐稿件表 1）+ ITU-R P.618-13 雨衰
  passes.py             逐终端过境表 + 决策网格上的活跃索引
  policies.py           六方案触发/目标星规则（转录自稿件表 7 注）
  sim.py                单终端切换时序；触发时刻精确求解
  metrics.py            C1/C2/C3 三种判据 + 容量检查
  metrics_v2.py         重建后的指标集 M1–M5（供 05/08/09 使用）
  rl/                   NumPy DQN 栈：env.py / nets.py / policy_target.py

scripts/                11 个编号入口（00 → 11）
tests/                  7 个模块、42 项断言
results/                随仓库提交的运行产物（JSON + 日志 + 1 张 PNG）
requirements.txt        numpy / matplotlib / pytest
```

> `03_*`、`04_*` 两个脚本**不存在**——Stage 0 闸门判定「生成论文格式对照表」为时过早，
> 计划中止。编号保持原样不重排，以免别处引用的脚本编号失去指向。

---

## 输入与输出

**本包没有外部输入数据集。** 世界由 `gapmoh/constants.py` 确定性生成：虚构的
Walker-delta 星座（72 面 × 70 颗、550 km、53°）、合成终端轨迹、闭式 Ka 链路预算。
不随仓库附带也不需要任何 TLE、星历文件或实测轨迹——这也是本包不依赖
`skyfield`/`sgp4` 的原因（记为偏差 D1，见作者处的溯源记录）。

**输出**位于 `results/`，每次运行写一个 JSON 结果文件加一份 `*_log.txt` 人类可读日志，
均作为运行产物提交。

关于已提交产物的两点说明（如实声明，未做美化）：

1. **日志内含原机器的绝对路径。** `wrote D:\...\results\...` 一类行反映该次运行的实际位置，
   **刻意未改写**——它们是某次具体运行的证据，改写即等于篡改记录。中文 Windows 控制台的
   编码乱码同样原样保留。
2. **`results/` 内的部分文档性文字引用了稿件源文件**，而稿件不属本仓库。数值本身是自洽的。

---

## 三条不可逾越的线

1. **`gapmoh/` 包内不得出现论文表 7 的任何数值**，除非该行带 `[SPEC]` 出处标签。
   由 `tests/test_integrity_lint.py` 机械强制。
2. **`simulate_terminal` 的 `dt_exec_s` 无默认值。** 表 7 的执行时长是判据的**输入**，
   由调用方给出，不得回流为包常量。
3. **绝不调参去凑表 7。** 结果与表 7 不符时报告差异，两个方向都不改数据。

---

## 已验证的锚点

| 量 | 值 | 对照 |
|---|---|---|
| 最高仰角切换频次 | 18.9167 次/h | 已发布 `handover_frequency_sim/` |
| 剩余最长切换频次 | 9.1667 次/h | 同上 |
| 窗口均值（前两者） | 3.1673 / 6.5127 min（n = 227 / 110） | 同上 |
| 噪声底 | −93.0 dBm | 稿件表 1 |
| 自由空间损耗 @15° | 185.6 dB | 稿件表 1 |
| 晴空 SNR @15° / 天顶 | 20.6 dB / 29.4 dB | 稿件表 1 |
| 斜距 @15° | 1518 km | 稿件表 1 |
| G/T | 10.44 dB/K | 稿件表 1 |

`handover_frequency_sim/` 是另一个同级包，**不包含在本仓库内**；上表所列为其已发布的对照值。

---

## 适用范围与限制

- **本发布环境只实现了论文四个奖励项中的两个。** `r_interruption` 与 `r_load` 在所评估
  工况下**恒为零**（该工况下切换不会失败，且每星负载远低于 $C$=50 上限），故省略不改变
  任何已报数值。观测向量的负载槽同样硬编码为 0。此点在稿件中已披露。
- **RL 后端为 NumPy，非 PyTorch。** 稿件规定 PyTorch 2.0 + CUDA，本环境两者皆无。
  演员网络结构、学习率与优化器均按规格实现，差异仅在张量后端。
- **已提交的训练产物是缩规模运行**（M=20、1800 s、60 回合、3 种子），并非稿件的
  10000 回合 × 10 种子 × M=100。超参一致，规模不同。
- **消融表中标为 `MADRL-Basic` / `Grid-Only` 的臂是脚本化策略替身**，不是独立训练的网络。

---

## 依赖与环境

```
numpy>=2.0
matplotlib>=3.8
pytest>=8.0
```

假定 Python 3.10+（使用了 `dataclasses`、`pathlib`、PEP 604 联合类型）。

---

## 引用

稿件修订中：

> *GAP-MOH: Grid-Boundary Proactive Handover for Asia-Pacific Maritime LEO
> Satellite Networks* —— 作者与单位待定稿。

请引用论文，而非本仓库。

---

## 许可

**尚未附加许可证。** 无 `LICENSE` 文件时，默认保留全部权利。
在转为公开仓库之前，请先选定并添加许可证。
