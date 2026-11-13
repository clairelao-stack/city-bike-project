# City Bike Project

Citi Bike 夜间调度：**保留原 MILP 的全部变量、目标函数和约束**，在求解之前独立读取、检查并可选改进初始路线。

## 文件分工

| 文件 | 作用 |
|---|---|
| `run_experiment.py` | 统一设置 warm start 路径、启发式时间、求解时间和结果文件夹 |
| `warm_start.py` | 读取 JSON、检查路线、保存准备好的方案、设置 Gurobi `.Start` |
| `heuristic.py` | 可选的简单局部搜索；尝试换顺序、增删站点、调整车辆分工 |
| `main.py` | 原模型与求解流程，只调用初始方案模块 |
| `prepare_start.py` | 单独生成并检查初始方案，不启动 Gurobi |
| `Run_Experiment.ipynb` | Jupyter 运行入口 |
| `warm_start_83457.json` | 已验证的原模型可行方案，目标值 834.5695 |
| `model_station_inputs.json` | 固定的 165 站点库存和目标数据 |
| `selected_depots.json` | 两辆车的固定车库位置 |

流程：

```text
读取保存的路线 → 可行性检查 → 可选局部搜索 → 再检查 → 设置 .Start → 原模型求解
```

## 最简单的运行方式

下载并解压仓库，在项目文件夹中打开 `Run_Experiment.ipynb`，运行唯一的代码单元：

```python
%run run_experiment.py
```

使用已经能运行 Gurobi 的 Python kernel。需要安装依赖时：

```python
%pip install -r requirements.txt
```

运行完整 MILP 需要你自己有效的 Gurobi 许可证；本仓库不包含许可证或凭据。

## 只需要改这里

打开 `run_experiment.py`，顶部设置如下：

```python
WARM_START_FILE = ROOT / 'warm_start_83457.json'
HEURISTIC_SECONDS = 30
SOLVER_SECONDS = 600
OUTPUT_FOLDER = ROOT / ('results_heuristic' if HEURISTIC_SECONDS else 'results_warm_only')
```

- `HEURISTIC_SECONDS = 0`：只使用保存的 warm start，不运行新增局部搜索。
- `HEURISTIC_SECONDS = 30`：先用最多约 30 秒改进初始路线，再求解原模型。
- `SOLVER_SECONDS = 600`：原模型最多求解 600 秒；启发式时间另计。
- 先保持容量 25 辆、每车距离预算 25 km，方便对照。
- 同一结果目录会被下次运行覆盖；需要保留时修改 `OUTPUT_FOLDER`。

## 单独试启发式，不求解模型

```python
%run prepare_start.py --seconds 30
```

结果在 `prepared_start/`。把 `run_experiment.py` 中的 `WARM_START_FILE` 改为
`ROOT / 'prepared_start' / 'prepared_warm_start.json'`，并将 `HEURISTIC_SECONDS` 设为 0，
即可把单独准备好的方案交给原模型。

也可以直接运行：

```python
%run main.py --warm-start prepared_start/prepared_warm_start.json --output-dir results_prepared -p 600
```

## 新增启发式是什么？

这是简单局部搜索，**不是 Clarke–Wright**。它从已有好路线出发，随机尝试移动或交换站点、反转部分访问顺序、加入或删除站点，以及插入先取后送的站点对。每个候选访问顺序都重新生成取送量，检查容量和距离；只接受目标值更低的可行方案。

启发式只构造朝库存目标调整的候选方案；这条选择规则**没有加入原 MIP 的约束**。原 MIP 仍保留原来允许的全部取送选择、MTZ、Big-M 载重传递、全路段和每站最多访问一次等限制。局部搜索不保证找到全局最优，也不保证有明显改善。

## 查看结果

| 输出 | 内容 |
|---|---|
| `prepared_warm_start.json` | 实际交给原模型的初始方案 |
| `warm_start_report.json` | 初始/改进后的成本、启发式耗时和验证状态 |
| `gurobi.log` | 求解日志，检查是否出现 `Loaded user MIP start` |
| `solution.json` | 原模型最终找到的可行方案、下界和 gap |
| `route_steps.csv` | 访问顺序、每站取送量、车上余量 |
| `summary.csv` | 成本、库存偏差和里程汇总 |
| `validation_checks.csv` | 原有验证结果 |
| `routes.png` | 路线图；街道底图需要网络 |

原模型使用 600 秒默认上限和原来的求解停止容差；没有额外设置 50% 的 gap 阈值。
Warm start 和启发式不保证降低最终 gap。比较时同时报告启发式/准备时间与 MIP 时间、目标值、下界和 gap。

## 已做验证

- 原模型的变量、目标函数和约束代码与用户提供的原版逐字对比一致。
- 保存的 834.5695 方案通过独立库存、载重、访问和距离重放检查。
- 165 站点短局部搜索：初始成本由 834.5695 小幅降到 834.4510，方案通过检查。
- 使用 2 秒启发式和 10 秒原 MIP 做过集成验证，Gurobi 接收了 834.4510 的 warm start，原有 12 类验证全部通过。
- 以上是短验证，**不是默认 30 秒启发式＋600 秒求解的性能结论**。计时停止可能使相同种子得到略不同结果。

输入和初始方案沿用项目已有固定数据；没有随仓库重新分发原始骑行档案、许可证或本机日志。

本仓库先作为实验代码存放，不提供额外开源许可证；依赖软件的许可证保持各自条款。
