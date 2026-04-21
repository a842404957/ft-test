# FT-Oriented Fault Tolerance V1.3.4

`simulator/Fault_Tolerance/` 现在默认服务于 `ft_group_cluster_translate` 主路径，不再把旧 PRAP 路径当作默认示例。

如果当前 shell 不是装好依赖的环境，请把下面的 `python` 替换成 `conda run -n <env> python`。在 `aris-gpu` 上，当前建议使用 `conda run -n ming python`。

## 模块结构

```text
simulator/Fault_Tolerance/
├── config.py
├── pattern_data_loader.py
├── redundancy_group_parser.py
├── fault_injector.py
├── majority_voter.py
├── nearest_pattern_corrector.py
├── metrics_collector.py
├── report_generator.py
├── fault_tolerance_simulation.py
└── README.md
```

## 推荐工作流

1. 训练 / 转换

```bash
python main.py --model Vgg16 --translate ft_group_cluster_translate --run-tag vgg16_demo
```

快速探索时，推荐先只构组：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --build-only \
  --run-tag res18_build_only
```

若要跳过缓存、强制重建 artifacts：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --build-only \
  --force-rebuild \
  --run-tag res18_build_rebuild
```

可用的 FT 入口参数：

- `--build-only`：只生成 FT artifacts 和投影后的 `after_translate_parameters.pth`
- `--force-rebuild`：忽略已有 FT artifacts 缓存，强制从 `--base-checkpoint-epoch` 对应 checkpoint 重建
- `--run-tag`：给本次实验分配独立 tag；若未显式传 `--artifact-dir`，则 artifacts 会写到 `results/ft_runs/<model>/<translate>/<run-tag>/artifacts`
- `--artifact-dir`：显式指定 artifact 输出目录；适合把不同 preset 分开保存
- `--ft-low-cost`：启用低成本 FT 训练 preset；默认会把 FT 训练终点压到 `160`，并保持至少一次 refresh
- `--ft-end-epoch`：控制 FT 训练终点；默认 `200`，`--ft-low-cost` 默认 `160`
- `--ft-reg-interval`：每 N 个 batch 才计算一次 FT 正则；默认 `1`，`--ft-low-cost` 默认 `10`
- `--ft-reg-min-coverage`：只对 coverage 不低于该阈值的层计算 FT 正则；默认 `0.0`，`--ft-low-cost` 默认 `0.1`
- `--ft-reg-min-groups`：只对 repairable group 数不少于该阈值的层计算 FT 正则；默认 `1`，`--ft-low-cost` 默认 `64`
- `--translate-epochs`：FT 训练期间做 before/after translate 评估的 epoch，默认 `200`
- `--refresh-epochs`：动态 refresh 节点，默认 `190,200`
- `--base-checkpoint-epoch`：FT 起始原始 checkpoint，默认 `150`

探索性 FT 训练建议先用低成本模式：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --ft-low-cost \
  --run-tag res18_fast
```

2. 单次三级容错

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_low_fault_rate.json \
  --samples 256 \
  --artifact-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/artifacts \
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim
```

3. 策略对比

```bash
python run_hierarchical_fault_tolerance.py \
  --mode compare \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_high_fault_rate.json \
  --samples 256 \
  --artifact-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/artifacts \
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim
```

4. FT 分析

```bash
python fault_tolerance_analyse.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --data-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/artifacts \
  --output-json results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/analysis/ft_report.json \
  --output-csv results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/analysis/ft_layers.csv
```

5. 结果收集

```bash
python scripts/collect_ft_results.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_high_fault_rate.json \
  --samples 256 \
  --artifact-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/artifacts \
  --report-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim \
  --results-root results/ft_runs \
  --tag vgg16_demo
```

## 真实实验操作

推荐把同一轮真实实验统一落到：

```text
results/ft_runs/<model>/<translate>/<tag>/
```

其中 `single` 和 `compare` 共用同一个 `sim/` 输出目录，`analyse` 写到 `analysis/`，`main.py` 的产物写到 `artifacts/`，`collect` 最终在根目录生成 `summary.csv`、`summary.md` 和 `run_metadata.json`。

## 协议说明

FT 主路径优先读取：

- `model_{model_name}_{translate_name}_group_information.pkl`
- `model_{model_name}_{translate_name}_coverage_ratio_information.pkl`
- `model_{model_name}_{translate_name}_mask.pkl`
- `model_{model_name}_{translate_name}_map_information.pkl`
- `model_{model_name}_{translate_name}_multiple_relationship_information.pkl`
- `model_{model_name}_{translate_name}_after_translate_parameters.pth`

其中：

- `group_information.pkl` 是主协议，记录显式 block-aware groups
- `coverage_ratio_information.pkl` 是正式统计命名
- `reuse_ratio_information.pkl` 只作为兼容别名保留

## 当前行为

- `PatternDataLoader` 优先读取 `group_information.pkl` 和 `coverage_ratio_information.pkl`
- `RedundancyGroupParser` 优先按显式 group 解析
- `FaultToleranceSimulator` 的 Level 1 已按 block-aware 成员做替换
- `fault_tolerance_analyse.py`、`run_hierarchical_fault_tolerance.py` 和 `scripts/collect_ft_results.py` 现在都支持指向独立 artifact 目录

## 兼容说明

- 旧 PRAP 文件名和 `weight_pattern_shape_and_value_similar_translate` 仍保留在 loader / parser fallback 中
- 默认示例命令不再使用旧 PRAP 主路径
