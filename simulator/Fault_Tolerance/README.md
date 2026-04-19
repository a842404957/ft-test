# FT-Oriented Fault Tolerance V1.3.1

`simulator/Fault_Tolerance/` 现在默认服务于 `ft_group_cluster_translate` 主路径，不再把旧 PRAP 路径当作默认示例。

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
python main.py --model Vgg16 --translate ft_group_cluster_translate
```

2. 单次三级容错

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_low_fault_rate.json \
  --samples 256 \
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
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim
```

4. FT 分析

```bash
python fault_tolerance_analyse.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
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
  --report-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim \
  --results-root results/ft_runs \
  --tag vgg16_demo
```

## 真实实验操作

推荐把同一轮真实实验统一落到：

```text
results/ft_runs/<model>/<translate>/<tag>/
```

其中 `single` 和 `compare` 共用同一个 `sim/` 输出目录，`analyse` 写到 `analysis/`，`collect` 最终在根目录生成 `summary.csv`、`summary.md` 和 `run_metadata.json`。

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
- `fault_tolerance_analyse.py` 和 `scripts/collect_ft_results.py` 与这套协议保持一致

## 兼容说明

- 旧 PRAP 文件名和 `weight_pattern_shape_and_value_similar_translate` 仍保留在 loader / parser fallback 中
- 默认示例命令不再使用旧 PRAP 主路径
