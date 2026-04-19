# FT-Oriented Fault Tolerance

`simulator/Fault_Tolerance/` 已经切到 FT-oriented 主路径，默认围绕 `ft_group_cluster_translate` 产物工作。

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

## 主链路

训练 / 转换：

```bash
python main.py --model Vgg16 --translate ft_group_cluster_translate
```

单次三级容错：

```bash
python run_hierarchical_fault_tolerance.py --mode single --model Vgg16 --translate ft_group_cluster_translate --config fault_tolerance_config_low_fault_rate.json --samples 256
```

策略对比：

```bash
python run_hierarchical_fault_tolerance.py --mode compare --model Vgg16 --translate ft_group_cluster_translate --config fault_tolerance_config_high_fault_rate.json --samples 256
```

FT 分析：

```bash
python fault_tolerance_analyse.py --model Vgg16 --translate ft_group_cluster_translate --output-json ft_report.json --output-csv ft_layers.csv
```

最小回归：

```bash
python scripts/regression_check.py
```

结果收集：

```bash
python scripts/collect_ft_results.py --model Vgg16 --translate ft_group_cluster_translate --config fault_tolerance_config_high_fault_rate.json --samples 256 --tag vgg16_demo
```

## 产物协议

FT 主路径优先读取以下产物：

- `model_{model_name}_{translate_name}_mask.pkl`
- `model_{model_name}_{translate_name}_map_information.pkl`
- `model_{model_name}_{translate_name}_multiple_relationship_information.pkl`
- `model_{model_name}_{translate_name}_group_information.pkl`
- `model_{model_name}_{translate_name}_coverage_ratio_information.pkl`
- `model_{model_name}_{translate_name}_reuse_ratio_information.pkl`
- `model_{model_name}_{translate_name}_after_translate_parameters.pth`

其中：

- `group_information.pkl` 是主协议，记录显式 block-aware 冗余组。
- `map_information.pkl` 和 `multiple_relationship_information.pkl` 继续保留，用于兼容旧解析路径。
- `coverage_ratio_information.pkl` 是正式统计命名，`reuse_ratio_information.pkl` 仅保留为兼容别名。

## 当前行为

- `PatternDataLoader` 优先读取 `group_information.pkl`，缺失时再 fallback 到旧 `map_information.pkl`。
- `RedundancyGroupParser` 优先按显式 group 解析，保留 block member / block offset / multiplier 信息。
- `FaultToleranceSimulator` 的 Level 1 已按 block-aware 成员做替换，Level 2 / Level 3 保持兼容。

## Python 用法

```python
import torch

from model import Vgg16
from simulator.Fault_Tolerance import FaultToleranceSimulator

model = Vgg16(num_classes=10)
model.load_state_dict(
    torch.load('model_Vgg16_ft_group_cluster_translate_after_translate_parameters.pth')
)

simulator = FaultToleranceSimulator(
    model=model,
    model_name='Vgg16',
    translate_name='ft_group_cluster_translate',
    config_file='fault_tolerance_config_low_fault_rate.json',
)
```

## 兼容说明

- 旧 PRAP 文件名和 `weight_pattern_shape_and_value_similar_translate` 仍保留在 loader / parser fallback 中。
- 文档和示例命令不再把旧路径当作主路径。
