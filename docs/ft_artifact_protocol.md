# FT Artifact Protocol

V1.3.1 的 `ft_group_cluster_translate` 主路径当前使用以下协议：

- `model_{model}_{translate}_mask.pkl`
  结构化 mask。
- `model_{model}_{translate}_map_information.pkl`
  兼容旧解析器的映射表。
- `model_{model}_{translate}_multiple_relationship_information.pkl`
  兼容旧解析器的 member multiplier 信息。
- `model_{model}_{translate}_group_information.pkl`
  FT 主协议，记录显式 block-aware groups。
- `model_{model}_{translate}_coverage_ratio_information.pkl`
  正式统计命名，表示按层覆盖率/压缩比统计。
- `model_{model}_{translate}_reuse_ratio_information.pkl`
  兼容别名，与 `coverage_ratio_information.pkl` 内容保持一致。
- `model_{model}_{translate}_after_translate_parameters.pth`
  转换训练后的模型参数。
- `model_{model}_{translate}_refresh_log.csv`
  动态 regroup refresh 摘要。

## 兼容关系

- 新主路径应优先读取 `coverage_ratio_information.pkl`。
- 若缺失，再 fallback 到 `reuse_ratio_information.pkl`。
- 旧 PRAP 路径仍可能只存在 `shape_and_value_*_reuse_ratio_information.pkl`，该命名继续被 loader 接受。

## 推荐工作流里的结果目录

推荐把同一轮实验的输出整理到：

```text
results/ft_runs/<model>/<translate>/<tag>/
```

其中：

- `sim/` 放 single / compare 的仿真报告
- `analysis/` 放 `fault_tolerance_analyse.py` 的 json/csv
- 根目录放 `summary.csv`、`summary.md`、`run_metadata.json`
