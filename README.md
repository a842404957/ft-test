# ft-test

当前推荐方法是 `ft_group_cluster_translate`。这条主路径已经从 PRAP 风格 reuse 改成了 FT-oriented grouping：先做 FTScore mask 选择，再做 block-aware 冗余分组，再接入三级容错仿真。

## 主流程

训练 / 转换入口：

```bash
python main.py --model Vgg16 --translate ft_group_cluster_translate
```

三级容错仿真：

```bash
python run_hierarchical_fault_tolerance.py --mode single --model Vgg16 --translate ft_group_cluster_translate --samples 256
```

策略对比：

```bash
python run_hierarchical_fault_tolerance.py --mode compare --model Vgg16 --translate ft_group_cluster_translate --config fault_tolerance_config_high_fault_rate.json --samples 256
```

FT 专属分析：

```bash
python fault_tolerance_analyse.py --model Vgg16 --translate ft_group_cluster_translate --output-json ft_report.json --output-csv ft_layers.csv
```

## 产物文件

`ft_group_cluster_translate` 会生成以下产物：

- `model_{model_name}_{translate_name}_mask.pkl`
- `model_{model_name}_{translate_name}_map_information.pkl`
- `model_{model_name}_{translate_name}_multiple_relationship_information.pkl`
- `model_{model_name}_{translate_name}_group_information.pkl`
- `model_{model_name}_{translate_name}_reuse_ratio_information.pkl`
- `model_{model_name}_{translate_name}_after_translate_parameters.pth`
- `model_{model_name}_{translate_name}_refresh_log.csv`

其中 `group_information.pkl` 是 FT 主协议，仿真和分析脚本都会优先读取它；`map_information.pkl` 只保留为旧路径兼容 fallback。

## V1.2 要点

- `extract_ou_patterns(...)` 现在按真实 `channel_number` / block 边界提取成员。
- `ft_group_score_mask(...)` 现在走 FTScore 候选搜索，而不是单纯包装 `get_shape_mask(...)`。
- `ft_group_translate_train(...)` 已支持动态 refresh，并记录 refresh 前后 `coverage / group_count / singleton_ratio`。
- `PatternDataLoader / RedundancyGroupParser / FaultToleranceSimulator` 已优先使用显式 `group_information.pkl`。
