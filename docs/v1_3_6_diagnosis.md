# V1.3.6 Diagnosis Baseline

V1.3.6 的结论是：当前 `ft_group_cluster_translate` 主路径已经完成了 repair validity 与 redundancy construction audit，但结构指标不足以支撑后续继续烧全量 mask sweep。

## 当前 FT 主路径基线

- `avg_group_size = 1.3072`
- `avg_singleton_ratio = 0.8891`
- `avg_repairable_ou_ratio = 0.2302`
- `21` 层里有 `20` 层 `mask_density = 1.0`
- 大多数层选择 `shared_topk_1.0000` 或 `shape_seed`

这意味着当前 FT 主路径几乎没有形成稳定的 OU 冗余结构，group 大量退化为 singleton。

## PRAP 对照

- `repairable_ou_ratio = 94.94%`
- `singleton_groups = 0`
- `avg_group_size = 415.41`

PRAP fallback 的冗余构建能力仍然显著强于当前 FT 主路径。

## Oracle Restore

在 `stress_3pct` 下，`oracle restore` 可以把准确率恢复到接近 baseline。这说明：

- fault injection 没有坏
- evaluation pipeline 没有坏
- restore pipeline 没有坏

因此，当前瓶颈不在仿真框架，而在冗余结构与替换质量。

## Level1-only 结果

`Level1-only` 在 `stress_3pct` 下不能恢复模型准确率，且有时会比 faulty 更差。结合 oracle 结果，可以判断：

- 不是 fault/eval pipeline 的假象
- 问题集中在 redundancy structure 和 assignment quality

## 为什么不继续全量 mask sweep

V1.3.7 的离线 FTScore 重选仍然只有：

- `avg_repairable_ou_ratio = 0.2394`
- `avg_singleton_ratio = 0.8843`
- `avg_group_size = 1.3135`

仍远低于 gate：

- `repairable >= 0.45`
- `singleton <= 0.65`
- `avg_group_size >= 1.8`

并且 `Res18 build-only + full mask sweep` 已经表现出明显的构建成本问题。因此，不继续把时间投入到更大的 sweep 上。

## 为什么转向 Redundancy-Budgeted OU Grouping

当前 `FTScore + natural similarity` 的主路径本质上还是在“发现自然冗余”。但当前结构指标已经说明：

- dense / near-dense mask 太容易胜出
- singleton 太多
- 只靠更稀疏的候选重选，增益很小

因此 V1.3.8 转向：

- 在准确率损失预算下主动制造 OU 冗余
- 用 prototype budget、coverage target、singleton 上限和 assignment error 上限来约束 grouping
- 不依赖 dense mask 自然出现大 group

这份文档固定为 V1.3.6 诊断基线，后续 budgeted grouping 的所有结构改进都应以此作为对照。
